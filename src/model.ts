import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';

export const GRID_DEPTH = 5;

const LAMBDA_OBJ = 1;
const LAMBDA_NO_OBJ = 0.5;
const LAMBDA_IOU = 5;

const LR = 1e-3;

export class Model {
  public readonly model: tf.Sequential;

  constructor() {
    const model = tf.sequential();

    // Initial layers
    model.add(tf.layers.conv2d({
      inputShape: [ TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ],
      kernelSize: 3,
      filters: 16,
      activation: 'relu',
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [ 2, 2 ],
      strides: [ 2, 2 ],
    }));

    function convPool(kernel: number, filters: number, pool: number,
                      stride: number) {
      model.add(tf.layers.batchNormalization({}));

      model.add(tf.layers.conv2d({
        kernelSize: kernel,
        filters,
        activation: 'relu',
      }));

      model.add(tf.layers.maxPooling2d({
        poolSize: [ pool, pool ],
        strides: [ stride, stride ],
      }));
    }

    convPool(3, 32, 2, 2);
    convPool(3, 64, 2, 2);
    convPool(3, 128, 2, 2);

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 512,
      activation: 'relu'
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 512,
      activation: 'relu'
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: GRID_CHANNELS * GRID_DEPTH,
      activation: 'relu'
    }));

    model.add(tf.layers.reshape({
      targetShape: [ GRID_SIZE, GRID_SIZE, GRID_DEPTH, GRID_CHANNELS ]
    }));

    model.compile({
      loss: (xs, ys) => this.loss(xs, ys),
      optimizer: tf.train.adam(LR),
    });

    this.model = model;
  }

  private loss(xs: tf.Tensor, ys: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // shape == [ batch, grid_size, grid_size, grid_depth, grid_channels ]
      const centerXIndex = tf.tensor1d([ 0 ], 'int32');
      const centerYIndex = tf.tensor1d([ 1 ], 'int32');
      const widthIndex = tf.tensor1d([ 2 ], 'int32');
      const heightIndex = tf.tensor1d([ 3 ], 'int32');
      const angleIndex = tf.tensor1d([ 4 ], 'int32');
      const confidenceIndex = tf.tensor1d([ 5 ], 'int32');

      function select(t: tf.Tensor, index: tf.Tensor1D): tf.Tensor {
        return t.gather(index, -1).squeeze([ 4 ]);
      }

      function parseBox(out: tf.Tensor) {
        const box = {
          cx: select(out, centerXIndex),
          cy: select(out, centerYIndex),
          width: select(out, widthIndex),
          height: select(out, heightIndex),
          angle: select(out, angleIndex),
        };

        const confidence = select(out, confidenceIndex);

        const corners = {
          leftTop: {
            x: box.cx.sub(box.width),
            y: box.cy.sub(box.height),
          },
          rightBottom: {
            x: box.cx.add(box.width),
            y: box.cy.add(box.height),
          },
        };

        return { box, confidence, corners };
      }

      const x = parseBox(xs);
      const y = parseBox(ys);

      const intersection = {
        leftTop: {
          x: tf.maximum(x.corners.leftTop.x, y.corners.leftTop.x),
          y: tf.maximum(x.corners.leftTop.y, y.corners.leftTop.y),
        },
        rightBottom: {
          x: tf.minimum(x.corners.rightBottom.x, y.corners.rightBottom.x),
          y: tf.minimum(x.corners.rightBottom.y, y.corners.rightBottom.y),
        }
      };

      const interArea = tf.mul(
        intersection.rightBottom.x.sub(intersection.leftTop.x),
        intersection.rightBottom.y.sub(intersection.leftTop.y));

      const xArea = tf.mul(
        x.corners.rightBottom.x.sub(x.corners.leftTop.x),
        x.corners.rightBottom.y.sub(x.corners.leftTop.y));

      const yArea = tf.mul(
        y.corners.rightBottom.x.sub(y.corners.leftTop.x),
        y.corners.rightBottom.y.sub(y.corners.leftTop.y));

      const epsilon = tf.scalar(1e-23);
      const iou = interArea.div(xArea.add(yArea).sub(interArea).add(epsilon));

      const angleDiff = tf.abs(tf.cos(x.box.angle.sub(y.box.angle)));

      const angleIOU = iou.mul(angleDiff).squeeze([ 4 ]);
      const argMax = angleIOU.argMax(-1).flatten();

      const maskShape = angleIOU.shape;

      const onMask = tf.cast(tf.oneHot(argMax, GRID_DEPTH, 1, 0), 'float32')
          .reshape(maskShape);
      const offMask = tf.cast(tf.oneHot(argMax, GRID_DEPTH, 0, 1), 'float32')
          .reshape(maskShape);

      const objLoss = tf.sub(x.confidence, y.confidence).mul(onMask)
          .sum(-1)
          .mul(tf.scalar(LAMBDA_OBJ));

      const noObjLoss = tf.sub(x.confidence, y.confidence).mul(onMask)
          .sum(-1)
          .mul(tf.scalar(LAMBDA_NO_OBJ));

      const iouLoss = tf.sub(tf.scalar(1), iou.mul(onMask).sum(-1))
        .mul(tf.scalar(LAMBDA_IOU));

      return objLoss.add(noObjLoss).add(iouLoss);
    });
  }
}
