import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';

export const GRID_DEPTH = 1;

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
      function parseBox(out: tf.Tensor, normalize: boolean = false) {
        let [ center, size, angle, confidence ] =
            tf.split(out, [ 2, 2, 1, 1 ], -1);

        angle = tf.squeeze(angle, [ angle.rank - 1 ]);
        confidence = tf.squeeze(confidence, [ confidence.rank - 1 ]);

        if (normalize) {
          confidence = tf.sigmoid(confidence);
        }

        const box = {
          center,
          size,
          angle,
        };

        const halfSize = box.size.div(tf.scalar(2));

        const corners = {
          topLeft: box.center.sub(halfSize),
          bottomRight: box.center.add(halfSize),
        };

        return { box, confidence, corners };
      }

      function area(size: tf.Tensor) {
        let [ width, height ] = tf.split(size, [ 1, 1 ], -1);

        width = width.squeeze([ width.rank - 1 ]);
        height = height.squeeze([ height.rank - 1 ]);

        return width.mul(height);
      }

      const x = parseBox(xs);
      const y = parseBox(ys, true);

      const intersection = {
        topLeft: tf.maximum(x.corners.topLeft, y.corners.topLeft),
        bottomRight: tf.minimum(x.corners.bottomRight, y.corners.bottomRight),
      };

      const intersectionSize =
          tf.relu(intersection.bottomRight.sub(intersection.topLeft));

      const interArea = area(intersectionSize);
      const xArea = area(x.box.size);
      const yArea = area(y.box.size);

      const epsilon = tf.scalar(1e-7);
      const iou = interArea.div(xArea.add(yArea).sub(interArea).add(epsilon));

      const angleDiff = tf.abs(tf.cos(x.box.angle.sub(y.box.angle)));

      const angleIOU = iou.mul(angleDiff);

      let onMask: tf.Tensor;
      if (GRID_DEPTH === 1) {
        onMask = tf.onesLike(angleIOU);
      } else {
        const argMax = angleIOU.argMax(-1).flatten();
        const maskShape = angleIOU.shape;
        onMask = tf.oneHot(argMax, GRID_DEPTH, 1, 0).cast('float32')
            .reshape(maskShape);
      }

      const hasObject = x.confidence.mean(-1);
      const noObject = tf.scalar(1).sub(hasObject);

      const objLoss = tf.squaredDifference(x.confidence, y.confidence)
          .mul(onMask).sum(-1)
          .mul(hasObject)
          .mul(tf.scalar(LAMBDA_OBJ));

      const noObjLoss = tf.squaredDifference(x.confidence, y.confidence)
          .mean(-1)
          .mul(noObject)
          .mul(tf.scalar(LAMBDA_NO_OBJ));

      const iouLoss = tf.sub(tf.scalar(1), angleIOU)
          .mul(onMask).sum(-1)
          .mul(hasObject)
          .mul(tf.scalar(LAMBDA_IOU));

      return objLoss;
    });
  }
}
