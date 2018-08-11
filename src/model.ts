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

const EPSILON = tf.scalar(1e-7);
const PI = tf.scalar(Math.PI);

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
      activation: 'sigmoid'
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
      function parseBox(out: tf.Tensor) {
        let [ center, size, angle, confidence ] =
            tf.split(out, [ 2, 2, 1, 1 ], -1);

        angle = PI.mul(tf.squeeze(angle, [ angle.rank - 1 ]));
        confidence = tf.squeeze(confidence, [ confidence.rank - 1 ]);

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
      const y = parseBox(ys);

      // Find intersection
      const intersection = {
        topLeft: tf.maximum(x.corners.topLeft, y.corners.topLeft),
        bottomRight: tf.minimum(x.corners.bottomRight, y.corners.bottomRight),
      };

      const intersectionSize =
          tf.relu(intersection.bottomRight.sub(intersection.topLeft));

      // Calculate all area
      const interArea = area(intersectionSize);
      const xArea = area(x.box.size);
      const yArea = area(y.box.size);
      const unionArea = xArea.add(yArea).sub(interArea);

      // Calculate Intersection over Union
      const iou = interArea.div(unionArea.add(EPSILON));

      // Multiply by angle difference
      const angleDiff = tf.abs(tf.cos(x.box.angle.sub(y.box.angle)));
      const angleIOU = iou.mul(angleDiff);

      // Mask out maximum angleIOU in each grid group
      const argMax = angleIOU.argMax(-1).flatten();
      const maskShape = angleIOU.shape;
      let onMask = tf.oneHot(argMax, GRID_DEPTH, 1, 0).cast('float32')
          .reshape(maskShape);

      // TODO(indutny): figure out why mask doesn't work...
      onMask = tf.onesLike(onMask).div(tf.scalar(GRID_DEPTH));

      // Find masks for object presence (`x` is a ground truth)
      const hasObject = x.confidence.mean(-1);
      const noObject = tf.scalar(1).sub(hasObject);

      const objectCount = hasObject.sum(-1).sum(-1);
      const noObjectCount = noObject.sum(-1).sum(-1);

      // Compute losses
      const objLoss = tf.squaredDifference(x.confidence, y.confidence)
          .mul(onMask).sum(-1)
          .mul(hasObject).div(objectCount).sum(-1).sum(-1)
          .mul(tf.scalar(LAMBDA_OBJ));

      const noObjLoss = tf.squaredDifference(x.confidence, y.confidence)
          .mean(-1)
          .mul(noObject).div(noObjectCount).sum(-1).sum(-1)
          .mul(tf.scalar(LAMBDA_NO_OBJ));

      const centerLoss =
          tf.squaredDifference(x.box.center, y.box.center).sum(-1);
      const sizeLoss = tf.squaredDifference(x.box.size, y.box.size).sum(-1);

      // TODO(indutny): use periodic function here
      const angleLoss = tf.squaredDifference(x.box.angle, y.box.angle);

      const boxLoss = centerLoss.add(sizeLoss).add(angleLoss)
          .mul(onMask).sum(-1)
          .mul(hasObject).div(objectCount).sum(-1).sum(-1)
          .mul(tf.scalar(LAMBDA_IOU));

      return objLoss.add(noObjLoss).add(boxLoss);
    });
  }
}
