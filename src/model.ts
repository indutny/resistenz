import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';

import { Output } from './layers/output';
import { MobileNetLayer } from './layers/mobilenet';

export const GRID_DEPTH = 5;

const LAMBDA_OBJ = 1;
const LAMBDA_NO_OBJ = 0.5;
const LAMBDA_IOU = 5;

const LR = 1e-4;
const MOMENTUM = 0.9;
const USE_NESTEROV = true;

const EPSILON = tf.scalar(1e-23);
const PI = tf.scalar(Math.PI);

export class Model {
  public readonly model: tf.Sequential;

  constructor(mobilenet: tf.Model) {
    const model = tf.sequential();

    // Just a no-op to specify input layer shape
    model.add(tf.layers.activation({
      inputShape: [ TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ],
      activation: 'linear',
    }));

    model.add(new MobileNetLayer(mobilenet));

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 256,
      activation: 'relu',
      padding: 'same',
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 512,
      activation: 'relu',
      padding: 'same',
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: GRID_CHANNELS * GRID_DEPTH,
    }));

    model.add(tf.layers.reshape({
      targetShape: [ GRID_SIZE, GRID_SIZE, GRID_DEPTH, GRID_CHANNELS ]
    }));

    model.add(new Output({}));

    model.compile({
      loss: (xs, ys) => this.loss(xs, ys),
      optimizer: tf.train.momentum(LR, MOMENTUM, USE_NESTEROV),
    });

    model.summary();

    this.model = model;
  }

  private loss(xs: tf.Tensor, ys: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // shape == [ batch, grid_size, grid_size, grid_depth, grid_channels ]
      function parseBox(out: tf.Tensor) {
        let [ center, size, angle, confidence ] =
            tf.split(out, [ 2, 2, 1, 1 ], -1);

        angle = tf.squeeze(angle, [ angle.rank - 1 ]);
        confidence = tf.squeeze(confidence, [ confidence.rank - 1 ]);

        angle = angle.mul(PI);

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
      const onMask = tf.oneHot(argMax, GRID_DEPTH, 1, 0).cast('float32')
          .reshape(maskShape);

      // Find masks for object presence (`x` is a ground truth)
      const hasObject = x.confidence.mul(onMask);
      const noObject = tf.scalar(1).sub(hasObject);

      // Compute losses
      const objLoss = tf.squaredDifference(tf.scalar(1), y.confidence)
          .mul(hasObject).sum(-1)
          .mul(tf.scalar(LAMBDA_OBJ));

      const noObjLoss = y.confidence.square()
          .mul(noObject).mean(-1)
          .mul(tf.scalar(LAMBDA_NO_OBJ));

      const centerLoss =
          tf.squaredDifference(x.box.center, y.box.center).sum(-1);

      // Use square root of size as in YOLO's paper
      const sizeLoss = tf.squaredDifference(
          x.box.size.sqrt(), y.box.size.sqrt()).sum(-1);

      const angleLoss = tf.sin(x.box.angle.sub(y.box.angle)).square();

      const boxLoss = centerLoss.add(sizeLoss).add(angleLoss)
          .mul(hasObject).sum(-1)
          .mul(tf.scalar(LAMBDA_IOU));

      return objLoss.add(noObjLoss).add(boxLoss);
    });
  }
}
