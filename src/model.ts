import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';

import { Noise } from './layers/noise';
import { Output, PRIOR_SIZES } from './layers/output';
import { MobileNetLayer } from './layers/mobilenet';

export const GRID_DEPTH = 5;

const LAMBDA_OBJ = 1;
const LAMBDA_NO_OBJ = 0.5;
const LAMBDA_COORD = 5;

const WEIGHT_DECAY = 0.0005;

const IOU_THRESHOLD = 0.7;

const LR = 1e-2;
const MOMENTUM = 0.9;
const USE_NESTEROV = true;

export class Model {
  public readonly model: tf.Sequential;

  constructor() {
    const model = tf.sequential();

    // Just a no-op to specify input layer shape
    model.add(tf.layers.activation({
      inputShape: [ TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ],
      activation: 'linear',
    }));

    model.add(new Noise({}));

    // model.add(new MobileNetLayer(mobilenet));

    function convPool(kernel: number, filters: number, pool: number,
                      stride: number) {
      model.add(tf.layers.conv2d({
        kernelSize: kernel,
        filters,
      }));

      model.add(tf.layers.batchNormalization({}));

      model.add(tf.layers.leakyReLU());

      model.add(tf.layers.maxPooling2d({
        poolSize: [ pool, pool ],
        strides: [ stride, stride ],
        padding: 'same',
      }));
    }

    // TinyYOLO v3 (more or less)
    convPool(3, 16, 2, 2);
    convPool(3, 32, 2, 2);
    convPool(3, 64, 2, 2);
    convPool(3, 128, 2, 2);
    convPool(3, 256, 2, 2);
    convPool(3, 512, 2, 1);

    function convBN(kernel: number, filters: number,
                    activation: string = 'leaky') {
      model.add(tf.layers.conv2d({
        kernelSize: kernel,
        filters,
      }));

      model.add(tf.layers.batchNormalization({}));
      if (activation === 'leaky') {
        model.add(tf.layers.leakyReLU());
      } else {
        model.add(tf.layers.activation({ activation }));
      }
    }

    // Detection layer
    convBN(1, 256);
    convBN(3, 512);
    convBN(1, GRID_DEPTH * GRID_CHANNELS, 'linear');

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
            tf.split(out, [ 2, 2, 2, 1 ], -1);

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

      const epsilon = tf.scalar(1e-23);

      // Calculate Intersection over Union
      const iou = interArea.div(unionArea.add(epsilon));

      // Multiply by angle difference
      // NOTE: (cos x - cos y)^2 + (sin x - sin y)^2 = 2 (1 - cos (x - y))
      const angleLoss = tf.squaredDifference(x.box.angle, y.box.angle).sum(-1)
          .div(tf.scalar(2));
      const angleMul = tf.scalar(1).sub(angleLoss);
      const angleIOU = iou.mul(angleMul);

      // Mask out maximum angleIOU in each grid group, and everything higher
      // than threshold
      const thresholdMask = angleIOU.greaterEqual(tf.scalar(IOU_THRESHOLD))
          .cast('float32');

      const argMax = angleIOU.argMax(-1).flatten();
      const maskShape = angleIOU.shape;
      const maxMask = tf.oneHot(argMax, GRID_DEPTH, 1, 0)
          .cast('float32').reshape(maskShape);

      const onMask = tf.maximum(thresholdMask, maxMask);

      // Find masks for object presence (`x` is a ground truth)
      const hasObject = x.confidence.mul(onMask);
      const noObject = tf.scalar(1).sub(hasObject);

      const objCount = hasObject.sum(-1).sum(-1).sum(-1).add(epsilon);
      const noObjCount = noObject.sum(-1).sum(-1).sum(-1).add(epsilon)

      // Compute losses
      const objLoss = tf.squaredDifference(tf.scalar(1), y.confidence)
          .mul(hasObject).sum(-1)
          .sum(-1).sum(-1).div(objCount)
          .mul(tf.scalar(LAMBDA_OBJ));

      const noObjLoss = y.confidence.square()
          .mul(noObject).sum(-1)
          .sum(-1).sum(-1).div(noObjCount)
          .mul(tf.scalar(LAMBDA_NO_OBJ));

      const centerLoss =
          tf.squaredDifference(x.box.center, y.box.center).sum(-1);

      const sizeLoss = tf.squaredDifference(
        x.box.size.sqrt(), y.box.size.sqrt()).sum(-1);

      const boxLoss = centerLoss.add(sizeLoss).add(angleLoss)
          .mul(hasObject).sum(-1)
          .sum(-1).sum(-1).div(objCount)
          .mul(tf.scalar(LAMBDA_COORD));

      const weights = this.model.trainableWeights.filter((weight) => {
        return /conv2d/.test(weight.name);
      }).map((weight) => {
        return weight.read().cast('float32');
      });
      const decayLoss = weights.reduce((acc, weight) => {
        return acc.add(weight.square().mean());
      }, tf.scalar(0)).mul(tf.scalar(WEIGHT_DECAY / 2));

      return objLoss.add(noObjLoss).add(boxLoss).add(decayLoss);
    });
  }
}
