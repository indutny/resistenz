import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

// K-means
export const PRIOR_SIZES = [
  0.10768077077307367, 0.26221649485269616,
  0.2029919860148871, 0.5090933602720991,
  0.2066486075129398, 0.08824253371241374,
  0.35765892607512195, 0.14442448327435234,
  0.6069002757259446, 0.24344832037024136,
];

function revSigmoid(value: number): number {
  return -Math.log(1 / value - 1);
}

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return tf.tidy(() => {
      if (Array.isArray(inputs)) {
        inputs = inputs[0];
      }

      const [ center, size, angle, confidence ] =
          tf.split(inputs, [ 2, 2, 2, 1 ], -1);

      const depth = size.shape[size.shape.length - 2];

      const priorSizes = tf.tensor2d(PRIOR_SIZES.map(revSigmoid), [ depth, 2 ]);

      const angleNorm = angle.square().sum(-1).sqrt().add(tf.scalar(1e-23));

      return tf.concat([
        tf.sigmoid(center),

        // Offset sigmoid
        tf.sigmoid(size.add(priorSizes)),
        angle.div(angleNorm),
        tf.sigmoid(confidence),
      ], -1);
    });
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
