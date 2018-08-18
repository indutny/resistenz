import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

// K-means
export const PRIOR_SIZES = [
  0.4647230700573447, 0.19197944960394014,
  0.4700954862219095, 0.16818836604194692,
  0.49641665606713564, 0.19340740753568383,
  0.5922471314865069, 0.2264876137542159,
  0.628718604266798, 0.24043506305528314,
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
