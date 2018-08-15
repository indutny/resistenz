import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

// K-means
const PRIOR_SIZES = [
  0.10768077077307367, 0.26221649485269616,
  0.2029919860148871, 0.5090933602720991,
  0.2066486075129398, 0.08824253371241374,
  0.35765892607512195, 0.14442448327435234,
  0.6069002757259446, 0.24344832037024136,
];

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return tf.tidy(() => {
      if (Array.isArray(inputs)) {
        inputs = inputs[0];
      }

      const [ center, size, angle, confidence ] =
          tf.split(inputs, [ 2, 2, 2, 1 ], -1);

      const depth = size.shape[size.shape.length - 2];

      return tf.concat([
        tf.sigmoid(center),
        tf.exp(size).mul(tf.tensor2d(PRIOR_SIZES, [ depth, 2 ])),
        tf.softmax(angle, -1),
        tf.sigmoid(confidence),
      ], -1);
    });
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
