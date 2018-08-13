import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

const PRIOR_SIZE = 0.2;

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (Array.isArray(inputs)) {
      inputs = inputs[0];
    }
    const [ center, size, angle, confidence ] =
        tf.split(inputs, [ 2, 2, 2, 1 ], -1);

    return tf.concat([
      tf.sigmoid(center),
      tf.exp(size).mul(tf.scalar(PRIOR_SIZE)),
      tf.softmax(angle, -1),
      tf.sigmoid(confidence),
    ], -1);
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
