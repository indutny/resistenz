import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

const PRIOR_SIZES = [
  0.7080671, 0.1955982,
  0.0387221, 0.0358943,
  0.2332013, 0.4570694,
  0.5587609, 0.0560221,
  0.8829163, 0.140842
];

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
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
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
