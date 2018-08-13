import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

// K-means
const PRIOR_SIZES = [
  0.022357168681685333, 0.036641368184605824,
  0.05664291044986954, 0.14495684544563542,
  0.1511164971561785, 0.04987776541909509,
  0.06825699641427217, 0.025930038642782437,
  0.44592230184277365, 0.1599482396657784,
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
