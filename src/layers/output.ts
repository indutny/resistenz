import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (Array.isArray(inputs)) {
      inputs = inputs[0];
    }
    this.invokeCallHook(inputs, kwargs);
    const [ coords, angle, confidence ] = tf.split(inputs, [ 4, 1, 1 ], -1);
    return tf.concat([
      tf.sigmoid(coords),
      angle,
      tf.sigmoid(confidence)
    ], -1);
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
