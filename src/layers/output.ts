import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

export class Output extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (Array.isArray(inputs)) {
      inputs = inputs[0];
    }
    this.invokeCallHook(inputs, kwargs);
    const [ center, size, angle, confidence ] =
        tf.split(inputs, [ 2, 2, 1, 1 ], -1);

    const depth = inputs.shape[inputs.shape.length - 2];
    const anglePrior = tf.linspace(0, 0.5, depth)
      .reshape(inputs.shape.slice(0, -2).fill(1).concat(depth)).expandDims(-1);

    return tf.concat([
      tf.sigmoid(center),
      tf.exp(size),
      angle.add(anglePrior),
      tf.sigmoid(confidence),
    ], -1);
  }

  public getClassName(): string {
    return 'resistenz_output';
  }
}
