import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

export class MobileNetLayer extends tf.layers.Layer {
  private readonly net: tf.Model;

  constructor(mobilenet: tf.Model) {
    super({});

    const layer = mobilenet.getLayer('conv_pw_13_relu');

    this.net = tf.model({
      inputs: mobilenet.inputs,
      outputs: layer.output,
    });
  }

  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    if (Array.isArray(inputs)) {
      inputs = inputs[0];
    }

    return this.net.predict(inputs);
  }

  public get trainableWeights() {
    return this.net.trainableWeights;
  }

  public computeOutputShape(inputShape: tf.Shape|tf.Shape[]) {
    return this.net.outputs[0].shape;
  }

  public getClassName(): string {
    return 'mobilenet';
  }
}
