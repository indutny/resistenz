import * as tf from '@tensorflow/tfjs';

import Tensor = tf.Tensor;

const NOISE_LEVEL = 0.5;

export class Noise extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return tf.tidy(() => {
      if (Array.isArray(inputs)) {
        inputs = inputs[0];
      }

      if (!kwargs.training) {
        return inputs;
      }

      const noise = tf.randomNormal(inputs.shape, 0, 0.2).mul(
          kwargs.training.cast('float32'));

      return inputs.add(noise);
    });
  }

  public getClassName(): string {
    return 'resistenz_noise';
  }
}
