import * as tf from '@tensorflow/tfjs';
import { Buffer } from 'buffer';
import jimp = require('jimp');

import Tensor = tf.Tensor;

export class SaveImage extends tf.layers.Layer {
  public call(inputs: Tensor|Tensor[], kwargs: any): Tensor|Tensor[] {
    return tf.tidy(() => {
      if (Array.isArray(inputs)) {
        inputs = inputs[0];
      }

      const width = inputs.shape[2];
      const height = inputs.shape[1];

      const raw = inputs.dataSync();

      const data = Buffer.alloc(width * height * 4);
      for (let i = 0, j = 0; i < width * height * 3; i += 3, j += 4) {
        data[j + 0] = raw[i + 0] * 255;
        data[j + 1] = raw[i + 1] * 255;
        data[j + 2] = raw[i + 2] * 255;
        data[j + 3] = 255;
      }

      const img = new jimp({ data, width, height });
      img.write('/tmp/1.png');

      return inputs;
    });
  }

  public getClassName(): string {
    return 'save_image';
  }
}
