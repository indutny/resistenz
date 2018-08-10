import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';

export const GRID_DEPTH = 5;

export class Model {
  public readonly model: tf.Model;

  constructor() {
    const image = tf.input({
      name: 'image',
      shape: [ TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ],
    });

    const targetGrid = tf.input({
      name: 'target_grid',
      shape: [ GRID_SIZE, GRID_SIZE, GRID_CHANNELS ],
    });

    const training = tf.input({
      name: 'training',
      batchShape: [ 1 ],
    });

    // TODO(indutny): replace `any` with real type
    let x: any = image;

    function convPool(x: any,
                      kernel: number, filters: number, pool: number,
                      stride: number): any {
      x = tf.layers.batchNormalization({}).apply(x, { training });

      x = tf.layers.conv2d({
        kernelSize: kernel,
        filters,
        activation: 'relu',
      }).apply(x);

      return tf.layers.maxPooling2d({
        poolSize: [ pool, pool ],
        strides: [ stride, stride ],
      }).apply(x);
    }

    x = convPool(x, 3, 16, 2, 2);
    x = convPool(x, 3, 32, 2, 2);
    x = convPool(x, 3, 64, 2, 2);
    x = convPool(x, 3, 128, 2, 2);

    // Final mapping
    for (let i = 0; i < 2; i++) {
      x = tf.layers.conv2d({
        kernelSize: 3,
        filters: 512,
        activation: 'relu'
      }).apply(x);
    }

    x = tf.layers.conv2d({
      kernelSize: 1,
      filters: GRID_DEPTH * GRID_CHANNELS,
      activation: 'relu'
    }).apply(x);

    x = tf.layers.reshape({
      targetShape: [ GRID_SIZE, GRID_SIZE, GRID_DEPTH, GRID_CHANNELS ],
    }).apply(x);

    const output = x;

    const model = tf.model({
      inputs: [ image, targetGrid, training ],
      outputs: output,
    });

    model.compile({ loss: (xs, ys) => this.loss(xs, ys), optimizer: 'adam' });

    this.model = model;
  }

  private loss(xs: tf.Tensor, ys: tf.Tensor): tf.Tensor {
    ys.print();
    return tf.tensor(0);
  }
}
