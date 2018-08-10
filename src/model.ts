import * as tf from '@tensorflow/tfjs';

import {
  TARGET_WIDTH, TARGET_HEIGHT, GRID_SIZE, GRID_DIMS
} from './dataset';

export const GRID_DEPTH = 5;

export class Model {
  public readonly model: tf.Sequential;

  constructor() {
    const model = tf.sequential();

    // Initial layers
    model.add(tf.layers.conv2d({
      inputShape: [ TARGET_WIDTH, TARGET_HEIGHT, 3 ],
      kernelSize: 3,
      filters: 16,
      activation: 'relu',
    }));

    model.add(tf.layers.maxPooling2d({
      poolSize: [ 2, 2 ],
      strides: [ 2, 2 ],
    }));

    function convPool(kernel: number, filters: number, pool: number,
                      stride: number) {
      model.add(tf.layers.batchNormalization({}));

      model.add(tf.layers.conv2d({
        kernelSize: kernel,
        filters,
        activation: 'relu',
      }));

      model.add(tf.layers.maxPooling2d({
        poolSize: [ pool, pool ],
        strides: [ stride, stride ],
      }));
    }

    convPool(3, 32, 2, 2);
    convPool(3, 64, 2, 2);
    convPool(3, 128, 2, 2);

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 512,
      activation: 'relu'
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 512,
      activation: 'relu'
    }));

    model.add(tf.layers.conv2d({
      kernelSize: 1,
      filters: 6 * GRID_DEPTH,
      activation: 'relu'
    }));

    model.compile({ loss: (xs, ys) => this.loss(xs, ys), optimizer: 'adam' });

    this.model = model;
  }

  private loss(xs: tf.Tensor, ys: tf.Tensor): tf.Tensor {
    const deepGrid = xs.reshape([
      xs.shape[0],
      GRID_SIZE,
      GRID_SIZE,
      GRID_DEPTH,
      GRID_DIMS,
    ]);
    const actualGrid = ys.reshape([
      ys.shape[0],
      GRID_SIZE,
      GRID_SIZE,
      GRID_DIMS,
    ]);
    actualGrid.print();
    return tf.tensor(0);
  }
}
