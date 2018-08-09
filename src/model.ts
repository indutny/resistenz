import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

class Model {
  public readonly model: tf.Sequential;

  constructor() {
    const model = tf.sequential();

    // Initial layers
    model.add(tf.layers.conv2d({
      inputShape: [ 416, 416, 3 ],
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
    convPool(3, 256, 2, 2);

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
      filters: 9 * 5,
      activation: 'relu'
    }));

    model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

    this.model = model;
  }
}

const m = new Model();
