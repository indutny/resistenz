import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { load } from './dataset';
import { TARGET_WIDTH, TARGET_HEIGHT } from './input';
import { IOrientedRect } from './utils';
import { GRID_DEPTH } from './model';

const LR = 0.001;

async function generate() {
  const inputs = await load();

  const rects: IOrientedRect[] = [];
  for (const input of inputs) {
    for (const rect of input.resize().computeRects()) {
      rects.push(rect);
    }
  }

  const sizesSrc = new Float32Array(rects.length * 2);
  let off = 0;
  for (const rect of rects) {
    sizesSrc[off++] = rect.width / TARGET_WIDTH;
    sizesSrc[off++] = rect.height / TARGET_HEIGHT;
  }

  const centers = tf.variable(tf.randomUniform([ GRID_DEPTH, 2 ]),
    true, 'points');

  const optimizer = tf.train.sgd(LR);

  for (let i = 0; i < 10000; i++) {
    const lossValue = optimizer.minimize(() => {
      const sizes =
          tf.tensor2d(sizesSrc, [ rects.length, 2 ]).expandDims(1);
      const expandedCenters = centers.expandDims(0);

      const distances = sizes.sub(expandedCenters).square().sum(-1);

      const argMin = distances.argMin(-1).flatten();
      const oneHot = tf.oneHot(argMin, GRID_DEPTH).cast('float32')
          .reshape(distances.shape);

      return distances.mul(oneHot).sum(-1).mean() as tf.Scalar;
    }, true, [ centers ]);
    if (lossValue) {
      lossValue.print();
      lossValue.dispose();
    }
  }

  centers.print();
}

generate().catch((e) => {
  console.error(e);
  process.exit(1);
});
