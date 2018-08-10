import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { load } from './dataset';
import {
  ITrainingPair, TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';
import { Model, GRID_DEPTH } from './model';

async function train() {
  const m = new Model();
  const inputs = await load();

  const validationCount = Math.floor(inputs.length * 0.1);
  const trainSrc = inputs.slice(validationCount);
  const validateSrc = inputs.slice(0, validationCount)
    .map((val) => val.resize());

  function tensorify(pairs: ReadonlyArray<ITrainingPair>) {
    const xs = new Float32Array(
      pairs.length * TARGET_WIDTH * TARGET_HEIGHT * TARGET_CHANNELS);
    const ys = new Float32Array(
      pairs.length * GRID_SIZE * GRID_SIZE * GRID_CHANNELS);

    let offX = 0;
    let offY = 0;
    for (const pair of pairs) {
      for (let i = 0; i < pair.rgb.length; i++) {
        xs[offX++] = pair.rgb[i];
      }
      for (let i = 0; i < pair.grid.length; i++) {
        ys[offY++] = pair.grid[i];
      }
    }

    return [
      tf.tensor(xs, [
        pairs.length, TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ]),
      tf.tensor(ys, [
        pairs.length, GRID_SIZE, GRID_SIZE, GRID_CHANNELS ])
    ];
  }

  const [ valXs, valYs ] =
    tensorify(validateSrc.map((input) => input.toTrainingPair()));

  console.log('Running fit');
  for (let epoch = 1; epoch < 100; epoch++) {
    console.log('Randomizing training data...');
    let ts = Date.now();
    const train = trainSrc.map((input) => input.randomize().toTrainingPair());
    console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));

    console.log('Epoch %d', epoch);
    ts = Date.now();
    const [ xs, ys ] = tensorify(train);
    const history = await m.model.fit(xs, ys, {
      epochs: 1,
      validationData: [ valXs, valYs ],
    });
    console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));
    console.log(history);

    // Clean-up memory?
    tf.dispose(xs);
    tf.dispose(ys);
  }
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
