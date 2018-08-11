import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { load } from './dataset';
import {
  ITrainingPair, TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';
import { Model, GRID_DEPTH } from './model';

const IMAGE_DIR = path.join(__dirname, '..', 'images');

async function train() {
  const m = new Model();
  const inputs = await load();

  const validationCount = Math.floor(inputs.length * 0.1);
  let trainSrc = inputs.slice(validationCount);
  trainSrc = trainSrc.concat(trainSrc);
  trainSrc = trainSrc.concat(trainSrc);

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

    const image = tf.tensor(xs, [
        pairs.length, TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS ]);

    return {
      image,
      targetGrid: tf.tidy(() => {
        const shallow = tf.tensor(ys, [
            pairs.length, GRID_SIZE, GRID_SIZE, 1, GRID_CHANNELS ]);
        return shallow.tile([ 1, 1, 1, GRID_DEPTH, 1 ]);
      }),
    };
  }

  const validationData =
    tensorify(validateSrc.map((input) => input.toTrainingPair()));

  console.log('Running fit');
  for (let epoch = 1; epoch < 1000; epoch++) {
    console.log('Randomizing training data...');
    let ts = Date.now();

    const trainInputs = trainSrc.map((input) => input.randomize());
    const train = trainInputs.map((input) => input.toTrainingPair());
    console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));

    console.log('Epoch %d', epoch);
    ts = Date.now();
    const trainingData = tensorify(train);
    const history = await m.model.fit(
      trainingData.image,
      trainingData.targetGrid,
      {
        initialEpoch: epoch,
        epochs: epoch + 1,
      });
    console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));

    console.log('metrics %j', history.history);
    console.log('memory %j', tf.memory());

    const test = tensorify([ trainInputs[0].toTrainingPair() ]);
    const prediction = await (m.model.predict(test.image) as tf.Tensor).data();

    let rects = trainInputs[0].predictionToRects(prediction, GRID_DEPTH, 0.2);
    let svg = await trainInputs[0].toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, 'train.svg'), svg);

    rects = trainInputs[0].predictionToRects(await test.targetGrid.data(), GRID_DEPTH);
    svg = await trainInputs[0].toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, 'train_ground.svg'), svg);

    // Clean-up memory?
    tf.dispose(test);
    tf.dispose(trainingData);
  }
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
