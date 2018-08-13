import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { load } from './dataset';
import {
  Input, ITrainingPair, TARGET_WIDTH, TARGET_HEIGHT, TARGET_CHANNELS,
  GRID_SIZE, GRID_CHANNELS,
} from './input';
import { Model, GRID_DEPTH } from './model';

const IMAGE_DIR = path.join(__dirname, '..', 'images');
const SAVE_DIR = path.join(__dirname, '..', 'saves');
const SAVE_FILE = path.join(SAVE_DIR, 'model');

async function train() {
  const m = new Model();
  const inputs = await load();

  const validationCount = (inputs.length * 0.1) | 0;
  const trainSrc = inputs.slice(validationCount);

  // TODO(indutny): proper validation
  const validateSrc = inputs.slice(0, Math.min(1, validationCount))
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

  async function predict(file: string, input: Input,
                         image: tf.Tensor, grid: tf.Tensor) {
    const prediction = await (m.model.predict(image) as tf.Tensor).data();

    let rects = input.predictionToRects(prediction, GRID_DEPTH, 0.2);
    let svg = await input.toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}.svg`), svg);

    rects = input.predictionToRects(await grid.data(), GRID_DEPTH);
    svg = await input.toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}_ground.svg`), svg);
  }

  console.log('Randomizing training data...');
  let ts = Date.now();

  const trainInputs = trainSrc.map((input) => input.randomize());
  const train = trainInputs.map((input) => input.toTrainingPair());
  console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));
  const trainingData = tensorify(train);

  const test = tensorify([ trainInputs[0].toTrainingPair() ]);

  console.log('Running fit');
  for (let epoch = 1; epoch < 1000000; epoch += 25) {
    console.log('Epoch %d', epoch);
    ts = Date.now();
    const history = await m.model.fit(
      trainingData.image,
      trainingData.targetGrid,
      {
        initialEpoch: epoch,
        batchSize: 8,
        epochs: epoch + 25,
        callbacks: {
          onBatchEnd: async () => {
            process.stdout.write('.');
          },
          onEpochEnd: async (epoch, logs) => {
            process.stdout.write('\n');
            console.log('epoch %d end %j', epoch, logs);

            await predict('train', trainInputs[0], test.image, test.targetGrid);

            if (validateSrc.length === 1) {
              await predict('validate', validateSrc[0],
                validationData.image, validationData.targetGrid);
            }
          },
        },
      });
    console.log('Took %s sec', ((Date.now() - ts) / 1000).toFixed(2));

    console.log('metrics %j', history.history);
    console.log('memory %j', tf.memory());
    await m.model.save(`file://${SAVE_FILE}`);
  }

  // Clean-up memory?
  tf.dispose(test);
  tf.dispose(trainingData);
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
