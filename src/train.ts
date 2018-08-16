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
import { ImagePool } from './image-pool';

const IMAGE_DIR = path.join(__dirname, '..', 'images');
const SAVE_DIR = path.join(__dirname, '..', 'saves');
const SAVE_FILE = path.join(SAVE_DIR, 'model');
const MOBILE_NET =
    path.join(__dirname, '..', 'pretrained', 'mobilenet_224', 'model.json');

interface ITensorifyResult {
  readonly image: tf.Tensor;
  readonly grid: tf.Tensor;
}

type Batch = ReadonlyArray<Input>;

async function *augmentTrain(
    pool: ImagePool,
    batchSize: number = 10,
    minPercent: number = 0.2,
    maxParallelBatches: number = 4) {
  const replacements: Set<number> = new Set();

  const minCount = Math.ceil(pool.size * minPercent);
  for (let i = 0; i < minCount; i++) {
    let index: number;
    do {
      index = (Math.random() * pool.size) | 0;
    } while (replacements.has(index));
    replacements.add(index);
  }

  // Shuffle
  const indices: number[] = [];
  for (let i = 0; i < pool.size; i++) {
    indices.push(i);
  }

  for (let i = indices.length - 2; i >= 0; i--) {
    const j = (Math.random() * i) | 0;

    const t = indices[i];
    indices[i] = indices[j];
    indices[j] = t;
  }

  const pendingBatches: Array<Promise<Batch>> = [];
  let fillIndex = 0;
  for (let i = 0; i < indices.length; i += batchSize) {
    const batch = indices.slice(i, i + batchSize);

    pendingBatches.push(Promise.all(batch.map(async (index) => {
      let res: Input;
      if (replacements.has(index)) {
        res = await pool.randomize(index);
      } else {
        res = await pool.getLast(index);
      }

      return res;
    })));

    if (pendingBatches.length >= maxParallelBatches) {
      yield await pendingBatches.shift()!;
    }
  }

  while (pendingBatches.length > 0) {
    yield await pendingBatches.shift()!;
  }
}

function tensorify(pairs: ReadonlyArray<ITrainingPair>): ITensorifyResult {
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
    grid: tf.tidy(() => {
      const shallow = tf.tensor(ys, [
          pairs.length, GRID_SIZE, GRID_SIZE, 1, GRID_CHANNELS ]);
      return shallow.tile([ 1, 1, 1, GRID_DEPTH, 1 ]);
    }),
  };
}

function disposeTensorify(result: ITensorifyResult) {
  tf.dispose(result.image);
  tf.dispose(result.grid);
}

async function train() {
  // const mobilenet = await tf.loadModel(`file://${MOBILE_NET}`);

  const m = new Model();
  const dataset = await load();

  const pool = new ImagePool(dataset.train);

  const trainSrc = dataset.train;

  console.log(`Validate count: ${dataset.validate.length}`);

  const validateSrc = dataset.validate
    .map((val) => val.resize());

  console.time('validation tensorify');
  const validation = {
    all: tensorify(validateSrc.map((input) => input.toTrainingPair())),
  };
  console.timeEnd('validation tensorify');

  async function predict(file: string, input: Input, entry: ITensorifyResult) {
    const predictionTensor = m.model.predict(entry.image) as tf.Tensor;
    const prediction = await predictionTensor.data();
    predictionTensor.dispose();

    let rects = input.predictionToRects(prediction, GRID_DEPTH, 0.2);
    let svg = await input.toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}.svg`), svg);

    rects = input.predictionToRects(await entry.grid.data(), GRID_DEPTH);
    svg = await input.toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}_ground.svg`), svg);
  }

  let iterator: AsyncIterableIterator<Batch> | undefined;

  console.log('Running fit');
  for (let epoch = 1; epoch < 1000000; epoch += 1) {
    console.log('Epoch %d', epoch);

    console.time('fit');
    const losses: number[] = [];

    if (!iterator) {
      iterator = augmentTrain(pool);
    }
    for await (const batch of iterator!) {
      const batchTensor = tensorify(
        batch.map((input) => input.toTrainingPair()));

      const history = await m.model.fit(batchTensor.image, batchTensor.grid, {
        shuffle: false,

        initialEpoch: epoch,
        batchSize: batch.length,
        epochs: epoch + 1,
      });
      process.stdout.write('.');

      losses.push((history.history.loss as number[])[0]);
      disposeTensorify(batchTensor);
    }

    // Restart iterator as soon as possible
    iterator = augmentTrain(pool);

    process.stdout.write('\n');
    console.timeEnd('fit');
    console.log(`losses=${JSON.stringify(losses)}`);

    let valLoss = await m.model.evaluate(validation.all.image,
                                         validation.all.grid, {
      batchSize: 10,
    });
    if (!Array.isArray(valLoss)) {
      valLoss = [ valLoss ];
    }

    if (valLoss.length > 0) {
      const loss = await valLoss[0].data();
      console.log(`val_loss=${loss[0]}`);
    }
    tf.dispose(valLoss);

    {
      const src = await pool.getLast((Math.random() * pool.size) | 0);
      const single = tensorify([ src.toTrainingPair() ]);
      await predict('train', src, single);
      disposeTensorify(single);
    }

    if (validateSrc.length >= 1) {
      const src = validateSrc[(Math.random() * validateSrc.length) | 0];
      const single = tensorify([ src.toTrainingPair() ]);
      await predict('validate', src, single);
      disposeTensorify(single);
    }

    console.log('memory %j', tf.memory());
    await m.model.save(`file://${SAVE_FILE}`);
  }

  // Clean-up memory?
  disposeTensorify(validation.all);

  pool.close();
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
