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

const AUGMENT_MULTIPLY = 1;

async function augmentTrain(pool: ImagePool,
    src: ReadonlyArray<Input>,
    list: Input[], minPercent: number = 0.25) {

  const targetSize = src.length * AUGMENT_MULTIPLY;
  const minCount =
      Math.max(targetSize - list.length, list.length * minPercent) | 0;

  // Add random entries
  let done = 0;
  await Promise.all(new Array(minCount).fill(0).map(async () => {
    const index = (src.length * Math.random()) | 0;
    list.push(await pool.randomize(src[index]));
    done++;
    if (done % 100 === 0 || done === minCount) {
      console.log(`${done}/${minCount}`);
    }
  }));

  // Remove entries from the list
  while (list.length > targetSize) {
    list.shift();
  }
}

async function randomizeInputs(pool: ImagePool, src: ReadonlyArray<Input>) {
  let done = 0;
  return Promise.all(src.map(async (input) => {
    const res = await pool.randomize(input);
    done++;
    if (done % 100 === 0 || done === src.length) {
      console.log(`${done}/${src.length}`);
    }
    return res;
  }));
}

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

async function train() {
  const pool = new ImagePool();

  const mobilenet = await tf.loadModel(`file://${MOBILE_NET}`);

  const m = new Model(mobilenet);
  const dataset = await load();

  const trainSrc = dataset.train;

  const validateSrc = dataset.validate
    .map((val) => val.resize());

  console.time('validation tensorify');
  const validation = {
    all: tensorify(validateSrc.map((input) => input.toTrainingPair())),
    single: validateSrc.length ?
      tensorify([ validateSrc[0].toTrainingPair() ]) : undefined,
  };
  console.timeEnd('validation tensorify');

  async function predict(file: string, input: Input,
                         image: tf.Tensor, grid: tf.Tensor) {
    const prediction = await (m.model.predict(image) as tf.Tensor).data();

    let rects = input.predictionToRects(prediction, GRID_DEPTH, 0.2);
    let svg = await input.toSVG(rects);
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}.svg`), svg);

    svg = await input.toSVG();
    fs.writeFileSync(path.join(IMAGE_DIR, `${file}_ground.svg`), svg);
  }

  // Shared training data
  const trainInputs = await randomizeInputs(pool, trainSrc);
  const train = trainInputs.map((input) => input.toTrainingPair());
  const training = {
    all: tensorify(train),
    single: tensorify([ train[train.length - 1] ]),
  };

  console.log('Running fit');
  for (let epoch = 1; epoch < 1000000; epoch += 100) {
    console.log('Epoch %d', epoch);

    /*
    console.log('Randomizing training data... [%d]', trainSrc.length);
    console.time('randomize');

    await augmentTrain(pool, trainSrc, trainInputs);

    console.timeEnd('randomize');

    console.log('Translating to tensors...');
    console.time('tensorify');
    const train = trainInputs.map((input) => input.toTrainingPair());
    const training = {
      all: tensorify(train),
      single: tensorify([ train[0] ]),
    };

    console.timeEnd('tensorify');
    */

    console.time('fit');
    const history = await m.model.fit(
      training.all.image,
      training.all.targetGrid,
      {
        initialEpoch: epoch,
        batchSize: 10,
        epochs: epoch + 100,
        validationData: validateSrc.length >= 1 ?
          [ validation.all.image, validation.all.targetGrid ] : undefined,
        callbacks: {
          onBatchEnd: async () => {
            process.stdout.write('.');
          },
          onEpochEnd: async (epoch, logs) => {
            process.stdout.write('\n');
            console.log('epoch %d end %j', epoch, logs);

            await predict('train', trainInputs[trainInputs.length - 1],
              training.single.image, training.single.targetGrid);

            if (validateSrc.length >= 1) {
              await predict('validate', validateSrc[0],
                validation.single!.image, validation.single!.targetGrid);
            }
          },
        },
      });
    console.timeEnd('fit');

    console.log('memory %j', tf.memory());
    await m.model.save(`file://${SAVE_FILE}`);
  }

  // Clean-up memory?
  tf.dispose(training.single);
  tf.dispose(training.all);

  // Clean-up memory?
  if (validation.single) {
    tf.dispose(validation.single);
  }
  tf.dispose(validation.all);

  pool.close();
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
