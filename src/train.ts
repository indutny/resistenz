import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import {
  load, TARGET_WIDTH, TARGET_HEIGHT, GRID_SIZE,
  IDataset
} from './dataset';
import { ITrainingPair } from './input';
import { Model, GRID_DEPTH } from './model';

async function train() {
  const m = new Model();
  const inputs = await load();

  const validationCount = Math.floor(inputs.length * 0.1);
  const trainSrc = inputs.slice(validationCount);
  const validateSrc = inputs.slice(0, validationCount);

  function tensorify(pairs: ReadonlyArray<ITrainingPair>) {
    const xs: tf.Tensor[] = [];
    const ys: tf.Tensor[] = [];

    for (const pair of pairs) {
      xs.push(tf.tensor(pair.rgb));
      ys.push(tf.tensor(pair.grid));
    }

    return [ xs, ys ];
  }

  const validationData =
    tensorify(validateSrc.map((input) => input.toTrainingPair()));

  console.log('Running fit');
  for (let epoch = 1; epoch < 100; i++) {
    const train = trainSrc.map((input) => input.randomize().toTrainingPair());

    const [ xs, ys ] = tensorify(train);
    const history = await m.model.fit(xs, ys, {
      epochs: 1,
      validationData,
    });
    console.log(history);
  }
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
