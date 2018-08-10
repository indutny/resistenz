import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import {
  load, TARGET_WIDTH, TARGET_HEIGHT, GRID_SIZE, GRID_DEPTH,
  IDataset
} from './dataset';
import { Model } from './model';

async function train() {
  const m = new Model();
  const dataset = await load();

  function flatten(arr: ReadonlyArray<ReadonlyArray<number>>): Float32Array {
    let res = new Float32Array(arr[0].length * arr.length);
    let off = 0;
    for (const entry of arr) {
      for (let i = 0; i < entry.length; i++) {
        res[off++] = entry[i];
      }
    }
    return res;
  }

  const validationCount = Math.floor(dataset.rgbs.length * 0.1);
  const train: IDataset = {
    rgbs: dataset.rgbs.slice(validationCount),
    images: dataset.images.slice(validationCount),
    grids: dataset.grids.slice(validationCount),
  };
  const validate: IDataset = {
    rgbs: dataset.rgbs.slice(0, validationCount),
    images: dataset.images.slice(0, validationCount),
    grids: dataset.grids.slice(0, validationCount),
  };

  function tensorifyDataset(dataset: IDataset): [ tf.Tensor, tf.Tensor ] {
    const images = tf.tensor(flatten(dataset.rgbs))
        .reshape([ dataset.images.length, TARGET_WIDTH, TARGET_HEIGHT, 3 ]);
    const grids = tf.tensor(flatten(dataset.grids))
        .reshape([ dataset.grids.length, GRID_SIZE, GRID_SIZE, GRID_DEPTH * 9 ]);

    return [ images, grids ];
  }

  console.log('Preparing train dataset...');
  const [ xs, ys ] = tensorifyDataset(train);

  console.log('Preparing validate dataset...');
  const validationData = tensorifyDataset(validate);

  const example = tensorifyDataset({
    rgbs: [ validate.rgbs[0] ],
    images: [ validate.images[0] ],
    grids: [ validate.grids[0] ],
  });

  console.log('Running fit');
  await m.model.fit(xs, ys, {
    epochs: 100,
    validationData,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        console.log(`Epoch ${epoch}: log = %j`, log);

        const prediction = m.model.predict(example[0]) as tf.Tensor;
        const polygons = await prediction.data();

        const image = validate.images[0].clone();

        for (let i = 0; i < polygons.length; i += 9) {
          const prob = polygons[i + 8];

          const gridOff = (i / (9 * GRID_DEPTH)) | 0;
          const gridX = (gridOff / GRID_SIZE) | 0;
          const gridY = gridOff % GRID_SIZE;

          if (prob > 0.1) {
            console.log(prob, polygons.slice(i, i + 7));
          }
        }
      }
    }
  });
}

train().then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
