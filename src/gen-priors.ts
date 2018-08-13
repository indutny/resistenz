import { promisify } from 'util';

import { load } from './dataset';
import { TARGET_WIDTH, TARGET_HEIGHT } from './input';
import { IOrientedRect } from './utils';
import { GRID_DEPTH } from './model';

const kmeans = require('node-kmeans');

const LR = 0.001;
const RANDOM_COUNT = 2;

async function generate() {
  const inputs = await load();

  const rects: IOrientedRect[] = [];
  let counter = 0;
  for (const input of inputs) {
    console.error('image: %d', counter++);
    for (let i = 0; i < RANDOM_COUNT; i++) {
      for (const rect of input.randomize().computeRects()) {
        rects.push(rect);
      }
      process.stderr.write('.');
    }
    process.stderr.write('\n');
  }

  const points = [];
  for (const rect of rects) {
    const width = rect.width / TARGET_WIDTH;
    const height = rect.height / TARGET_HEIGHT;

    points.push([ width, height ]);
  }

  const clusterize = promisify(kmeans.clusterize);
  const res =await clusterize.call(kmeans, points, { k: GRID_DEPTH });

  const centers = res.map((obj: any) => obj.centroid);

  console.log(centers);
}

generate().catch((e) => {
  console.error(e);
  process.exit(1);
});
