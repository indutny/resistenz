#!/usr/bin/env npx ts-node
import { promisify } from 'util';

import { load } from '../src/dataset';
import { TARGET_WIDTH, TARGET_HEIGHT } from '../src/input';
import { IOrientedRect } from '../src/utils';
import { GRID_DEPTH } from '../src/model';
import { ImagePool } from '../src/image-pool';

const kmeans = require('node-kmeans');

const LR = 0.001;
const RANDOM_COUNT = 2;

const MAX_PARALLEL = 8;

async function generate() {
  const dataset = await load();

  const pool = new ImagePool(dataset.train);

  const rects: IOrientedRect[] = [];

  const indices = [];
  for (let i = 0; i < pool.size; i++) {
    indices.push(i);
  }

  for (let i = 0; i < indices.length; i += MAX_PARALLEL) {
    console.log('%d/%d', i, indices.length);
    await Promise.all(indices.slice(i, i + MAX_PARALLEL).map(async (i) => {
      for (let i = 0; i < RANDOM_COUNT; i++) {
        const input = await pool.randomize(i);
        for (const rect of input.computeRects()) {
          rects.push(rect);
        }
        process.stderr.write('.');
      }
    }));
  }

  const points = [];
  for (const rect of rects) {
    const width = rect.width / TARGET_WIDTH;
    const height = rect.height / TARGET_HEIGHT;

    points.push([ width, height ]);
  }

  const clusterize = promisify(kmeans.clusterize);

  // TODO(indutny): apparently this is very wrong, we should clusterize by
  // IoU!!!
  const res = await clusterize.call(kmeans, points, { k: GRID_DEPTH });

  const centers = res.map((obj: any) => obj.centroid);

  console.log(centers);

  pool.close();
}

generate().catch((e) => {
  console.error(e);
  process.exit(1);
});
