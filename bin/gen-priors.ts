#!/usr/bin/env npx ts-node
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import imageSize = require('image-size');

import { IMAGE_DIR } from '../src/dataset';
import { Polygon, polygonToRect } from '../src/utils';
import { GRID_DEPTH } from '../src/model';

const kmeans = require('node-kmeans');

const LR = 0.001;
const RANDOM_COUNT = 2;

const MAX_PARALLEL = 8;

async function generate() {
  let files = await promisify(fs.readdir)(IMAGE_DIR);
  files = files.filter((file) => /\.json$/.test(file));

  const rects = await Promise.all(files.map(async (file) => {
    const jsonFile = path.join(IMAGE_DIR, file);
    const jpegFile = jsonFile.replace(/\.json$/, '.jpg');
    const [ image, json ] = await Promise.all([
      promisify(fs.readFile)(jpegFile),
      promisify(fs.readFile)(jsonFile),
    ]);

    const size = imageSize(image);

    const minDim = Math.min(size.width, size.height);

    const polygons = JSON.parse(json.toString()).polygons as Polygon[];
    const rects = polygons.map((poly) => {
      return poly.map((p) => {
        return { x: p.x / minDim, y: p.y / minDim };
      });
    }).map((poly) => polygonToRect(poly));

    return rects.map((rect) => {
      return [ rect.width, rect.height ];
    });
  }));

  const points = rects.reduce((acc, curr) => acc.concat(curr));
  const clusterize = promisify(kmeans.clusterize);

  const res = await clusterize.call(kmeans, points, {
    k: GRID_DEPTH,
    distance: (a: [ number, number ], b: [ number, number ]) => {
      const aArea = a[0] * a[1];
      const bArea = b[0] * b[1];
      const intersection = Math.min(a[0], b[0]) * Math.min(a[1], b[1]);
      const iou = intersection / (aArea + bArea - intersection + 1e-23);
      return 1 - iou;
    },
  });

  const centers = res.map((obj: any) => obj.centroid);

  centers.sort((a: [ number, number ], b: [ number, number ]) => {
    return a[0] - b[0];
  });
  console.log(centers);
}

generate().catch((e) => {
  console.error(e);
  process.exit(1);
});
