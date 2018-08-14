import * as assert from 'assert';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import jimp = require('jimp');

import { Polygon, IPoint, IOrientedRect } from './utils';
import { Input, TARGET_WIDTH, TARGET_HEIGHT } from './input';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const IMAGE_DIR = path.join(DATASET_DIR, 'processed');

export async function load(): Promise<ReadonlyArray<Input>> {
  const dir = await promisify(fs.readdir)(IMAGE_DIR);

  let files = dir.filter((file) => /\.json$/.test(file));

  // Stupid, but stable semi-random sort
  files = files.map((file) => {
    return {
      file,
      hash: crypto.createHash('sha256').update(file).digest('hex'),
    };
  }).sort((a, b) => {
    return a.hash === b.hash ? 0 : a.hash < b.hash ? -1 : 1;
  }).map((entry) => entry.file);

  console.log('Loading images...');
  let done = 0;
  return await Promise.all(files.map(async (labelsFile) => {
    const imageFile = labelsFile.replace(/\.json$/, '') + '.jpg';
    const image = await jimp.read(path.join(IMAGE_DIR, imageFile));
    const rawLabels = await promisify(fs.readFile)(
        path.join(IMAGE_DIR, labelsFile));

    const labels = JSON.parse(rawLabels.toString());

    console.log(`${done++}/${files.length}`);
    return new Input(image, labels.polygons);
  }));
}

/*
load().then(async (inputs) => {
  const random = inputs[0];

  let svg = await random.toSVG();
  fs.writeFileSync('/tmp/1.svg', svg);

  const grid = random.toTrainingPair().grid;

  svg = await random.toSVG(random.predictionToRects(grid, 1));
  fs.writeFileSync('/tmp/2.svg', svg);
});
 */
