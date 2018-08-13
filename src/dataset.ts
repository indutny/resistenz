import * as assert from 'assert';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import jimp = require('jimp');

import { Polygon, IPoint, IOrientedRect } from './utils';
import { Input, TARGET_WIDTH, TARGET_HEIGHT } from './input';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export async function load(): Promise<ReadonlyArray<Input>> {
  const json = await util.promisify(fs.readFile)(LABELS);
  const globalGeos: Map<string, ReadonlyArray<Polygon>> = new Map();

  JSON.parse(json.toString()).forEach((entry: any) => {
    if (!entry || !entry.Label || !entry.Label.Resistor) {
      return;
    }

    const geometry = entry.Label.Resistor.filter((entry: any) => {
      return entry.geometry.length === 4;
    });

    if (geometry.length === 0) {
      return;
    }

    globalGeos.set(entry['External ID'], geometry.map((entry: any) => {
      return entry.geometry;
    }));
  });

  const dir = await util.promisify(fs.readdir)(IMAGE_DIR);
  let files = dir.filter((file) => globalGeos.has(file));

  // Stupid, but stable sort
  files = files.map((file) => {
    return {
      file,
      hash: crypto.createHash('sha256').update(file).digest('hex'),
    };
  }).sort((a, b) => {
    return a.hash === b.hash ? 0 : a.hash < b.hash ? -1 : 1;
  }).map((entry) => entry.file);

  console.log('Loading images...');
  return await Promise.all(files.map(async (file) => {
    const image = await jimp.read(path.join(IMAGE_DIR, file));

    const originalWidth = image.bitmap.width;
    const originalHeight = image.bitmap.height;

    // Reduce processing time (and memory usage)
    image.scaleToFit(2 * TARGET_WIDTH, 2 * TARGET_HEIGHT,
        jimp.RESIZE_NEAREST_NEIGHBOR);

    const scaleX = image.bitmap.width / originalWidth;
    const scaleY = image.bitmap.height / originalHeight;

    const geos = globalGeos.get(file)!.map((geo) => {
      return geo.map((point) => {
        // The points have inverted y axis in the data :(
        // (A https://www.labelbox.com/ quirk)
        return {
          x: point.x * scaleX,
          y: (originalHeight - point.y) * scaleY,
        };
      });
    });

    return new Input(image, geos);
  }));
}

/*
load().then(async (inputs) => {
  const random = inputs[0].randomize();

  let svg = await random.toSVG();
  fs.writeFileSync('/tmp/1.svg', svg);

  const grid = random.toTrainingPair().grid;

  svg = await random.toSVG(random.predictionToRects(grid, 1));
  fs.writeFileSync('/tmp/2.svg', svg);
});
*/
