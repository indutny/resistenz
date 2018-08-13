import * as assert from 'assert';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import jimp = require('jimp');

import { Polygon, IPoint, IOrientedRect } from './utils';
import { Input } from './input';

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
  let files = dir.filter((file) => globalGeos.has(file)).slice(32, 40);

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

    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const geos = globalGeos.get(file)!.map((geo) => {
      return geo.map((point) => {
        // The points have inverted y axis in the data :(
        // (A https://www.labelbox.com/ quirk)
        return {
          x: point.x,
          y: height - point.y,
        };
      });
    });

    return new Input(image, geos);
  }));
}
