import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import jimp = require('jimp');

import { IPoint, IOrientedRect, polygonToRect } from './utils';
import { Input } from './input';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = 416;
export const GRID_SIZE = 20;
export const GRID_DIMS = 6;

type Polygon = ReadonlyArray<IPoint>;

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

  console.log('Loading images...');
  return await Promise.all(files.map(async (file) => {
    const image = await jimp.read(path.join(IMAGE_DIR, file));

    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const rects = globalGeos.get(file)!.map((geo) => {
      return polygonToRect(geo.map((point) => {
        // The points have inverted y axis in the data :(
        return {
          x: point.x,
          y: height - point.y,
        };
      }));
    });

    return new Input(image, rects);
  }));
}

load().then(async (inputs) => {
  const svg = await inputs[7].toSVG();
  fs.writeFileSync('/tmp/1.svg', svg);
});
