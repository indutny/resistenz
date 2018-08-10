import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import jimp = require('jimp');

import { IOrientedRect, polygonToRect } from './utils';
import { Input } from './input';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = 416;
export const GRID_SIZE = 20;
export const GRID_DIMS = 6;

export async function load(): Promise<ReadonlyArray<Input>> {
  const json = await util.promisify(fs.readFile)(LABELS);
  const globalRects: Map<string, ReadonlyArray<IOrientedRect>> = new Map();

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

    globalRects.set(entry['External ID'], geometry.map((entry: any) => {
      return polygonToRect(entry.geometry);
    }));
  });

  const dir = await util.promisify(fs.readdir)(IMAGE_DIR);
  let files = dir.filter((file) => globalRects.has(file));

  console.log('Loading images...');
  return await Promise.all(files.map(async (file) => {
    const image = await jimp.read(path.join(IMAGE_DIR, file));

    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const rects = globalRects.get(file)!.map((rect) => {
      return {
        cx: rect.cx / width,
        cy: rect.cy / height,
        width: rect.width / width,
        height: rect.height / height,
        angle: rect.angle,
      };
    });

    return new Input(image, rects);
  }));
}
