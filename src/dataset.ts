import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import * as crypto from 'crypto';
import jimp = require('jimp');

import { IOrientedRect, polygonToRect } from './utils';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export type RawGrid = ReadonlyArray<number>;

export interface IDataset {
  readonly rgbs: ReadonlyArray<ReadonlyArray<number>>;
  readonly grids: ReadonlyArray<RawGrid>;
  readonly images: ReadonlyArray<jimp>;
}

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = 416;
export const GRID_SIZE = 20;
export const GRID_DEPTH = 5;

export async function load(): Promise<IDataset> {
  const json = await util.promisify(fs.readFile)(LABELS);
  const rects: Map<string, ReadonlyArray<IOrientedRect>> = new Map();

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

    rects.set(entry['External ID'], geometry.map((entry: any) => {
      return polygonToRect(entry.geometry);
    }));
  });

  const dir = await util.promisify(fs.readdir)(IMAGE_DIR);
  let files = dir.filter((file) => rects.has(file));

  // Semi-random sort
  files = files.map((file) => {
    return {
      file,
      hash: crypto.createHash('sha256').update(file).digest('hex'),
    };
  }).sort((a, b) => {
    return a.hash > b.hash ? 1 : a.hash < b.hash ? -1 : 0;
  }).map((a) => a.file);

  console.log('Loading images...');
  const images = await Promise.all(files.map(async (file) => {
    return jimp.read(path.join(IMAGE_DIR, file));
  }));

  console.log('Processing images...');

  const rgbs: ReadonlyArray<number>[] = [];
  const grids: RawGrid[] = [];
  for (const [ index, image ] of images.entries()) {
    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const imageRects: ReadonlyArray<IOrientedRect> =
      rects.get(files[index])!
      .map((rect) => {
        return {
          cx: rect.cx / width,
          cy: rect.cy / height,
          width: rect.width / width,
          height: rect.height / height,
          angle: rect.angle,
        };
      });

    const grid: Array<IOrientedRect | undefined> =
        new Array(GRID_SIZE * GRID_SIZE);

    for (const rect of imageRects) {
      const gridX = Math.round(rect.cx * (GRID_SIZE - 1));
      const gridY = Math.round(rect.cy * (GRID_SIZE - 1));

      const gridIndex = gridX * GRID_SIZE + gridY;

      // Busy slot
      // TODO(indutny): multiple rects per grid slot?
      if (grid[gridIndex] !== undefined) {
        continue;
      }
      console.log('+');

      // Make coordinates relative to the grid
      grid[gridIndex] = {
        cx: rect.cx - (gridX / GRID_SIZE),
        cy: rect.cy - (gridY / GRID_SIZE),
        width: rect.width,
        height: rect.height,
        angle: rect.angle,
      };
    }

    const rawGrid: number[] = [];
    for (const maybeRect of grid) {
      if (maybeRect === undefined) {
        rawGrid.push(0, 0, 0, 0, 0, 0);
        continue;
      }

      const rect = maybeRect!;
      rawGrid.push(rect.cx, rect.cy, rect.width, rect.height, rect.angle, 1);
    }

    image.resize(TARGET_WIDTH, TARGET_HEIGHT);

    const rgb: number[] = new Array(TARGET_WIDTH * TARGET_HEIGHT * 3);
    for (let i = 0, j = 0; i < image.bitmap.data.length; i += 4, j += 3) {
      rgb[j] = image.bitmap.data[i] / 255;
      rgb[j + 1] = image.bitmap.data[i + 1] / 255;
      rgb[j + 2] = image.bitmap.data[i + 2] / 255;
    }

    rgbs.push(rgb);
    grids.push(rawGrid);
  }

  return { rgbs, grids, images };
}
