import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import { Buffer } from 'buffer';
import jimp = require('jimp');

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export interface IPoint { readonly x: number, readonly y: number };
export type Polygon = ReadonlyArray<IPoint>;

export interface IDatasetEntry {
  readonly rgb: Buffer;
  readonly grid: ReadonlyArray<Polygon | undefined>;
}

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = 416;
export const GRID_SIZE = 13;
export const GRID_DEPTH = 5;

export async function load(): Promise<ReadonlyArray<IDatasetEntry>> {
  const json = await util.promisify(fs.readFile)(LABELS);
  const polygons: Map<string, ReadonlyArray<Polygon>> = new Map();
  JSON.parse(json.toString()).forEach((entry: any) => {
    if (!entry || !entry.Label || !entry.Label.Resistor) {
      return;
    }

    const geometry = entry.Label.Resistor.filter((geo: any) => {
      return geo.geometry.length === 4;
    });

    if (geometry.length === 0) {
      return;
    }

    polygons.set(entry['External ID'], geometry.map((geo: any) => {
      const res: IPoint[] = [];
      for (const point of geo.geometry) {
        res.push({ x: point.x, y: point.y });
      }
      return res;
    }));
  });

  const dir = await util.promisify(fs.readdir)(IMAGE_DIR);
  const files = dir.filter((file) => polygons.has(file));

  const images = await Promise.all(files.map(async (file) => {
    return jimp.read(path.join(IMAGE_DIR, file));
  }));

  const entries: IDatasetEntry[] = [];
  for (const [ index, image ] of images.entries()) {
    const width = image.bitmap.width;
    const height = image.bitmap.height;
    const imagePolys: ReadonlyArray<Polygon> = polygons.get(files[index])!
      .map((geo) => {
        return geo.map((p) => {
          return {
            x: p.x / width,
            y: p.y / height,
          };
        });
      });

    const centers: ReadonlyArray<IPoint> = imagePolys.map((poly) => {
      let x: number = 0;
      let y: number = 0;

      for (const point of poly) {
        x += point.x;
        y += point.y;
      }
      x /= poly.length;
      y /= poly.length;

      return { x, y };
    });

    const grid: Array<Polygon | undefined> =
        new Array(GRID_SIZE * GRID_SIZE * GRID_DEPTH);

    for (const [ i, poly ] of imagePolys.entries()) {
      const center = centers[i];

      const gridX = Math.round(center.x * (GRID_SIZE - 1));
      const gridY = Math.round(center.y * (GRID_SIZE - 1));

      let gridIndex = GRID_DEPTH * (gridX * GRID_SIZE + gridY);
      let hasSpace = false;
      for (let i = 0; i < GRID_DEPTH; i++, gridIndex++) {
        if (grid[gridIndex] === undefined) {
          hasSpace = true;
          break;
        }
      }

      if (!hasSpace) {
        continue;
      }

      grid[gridIndex] = poly;
    }

    image.resize(TARGET_WIDTH, TARGET_HEIGHT);

    const rgb = Buffer.alloc(TARGET_WIDTH * TARGET_HEIGHT * 3);
    for (let i = 0, j = 0; i < image.bitmap.data.length; i += 4, j += 3) {
      rgb[j] = image.bitmap.data[i];
      rgb[j + 1] = image.bitmap.data[i + 1];
      rgb[j + 2] = image.bitmap.data[i + 2];
    }

    entries.push({ rgb, grid });
  }

  return entries;
}

load();
