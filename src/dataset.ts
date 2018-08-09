import * as fs from 'fs';
import * as path from 'path';
import * as util from 'util';
import { Buffer } from 'buffer';
import jimp = require('jimp');

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS = path.join(DATASET_DIR, 'labels.json');
const IMAGE_DIR = path.join(DATASET_DIR, 'resized');

export type Polygon = ReadonlyArray<number>;

export interface IDatasetEntry {
  readonly rgb: Buffer;
  readonly polygons: ReadonlyArray<Polygon>;
}

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
      const res: number[] = [];
      for (const point of geo.geometry) {
        res.push(point.x, point.y);
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
        const res: number[] = [];
        for (let i = 0; i < geo.length; i += 2) {
          res.push(geo[i] * 416 / width);
          res.push(geo[i + 1] * 416 / height);
        }
        return res;
      });

    image.resize(416, 416);

    const rgb = Buffer.alloc(416 * 416 * 3);
    for (let i = 0, j = 0; i < image.bitmap.data.length; i += 4, j += 3) {
      rgb[j] = image.bitmap.data[i];
      rgb[j + 1] = image.bitmap.data[i + 1];
      rgb[j + 2] = image.bitmap.data[i + 2];
    }

    entries.push({ rgb, polygons: imagePolys });
  }

  return entries;
}
