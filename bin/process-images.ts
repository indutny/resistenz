#!/usr/bin/env npx ts-node
import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import jimp = require('jimp');

import { IPoint, IRect, Polygon, polygonCenter } from '../src/utils';

const TARGET_WIDTH = 416;
const TARGET_HEIGHT = TARGET_WIDTH;

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS_FILE = path.join(DATASET_DIR, 'labels.json');
const RAW_DIR = path.join(DATASET_DIR, 'raw');
const RESISTOR_DIR = path.join(DATASET_DIR, 'resistors');
const PROCESSED_DIR = path.join(DATASET_DIR, 'processed');

const labels = JSON.parse(fs.readFileSync(LABELS_FILE).toString());

interface IResistor {
  readonly polygon: Polygon;
  readonly colors: ReadonlyArray<string> | undefined;
}

class Image {
  constructor(
      public readonly id: string,
      public readonly raw: jimp,
    public readonly resistors: ReadonlyArray<IResistor>) {
  }

  public slice(frames: ReadonlyArray<IRect>): ReadonlyArray<Image> {
    const res: Image[] = [];

    let i = 0;
    for (const frame of frames) {
      if (frame.width < TARGET_WIDTH || frame.height < TARGET_HEIGHT) {
        continue;
      }

      const clone = this.raw.clone();

      try {
        clone.crop(frame.x, frame.y, frame.width, frame.height);
      } catch (e) {
        console.error(e.stack);
        console.error('Skipping...');
        continue;
      }

      let subResistors = this.resistors.filter((res) => {
        const c = polygonCenter(res.polygon);

        return res.colors &&
          c.x >= frame.x && c.x <= (frame.x + frame.width) &&
          c.y >= frame.y && c.y <= (frame.y + frame.height);
      }).map((res) => {
        const polygon = res.polygon.map((p) => {
          return { x: p.x - frame.x, y: p.y - frame.y };
        });

        return { polygon, colors: res.colors };
      });

      assert.strictEqual(TARGET_WIDTH, TARGET_HEIGHT);
      const scale =
          Math.min(1, 2 * TARGET_WIDTH / Math.min(frame.width, frame.height));

      // Resize to save space
      clone.scale(scale, jimp.RESIZE_NEAREST_NEIGHBOR);

      subResistors = subResistors.map((res) => {
        const polygon = res.polygon.map((p) => {
          return { x: p.x * scale, y: p.y * scale };
        });

        return { polygon, colors: res.colors };
      });

      res.push(new Image(`${this.id}_${i++}`, clone, subResistors));
    }

    return res;
  }

  public async save(file: string) {
    const json = JSON.stringify({
      resistors: this.resistors,
    });

    return Promise.all([
      this.raw.writeAsync(file + '.jpg'),
      promisify(fs.writeFile)(file + '.json', json)
    ]);
  }
}

async function run() {
  let done = 0;
  await Promise.all(labels.map(async (data: any) => {
    if (!data || !data['Label']['Suggested Frame']) {
      return;
    }

    const id: string = data['External ID'];
    const imageFile = path.join(RAW_DIR, id + '.jpg');

    const exists = await promisify(fs.exists)(imageFile);
    if (!exists) {
      return;
    }

    const rawImage = await jimp.read(imageFile);

    const height = rawImage.bitmap.height;

    // labelbox.com quirks
    function translate(point: IPoint): IPoint {
      return { x: point.x, y: height - point.y };
    }

    const frames: IRect[] = [];
    for (let { geometry } of data['Label']['Suggested Frame']) {
      geometry = geometry.map(translate);

      frames.push({
        x: geometry[0].x,
        y: geometry[0].y,
        width: geometry[2].x - geometry[0].x,
        height: geometry[2].y - geometry[0].y,
      });
    }

    let polygons: Polygon[] = [];
    for (let { geometry } of (data['Label']['Resistor'] || [])) {
      geometry = geometry.map(translate);

      polygons.push(geometry);
    }

    polygons = polygons.filter((poly) => {
      const c = polygonCenter(poly);

      // Leave only those resistors that lie in frames
      // TODO(indutny): reduce copy-paste between this file and slice-resistors
      for (const frame of frames) {
        if (frame.x <= c.x && frame.y <= c.y &&
            c.x <= (frame.x + frame.width) && c.y <= (frame.y + frame.height)) {
          return true;
        }
      }

      return false;
    });

    const resistors = await Promise.all(polygons.map(async (polygon, index) => {
      const colorFile = path.join(RESISTOR_DIR, `${id}_${index}.json`);
      const exists = await promisify(fs.exists)(colorFile);

      if (!exists) {
        return { polygon, colors: undefined };
      }

      const raw = await promisify(fs.readFile)(colorFile);
      return {
        polygon,
        colors: JSON.parse(raw.toString()).colors,
      };
    }));

    const image = new Image(id, rawImage, resistors);

    await Promise.all(image.slice(frames).map(async (slice) => {
      await slice.save(path.join(PROCESSED_DIR, slice.id));
    }));
  }).map((promise: Promise<any>) => {
    return promise.then(() => {
      console.log(`${done++}/${labels.length}`);
    });
  }));
}

run().then(() => {
  console.log('done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
