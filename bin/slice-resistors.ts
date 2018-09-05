#!/usr/bin/env npx ts-node
import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import jimp = require('jimp');

import { IPoint, Polygon, IOrientedRect, polygonToRect } from '../src/utils';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const LABELS_FILE = path.join(DATASET_DIR, 'labels.json');
const RAW_DIR = path.join(DATASET_DIR, 'raw');
const RESISTORS_DIR = path.join(DATASET_DIR, 'resistors');

const BATCH_SIZE = 32;

async function sliceRect(image: jimp, rect: IOrientedRect, id: string,
                         subId: number) {
  const radius = Math.max(rect.width, rect.height) / 2;

  image.crop(rect.cx - radius, rect.cy - radius, 2 * radius,
    2 * radius);
  image.rotate(rect.angle * 180 / Math.PI, false);

  image.crop(radius - rect.width / 2, radius - rect.height / 2,
             rect.width, rect.height);
  image.normalize();

  await image.writeAsync(path.join(RESISTORS_DIR, id + '_' + subId + '.jpg'));
}

async function runSingle(label: any) {
  const id = label['External ID'];
  const rawResistors = label['Label']['Resistor'] || [];
  const rawFrames = label['Label']['Suggested Frame'] || [];
  const resistors =
      rawResistors.map((elem: any) => elem.geometry) as ReadonlyArray<Polygon>;
  let frames =
      rawFrames.map((elem: any) => elem.geometry) as ReadonlyArray<Polygon>;

  const imageFile = path.join(RAW_DIR, id + '.jpg');
  if (!await promisify(fs.exists)(imageFile)) {
    return;
  }

  let image: jimp;
  try {
    image = await jimp.read(imageFile);
  } catch (e) {
    return;
  }

  const height = image.bitmap.height;

  // labelbox.com quirks
  function translate(point: IPoint): IPoint {
    return { x: point.x, y: height - point.y };
  }

  frames = frames.map((frame) => frame.map((point) => translate(point)));

  const rects: ReadonlyArray<IOrientedRect> = resistors.map((poly) => {
    return polygonToRect(poly.map((point) => translate(point)));
  }).filter((rect: IOrientedRect) => {
    // Leave only those resistors that lie in frames
    for (const frame of frames) {
      const tl = frame[0];
      const br = frame[2];

      if (tl.x <= rect.cx && tl.y <= rect.cy &&
          rect.cx <= br.x && rect.cy <= br.y) {
        return true;
      }
    }
    return false;
  });

  // Slice up
  await Promise.all(rects.map(async (rect, index) => {
    try {
      await sliceRect(image.clone(), rect, id, index);
    } catch (e) {
      // ignore
    }
  }));
}

async function runBatch(labels: ReadonlyArray<any>) {
  return await Promise.all(labels.map(async (label: any) => {
    return await runSingle(label);
  }));
}

async function run() {
  const rawLabels = await promisify(fs.readFile)(LABELS_FILE);
  const labels = JSON.parse(rawLabels.toString());

  for (let i = 0; i < labels.length; i += BATCH_SIZE) {
    console.log('%d/%d', i, labels.length);
    await runBatch(labels.slice(i, i + BATCH_SIZE).filter((l: any) => l));
  }
}

run().then(() => {
  console.log('done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
