#!/usr/bin/env npx ts-node
import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import jimp = require('jimp');

// No types :(
const exifParser = require('exif-parser');

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const RAW_DIR = path.join(DATASET_DIR, 'raw');

Promise.all(fs.readdirSync(RAW_DIR).filter((file) => {
  return /\.jpg$/.test(file);
}).map(async (file) => {
  const raw = await promisify(fs.readFile)(path.join(RAW_DIR, file));

  const parser = exifParser.create(raw);
  const result = parser.parse();

  const orientation: number | undefined = result.tags.Orientation;
  if (!orientation || orientation === 1) {
    return;
  }

  const image = await jimp.read(raw);

  if (orientation === 6) {
    image.rotate(90);
  } else if (orientation === 8) {
    image.rotate(-90);
  } else if (orientation === 3) {
    image.rotate(180);
  }

  await image.write(path.join(RAW_DIR, file));
})).then(() => {
  console.log('Done');
}).catch((e) => {
  console.error(e.stack);
  process.exit(1);
});
