import * as assert from 'assert';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import jimp = require('jimp');

import { Polygon, IPoint, IOrientedRect } from './utils';
import { Input, TARGET_WIDTH, TARGET_HEIGHT } from './input';

const DATASET_DIR = path.join(__dirname, '..', 'dataset');
const IMAGE_DIR = path.join(DATASET_DIR, 'processed');

export interface IDataset {
  readonly train: ReadonlyArray<Input>;
  readonly validate: ReadonlyArray<Input>;
}

export async function load(validateSplit: number = 0.1): Promise<IDataset> {
  const dir = await promisify(fs.readdir)(IMAGE_DIR);

  let files = dir.filter((file) => /\.json$/.test(file));

  function getFileHash(file: string) {
    return file.replace(/_.*$/, '');
  }

  const hashes = Array.from(new Set(files.map(getFileHash)));

  const validateCount = (validateSplit * hashes.length) | 0;

  console.log('Loading images...');
  let done = 0;
  let total = files.length;

  function filterByHashes(hashes: ReadonlyArray<string>): string[] {
    const set = new Set(hashes);
    return files.filter((file) => {
      const hash = getFileHash(file);

      return set.has(hash);
    });
  }

  async function getInputs(hashes: ReadonlyArray<string>) {
    const files = filterByHashes(hashes);

    return await Promise.all(files.map(async (labelsFile) => {
      const imageFile = labelsFile.replace(/\.json$/, '') + '.jpg';
      const image = await jimp.read(path.join(IMAGE_DIR, imageFile));
      const rawLabels = await promisify(fs.readFile)(
          path.join(IMAGE_DIR, labelsFile));

      const labels = JSON.parse(rawLabels.toString());

      done++;
      if (done % 100 === 0 || done === total) {
        console.log(`${done}/${total}`);
      }
      return new Input(image, labels.polygons);
    }));
  }

  const [ validate, train ] = await Promise.all([
    getInputs(hashes.slice(0, validateCount)),
    getInputs(hashes.slice(validateCount)),
  ]);

  return { validate, train };
}

/*
import { ImagePool } from './image-pool';

load().then(async (inputs) => {
  const pool = new ImagePool();
  const random = await pool.randomize(inputs.train[0]);

  let svg = await random.toSVG();
  fs.writeFileSync('/tmp/1.svg', svg);

  const grid = random.toTrainingPair().grid;

  svg = await random.toSVG(random.predictionToRects(grid, 1));
  fs.writeFileSync('/tmp/2.svg', svg);

  pool.close();
});
 */
