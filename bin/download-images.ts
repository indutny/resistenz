#!/usr/bin/env npx ts-node

import * as fs from 'fs';
import * as crypto from 'crypto';
import * as path from 'path';
import { promisify } from 'util';
import fetch from 'node-fetch';

const URL_FILE = process.argv[2];
const OUT_DIR = process.argv[3];

const urls = fs.readFileSync(URL_FILE).toString().split(/(\r\n|\r|\n)+/g)
  .map((line) => line.trim())
  .filter((line) => line)
  .map((line) => {
    const match = line.match(
      /^(https:\/\/www\.flickr\.com\/photos\/[^\/]+\/[^\/]+)/);
    if (match === null) {
      throw new Error(`Invalid URL: "${line}"`);
    }
    return match![1];
  });

const uniqueURLs = Array.from(new Set(urls)).sort();

async function download(urls: ReadonlyArray<string>, maxParallel: number = 4) {
  for (let i = 0; i < urls.length; i += maxParallel) {
    await Promise.all(urls.slice(i, i + maxParallel).map(async (url) => {
      console.log(`Downloading: "${url}"`);

      const res = await fetch(url + '/sizes/o');
      const body = await res.text();

      const match = body.match(/src="(.*?_o.(?:jpg|png|jpeg))"/);
      if (!match) {
        throw new Error(`Image not found at "${url}"`);
      }

      const downloadURL = match[1];
      const extension = path.extname(downloadURL);

      const imageRes = await fetch(downloadURL);
      const image = await imageRes.buffer();

      const hash = crypto.createHash('sha256').update(url).digest('hex');
      const file = path.join(OUT_DIR, hash + extension);
      await promisify(fs.writeFile)(file, image);
    }));
  }
}

download(uniqueURLs).then(() => {
  console.log(uniqueURLs.length);
}).catch((e) => {
  console.error(e);
  process.exit(1);
});
