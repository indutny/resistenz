#!/usr/bin/env npx ts-node
import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';

const RAW_DIR = process.argv[2];
const RELATIVE = path.relative('.', RAW_DIR);

const header = 'External ID,URL';

const data = fs.readdirSync(RAW_DIR).filter((file) => {
  return /\.jpg$/.test(file);
}).map((file) => {
  const id = file.replace(/\.jpg$/, '');
  return `${id},https://raw.githubusercontent.com/indutny/resistenz/` +
    `master/${RELATIVE}/${file}`;
}).join('\n');

console.log(header + '\n' + data + '\n');
