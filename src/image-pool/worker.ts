import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Input } from '../input';

const images: Map<number, Input> = new Map();

process.on('message', async (msg) => {
  let input: Input;
  if (images.has(msg.index)) {
    input = images.get(msg.index)!;
  } else {
    const image = new jimp({
      width: msg.width,
      height: msg.height,
      data: Buffer.from(msg.image, 'base64'),
    });
    input = new Input(image, msg.polys);

    images.set(msg.index, input);
  }

  const random = input.randomize();

  const bitmap = random.image.bitmap;

  process.send!({
    seq: msg.seq,
    width: bitmap.width,
    height: bitmap.height,
    image: bitmap.data.toString('base64'),
    polys: random.polys,
  });
});
