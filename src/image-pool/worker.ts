import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Input } from '../input';

const images: Map<number, Input> = new Map();

process.on('message', async (msg) => {
  let input: Input;
  if (images.has(msg.index)) {
    input = images.get(msg.index)!;
  } else {
    const image = await jimp.read(Buffer.from(msg.image, 'base64'));
    input = new Input(image, msg.polys);

    images.set(msg.index, input);
  }

  const random = input.randomize();

  const buffer = await random.image.getBufferAsync(jimp.MIME_PNG);

  process.send!({
    seq: msg.seq,
    image: buffer.toString('base64'),
    polys: random.polys,
  });
});
