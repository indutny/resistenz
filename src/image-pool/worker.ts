import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Input } from '../input';

process.on('message', async (msg) => {
  const image = await jimp.read(Buffer.from(msg.image, 'base64'));
  const input = new Input(image, msg.polys);

  const random = input.randomize();

  const buffer = await random.image.getBufferAsync(jimp.MIME_PNG);

  process.send!({
    seq: msg.seq,
    image: buffer.toString('base64'),
    polys: random.polys,
  });
});
