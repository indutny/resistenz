import * as os from 'os';
import * as path from 'path';
import { ChildProcess, fork } from 'child_process';
import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Input } from '../input';

export class Master {
  private seq: number = 0;
  private lastWorker: number = 0;
  private workers: ChildProcess[] = [];
  private readonly callbacks: Map<number, (input: Input) => void> = new Map();

  constructor(private readonly size: number = os.cpus().length) {
    while (this.workers.length < this.size) {
      const worker = fork(path.join(__dirname, 'worker.ts'));

      this.workers.push(worker);

      worker.on('message', async (msg) => {
        const callback = this.callbacks.get(msg.seq)!;
        this.callbacks.delete(msg.seq);

        const image = await jimp.read(Buffer.from(msg.image, 'base64'));
        callback(new Input(image, msg.polys));
      });
    }
  }

  public async randomize(input: Input): Promise<Input> {
    const image = await input.image.getBufferAsync(jimp.MIME_PNG);

    return new Promise<Input>((resolve) => {
      const worker = this.workers[this.lastWorker];
      this.lastWorker = (this.lastWorker + 1) % this.workers.length;

      const seq = this.seq++;

      this.callbacks.set(seq, resolve);

      worker.send({
        seq,
        image: image.toString('base64'),
        polys: input.polys,
      });
    });
  }

  public close() {
    const workers = this.workers;
    this.workers = [];
    for (const worker of workers) {
      worker.kill();
    }
  }
}
