import * as os from 'os';
import * as path from 'path';
import { ChildProcess, fork } from 'child_process';
import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Input } from '../input';
import { Polygon } from '../utils';

interface IWorker {
  readonly proc: ChildProcess;
  readonly images: Set<number>;
}

export class Master {
  private seq: number = 0;
  private workers: IWorker[] = [];
  private readonly callbacks: Map<number, (input: Input) => void> = new Map();

  public readonly size: number;

  constructor(private readonly images: ReadonlyArray<Input>,
              private readonly maxWorkers: number = os.cpus().length) {
    this.size = this.images.length;

    while (this.workers.length < this.maxWorkers) {
      const worker = fork(path.join(__dirname, 'worker.ts'));

      this.workers.push({ proc: worker, images: new Set() });

      worker.on('message', async (msg) => {
        const callback = this.callbacks.get(msg.seq)!;
        this.callbacks.delete(msg.seq);

        const image = await jimp.read(Buffer.from(msg.image, 'base64'));
        callback(new Input(image, msg.polys));
      });
    }
  }

  public get(index: number): Input {
    return this.images[index];
  }

  public async randomize(index: number): Promise<Input> {
    return new Promise<Input>(async (resolve) => {
      const worker = this.workers[index % this.workers.length];

      const seq = this.seq++;

      this.callbacks.set(seq, resolve);

      let maybeImage: string | undefined = undefined;
      let maybePolys: ReadonlyArray<Polygon> | undefined = undefined;
      if (!worker.images.has(index)) {
        const input = this.images[index];
        const buffer = await input.image.getBufferAsync(jimp.MIME_PNG);
        maybeImage = buffer.toString('base64');
        maybePolys = input.polys;
      }

      worker.proc.send({
        seq,
        index,
        image: maybeImage,
        polys: maybePolys,
      });

      worker.images.add(index);
    });
  }

  public close() {
    const workers = this.workers;
    this.workers = [];
    for (const worker of workers) {
      worker.proc.kill();
    }
  }
}
