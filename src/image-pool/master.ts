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
  private readonly history: Map<number, Input> = new Map();

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

        const image = new jimp({
          width: msg.width,
          height: msg.height,
          data: Buffer.from(msg.image, 'base64'),
        });

        const input = new Input(image, msg.polys);
        this.history.set(msg.index, input);
        callback(input);
      });
    }
  }

  public async randomize(index: number): Promise<Input> {
    return new Promise<Input>(async (resolve) => {
      const worker = this.workers[index % this.workers.length];

      const seq = this.seq++;

      this.callbacks.set(seq, resolve);

      let maybeImage: string | undefined;
      let maybeWidth: number | undefined;
      let maybeHeight: number | undefined;
      let maybePolys: ReadonlyArray<Polygon> | undefined;
      if (!worker.images.has(index)) {
        const input = this.images[index];
        const bitmap = input.image.bitmap;

        maybeWidth = bitmap.width;
        maybeHeight = bitmap.height;
        maybeImage = bitmap.data.toString('base64');
        maybePolys = input.polys;
      }

      worker.proc.send({
        seq,
        index,
        width: maybeWidth,
        height: maybeHeight,
        image: maybeImage,
        polys: maybePolys,
      });

      worker.images.add(index);
    });
  }

  public async getLast(index: number): Promise<Input>{
    if (this.history.has(index)) {
      return this.history.get(index)!;
    }

    return await this.randomize(index);
  }

  public close() {
    const workers = this.workers;
    this.workers = [];
    for (const worker of workers) {
      worker.proc.kill();
    }
  }
}
