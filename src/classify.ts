import * as http from 'http';
import * as fs from 'fs';
import { run as microRun, json, send } from 'micro';
import * as path from 'path';
import * as debugAPI from 'debug';
import * as serveStatic from 'serve-static';
import * as Joi from 'joi';

type Req = http.IncomingMessage;
type Res = http.ServerResponse;

const debug = debugAPI('resistenz:classify:cli');

const PUBLIC_DIR = path.join(__dirname, '..', 'public', 'classify');
const IMAGES_DIR = path.join(PUBLIC_DIR, 'images');

const DIGIT = [
  'black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
  'grey', 'white',
];

const TOLERANCE = [
  'brown', 'red', 'green', 'blue', 'violet', 'grey', 'gold', 'silver', 'none',
];

const TEMPERATURE = [
  'black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
  'grey',
];

export class Server extends http.Server {
  private readonly dataset = fs.readdirSync(IMAGES_DIR);
  private readonly images = this.dataset.filter((f) => /\.jpg$/.test(f));
  private readonly hashes = new Set(this.images.map((f) => f.slice(0, -4)));
  private readonly completed: Set<string>;
  private readonly labelQueue: string[];
  private readonly publicHandler = serveStatic(PUBLIC_DIR);

  private readonly labelSchema = Joi.object().keys({
    hash: Joi.string().max(128).required(),
    colors: Joi.array().ordered(
      Joi.valid(DIGIT).required(),
      Joi.valid(DIGIT).required(),
      Joi.valid(DIGIT.concat('none')).required(),
      Joi.valid(DIGIT.concat('none')).required(),
      Joi.valid(TOLERANCE).required(),
      Joi.valid(TEMPERATURE.concat('none')).required(),
    ).max(6).required(),
  });

  private readonly skipSchema = Joi.object().keys({
    hash: Joi.string().max(128).required(),
  });

  constructor() {
    super();

    this.completed  = new Set(this.dataset.filter((f) => {
      return /\.json$/.test(f);
    }).map((f) => f.slice(0, -5)).filter((hash) => this.hashes.has(hash)));

    this.labelQueue = Array.from(this.hashes).filter((hash) => {
      return !this.completed.has(hash);
    });

    this.on('request', (req, res) => {
      this.publicHandler(req, res, () => {
        microRun(req, res, (req, res) => this.handleAPI(req, res));
      });
    });
  }

  private async handleAPI(req: Req, res: Res) {
    if (req.method === 'GET') {
      if (req.url === '/api/stats') {
        return this.handleStats(req, res);
      }

      if (req.url === '/api/next') {
        return this.handleNext(req, res);
      }
    } else if (req.method === 'PUT') {
      if (req.url === '/api/label') {
        return this.handleUpdateLabel(req, res);
      }
    } else if (req.method === 'DELETE') {
      if (req.url === '/api/label') {
        return this.handleSkipLabel(req, res);
      }
    }

    return send(res, 404, { error: 'Not found' });
  }

  private async handleStats(req: Req, res: Res) {
    return {
      completed: this.hashes.size - this.labelQueue.length,
      total: this.hashes.size,
    };
  }

  private async handleNext(req: Req, res: Res) {
    if (this.labelQueue.length === 0) {
      return { done: true, image: null };
    }

    return {
      done: false,
      hash: this.labelQueue[0],
    };
  }

  private async handleUpdateLabel(req: Req, res: Res) {
    const raw = await json(req);
    const { error, value } = this.labelSchema.validate(raw);
    if (error) {
      return send(res, 400, { error: error.message });
    }

    console.log(value);
    return { ok: true };
  }

  private async handleSkipLabel(req: Req, res: Res) {
    const raw = await json(req);
    const { error, value } = this.skipSchema.validate(raw);
    if (error) {
      return send(res, 400, { error: error.message });
    }

    console.log(value);
    return { ok: true };
  }
}
