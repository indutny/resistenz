import * as assert from 'assert';
import { Buffer } from 'buffer';
import jimp = require('jimp');

import { Polygon, IPoint, IOrientedRect, polygonToRect, rotate } from './utils';

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = 416;
export const TARGET_CHANNELS = 3;
export const GRID_SIZE = 20;
export const GRID_CHANNELS = 6;

// Maximum angle of rotation
const MAX_ROT_ANGLE = 180;

// Max amount of crop from each side
const MAX_CROP_PERCENT = 0.2;

// How much brightness can be adjusted [ 0, 1 ]
const MAX_BRIGHTNESS_DELTA = 0.2;

// How much contrast can be adjusted [ 0, 1 ]
const MAX_CONTRAST_DELTA = 0.2;

export interface ITrainingPair {
  readonly rgb: Float32Array;
  readonly grid: Float32Array;
}

export interface IColoredRect extends IOrientedRect {
  readonly alpha?: number;
}

export class Input {
  constructor(public readonly image: jimp,
              public readonly polys: ReadonlyArray<Polygon>) {
  }

  private random(): number {
    return 0.5;
    // return Math.random();
  }

  public randomize(): Input {
    const clone = this.image.clone();
    let polys = this.polys.slice();

    const width = clone.bitmap.width;
    const height = clone.bitmap.height;

    const center = { x: width / 2, y: height / 2 };

    // TODO(indutny): add noise?

    // Randomly rotate
    const angleDeg = (this.random() - 0.5) * 2 * MAX_ROT_ANGLE;
    const angleRad = angleDeg * Math.PI / 180;
    clone.background(0xffffffff);
    clone.rotate(-angleDeg, false);
    polys = polys.map((points) => {
      return points.map((p) => {
        const t = rotate({ x: p.x - center.x, y: p.y - center.y }, angleRad);
        return { x: t.x + center.x, y: t.y + center.y };
      });
    });

    // Randomly crop
    const crop = {
      top: this.random() * MAX_CROP_PERCENT,
      bottom: this.random() * MAX_CROP_PERCENT,
      left: this.random() * MAX_CROP_PERCENT,
      right: this.random() * MAX_CROP_PERCENT,
    };

    let cropX = Math.floor(crop.left * width);
    let cropY = Math.floor(crop.top * height);
    let cropW = width - cropX - Math.floor(crop.right * width);
    let cropH = height - cropY - Math.floor(crop.bottom * height);

    // Preserve x-y scale
    if (cropW > cropH) {
      cropX += this.random() * (cropW - cropH);
      cropW = cropH;
    } else {
      cropY += this.random() * (cropH - cropW);
      cropH = cropW;
    }

    clone.crop(cropX, cropY, cropW, cropH);
    polys = polys.filter((points) => {
      return points.every((p) => {
        return p.x >= cropX && p.x <= cropX + cropW &&
               p.y >= cropY && p.y <= cropY + cropH;
      });
    }).map((points) => {
      return points.map((p) => {
        return {
          x: p.x - cropX,
          y: p.y - cropY,
        };
      });
    });

    // Random brightness/contrast adjustment
    clone.brightness((this.random() - 0.5) * 2 * MAX_BRIGHTNESS_DELTA);
    clone.contrast((this.random() - 0.5) * 2 * MAX_CONTRAST_DELTA);

    // Return new network input
    return new Input(clone, polys).resize();
  }

  public resize(): Input {
    const clone = this.image.clone();
    let polys = this.polys.slice();

    const scaleX = TARGET_WIDTH / clone.bitmap.width;
    const scaleY = TARGET_HEIGHT / clone.bitmap.height;
    clone.resize(TARGET_WIDTH, TARGET_HEIGHT);

    polys = polys.map((points) => {
      return points.map((p) => {
        return { x: p.x * scaleX, y: p.y * scaleY };
      });
    });

    return new Input(clone, polys);
  }

  public computeRects(): ReadonlyArray<IOrientedRect> {
    return this.polys.map((poly) => {
      return polygonToRect(poly);
    });
  }

  public toTrainingPair(): ITrainingPair {
    const bitmap = this.image.bitmap;
    const rgb = new Float32Array(
        bitmap.width * bitmap.height * TARGET_CHANNELS);
    for (let i = 0, j = 0; i < bitmap.data.length; i += 4, j += 3) {
      rgb[j + 0] = bitmap.data[i + 0] / 0xff;
      rgb[j + 1] = bitmap.data[i + 1] / 0xff;
      rgb[j + 2] = bitmap.data[i + 2] / 0xff;
    }

    const rects = this.computeRects();

    const grid = new Float32Array(GRID_SIZE * GRID_SIZE * GRID_CHANNELS);
    for (const rect of rects) {
      const scaledRect: IOrientedRect = {
        cx: rect.cx / bitmap.width,
        cy: rect.cy / bitmap.height,

        // This is fine only for square output
        width: rect.width / bitmap.width,
        height: rect.height / bitmap.height,

        angle: rect.angle,
      };

      const gridX = (scaledRect.cx * GRID_SIZE) | 0;
      const gridY = (scaledRect.cy * GRID_SIZE) | 0;

      const gridOff = (gridY * GRID_SIZE + gridX) * GRID_CHANNELS;

      // Cell is busy
      if (grid[gridOff + 5] !== 0) {
        continue;
      }

      let angle = scaledRect.angle / Math.PI;
      if (angle < 0) {
        angle += 1;
      }

      const cx = scaledRect.cx * GRID_SIZE - gridX;
      const cy = scaledRect.cy * GRID_SIZE - gridY;
      assert(0 <= cx && cx <= 1, '`cx` out of bounds');
      assert(0 <= cy && cy <= 1, '`cy` out of bounds');

      grid[gridOff + 0] = cx;
      grid[gridOff + 1] = cy;

      grid[gridOff + 2] = scaledRect.width;
      grid[gridOff + 3] = scaledRect.height;
      grid[gridOff + 4] = angle;
      grid[gridOff + 5] = 1;
    }

    return {
      rgb,
      grid,
    };
  }

  public async toSVG(rects?: ReadonlyArray<IColoredRect>): Promise<string> {
    // TODO(indutny): remove this after Jimp bug is fixed
    const jpeg: Buffer =
        (await this.image.getBufferAsync(jimp.MIME_JPEG) as any);
    const src = `data:${jimp.MIME_JPEG};base64,${jpeg.toString('base64')}`;
    const img = `<image xlink:href="${src}" />`;

    const width = this.image.bitmap.width;
    const height = this.image.bitmap.height;

    if (!rects) {
      rects = this.computeRects();
    }
    const polygons = rects.map((rect) => {
      const halfWidth = rect.width / 2;
      const halfHeight = rect.height / 2;

      const points = [
        { x: halfWidth, y: halfHeight },
        { x: halfWidth, y: -halfHeight },
        { x: -halfWidth, y: -halfHeight },
        { x: -halfWidth, y: halfHeight },
      ].map((point) => {
        return {
          x: point.x,
          y: point.y,
        };
      }).map((point) => {
        return rotate(point, rect.angle);
      }).map((point) => {
        return `${point.x + rect.cx},${point.y + rect.cy}`;
      }).join(' ');

      const alpha = rect.alpha === undefined ? 1 : rect.alpha;

      return `<polygon points="${points}" fill="none" stroke="rgba(255,0,0,${alpha})"/>`;
    });

    return `
      <svg width="${width}" height="${height}"
           xmlns="http://www.w3.org/2000/svg"
           xmlns:xlink="http://www.w3.org/1999/xlink">
        ${img}
        ${polygons.join('\n')}
      </svg>`;
  }

  public predictionToRects(
      prediction: Float32Array | Int32Array | Uint8Array,
      depth: number,
      threshold: number = 0.5): ReadonlyArray<IColoredRect> {
    assert.strictEqual(prediction.length,
        GRID_SIZE * GRID_SIZE * depth * GRID_CHANNELS);

    const bitmap = this.image.bitmap;

    const rects: IOrientedRect[] = [];
    for (let i = 0; i < prediction.length; i += GRID_CHANNELS) {
      const confidence = prediction[i + 5];
      if (confidence < threshold) {
        continue;
      }

      const gridOff = (i / (GRID_CHANNELS * depth)) | 0;
      const gridX = gridOff % GRID_SIZE;
      const gridY = (gridOff / GRID_SIZE) | 0;

      const rect: IColoredRect = {
        cx: (prediction[i + 0] + gridX) * bitmap.width / GRID_SIZE,
        cy: (prediction[i + 1] + gridY) * bitmap.height / GRID_SIZE,
        width: (prediction[i + 2]) * bitmap.width,
        height: (prediction[i + 3]) * bitmap.height,
        angle: prediction[i + 4] * Math.PI,

        alpha: confidence,
      };

      rects.push(rect);
    }
    return rects;
  }
}
