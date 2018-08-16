import * as assert from 'assert';
import { Buffer } from 'buffer';
import jimp = require('jimp');

import {
  Polygon, IPoint, IOrientedRect, polygonToRect, rotate,
  vector, norm, polygonCenter, rectColor,
} from './utils';

export const TARGET_WIDTH = 416;
export const TARGET_HEIGHT = TARGET_WIDTH;
export const TARGET_CHANNELS = 3;
export const GRID_SIZE = 8;
export const GRID_CHANNELS = 7;

// Max amount of crop from each side
const MAX_CROP_PERCENT = 0.05;

const MAX_BRIGHTNESS = 0.1;
const MAX_CONTRAST = 0.1;
const MAX_HUE = 10;
const MAX_ROTATE = 0;

export interface ITrainingPair {
  readonly rgb: Float32Array;
  readonly grid: Float32Array;
}

export interface IColoredRect extends IOrientedRect {
  readonly confidence?: number;
}

export class Input {
  constructor(public readonly image: jimp,
              public readonly polys: ReadonlyArray<Polygon>) {
  }

  private random(): number {
    return 0.0;
    return Math.random();
  }

  public randomize(): Input {
    const clone = this.image.clone();
    let polys = this.polys.slice();

    let width = clone.bitmap.width;
    let height = clone.bitmap.height;

    const originalCenter = { x: width / 2, y: height / 2 };

    // TODO(indutny): add noise?

    // Randomly rotate by 90, 180, or 270 deg + small continuous degree
    const angleDeg = (this.random() - 0.5) * 2 * MAX_ROTATE +
        ((this.random() * 4) | 0) * 90;
    const angleRad = angleDeg * Math.PI / 180;
    clone.background(0x7f7f7fff);
    clone.rotate(-angleDeg, false);

    width = clone.bitmap.width;
    height = clone.bitmap.height;
    const rotatedCenter = { x: width / 2, y: height / 2 };

    polys = polys.map((points) => {
      return points.map((p) => {
        const t = rotate({
          x: p.x - originalCenter.x,
          y: p.y - originalCenter.y
        }, angleRad);
        return { x: t.x + rotatedCenter.x, y: t.y + rotatedCenter.y };
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
      cropX += (cropW - cropH) / 2;
      cropW = cropH;
    } else {
      cropY += (cropH - cropW) / 2;
      cropH = cropW;
    }

    clone.crop(cropX, cropY, cropW, cropH);
    polys = polys.filter((points) => {
      const center = polygonCenter(points);

      return center.x >= cropX && center.x <= cropX + cropW &&
             center.y >= cropY && center.y <= cropY + cropH;
    }).map((points) => {
      return points.map((p) => {
        return {
          x: p.x - cropX,
          y: p.y - cropY,
        };
      });
    });

    clone.brightness((this.random() - 0.5) * 2 * MAX_BRIGHTNESS);
    clone.contrast((this.random() - 0.5) * 2 * MAX_CONTRAST);
    clone.color([
      { apply: 'hue', params: [ (this.random() - 0.5) * 2 * MAX_HUE ] },
    ]);

    // Return new network input
    return new Input(clone, polys).resize();
  }

  public resize(): Input {
    const clone = this.image.clone();
    let polys = this.polys.slice();

    const scaleX = TARGET_WIDTH / clone.bitmap.width;
    const scaleY = TARGET_HEIGHT / clone.bitmap.height;
    clone.resize(TARGET_WIDTH, TARGET_HEIGHT, jimp.RESIZE_NEAREST_NEIGHBOR);

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
      rgb[j + 0] = (bitmap.data[i + 0] / 0xff) * 2 - 1;
      rgb[j + 1] = (bitmap.data[i + 1] / 0xff) * 2 - 1;
      rgb[j + 2] = (bitmap.data[i + 2] / 0xff) * 2 - 1;
    }

    const rects = this.computeRects();

    const cellCenter = {
      x: 0.5 / GRID_SIZE,
      y: 0.5 / GRID_SIZE,
    };

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

      const cx = scaledRect.cx * GRID_SIZE - gridX;
      const cy = scaledRect.cy * GRID_SIZE - gridY;

      // Cell is busy
      if (grid[gridOff + 6] !== 0) {
        const distance = {
          present: norm(vector(cellCenter, {
            x: grid[gridOff + 0],
            y: grid[gridOff + 1],
          })),
          candidate: norm(vector(cellCenter, { x: cx, y: cy })),
        };

        // Contents of cell are closer to the cell center
        if (distance.candidate >= distance.present) {
          continue;
        }
      }

      assert(0 <= cx && cx <= 1, `'cx' out of bounds: ${cx}`);
      assert(0 <= cy && cy <= 1, `'cy' out of bounds: ${cy}`);

      grid[gridOff + 0] = cx;
      grid[gridOff + 1] = cy;

      const width = scaledRect.width;
      const height = scaledRect.height;
      assert(0 <= width && width <= 1, '`width` out of bounds');
      assert(0 <= height && height <= 1, '`height` out of bounds');

      grid[gridOff + 2] = width;
      grid[gridOff + 3] = height;

      const cos = Math.cos(scaledRect.angle);
      const sin = Math.sin(scaledRect.angle);

      // Only 1st quadrant
      assert(0 <= cos && cos <= 1, `'cos' out of bounds: ${cos}`);
      assert(0 <= sin && sin <= 1, `'sin' out of bounds: ${sin}`);
      grid[gridOff + 4] = cos;
      grid[gridOff + 5] = sin;

      // Confidence
      grid[gridOff + 6] = 1;
    }

    return {
      rgb,
      grid,
    };
  }

  public async toSVG(rects?: ReadonlyArray<IColoredRect>): Promise<string> {
    const src =  await this.image.getBase64Async(jimp.MIME_PNG);
    const width = this.image.bitmap.width;
    const height = this.image.bitmap.height;

    const img = `<image width="${width}" height="${height}" ` +
      `xlink:href="${src}" />`;

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

      const confidence = rect.confidence === undefined ? 1 : rect.confidence;

      function computeAlpha(x: number): number {
        return Math.exp(1 - 1 / (x + 1e-18));
      }

      const alpha = computeAlpha(confidence);
      const fillAlpha = computeAlpha(Math.max(0, confidence - 0.5));

      const color = rectColor(confidence);

      const stroke = `rgba(${color.join(',')},${alpha})`;
      const fill = `rgba(${color.join(',')},${fillAlpha})`;

      return `<polygon points="${points}" fill="${fill}" stroke="${stroke}"/>`;
    });

    return `
      <svg version="1.1" baseProfile="full"
           width="${width}" height="${height}"
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
      const confidence = prediction[i + 6];
      if (confidence < threshold) {
        continue
      }

      const gridOff = (i / (GRID_CHANNELS * depth)) | 0;
      const gridX = gridOff % GRID_SIZE;
      const gridY = (gridOff / GRID_SIZE) | 0;

      const cos = prediction[i + 4];
      const sin = prediction[i + 5];

      const rect: IColoredRect = {
        cx: (prediction[i + 0] + gridX) * bitmap.width / GRID_SIZE,
        cy: (prediction[i + 1] + gridY) * bitmap.height / GRID_SIZE,
        width: (prediction[i + 2]) * bitmap.width,
        height: (prediction[i + 3]) * bitmap.height,
        angle: Math.atan2(sin, cos),

        confidence,
      };

      rects.push(rect);
    }
    return rects;
  }
}
