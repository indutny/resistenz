import jimp = require('jimp');
import { Buffer } from 'buffer';

import { IPoint, IOrientedRect, rotate } from './utils';
import { TARGET_WIDTH, TARGET_HEIGHT } from './dataset';

// Max amount of crop from each side
const MAX_CROP_PERCENT = 0.2;

// How much brightness can be adjusted [ 0, 1 ]
const MAX_BRIGHTNESS_DELTA = 0.5;

// How much contrast can be adjusted [ 0, 1 ]
const MAX_CONTRAST_DELTA = 0.5;

export class Input {
  constructor(public readonly image: jimp,
              public readonly rects: ReadonlyArray<IOrientedRect>) {
  }

  public randomize(): Input {
    const clone = this.image.clone();
    let rects = this.rects.slice();

    const width = clone.bitmap.width;
    const height = clone.bitmap.height;

    // Randomly crop
    const crop = {
      top: Math.random() * MAX_CROP_PERCENT,
      bottom: Math.random() * MAX_CROP_PERCENT,
      left: Math.random() * MAX_CROP_PERCENT,
      right: Math.random() * MAX_CROP_PERCENT,
    };

    const cropX = Math.floor(crop.left * width);
    const cropY = Math.floor(crop.top * height);
    const cropW = width - cropX - Math.floor(crop.right * width);
    const cropH = height - cropY - Math.floor(crop.bottom * height);

    clone.crop(cropX, cropY, cropW, cropH);
    rects = rects.filter((rect) => {
      return rect.cx >= cropX && rect.cx <= cropX + cropW &&
             rect.cy >= cropY && rect.cy <= cropY + cropH;
    });

    // Resize
    const scaleX = TARGET_WIDTH / width;
    const scaleY = TARGET_HEIGHT / height;
    clone.resize(TARGET_WIDTH, TARGET_HEIGHT);

    rects = rects.map((rect) => {
      return {
        cx: rect.cx * scaleX,
        cy: rect.cy * scaleY,
        // TODO(indutny): rotate and scale
        width: rect.width,
        height: rect.height,
        angle: rect.angle,
      };
    });

    // Random brightness/contrast adjustment
    clone.brightness((Math.random() - 0.5) * MAX_BRIGHTNESS_DELTA);
    clone.contrast((Math.random() - 0.5) * MAX_CONTRAST_DELTA);

    // Return new network input
    return new Input(clone, rects);
  }

  public async toSVG(): Promise<string> {
    // TODO(indutny): remove this after Jimp bug is fixed
    const png: Buffer =
      (await this.image.getBufferAsync(jimp.MIME_JPEG)) as any;
    const src = 'data:image/png;base64,' + png.toString('base64');
    const img = `<image xlink:href="${src}" />`;

    const width = this.image.bitmap.width;
    const height = this.image.bitmap.height;

    const polygons = this.rects.map((rect) => {
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

      return `<polygon points="${points}" fill="none" stroke="red"/>`;
    });

    return `
      <svg width="${width}" height="${height}"
           xmlns="http://www.w3.org/2000/svg"
           xmlns:xlink="http://www.w3.org/1999/xlink">
        ${img}
        ${polygons.join('\n')}
      </svg>`;
  }
}
