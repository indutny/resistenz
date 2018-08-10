import jimp = require('jimp');
import { Buffer } from 'buffer';

import { Polygon, IPoint, IOrientedRect, polygonToRect, rotate } from './utils';
import { TARGET_WIDTH, TARGET_HEIGHT } from './dataset';

// Maximum angle of rotation
const MAX_ROT_ANGLE = 360;

// Max amount of crop from each side
const MAX_CROP_PERCENT = 0.2;

// How much brightness can be adjusted [ 0, 1 ]
const MAX_BRIGHTNESS_DELTA = 0.5;

// How much contrast can be adjusted [ 0, 1 ]
const MAX_CONTRAST_DELTA = 0.5;

export class Input {
  constructor(public readonly image: jimp,
              public readonly polys: ReadonlyArray<Polygon>) {
  }

  public randomize(): Input {
    const clone = this.image.clone();
    let polys = this.polys.slice();

    const width = clone.bitmap.width;
    const height = clone.bitmap.height;

    const center = { x: width / 2, y: height / 2 };

    // Randomly rotate
    const angleDeg = Math.random() * MAX_ROT_ANGLE;
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

    // Resize
    const scaleX = TARGET_WIDTH / cropW;
    const scaleY = TARGET_HEIGHT / cropH;
    clone.resize(TARGET_WIDTH, TARGET_HEIGHT);

    polys = polys.map((points) => {
      return points.map((p) => {
        return { x: p.x * scaleX, y: p.y * scaleY };
      });
    });

    // Random brightness/contrast adjustment
    clone.brightness((Math.random() - 0.5) * MAX_BRIGHTNESS_DELTA);
    clone.contrast((Math.random() - 0.5) * MAX_CONTRAST_DELTA);

    // Return new network input
    return new Input(clone, polys);
  }

  public async toSVG(): Promise<string> {
    // TODO(indutny): remove this after Jimp bug is fixed
    const png: Buffer =
      (await this.image.getBufferAsync(jimp.MIME_JPEG)) as any;
    const src = 'data:image/png;base64,' + png.toString('base64');
    const img = `<image xlink:href="${src}" />`;

    const width = this.image.bitmap.width;
    const height = this.image.bitmap.height;

    const polygons = this.polys.map((poly) => {
      return polygonToRect(poly);
    }).map((rect) => {
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
