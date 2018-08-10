import jimp = require('jimp');
import { IOrientedRect } from './utils';
import { TARGET_WIDTH, TARGET_HEIGHT } from './dataset';

// Max amount of crop from each side
const MAX_CROP_PERCENT = 0.1;

// How much brightness can be adjusted [ 0, 1 ]
const MAX_BRIGHTNESS_DELTA = 0.3;

// How much contrast can be adjusted [ 0, 1 ]
const MAX_CONTRAST_DELTA = 0.3;

export class Input {
  constructor(public readonly image: jimp,
              public readonly rects: ReadonlyArray<IOrientedRect>) {
  }

  public randomize() {
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
      return rect.cx >= crop.left && rect.cx <= (1 - crop.right) &&
             rect.cy >= crop.top && rect.cy <= (1 - crop.bottom);
    });

    // Resize
    clone.resize(TARGET_WIDTH, TARGET_HEIGHT);

    // Random brightness/contrast adjustment
    clone.brightness((Math.random() - 0.5) * MAX_BRIGHTNESS_DELTA);
    clone.contrast((Math.random() - 0.5) * MAX_CONTRAST_DELTA);

    // Return new network input
    return new Input(clone, rects);
  }
}
