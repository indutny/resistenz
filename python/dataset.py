import tensorflow as tf
import os
import json

IMAGE_SIZE = 416
MAX_CROP = 0.1

class Dataset:
  def __init__(self, validate_split=0.15):
    self.validate_split = 0.15
    self.images = [
        './dataset/processed/{}'.format(f)
        for f in os.listdir('./dataset/processed')
        if f.endswith('.jpg')
    ]
    self.images = self.images[:2]

    self.polygons = []
    max_polys = 0
    for image_file in self.images:
      json_file = image_file[:-4] + '.json'
      with open(json_file, 'r') as f:
        raw_labels = json.load(f)

      image_polys = []
      for poly in raw_labels['polygons']:
        if len(poly) != 4:
          continue
        poly_points = [ [ float(p['x']), float(p['y']) ] for p in poly ]
        image_polys.append(poly_points)

      self.polygons.append(image_polys)
      max_polys = max(max_polys, len(image_polys))

    # Pad polygons
    for image_polys in self.polygons:
      while (len(image_polys) < max_polys):
        image_polys.append(4 * [ [ 0.0, 0.0 ] ])

  def load(self):
    validate_count = int(len(self.images) * self.validate_split)
    validation = self.load_single(self.images[:validate_count], \
        self.polygons[:validate_count])
    training = self.load_single(self.images[validate_count:], \
        self.polygons[validate_count:])

    training = training.map( \
        lambda img, polys: self.process_image(img, polys, True))
    validation = validation.map( \
        lambda img, polys: self.process_image(img, polys, False))

    training = training.shuffle(buffer_size=10000)

    return training, validation

  def process_image(self, image, polygons, training):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image)

    size = tf.shape(image)[:2]
    width = size[1]
    height = size[0]

    crop_size = tf.reduce_min(size, axis=-1, name='crop_size')
    size_delta = size - crop_size
    if training:
      # Random crop for training
      crop_off = tf.cast(size_delta, dtype=tf.float32) * \
          tf.random_uniform([ 2 ])
      crop_off = tf.cast(crop_off, dtype=tf.int32)
    else:
      # Central crop for validation
      crop_off = size_delta // 2

    image = tf.image.crop_to_bounding_box(image, crop_off[0], crop_off[1], \
        crop_size, crop_size)
    print(polygons)

    image = tf.image.resize_images(image, [ IMAGE_SIZE, IMAGE_SIZE ])
    return image, polygons

  def load_single(self, images, polygons):
    return tf.data.Dataset.from_tensor_slices( \
        (tf.constant(images, dtype=tf.string), \
         tf.constant(polygons, dtype=tf.float32),))
