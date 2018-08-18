import tensorflow as tf
import math
import numpy as np
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
        image_polys.append(4 * [ [ -1.0, -1.0 ] ])

    # Just to have stable shape for empty validation data
    self.polygons = np.array(self.polygons)

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

    #
    # TODO(indutny): do a minor crop
    #

    #
    # Do a major crop to fit image into a square
    #
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

    polygons = self.crop_polygons(polygons, crop_off, crop_size)

    image = tf.image.crop_to_bounding_box(image, crop_off[0], crop_off[1], \
        crop_size, crop_size)

    #
    # Resize all images to target size
    #
    image = tf.image.resize_images(image, [ IMAGE_SIZE, IMAGE_SIZE ])
    return image, self.polygons_to_grid(polygons)

  def load_single(self, images, polygons):
    return tf.data.Dataset.from_tensor_slices( \
        (tf.constant(images, dtype=tf.string), \
         tf.constant(polygons, dtype=tf.float32),))

  def crop_polygons(self, polygons, crop_off, crop_size):
    polygons -= tf.cast(crop_off, dtype=tf.float32)
    polygon_centers = tf.reduce_mean(polygons, axis=1)

    # Coordinate-wise mask
    polygon_mask = tf.logical_and(polygon_centers >= 0.0, \
        polygon_centers <= tf.cast(crop_size, dtype=tf.float32))

    # Polygon-wise mask
    polygon_mask = tf.logical_and(polygon_mask[:, 0], polygon_mask[:, 1])

    return tf.where(polygon_mask, polygons, -tf.ones_like(polygons))

  def polygons_to_grid(self, polygons):
    rects = self.polygons_to_rects(polygons)
    return rects

  def polygons_to_rects(self, polygons):
    center = tf.reduce_mean(polygons, axis=1)

    p0, p1, p2, p3 = \
        polygons[:, 0], polygons[:, 1], polygons[:, 2], polygons[:, 3]

    diag02 = p0 - p2
    diag13 = p1 - p3

    diag = (tf.norm(diag02, axis=-1) + tf.norm(diag13, axis=-1)) / 2.0

    v01 = p0 - p1
    v03 = p0 - p3
    v21 = p2 - p1
    v23 = p2 - p3

    area = self.triangle_area(v01, v03) + self.triangle_area(v21, v23)

    # Compute box width/height using quadratic equation
    disc = tf.sqrt(diag ** 4 - 4 * area ** 2)

    # NOTE: `abs` is added just in case, to prevent nan on disc close to 0
    width = tf.sqrt(tf.abs(diag ** 2 + disc) / 2.0)
    height = tf.sqrt(tf.abs(diag ** 2 - disc) / 2.0)
    size = tf.stack([ width, height ], axis=1)

    # Find longest side
    sides = tf.stack([ v01, v03, v21, v23 ], axis=1)
    side_lens = tf.norm(sides, axis=-1)
    max_side_i = tf.argmax(side_lens, axis=1)
    max_side_hot = tf.expand_dims(tf.one_hot(max_side_i, 4), axis=-1)
    max_side = tf.reduce_sum(max_side_hot * sides, axis=1)

    angle = tf.atan2(max_side[:, 1], max_side[:, 0])
    angle = tf.where(angle < 0.0, angle + math.pi, angle)

    return tf.concat([ center, size, tf.expand_dims(angle, axis=-1) ], axis=1)

  def triangle_area(self, side1, side2):
    return tf.abs(side1[:, 0] * side2[:, 1] - side1[:, 1] * side2[:, 0]) / 2.0
