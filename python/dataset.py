import tensorflow as tf
import math
import numpy as np
import os
import json

from utils import create_cell_starts, normalize_image

DIR = os.path.join('.', 'dataset', 'processed')

class Dataset:
  def __init__(self, image_size, grid_size, validation_split=0.15, max_crop=0.1,
               saturation=0.5, exposure=0.2, noise_dev=0.03,
               hue=0.1):
    self.image_size = image_size
    self.grid_size = grid_size
    self.validation_split = validation_split

    self.max_crop = max_crop
    self.saturation = saturation
    self.exposure = exposure
    self.noise_dev = noise_dev
    self.hue = hue

    self.images = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if f.endswith('.jpg')
    ]
    self.images = sorted(self.images)

    self.base_hashes = sorted(
        list(set([ f.split('_', 1)[0] for f in self.images ])))

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
        image_polys.append(4 * [ [ -1000.0, -1000.0 ] ])

    # Just to have stable shape for empty validation data
    self.polygons = np.array(self.polygons)

  def load(self):
    validation_count = int(len(self.base_hashes) * self.validation_split)
    validation_hashes = set(self.base_hashes[:validation_count])

    validation_images = []
    validation_polygons = []
    training_images = []
    training_polygons = []
    for i, image in enumerate(self.images):
      if image.split('_', 1)[0] in validation_hashes:
        validation_images.append(image)
        validation_polygons.append(i)
      else:
        training_images.append(image)
        training_polygons.append(i)

    print('Training dataset has {} images'.format(len(training_images)))
    print('Validation dataset has {} images'.format(len(validation_images)))

    # Do this trick to preserve shape
    training_polygons = self.polygons[training_polygons]
    validation_polygons = self.polygons[validation_polygons]

    validation = self.load_single(validation_images, validation_polygons)
    training = self.load_single(training_images, training_polygons)

    training = training.map( \
        lambda img, polys: self.process_image(img, polys, True))
    validation = validation.map( \
        lambda img, polys: self.process_image(img, polys, False))

    return training, validation

  def load_single(self, images, polygons):
    dataset =  tf.data.Dataset.from_tensor_slices( \
        (tf.constant(images, dtype=tf.string), \
         tf.constant(polygons, dtype=tf.float32),))

    dataset = dataset.map(lambda img, polys: (self.load_image(img), polys,))
    dataset.cache()
    return dataset.shuffle(buffer_size=10000)

  def load_image(self, image):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

  def process_image(self, image, polygons, training):
    #
    # Do a major crop to fit image into a square
    #
    image, polygons = self.major_crop(image, polygons, training)

    if training:
      #
      # Do a rotation on full-size image
      #
      image, polygons = self.random_rotate(image, polygons)

      #
      # Do a minor crop
      #
      image, polygons = self.minor_crop(image, polygons)

    #
    # Resize all images to target size
    #
    crop_size = tf.shape(image)[0]

    check_size = tf.assert_greater_equal(crop_size, self.image_size)
    with tf.control_dependencies([ check_size ]):
      image = tf.image.resize_images(image,
          [ self.image_size, self.image_size ],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    polygons = polygons * float(self.image_size) / \
        tf.cast(crop_size, dtype=tf.float32)

    #
    # Change image's type and value range
    #
    image = tf.cast(image, dtype=tf.float32)
    image /= 255.0

    #
    # Color/exposure manipulation, rotation
    #
    if training:
      image = tf.image.rgb_to_hsv(image)

      # Color
      h, s, v = tf.split(image, [ 1, 1, 1 ], axis=-1)

      saturation_coeff = tf.random_uniform([], 1.0 - self.saturation,
          1.0 + self.saturation)
      exposure_coeff = tf.random_uniform([], 1.0 - self.exposure,
          1.0 + self.exposure)

      s *= saturation_coeff
      s = tf.clip_by_value(s, 0.0, 1.0)

      v *= exposure_coeff
      v = tf.clip_by_value(v, 0.0, 1.0)

      image = tf.concat([ h, s, v ], axis=-1)

      # Exposure

      image = tf.image.hsv_to_rgb(image)

      # TODO(indutny): change hue above too
      image = tf.image.random_hue(image, self.hue)

      # Rotation
      rot_count = tf.random_uniform([], 0, 4, dtype=tf.int32)
      image = tf.image.rot90(image, rot_count)
      polygons = self.rot90_polygons(polygons, rot_count)

    #
    # Add gaussian noise
    #
    if training:
      image += tf.random_normal(shape=tf.shape(image), stddev=self.noise_dev)
      image = tf.clip_by_value(image, 0.0, 1.0)

    image = normalize_image(image)

    return image, self.polygons_to_grid(polygons)

  def random_rotate(self, image, polygons):
    # TODO(indutny): implement me
    return image, polygons

  def major_crop(self, image, polygons, training):
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
    polygons = self.crop_polygons(polygons, crop_off, crop_size)
    return image, polygons

  def minor_crop(self, image, polygons):
    width = tf.shape(image)[0]
    float_width = tf.cast(width, dtype=tf.float32)

    max_allowed_crop = 1 - self.image_size / (float_width + 1e-23)
    max_allowed_crop /= 2.0
    max_allowed_crop = tf.minimum(self.max_crop, max_allowed_crop)

    crop_off = tf.random_uniform([], 0.0, max_allowed_crop) * float_width
    crop_off = tf.cast(crop_off, dtype=tf.int32)

    crop_size = width - 2 * crop_off

    image = tf.image.crop_to_bounding_box(image, crop_off, crop_off,
        crop_size, crop_size)

    # `crop_polygons` expend 2-dim off and size
    crop_off = tf.tile(tf.expand_dims(crop_off, axis=0), [ 2 ])
    crop_size = tf.tile(tf.expand_dims(crop_size, axis=0), [ 2 ])
    polygons = self.crop_polygons(polygons, crop_off, crop_size)
    return image, polygons

  def crop_polygons(self, polygons, crop_off, crop_size):
    # NOTE: `crop_off = [ height, width ]`
    polygons -= tf.cast(tf.gather(crop_off, [ 1, 0 ]), dtype=tf.float32)
    polygon_centers = tf.reduce_mean(polygons, axis=1)

    # Coordinate-wise mask
    polygon_mask = tf.logical_and(polygon_centers >= 0.0, \
        polygon_centers <= tf.cast(crop_size, dtype=tf.float32))

    # Polygon-wise mask
    polygon_mask = tf.logical_and(polygon_mask[:, 0], polygon_mask[:, 1])

    return tf.where(polygon_mask, polygons, -tf.ones_like(polygons))

  def rot90_polygons(self, polygons, rot_count):
    angle = (math.pi / 2.0) * tf.cast(rot_count, dtype=tf.float32)

    cos = tf.cos(angle)
    sin = tf.sin(angle)

    center = tf.constant([ self.image_size / 2.0 ], dtype=tf.float32)
    matrix = tf.reshape(tf.stack([ cos, -sin, sin, cos ]), shape=[ 2, 2 ])

    # Flatten
    old_shape = polygons.shape
    polygons = tf.reshape(polygons, [ old_shape[0] * old_shape[1], 2 ])

    # Rotate
    polygons = tf.matmul(polygons - center, matrix) + center

    # Restore shape
    polygons = tf.reshape(polygons, old_shape)
    return polygons

  def polygons_to_grid(self, polygons):
    rects = self.polygons_to_rects(polygons)

    cell_starts = create_cell_starts(self.grid_size)

    # Broadcast
    center = tf.expand_dims(rects['center'], axis=0)
    center = tf.expand_dims(center, axis=0, name='broadcast_center')
    center -= cell_starts

    rect_count = rects['center'].shape[0]

    # Test
    is_in_cell = tf.logical_and(center >= 0.0, center < 1 / self.grid_size)
    is_in_cell = tf.reduce_min(tf.cast(is_in_cell, dtype=tf.float32), axis=-1,
        name='is_in_cell')
    is_non_empty_cell = tf.reduce_max(is_in_cell, axis=-1, keepdims=True,
        name='is_non_empty_cell')

    first_in_cell = tf.one_hot(tf.argmax(is_in_cell, axis=-1), depth=rect_count,
        axis=-1, name='first_in_cell') * is_non_empty_cell

    # Tile sizes, angles, and confidence
    rest = rects['rest']
    rest = tf.reshape(rest, [ 1, 1, rest.shape[0], rest.shape[-1] ])
    rest = tf.tile(rest, [ self.grid_size, self.grid_size, 1, 1 ],
        name='broadcast_rest')

    # Rescale center so that it would be in [ 0, 1) range
    center *= float(self.grid_size)

    rect = tf.concat([ center, rest ], axis=-1)

    grid = tf.expand_dims(first_in_cell, axis=-1) * rect
    grid = tf.reduce_sum(grid, axis=2, name='shallow_grid')

    # Add extra dimension for grid depth
    grid = tf.expand_dims(grid, axis=2, name='grid')

    return grid

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
    angle = tf.stack([ tf.cos(angle), tf.sin(angle) ], axis=-1, name='angle')

    # Rescale offsets, sizes to be a percent of image size
    center /= float(self.image_size)
    size /= float(self.image_size)

    rect_count = center.shape[0]
    confidence = tf.ones([ rect_count, 1 ], dtype=tf.float32)

    rest = tf.concat([ size, angle, confidence ], axis=-1)

    return { 'center': center, 'rest': rest }

  def triangle_area(self, side1, side2):
    return tf.abs(side1[:, 0] * side2[:, 1] - side1[:, 1] * side2[:, 0]) / 2.0
