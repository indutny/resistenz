import tensorflow as tf
import math
import numpy as np
import os
import json

DIR = os.path.join('.', 'dataset', 'processed')

class Dataset:
  def __init__(self, image_size, grid_size, validate_split=0.15, max_crop=0.1,
               saturation=0.5, brightness=0.2, contrast=0.2):
    self.image_size = image_size
    self.grid_size = grid_size
    self.validate_split = validate_split

    self.max_crop = max_crop
    self.saturation = saturation
    self.brightness = brightness
    self.contrast = contrast

    self.images = [
        os.path.join(DIR, f)
        for f in os.listdir(DIR)
        if f.endswith('.jpg')
    ]

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
    image = tf.image.decode_jpeg(image, channels=3)

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
    image = tf.image.resize_images(image, [ self.image_size, self.image_size ])
    polygons = polygons * float(self.image_size) / \
        tf.cast(crop_size, dtype=tf.float32)

    #
    # Augment image
    #
    if training:
      image = tf.image.random_saturation(image, 1.0 - self.saturation,
          1.0 + self.saturation)
      image = tf.image.random_brightness(image, 1.0 - self.brightness,
          1.0 + self.brightness)
      image = tf.image.random_contrast(image, 1.0 - self.contrast,
          1.0 + self.contrast)
      image = tf.image.rot90(image, tf.random_uniform([], 0, 4, dtype=tf.int32))

    #
    # Change image's type and value range
    #
    image = tf.cast(image, dtype=tf.float32)
    image /= 255.0
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

    cell_offsets = tf.linspace(0.0, 1.0 - 1 / self.grid_size, self.grid_size)
    cell_offsets_x = tf.tile(tf.expand_dims(cell_offsets, axis=0),
        [ self.grid_size, 1 ], name='cell_offsets_x')
    cell_offsets_y = tf.tile(tf.expand_dims(cell_offsets, axis=1),
        [ 1, self.grid_size ], name='cell_offsets_y')

    cell_starts = tf.stack([ cell_offsets_x, cell_offsets_y ], axis=2)
    cell_ends = cell_starts + (1.0 / self.grid_size)

    cell_starts = tf.expand_dims(cell_starts, axis=2, name='cell_starts')
    cell_ends = tf.expand_dims(cell_ends, axis=2, name='cell_ends')

    # Broadcast
    center = tf.expand_dims(rects['center'], axis=0)
    center = tf.expand_dims(center, axis=0, name='broadcast_center')

    rect_count = rects['center'].shape[0]

    indices = tf.range(0, rect_count)
    indices = tf.expand_dims(indices, axis=0)
    indices = tf.expand_dims(indices, axis=0, name='broadcast_indices')

    # Test
    is_in_cell = tf.logical_and(center >= cell_starts, center < cell_ends)
    is_in_cell = tf.reduce_min(tf.cast(is_in_cell, dtype=tf.float32), axis=-1,
        name='is_in_cell')
    is_non_empty_cell = tf.reduce_max(is_in_cell, axis=-1, keepdims=True,
        name='is_non_empty_cell')

    first_in_cell = tf.one_hot(tf.argmax(is_in_cell, axis=-1), depth=rect_count,
        axis=-1, name='first_in_cell') * is_non_empty_cell

    grid = tf.expand_dims(first_in_cell, axis=-1) * rects['rect']
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

    center /= float(self.image_size)
    size /= float(self.image_size)

    rect_count = center.shape[0]

    rect = tf.concat([
      center, size,
      tf.stack([ tf.cos(angle), tf.sin(angle) ], axis=-1, name='angle'),
      tf.tile(tf.constant(1.0, shape=(1,1,)), [ rect_count, 1 ]), # confidence
    ], axis=1)

    return { 'center': center, 'size': size, 'angle': angle, 'rect': rect }

  def triangle_area(self, side1, side2):
    return tf.abs(side1[:, 0] * side2[:, 1] - side1[:, 1] * side2[:, 0]) / 2.0
