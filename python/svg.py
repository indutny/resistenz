import os
import tensorflow as tf

from utils import create_cell_starts

class SVG:
  def __init__(self, raw, grid, label):
    self.raw = raw
    self.grid = grid
    self.label = label

  def write_file(self, filename):
    with tf.name_scope('svg'):
      image = tf.cast(self.raw * 255.0, dtype=tf.uint8)
      height = int(image.shape[0])
      width = int(image.shape[1])

      jpeg = tf.image.encode_jpeg(image, quality=80)
      web_base64 = tf.encode_base64(jpeg, pad=True)

      # I'm not fond of this, but...
      base64 = tf.regex_replace(web_base64, '-', '+')
      base64 = tf.regex_replace(base64, '_', '/')

      svg = '<svg version="1.1" baseProfile="full" ' + \
          'width="{}" height="{}" '.format(width, height) + \
          'xmlns="http://www.w3.org/2000/svg" ' + \
          'xmlns:xlink="http://www.w3.org/1999/xlink">\n'

      # Write image at bottom to make reading SVG source easier
      svg += '  <image width="{}" height="{}" '.format(width, height) + \
          'xlink:href="data:image/png;base64,' + base64 + '"/>\n'

      # Compute cell offsets
      grid_size = int(self.grid.shape[0])
      grid_depth = int(self.grid.shape[2])
      grid_channels = int(self.grid.shape[3])

      cell_starts = create_cell_starts(grid_size) * float(grid_size)
      rest = tf.zeros([ grid_size, grid_size, 1, grid_channels - 2 ])
      cell_starts = tf.concat([ cell_starts, rest ], axis=-1)

      # Add cell offsets
      grid = self.grid + cell_starts

      # Fix dimensions
      grid = grid * tf.constant([
        float(width) / grid_size , float(height) / grid_size,
        float(width), float(height),
        1, 1, 1 ]);

      # Make grid linear
      grid = tf.reshape(grid,
          [ grid_size * grid_size * grid_depth, grid_channels ])

      svg += tf.foldl(lambda acc, cell: acc + self.cell_to_polygon(cell), grid,
          initializer=tf.constant('', tf.string))

      svg += '</svg>\n'

      return tf.write_file(filename, svg)

  def cell_to_polygon(self, cell):
    center, size, angle, confidence = tf.split(cell, [ 2, 2, 2, 1 ], axis=-1)
    confidence = tf.squeeze(confidence, axis=-1)

    flip_vec = tf.constant([ -1.0, 1.0 ])

    half_size = size / 2.0
    skew_half_size = half_size * flip_vec

    rot_matrix = tf.stack([ angle, tf.gather(angle, [ 1, 0 ]) * flip_vec ],
        axis=0)

    top_left = -half_size
    top_right = -skew_half_size
    bottom_right = half_size
    bottom_left = skew_half_size

    rect = tf.stack([ top_left, top_right, bottom_right, bottom_left ], axis=0)
    rect = tf.matmul(rect, rot_matrix)
    rect += center

    def point_to_str(point):
      return tf.as_string(point[0]) + ',' + tf.as_string(point[1])

    points = point_to_str(rect[0]) + ' ' + point_to_str(rect[1]) + ',' + \
        point_to_str(rect[2]) + ' ' + point_to_str(rect[3])

    alpha = tf.exp(1.0 - 1.0 / (confidence + 1e-23), name='alpha')

    color = tf.where(confidence >= 0.5, '0,255,0', '255,0,0', name='color')

    fill = 'none'
    stroke = 'rgba(' +  color + ',' + tf.as_string(alpha) + ')'
    return tf.where(confidence >= 0.1, \
        '  <polygon points="' + points + '" fill="' + fill + \
        '" stroke="' + stroke + '"/>\n', '')
