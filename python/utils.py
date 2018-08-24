import tensorflow as tf
import math

def create_cell_starts(grid_size):
  cell_offsets = tf.cast(tf.range(grid_size), dtype=tf.float32)
  cell_offsets /= float(grid_size)
  cell_offsets_x = tf.tile(tf.expand_dims(cell_offsets, axis=0),
      [ grid_size, 1 ], name='cell_offsets_x')
  cell_offsets_y = tf.tile(tf.expand_dims(cell_offsets, axis=1),
      [ 1, grid_size ], name='cell_offsets_y')

  cell_starts = tf.stack([ cell_offsets_x, cell_offsets_y ], axis=2)

  return tf.expand_dims(cell_starts, axis=2, name='cell_starts')

def gen_rot_matrix(angle):
  angle *= math.pi

  cos = math.cos(angle)
  sin = math.sin(angle)

  return [
    [ cos, -sin ],
    [ sin, cos ],
  ]

def normalize_image(image):
  min_val = tf.reduce_min(image)
  max_val = tf.reduce_max(image)
  image -= min_val
  image /= (max_val - min_val + 1e-23)
  return image
