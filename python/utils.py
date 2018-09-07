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

DIGIT = [
  'black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
  'grey', 'white',
]

MULTIPLY = [
  'none',
  'black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
  'gold', 'silver',
]

TOLERANCE = [
  'none',
  'brown', 'red', 'green', 'blue', 'violet', 'grey', 'gold', 'silver',
]

TEMPERATURE = [
  'none',
  'black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
  'grey',
]

COLOR_DIMS = [ \
    len(DIGIT), len(DIGIT), len(DIGIT), len(MULTIPLY), len(TOLERANCE), \
    len(TEMPERATURE) ]

def colors_to_int(colors):
  dig1 = DIGIT.index(colors[0])
  dig2 = DIGIT.index(colors[1])
  dig3 = DIGIT.index(colors[2])
  multiply = MULTIPLY.index(colors[3])
  tolerance = TOLERANCE.index(colors[4])
  temperature = TEMPERATURE.index(colors[5])

  if dig1 is None:
    raise Exception('Invalid first digit')
  if dig2 is None:
    raise Exception('Invalid first digit')
  if dig3 is None:
    raise Exception('Invalid first digit')
  if multiply is None:
    raise Exception('Invalid first digit')
  if tolerance is None:
    raise Exception('Invalid first digit')
  if temperature is None:
    raise Exception('Invalid first digit')

  return [ dig1, dig2, dig3, multiply, tolerance, temperature ]
