import tensorflow as tf

from utils import gen_rot_matrix

IMAGE_SIZE = 416
# TODO(indutny): there is no reason to not calculate grid_size automatically
GRID_SIZE = 13
GRID_CHANNELS = 7

#
# Last computed priors:

# [ [ 0.4203329600221457, 0.15038458315749056 ],
#   [ 0.28172578032055906, 0.11638234408829319 ],
#   [ 0.2833327856632452, 0.10136935933972972 ],
#   [ 0.4179489178405124, 0.17265681086089144 ],
#   [ 0.28889946330244515, 0.11048115994080869 ] ]
#
# We're using the previous ones

PRIOR_SIZE = [ 0.15653530649021333, 0.0697987945243159 ]
PRIOR_ANGLES = [ 0.0, 0.2, 0.4, 0.6, 0.8 ]

class Model:
  def __init__(self, config, prior_size=PRIOR_SIZE, prior_angles=PRIOR_ANGLES):
    self.prior_size = tf.constant(prior_size, dtype=tf.float32,
        name='prior_size')
    self.prior_rotations = tf.constant([
      gen_rot_matrix(angle) for angle in prior_angles
    ], name='prior_rotations')
    self.iou_threshold = config.iou_threshold
    self.weight_decay = config.weight_decay
    self.grid_depth = config.grid_depth

    self.lambda_angle = config.lambda_angle
    self.lambda_obj = config.lambda_obj
    self.lambda_no_obj = config.lambda_no_obj
    self.lambda_coord = config.lambda_coord

    self.trainable_variables = None

  def forward(self, image, training=False):
    with tf.variable_scope('resistenz', reuse=tf.AUTO_REUSE, \
        values=[ image ]) as scope:
      x = image

      # TODO(indutny): noise during training

      x = self.conv_bn(x, filters=16, size=3, name='1', training=training)
      x = self.max_pool(x, size=2, stride=2, name='1')
      x = self.conv_bn(x, filters=32, size=3, name='2', training=training)
      x = self.max_pool(x, size=2, stride=2, name='2')
      x = self.conv_bn(x, filters=64, size=3, name='3', training=training)
      x = self.max_pool(x, size=2, stride=2, name='3')
      x = self.conv_bn(x, filters=128, size=3, name='4', training=training)
      x = self.max_pool(x, size=2, stride=2, name='4')
      x = self.conv_bn(x, filters=256, size=3, name='5', training=training)
      x = self.max_pool(x, size=2, stride=2, name='5')
      x = self.conv_bn(x, filters=512, size=3, name='6', training=training)
      x = self.max_pool(x, size=2, stride=1, name='6')
      x = self.conv_bn(x, filters=1024, size=3, name='pre_final',
          training=training)

      # TODO(indutny): residual routes

      ####

      x = self.conv_bn(x, filters=256, size=1, name='final_1',
          training=training)
      x = self.conv_bn(x, filters=512, size=3, name='final_2',
          training=training)
      x = self.conv_bn(x, filters=self.grid_depth * GRID_CHANNELS, size=1,
          name='last', activation=None, training=training)

      x = self.output(x)

      self.trainable_variables = scope.trainable_variables()

      return x

  def loss_and_metrics(self, prediction, labels, tag='train'):
    # Just a helpers
    def sum_over_cells(x, name=None):
      return tf.reduce_sum(x, axis=3, name=name)

    def sum_over_grid(x, name=None):
      return tf.reduce_sum(tf.reduce_sum(x, axis=2), axis=1, name=name)

    with tf.variable_scope('resistenz_loss_{}'.format(tag), reuse=False, \
        values=[ prediction, labels ]):
      prediction = self.parse_box(prediction, 'prediction')
      labels = self.parse_box(labels, 'labels')

      iou = self.iou(prediction, labels)

      # (cos x - cos y)^2 + (sin x - sin y)^2 = 2 ( 1 - cos [ x - y ] )
      angle_diff = tf.reduce_mean(
          (prediction['angle'] - labels['angle']) ** 2, axis=-1,
          name='angle_diff')
      abs_cos_diff = tf.abs(1.0 - angle_diff, name='abs_cos_diff')

      iou *= abs_cos_diff
      max_iou = tf.one_hot(tf.argmax(iou, axis=-1), depth=self.grid_depth,
          axis=-1, on_value=True, off_value=False, dtype=tf.bool)

      # Compute masks
      active_anchors = tf.logical_or(iou >= self.iou_threshold, max_iou)
      active_anchors = tf.cast(active_anchors, dtype=tf.float32,
          name='active_anchors')
      active_anchors *= labels['confidence']

      inactive_anchors = 1.0 - active_anchors

      expected_confidence = active_anchors

      # Normalize masks
      active_count = sum_over_grid(sum_over_cells(active_anchors),
          name='active_count')

      active_count = tf.expand_dims(active_count, axis=-1)
      active_count = tf.expand_dims(active_count, axis=-1)
      active_count = tf.expand_dims(active_count, axis=-1)

      inactive_count = GRID_SIZE * GRID_SIZE * self.grid_depth - active_count

      active_anchors /= active_count + 1e-23
      inactive_anchors /= inactive_count + 1e-23

      # Confidence loss
      confidence_loss = \
          (prediction['confidence'] - expected_confidence) ** 2 / 2.0

      obj_loss = sum_over_cells( \
          self.lambda_obj * active_anchors * confidence_loss, name='obj_loss')
      no_obj_loss = sum_over_cells( \
          self.lambda_no_obj * inactive_anchors * confidence_loss,
          name='no_obj_loss')

      # Coordinate loss
      center_loss = tf.reduce_mean(
          (GRID_SIZE * (prediction['center'] - labels['center'])) ** 2,
          axis=-1, name='center_loss')
      size_loss = tf.reduce_mean(
          (tf.sqrt(prediction['size']) - tf.sqrt(labels['size'])) ** 2,
          axis=-1, name='size_loss')
      angle_loss = self.lambda_angle * (1.0 - abs_cos_diff)

      coord_loss = self.lambda_coord * active_anchors * \
          (center_loss + size_loss + angle_loss)
      coord_loss = sum_over_cells(coord_loss, name='coord_loss')

      # To batch losses
      obj_loss = sum_over_grid(obj_loss)
      no_obj_loss = sum_over_grid(no_obj_loss)
      coord_loss = sum_over_grid(coord_loss)

      # To scalars
      obj_loss = tf.reduce_mean(obj_loss)
      no_obj_loss = tf.reduce_mean(no_obj_loss)
      coord_loss = tf.reduce_mean(coord_loss)

      # Weight decay
      weight_loss = 0.0
      for var in self.trainable_variables:
        if not 'bn_' in var.name:
          weight_loss += tf.nn.l2_loss(var)
      weight_loss *= self.weight_decay

      # Total
      total_loss = obj_loss + no_obj_loss + coord_loss + weight_loss

      # Some metrics
      mean_iou = sum_over_grid(sum_over_cells(iou * active_anchors))
      mean_iou = tf.reduce_mean(mean_iou)

      center_loss = self.lambda_coord * center_loss * active_anchors
      size_loss = self.lambda_coord * size_loss * active_anchors
      angle_loss = self.lambda_coord * angle_loss * active_anchors

      center_loss = sum_over_grid(sum_over_cells(center_loss))
      size_loss = sum_over_grid(sum_over_cells(size_loss))
      angle_loss = sum_over_grid(sum_over_cells(angle_loss))

      center_loss = tf.reduce_mean(center_loss)
      size_loss = tf.reduce_mean(size_loss)
      angle_loss = tf.reduce_mean(angle_loss)

    # NOTE: create metrics outside of variable scope for clearer name
    metrics = [
      tf.summary.scalar('{}/iou'.format(tag), mean_iou),
      tf.summary.scalar('{}/obj_loss'.format(tag), obj_loss),
      tf.summary.scalar('{}/no_obj_loss'.format(tag), no_obj_loss),
      tf.summary.scalar('{}/coord_loss'.format(tag), coord_loss),
      tf.summary.scalar('{}/center_loss'.format(tag), center_loss),
      tf.summary.scalar('{}/size_loss'.format(tag), size_loss),
      tf.summary.scalar('{}/angle_loss'.format(tag), angle_loss),
      tf.summary.scalar('{}/loss'.format(tag), total_loss),
      tf.summary.scalar('{}/weight_loss'.format(tag), weight_loss),
    ]

    return total_loss, tf.summary.merge(metrics)

  # Helpers

  def conv_bn(self, input, filters, size, name, \
              activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
              training=False):
    x = tf.layers.conv2d(input, filters=filters, kernel_size=size, \
        padding='SAME',
        name='conv_{}'.format(name))
    if not activation is None:
      x = tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5,
          training=training,
          name='bn_{}'.format(name))
      x = activation(x)
    return x

  def max_pool(self, input, size, stride, name):
    return tf.layers.max_pooling2d(input, pool_size=size, strides=stride,
        padding='SAME')

  def output(self, x):
    with tf.name_scope('output', values=[ x ]):
      batch_size = tf.shape(x)[0]

      x = tf.reshape(x, [
        batch_size, GRID_SIZE, GRID_SIZE, self.grid_depth, GRID_CHANNELS,
      ])
      center, size, angle, confidence = tf.split(x, [ 2, 2, 2, 1 ], axis=-1)

      center = tf.sigmoid(center)
      size = tf.exp(size)
      angle = tf.nn.l2_normalize(angle, axis=-1)
      confidence = tf.sigmoid(confidence)

      # Apply priors
      with tf.name_scope('apply_prior_size', values=[ size, self.prior_size ]):
        size *= self.prior_size

      with tf.name_scope('apply_prior_angles',
          values=[ angle, self.prior_rotations ]):
        old_shape = tf.shape(angle)
        angle = tf.reshape(angle,
            shape=[ batch_size * GRID_SIZE * GRID_SIZE, self.grid_depth, 1, 2 ])
        prior_rotations = tf.expand_dims(self.prior_rotations, axis=0)
        prior_rotations = tf.tile(prior_rotations,
            [ batch_size * GRID_SIZE * GRID_SIZE, 1, 1, 1 ])

        angle = tf.matmul(angle, prior_rotations, transpose_b=True)
        angle = tf.reshape(angle, shape=old_shape)

      return tf.concat([ center, size, angle, confidence ], axis=-1,
          name='output')

  def parse_box(self, input, name):
    center, size, angle, confidence = tf.split(input, [ 2, 2, 2, 1 ], axis=-1)
    confidence = tf.squeeze(confidence, axis=-1,
        name='{}_confidence'.format(name))

    center /= GRID_SIZE
    half_size = size / 2.0

    return {
      'center': center,
      'size': size,
      'angle': angle,
      'confidence': confidence,

      'top_left': center - half_size,
      'bottom_right': center + half_size,
      'area': self.area(size, name),
    }

  def area(self, size, name):
    width, height = tf.split(size, [ 1, 1 ], axis=-1)

    return tf.squeeze(width * height, axis=-1, name='{}_area'.format(name))

  def iou(self, a, b):
    top_left = tf.maximum(a['top_left'], b['top_left'], name='iou_top_left')
    bottom_right = tf.minimum(a['bottom_right'], b['bottom_right'],
        name='iou_bottom_right')

    size = tf.nn.relu(bottom_right - top_left, name='iou_size')
    intersection = self.area(size, 'iou_area')
    union = a['area'] + b['area'] - intersection

    return intersection / (union + 1e-23)
