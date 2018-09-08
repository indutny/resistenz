import tensorflow as tf

from utils import FLAT_COLOR_DIMS, COLOR_DIMS

IMAGE_SIZE = 416
# TODO(indutny): there is no reason to not calculate grid_size automatically
GRID_SIZE = 13
GRID_CHANNELS = 7

PRIOR_SIZES = [
  [ 0.14377480392797287, 0.059023397839700086 ],
  [ 0.20904473801128326, 0.08287369797830041 ],
  [ 0.2795802996888472, 0.11140121237843759 ],
  [ 0.3760081365223815, 0.1493933380505552 ],
  [ 0.5984967942142249, 0.2427157057261726 ],
]

class Model:
  def __init__(self, config, prior_sizes=PRIOR_SIZES):
    self.config = config

    self.prior_sizes = tf.constant(prior_sizes, dtype=tf.float32,
        name='prior_sizes')
    self.iou_threshold = config.iou_threshold
    self.weight_decay = config.weight_decay
    self.grid_depth = config.grid_depth

    self.lambda_angle = config.lambda_angle
    self.lambda_obj = config.lambda_obj
    self.lambda_no_obj = config.lambda_no_obj
    self.lambda_coord = config.lambda_coord

    self.trainable_variables = None

  def forward(self, image, training=False, coreml=False):
    with tf.variable_scope('resistenz', reuse=tf.AUTO_REUSE, \
        values=[ image ]) as scope:
      x = image

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

      # TODO(indutny): residual routes
      if not self.config.minimal:
        x = self.conv_bn(x, filters=1024, size=3, name='pre_final',
                         training=training)

      ####

      if not self.config.minimal:
        x = self.conv_bn(x, filters=256, size=1, name='final_1',
                         training=training)
        x = self.conv_bn(x, filters=512, size=3, name='final_2',
                         training=training)
      else:
        x = self.conv_bn(x, filters=128, size=3, name='final_2',
                         training=training)
      x = self.conv_bn(x, filters=self.grid_depth * GRID_CHANNELS + \
          FLAT_COLOR_DIMS, size=1,
          name='last', activation=None, training=training)

      x, raw_colors = self.output(x, coreml=coreml)

      self.trainable_variables = scope.trainable_variables()

      return x, raw_colors

  def loss_and_metrics(self, prediction, prediction_colors, labels, \
                       tag='train'):
    # Just a helpers
    def sum_over_cells(x, name=None, max=False):
      if max:
        return tf.reduce_max(x, axis=3, name=name)
      else:
        return tf.reduce_sum(x, axis=3, name=name)

    def sum_over_grid(x, name=None, max=False):
      if max:
        return tf.reduce_max(tf.reduce_max(x, axis=2), axis=1, name=name)
      else:
        return tf.reduce_sum(tf.reduce_sum(x, axis=2), axis=1, name=name)

    with tf.variable_scope('resistenz_loss_{}'.format(tag), reuse=False, \
        values=[ prediction, prediction_colors, labels ]):
      prediction = self.parse_box(prediction, 'prediction')
      labels = self.parse_box(labels, 'labels')

      iou = self.iou(prediction, labels)

      # (cos x - cos y)^2 + (sin x - sin y)^2 = 2 ( 1 - cos [ x - y ] )
      angle_diff = tf.reduce_mean(
          (prediction['angle'] - labels['angle']) ** 2, axis=-1,
          name='angle_diff')
      abs_cos_diff = tf.abs(1.0 - angle_diff, name='abs_cos_diff')

      iou *= abs_cos_diff

      # Compute masks
      active_anchors = tf.one_hot(tf.argmax(iou, axis=-1), depth=self.grid_depth,
          axis=-1, on_value=1.0, off_value=0.0, dtype=tf.float32,
          name='active_anchors')
      active_anchors *= labels['confidence']

      # Disable training for anchors with high IoU
      passive_anchors = labels['confidence']
      passive_anchors *= tf.cast(iou >= self.iou_threshold, dtype=tf.float32)

      inactive_anchors = 1.0 - tf.maximum(active_anchors, passive_anchors)
      inactive_anchors = tf.identity(inactive_anchors, name='inactive_anchors')

      expected_confidence = active_anchors

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
      total_loss = obj_loss + no_obj_loss + coord_loss
      regularization_loss = weight_loss

      # Count objects for metrics below
      active_count = sum_over_grid(sum_over_cells(active_anchors),
          name='active_count')

      active_count = tf.expand_dims(active_count, axis=-1)
      active_count = tf.expand_dims(active_count, axis=-1)
      active_count = tf.expand_dims(active_count, axis=-1)

      # Some metrics
      mean_anchors = active_anchors / (active_count + 1e-23)
      mean_iou = sum_over_grid(sum_over_cells(iou * mean_anchors))
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

    return total_loss + regularization_loss, tf.summary.merge(metrics)

  # Helpers

  def conv_bn(self, input, filters, size, name, training, \
              activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1)) :
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

  def output(self, x, coreml=False):
    with tf.name_scope('output', values=[ x ]):
      batch_size = tf.shape(x)[0]

      if coreml:
        # CoreML does not support rank-5 tensors, strided slices, and so on
        x = tf.reshape(x, [
          batch_size, GRID_SIZE, GRID_SIZE,
          FLAT_COLOR_DIMS + self.grid_depth * GRID_CHANNELS,
        ], name='output')
        return x

      x, colors = tf.split(x, \
          [ self.grid_depth * GRID_CHANNELS, FLAT_COLOR_DIMS ], axis=-1)

      x = tf.reshape(x, [
        batch_size, GRID_SIZE, GRID_SIZE, self.grid_depth, GRID_CHANNELS,
      ])

      center, size, angle, confidence = \
          tf.split(x, [ 2, 2, 2, 1 ], axis=-1)

      center = tf.sigmoid(center)
      size = tf.exp(size)
      angle = tf.nn.l2_normalize(angle, axis=-1)
      confidence = tf.sigmoid(confidence)

      # Apply softmax over each color group
      raw_colors = tf.split(colors, COLOR_DIMS, axis=-1)
      split_colors = [ tf.nn.softmax(l, axis=-1) for l in raw_colors ]
      colors = tf.concat(split_colors, axis=-1)

      # Apply priors
      with tf.name_scope('apply_prior_sizes',
                         values=[ size, self.prior_sizes ]):
        size *= self.prior_sizes

      colors = tf.expand_dims(colors, axis=-2)
      colors = tf.tile(colors, [ 1, 1, 1, self.grid_depth, 1 ])
      x = tf.concat([ center, size, angle, confidence, colors ], axis=-1,
          name='output')

      # Return raw_colors for use in the loss
      return x, raw_colors

  def parse_box(self, input, name):
    center, size, angle, confidence, colors = tf.split(input, \
        [ 2, 2, 2, 1, FLAT_COLOR_DIMS ], \
        axis=-1, name='{}_box_split'.format(name))
    confidence = tf.squeeze(confidence, axis=-1,
        name='{}_confidence'.format(name))

    center /= GRID_SIZE
    half_size = size / 2.0

    return {
      'center': center,
      'size': size,
      'angle': angle,
      'confidence': confidence,

      'colors': colors,

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
