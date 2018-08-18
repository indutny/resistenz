import tensorflow as tf

IMAGE_SIZE = 416
# TODO(indutny): there is no reason to not calculate grid_size automatically
GRID_SIZE = 13
GRID_DEPTH = 5
GRID_CHANNELS = 7

# TODO(indutny): add angles (sizes are almost the same)
PRIOR_SIZES = [
  [ 0.15653530649021333, 0.0697987945243159 ],
  [ 0.1683968620145891, 0.07056800049810737 ],
  [ 0.15653530649021333, 0.0697987945243159 ],
  [ 0.1683968620145891, 0.07056800049810737 ],
  [ 0.15653530649021333, 0.0697987945243159 ],
]

class Model:
  def __init__(self,
               prior_sizes=PRIOR_SIZES, iou_threshold=0.5,
               lambda_obj=1.0, lambda_no_obj=0.5, lambda_coord=5.0):
    self.prior_sizes = tf.constant(prior_sizes, dtype=tf.float32)
    self.iou_threshold = iou_threshold

    self.lambda_obj = lambda_obj
    self.lambda_no_obj = lambda_no_obj
    self.lambda_coord = lambda_coord

  def forward(self, image):
    with tf.variable_scope('resistenz', reuse=tf.AUTO_REUSE, values=[ image ]):
      x = image

      # TODO(indutny): noise during training

      x = self.conv_bn(x, filters=16, size=3, name='1')
      x = self.max_pool(x, size=2, stride=2, name='1')
      x = self.conv_bn(x, filters=32, size=3, name='2')
      x = self.max_pool(x, size=2, stride=2, name='2')
      x = self.conv_bn(x, filters=64, size=3, name='3')
      x = self.max_pool(x, size=2, stride=2, name='3')
      x = self.conv_bn(x, filters=128, size=3, name='4')
      x = self.max_pool(x, size=2, stride=2, name='4')
      x = self.conv_bn(x, filters=256, size=3, name='5')
      x = self.max_pool(x, size=2, stride=2, name='5')
      x = self.conv_bn(x, filters=512, size=3, name='6')
      x = self.max_pool(x, size=2, stride=1, name='6')
      x = self.conv_bn(x, filters=1024, size=3, name='pre_final')

      # TODO(indutny): residual routes

      ####

      x = self.conv_bn(x, filters=256, size=1, name='final_1')
      x = self.conv_bn(x, filters=512, size=3, name='final_2')
      x = self.conv_bn(x, filters=GRID_DEPTH * GRID_CHANNELS, size=1,
          name='last', activation=None)

      x = self.output(x)

      return x

  def loss_and_metrics(self, prediction, labels, tag='train'):
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
      max_iou = tf.one_hot(tf.argmax(iou, axis=-1), depth=GRID_DEPTH, axis=-1,
          on_value=True, off_value=False, dtype=tf.bool)

      active_anchors = tf.logical_or(iou >= self.iou_threshold, max_iou)
      active_anchors = tf.cast(active_anchors, dtype=tf.float32,
          name='active_anchors')
      active_anchors *= labels['confidence']

      inactive_anchors = 1.0 - active_anchors

      active_count = tf.reduce_sum(
          tf.reduce_sum(tf.reduce_sum(active_anchors, axis=-1), axis=-1),
          axis=-1, name='active_count') + 1e-29
      inactive_count = tf.reduce_sum(
          tf.reduce_sum(tf.reduce_sum(inactive_anchors, axis=-1), axis=-1),
          axis=-1, name='inactive_count') + 1e-29

      # Confidence loss
      expected_confidence = active_anchors

      confidence_loss = \
          (prediction['confidence'] - expected_confidence) ** 2 / 2.0

      obj_loss = tf.reduce_sum( \
          self.lambda_obj * active_anchors * confidence_loss, axis=-1,
          name='obj_loss')
      no_obj_loss = tf.reduce_sum( \
          self.lambda_no_obj * inactive_anchors * confidence_loss, axis=-1,
          name='no_obj_loss')

      # Coordinate loss
      center_loss = tf.reduce_mean(
          ((prediction['center'] - labels['center']) * GRID_SIZE) ** 2,
          axis=-1, name='center_loss')
      size_loss = tf.reduce_mean(
          (tf.sqrt(prediction['size']) - tf.sqrt(labels['size'])) ** 2,
          axis=-1, name='size_loss')
      angle_loss = angle_diff

      coord_loss = self.lambda_coord * active_anchors * \
          (center_loss + size_loss + angle_loss)
      coord_loss = tf.reduce_sum(coord_loss, axis=-1)

      # To batch losses
      obj_loss = tf.reduce_sum(tf.reduce_sum(obj_loss, axis=-1), axis=-1)
      no_obj_loss = tf.reduce_sum(tf.reduce_sum(no_obj_loss, axis=-1), axis=-1)
      coord_loss = tf.reduce_sum(tf.reduce_sum(coord_loss, axis=-1), axis=-1)

      # Normalize by active/inactive count
      obj_loss /= active_count
      no_obj_loss /= inactive_count
      coord_loss /= active_count

      # To scalars
      obj_loss = tf.reduce_mean(obj_loss)
      no_obj_loss = tf.reduce_mean(no_obj_loss)
      coord_loss = tf.reduce_mean(coord_loss)

      # TODO(indutny): weight decay

      # Total
      total_loss = obj_loss + no_obj_loss + coord_loss

      # Some metrics
      mean_iou = tf.reduce_sum(tf.reduce_sum(iou * active_anchors, axis=-1) / \
          (tf.reduce_sum(active_anchors) + 1e-23))

    # NOTE: create metrics outside of variable scope for clearer name
    metrics = [
      tf.summary.scalar('{}/iou'.format(tag), mean_iou),
      tf.summary.scalar('{}/obj_loss'.format(tag), obj_loss),
      tf.summary.scalar('{}/no_obj_loss'.format(tag), no_obj_loss),
      tf.summary.scalar('{}/coord_loss'.format(tag), coord_loss),
      tf.summary.scalar('{}/loss'.format(tag), total_loss),
    ]

    return total_loss, tf.summary.merge(metrics)

  # Helpers

  def conv_bn(self, input, filters, size, name, activation=tf.nn.leaky_relu):
    x = tf.layers.conv2d(input, filters=filters, kernel_size=size,
        padding='SAME', name='conv_{}'.format(name))
    x = tf.layers.batch_normalization(x, name='bn_{}'.format(name))
    if not activation is None:
      x = activation(x)
    return x

  def max_pool(self, input, size, stride, name):
    return tf.layers.max_pooling2d(input, pool_size=size, strides=stride,
        padding='SAME')

  def output(self, x):
    x = tf.reshape(x, [
      tf.shape(x)[0], GRID_SIZE, GRID_SIZE, GRID_DEPTH, GRID_CHANNELS,
    ])
    center, size, angle, confidence = tf.split(x, [ 2, 2, 2, 1 ], axis=-1)

    center = tf.sigmoid(center)
    size = tf.exp(size) * self.prior_sizes
    angle = tf.nn.l2_normalize(angle, axis=-1)
    confidence = tf.sigmoid(confidence)

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
