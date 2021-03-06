import os
import tensorflow as tf

from dataset import Dataset
from model import Model, IMAGE_SIZE, GRID_SIZE
from args import parse_args
from svg import SVG

# TODO(indutny): move these to args?
LOG_DIR = os.path.join('.', 'logs')
IMAGE_DIR = os.path.join('.', 'images')

args, tag = parse_args()
print('Running with a tag "{}"'.format(tag))

SAVE_DIR = os.path.join('.', 'saves', tag)

model = Model(args)

with tf.Session() as sess:
  dataset = Dataset(image_size=IMAGE_SIZE, grid_size=GRID_SIZE)

  training, validation = dataset.load()

  # Real datasets and their iterators
  training = training.batch(args.batch_size).prefetch(4 * args.batch_size)
  validation = validation.batch(args.batch_size).cache()

  training_iter = training.make_initializable_iterator()
  validation_iter = validation.make_initializable_iterator()

  training_batch = training_iter.get_next()
  validation_batch = validation_iter.get_next()

  # Predictions
  # NOTE: yes, this compiles both twice... but perhaps it is faster this way?
  training_pred, training_colors, training_raw_colors = \
      model.forward(training_batch[0], training=True)
  validation_pred, validation_colors, validation_raw_colors = \
      model.forward(validation_batch[0])

  # Encode first images of each epoch for debugging purposes
  def svg_op(batch, pred, pred_colors, fname):
    svg = SVG(batch[0][0], pred[0], pred_colors[0], batch[1][0])
    return svg.write_file(os.path.join(IMAGE_DIR, fname))

  svg_op = {
    'training': svg_op(training_batch, training_pred, training_colors, \
        'train.svg'),
    'validation': svg_op(validation_batch, validation_pred, validation_colors, \
        'validate.svg'),
  }

  # Steps
  epoch = tf.Variable(name='epoch', initial_value=0, dtype=tf.int32)
  epoch_inc = epoch.assign_add(1)

  global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)
  validation_step = tf.Variable(name='validation_step', initial_value=0,
      dtype=tf.int32)
  validation_step_inc = validation_step.assign_add(1)

  # Losses and metrics
  training_loss, training_metrics = \
      model.loss_and_metrics(training_pred, training_raw_colors, \
                             training_batch[1])
  validation_loss, validation_metrics = \
      model.loss_and_metrics(validation_pred, validation_raw_colors, \
                             validation_batch[1], 'val')

  # Learing rate schedule
  def lr_schedule(epoch):
    def spline(from_epoch, from_val, to_epoch, to_val):
      t = tf.to_float(epoch) - tf.constant(from_epoch, dtype=tf.float32)
      t /= tf.constant(to_epoch - from_epoch, dtype=tf.float32)
      t = tf.clip_by_value(t, 0.0, 1.0)

      a = 2.0 * (from_val - to_val)
      b = 3.0 * (to_val - from_val)
      d = from_val

      res = a * t ** 3 + b * t ** 2 + d

      return res

    with tf.name_scope('lr', values=[ epoch ]):
      initial = args.lr
      fast = args.lr_fast
      fast_epoch = args.lr_fast_epoch
      slow = args.lr_slow
      slow_epoch = args.lr_slow_epoch

      lr = tf.where(epoch < slow_epoch,
          spline(fast_epoch, fast, slow_epoch, slow), slow)
      lr = tf.where(epoch < fast_epoch,
          spline(0, initial, fast_epoch, fast), lr)

      return lr

  lr = lr_schedule(epoch)
  training_metrics = tf.summary.merge([
    training_metrics,
    tf.summary.scalar('train/lr', lr),
  ])

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(lr, args.momentum)
    minimize = optimizer.minimize(training_loss, global_step)

  sess.run(tf.global_variables_initializer())

  writer = tf.summary.FileWriter(os.path.join(LOG_DIR, tag))
  writer.add_graph(tf.get_default_graph())

  saver = tf.train.Saver(max_to_keep=10, name=tag)

  sess.graph.finalize()

  if not args.restore is None:
    print('Restoring from "{}"'.format(args.restore))
    saver.restore(sess, args.restore)

  epoch_value = sess.run(epoch)
  while epoch_value < args.epochs:
    print('Epoch {}'.format(epoch_value))

    sess.run([ training_iter.initializer, validation_iter.initializer ])

    batches = 0
    while True:
      try:
        _, metrics, step, _ = sess.run([
          minimize, training_metrics, global_step,
          svg_op['training'],
        ])
        batches += 1

        writer.add_summary(metrics, step)
        writer.flush()
      except tf.errors.OutOfRangeError:
        break
    print('Completed {} training batches'.format(batches))

    batches = 0
    while True:
      try:
        metrics, step, _ = sess.run([
          validation_metrics, validation_step_inc,
          svg_op['validation'],
        ])
        batches += 1

        writer.add_summary(metrics, step)
        writer.flush()
      except tf.errors.OutOfRangeError:
        break
    print('Completed {} validation batches'.format(batches))

    epoch_value = sess.run(epoch_inc)

    if epoch_value % args.save_every == 0:
      print('Saving...')
      saver.save(sess, os.path.join(SAVE_DIR, '{:08d}'.format(epoch_value)))
