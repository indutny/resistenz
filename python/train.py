import os
import tensorflow as tf

from dataset import Dataset
from model import Model, IMAGE_SIZE, GRID_SIZE
from args import parse_args

args, tag = parse_args()
print('Running with a tag "{}"'.format(tag))

model = Model()

optimizer = tf.train.MomentumOptimizer(args.lr, args.momentum)

with tf.Session() as sess:
  dataset = Dataset(image_size=IMAGE_SIZE, grid_size=GRID_SIZE)

  training, validation = dataset.load()

  # Just a single image for epoch to save to SVG
  training_single = training.batch(1).repeat()
  validation_single = validation.batch(1).repeat()

  # Real datasets and their iterators
  training = training.batch(args.batch_size)
  validation = validation.batch(args.batch_size)

  training_iter = training.make_initializable_iterator()
  validation_iter = validation.make_initializable_iterator()

  training_batch = training_iter.get_next()
  validation_batch = validation_iter.get_next()

  # Predictions
  # NOTE: yes, this compiles both twice... but perhaps it is faster this way?
  training_pred = model.forward(training_batch[0])
  validation_pred = model.forward(validation_batch[0])

  # Steps
  global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)
  validation_step = tf.Variable(name='validation_step', initial_value=0,
      dtype=tf.int32)
  validation_step_inc = validation_step.assign_add(1)

  # Losses and metrics
  training_loss, training_metrics = \
      model.loss_and_metrics(training_pred, training_batch[1])
  validation_loss, validation_metrics = \
      model.loss_and_metrics(validation_pred, validation_batch[1], 'val')

  minimize = optimizer.minimize(training_loss, global_step)

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  writer = tf.summary.FileWriter(os.path.join('.', 'logs', tag))
  writer.add_graph(tf.get_default_graph())

  for i in range(0, args.epochs):
    print('Epoch {}'.format(i))
    batches = 0

    sess.run([ training_iter.initializer, validation_iter.initializer ])
    while True:
      try:
        _, metrics, step = sess.run([ minimize, training_metrics, global_step ])
        batches += 1
      except tf.errors.OutOfRangeError:
        break
      writer.add_summary(metrics, step)
      writer.flush()

    while True:
      try:
        metrics, step = sess.run([ validation_metrics, validation_step_inc ])
      except tf.errors.OutOfRangeError:
        break
      writer.add_summary(metrics, step)
      writer.flush()

    # TODO(indutny): validation
    print('Completed {} batches'.format(batches))
