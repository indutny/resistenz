import tensorflow as tf

from dataset import Dataset
from model import Model, IMAGE_SIZE, GRID_SIZE

NUM_EPOCHS = 100000
LR = 0.01

model = Model()

optimizer = tf.train.MomentumOptimizer(LR, 0.9)

global_step = tf.Variable(name='global_step', initial_value=0, dtype=tf.int32)

with tf.Session() as sess:
  dataset = Dataset(image_size=IMAGE_SIZE, grid_size=GRID_SIZE)

  training, validation = dataset.load()

  training = training.batch(32)
  validation = validation.batch(32)

  training_iter = training.make_initializable_iterator()
  validation_iter = validation.make_initializable_iterator()

  training_batch = training_iter.get_next()
  validation_batch = validation_iter.get_next()

  training_pred = model.forward(training_batch[0])
  validation_pred = model.forward(validation_batch[0])

  training_loss = model.loss(training_pred, training_batch[1])
  validation_loss = model.loss(validation_pred, validation_batch[1])

  minimize = optimizer.minimize(training_loss, global_step)

  sess.run(tf.global_variables_initializer())
  sess.graph.finalize()

  for i in range(0, NUM_EPOCHS):
    print('Epoch {}'.format(i))

    sess.run([ training_iter.initializer, validation_iter.initializer ])
    while True:
      try:
        _, loss = sess.run([ minimize, training_loss ])
        print(loss)
      except tf.errors.OutOfRangeError:
        break
