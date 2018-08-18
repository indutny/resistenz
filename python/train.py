import tensorflow as tf

from dataset import Dataset

with tf.Session() as sess:
  dataset = Dataset()

  training, validation = dataset.load()

  training = training.batch(32)
  validation = validation.batch(32)

  training_iter = training.make_one_shot_iterator()
  sess.run(training_iter.get_next())
