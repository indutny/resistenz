import tensorflow as tf

from dataset import Dataset
from model import IMAGE_SIZE, GRID_SIZE

with tf.Session() as sess:
  dataset = Dataset(image_size=IMAGE_SIZE, grid_size=GRID_SIZE)

  training, validation = dataset.load()

  training = training.batch(32)
  validation = validation.batch(32)

  training_iter = training.make_one_shot_iterator()
  sess.run(training_iter.get_next())
