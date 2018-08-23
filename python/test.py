import os
import tensorflow as tf

from model import Model, IMAGE_SIZE
from args import parse_args
from svg import SVG

IMAGE_DIR = os.path.join('.', 'images')

args, tag = parse_args('test')
print('Running with a tag "{}"'.format(tag))

model = Model(args)

with tf.Session() as sess:
  image = tf.read_file(tf.constant(args.image, dtype=tf.string))
  image = tf.image.decode_jpeg(image, channels=3)

  size = tf.shape(image)[:2]

  crop_size = tf.reduce_min(size, axis=-1, keepdims=True)
  crop_off = tf.cast((size - crop_size) / 2, dtype=tf.int32)

  image = tf.image.crop_to_bounding_box(image, crop_off[0], crop_off[1], \
      crop_size[0], crop_size[0])
  image = tf.image.resize_images(image,
      [ IMAGE_SIZE, IMAGE_SIZE ],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  image = tf.cast(image, dtype=tf.float32)
  image /= 255.0

  prediction = model.forward(tf.expand_dims(image, axis=0))[0]

  svg = SVG(image, prediction).write_file(os.path.join(IMAGE_DIR, 'test.svg'))

  saver = tf.train.Saver(max_to_keep=10, name=tag)
  saver.restore(sess, args.save)

  sess.graph.finalize()

  sess.run(svg)
