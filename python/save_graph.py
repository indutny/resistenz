import os
import tensorflow as tf

from model import Model, IMAGE_SIZE
from args import parse_args
from utils import normalize_image

args, tag = parse_args('freeze')
print('Running with a tag "{}"'.format(tag))

model = Model(args)

with tf.Session() as sess:
  image = tf.placeholder(name='image',
      shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3),
      dtype=tf.float32)
  image = normalize_image(image)
  prediction = model.forward(image, coreml=True)

  saver = tf.train.Saver()
  saver.restore(sess, args.save)
  saver.save(sess, os.path.join(args.out, 'graph.ckpt'))

  tf.train.write_graph(sess.graph, args.out, 'graph.pbtxt', as_text=True)
