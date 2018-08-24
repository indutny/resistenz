import argparse
import tensorflow as tf
import tfcoreml as tf_converter

parser = argparse.ArgumentParser(description='Convert resistenz network.')

parser.add_argument('model')
parser.add_argument('out')

args = parser.parse_args()

with tf.gfile.GFile(args.model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')

tf_converter.convert(tf_model_path=args.model, mlmodel_path=args.out,
    image_input_names=[ 'image:0' ],
    image_scale=1.0/255.0,
    output_feature_names=[ 'resistenz/output/output:0' ])
