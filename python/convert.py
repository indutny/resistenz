import argparse
import tensorflow as tf
import coremltools
import tfcoreml as tf_converter

parser = argparse.ArgumentParser(description='Convert resistenz network.')

parser.add_argument('model')
parser.add_argument('out')
parser.add_argument('--fp16', type=str)

args = parser.parse_args()

with tf.gfile.GFile(args.model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')

tf_converter.convert(tf_model_path=args.model, mlmodel_path=args.out,
    image_input_names=[ 'image:0' ],
    image_scale=1.0/255.0,
    output_feature_names=[ 'resistenz/output/output:0' ])

if not args.fp16 is None:
  # Load a model, lower its precision, and then save the smaller model.
  model_spec = coremltools.utils.load_spec(args.out)
  model_fp16_spec = \
      coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
  coremltools.utils.save_spec(model_fp16_spec, args.fp16)
