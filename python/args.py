import argparse

def parse_args(kind='train'):
  parser = argparse.ArgumentParser(description='Train resistenz network.')

  if kind == 'test':
    parser.add_argument('save', type=str)
    parser.add_argument('image', type=str)

  # TODO(indutny): add image_size
  parser.add_argument('--name', type=str, default='default')
  parser.add_argument('--epochs', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--iou_threshold', type=float, default=0.5)
  parser.add_argument('--grid_depth', type=int, default=5)

  parser.add_argument('--lambda_angle', type=float, default=0.1)
  parser.add_argument('--lambda_obj', type=float, default=1.0)
  parser.add_argument('--lambda_no_obj', type=float, default=2.5)
  parser.add_argument('--lambda_coord', type=float, default=5.0)

  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--lr_fast', type=float, default=0.01)
  parser.add_argument('--lr_fast_epoch', type=int, default=100)
  parser.add_argument('--lr_slow', type=float, default=0.001)
  parser.add_argument('--lr_slow_epoch', type=int, default=5000)

  parser.add_argument('--save_every', type=int, default=100)

  args = parser.parse_args()

  tag = args.name
  if args.batch_size != 64:
    tag += '_bs{}'.format(args.batch_size)
  if args.weight_decay != 5e-4:
    tag += '_wd{}'.format(args.weight_decay)
  if args.iou_threshold != 0.5:
    tag += '_iou{}'.format(args.iou_threshold)
  if args.grid_depth != 5:
    tag += '_gd{}'.format(args.grid_depth)

  if args.lambda_angle != 0.1:
    tag += '_la{}'.format(args.lambda_angle)
  if args.lambda_obj != 1.0:
    tag += '_lo{}'.format(args.lambda_obj)
  if args.lambda_no_obj != 2.5:
    tag += '_lno{}'.format(args.lambda_no_obj)
  if args.lambda_coord != 5.0:
    tag += '_lc{}'.format(args.lambda_coord)

  if args.lr != 0.001:
    tag += '_lr{}'.format(args.lr)
  if args.momentum != 0.9:
    tag += '_mom{}'.format(args.momentum)
  if args.lr_fast != 0.01:
    tag += '_lrf{}'.format(args.lr_fast)
  if args.lr_fast_epoch != 100:
    tag += '_lrfe{}'.format(args.lr_fast_epoch)
  if args.lr_slow != 0.001:
    tag += '_lrs{}'.format(args.lr_slow)
  if args.lr_slow_epoch != 5000:
    tag += '_lrse{}'.format(args.lr_slow_epoch)

  return args, tag
