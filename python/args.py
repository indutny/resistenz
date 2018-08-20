import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Train resistenz network.')

  # TODO(indutny): add image_size
  parser.add_argument('--name', type=str, default='default')
  parser.add_argument('--epochs', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--lr_fast', type=float, default=0.01)
  parser.add_argument('--lr_fast_epoch', type=int, default=100)
  parser.add_argument('--lr_slow', type=float, default=0.001)
  parser.add_argument('--lr_slow_epoch', type=int, default=500)
  parser.add_argument('--save_every', type=int, default=10)

  args = parser.parse_args()

  tag = args.name
  if args.batch_size != 32:
    tag += '_bs{}'.format(args.batch_size)
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
  if args.lr_slow_epoch != 500:
    tag += '_lrse{}'.format(args.lr_slow_epoch)

  return args, tag
