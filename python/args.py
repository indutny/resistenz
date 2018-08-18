import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Train resistenz network.')

  # TODO(indutny): add image_size
  parser.add_argument('--name', type=str, default='default')
  parser.add_argument('--epochs', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--momentum', type=float, default=0.9)

  args = parser.parse_args()

  tag = '{}_bs{}_lr{}_m{}'.format(args.name, args.batch_size, args.lr,
      args.momentum)

  return args, tag
