import argparse
import GeneralTools.architectures


def parse_run_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--filename', '-file', type=str, default=None)
  parser.add_argument('--dataset', '-data', type=str, default='mnist')

  parser.add_argument('--debug-mode', action='store_true', default=False)
  parser.add_argument('--optimizer', '-opt', type=str, default='adam')
  parser.add_argument('--n-instance', type=int, default=None)
  parser.add_argument('--save-per-step', type=int, default=5000)
  parser.add_argument('--batch-size', '-bs', type=int, default=64)
  parser.add_argument('--n-class', type=int, default=None)

  parser.add_argument('--lr-dis', '-lr-dis', type=float, default=5e-4)
  parser.add_argument('--lr-gen', '-lr-gen', type=float, default=2e-4)
  parser.add_argument('--lr-end', '-lr-end', type=float, default=1e-7)

  parser.add_argument('--loss-type', '-loss', type=str, default='rep')
  parser.add_argument('--rep-weight-0', '-rep-w1', type=float, default=0.0)
  parser.add_argument('--rep-weight-1', '-rep-w2', type=float, default=-1.0)
  parser.add_argument('--sample-same-class', action='store_true', default=False)
  parser.add_argument('--imbalanced-update', type=str, default=None)

  parser.add_argument('--debug-step', '-dstep', type=int, default=400)
  parser.add_argument('--query-step', '-qstep', type=int, default=100)
  parser.add_argument('--n-threads', type=int, default=7)
  parser.add_argument('--n-iterations', '-n-it', type=int, default=8)

  # MOG
  # parser.add_argument('--d-encoding', '-denc', type=int, default=4)
  parser.add_argument('--n-clusters', '-n-clusters', type=int, default=10)
  parser.add_argument('--em-steps', type=int, default=1)
  parser.add_argument('--cov-type', '-cov', type=str, default='full')
  parser.add_argument('--train-without-mog', action='store_true', default=False)

  parser.add_argument('--compute-fid', action='store_true', default=False)

  parser.add_argument('--architecture_key', type=str, default=None)

  parser.add_argument('--seed', '-seed', type=int, default=None)

  args = parser.parse_args()

  post_parse_processing(args)
  return args


def post_parse_processing(args):
  args.filename = '{}_{}'.format(args.dataset, args.filename)


def dataset_defaults(dataset, architecture_key):
  assert dataset in ['mnist', 'cifar', 'fashion']

  if dataset in ['mnist', 'fashion']:
    num_instance = 50000
    if architecture_key == '2d':
      architecture, code_dim, act_k, d_enc = GeneralTools.architectures.mnist_2d_enc()
    elif architecture_key is not None:
      raise ValueError
    else:
      architecture, code_dim, act_k, d_enc = GeneralTools.architectures.mnist_default()
  elif dataset == 'cifar':
    num_instance = 50000
    if architecture_key is not None:
      raise ValueError
    else:
      architecture, code_dim, act_k, d_enc = GeneralTools.architectures.cifar_default()
  else:
    raise ValueError
  return num_instance, architecture, code_dim, act_k, d_enc
