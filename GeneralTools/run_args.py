import argparse
from GeneralTools.architectures import get_mnist_architectures, cifar_default


def parse_run_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--filename', '-file', type=str, default=None)
  parser.add_argument('--dataset', '-data', type=str, default='mnist')

  parser.add_argument('--debug-mode', action='store_true', default=False)
  parser.add_argument('--optimizer_dis', '-opt_dis', type=str, default='adam')
  parser.add_argument('--optimizer_gen', '-opt_gen', type=str, default='adam')
  parser.add_argument('--n-instance', type=int, default=None)
  parser.add_argument('--save-per-step', type=int, default=1000)
  parser.add_argument('--batch-size', '-bs', type=int, default=128)
  parser.add_argument('--n-class', type=int, default=None)

  parser.add_argument('--lr-dis', '-lr-dis', type=float, default=5e-4)
  parser.add_argument('--lr-gen', '-lr-gen', type=float, default=2e-4)
  parser.add_argument('--lr-end', '-lr-end', type=float, default=1e-7)

  parser.add_argument('--loss-type', '-loss', type=str, default='rep')
  parser.add_argument('--rep-weight-0', '-rep-w1', type=float, default=0.0)
  parser.add_argument('--rep-weight-1', '-rep-w2', type=float, default=-1.0)
  parser.add_argument('--sample-same-class', action='store_true', default=False)
  parser.add_argument('--imbalanced-update', type=str, default=None)

  parser.add_argument('--query-step', '-qstep', type=int, default=250)
  parser.add_argument('--n-threads', type=int, default=7)
  parser.add_argument('--n-iterations', '-n-it', type=int, default=4)

  parser.add_argument('--architecture-gen-key', '-archg', type=str, default=None)
  parser.add_argument('--architecture-dis-key', '-archd', type=str, default=None)

  # DP
  parser.add_argument('--l2-clip-loss', '-lclip', type=float, default=100.)
  parser.add_argument('--l2-clip-grad', '-gclip', type=str, default='100.')
  parser.add_argument('--l2-clip-mog',  '-mclip', type=float, default=None)

  parser.add_argument('--clip-by-layer-norm', action='store_true', default=False)

  parser.add_argument('--noise-factor-loss',    '-lnoise',   type=float, default=None)
  parser.add_argument('--noise-factor-grad',    '-gnoise',   type=float, default=None)
  parser.add_argument('--noise-factor-mog-pi',  '-pinoise',  type=float, default=None)
  parser.add_argument('--noise-factor-mog-mu',  '-munoise',  type=float, default=None)
  parser.add_argument('--noise-factor-mog-sig', '-signoise', type=float, default=None)

  # parser.add_argument('--num_microbatches', '-micro', type=int, default=None)

  # MOG
  parser.add_argument('--mog-type', '-mog', type=str, default='map_lhs')

  parser.add_argument('--n-comp', '-n-comp', type=int, default=5)
  parser.add_argument('--em-steps', '-em', type=int, default=1)
  parser.add_argument('--cov-type', '-cov', type=str, default='full')
  parser.add_argument('--train-without-mog', action='store_true', default=False)
  parser.add_argument('--re-init-step', type=int, default=None)
  parser.add_argument('--reg-covar', type=float, default=None)

  parser.add_argument('--decay-gamma', '-decay', type=float, default=None)

  parser.add_argument('--compute-fid', '-fid', action='store_true', default=False)
  parser.add_argument('--n-fid-batches', type=int, default=781)  # 781 is max for 50k data and bs=64, 100 for bs 500

  parser.add_argument('--fix-cov', action='store_true', default=False)
  parser.add_argument('--fix-pi', action='store_true', default=False)
  parser.add_argument('--map-em', action='store_true', default=False)
  parser.add_argument('--d-enc', '-d', type=int, default=None)

  parser.add_argument('--seed', '-seed', type=int, default=None)

  # random fourier features
  parser.add_argument('--rff-sigma', type=float, default=None)
  parser.add_argument('--rff-dims', type=int, default=None)
  parser.add_argument('--rff-const-noise', action='store_true', default=False)
  parser.add_argument('--rff-gen-loss', type=str, default='data')  # options: data, mog, rff, (not yet: mog-rff

  parser.add_argument('--store-encodings', action='store_true', default=False)
  parser.add_argument('--stop-snorm-grads', action='store_true', default=False)

  args = parser.parse_args()

  post_parse_processing(args)
  return args


def post_parse_processing(args):
  args.filename = '{}_{}'.format(args.dataset, args.filename)

  assert isinstance(args.l2_norm_clip_grad, str)
  if args.clip_by_layer_norm:
    args.l2_norm_clip_grad = [float(k) for k in args.l2_norm_clip_grad.split(',')]
  else:
    args.l2_norm_clip_grad = float(args.l2_norm_clip_grad)


def dataset_defaults(dataset, d_enc, gen_key, dis_key):
  assert dataset in ['mnist', 'cifar', 'fashion']

  def clean_key(key):
    key = 'default' if key is None else key
    key = 'dim' if key == 'diminuitive' else key
    assert key in {'default', 'lean', 'small', 'tiny', 'dim', 'min'}
    return key

  if dataset in ['mnist', 'fashion']:
    num_instance = 50000
    d_enc = 2 if d_enc is None else d_enc
    architecture, code_dim, act_k, d_enc = get_mnist_architectures(clean_key(gen_key), clean_key(dis_key), d_enc)

  elif dataset == 'cifar':
    num_instance = 50000
    d_enc = 16 if d_enc is None else d_enc
    architecture, code_dim, act_k, d_enc = cifar_default(d_enc)
  else:
    raise ValueError
  return num_instance, architecture, code_dim, act_k, d_enc
