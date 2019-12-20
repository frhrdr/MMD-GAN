import numpy as np


def mnist_default(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1', 'out': 64 * 7 * 7, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [64, 7, 7]},
       {'name': 'l2_up',  'out': 32, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up',  'out': 16, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_t28', 'out': 1, 'act': 'tanh'}]

  d = [{'name': 'l1_f28', 'out': 16, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l2_ds',  'out': 32, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l3',     'out': 32, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l4_ds',  'out': 64, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l5',   'out': 64, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [7*7*64]},
       {'name': 'l6_s', 'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]

  code_dim = 32
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def mnist_lean(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1', 'out': 16 * 7 * 7, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [16, 7, 7]},
       {'name': 'l2_up',  'out': 16, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up',  'out':  8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_t28', 'out':  1, 'act': 'tanh'}]

  d = [{'name': 'l1_f28', 'out':  8, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l2_ds',  'out': 16, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l3',     'out': 16, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l4_ds',  'out': 32, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l5',   'out': 32, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [7*7*32]},
       {'name': 'l6_s', 'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]

  code_dim = 16
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def mnist_small(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1', 'out': 8 * 7 * 7, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [8, 7, 7]},
       {'name': 'l2_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_t28', 'out': 1, 'act': 'tanh'}]

  d = [{'name': 'l1_f28', 'out':  4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l2_ds',  'out':  8, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l3',     'out':  8, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l4_ds',  'out': 16, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l5',     'out': 16, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [7*7*16]},
       {'name': 'l6_s',   'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]


  # discriminator weight count:
  # L1: 3x3x1x4, L2: 4x4x4x8, L3: 3x3x8x8, L4: 4x4x8x16, L5: 3x3x16x16, L6: 7*7*16x(d_enc=2)
  # 36 + 512 + 576 + 2048 + 2304 + 1568 = 7044

  code_dim = 8
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def mnist_tiny(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1', 'out': 8 * 7 * 7, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [8, 7, 7]},
       {'name': 'l2_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_t28', 'out': 1, 'act': 'tanh'}]

  d = [{'name': 'l1_f28', 'out': 4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l2_ds',  'out': 4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l3',     'out': 4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l4_ds',  'out': 8, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l5',     'out': 8, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [7*7*8]},
       {'name': 'l6_s',   'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]

  # discriminator weight count:
  # L1: 3x3x1x4, L2: 4x4x4x4, L3: 3x3x4x4, L4: 4x4x4x8, L5: 3x3x8x8, L6: 7*7*8x(d_enc=2)
  # 36 + 256 + 144 + 512 + 576 + 784 = 2308

  code_dim = 8
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def mnist_diminuitive(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1', 'out': 8 * 7 * 7, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [8, 7, 7]},
       {'name': 'l2_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up',  'out': 8, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_t28', 'out': 1, 'act': 'tanh'}]

  d = [{'name': 'l1_ds',  'out': 4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l2',     'out': 4, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 3, 'strides': 1},
       {'name': 'l3_ds',  'out': 4, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2,
        'out_reshape': [7*7*4]},
       {'name': 'l4_s',   'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]

  # discriminator weight count:
  # L1: 4x4x1x4, L2: 3x3x4x4, L3: 4x4x4x4, L4: 7*7*4x(d_enc=2)
  # 64 + 144 + 256 + 392 = 856

  code_dim = 8
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def mnist_minimal(d_enc=2):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  # w_nm = None
  g = [{'name': 'l1',    'out': 28 * 28, 'op': 'd', 'act': 'tanh', 'act_nm': None, 'out_reshape': [1, 28, 28]}]

  d = [{'name': 'l5',   'out': 1, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [1*28*28]},
       {'name': 'l2_s', 'out': d_enc,  'op': 'd', 'act_k': act_k, 'bias': 'b'}]

  code_dim = 8
  architecture = {'input': [(1, 28, 28)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}

  return architecture, code_dim, act_k, d_enc


def cifar_default(d_enc=16):
  act_k = np.power(64.0, 0.125)  # multiplier
  w_nm = 's'  # spectral normalization
  g = [{'name': 'l1',    'out': 512 * 4 * 4, 'op': 'd', 'act': 'linear', 'act_nm': None, 'out_reshape': [512, 4, 4]},
       {'name': 'l2_up', 'out': 256, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l3_up', 'out': 128, 'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l4_up', 'out': 64,  'op': 'tc', 'act': 'relu', 'act_nm': 'bn', 'kernel': 4, 'strides': 2},
       {'name': 'l5_t32', 'out': 3,  'act': 'tanh'}]
  d = [{'name': 'l1_f32', 'out': 64, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
       {'name': 'l2_ds', 'out': 128, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l3',    'out': 128, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
       {'name': 'l4_ds', 'out': 256, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l5',    'out': 256, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm},
       {'name': 'l6_ds', 'out': 512, 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'kernel': 4, 'strides': 2},
       {'name': 'l7',    'out': 512, 'op': 'c', 'act': 'lrelu', 'act_k': act_k, 'w_nm': w_nm, 'out_reshape': [4*4*512]},
       {'name': 'l8_s',  'out': d_enc, 'op': 'd', 'act_k': act_k, 'bias': 'b', 'w_nm': w_nm}]
  code_dim = 128
  architecture = {'input': [(3, 32, 32)],
                  'code': [(code_dim, 'linear')],
                  'generator': g,
                  'discriminator': d}
  return architecture, code_dim, act_k, d_enc
