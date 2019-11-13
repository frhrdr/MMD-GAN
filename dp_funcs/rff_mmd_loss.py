import numpy as np
import tensorflow as tf
from collections import namedtuple


# ABANDONED BECAUSE LOSS CANNOT BE NEATLY SEPARATED ACROSS DATAPOINTS

class RFFKMap:
  def __init__(self, rff_sigma, rff_dims, enc_dims, const_noise, gen_loss='data'):
    # initialize omega and b in numpy following the abcdp code by wittawat
    # set up tensorflow equivalents and initialize them with the numpy code
    #
    assert rff_sigma > 0, 'sigma2 must be positive. Was {}'.format(rff_sigma)
    assert rff_dims > 0 and rff_dims % 2 == 0, 'rff_dims must be even positive int. Was {}'.format(rff_dims)
    half_dims = rff_dims // 2
    assert gen_loss in {'rff', 'mog', 'data'}, 'gen loss must be in {rff, mog, data}'
    self.gen_loss = gen_loss
    self.const_noise = const_noise

    if self.const_noise:
      # with NumpySeedContext(seed=self.seed):
      self.tf_w = tf.constant(np.random.randn(enc_dims, half_dims) / np.sqrt(tf.cast(rff_sigma * 2.0**0.5, tf.float32)))
    else:
      # num = tf.random_normal(shape=(enc_dims, rff_dims // 2))
      # den = tf.sqrt(tf.cast(rff_sigma * 2.0**0.5, tf.float32))
      # print('num', num.dtype, 'den', den.dtype)
      # self.tf_w = num / den
      self.tf_w = tf.random_normal(shape=(enc_dims, half_dims)) / tf.sqrt(tf.cast(rff_sigma * 2.0**0.5, tf.float32))

  def gen_features(self, encoding):
    # The following block of code is deterministic given seed.
    # Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html

    print(encoding.dtype, self.tf_w.dtype)
    enc_w = tf.matmul(encoding, self.tf_w)  # (bs, d_enc) (d_enc, rff) -> (bs, rff)
    enc_z1 = tf.math.cos(enc_w)  # (bs, rff)
    enc_z2 = tf.math.sin(enc_w)  # (bs, rff)
    enc_rff = tf.concat((enc_z1, enc_z2), axis=1)
    return enc_rff


class NumpySeedContext(object):
  """
  A context manager to reset the random seed by numpy.random.seed(..).
  Set the seed back at the end of the block.
  """

  def __init__(self, seed):
    self.seed = seed

  def __enter__(self):
    rstate = np.random.get_state()
    self.cur_state = rstate
    np.random.seed(self.seed)
    return self

  def __exit__(self, *args):
    np.random.set_state(self.cur_state)


rff_specs = namedtuple('rff_specs', ['sigma', 'dims', 'const_noise', 'gen_loss'])
