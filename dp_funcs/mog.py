import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
from dp_funcs.net_picker import NetPicker
from scipy.stats import multivariate_normal
import os


class EncodingMoG:
  def __init__(self, n_dims, n_comp, linked_gan, np_mog, n_data_samples, enc_batch_size, filename=None, cov_type='full',
               fix_cov=False, fix_pi=False, re_init_at_step=None, decay_gamma=None, map_em=False):
    self.n_dims = n_dims
    self.n_comp = n_comp
    self.cov_type = cov_type

    self.pi = None
    self.mu = None
    self.sigma = None

    self.data_filename = filename
    self.linked_gan = linked_gan
    self.encoding = None
    self.batch_encoding = None

    self.enc_batch_size = enc_batch_size
    self.n_data_samples = n_data_samples
    self.np_mog = default_mogs(np_mog, n_comp, n_dims, cov_type, decay_gamma, map_em) if isinstance(np_mog, str) else np_mog

    self.tfp_mog = None

    self.pi_ph = None
    self.mu_ph = None
    self.sigma_ph = None
    self.param_update_op = None
    self.gen_data_op = None
    # self.gan_loss = None
    self.s_x_ph = None
    self.loss_gen = None
    self.loss_dis = None
    self.loss_list = []

    self.re_init_at_step = re_init_at_step
    self.fix_cov = fix_cov
    self.fix_pi = fix_pi
    self.last_batch = None
    self.starting_means = None
    self.means_summary_op = None
    self.approx_test = False  # approximation quality test. now mostly obsolete

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

  def init_np_mog(self):
    if hasattr(self.np_mog, 'converged_'):
      print('(re)-initializing MoG')
      del self.np_mog.converged_
    else:
      print('MoG not initialized yet')

  def define_tfp_mog_vars(self, do_summary):
    self.pi = tf.compat.v1.get_variable('mog_pi', dtype=tf.float32,
                                        initializer=tf.ones((self.n_comp,)) / self.n_comp)
    print('-------made a pi variable:', self.pi)
    self.mu = tf.compat.v1.get_variable('mog_mu', dtype=tf.float32,
                                        initializer=tf.random.normal((self.n_comp, self.n_dims)))

    if self.cov_type == 'full':
      sig_init = tf.eye(self.n_dims, batch_shape=(self.n_comp,))
    elif self.cov_type == 'diag':
      sig_init = tf.ones((self.n_comp, self.n_dims))
    elif self.cov_type == 'spherical':
      sig_init = tf.ones((self.n_comp,))
    else:
      raise ValueError

    self.sigma = tf.compat.v1.get_variable('mog_sigma', dtype=tf.float32, initializer=sig_init)

    tfp_cat = tfp.distributions.Categorical(probs=self.pi)

    if self.cov_type == 'full':
      tfp_nrm = tfp.distributions.MultivariateNormalFullCovariance(self.mu, self.sigma, allow_nan_stats=False)
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.n_dims, self.n_dims))
    elif self.cov_type == 'diag':
      tfp_nrm = tfp.distributions.MultivariateNormalDiag(self.mu, self.sigma, allow_nan_stats=False)
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.n_dims))
    elif self.cov_type == 'spherical':
      tfp_nrm = tfp.distributions.MultivariateNormalDiag(self.mu, self.sigma * tf.eye(self.n_dims), allow_nan_stats=False)
      # TODO eye not correct
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp,))
    else:
      raise ValueError

    self.tfp_mog = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp_cat, components_distribution=tfp_nrm)

    self.pi_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp,))
    self.mu_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.n_dims))

    self.param_update_op = tf.group(tf.compat.v1.assign(self.pi, self.pi_ph),
                                    tf.compat.v1.assign(self.mu, self.mu_ph),
                                    tf.compat.v1.assign(self.sigma, self.sigma_ph))

    if do_summary:
      with tf.name_scope(None):  # return to root scope to avoid scope overlap
        tf.compat.v1.summary.scalar('MoG/pi/max_val', tf.reduce_max(self.pi))
        tf.compat.v1.summary.scalar('MoG/pi/min_val', tf.reduce_min(self.pi))
        mu_norms = tf.norm(self.mu, axis=1)
        tf.compat.v1.summary.scalar('MoG/mu/max_norm', tf.reduce_max(mu_norms))
        tf.compat.v1.summary.scalar('MoG/mu/mean_norm', tf.reduce_mean(mu_norms))
        tf.compat.v1.summary.scalar('MoG/mu/min_norm', tf.reduce_min(mu_norms))
        sig_diag = tf.linalg.diag_part(self.sigma) if self.cov_type == 'full' else self.sigma
        tf.compat.v1.summary.scalar('MoG/sig/diag_max_val', tf.reduce_max(sig_diag))
        tf.compat.v1.summary.scalar('MoG/sig/diag_mean_val', tf.reduce_mean(sig_diag))
        tf.compat.v1.summary.scalar('MoG/sig/diag_min_val', tf.reduce_min(sig_diag))

  def time_to_update(self, global_step_value, update_flag):
    # only relevant for alternating update schemes
    if isinstance(update_flag, tuple) or isinstance(update_flag, list):
      update_freq = update_flag[1]
      assert update_freq > 0  # should active be less than 50% of steps
      return global_step_value % update_freq == 0
    elif isinstance(update_flag, NetPicker):
      return update_flag.do_mog_update()
    else:
      raise ValueError

  def update(self, session):
    new_encodings = self.collect_encodings(session)
    self.fit(new_encodings, session)

  def set_batch_encoding(self):
    if max(0, 1) == 1:
      self.batch_encoding = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
    else:
      k = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
      k['x'] = tf.Print(k['x'], [tf.norm(k['x']), tf.reduce_mean(k['x']), tf.reduce_max(k['x'])], message='x_enc')
      self.batch_encoding = k

    self.means_summary_op, self.encoding = None, None  # because it convenient happens at the same time:

  def update_by_batch(self, session):
    encodings_mat = session.run(self.batch_encoding)['x']
    self.last_batch = encodings_mat
    self.fit(encodings_mat, session)

  def collect_encodings(self, session):
    assert self.enc_batch_size
    assert self.n_data_samples

    assert self.n_data_samples % self.enc_batch_size == 0
    n_steps = self.n_data_samples // self.enc_batch_size
    encoding_mats = []

    if self.encoding is None:
      self.encoding = self.linked_gan.Dis(self.linked_gan.get_data_batch(self.data_filename, self.enc_batch_size))
    for step in range(n_steps):
      encoding_mat = session.run(self.encoding)['x']
      encoding_mats.append(encoding_mat)

    return np.concatenate(encoding_mats)

  def store_encodings_and_params(self, session, save_dir, train_step):
    # retrieve encodings and MoG parameters and save them in a numpy file
    encodings_mat = self.collect_encodings(session)
    save_dict = {'enc': encodings_mat,
                 'batch': self.last_batch,
                 'pi': self.np_mog.weights_,
                 'mu': self.np_mog.means_,
                 'sig': self.np_mog.covariances_}
    save_path = os.path.join(save_dir, 'encodings_step_{}.npz'.format(train_step))
    np.savez(save_path, **save_dict)

  def fit(self, encodings, session):
    self.np_mog.fit(encodings)

    if self.fix_cov:
      fix_cov = np.stack([np.eye(self.n_dims)] * self.n_comp) if self.cov_type == 'full' else np.ones(self.n_comp,
                                                                                                      self.n_dims)
      cov_scale = 1.
      self.np_mog.covariances_ = fix_cov * cov_scale

    if self.fix_pi:
      self.np_mog.weights_ = np.ones(self.n_comp) / self.n_comp

    if self.means_summary_op is None:
      if self.starting_means is None:
        self.starting_means = np.copy(self.np_mog.means_)
        # print('\033[1;31m', self.starting_means, '\033[1;34m')

      mu_dist_to_start = tf.norm(self.mu - self.starting_means, axis=1)
      self.means_summary_op = tf.compat.v1.summary.scalar('MoG/mu/mean_dist_to_start', tf.reduce_mean(mu_dist_to_start))

    session.run(self.param_update_op, feed_dict={self.pi_ph: self.np_mog.weights_,
                                                 self.mu_ph: self.np_mog.means_,
                                                 self.sigma_ph: self.np_mog.covariances_})

  def sample_batch(self, batch_size):
    return self.tfp_mog.sample(batch_size)


class NumpyMAPMoG:
  def __init__(self, n_comp, d_enc, do_map=True, reg_covar=None, dir_a=None, niw_k=None, niw_v=None, niw_s=None):
    self.d_enc = d_enc
    self.n_comp = n_comp

    self.weights_ = None
    self.means_ = None
    self.covariances_ = None

    self.do_map = do_map
    self.reg_covar = None if reg_covar is None else np.eye(d_enc) * reg_covar

    self.dir_a = dir_a if dir_a is not None else np.ones((self.n_comp,)) * 2
    self.niw_k = niw_k if niw_k is not None else 1.
    self.niw_v = niw_v if niw_v is not None else self.d_enc + 2
    self.niw_s = niw_s if niw_s is not None else np.eye(self.d_enc) * 0.1

  def _init_params(self, encodings):
    # random init only
    self.converged_ = False
    n_data = encodings.shape[0]
    resp = np.random.rand(self.n_comp, n_data)
    resp = resp / resp.sum(axis=0)
    n_k = np.sum(resp, axis=1)  # (n_comp)
    pi_mle, mu_mle, sig_mle = self.m_step_mle(n_k, n_data, encodings, resp)
    pi_map, mu_map, sig_map = self.map_from_mle(pi_mle, mu_mle, sig_mle, n_data, n_k)
    self.weights_, self.means_, self.covariances_ = pi_map, mu_map, sig_map

  def fit(self, encodings):
    if not hasattr(self, 'converged_'):
      self._init_params(encodings)

    n_data = encodings.shape[0]

    n_k, resp = self.e_step(encodings)

    pi, mu, sig = self.m_step_mle(n_k, n_data, encodings, resp)
    if self.do_map:
      pi, mu, sig = self.map_from_mle(pi, mu, sig, n_data, n_k)
    self.weights_, self.means_, self.covariances_ = pi, mu, sig

  def e_step(self, x):
    # following bishop p. 438
    pi, mu, sig = self.weights_, self.means_, self.covariances_

    data_resp = np.stack([pi[k] * multivariate_normal.pdf(x, mean=mu[k, :], cov=sig[k, :, :])
                          for k in range(self.n_comp)])  # (n_comp, n_data)
    data_resp_normed = data_resp / np.sum(data_resp, axis=0)
    n_k = np.sum(data_resp_normed, axis=1)  # (n_comp)
    return n_k, data_resp_normed

  def m_step_mle(self, n_k, n_data, x, resp):
    # following bishop p. 439
    pi_mle = n_k / n_data
    mu_mle = resp @ x / n_k[:, None]  # (n_c, n_d) (n_d, d) / (n_c) -> (n_c, d) / (n_c) -> (n_c,d)
    # centering: (n_d, d) (n_c, d) -> (n_c, n_d, d)
    x_centered = x[None, :, :] - mu_mle[:, None, :]

    resp_n_k = resp / n_k[:, None]
    sig_mle = (resp_n_k[:, None, :] * np.transpose(x_centered, (0, 2, 1))) @ x_centered  # -> (n_c, d,d)
    if self.reg_covar is not None:
      sig_mle = sig_mle + self.reg_covar
    return pi_mle, mu_mle, sig_mle

  def map_from_mle(self, pi_mle, mu_mle, sig_mle, n_data, n_k):
    # following the dp-em appendix 1
    pi_map = (n_data * pi_mle + self.dir_a - 1) / (n_data + np.sum(self.dir_a) - self.n_comp)  # (n_c)

    mu_map = (n_k[:, None] * mu_mle) / (n_k + self.niw_k)[:, None]  # (n_c) (n_c, d) / (n_c) -> (n_c, d)
    sig_map_t1 = self.niw_s + n_k[:, None, None] * sig_mle  # (n_c, d,d)
    sig_map_t2 = (self.niw_k * n_k / (self.niw_k + n_k))[:, None, None] * (mu_mle[:, :, None] @ mu_mle[:, None, :])
    sig_map = (sig_map_t1 + sig_map_t2) / (self.niw_v + n_k + self.d_enc + 2)[:, None, None]
    return pi_map, mu_map, sig_map

  def sample(self, n_samples):
    pass


class NowlanMoG:
  def __init__(self, n_comp, d_enc, decay_gamma, do_map=False, dir_a=None, niw_k=None, niw_v=None, niw_s=None):

    self.d_enc = d_enc  # noted in comments as d
    self.n_comp = n_comp  # noted in comments as k
    # batch size noted in comments as n

    self.weights_ = None
    self.means_ = None
    self.covariances_ = None

    self.n_agg = np.zeros((self.n_comp,))
    self.m_agg = np.zeros((self.n_comp, self.d_enc))
    self.q_agg = np.zeros((self.n_comp, self.d_enc, self.d_enc))
    self.decay_gamma = decay_gamma

    self.do_map = do_map
    self.dir_a = dir_a if dir_a is not None else np.ones((self.n_comp,)) * 2
    self.niw_k = niw_k if niw_k is not None else 1.
    self.niw_v = niw_v if niw_v is not None else self.d_enc + 2
    self.niw_s = niw_s if niw_s is not None else np.eye(self.d_enc) * 0.1

  def fit(self, data):
    n_batch, m_batch, q_batch = self.e_step(data)
    pi, mu, sig = self.m_step_mle(n_batch, m_batch, q_batch)
    if self.do_map:
      pi, mu, sig = self.map_from_mle(pi, mu, sig, data.shape[0], n_batch)
    self.weights_, self.means_, self.covariances_ = pi, mu, sig

  def e_step(self, data):
    # following bishop p. 438 and then Neal and Hinton citing Nowlan 1991
    if not hasattr(self, 'converged_'):  # for init
      self.converged_ = False
      resp = np.random.rand(self.n_comp, data.shape[0])
    else:
      print(self.covariances_)
      resp = np.stack([self.weights_[k] * multivariate_normal.pdf(data, self.means_[k, :], self.covariances_[k, :, :])
                       for k in range(self.n_comp)])  # (n_comp, n_data)
    resp = resp / np.sum(resp, axis=0)

    n_batch = resp  # just resp (k, n)
    m_batch = resp[:, :, None] * data[None, :, :]  # (k, n) * (n, d) -> (k, n, d)
    data_outer = data[:, :, None] * data[:, None, :]  # (n, d) * (n, d) -> (n, d, d)
    q_batch = resp[:, :, None, None] * data_outer[None, :, :, :]  # (k, n) * (n, d, d) -> (k, n, d, d)

    n_batch = np.sum(n_batch, axis=1)  # (k)
    m_batch = np.sum(m_batch, axis=1)  # (k, d)
    q_batch = np.sum(q_batch, axis=1)  # (k, d, d)

    return n_batch, m_batch, q_batch

  def m_step_mle(self, n_batch, m_batch, q_batch):
    # following my derivations extending Nowlan 1991 to arbitrary dimensions and number of components

    # update sufficient statistics
    self.n_agg = self.decay_gamma * self.n_agg + n_batch
    self.m_agg = self.decay_gamma * self.m_agg + m_batch
    self.q_agg = self.decay_gamma * self.q_agg + q_batch

    # infer MoG parameters
    n, m, q = self.n_agg, self.m_agg, self.q_agg
    pi_mle = n / np.sum(n)
    mu_mle = m / n[:, None]
    sig_mle = q / n[:, None, None] - m[:, :, None] * m[:, None, :] / (n**2)[:, None, None]
    return pi_mle, mu_mle, sig_mle

  def map_from_mle(self, pi_mle, mu_mle, sig_mle, n_data, n_k):
    # following the dp-em appendix 1
    pi_map = (n_data * pi_mle + self.dir_a - 1) / (n_data + np.sum(self.dir_a) - self.n_comp)  # (n_c)

    mu_map = (n_k[:, None] * mu_mle) / (n_k + self.niw_k)[:, None]  # (n_c) (n_c, d) / (n_c) -> (n_c, d)
    sig_map_t1 = self.niw_s + n_k[:, None, None] * sig_mle  # (n_c, d,d)
    sig_map_t2 = (self.niw_k * n_k / (self.niw_k + n_k))[:, None, None] * (mu_mle[:, :, None] @ mu_mle[:, None, :])
    sig_map = (sig_map_t1 + sig_map_t2) / (self.niw_v + n_k + self.d_enc + 2)[:, None, None]
    return pi_map, mu_map, sig_map

  def sample(self, n_samples):
    pass


def default_mogs(key, n_comp, d_enc, cov_type, decay_gamma=None, map_em=False):
  if key == 'sklearn':
    assert map_em is False
    mog = GaussianMixture(n_comp, cov_type, max_iter=1, init_params='random', n_init=1, warm_start=True)
  elif key == 'map':
    assert cov_type == 'full'
    mog = NumpyMAPMoG(n_comp, d_enc, do_map=map_em)
  elif key == 'nowlan':
    assert cov_type == 'full'
    assert decay_gamma is not None
    mog = NowlanMoG(n_comp, d_enc, decay_gamma, do_map=map_em)
  else:
    raise ValueError
  return mog
