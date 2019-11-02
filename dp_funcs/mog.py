import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
from dp_funcs.net_picker import NetPicker
from scipy.stats import multivariate_normal
import os


class MoG:
  def __init__(self, n_dims, n_comp, max_iter, linked_gan, enc_batch_size=None, n_data_samples=None,
               filename=None, cov_type='full', fix_cov=False, fix_pi=False, re_init_at_step=None):
    self.d_enc = n_dims
    self.n_comp = n_comp
    self.cov_type = cov_type

    self.pi = None
    self.mu = None
    self.sigma = None

    self.data_filename = filename
    self.linked_gan = linked_gan
    self.encoding = None
    self.batch_encoding = None

    self.mog_init_type = 'random'
    self.mog_init_tries = 1
    self.max_iter = max_iter
    self.print_convergence_warning = False
    self.warm_start = True
    self.enc_batch_size = enc_batch_size
    self.n_data_samples = n_data_samples
    self.scikit_mog = None
    self.init_scikit_mog()

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

    if not self.print_convergence_warning:
      warnings.filterwarnings("ignore", category=ConvergenceWarning)

  def init_scikit_mog(self):
    print('(re)-initializing MoG')
    self.scikit_mog = GaussianMixture(n_components=self.n_comp,
                                      covariance_type=self.cov_type,
                                      max_iter=self.max_iter,
                                      init_params=self.mog_init_type,
                                      n_init=self.mog_init_type,
                                      warm_start=self.warm_start)  # may be worth considering

  def define_tfp_mog_vars(self, do_summary):
    self.pi = tf.compat.v1.get_variable('mog_pi', dtype=tf.float32,
                                        initializer=tf.ones((self.n_comp,)) / self.n_comp)
    print('-------made a pi variable:', self.pi)
    self.mu = tf.compat.v1.get_variable('mog_mu', dtype=tf.float32,
                                        initializer=tf.random.normal((self.n_comp, self.d_enc)))

    if self.cov_type == 'full':
      sig_init = tf.eye(self.d_enc, batch_shape=(self.n_comp,))
    elif self.cov_type == 'diag':
      sig_init = tf.ones((self.n_comp, self.d_enc))
    elif self.cov_type == 'spherical':
      sig_init = tf.ones((self.n_comp,))
    else:
      raise ValueError

    self.sigma = tf.compat.v1.get_variable('mog_sigma', dtype=tf.float32, initializer=sig_init)

    tfp_cat = tfp.distributions.Categorical(probs=self.pi)

    if self.cov_type == 'full':
      tfp_nrm = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=self.sigma,
                                                                   allow_nan_stats=False)
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.d_enc, self.d_enc))
    elif self.cov_type == 'diag':
      tfp_nrm = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma, allow_nan_stats=False)
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.d_enc))
    elif self.cov_type == 'spherical':
      tfp_nrm = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma * tf.eye(self.d_enc),
                                                         allow_nan_stats=False)  # TODO eye not correct
      self.sigma_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp,))
    else:
      raise ValueError

    self.tfp_mog = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp_cat, components_distribution=tfp_nrm)

    self.pi_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp,))
    self.mu_ph = tf.compat.v1.placeholder(tf.float32, shape=(self.n_comp, self.d_enc))

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
    if isinstance(update_flag, tuple) or isinstance(update_flag, list):
      update_freq = update_flag[1]
      assert update_freq > 0  # should active be less than 50% of steps
      return global_step_value % update_freq == 0
    elif isinstance(update_flag, NetPicker):
      return update_flag.do_mog_update()
    else:
      raise ValueError

  def update(self, session):
    # - collect all data encodings
    # need a second data iterator
    # need an encoder op to get embeddings
    # need to store embeddings somewher (probs can do this in memory if they're low-dimensional, might need to save 'em)
    new_encodings = self.collect_encodings(session)

    # - update MoG parameters
    # compute params from embeddings
    # need existing MoG model params to update
    self.fit(new_encodings, session)

    # - proceed with training, sampling from updated MoG
    # this should not query the dataset at all, ignoring the next-batch op. does it do that?

  def set_batch_encoding(self):
    if max(0, 1) == 1:
      self.batch_encoding = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
    else:
      k = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
      k['x'] = tf.Print(k['x'], [tf.norm(k['x']), tf.reduce_mean(k['x']), tf.reduce_max(k['x'])], message='x_enc')
      self.batch_encoding = k

    # because it convenient happens at the same time:
    self.means_summary_op = None
    self.encoding = None

  def update_by_batch(self, session):
    encodings_mat = session.run(self.batch_encoding)['x']
    self.last_batch = encodings_mat
    self.fit(encodings_mat, session)

  def collect_encodings(self, session):
    # assert self.data_loader
    # assert self.encode_op
    assert self.enc_batch_size
    assert self.n_data_samples

    assert self.n_data_samples % self.enc_batch_size == 0
    n_steps = self.n_data_samples // self.enc_batch_size
    encoding_mats = []

    if self.encoding is None:
      # print('init encoding')
      self.encoding = self.linked_gan.Dis(self.linked_gan.get_data_batch(self.data_filename, self.enc_batch_size))
    # print('colllecting encodings')
    for step in range(n_steps):
      encoding_mat = session.run(self.encoding)['x']
      encoding_mats.append(encoding_mat)

    encoding_mat = np.concatenate(encoding_mats)
    # print('encoded data shape:', encoding_mat.shape)
    return encoding_mat

  def store_encodings_and_params(self, session, save_dir, train_step):
    # retrieve encodings and MoG parameters and save them in a numpy file
    encodings_mat = self.collect_encodings(session)
    save_dict = {'enc': encodings_mat,
                 'batch': self.last_batch,
                 'pi': self.scikit_mog.weights_,
                 'mu': self.scikit_mog.means_,
                 'sig': self.scikit_mog.covariances_}
    save_path = os.path.join(save_dir, 'encodings_step_{}.npz'.format(train_step))
    np.savez(save_path, **save_dict)

  def fit(self, encodings, session):

    # encodings = np.random.normal(size=(64, 16))  # debug
    # if hasattr(self.scikit_mog, 'weights_'):
    #   self.scikit_mog.weights_ = np.ones_like(self.scikit_mog.weights_) / self.n_clusters
    self.scikit_mog.fit(encodings)

    if self.fix_cov:
      fix_cov = np.stack([np.eye(self.d_enc)] * self.n_comp) if self.cov_type == 'full' else np.ones(self.n_comp,
                                                                                                     self.d_enc)
      cov_scale = 1.
      self.scikit_mog.covariances_ = fix_cov * cov_scale

    if self.fix_pi:
      self.scikit_mog.weights_ = np.ones(self.n_comp) / self.n_comp

    if self.means_summary_op is None:
      if self.starting_means is None:
        self.starting_means = np.copy(self.scikit_mog.means_)
        # print('\033[1;31m', self.starting_means, '\033[1;34m')

      mu_dist_to_start = tf.norm(self.mu - self.starting_means, axis=1)
      self.means_summary_op = tf.compat.v1.summary.scalar('MoG/mu/mean_dist_to_start', tf.reduce_mean(mu_dist_to_start))
    # if self.pi is None:  # this must be done elsewhere in the linked sngan
    #   print('setting up tfp mog vars')
    #   self.define_tfp_mog_vars(do_summary=False)

    feed_dict = {self.pi_ph: self.scikit_mog.weights_,
                 self.mu_ph: self.scikit_mog.means_,
                 self.sigma_ph: self.scikit_mog.covariances_}

    # feed_dict = {self.pi_ph: np.ones(1),  # debug
    #              self.mu_ph: np.zeros((1, 16)),
    #              self.sigma_ph: np.eye(16).reshape(1, 16, 16) * 10}

    session.run(self.param_update_op, feed_dict=feed_dict)

    # DEBUG
    # mu_val = session.run(self.mu)
    # print('== mu comp:', np.linalg.norm(mu_val), np.linalg.norm(self.scikit_mog.means_))

  def sample_batch(self, batch_size):
    return self.tfp_mog.sample(batch_size)

  # def test_mog_approx(self, session, n_samples=500):
  #   # get encodings
  #   print('---------------------- starting test mog approx')
  #   new_encodings = self.collect_encodings(session)
  #   print('---------------------- collected encodings')
  #   # fit mog
  #   self.fit(new_encodings, session)
  #   print('---------------------- fit mog')
  #
  #   # sample both data and aproximate encodings
  #   x_data_sample = new_encodings[np.random.choice(new_encodings.shape[0], n_samples, replace=False), :]
  #   x_mog_sample = self.scikit_mog.sample(n_samples)[0]
  #   print('data sample shapes should match:', x_data_sample.shape, x_mog_sample.shape)
  #
  #   # sample generator encodings (how?)
  #   if self.loss_gen is None:
  #     print('---------------------- init for gen and loss functions')
  #     code_batch = self.linked_gan.sample_codes(batch_size=n_samples, name='code_tr')
  #     gen_batch = self.linked_gan.Gen(code_batch, is_training=False)
  #     s_gen = self.linked_gan.Dis(gen_batch, is_training=True)['x']
  #     gan_loss = GANLoss(do_summary=False)
  #     self.s_x_ph = tf.placeholder(tf.float32, shape=(n_samples, self.d_enc))
  #     assert self.linked_gan.loss_type in {'rep', 'rmb'}
  #     self.loss_gen, self.loss_dis = gan_loss.apply(s_gen, self.s_x_ph, self.linked_gan.loss_type,
  #                                                   batch_size=n_samples, d=self.linked_gan.score_size,
  #                                                   rep_weights=self.linked_gan.rep_weights)
  #
  #   print('---------------------- computing losses')
  #   l_gen_data, l_dis_data = session.run([self.loss_gen, self.loss_dis], feed_dict={self.s_x_ph: x_data_sample})
  #   l_gen_mog, l_dis_mog = session.run([self.loss_gen, self.loss_dis], feed_dict={self.s_x_ph: x_mog_sample})
  #   print('-> Losses: True Data Dis {} \t Gen {} \t- MoG Approx Dis {} \t Gen {} \n'.format(l_dis_data, l_gen_data,
  #                                                                                           l_dis_mog, l_gen_mog))
  #   self.loss_list.append((l_dis_data, l_gen_data, l_dis_mog, l_gen_mog))
  #   print('---------------------- test mog approx done')
  #
  # def save_loss_list(self, save_file):
  #   loss_mat = np.asarray(self.loss_list)
  #   np.save(save_file, loss_mat)


class NumpyMAPMoG:
  def __init__(self, n_comp, d_enc):
    self.d_enc = d_enc
    self.n_comp = n_comp

    self.weights_ = None
    self.means_ = None
    self.covariances_ = None

    self.dir_a = np.ones((self.n_comp,)) * 2
    self.niw_k = 1.
    self.niw_v = self.d_enc + 2
    self.niw_s = np.eye(self.d_enc) * 0.1

  def _init_params(self):
    pass

  def fit(self, encodings):
    if self.weights_ is None:
      self._init_params()

    n_data = encodings.shape[0]

    n_k, resp = self.e_step(encodings)

    pi_mle, mu_mle, sig_mle = self.m_step_mle(n_k, n_data, encodings, resp)
    pi_map, mu_map, sig_map = self.map_from_mle(pi_mle, mu_mle, sig_mle, n_data, n_k)
    self.weights_, self.means_, self.covariances_ = pi_map, mu_map, sig_map

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
    sig_mle = (resp_n_k[:, :, None] * np.transpose(x_centered, (0, 2, 1))) @ x_centered  # -> (n_c, d,d)
    return pi_mle, mu_mle, sig_mle

  def map_from_mle(self, pi_mle, mu_mle, sig_mle, n_data, n_k):
    # following the dp-em appendix 1
    pi_map = (n_data * pi_mle + self.dir_a - 1) / (n_data + np.sum(self.dir_a) - self.n_comp)  # (n_c)

    mu_map = (n_k[:, None] * mu_mle) / (n_k + self.niw_k)[:, None]  # (n_c) (n_c, d) / (n_c) -> (n_c, d)
    sig_map_t1 = self.niw_s + n_k[:, None, None] * sig_mle  # (n_c, d,d)
    sig_map_t2 = (self.niw_k * n_k / (self.niw_k + n_k))[:, None, None] * (mu_mle[:, :, None] @ mu_mle[:, None, :])
    sig_map = (sig_map_t1 + sig_map_t2) / (self.niw_v + n_k + self.d_enc + 2)
    return pi_map, mu_map, sig_map

  def sample(self, n_samples):
    pass
