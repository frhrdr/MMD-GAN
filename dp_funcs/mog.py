import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from dp_funcs.net_picker import NetPicker
from GeneralTools.math_funcs.gan_losses import GANLoss


class MoG:
  def __init__(self, n_dims, n_clusters, linked_gan, enc_batch_size=None, n_data_samples=None,
               filename=None, cov_type='full'):
    self.d_enc = n_dims
    self.n_clusters = n_clusters
    self.cov_type = cov_type

    self.pi = None
    self.mu = None
    self.sigma = None

    self.data_filename = filename
    self.linked_gan = linked_gan
    self.encoding = None
    self.batch_encoding = None

    self.max_iter = 100
    self.print_convergence_warning = True
    self.warm_start = False
    self.enc_batch_size = enc_batch_size
    self.n_data_samples = n_data_samples
    self.scikit_mog = GaussianMixture(n_components=n_clusters,
                                      covariance_type=cov_type,
                                      max_iter=self.max_iter,
                                      n_init=3,
                                      warm_start=self.warm_start)  # may be worth considering
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

  def define_tfp_mog_vars(self):
    self.pi = tf.compat.v1.get_variable('mog_pi', dtype=tf.float32,
                                        initializer=tf.ones((self.n_clusters,)) / self.n_clusters)
    print('-------made a pi variable:', self.pi)
    self.mu = tf.compat.v1.get_variable('mog_mu', dtype=tf.float32,
                                        initializer=tf.random.normal((self.n_clusters, self.d_enc)))

    if self.cov_type == 'full':
      sig_init = tf.eye(self.d_enc, batch_shape=(self.n_clusters,))
    elif self.cov_type == 'diag':
      sig_init = tf.ones((self.n_clusters, self.d_enc))
    else:
      raise ValueError

    self.sigma = tf.compat.v1.get_variable('mog_sigma', dtype=tf.float32,
                                           initializer=sig_init)

    tfp_cat = tfp.distributions.Categorical(probs=self.pi)

    if self.cov_type == 'full':
      tfp_nrm = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=self.sigma)
      self.sigma_ph = tf.placeholder(tf.float32, shape=(self.n_clusters, self.d_enc, self.d_enc))
    elif self.cov_type == 'diag':
      tfp_nrm = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)
      self.sigma_ph = tf.placeholder(tf.float32, shape=(self.n_clusters, self.d_enc))
    else:
      raise ValueError

    self.tfp_mog = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp_cat, components_distribution=tfp_nrm)

    self.pi_ph = tf.placeholder(tf.float32, shape=(self.n_clusters,))
    self.mu_ph = tf.placeholder(tf.float32, shape=(self.n_clusters, self.d_enc))

    self.param_update_op = tf.group(tf.assign(self.pi, self.pi_ph),
                                    tf.assign(self.mu, self.mu_ph),
                                    tf.assign(self.sigma, self.sigma_ph))

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

  def update_by_batch(self, session):
    if self.batch_encoding is None:
      if max(0, 1) == 2:
        self.batch_encoding = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
      else:
        k = self.linked_gan.Dis(self.linked_gan.data_batch, is_training=False)
        k['x'] = tf.Print(k['x'], [tf.norm(k['x']), tf.reduce_mean(k['x']), tf.reduce_max(k['x'])], message='x_enc')
        self.batch_encoding = k
    encodings_mat = session.run(self.batch_encoding)['x']

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
      print('init encoding')
      self.encoding = self.linked_gan.Dis(self.linked_gan.get_data_batch(self.data_filename, self.enc_batch_size))
    print('colllecting encodings')
    for step in range(n_steps):
      encoding_mat = session.run(self.encoding)['x']
      encoding_mats.append(encoding_mat)

    encoding_mat = np.concatenate(encoding_mats)
    print('encoded data shape:', encoding_mat.shape)
    return encoding_mat

  def fit(self, encodings, session):
    # print('fitting mog')
    try:
      self.scikit_mog.fit(encodings)
    except ConvergenceWarning as cw:
      if self.print_convergence_warning:
        print(cw)

    if self.pi is None:
      print('setting up tfp mog vars')
      self.define_tfp_mog_vars()

    # print('mog pi:', self.scikit_mog.weights_)
    # print('mog mu_0', self.scikit_mog.means_[0, :])
    feed_dict = {self.pi_ph: self.scikit_mog.weights_,
                 self.mu_ph: self.scikit_mog.means_,
                 self.sigma_ph: self.scikit_mog.covariances_}
    # print('updating tfp mog params')
    session.run(self.param_update_op, feed_dict=feed_dict)

    # DEBUG
    mu_val = session.run(self.mu)
    print('== mu comp:', np.norm(mu_val), np.norm(self.scikit_mog.means_))

  def sample_batch(self, batch_size):
    return self.tfp_mog.sample(batch_size)

  def test_mog_approx(self, session, n_samples=500):
    # get encodings
    print('---------------------- starting test mog approx')
    new_encodings = self.collect_encodings(session)
    print('---------------------- collected encodings')
    # fit mog
    self.fit(new_encodings, session)
    print('---------------------- fit mog')

    # sample both data and aproximate encodings
    x_data_sample = new_encodings[np.random.choice(new_encodings.shape[0], n_samples, replace=False), :]
    x_mog_sample = self.scikit_mog.sample(n_samples)[0]
    print('data sample shapes should match:', x_data_sample.shape, x_mog_sample.shape)

    # sample generator encodings (how?)
    if self.loss_gen is None:
      print('---------------------- init for gen and loss functions')
      code_batch = self.linked_gan.sample_codes(batch_size=n_samples, name='code_tr')
      gen_batch = self.linked_gan.Gen(code_batch, is_training=False)
      s_gen = self.linked_gan.Dis(gen_batch, is_training=True)['x']
      gan_loss = GANLoss(do_summary=False)
      self.s_x_ph = tf.placeholder(tf.float32, shape=(n_samples, self.d_enc))
      assert self.linked_gan.loss_type in {'rep', 'rmb'}
      self.loss_gen, self.loss_dis = gan_loss.apply(s_gen, self.s_x_ph, self.linked_gan.loss_type, batch_size=n_samples,
                                                    d=self.linked_gan.score_size,
                                                    rep_weights=self.linked_gan.rep_weights)

    print('---------------------- computing losses')
    l_gen_data, l_dis_data = session.run([self.loss_gen, self.loss_dis], feed_dict={self.s_x_ph: x_data_sample})
    l_gen_mog, l_dis_mog = session.run([self.loss_gen, self.loss_dis], feed_dict={self.s_x_ph: x_mog_sample})
    print('-> Losses: True Data Dis {} \t Gen {} \t- MoG Approx Dis {} \t Gen {} \n'.format(l_dis_data, l_gen_data,
                                                                                            l_dis_mog, l_gen_mog))
    self.loss_list.append((l_dis_data, l_gen_data, l_dis_mog, l_gen_mog))
    print('---------------------- test mog approx done')

  def save_loss_list(self, save_file):
    loss_mat = np.asarray(self.loss_list)
    np.save(save_file, loss_mat)
