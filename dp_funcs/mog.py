import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture


class MoG:
  def __init__(self, n_dims, n_clusters, linked_gan, enc_batch_size, n_data_samples, filename):
    self.n_dims = n_dims
    self.n_clusters = n_clusters

    self.pi = tf.get_variable('mog_pi', dtype=tf.float32,
                              initializer=tf.ones((n_clusters,)) / n_clusters)
    self.mu = tf.get_variable('mog_mu', dtype=tf.float32,
                              initializer=tf.random_normal((n_clusters, n_dims)))
    self.sigma = tf.get_variable('mog_sigma', dtype=tf.float32,
                                 initializer=tf.eye(n_dims, batch_shape=(n_clusters,)))

    self.data_filename = filename
    self.linked_gan = linked_gan
    self.encoding = None

    self.enc_batch_size = enc_batch_size
    self.n_data_samples = n_data_samples
    self.scikit_mog = GaussianMixture(n_components=n_clusters,
                                      covariance_type='full',
                                      max_iter=100,
                                      n_init=3,
                                      warm_start=False)  # may be worth considering
    tfp_cat = tfp.distributions.Categorical(probs=self.pi)
    tfp_nrm = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=self.sigma)
    self.tfp_mog = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp_cat,
                                                       components_distribution=tfp_nrm)

    self.pi_ph = tf.placeholder(tf.float32, shape=(n_clusters,))
    self.mu_ph = tf.placeholder(tf.float32, shape=(n_clusters, n_dims))
    self.sigma_ph = tf.placeholder(tf.float32, shape=(n_clusters, n_dims, n_dims))
    self.param_update_op = tf.group(tf.assign(self.pi, self.pi_ph),
                                    tf.assign(self.mu, self.mu_ph),
                                    tf.assign(self.sigma, self.sigma_ph))

  def check_and_update(self, global_step_value, update_freq, session):
    assert update_freq > 0  # should active be less than 50% of steps
    if global_step_value % update_freq != 0:
      return

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

  def collect_encodings(self, session):
    # assert self.data_loader
    # assert self.encode_op
    assert self.enc_batch_size
    assert self.n_data_samples

    assert self.n_data_samples % self.enc_batch_size == 0
    n_steps = self.n_data_samples // self.enc_batch_size
    print('n_steps:', n_steps)
    encoding_mats = []

    if self.encoding is None:
      print('init encoding')
      self.encoding = self.linked_gan.Dis(self.linked_gan.get_data_batch(self.data_filename, self.enc_batch_size))
    print('colllecting encodings')
    for step in range(n_steps):
      encoding_mat = session.run(self.encoding)
      print(encoding_mat.shape)
      encoding_mats.append(encoding_mat)
    print('done')

    encoding_mat = np.concatenate(encoding_mats)
    return encoding_mat

  def fit(self, encodings, session):
    print('fitting mog')
    self.scikit_mog.fit(encodings)
    feed_dict = {self.pi_ph: self.scikit_mog.weights_,
                 self.mu_ph: self.scikit_mog.means_,
                 self.sigma_ph: self.scikit_mog.covariances_}
    print('updating tfp mog params')
    session.run(self.param_update_op, feed_dict=feed_dict)

  def sample_batch(self, batch_size):
    return self.tfp_mog.sample(batch_size)
