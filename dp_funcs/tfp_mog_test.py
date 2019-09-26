import tensorflow as tf
import tensorflow_probability as tfp
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

pi = [0.1, 0.4, 0.5]
mu = [[-10., -10.],
      [0., 15.],
      [3., 0.]]
sigma = tf.eye(2, batch_shape=(3,))

tfp_cat = tfp.distributions.Categorical(probs=pi)
tfp_nrm = tfp.distributions.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
tfp_mog = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp_cat,
                                              components_distribution=tfp_nrm)

sess = tf.Session()
sample_mat = sess.run(tfp_mog.sample(500))

print(sample_mat.shape)

plt.scatter(sample_mat[:, 0], sample_mat[:, 1])
plt.savefig('tfp_mog_test_plot.png')
