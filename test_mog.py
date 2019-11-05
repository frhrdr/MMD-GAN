import numpy as np
import matplotlib.pyplot as plt
from dp_funcs.mog import NumpyMAPMoG, GaussianMixture, NowlanMoG
from plotting import plot_mog_ellipses
from sklearn.exceptions import ConvergenceWarning
import warnings



def make_sample_data(d=2, n_comp=10, n_samples=3000):
  pi = np.random.rand(n_comp)
  pi = pi / np.sum(pi)
  samples_made = 0
  samples = []
  for k in range(n_comp):
    mu_k = (np.random.randn(d) - 0.5) * 3
    sig_k = np.random.normal(loc=np.zeros((d, d+2)))
    sig_k = sig_k @ sig_k.T
    n_to_sample = int(n_samples * pi[k]) if (k < n_comp - 1) else (n_samples - samples_made)
    samples.append(np.random.multivariate_normal(mean=mu_k, cov=sig_k, size=(n_to_sample,)))
    samples_made += n_to_sample
  samples = np.concatenate(samples)
  np.random.shuffle(samples)
  return samples


def test_sklearn_mog(n_comp=5, max_iter=200, batch_size=50, n_steps=200, n_samples=3000):

  warnings.filterwarnings("ignore", category=ConvergenceWarning)

  data = make_sample_data(n_comp=n_comp, n_samples=n_samples)

  mog_base_sklearn = GaussianMixture(n_components=n_comp, covariance_type='full', max_iter=max_iter,
                                     init_params='random', n_init=3, warm_start=False, tol=1e-7)
  mog_base_sklearn.fit(data)

  d_enc = data.shape[1]
  mog_base_mle = NumpyMAPMoG(n_comp, d_enc, do_map=False)
  mog_base_map = NumpyMAPMoG(n_comp, d_enc, do_map=True)
  for step in range(n_steps):
    mog_base_mle.fit(data)
    mog_base_map.fit(data)

  mog_it_sklearn = GaussianMixture(n_components=n_comp, covariance_type='full', max_iter=1,
                                   init_params='random', n_init=1, warm_start=True)

  # mog_it_mle = NumpyMAPMoG(n_comp, data.shape[1], do_map=False, reg_covar=1e-3)
  mog_it_map = NumpyMAPMoG(n_comp, data.shape[1], do_map=True,
                           dir_a=np.ones((n_comp,)) * 10.,  # higher number -> more even distribution among components
                           niw_k=1.,  # higher number ->
                           niw_v=d_enc + 2,  # higher number -> ?
                           niw_s=np.eye(d_enc) * 10.)  # higher number -> larger diagonal covariance (5 maybe?)

  assert n_samples % n_steps == 0
  idx = 0
  batch = None
  for step in range(n_steps):
    batch = data[idx:idx + batch_size, :]
    mog_it_sklearn.fit(batch)
    # mog_it_mle.fit(batch)
    mog_it_map.fit(batch)
    idx = (idx + batch_size) % n_samples

  plt.figure()
  plt.scatter(data[:, 0], data[:, 1], c='xkcd:aqua blue', s=3, alpha=0.2, label='data')
  plt.scatter(batch[:, 0], batch[:, 1], c='xkcd:sea blue', s=5, label='minibatch')

  def plot_mog(mog, color, label, verbose=False):
    plot_mog_ellipses(mog.weights_, mog.means_, mog.covariances_, color, label, verbose)

  plot_mog(mog_base_sklearn, 'yellow', 'batch mog sklearn')
  # plot_mog(mog_base_map, 'green', 'batch mog map')
  # plot_mog(mog_base_mle, 'red', 'batch mog mle')

  # plot_mog(mog_it_sklearn, 'xkcd:tan', 'it mog map', verbose=True)
  plot_mog(mog_it_map, 'xkcd:olive', 'it mog map', verbose=True)
  # plot_mog(mog_it_mle, 'xkcd:orange', 'it mog mle', verbose=True)
  plt.legend()
  plt.show()


def test_nowlan_mog(n_comp=5, max_iter=200, batch_size=50, n_steps=300, n_samples=3000):

  warnings.filterwarnings("ignore", category=ConvergenceWarning)

  data = make_sample_data(n_comp=n_comp, n_samples=n_samples)

  mog_base_sklearn = GaussianMixture(n_components=n_comp, covariance_type='full', max_iter=max_iter,
                                     init_params='random', n_init=3, warm_start=False, tol=1e-7)
  mog_base_sklearn.fit(data)

  d_enc = data.shape[1]
  gammas = [0.8, 0.90, 0.95]
  do_map = True
  nowlan_mogs = [NowlanMoG(n_comp, d_enc, g, do_map) for g in gammas]

  assert n_samples % n_steps == 0
  idx = 0
  batch = None
  for step in range(n_steps):
    batch = data[idx:idx + batch_size, :]
    idx = (idx + batch_size) % n_samples

    for mog in nowlan_mogs:
      mog.fit(data)

  plt.figure()
  plt.scatter(data[:, 0], data[:, 1], c='xkcd:aqua blue', s=3, alpha=0.2, label='data')
  plt.scatter(batch[:, 0], batch[:, 1], c='xkcd:sea blue', s=5, label='minibatch')

  def plot_mog(mog, color, label, verbose=False):
    plot_mog_ellipses(mog.weights_, mog.means_, mog.covariances_, color, label, verbose)

  # plot_mog(mog_base_sklearn, 'yellow', 'batch mog sklearn')

  # plot_mog(mog_it_sklearn, 'xkcd:tan', 'it mog map', verbose=True)
  colors = ['xkcd:orange', 'xkcd:violet', 'green', 'xkcd:olive']
  for mog, color, gamma in zip(nowlan_mogs, colors, gammas):
    plot_mog(mog, color, 'it mog g={}'.format(gamma), verbose=True)

  plt.legend()
  plt.show()


if __name__ == '__main__':
    # m = make_sample_data()
    # test_sklearn_mog()
    test_nowlan_mog()
    # plt.figure()
    # plt.scatter(m[:, 0], m[:, 1])
    # plt.show()