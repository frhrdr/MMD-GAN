import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.mixture import GaussianMixture


def plot_encodings_2d(step, empirical_mog_clusters, dataset, setup):

  base_path = 'Results/{}_{}_log/sngan_rep_5e-04_2e-04_k1.68_0.0_-1.0/encodings_step_{}.npz'.format(dataset, setup,
                                                                                                    step)
  mats = np.load(base_path)
  plt.figure()
  plt.scatter(mats['enc'][:, 0], mats['enc'][:, 1], c='xkcd:sea blue', s=3)
  plt.scatter(mats['batch'][:, 0], mats['batch'][:, 1], c='xkcd:aqua blue', s=5)
  # plt.xlim(-2e-4, 2e-4)
  # plt.ylim(-2e-4, 2e-4)
  # plt.xlim(5.09e-6, 5.105e-6)
  # plt.ylim(-8.395e-6, -8.38e-6)
  # learned ellipse(s)
  for idx in range(mats['mu'].shape[0]):
    confidence_ellipse(mats['mu'][idx, :], mats['sig'][idx, :, :], plt.gca(), n_std=2., edgecolor='red')
    plt.scatter(mats['mu'][idx, 0], mats['mu'][idx, 1], c='red', s=5)
    plt.scatter(mats['mu'][idx, 0], mats['mu'][idx, 1], c='red', s=100, alpha=mats['pi'][idx])
    print('component', idx)
    print('pi', mats['pi'][idx], 'mu', mats['mu'][idx, :])
    print('sig', mats['sig'][idx, :, :])

  # empirical data ellipse:
  if empirical_mog_clusters == 1:
    mean = np.mean(mats['enc'], axis=0)[None, :]
    cov = np.cov(mats['enc'][:, 0], mats['enc'][:, 1])[None, :, :]
    weight = 1.
  else:
    mog = GaussianMixture(n_components=empirical_mog_clusters, n_init=3)
    mog.fit(mats['enc'])
    weight = mog.weights_
    mean = mog.means_
    cov = mog.covariances_

  for idx in range(mean.shape[0]):
    confidence_ellipse(mean[idx, :], cov[idx, :, :], plt.gca(), n_std=2., edgecolor='yellow')
    plt.scatter(mean[idx, 0], mean[idx, 1], c='yellow', s=5)
    plt.scatter(mean[idx, 0], mean[idx, 1], c='yellow', s=100, alpha=weight[idx])

  plt.xlabel('D_enc1')
  plt.ylabel('D_enc2')
  plt.title('{}_{}_encodings_step_{}'.format(dataset, setup, step))
  # plt.show()
  plt.savefig('Plots/{}/{}_encodings_step_{}.png'.format(dataset, setup, step))


def confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
  pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
  # Using a special case to obtain the eigenvalues of this
  # two-dimensionl dataset.
  ell_radius_x = np.sqrt(1 + pearson)
  ell_radius_y = np.sqrt(1 - pearson)
  ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

  # Calculating the stdandard deviation of x from the squareroot of the variance and multiplying
  # with the given number of standard deviations.
  scale_x = np.sqrt(cov[0, 0]) * n_std
  scale_y = np.sqrt(cov[1, 1]) * n_std
  transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])
  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)


def plot_mog_ellipses(pi, mu, sig, color, label, verbose=False):
  if verbose:
    print('MoG plot', label)
    print('Pi:', pi)
  for idx in range(mu.shape[0]):
    confidence_ellipse(mu[idx, :], sig[idx, :, :], plt.gca(), n_std=2., edgecolor=color)
    plt.scatter(mu[idx, 0], mu[idx, 1], c=color, s=5, label=label)
    plt.scatter(mu[idx, 0], mu[idx, 1], c=color, s=100, alpha=pi[idx])
    label = None
    # if verbose:
    #   print('component', idx)
    #   print('pi', pi[idx], 'mu', mu[idx, :])
    #   print('sig', sig[idx, :, :])

def first_loss_approx_plots():
  mat = np.load('cifar_logging4_losses.npy')  # columns: True Data 0:Dis 1:Gen - MoG Approx 2:Dis 3:Gen

  plt.figure()
  plt.scatter(mat[:, 0], mat[:, 2])
  plt.plot([-0.6, -0.2], [-0.6, -0.2])
  plt.xlabel('True Data Discriminator Loss')
  plt.ylabel('MoG Data Discriminator Loss')

  plt.savefig('loss_comp_dis.png')

  plt.figure()
  plt.scatter(mat[:, 1], mat[:, 3])
  plt.plot([0., 0.25], [0, 0.25])
  plt.xlabel('True Data Generator Loss')
  plt.ylabel('MoG Data Generator Loss')

  plt.savefig('loss_comp_gen.png')

  plt.figure()
  plt.plot(mat[:, 0], label='True Dis')
  plt.plot(mat[:, 1], label='True Gen')
  plt.plot(mat[:, 2], label='MoG Dis')
  plt.plot(mat[:, 3], label='MoG Gen')
  plt.xlabel('log step')
  plt.ylabel('Loss')
  plt.legend()

  plt.savefig('loss_by_step.png')


if __name__ == '__main__':
  # steps = [0, 499, 1999, 2499, 3999]
  steps = [0, 49, 199, 249, 499, 1999, 4999]
  empirical_mog_clusters = 5
  dataset = 'fashion'
  setup = 'skfi_kmeans_c5_ri200_bs500'
  for step in steps:
    plot_encodings_2d(step, empirical_mog_clusters, dataset, setup)
