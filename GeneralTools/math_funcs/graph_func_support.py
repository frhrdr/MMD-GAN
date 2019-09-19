import numpy as np

from GeneralTools.misc_fun import FLAGS


def trace_sqrt_product_np(cov1, cov2):
    """ This function calculates trace(sqrt(cov1 * cov2))

    This code is inspired from:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

    :param cov1:
    :param cov2:
    :return:
    """
    sqrt_cov1 = sqrt_sym_mat_np(cov1)
    cov_121 = np.matmul(np.matmul(sqrt_cov1, cov2), sqrt_cov1)

    return np.trace(sqrt_sym_mat_np(cov_121))


def sqrt_sym_mat_np(mat, eps=None):
    """ This function calculates the square root of symmetric matrix

    :param mat:
    :param eps:
    :return:
    """
    if eps is None:
        eps = FLAGS.EPSI
    u, s, vh = np.linalg.svd(mat)
    si = np.where(s < eps, 0.0, np.sqrt(s))

    return np.matmul(np.matmul(u, np.diag(si)), vh)


def mean_cov_np(x):
    """ This function calculates mean and covariance for 2d array x.
    This function is faster than separately running np.mean and np.cov

    :param x: 2D array, columns of x represents variables.
    :return:
    """
    mu = np.mean(x, axis=0)
    x_centred = x - mu
    cov = np.matmul(x_centred.transpose(), x_centred) / (x.shape[0] - 1.0)

    return mu, cov