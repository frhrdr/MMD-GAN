# default modules


########################################################################


########################################################################
# def scale_range(x, scale_min=-1.0, scale_max=1.0, axis=1):
#     """ This function scales numpy matrix to range [scale_min, scale_max]
#
#     """
#     x_min = np.amin(x, axis=axis, keepdims=True)
#     x_range = np.amax(x, axis=axis, keepdims=True) - x_min
#     x_range[x_range == 0.0] = 1.0
#     # scale to [0,1]
#     x = (x - x_min) / x_range
#     # scale to [scale_min, scale_max]
#     x = x * (scale_max - scale_min) + scale_min
#
#     return x


########################################################################


########################################################################
# def mean_cov_tf(x):
#     """ This function calculates mean and covariance for 2d array x.
#
#     :param x: 2D array, columns of x represents variables.
#     :return:
#     """
#     mu = tf.reduce_mean(x, axis=0, keepdims=True)  # 1-D
#     x_centred = x - mu
#     cov = tf.matmul(x_centred, x_centred, transpose_a=True) / (x.get_shape().as_list()[0] - 1.0)
#
#     return mu, cov


########################################################################
# def scale_image_range(image, scale_min=-1.0, scale_max=1.0, image_format='channels_last'):
#     """ This function scales images per channel to [-1,1]. The max and min are calculated over all samples.
#
#     Note that, in batch normalization, they also calculate the mean and std for each feature map.
#
#     :param image: 4-D numpy array, either in channels_first format or channels_last format
#     :param scale_min:
#     :param scale_max:
#     :param image_format
#     :return:
#     """
#     if len(image.shape) != 4:
#         raise AttributeError('Input must be 4-D tensor.')
#
#     if image_format == 'channels_last':
#         num_instance, height, width, num_channel = image.shape
#         pixel_channel = image.reshape((-1, num_channel))  # [pixels, channel]
#         pixel_channel = scale_range(pixel_channel, scale_min=scale_min, scale_max=scale_max, axis=0)
#         image = pixel_channel.reshape((num_instance, height, width, num_channel))
#     elif image_format == 'channels_first':
#         # scale_range works faster when axis=1, work on this
#         image = np.transpose(image, axes=(1, 0, 2, 3))
#         num_channel, num_instance, height, width = image.shape
#         pixel_channel = image.reshape((num_channel, -1))  # [channel, pixels]
#         pixel_channel = scale_range(pixel_channel, scale_min=scale_min, scale_max=scale_max, axis=1)
#         image = pixel_channel.reshape((num_channel, num_instance, height, width))
#         image = np.transpose(image, axes=(1, 0, 2, 3))  # convert back to channels_first
#
#     return image


########################################################################
# def pairwise_dist(mat1, mat2=None):
#     """ This function calculates the pairwise distance matrix dist. If mat2 is not provided,
#     dist is defined among row vectors of mat1.
#
#     The distance is formed as sqrt(mat1*mat1' - 2*mat1*mat2' + mat2*mat2')
#
#     :param mat1:
#     :param mat2:
#     :return:
#     """
#     # tf.reduce_sum() will produce result of shape (N,), which, when transposed, is still (N,)
#     # Thus, to force mm1 and mm2 (or mm1') to have different shape, tf.expand_dims() is used
#     mm1 = tf.expand_dims(tf.reduce_sum(tf.multiply(mat1, mat1), axis=1), axis=1)
#     if mat2 is None:
#         mmt = tf.multiply(tf.matmul(mat1, mat1, transpose_b=True), -2)
#         dist = tf.sqrt(tf.add(tf.add(tf.add(mm1, tf.transpose(mm1)), mmt), FLAGS.EPSI))
#     else:
#         mm2 = tf.expand_dims(tf.reduce_sum(tf.multiply(mat2, mat2), axis=1), axis=0)
#         mrt = tf.multiply(tf.matmul(mat1, mat2, transpose_b=True), -2)
#         dist = tf.sqrt(tf.add(tf.add(tf.add(mm1, mm2), mrt), FLAGS.EPSI))
#         # dist = tf.sqrt(tf.add(tf.add(mm1, mm2), mrt))
#
#     return dist


########################################################################
# def slerp(p0, p1, t):
#     """ This function calculates the spherical linear interpolation of p0 and p1
#
#     :param p0: a vector of shape (d, )
#     :param p1: a vector of shape (d, )
#     :param t: a scalar, or a vector of shape (n, )
#     :return:
#
#     Numeric instability may occur when theta is close to zero or pi. In these cases,
#     sin(t * theta) >> sin(theta). These cases are common, e.g. p0 = -p1.
#
#     """
#     from numpy.linalg import norm
#
#     theta = np.arccos(np.dot(p0 / norm(p0), p1 / norm(p1)), dtype=np.float32)
#     st = np.sin(theta)  # there is no dtype para for np.sin
#     # in case t is a vector, output is a row matrix
#     if not np.isscalar(t):
#         p0 = np.expand_dims(p0, axis=0)
#         p1 = np.expand_dims(p1, axis=0)
#         t = np.expand_dims(t, axis=1)
#     if st > 0.1:
#         p2 = np.sin((1.0 - t) * theta) / st * p0 + np.sin(t * theta) / st * p1
#     else:
#         p2 = (1.0 - t) * p0 + t * p1
#
#     return p2


########################################################################


########################################################################


########################################################################
# def l2normalization(w):
#     """ This function applies l2 normalization to the input vector.
#     If w is a matrix / tensor, the Frobenius norm is used for normalization.
#
#     :param w:
#     :return:
#     """
#
#     # tf.norm is slightly faster than tf.sqrt(tf.reduce_sum(tf.square()))
#     # it is important that axis=None; in this case, norm(w) = norm(vec(w))
#     return w / (tf.norm(w, ord='euclidean', axis=None) + FLAGS.EPSI)


########################################################################
# def batch_norm(tensor, axis=None, keepdims=False, name='norm'):
#     """ This function calculates the l2 norm for each instance in a batch
#
#     :param tensor: shape [batch_size, ...]
#     :param axis: the axis to calculate norm, could be integer or list/tuple of integers
#     :param keepdims: whether to keep dimensions
#     :param name:
#     :return:
#     """
#     with tf.name_scope(name):
#         return tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=keepdims))


########################################################################


# def get_squared_dist_ref(x, y):
#     """ This function calculates the pairwise distance between x and x, x and y, y and y.
#     It is more accurate than get_dist at the cost of higher memory and complexity.
#
#     :param x:
#     :param y:
#     :return:
#     """
#     with tf.name_scope('squared_dist_ref'):
#         if len(x.get_shape().as_list()) > 2:
#             raise AttributeError('get_dist: Input must be a matrix.')
#
#         x_expand = tf.expand_dims(x, axis=2)  # m-by-d-by-1
#         x_permute = tf.transpose(x_expand, perm=(2, 1, 0))  # 1-by-d-by-m
#         dxx = x_expand - x_permute  # m-by-d-by-m, the first page is ai - a1
#         dist_xx = tf.reduce_sum(tf.multiply(dxx, dxx), axis=1)  # m-by-m, the first column is (ai-a1)^2
#
#         if y is None:
#             return dist_xx
#         else:
#             y_expand = tf.expand_dims(y, axis=2)  # m-by-d-by-1
#             y_permute = tf.transpose(y_expand, perm=(2, 1, 0))
#             dxy = x_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
#             dist_xy = tf.reduce_sum(tf.multiply(dxy, dxy), axis=1)  # m-by-m, the first column is (ai-b1)^2
#             dyy = y_expand - y_permute  # m-by-d-by-m, the first page is ai - b1
#             dist_yy = tf.reduce_sum(tf.multiply(dyy, dyy), axis=1)  # m-by-m, the first column is (ai-b1)^2
#
#             return dist_xx, dist_xy, dist_yy


########################################################################
# def squared_dist_triplet(x, y, z, name='squared_dist', do_summary=False, scope_prefix=''):
#     """ This function calculates the pairwise distance between x and x, x and y, y and y, y and z, z and z in 'seq'
#     mode, or any two pairs in 'all' mode
#
#     :param x:
#     :param y:
#     :param z:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         x_x = tf.matmul(x, x, transpose_b=True)
#         y_y = tf.matmul(y, y, transpose_b=True)
#         z_z = tf.matmul(z, z, transpose_b=True)
#         x_y = tf.matmul(x, y, transpose_b=True)
#         y_z = tf.matmul(y, z, transpose_b=True)
#         x_z = tf.matmul(x, z, transpose_b=True)
#         d_x = tf.diag_part(x_x)
#         d_y = tf.diag_part(y_y)
#         d_z = tf.diag_part(z_z)
#
#         d_x_x = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_x + tf.expand_dims(d_x, axis=0), 0.0)
#         d_y_y = tf.maximum(tf.expand_dims(d_y, axis=1) - 2.0 * y_y + tf.expand_dims(d_y, axis=0), 0.0)
#         d_z_z = tf.maximum(tf.expand_dims(d_z, axis=1) - 2.0 * z_z + tf.expand_dims(d_z, axis=0), 0.0)
#         d_x_y = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_y + tf.expand_dims(d_y, axis=0), 0.0)
#         d_y_z = tf.maximum(tf.expand_dims(d_y, axis=1) - 2.0 * y_z + tf.expand_dims(d_z, axis=0), 0.0)
#         d_x_z = tf.maximum(tf.expand_dims(d_x, axis=1) - 2.0 * x_z + tf.expand_dims(d_z, axis=0), 0.0)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.histogram(scope_prefix + name + '/dxx', d_x_x)
#                 tf.summary.histogram(scope_prefix + name + '/dyy', d_y_y)
#                 tf.summary.histogram(scope_prefix + name + '/dzz', d_z_z)
#                 tf.summary.histogram(scope_prefix + name + '/dxy', d_x_y)
#                 tf.summary.histogram(scope_prefix + name + '/dyz', d_y_z)
#                 tf.summary.histogram(scope_prefix + name + '/dxz', d_x_z)
#
#         return d_x_x, d_y_y, d_z_z, d_x_y, d_x_z, d_y_z


########################################################################
# def get_dist_np(x, y):
#     """ This function calculates the pairwise distance between x and y using numpy
#
#     :param x: m-by-d array
#     :param y: n-by-d array
#     :return:
#     """
#     x = np.array(x, dtype=np.float32)
#     y = np.array(y, dtype=np.float32)
#     x_expand = np.expand_dims(x, axis=2)  # m-by-d-by-1
#     y_expand = np.expand_dims(y, axis=2)  # n-by-d-by-1
#     y_permute = np.transpose(y_expand, axes=(2, 1, 0))  # 1-by-d-by-n
#     dxy = x_expand - y_permute  # m-by-d-by-n, the first page is ai - b1
#     dist_xy = np.sqrt(np.sum(np.multiply(dxy, dxy), axis=1, dtype=np.float32))  # m-by-n, the first column is (ai-b1)^2
#
#     return dist_xy


#########################################################################


#######################################################################


#######################################################################


########################################################################
# def row_mean_wo_diagonal(matrix, num_col, name='mu_wo_diag'):
#     """ This function calculates the mean of each row of the matrix elements excluding the diagonal
#
#     :param matrix:
#     :param num_col:
#     :type num_col: float
#     :param name:
#     :return:
#     """
#     with tf.name_scope(name):
#         return (tf.reduce_sum(matrix, axis=1) - tf.matrix_diag_part(matrix)) / (num_col - 1.0)


#########################################################################


#########################################################################


#########################################################################


#########################################################################


#########################################################################
# def cramer(dist_xx, dist_xy, dist_yy, batch_size, name='mmd', epsi=1e-16, do_summary=False, scope_prefix=''):
#     """ This function calculates the energy distance without the need of independent samples.
#
#     The energy distance is taken originall from following paper:
#     Bellemare1, M.G., Danihelka1, I., Dabney, W., Mohamed S., Lakshminarayanan B., Hoyer S., Munos R. (2017).
#     The Cramer Distance as a Solution to Biased Wasserstein Gradients
#     However, the original method requires two batches to calculate the kernel.
#
#     :param dist_xx:
#     :param dist_xy:
#     :param dist_yy:
#     :param batch_size:
#     :param name:
#     :param epsi:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         k_xx = -tf.sqrt(dist_xx + epsi)
#         k_xy = -tf.sqrt(dist_xy + epsi)
#         k_yy = -tf.sqrt(dist_yy + epsi)
#
#         m = tf.constant(batch_size, tf.float32)
#         e_kxx = matrix_mean_wo_diagonal(k_xx, m)
#         e_kxy = matrix_mean_wo_diagonal(k_xy, m)
#         e_kyy = matrix_mean_wo_diagonal(k_yy, m)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
#
#         # return e_kxx, e_kxy, e_kyy
#         return e_kxx + e_kyy - 2.0 * e_kxy


#########################################################################


#########################################################################


#########################################################################


#########################################################################


#########################################################################


# def mmd_g_xn(
#         batch_size, d, sigma, x, dist_xx=None, y_mu=0.0, y_var=1.0, name='mmd',
#         do_summary=False, scope_prefix=''):
#     """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
#     with zero mean and specified variance.
#
#     :param x:
#     :param y_var:
#     :param batch_size:
#     :param d:
#     :param sigma:
#     :param y_mu:
#     :param dist_xx:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         # get dist_xx
#         if dist_xx is None:
#             xxt = tf.matmul(x, x, transpose_b=True)
#             dx = tf.diag_part(xxt)
#             dist_xx = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xxt + tf.expand_dims(dx, axis=0), 0.0)
#         # get dist(x, Ey)
#         dist_xy = tf.reduce_sum(tf.multiply(x - y_mu, x - y_mu), axis=1)
#
#         k_xx = tf.exp(-dist_xx / (2.0 * sigma), name='k_xx')
#         k_xy = tf.multiply(
#             tf.exp(-dist_xy / (2.0 * (sigma + y_var))),
#             tf.pow(sigma / (sigma + y_var), d / 2.0), name='k_xy')
#
#         m = tf.constant(batch_size, tf.float32)
#         e_kxx = matrix_mean_wo_diagonal(k_xx, m)
#         e_kxy = tf.reduce_mean(k_xy)
#         e_kyy = tf.pow(sigma / (sigma + 2.0 * y_var), d / 2.0)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
#
#         return e_kxx + e_kyy - 2.0 * e_kxy


# def mixture_g_xn(batch_size, d, sigma, x, dist_xx=None, y_mu=0.0, y_var=1.0, name='mmd', do_summary=False):
#     """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
#     with zero mean and specified variance. A mixture of sigma is used.
#
#     :param batch_size:
#     :param d:
#     :param sigma:
#     :param x:
#     :param dist_xx:
#     :param y_mu:
#     :param y_var:
#     :param name:
#     :param do_summary:
#     :return:
#     """
#     num_sigma = len(sigma)
#     with tf.name_scope(name):
#         mmd = 0.0
#         for i in range(num_sigma):
#             mmd_i = mmd_g_xn(
#                 batch_size, d, sigma[i], x=x, dist_xx=dist_xx, y_mu=y_mu, y_var=y_var,
#                 name='d{}'.format(i), do_summary=do_summary)
#             mmd = mmd + mmd_i
#
#         return mmd


#########################################################################
# def rand_mmd_g(dist_all, batch_size, omega=0.5, max_iter=0, name='mmd', do_summary=False, scope_prefix=''):
#     """ This function uses a global sigma to make e_k match the given omega which is sampled uniformly. The sigma is
#     initialized with geometric mean of pairwise distances and updated with Newton's method.
#
#     :param dist_all:
#     :param batch_size:
#     :param omega:
#     :param max_iter:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         m = tf.constant(batch_size, tf.float32)
#
#         def kernel(b):
#             return tf.exp(-dist_all * b)
#
#         def f(b):
#             k = kernel(b)
#             e_k = matrix_mean_wo_diagonal(k, 2 * m)
#             return e_k - omega, k
#
#         def df(k):
#             kd = -k * dist_all  # gradient of exp(-d*w)
#             e_kd = matrix_mean_wo_diagonal(kd, 2 * m)
#             return e_kd
#
#         # initialize sigma as the geometric mean of all pairwise distances
#         dist_mean = matrix_mean_wo_diagonal(dist_all, 2 * m)
#         beta = -tf.log(omega) / (dist_mean + FLAGS.EPSI)  # beta = 1/2/sigma
#         # if max_iter is larger than one, do newton's update
#         if max_iter > 0:
#             beta, _ = tf.while_loop(
#                 cond=lambda _1, i: i < max_iter,
#                 body=lambda b, i: newton_root(b, f, df, step=i),
#                 loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#
#         k_all = kernel(beta)
#         k_xx = k_all[0:batch_size, 0:batch_size]
#         k_xy_0 = k_all[0:batch_size, batch_size:]
#         k_xy_1 = k_all[batch_size:, 0:batch_size]
#         k_yy = k_all[batch_size:, batch_size:]
#
#         e_kxx = matrix_mean_wo_diagonal(k_xx, m)
#         e_kxy_0 = matrix_mean_wo_diagonal(k_xy_0, m)
#         e_kxy_1 = matrix_mean_wo_diagonal(k_xy_1, m)
#         e_kyy = matrix_mean_wo_diagonal(k_yy, m)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.summary.scalar(scope_prefix + name + '/kxy_0', e_kxy_0)
#                 tf.summary.scalar(scope_prefix + name + '/kxy_1', e_kxy_1)
#                 # tf.summary.scalar(scope_prefix + name + 'omega', omega)
#
#         return e_kxx + e_kyy - e_kxy_0 - e_kxy_1


# def sqrt_sym_mat_tf(mat, eps=None):
#     """ This function calculates the square root of symmetric matrix
#
#     :param mat:
#     :param eps:
#     :return:
#     """
#     if eps is None:
#         eps = FLAGS.EPSI
#     s, u, v = tf.svd(mat)
#     si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
#
#     return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)


# def trace_sqrt_product_tf(cov1, cov2):
#     """ This function calculates trace(sqrt(cov1 * cov2))
#
#     This code is inspired from:
#     https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
#
#     :param cov1:
#     :param cov2:
#     :return:
#     """
#     sqrt_cov1 = sqrt_sym_mat_tf(cov1)
#     cov_121 = tf.matmul(tf.matmul(sqrt_cov1, cov2), sqrt_cov1)
#
#     return tf.trace(sqrt_sym_mat_tf(cov_121))


# def jacobian(y, x, name='jacobian'):
#     """ This function calculates the jacobian matrix: dy/dx and returns a list
#
#     :param y: batch_size-by-d matrix
#     :param x: batch_size-by-s tensor
#     :param name:
#     :return:
#     """
#     with tf.name_scope(name):
#         batch_size, d = y.get_shape().as_list()
#         if d == 1:
#             return tf.reshape(tf.gradients(y, x)[0], [batch_size, -1])  # b-by-s
#         else:
#             return tf.transpose(
#                 tf.stack(
#                     [tf.reshape(tf.gradients(y[:, i], x)[0], [batch_size, -1]) for i in range(d)], axis=0),  # d-b-s
#                 perm=(1, 0, 2))  # b-d-s tensor

# def jacobian_squared_frobenius_norm(y, x, name='J_fnorm', do_summary=False):
#     """ This function calculates the squared frobenious norm, e.g. sum of square of all elements in Jacobian matrix
#
#     :param y: batch_size-by-d matrix
#     :param x: batch_size-by-s tensor
#     :param name:
#     :param do_summary:
#     :return:
#     """
#     with tf.name_scope(name):
#         batch_size, d = y.get_shape().as_list()
#         # sfn - squared frobenious norm
#         if d == 1:
#             jaco_sfn = tf.reduce_sum(tf.square(tf.reshape(tf.gradients(y, x)[0], [batch_size, -1])), axis=1)
#         else:
#             jaco_sfn = tf.reduce_sum(
#                 tf.stack(
#                     [tf.reduce_sum(
#                         tf.square(tf.reshape(tf.gradients(y[:, i], x)[0], [batch_size, -1])),  # b-vector
#                         axis=1) for i in range(d)],
#                     axis=0),  # d-by-b
#                 axis=0)  # b-vector
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.histogram('Jaco_sfn', jaco_sfn)
#
#         return jaco_sfn


# def witness_mix_g(dist_zx, dist_zy, sigma=None, name='witness', do_summary=False):
#     """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on
#     a list of t-distribution kernels.
#
#     :param dist_zx:
#     :param dist_zy:
#     :param sigma:
#     :param name:
#     :param do_summary:
#     :return:
#     """
#     num_sigma = len(sigma)
#     with tf.name_scope(name):
#         witness = 0.0
#         for i in range(num_sigma):
#             wit_i = witness_g(
#                 dist_zx, dist_zy, sigma=sigma[i], name='d{}'.format(i), do_summary=do_summary)
#             witness = witness + wit_i
#
#         return witness
#
#
# def witness_g(dist_zx, dist_zy, sigma=2.0, name='witness', do_summary=False, scope_prefix=''):
#     """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on Gaussian kernel
#
#     :param dist_zx:
#     :param dist_zy:
#     :param sigma:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         # get dist between (x, z) and (y, z)
#         # dist_zx = get_squared_dist(z, x, mode='xy', name='dist_zx', do_summary=do_summary)
#         # dist_zy = get_squared_dist(z, y, mode='xy', name='dist_zy', do_summary=do_summary)
#
#         k_zx = tf.exp(-dist_zx / (2.0 * sigma), name='k_zx')
#         k_zy = tf.exp(-dist_zy / (2.0 * sigma), name='k_zy')
#
#         e_kx = tf.reduce_mean(k_zx, axis=1)
#         e_ky = tf.reduce_mean(k_zy, axis=1)
#
#         witness = e_kx - e_ky
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.histogram(scope_prefix + name + '/kzx', e_kx)
#                 tf.summary.histogram(scope_prefix + name + '/kzy', e_ky)
#
#         return witness


# def witness_mix_t(dist_zx, dist_zy, alpha=None, beta=2.0, name='witness', do_summary=False):
#     """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on
#     a list of t-distribution kernels.
#
#     :param dist_zx:
#     :param dist_zy:
#     :param alpha:
#     :param beta:
#     :param name:
#     :param do_summary:
#     :return:
#     """
#     num_alpha = len(alpha)
#     with tf.name_scope(name):
#         witness = 0.0
#         for i in range(num_alpha):
#             wit_i = witness_t(
#                 dist_zx, dist_zy, alpha=alpha[i], beta=beta, name='d{}'.format(i), do_summary=do_summary)
#             witness = witness + wit_i
#
#         return witness
#
#
# def witness_t(dist_zx, dist_zy, alpha=1.0, beta=2.0, name='witness', do_summary=False, scope_prefix=''):
#     """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on t-distribution kernel
#
#     :param dist_zx:
#     :param dist_zy:
#     :param alpha:
#     :param beta:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#         # get dist between (x, z) and (y, z)
#         # dist_zx = get_squared_dist(z, x, mode='xy', name='dist_zx', do_summary=do_summary)
#         # dist_zy = get_squared_dist(z, y, mode='xy', name='dist_zy', do_summary=do_summary)
#
#         log_k_zx = tf.log(dist_zx / (beta * alpha) + 1.0)
#         log_k_zy = tf.log(dist_zy / (beta * alpha) + 1.0)
#
#         k_zx = tf.exp(-alpha * log_k_zx)
#         k_zy = tf.exp(-alpha * log_k_zy)
#
#         e_kx = tf.reduce_mean(k_zx, axis=1)
#         e_ky = tf.reduce_mean(k_zy, axis=1)
#
#         witness = e_kx - e_ky
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.summary.histogram(scope_prefix + name + '/kzx', e_kx)
#                 tf.summary.histogram(scope_prefix + name + '/kzy', e_ky)
#
#         return witness
