import numpy as np
import tensorflow as tf

from GeneralTools.misc_fun import FLAGS


class GANLoss(object):
    def __init__(self, do_summary=False):
        """ This class defines all kinds of loss functions for generative adversarial nets

        Current losses include:

        """
        # IO
        self.do_summary = do_summary
        self.score_gen = None
        self.score_data = None
        self.batch_size = None
        self.num_scores = None
        # loss
        self.loss_gen = None
        self.loss_dis = None
        self.dis_penalty = None
        self.dis_scale = None
        self.debug_register = None  # output used for debugging
        # hyperparameters
        self.sigma = [1.0, np.sqrt(2.0), 2.0, np.sqrt(8.0), 4.0]
        # self.sigma = [1.0, 2.0, 4.0, 8.0, 16.0]  # mmd-g, kernel scales used in original paper
        self.alpha = [0.2, 0.5, 1, 2, 5.0]  # mmd-t, kernel scales used in original paper
        self.beta = 2.0  # mmd-t, kernel scales used in original paper
        self.omega_range = [0.05, 0.85]  # rand_g parameter
        self.ref_normal = 1.0  # rand_g parameter
        # weights[0] - weights[1] = 1.0
        self.repulsive_weights = [0.0, -1.0]  # weights for e_kxy and -e_kyy; note that kyy is for the real data!
        # self.repulsive_weights = [-1.0, -2.0]  # weights for e_kxy and -e_kyy

    def _add_summary_(self):
        """ This function adds summaries

        :return:
        """
        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.compat.v1.summary.scalar('GANLoss/gen', self.loss_gen)
                tf.compat.v1.summary.scalar('GANLoss/dis', self.loss_dis)

    # def _logistic_(self):
    #     """ non-saturate logistic loss
    #     :return:
    #     """
    #     with tf.name_scope('logistic_loss'):
    #         self.loss_dis = tf.reduce_mean(tf.nn.softplus(self.score_gen) + tf.nn.softplus(-self.score_data))
    #         self.loss_gen = tf.reduce_mean(tf.nn.softplus(-self.score_gen))
    #
    # def _hinge_(self):
    #     """ hinge loss
    #     :return:
    #     """
    #     with tf.name_scope('hinge_loss'):
    #         self.loss_dis = tf.reduce_mean(
    #             tf.nn.relu(1.0 + self.score_gen)) + tf.reduce_mean(tf.nn.relu(1.0 - self.score_data))
    #         self.loss_gen = tf.reduce_mean(-self.score_gen)
    #
    # def _wasserstein_(self):
    #     """ wasserstein distance
    #     :return:
    #     """
    #     assert self.dis_penalty is not None, 'Discriminator penalty must be provided for wasserstein GAN'
    #     with tf.name_scope('wasserstein'):
    #         self.loss_gen = tf.reduce_mean(self.score_data) - tf.reduce_mean(self.score_gen)
    #         self.loss_dis = - self.loss_gen + self.dis_penalty
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
    #
    #     return self.loss_dis, self.loss_gen

    # def _mmd_g_(self):
    #     """ maximum mean discrepancy with gaussian kernel
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #
    #     # mmd
    #     self.loss_gen = mixture_mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=self.sigma,
    #         name='mmd_g', do_summary=self.do_summary)
    #     self.loss_dis = -self.loss_gen
    #     if self.dis_penalty is not None:
    #         self.loss_dis = self.loss_dis + self.dis_penalty
    #
    # def _mmd_g_bound_(self):
    #     """ maximum mean discrepancy with gaussian kernel and bounds on dxy
    #
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #
    #     # mmd
    #     self.loss_gen = mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
    #         name='mmd_g', do_summary=self.do_summary, scope_prefix='')
    #     mmd_b = mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, upper_bound=4, lower_bound=0.25,
    #         name='mmd_g_b', do_summary=self.do_summary, scope_prefix='')
    #     self.loss_dis = -mmd_b
    #     if self.dis_penalty is not None:
    #         self.loss_dis = self.loss_dis + self.dis_penalty
    #
    # def _mmd_g_mix_(self, mix_threshold=1.0):
    #     """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
    #     if discriminator is too strong
    #
    #     :param mix_threshold:
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
    #     dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
    #
    #     # mmd
    #     with tf.compat.v1.variable_scope('mmd_g_mix', reuse=tf.compat.v1.AUTO_REUSE):
    #         self.loss_gen = mixture_mmd_g(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, sigma=self.sigma,
    #             name='mmd', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
    #         # mix data if self.loss_gen surpass loss_gen_threshold
    #         mix_indices, loss_average, mix_prob = get_mix_coin(
    #             self.loss_gen, mix_threshold, batch_size=self.batch_size, loss_average_name='gen_average')
    #         dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
    #         # mmd for mixed data
    #         loss_mix = mixture_mmd_g(
    #             dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, sigma=self.sigma,
    #             name='mmd_mix', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
    #         self.loss_dis = -loss_mix
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('GANLoss/gen_average', loss_average)
    #             tf.compat.v1.summary.scalar('GANLoss/mix_prob', mix_prob)
    #             tf.compat.v1.summary.histogram('squared_dist/dxx', dist_gg)
    #             tf.compat.v1.summary.histogram('squared_dist/dyy', dist_dd)
    #             tf.compat.v1.summary.histogram('squared_dist/dxy', dist_gd)
    #
    # def _single_mmd_g_mix_(self, mix_threshold=0.2):
    #     """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
    #     if discriminator is too strong
    #
    #     :param mix_threshold:
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
    #     dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
    #
    #     # mmd
    #     with tf.compat.v1.variable_scope('mmd_g_mix', reuse=tf.compat.v1.AUTO_REUSE):
    #         self.loss_gen = mmd_g(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
    #             name='mmd', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
    #         # mix data if self.loss_gen surpass loss_gen_threshold
    #         mix_indices, loss_average, mix_prob = get_mix_coin(
    #             self.loss_gen, mix_threshold, batch_size=self.batch_size, loss_average_name='gen_average')
    #         dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
    #         # mmd for mixed data
    #         loss_mix = mmd_g(
    #             dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, sigma=1.0,
    #             name='mmd_mix', do_summary=self.do_summary, scope_prefix='mmd_g_mix/')
    #         self.loss_dis = -loss_mix
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('GANLoss/gen_average', loss_average)
    #             tf.compat.v1.summary.scalar('GANLoss/mix_prob', mix_prob)
    #             tf.compat.v1.summary.histogram('squared_dist/dxx', dist_gg)
    #             tf.compat.v1.summary.histogram('squared_dist/dyy', dist_dd)
    #             tf.compat.v1.summary.histogram('squared_dist/dxy', dist_gd)
    #
    # def _mmd_t_(self):
    #     """ maximum mean discrepancy with t-distribution kernel
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #     # mmd
    #     self.loss_gen = mixture_mmd_t(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, alpha=self.alpha, beta=self.beta,
    #         name='mmd_t', do_summary=self.do_summary)
    #     self.loss_dis = -self.loss_gen
    #     if self.dis_penalty is not None:
    #         self.loss_dis = self.loss_dis + self.dis_penalty

    # def _rand_g_(self):
    #     """ maximum mean discrepancy with gaussian kernel and random kernel scale
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #
    #     # mmd
    #     with tf.name_scope('rand_g'):
    #         omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #             if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    #         loss_gr = rand_mmd_g_xy(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         loss_gn = rand_mmd_g_xn(
    #             self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
    #             max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         loss_rn = rand_mmd_g_xn(
    #             self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
    #             max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         # final loss
    #         self.loss_gen = loss_gr
    #         self.loss_dis = loss_rn - loss_gr
    #
    #     # self.debug_register = [omega, loss_gr, loss_gn, loss_rn]
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('rand_g/omega', omega)
    #             tf.compat.v1.summary.scalar('GANLoss/gr', loss_gr)
    #             tf.compat.v1.summary.scalar('GANLoss/gn', loss_gn)
    #             tf.compat.v1.summary.scalar('GANLoss/rn', loss_rn)
    #
    # def _rand_g_bounded_(self):
    #     """ maximum mean discrepancy with gaussian kernel and random kernel scale, and upper bounds on dxy
    #
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #
    #     with tf.name_scope('rand_g'):
    #         omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #             if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    #         loss_gr, loss_gr_b = rand_mmd_g_xy_bounded(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         # loss_gn = rand_mmd_g_xn(
    #         #     self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
    #         #     max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         # loss_rn = rand_mmd_g_xn(
    #         #     self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
    #         #     max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g/')
    #         # final loss
    #         self.loss_gen = loss_gr
    #         self.loss_dis = - loss_gr_b
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('rand_g/omega', omega)
    #             tf.compat.v1.summary.scalar('GANLoss/gr', loss_gr)
    #             # tf.compat.v1.summary.scalar('GANLoss/gn', loss_gn)
    #             # tf.compat.v1.summary.scalar('GANLoss/rn', loss_rn)
    #
    # def _rand_g_mix_(self, mix_threshold=0.2):
    #     """ maximum mean discrepancy with gaussian kernel and random kernel scale
    #     and mixing score_gen and score_data if discriminator is too strong
    #     """
    #     # calculate pairwise distance
    #     pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
    #     dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
    #     # mmd
    #     with tf.compat.v1.variable_scope('rand_g_mix', reuse=tf.compat.v1.AUTO_REUSE):
    #         omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #             if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    #         loss_gr = rand_mmd_g_xy(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
    #         loss_gn = rand_mmd_g_xn(
    #             self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
    #             max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
    #         loss_rn = rand_mmd_g_xn(
    #             self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
    #             max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
    #         # mix data if self.loss_gen surpass loss_gen_threshold
    #         mix_indices, loss_average, mix_prob = get_mix_coin(
    #             loss_gr, mix_threshold, batch_size=self.batch_size, loss_average_name='gr_average')
    #         dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
    #         # mmd for mixed data
    #         loss_gr_mix = rand_mmd_g_xy(
    #             dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr_mix', do_summary=self.do_summary, scope_prefix='rand_g_mix/')
    #         # final loss
    #         self.loss_gen = loss_gr
    #         self.loss_dis = loss_rn - loss_gr_mix
    #         # self.debug_register = loss_rn
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('rand_g_mix/omega', omega)
    #             tf.compat.v1.summary.scalar('GANLoss/gr_average', loss_average)
    #             tf.compat.v1.summary.scalar('GANLoss/mix_prob', mix_prob)
    #             tf.compat.v1.summary.histogram('squared_dist/dxx', dist_gg)
    #             tf.compat.v1.summary.histogram('squared_dist/dyy', dist_dd)
    #             tf.compat.v1.summary.histogram('squared_dist/dxy', dist_gd)
    #             tf.compat.v1.summary.scalar('GANLoss/gr', loss_gr)
    #             tf.compat.v1.summary.scalar('GANLoss/gn', loss_gn)
    #             tf.compat.v1.summary.scalar('GANLoss/rn', loss_rn)
    #             tf.compat.v1.summary.scalar('GANLoss/gr_mix', loss_gr_mix)
    #
    # def _sym_rg_mix_(self, mix_threshold=0.2):
    #     """ symmetric version of rand_g_mix
    #
    #     :param mix_threshold:
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
    #     dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
    #     # mmd
    #     with tf.compat.v1.variable_scope('sym_rg_mix', reuse=tf.compat.v1.AUTO_REUSE):
    #         omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #             if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    #         loss_gr = rand_mmd_g_xy(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         loss_gn = rand_mmd_g_xn(
    #             self.score_gen, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_gg, omega=omega,
    #             max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         loss_rn = rand_mmd_g_xn(
    #             self.score_data, self.ref_normal, self.batch_size, self.num_scores, dist_xx=dist_dd, omega=omega,
    #             max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         # mix data if self.loss_gen surpass loss_gen_threshold
    #         mix_indices, loss_average, mix_prob = get_mix_coin(
    #             loss_gr, mix_threshold, batch_size=self.batch_size, loss_average_name='gr_average')
    #         dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
    #         # mmd for mixed data
    #         loss_gr_mix = rand_mmd_g_xy(
    #             dist_gg_mix, dist_gd_mix, dist_dd_mix, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr_mix', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         # final loss
    #         self.loss_gen = loss_gr + loss_gn
    #         self.loss_dis = loss_rn - loss_gr_mix - loss_gn
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('rand_g_mix/omega', omega)
    #             tf.compat.v1.summary.scalar('GANLoss/gr_average', loss_average)
    #             tf.compat.v1.summary.scalar('GANLoss/mix_prob', mix_prob)
    #             tf.compat.v1.summary.histogram('squared_dist/dxx', dist_gg)
    #             tf.compat.v1.summary.histogram('squared_dist/dyy', dist_dd)
    #             tf.compat.v1.summary.histogram('squared_dist/dxy', dist_gd)
    #             tf.compat.v1.summary.scalar('GANLoss/gr', loss_gr)
    #             tf.compat.v1.summary.scalar('GANLoss/gn', loss_gn)
    #             tf.compat.v1.summary.scalar('GANLoss/rn', loss_rn)
    #             tf.compat.v1.summary.scalar('GANLoss/gr_mix', loss_gr_mix)
    #
    # def _sym_rand_g_(self):
    #     """ Version 2 of symmetric rand_g. This function does not use label smoothing
    #
    #     This function does not work.
    #
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     pair_dist = get_squared_dist(tf.concat((self.score_gen, self.score_data), axis=0))
    #     dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=self.batch_size)
    #     # mmd
    #     with tf.compat.v1.variable_scope('sym_rg_mix', reuse=tf.compat.v1.AUTO_REUSE):
    #         omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #             if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    #         loss_gr = rand_mmd_g_xy(
    #             dist_gg, dist_gd, dist_dd, self.batch_size, omega=omega,
    #             max_iter=3, name='mmd_gr', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         loss_gn = rand_mmd_g_xn(
    #             self.score_gen, self.ref_normal, self.batch_size, self.num_scores, y_mu=-0.5, dist_xx=dist_gg,
    #             omega=omega, max_iter=3, name='mmd_gn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         loss_rn = rand_mmd_g_xn(
    #             self.score_data, self.ref_normal, self.batch_size, self.num_scores, y_mu=0.5, dist_xx=dist_dd,
    #             omega=omega, max_iter=3, name='mmd_rn', do_summary=self.do_summary, scope_prefix='sym_rg_mix/')
    #         self.loss_gen = loss_gr
    #         self.loss_dis = 0.5*(loss_rn + loss_gn) - loss_gr
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('sym_rg_mix/omega', omega)
    #             tf.compat.v1.summary.histogram('squared_dist/dxx', dist_gg)
    #             tf.compat.v1.summary.histogram('squared_dist/dyy', dist_dd)
    #             tf.compat.v1.summary.histogram('squared_dist/dxy', dist_gd)
    #             tf.compat.v1.summary.scalar('GANLoss/gr', loss_gr)
    #             tf.compat.v1.summary.scalar('GANLoss/gn', loss_gn)
    #             tf.compat.v1.summary.scalar('GANLoss/rn', loss_rn)
    #
    # def _rand_g_instance_noise_(self, mix_threshold=0.2):
    #     """ This function tests instance noise
    #
    #     :param mix_threshold:
    #     :return:
    #     """
    #     with tf.compat.v1.variable_scope('ins_noise'):
    #         sigma = tf.compat.v1.get_variable(
    #             'sigma', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
    #         stddev = tf.log(sigma + 1.0)  # to slow down sigma increase
    #         noise_gen = tf.random.normal(
    #             self.score_gen.get_shape().as_list(), mean=0.0, stddev=stddev,
    #             name='noise_gen', dtype=tf.float32)
    #         noise_x = tf.random.normal(
    #             self.score_data.get_shape().as_list(), mean=0.0, stddev=stddev,
    #             name='noise_x', dtype=tf.float32)
    #         self.score_gen = self.score_gen + noise_gen
    #         self.score_data = self.score_data + noise_x
    #         # use rand_g loss
    #         self._rand_g_()
    #         # update sigma
    #         loss_average = moving_average_copy(self.loss_gen, 'mmd_mean')
    #         tf.compat.v1.add_to_collection(
    #             tf.GraphKeys.UPDATE_OPS,
    #             tf.assign(
    #                 sigma,
    #                 tf.clip_by_value(
    #                     sigma + 0.001 * (loss_average - mix_threshold),
    #                     clip_value_min=0.0, clip_value_max=1.7183)))
    #
    #     if self.do_summary:
    #         with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #             tf.compat.v1.summary.scalar('GANLoss/gr_average', loss_average)
    #             tf.compat.v1.summary.scalar('GANLoss/sigma', sigma)

    def _repulsive_mmd_g_(self):
        """ repulsive loss

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        # self.loss_gen, self.loss_dis = mmd_g(
        #     dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.6,
        #     name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        self.loss_gen, self.loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = (self.loss_dis - 1.0) * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _repulsive_mmd_g_with_gmm(self, extra_loss_vars):
        """ repulsive loss

        :return:
        """

        pi, mu, sig, lambda_mu, lambda_sig = [extra_loss_vars[k] for k in
                                              ['pi', 'mu', 'sig', 'lambda_mu', 'lambda_sig']]
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)

        # gmm_term = lambda_mu *

        self.loss_gen, self.loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = (self.loss_dis - 1.0) * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _repulsive_mmd_g_inv_disc(self):
        """ repulsive loss with inverted discriminator loss so sample encodings can better be modeled by MoG

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        # self.loss_gen, self.loss_dis = mmd_g(
        #     dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.6,
        #     name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        self.loss_gen, self.loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)

        self.loss_dis = -self.loss_dis  # THAT'S ALL THAT CHANGES!

        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = (self.loss_dis - 1.0) * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _repulsive_mmd_g_bounded_(self):
        """ rmb loss

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        self.loss_gen, self.loss_dis = mmd_g_bounded(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, lower_bound=0.25, upper_bound=4.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = self.loss_dis * self.dis_scale
            if self.do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar('GANLoss/dis_scale', self.dis_scale)

    def _test_(self):
        self.loss_dis = 0.0
        self.loss_gen = 0.0

    def __call__(self, score_gen, score_data, loss_type='logistic', **kwargs):
        """  This function calls one of the loss functions.

        :param score_gen:
        :param score_data:
        :param loss_type:
        :param kwargs:
        :return:
        """
        # IO and hyperparameters
        self.score_gen = score_gen  # ------- apparently these are just the batch embedding. what a misleading name..
        self.score_data = score_data
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'd' in kwargs:
            self.num_scores = kwargs['d']
        if 'dis_penalty' in kwargs:
            self.dis_penalty = kwargs['dis_penalty']
        if 'dis_scale' in kwargs:
            self.dis_scale = kwargs['dis_scale']
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'beta' in kwargs:
            self.beta = kwargs['beta']
        if 'omega' in kwargs:
            self.omega_range = kwargs['omega']
        if 'ref_normal' in kwargs:
            self.ref_normal = kwargs['ref_normal']
        if 'rep_weights' in kwargs:
            self.repulsive_weights = kwargs['rep_weights']
        # check inputs
        if loss_type in {'fixed_g', 'mmd_g', 'fixed_t', 'mmd_t', 'mmd_g_mix', 'fixed_g_mix',
                         'rand_g', 'rand_g_mix', 'sym_rg_mix', 'instance_noise', 'ins_noise',
                         'sym_rg', 'rgb', 'rep', 'rep_gp', 'rmb', 'rmb_gp',
                         'rep_inv_disc'}:
            assert self.batch_size is not None, 'GANLoss: batch_size must be provided'
            if loss_type in {'rand_g', 'rand_g_mix', 'sym_rg_mix', 'sym_rg'}:
                assert self.num_scores is not None, 'GANLoss: d must be provided'
        if loss_type in {'rep_gp', 'rmb_gp', 'wasserstein'}:
            assert self.dis_penalty is not None, 'Discriminator penalty must be provided.'
        if loss_type in {'rep_ds', 'rmb_ds'}:
            assert self.dis_scale is not None, 'Discriminator loss scale must be provided.'

        # loss

        # if loss_type in {'logistic', ''}:
        #     self._logistic_()
        # elif loss_type == 'hinge':
        #     self._hinge_()
        # elif loss_type == 'wasserstein':
        #     self._wasserstein_()
        # elif loss_type in {'fixed_g', 'mmd_g'}:
        #     self._mmd_g_()
        # elif loss_type in {'mgb'}:
        #     self._mmd_g_bound_()
        # elif loss_type in {'fixed_t', 'mmd_t'}:
        #     self._mmd_t_()
        # elif loss_type in {'mmd_g_mix', 'fixed_g_mix'}:
        #     if 'mix_threshold' in kwargs:
        #         self._mmd_g_mix_(kwargs['mix_threshold'])
        #     else:
        #         self._mmd_g_mix_()
        # elif loss_type in {'sgm'}:  # single mmd-g mix
        #     if 'mix_threshold' in kwargs:
        #         self._single_mmd_g_mix_(kwargs['mix_threshold'])
        #     else:
        #         self._single_mmd_g_mix_()
        # elif loss_type == 'rand_g':
        #     self._rand_g_()
        # elif loss_type == 'rgb':
        #     self._rand_g_bounded_()
        # elif loss_type == 'rand_g_mix':
        #     if 'mix_threshold' in kwargs:
        #         self._rand_g_mix_(kwargs['mix_threshold'])
        #     else:
        #         self._rand_g_mix_()
        # elif loss_type == 'sym_rg_mix':
        #     if 'mix_threshold' in kwargs:
        #         self._sym_rg_mix_(kwargs['mix_threshold'])
        #     else:
        #         self._sym_rg_mix_()
        # elif loss_type in {'sym_rg', 'sym_rand_g'}:
        #     self._sym_rand_g_()
        # elif loss_type in {'instance_noise', 'ins_noise'}:
        #     if 'mix_threshold' in kwargs:
        #         self._rand_g_instance_noise_(kwargs['mix_threshold'])
        #     else:
        #         self._rand_g_instance_noise_()
        # el-
        if loss_type in {'rep', 'rep_mmd_g', 'rep_gp', 'rep_ds'}:
            self._repulsive_mmd_g_()
        elif loss_type in {'rmb', 'rep_b', 'rep_mmd_b', 'rmb_gp', 'rmb_ds'}:
            self._repulsive_mmd_g_bounded_()
        elif loss_type == 'test':
            self._test_()
        elif loss_type == 'rep_inv_disc':
            self._repulsive_mmd_g_inv_disc()
        elif isinstance(loss_type, dict):
            assert 'type' in loss_type.keys()
            if loss_type['type'] == 'direct_gmm_loss':
                print('direct_gmm_loss queried')
                self._repulsive_mmd_g_with_gmm(loss_type)
        else:
            raise NotImplementedError('Not implemented.')

        self._add_summary_()

        return self.loss_gen, self.loss_dis

    def apply(self, score_gen, score_data, loss_type='logistic', **kwargs):
        return self.__call__(score_gen, score_data, loss_type=loss_type, **kwargs)

    def get_register(self):
        """ This function returns the registered tensor

        :return:
        """
        # loss object always forgets self.debug_register after its value returned
        registered_info = self.debug_register
        self.debug_register = None
        return registered_info


# def mixture_mmd_t(
#         dist_xx, dist_xy, dist_yy, batch_size, alpha=None, beta=2.0, var_targets=None, name='mmd',
#         do_summary=False, scope_prefix=''):
#     """ This function calculates the maximum mean discrepancy with a list of t-distribution kernels
#
#     :param dist_xx:
#     :param dist_xy:
#     :param dist_yy:
#     :param batch_size:
#     :param alpha: [0.2, 0.5, 1, 2, 25]
#     :type alpha: list
#     :param beta:
#     :param var_targets: if alpha is trainable, var_targets contain the target for each alpha
#     :type var_targets: list
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     num_alpha = len(alpha) if isinstance(alpha, list) else len(var_targets)
#     with tf.name_scope(name):
#         mmd = 0.0
#         if var_targets is None:
#             for i in range(num_alpha):
#                 mmd_i = mmd_t(
#                     dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta,
#                     name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
#                 mmd = mmd + mmd_i
#
#             return mmd
#         else:
#             loss_alpha = 0.0
#             for i in range(num_alpha):
#                 mmd_i, loss_i = mmd_t(
#                     dist_xx, dist_xy, dist_yy, batch_size, alpha=alpha[i], beta=beta, var_target=var_targets[i],
#                     name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
#                 mmd = mmd + mmd_i
#                 loss_alpha = loss_alpha + loss_i
#
#             return mmd, loss_alpha


def mmd_g_bounded(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel

    The kernel is taken from following paper:
    Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
    MMD GAN: Towards Deeper Understanding of Moment Matching Network.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :param var_target: if sigma is trainable, var_target contain the target for sigma
    :param upper_bound:
    :param lower_bound:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
    :type custom_weights: list
    :return:
    """
    with tf.name_scope(name):
        k_xx = tf.exp(-dist_xx / (2.0 * sigma ** 2), name='k_xx')
        k_yy = tf.exp(-dist_yy / (2.0 * sigma ** 2), name='k_yy')
        k_xy = tf.exp(-dist_xy / (2.0 * sigma ** 2), name='k_xy')

        # in rep loss, custom_weights[0] - custom_weights[1] = 1
        k_xx_b = tf.exp(-tf.maximum(dist_xx, lower_bound) / (2.0 * sigma ** 2), name='k_xx_lb')
        if custom_weights[0] > 0:
            k_xy_b = tf.exp(-tf.minimum(dist_xy, upper_bound) / (2.0 * sigma ** 2), name='k_xy_ub')
        else:
            k_xy_b = k_xy  # no lower bound should be enforced as k_xy may be zero at equilibrium
        if custom_weights[1] > 0:  # the original mmd-g
            k_yy_b = tf.exp(-tf.maximum(dist_yy, lower_bound) / (2.0 * sigma ** 2), name='k_yy_ub')
        else:  # the repulsive mmd-g
            k_yy_b = tf.exp(-tf.minimum(dist_yy, upper_bound) / (2.0 * sigma ** 2), name='k_yy_ub')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)
        e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
        e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)
        e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m) if custom_weights[0] < 0 else e_kxy

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx_b', e_kxx_b)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy_b', e_kyy_b)
                if custom_weights[0] > 0:
                    tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)

        if var_target is None:
            if custom_weights is None:
                mmd = e_kxx + e_kyy - 2.0 * e_kxy
                return mmd
            else:
                assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
                mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
                mmd2 = custom_weights[0] * e_kxy_b - e_kxx_b - custom_weights[1] * e_kyy_b
                return mmd1, mmd2
        else:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


# def mixture_mmd_g(
#         dist_xx, dist_xy, dist_yy, batch_size, sigma=None, var_targets=None, name='mmd_g',
#         do_summary=False, scope_prefix=''):
#     """ This function calculates the maximum mean discrepancy with a list of Gaussian distribution kernel
#
#     :param dist_xx:
#     :param dist_xy:
#     :param dist_yy:
#     :param batch_size:
#     :param sigma:
#     :type sigma: list
#     :param var_targets: if sigma is trainable, var_targets contain the target for each sigma
#     :type var_targets: list
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     num_sigma = len(sigma) if isinstance(sigma, list) else len(var_targets)
#     with tf.name_scope(name):
#         mmd = 0.0
#         if var_targets is None:
#             for i in range(num_sigma):
#                 mmd_i = mmd_g(
#                     dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i],
#                     name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
#                 mmd = mmd + mmd_i
#
#             return mmd
#         else:
#             loss_sigma = 0.0
#             for i in range(num_sigma):
#                 mmd_i, loss_i = mmd_g(
#                     dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i], var_target=var_targets[i],
#                     name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
#                 mmd = mmd + mmd_i
#                 loss_sigma = loss_sigma + loss_i
#
#             return mmd, loss_sigma


# def rand_mmd_g_xy(
#         dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
#         do_summary=False, scope_prefix=''):
#     """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
#     given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
#     with Newton's method.
#
#     :param dist_xx:
#     :param dist_xy:
#     :param dist_yy:
#     :param dist_yx: optional, if dist_xy and dist_yx are not the same
#     :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
#     :param omega:
#     :param max_iter:
#     :param name:
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#
#         def kernel(dist, b):
#             return tf.exp(-dist * b)
#
#         def f(b):
#             k = kernel(dist_xy, b)
#             e_k = tf.reduce_mean(k)
#             return e_k - omega, k
#
#         def df(k):
#             kd = -k * dist_xy  # gradient of exp(-d*w)
#             e_kd = tf.reduce_mean(kd)
#             return e_kd
#
#         def f_plus(b):
#             k0 = kernel(dist_xy, b)
#             e_k0 = tf.reduce_mean(k0)
#             k1 = kernel(dist_yx, b)
#             e_k1 = tf.reduce_mean(k1)
#             return e_k0 + e_k1 - 2.0 * omega, (k0, k1)
#
#         def df_plus(k):
#             kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
#             kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
#             e_kd = tf.reduce_mean(kd0) + tf.reduce_mean(kd1)
#             return e_kd
#
#         if dist_yx is None:
#             # initialize sigma as the geometric mean of dist_xy
#             beta = -tf.log(omega) / tf.reduce_mean(dist_xy + FLAGS.EPSI)  # beta = 1/2/sigma
#             # if max_iter is larger than one, do newton's update
#             if max_iter > 0:
#                 beta, _ = tf.while_loop(
#                     cond=lambda _1, i: i < max_iter,
#                     body=lambda b, i: newton_root(b, f, df, step=i),
#                     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#         else:
#             # initialize sigma as the geometric mean of dist_xy and dist_yx
#             # beta = 1/2/sigma
#             beta = -2.0 * tf.log(omega) / (tf.reduce_mean(dist_xy) + tf.reduce_mean(dist_yx) + FLAGS.EPSI)
#             # if max_iter is larger than one, do newton's update
#             if max_iter > 0:
#                 beta, _ = tf.while_loop(
#                     cond=lambda _1, i: i < max_iter,
#                     body=lambda b, i: newton_root(b, f_plus, df_plus, step=i),
#                     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#
#         k_xx = kernel(dist_xx, beta)
#         k_xy = kernel(dist_xy, beta)
#         k_yy = kernel(dist_yy, beta)
#
#         if batch_size is None:  # include diagonal elements in k**
#             e_kxx = tf.reduce_mean(k_xx)
#             e_kxy = tf.reduce_mean(k_xy)
#             e_kyy = tf.reduce_mean(k_yy)
#         else:  # exclude diagonal elements in k**
#             m = tf.constant(batch_size, tf.float32)
#             e_kxx = matrix_mean_wo_diagonal(k_xx, m)
#             e_kxy = matrix_mean_wo_diagonal(k_xy, m)
#             e_kyy = matrix_mean_wo_diagonal(k_yy, m)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
#                 # tf.compat.v1.summary.scalar(scope_prefix + name + 'omega', omega)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dxx', dist_xx)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dxy', dist_xy)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dyy', dist_yy)
#
#         if dist_yx is None:
#             return e_kxx + e_kyy - 2.0 * e_kxy
#         else:
#             k_yx = kernel(dist_yx, beta)
#             if batch_size is None:
#                 e_kyx = tf.reduce_mean(k_yx)
#             else:
#                 m = tf.constant(batch_size, tf.float32)
#                 e_kyx = matrix_mean_wo_diagonal(k_yx, m)
#             if do_summary:
#                 with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                     tf.compat.v1.summary.scalar(scope_prefix + name + 'kyx', e_kyx)
#             return e_kxx + e_kyy - e_kxy - e_kyx


# def rand_mmd_g_xy_bounded(
#         dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
#         beta_lb=0.125, beta_ub=2.0, do_summary=False, scope_prefix=''):
#     """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
#     given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
#     with Newton's method.
#
#     :param dist_xx:
#     :param dist_xy:
#     :param dist_yy:
#     :param dist_yx: optional, if dist_xy and dist_yx are not the same
#     :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
#     :param omega:
#     :param max_iter:
#     :param name:
#     :param beta_lb: lower bound for beta (upper bound for sigma)
#     :param beta_ub: upper bound for beta (lower bound for sigma)
#     :param do_summary:
#     :param scope_prefix:
#     :return:
#     """
#     with tf.name_scope(name):
#
#         def kernel(dist, b):
#             return tf.exp(-dist * b)
#
#         def f(b):
#             k = kernel(dist_xy, b)
#             e_k = tf.reduce_mean(k)
#             return e_k - omega, k
#
#         def df(k):
#             kd = -k * dist_xy  # gradient of exp(-d*w)
#             e_kd = tf.reduce_mean(kd)
#             return e_kd
#
#         def f_plus(b):
#             k0 = kernel(dist_xy, b)
#             e_k0 = tf.reduce_mean(k0)
#             k1 = kernel(dist_yx, b)
#             e_k1 = tf.reduce_mean(k1)
#             return e_k0 + e_k1 - 2.0 * omega, (k0, k1)
#
#         def df_plus(k):
#             kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
#             kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
#             e_kd = tf.reduce_mean(kd0) + tf.reduce_mean(kd1)
#             return e_kd
#
#         if dist_yx is None:
#             # initialize sigma as the geometric mean of dist_xy
#             beta = -tf.log(omega) / tf.reduce_mean(dist_xy + FLAGS.EPSI)  # beta = 1/2/sigma
#             # if max_iter is larger than one, do newton's update
#             if max_iter > 0:
#                 beta, _ = tf.while_loop(
#                     cond=lambda _1, i: i < max_iter,
#                     body=lambda b, i: newton_root(b, f, df, step=i),
#                     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#         else:
#             # initialize sigma as the geometric mean of dist_xy and dist_yx
#             # beta = 1/2/sigma
#             beta = -2.0 * tf.log(omega) / (tf.reduce_mean(dist_xy) + tf.reduce_mean(dist_yx) + FLAGS.EPSI)
#             # if max_iter is larger than one, do newton's update
#             if max_iter > 0:
#                 beta, _ = tf.while_loop(
#                     cond=lambda _1, i: i < max_iter,
#                     body=lambda b, i: newton_root(b, f_plus, df_plus, step=i),
#                     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#
#         beta = tf.clip_by_value(beta, beta_lb, beta_ub)
#         k_xx = kernel(dist_xx, beta)
#         k_xy = kernel(dist_xy, beta)
#         k_yy = kernel(dist_yy, beta)
#         k_xx_b = kernel(tf.maximum(dist_xx, 0.125/beta), beta)
#         k_xy_b = kernel(tf.minimum(dist_xy, 2.0/beta), beta)
#         k_yy_b = kernel(tf.maximum(dist_yy, 0.125/beta), beta)
#
#         if batch_size is None:  # include diagonal elements in k**
#             e_kxx = tf.reduce_mean(k_xx)
#             e_kxy = tf.reduce_mean(k_xy)
#             e_kyy = tf.reduce_mean(k_yy)
#             e_kxx_b = tf.reduce_mean(k_xx_b)
#             e_kxy_b = tf.reduce_mean(k_xy_b)
#             e_kyy_b = tf.reduce_mean(k_yy_b)
#         else:  # exclude diagonal elements in k**
#             m = tf.constant(batch_size, tf.float32)
#             e_kxx = matrix_mean_wo_diagonal(k_xx, m)
#             e_kxy = matrix_mean_wo_diagonal(k_xy, m)
#             e_kyy = matrix_mean_wo_diagonal(k_yy, m)
#             e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
#             e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m)
#             e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/beta', beta)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx_b', e_kxx_b)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy_b', e_kyy_b)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)
#                 # tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy_b', e_kxy_b)
#                 # tf.compat.v1.summary.scalar(scope_prefix + name + 'omega', omega)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dxx', dist_xx)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dxy', dist_xy)
#                 # tf.compat.v1.summary.histogram(scope_prefix + name + 'dyy', dist_yy)
#
#         if dist_yx is None:
#             return e_kxx + e_kyy - 2.0 * e_kxy, e_kxx_b - 2.0 * e_kyy_b + e_kxy_b
#         else:
#             k_yx = kernel(dist_yx, beta)
#             # k_yx_b = kernel(tf.minimum(dist_yx, upper_bound), beta)
#             if batch_size is None:
#                 e_kyx = tf.reduce_mean(k_yx)
#                 # e_kyx_b = tf.reduce_mean(k_yx_b)
#             else:
#                 m = tf.constant(batch_size, tf.float32)
#                 e_kyx = matrix_mean_wo_diagonal(k_yx, m)
#                 # e_kyx_b = matrix_mean_wo_diagonal(k_yx_b, m)
#             if do_summary:
#                 with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                     tf.compat.v1.summary.scalar(scope_prefix + name + 'kyx', e_kyx)
#                     # tf.compat.v1.summary.scalar(scope_prefix + name + 'kyx_b', e_kyx_b)
#             return e_kxx + e_kyy - e_kxy - e_kyx


# def rand_mmd_g_xn(
#         x, y_rho, batch_size, d, y_mu=0.0, dist_xx=None, omega=0.5, max_iter=0, name='mmd',
#         do_summary=False, scope_prefix=''):
#     """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
#     with zero mean and specified STD. This function uses a global sigma to make e_k match the given omega
#     which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated with
#     Newton's method.
#
#     :param x:
#     :param y_rho: y_std = sqrt(y_rho / 2.0 / d)
#     :param batch_size:
#     :param d: number of features in x
#     :param y_mu:
#     :param dist_xx:
#     :param omega:
#     :param max_iter:
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
#         def kernel(dist, b):
#             return tf.exp(-dist * b)
#
#         def f(b):
#             const_f = d / (d + b * y_rho)
#             k = tf.pow(const_f, d / 2.0) * tf.exp(-b * const_f * dist_xy)
#             e_k = tf.reduce_mean(k)
#             return e_k - omega, (const_f, k, e_k)
#
#         def df(k):
#             kd = -y_rho * k[0] / 2.0 * k[2] - tf.reduce_mean(tf.pow(k[0], 2) * dist_xy * k[1])  # gradient of exp(-d*w)
#             e_kd = tf.reduce_mean(kd)
#             return e_kd
#
#         # initialize sigma as the geometric mean of dist_xy
#         beta = -tf.log(omega) / (tf.reduce_mean(dist_xy) + y_rho / 2.0)  # beta = 1/2/sigma
#         # if max_iter is larger than one, do newton's update
#         if max_iter > 0:
#             beta, _ = tf.while_loop(
#                 cond=lambda _1, i: i < max_iter,
#                 body=lambda b, i: newton_root(b, f, df, step=i),
#                 loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
#
#         const_0 = d / (d + beta * y_rho)
#         k_xx = kernel(dist_xx, beta)
#         k_xy = tf.pow(const_0, d / 2.0) * tf.exp(-beta * const_0 * dist_xy)
#
#         e_kxx = matrix_mean_wo_diagonal(k_xx, tf.constant(batch_size, tf.float32))
#         e_kxy = tf.reduce_mean(k_xy)
#         e_kyy = tf.pow(d / (d + 2.0 * beta * y_rho), d / 2.0)
#
#         if do_summary:
#             with tf.name_scope(None):  # return to root scope to avoid scope overlap
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
#                 tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)
#
#         return e_kxx + e_kyy - 2.0 * e_kxy


# def slice_pairwise_distance(pair_dist, batch_size=None, indices=None):
#     """ This function slice pair-dist into smaller pairwise distance matrices
#
#     :param pair_dist: 2batch_size-by-2batch_size pairwise distance matrix
#     :param batch_size:
#     :param indices:
#     :return:
#     """
#     with tf.name_scope('slice_dist'):
#         if indices is None:
#             dist_g1 = pair_dist[0:batch_size, 0:batch_size]
#             dist_g2 = pair_dist[batch_size:, batch_size:]
#             dist_g1g2 = pair_dist[0:batch_size, batch_size:]
#         else:
#             mix_group_1 = tf.concat((indices, tf.logical_not(indices)), axis=0)
#             mix_group_2 = tf.concat((tf.logical_not(indices), indices), axis=0)
#             dist_g1 = mat_slice(pair_dist, mix_group_1)
#             dist_g2 = mat_slice(pair_dist, mix_group_2)
#             dist_g1g2 = mat_slice(pair_dist, mix_group_1, mix_group_2)
#
#     return dist_g1, dist_g1g2, dist_g2


# def get_mix_coin(
#         loss, loss_threshold, batch_size=None, loss_average_update=0.01, mix_prob_update=0.01,
#         loss_average_name='loss_ave'):
#     """ This function generate a mix_indices to mix data from two classes
#
#     :param loss:
#     :param loss_threshold:
#     :param batch_size:
#     :param loss_average_update:
#     :param mix_prob_update:
#     :param loss_average_name:
#     :return:
#     """
#     with tf.compat.v1.variable_scope('coin', reuse=tf.compat.v1.AUTO_REUSE):
#         # calculate moving average of loss
#         loss_average = moving_average_copy(loss, loss_average_name, rho=loss_average_update)
#         # update mixing probability
#         mix_prob = moving_average_update(
#             'prob', [], loss_average - loss_threshold, rho=mix_prob_update, clip_values=[0.0, 0.5])
#         # sample mix_indices
#         uni = tf.random_uniform([batch_size], 0.0, 1.0, dtype=tf.float32, name='uni')
#         mix_indices = tf.greater(uni, mix_prob, name='mix_indices')  # mix_indices for using original data
#
#     # loss_average and mix_prob is returned so that summary can be added outside of coin variable scope
#     return mix_indices, loss_average, mix_prob


# def moving_average_copy(tensor, name=None, rho=0.01, initializer=None, dtype=tf.float32):
#     """ This function creates a moving average copy of tensor
#
#     :param tensor:
#     :param name: name for the moving average
#     :param rho:
#     :param initializer:
#     :param dtype:
#     :return:
#     """
#     if initializer is None:
#         initializer = tf.zeros_initializer
#     if name is None:
#         name = get_tensor_name(tensor) + '_copy'
#
#     tensor_copy = tf.compat.v1.get_variable(
#         name, shape=tensor.get_shape().as_list(), dtype=dtype, initializer=initializer, trainable=False)
#     tf.compat.v1.add_to_collection(
#         tf.GraphKeys.UPDATE_OPS,
#         tf.assign(tensor_copy, (1.0 - rho) * tensor_copy + rho * tensor))
#
#     return tensor_copy


# def moving_average_update(name, shape, tensor_update, rho=0.01, initializer=None, clip_values=None, dtype=tf.float32):
#     """ This function creates a tensor that will be updated by tensor_update using moving average
#
#     :param tensor_update: update at each iteration
#     :param name: name for the tensor
#     :param shape: shape of tensor
#     :param rho:
#     :param initializer:
#     :param clip_values:
#     :param dtype:
#     :return:
#     """
#     if initializer is None:
#         initializer = tf.zeros_initializer
#
#     tensor = tf.compat.v1.get_variable(
#         name, shape=shape, dtype=dtype, initializer=initializer, trainable=False)
#     if clip_values is None:
#         tf.compat.v1.add_to_collection(
#             tf.GraphKeys.UPDATE_OPS,
#             tf.assign(tensor, tensor + rho * tensor_update))
#     else:
#         tf.compat.v1.add_to_collection(
#             tf.GraphKeys.UPDATE_OPS,
#             tf.assign(
#                 tensor,
#                 tf.clip_by_value(
#                     tensor + rho * tensor_update,
#                     clip_value_min=clip_values[0], clip_value_max=clip_values[1])))
#
#     return tensor


# def get_tensor_name(tensor):
#     """ This function return tensor name without scope
#
#     :param tensor:
#     :return:
#     """
#     import re
#     # split 'scope/name:0' into [scope, name, 0]
#     return re.split('[/:]', tensor.name)[-2]


def mmd_g(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel

    The kernel is taken from following paper:
    Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
    MMD GAN: Towards Deeper Understanding of Moment Matching Network.

    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :param var_target: if sigma is trainable, var_target contain the target for sigma
    :param upper_bound: bounds for pairwise distance in mmd-g.
    :param lower_bound:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
    :type custom_weights: list
    :return:
    """
    with tf.name_scope(name):
        if lower_bound is None:
            k_xx = tf.exp(-dist_xx / (2.0 * sigma**2), name='k_xx')
            k_yy = tf.exp(-dist_yy / (2.0 * sigma ** 2), name='k_yy')
        else:
            k_xx = tf.exp(-tf.maximum(dist_xx, lower_bound) / (2.0 * sigma ** 2), name='k_xx_lb')
            k_yy = tf.exp(-tf.maximum(dist_yy, lower_bound) / (2.0 * sigma ** 2), name='k_yy_lb')
        if upper_bound is None:
            k_xy = tf.exp(-dist_xy / (2.0 * sigma**2), name='k_xy')
        else:
            k_xy = tf.exp(-tf.minimum(dist_xy, upper_bound) / (2.0 * sigma ** 2), name='k_xy_ub')

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        if var_target is None:
            if custom_weights is None:
                mmd = e_kxx + e_kyy - 2.0 * e_kxy
                return mmd
            else:  # note that here kyy is for the real data!
                assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
                mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
                mmd2 = custom_weights[0] * e_kxy - e_kxx - custom_weights[1] * e_kyy  # w[0] = 1, w[1]= -1, so kyy - kxx
                return mmd1, mmd2
        else:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


def mmd_t(
        dist_xx, dist_xy, dist_yy, batch_size, alpha=1.0, beta=2.0, var_target=None, name='mmd',
        do_summary=False, scope_prefix=''):
    """This function calculates the maximum mean discrepancy with t-distribution kernel

    The code is inspired by the Github page of following paper:
    Binkowski M., Sutherland D., Arbel M., Gretton A. (2018)
    Demystifying MMD GANs.

    :param dist_xx: batch_size-by-batch_size matrix
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param alpha:
    :param beta:
    :param var_target: if alpha is trainable, var_target contain the target for sigma
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """

    with tf.name_scope(name):
        log_k_xx = tf.log(dist_xx / (beta * alpha) + 1.0)  # use log for better condition
        log_k_xy = tf.log(dist_xy / (beta * alpha) + 1.0)
        log_k_yy = tf.log(dist_yy / (beta * alpha) + 1.0)

        k_xx = tf.exp(-alpha * log_k_xx)  # [1.0, k(xi, xj); k(xi, xj), 1.0]
        k_xy = tf.exp(-alpha * log_k_xy)
        k_yy = tf.exp(-alpha * log_k_yy)

        m = tf.constant(batch_size, tf.float32)
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

        mmd = e_kxx + e_kyy - 2.0 * e_kxy

        if do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx', e_kxx)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy', e_kyy)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy', e_kxy)

        # return e_kxx, e_kxy, e_kyy
        if var_target is None:
            return mmd
        else:
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = tf.square(var - var_target)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma


def matrix_mean_wo_diagonal(matrix, num_row, num_col=None, name='mu_wo_diag'):
    """ This function calculates the mean of the matrix elements not in the diagonal

    2018.4.9 - replace tf.diag_part with tf.matrix_diag_part
    tf.matrix_diag_part can be used for rectangle matrix while tf.diag_part can only be used for square matrix

    :param matrix:
    :param num_row:
    :type num_row: float
    :param num_col:
    :type num_col: float
    :param name:
    :return:
    """
    with tf.name_scope(name):
        if num_col is None:
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) / (num_row * (num_row - 1.0))
        else:
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.matrix_diag_part(matrix))) \
                 / (num_row * num_col - tf.minimum(num_col, num_row))

    return mu


# def newton_root(x, f, df, step=None):
#     """ This function does one iteration update on x to find the root f(x)=0. It is primarily used as the body of
#     tf.while_loop.
#
#     :param x:
#     :param f: a function that receives x as input and outputs f(x) and other info for gradient calculation
#     :param df: a function that receives info as inputs and outputs the gradient of f at x
#     :param step:
#     :return:
#     """
#     fx, info2grad = f(x)
#     gx = df(info2grad)
#     x = x - fx / (gx + FLAGS.EPSI)
#
#     if step is None:
#         return x
#     else:
#         return x, step + 1


# def mat_slice(mat, row_index, col_index=None, name='slice'):
#     """ This function gets mat[index, index] where index is either bool or int32.
#
#     Note that:
#         if index is bool, output size is typically smaller than mat unless each element in index is True
#         if index is int32, output can be any size.
#
#     :param mat:
#     :param row_index:
#     :param col_index:
#     :param name;
#     :return:
#     """
#     if col_index is None:
#         col_index = row_index
#
#     with tf.name_scope(name):
#         if row_index.dtype != col_index.dtype:
#             raise AttributeError('dtype of row-index and col-index do not match.')
#         if row_index.dtype == tf.int32:
#             return tf.gather(tf.gather(mat, row_index, axis=0), col_index, axis=1)
#         elif row_index.dtype == tf.bool:
#             return tf.boolean_mask(tf.boolean_mask(mat, row_index, axis=0), col_index, axis=1)
#         else:
#             raise AttributeError('Type of index is: {}; expected either tf.int32 or tf.bool'.format(row_index.dtype))


def get_squared_dist(
        x, y=None, scale=None, z_score=False, mode='xxxyyy', name='squared_dist',
        do_summary=False, scope_prefix=''):
    """ This function calculates the pairwise distance between x and x, x and y, y and y

    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead

    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :param scale: 1-by-d vector, the precision vector. dxy = x*scale*y
    :param z_score:
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :param do_summary:
    :param scope_prefix: summary scope prefix
    :return:
    """
    with tf.name_scope(name):
        # check inputs
        if len(x.get_shape().as_list()) > 2:
            raise AttributeError('get_dist: Input must be a matrix.')
        if y is None:
            mode = 'xx'
        if z_score:
            if y is None:
                mu = tf.reduce_mean(x, axis=0, keepdims=True)
                x = x - mu
            else:
                mu = tf.reduce_mean(tf.concat((x, y), axis=0), axis=0, keepdims=True)
                x = x - mu
                y = y - mu

        if mode in ['xx', 'xxxy', 'xxxyyy']:
            if scale is None:
                xxt = tf.matmul(x, x, transpose_b=True)  # [xi_xi, xi_xj; xj_xi, xj_xj], batch_size-by-batch_size
            else:
                xxt = tf.matmul(x * scale, x, transpose_b=True)
            dx = tf.diag_part(xxt)  # [xxt], [batch_size]
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xxt + tf.expand_dims(dx, axis=0), 0.0)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.histogram(scope_prefix + name + '/dxx', dist_xx)

            if mode == 'xx':
                return dist_xx
            elif mode == 'xxxy':  # estimate dy without yyt
                if scale is None:
                    xyt = tf.matmul(x, y, transpose_b=True)
                    dy = tf.reduce_sum(tf.multiply(y, y), axis=1)
                else:
                    xyt = tf.matmul(x * scale, y, transpose_b=True)
                    dy = tf.reduce_sum(tf.multiply(y * scale, y), axis=1)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
                if do_summary:
                    with tf.name_scope(None):  # return to root scope to avoid scope overlap
                        tf.compat.v1.summary.histogram(scope_prefix + name + '/dxy', dist_xy)

                return dist_xx, dist_xy
            elif mode == 'xxxyyy':
                if scale is None:
                    xyt = tf.matmul(x, y, transpose_b=True)
                    yyt = tf.matmul(y, y, transpose_b=True)
                else:
                    xyt = tf.matmul(x * scale, y, transpose_b=True)
                    yyt = tf.matmul(y * scale, y, transpose_b=True)
                dy = tf.diag_part(yyt)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
                dist_yy = tf.maximum(tf.expand_dims(dy, axis=1) - 2.0 * yyt + tf.expand_dims(dy, axis=0), 0.0)
                if do_summary:
                    with tf.name_scope(None):  # return to root scope to avoid scope overlap
                        tf.compat.v1.summary.histogram(scope_prefix + name + '/dxy', dist_xy)
                        tf.compat.v1.summary.histogram(scope_prefix + name + '/dyy', dist_yy)

                return dist_xx, dist_xy, dist_yy

        elif mode == 'xy':
            if scale is None:
                dx = tf.reduce_sum(tf.multiply(x, x), axis=1)
                dy = tf.reduce_sum(tf.multiply(y, y), axis=1)
                xyt = tf.matmul(x, y, transpose_b=True)
            else:
                dx = tf.reduce_sum(tf.multiply(x * scale, x), axis=1)
                dy = tf.reduce_sum(tf.multiply(y * scale, y), axis=1)
                xyt = tf.matmul(x * scale, y, transpose_b=True)
            dist_xy = tf.maximum(tf.expand_dims(dx, axis=1) - 2.0 * xyt + tf.expand_dims(dy, axis=0), 0.0)
            if do_summary:
                with tf.name_scope(None):  # return to root scope to avoid scope overlap
                    tf.compat.v1.summary.histogram(scope_prefix + name + '/dxy', dist_xy)

            return dist_xy
        else:
            raise AttributeError('Mode {} not supported'.format(mode))