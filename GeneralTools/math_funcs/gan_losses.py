import numpy as np
import tensorflow as tf
from dp_funcs.rff_mmd_loss import RFFKMap
from collections import namedtuple


class GANLoss(object):
    def __init__(self, rff_specs, enc_dims, do_summary=False):
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

        self.rff_map = RFFKMap(rff_specs.sigma, rff_specs.dims, enc_dims, rff_specs.const_noise, rff_specs.gen_loss)

    def _add_summary_(self):
        """ This function adds summaries

        :return:
        """
        if self.do_summary:
            with tf.name_scope(None):  # return to root scope to avoid scope overlap
                tf.compat.v1.summary.scalar('GANLoss/gen', self.loss_gen)
                tf.compat.v1.summary.scalar('GANLoss/dis', self.loss_dis)

    def _default_loss_summary_(self):
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

        self._default_loss_summary_()

    def _repulsive_mmd_g_mog_approx(self):
        """ repulsive loss

        :return:
        """
        # DIS LOSS AS USUAL
        dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, self.score_data, do_summary=self.do_summary)
        _, self.loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)

        # GEN LOSS WITH MOG SAMPLES
        dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, self.score_mog, do_summary=self.do_summary)
        self.loss_gen, _ = mmd_g(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
            name='mmd_g_mog', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)

        self._default_loss_summary_()

    # def _repulsive_mmd_g_inv_disc(self):
    #     """ repulsive loss with inverted discriminator loss so sample encodings can better be modeled by MoG
    #     never quite figured out why this fails as badly as it does
    #     :return:
    #     """
    #     # calculate pairwise distance
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, self.score_data,
    #                                                  z_score=False, do_summary=self.do_summary)
    #     self.loss_gen, self.loss_dis = mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
    #         name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
    #
    #     self.loss_dis = -self.loss_dis  # THAT'S ALL THAT CHANGES!
    #     self._default_loss_summary_()

    # def _dp_repulsive_mmd_g_(self):
    #     """ repulsive loss
    #     :return:
    #     """
    #     # DIS LOSS AS USUAL
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
    #     _, self.loss_dis = mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
    #         name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
    #
    #     # GEN LOSS WITH MOG SAMPLES
    #     dist_gg, dist_gd, dist_dd = get_squared_dist(
    #         self.score_gen, self.score_mog, z_score=False, do_summary=self.do_summary)
    #     self.loss_gen, _ = mmd_g(
    #         dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0,
    #         name='mmd_g_mog', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
    #
    #     self._default_loss_summary_()

    def _rff_gaussian_kernel_approx(self):
        """
        random fourier feature approximation of MMD for discriminator (i.e. attractive loss)
        """
        # dis loss
        rff_gen = self.rff_map.gen_features(self.score_gen)  # (bs, d_rff)
        rff_dat = self.rff_map.gen_features(self.score_data)  # (bs, d_rff)
        rffk_gen = tf.compat.v1.reduce_mean(rff_gen, axis=0)  # (d_rff)
        rffk_dat = tf.compat.v1.reduce_mean(rff_dat, axis=0)  # (d_rff)
        self.loss_dis = -tf.compat.v1.reduce_sum((rffk_dat - rffk_gen) ** 2, name='rff_mmd_g')  # ()

        # gen loss
        if self.rff_map.gen_loss == 'rff':
            self.loss_gen = -self.loss_dis
        else:
            assert self.rff_map.gen_loss in {'data', 'mog'}
            comp_data = self.score_data if self.rff_map.gen_loss == 'data' else self.score_mog
            name = 'mmd_g' if self.rff_map.gen_loss == 'data' else 'mmd_g_mog'
            dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, comp_data, do_summary=self.do_summary)
            self.loss_gen, _ = mmd_g(dist_gg, dist_gd, dist_dd, self.batch_size, name=name,
                                     do_summary=self.do_summary, custom_weights=self.repulsive_weights)

    def _dp_rff_gaussian_kernel_approx(self):
        """
        random fourier feature approximation of MMD for discriminator (i.e. attractive loss) in DP setting
        provides the n+1 losses used to compute per-sample-clipped gradients
        """
        # dis loss
        rff_gen = self.rff_map.gen_features(self.score_gen)  # (bs, d_rff)
        rff_dat = self.rff_map.gen_features(self.score_data)  # (bs, d_rff)
        rffk_gen = tf.compat.v1.reduce_mean(rff_gen, axis=0)  # (d_rff)
        self.loss_dis = namedtuple('rff_loss', ['fdat', 'fgen'])(rff_dat, rffk_gen)

        # gen loss
        if self.rff_map.gen_loss == 'rff':
            rffk_dat = tf.compat.v1.reduce_mean(rff_dat, axis=0)
            self.loss_gen = tf.compat.v1.reduce_sum((rffk_dat - rffk_gen) ** 2, name='rff_mmd_g')
        else:
            assert self.rff_map.gen_loss in {'data', 'mog'}
            comp_data = self.score_data if self.rff_map.gen_loss == 'data' else self.score_mog
            name = 'mmd_g' if self.rff_map.gen_loss == 'data' else 'mmd_g_mog'
            dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, comp_data, do_summary=self.do_summary)
            self.loss_gen, _ = mmd_g(dist_gg, dist_gd, dist_dd, self.batch_size, name=name,
                                     do_summary=self.do_summary, custom_weights=self.repulsive_weights)

    def _repulsive_mmd_g_bounded_(self):
        """ rmb loss

        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(self.score_gen, self.score_data, do_summary=self.do_summary)
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

    def __call__(self, score_gen, score_data, score_mog=None, loss_type='logistic', **kwargs):
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
        self.score_mog = score_mog
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
        if loss_type in {'fixed_g', 'mmd_g', 'fixed_t', 'mmd_t', 'mmd_g_mix', 'fixed_g_mix', 'rand_g', 'rand_g_mix',
                         'sym_rg_mix', 'instance_noise', 'ins_noise', 'sym_rg', 'rgb', 'rep', 'rep_gp', 'rmb', 'rmb_gp',
                         'rep_inv_disc'}:
            assert self.batch_size is not None, 'GANLoss: batch_size must be provided'
            if loss_type in {'rand_g', 'rand_g_mix', 'sym_rg_mix', 'sym_rg'}:
                assert self.num_scores is not None, 'GANLoss: d must be provided'
        if loss_type in {'rep_gp', 'rmb_gp', 'wasserstein'}:
            assert self.dis_penalty is not None, 'Discriminator penalty must be provided.'
        if loss_type in {'rep_ds', 'rmb_ds'}:
            assert self.dis_scale is not None, 'Discriminator loss scale must be provided.'

        if loss_type in {'rep', 'rep_mmd_g', 'rep_gp', 'rep_ds'}:
            if self.score_mog is None:
                self._repulsive_mmd_g_()
            else:
                self._repulsive_mmd_g_mog_approx()
        elif loss_type in {'rmb', 'rep_b', 'rep_mmd_b', 'rmb_gp', 'rmb_ds'}:
            self._repulsive_mmd_g_bounded_()
        elif loss_type == 'test':
            self._test_()
        elif loss_type == 'rff':
            self._rff_gaussian_kernel_approx()
        elif loss_type == 'dp_rff':
            self._dp_rff_gaussian_kernel_approx()
        # elif loss_type == 'rep_inv_disc':
        #     self._repulsive_mmd_g_inv_disc()
        # elif isinstance(loss_type, dict):
        #     assert 'type' in loss_type.keys()
        #     if loss_type['type'] == 'direct_gmm_loss':
        #         print('direct_gmm_loss queried')
        #         self._repulsive_mmd_g_with_gmm(loss_type)
        else:
            raise NotImplementedError('Not implemented.')

        self._add_summary_()

        return self.loss_gen, self.loss_dis

    def apply(self, score_gen, score_data, score_mog=False, loss_type='logistic', **kwargs):
        return self.__call__(score_gen, score_data, score_mog=score_mog, loss_type=loss_type, **kwargs)

    def get_register(self):
        """ This function returns the registered tensor

        :return:
        """
        # loss object always forgets self.debug_register after its value returned
        registered_info = self.debug_register
        self.debug_register = None
        return registered_info


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
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxx_gen', e_kxx)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kyy_dat', e_kyy)
                tf.compat.v1.summary.scalar(scope_prefix + name + '/kxy_mix', e_kxy)

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
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.linalg.diag_part(matrix))) / (num_row * (num_row - 1.0))
        else:
            mu = (tf.reduce_sum(matrix) - tf.reduce_sum(tf.linalg.diag_part(matrix))) \
                 / (num_row * num_col - tf.minimum(num_col, num_row))

    return mu


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
            dx = tf.linalg.tensor_diag_part(xxt)  # [xxt], [batch_size]
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
                dy = tf.linalg.tensor_diag_part(yyt)
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
