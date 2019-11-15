import warnings

import numpy as np
import tensorflow as tf

from GeneralTools.misc_fun import FLAGS


class SpectralNorm(object):
    def __init__(self, sn_def, name_scope='SN', scope_prefix='', num_iter=1):
        """ This class contains functions to calculate the spectral normalization of the weight matrix
        using power iteration.

        The application of spectral normal to NN is proposed in following papers:
        Yoshida, Y., & Miyato, T. (2017).
        Spectral Norm Regularization for Improving the Generalizability of Deep Learning.
        Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2017).
        Spectral Normalization for Generative Adversarial Networks,
        Here spectral normalization is generalized for any linear ops or combination of linear ops

        Example of usage:
        Example 1.
        w = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        sn_def = {'op': 'tc', 'input_shape': [10, 64, 64, 64],
                  'output_shape': [10, 128, 64, 64],
                  'strides': 1, 'dilation': 1, 'padding': 'SAME',
                  'data_format': 'NCHW'}
        sigma = SpectralNorm(sn_def, name_scope='SN1', num_iter=20).apply(w)

        Example 2.
        w = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        w2 = tf.constant(np.random.randn(3, 3, 128, 64).astype(np.float32))
        sn_def = {'op': 'tc', 'input_shape': [10, 64, 64, 64],
                  'output_shape': [10, 128, 64, 64],
                  'strides': 1, 'dilation': 1, 'padding': 'SAME',
                  'data_format': 'NCHW'}

        SN = SpectralNorm(sn_def, num_iter=20)
        sigma1 = SN.apply(w)
        sigma2 = SN.apply(w2, name_scope='SN2', num_iter=30)


        :param sn_def: a dictionary with keys depending on the type of kernel:
            type     keys   value options
            dense:    'op'    'd' - common dense layer; 'cd' - conditional dense layers;
                            'dcd' - dense + conditional dense; 'dck' - dense * conditional scale
                            'project' - same to cd, except num_out is 1
            conv:    'op'    'c' - convolution; 'tc' - transpose convolution;
                            'cck' - convolution * conditional scale; 'tcck' - t-conv * conditional scale
                     'strides'    integer
                     'dilation'    integer
                     'padding'    'SAME' or 'VALID'
                     'data_format'    'NCHW' or 'NHWC'
                     'input_shape'    list of integers in format NCHW or NHWC
                     'output_shape'    for 'tc', output shape must be provided
        :param name_scope:
        :param scope_prefix:
        :param num_iter: number of power iterations per run
        """
        self.sn_def = sn_def.copy()
        self.name_scope = name_scope
        self.scope_prefix = scope_prefix
        self.name_in_err = self.scope_prefix + self.name_scope
        self.num_iter = num_iter
        # initialize
        self.w = None
        self.x = None
        self.use_u = None
        self.is_initialized = False
        self.forward = None
        self.backward = None

        # format stride
        if self.sn_def['op'] in {'c', 'tc', 'cck', 'tcck'}:
            if self.sn_def['data_format'] in ['NCHW', 'channels_first']:
                self.sn_def['strides'] = (1, 1, self.sn_def['strides'], self.sn_def['strides'])
            else:
                self.sn_def['strides'] = (1, self.sn_def['strides'], self.sn_def['strides'], 1)
            assert 'output_shape' in self.sn_def, \
                '{}: for conv, output_shape must be provided.'.format(self.name_in_err)

    def _init_routine(self):
        """ This function decides the routine to minimize memory usage

        :return:
        """
        if self.is_initialized is False:
            # decide the routine
            if self.sn_def['op'] in {'d', 'project'}:
                # for d kernel_shape [num_in, num_out]; for project, kernel shape [num_class, num_in]
                assert len(self.kernel_shape) == 2, \
                    '{}: kernel shape {} does not have length 2'.format(self.name_in_err, self.kernel_shape)
                num_in, num_out = self.kernel_shape
                # self.use_u = True
                self.use_u = True if num_in <= num_out else False
                x_shape = [1, num_in] if self.use_u else [1, num_out]
                self.forward = self._dense_ if self.use_u else self._dense_t_
                self.backward = self._dense_t_ if self.use_u else self._dense_
            elif self.sn_def['op'] in {'cd'}:  # kernel_shape [num_class, num_in, num_out]
                assert len(self.kernel_shape) == 3, \
                    '{}: kernel shape {} does not have length 3'.format(self.name_in_err, self.kernel_shape)
                num_class, num_in, num_out = self.kernel_shape
                self.use_u = True if num_in <= num_out else False
                x_shape = [num_class, 1, num_in] if self.use_u else [num_class, 1, num_out]
                self.forward = self._dense_ if self.use_u else self._dense_t_
                self.backward = self._dense_t_ if self.use_u else self._dense_
            elif self.sn_def['op'] in {'dck'}:  # convolution * conditional scale
                assert isinstance(self.kernel_shape, (list, tuple)) and len(self.kernel_shape) == 2, \
                    '{}: kernel shape must be a list of length 2. Got {}'.format(self.name_in_err, self.kernel_shape)
                assert len(self.kernel_shape[0]) == 2 and len(self.kernel_shape[1]) == 2, \
                    '{}: kernel shape {} does not have length 2'.format(self.name_in_err, self.kernel_shape)
                num_in, num_out = self.kernel_shape[0]
                num_class = self.kernel_shape[1][0]
                self.use_u = True if num_in <= num_out else False
                x_shape = [num_class, num_in] if self.use_u else [num_class, num_out]
                self.forward = (lambda x: self._scalar_(self._dense_(x, index=0), index=1, offset=1.0)) \
                    if self.use_u else (lambda y: self._dense_t_(self._scalar_(y, index=1, offset=1.0), index=0))
                self.backward = (lambda y: self._dense_t_(self._scalar_(y, index=1, offset=1.0), index=0)) \
                    if self.use_u else (lambda x: self._scalar_(self._dense_(x, index=0), index=1, offset=1.0))
            elif self.sn_def['op'] in {'c', 'tc'}:
                assert len(self.kernel_shape) == 4, \
                    '{}: kernel shape {} does not have length 4'.format(self.name_in_err, self.kernel_shape)
                # self.use_u = True
                self.use_u = True \
                    if np.prod(self.sn_def['input_shape'][1:]) <= np.prod(self.sn_def['output_shape'][1:]) \
                    else False
                if self.sn_def['op'] in {'c'}:  # input / output shape NCHW or NHWC
                    x_shape = self.sn_def['input_shape'].copy() if self.use_u else self.sn_def['output_shape'].copy()
                    x_shape[0] = 1
                    y_shape = self.sn_def['input_shape'].copy()
                    y_shape[0] = 1
                elif self.sn_def['op'] in {'tc'}:  # tc
                    x_shape = self.sn_def['output_shape'].copy() if self.use_u else self.sn_def['input_shape'].copy()
                    x_shape[0] = 1
                    y_shape = self.sn_def['output_shape'].copy()
                    y_shape[0] = 1
                else:
                    raise NotImplementedError('{}: {} not implemented.'.format(self.name_in_err, self.sn_def['op']))
                self.forward = self._conv_ if self.use_u else (lambda y: self._conv_t_(y, x_shape=y_shape))
                self.backward = (lambda y: self._conv_t_(y, x_shape=y_shape)) if self.use_u else self._conv_
            elif self.sn_def['op'] in {'cck', 'tcck'}:  # convolution * conditional scale
                assert isinstance(self.kernel_shape, (list, tuple)) and len(self.kernel_shape) == 2, \
                    '{}: kernel shape must be a list of length 2. Got {}'.format(self.name_in_err, self.kernel_shape)
                assert len(self.kernel_shape[0]) == 4 and len(self.kernel_shape[1]) == 4, \
                    '{}: kernel shape {} does not have length 4'.format(self.name_in_err, self.kernel_shape)
                self.use_u = True \
                    if np.prod(self.sn_def['input_shape'][1:]) <= np.prod(self.sn_def['output_shape'][1:]) \
                    else False
                num_class = self.kernel_shape[1][0]
                if self.sn_def['op'] in {'cck'}:  # input / output shape NCHW or NHWC
                    x_shape = self.sn_def['input_shape'].copy() if self.use_u else self.sn_def['output_shape'].copy()
                    x_shape[0] = num_class
                    y_shape = self.sn_def['input_shape'].copy()
                    y_shape[0] = num_class
                    self.forward = (lambda x: self._scalar_(self._conv_(x, index=0), index=1, offset=1.0)) \
                        if self.use_u \
                        else (lambda y: self._conv_t_(self._scalar_(y, index=1, offset=1.0), x_shape=y_shape, index=0))
                    self.backward = (lambda y: self._conv_t_(
                        self._scalar_(y, index=1, offset=1.0), x_shape=y_shape, index=0)) \
                        if self.use_u else (lambda x: self._scalar_(self._conv_(x, index=0), index=1, offset=1.0))
                elif self.sn_def['op'] in {'tcck'}:  # tcck
                    x_shape = self.sn_def['output_shape'].copy() if self.use_u else self.sn_def['input_shape'].copy()
                    x_shape[0] = num_class
                    y_shape = self.sn_def['output_shape'].copy()
                    y_shape[0] = num_class
                    self.forward = (lambda x: self._conv_(self._scalar_(x, index=1, offset=1.0), index=0)) \
                        if self.use_u \
                        else (lambda y: self._scalar_(self._conv_t_(y, x_shape=y_shape, index=0), index=1, offset=1.0))
                    self.backward = (lambda y: self._scalar_(
                        self._conv_t_(y, x_shape=y_shape, index=0), index=1, offset=1.0)) \
                        if self.use_u else (lambda x: self._conv_(self._scalar_(x, index=1, offset=1.0), index=0))
                else:
                    raise NotImplementedError('{}: {} not implemented.'.format(self.name_in_err, self.sn_def['op']))
            else:
                raise NotImplementedError('{}: {} is not implemented.'.format(self.name_in_err, self.sn_def['op']))

            self.x = tf.compat.v1.get_variable(
                'in_rand', shape=x_shape, dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(), trainable=False)

            self.is_initialized = True

    def _scalar_(self, x, index=None, offset=0.0):
        """ This function defines a elementwise multiplication op: y = x * w, where x shape [N, C, ...] or [N, ..., C],
        w shape [N, C, 1,..,1] or [N, 1,...,1, C], y shape [N, C, ...] or [N, ..., C]

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :param offset: add a constant offset
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.multiply(x, w, name='scalar') if offset == 0.0 else tf.multiply(x, w + offset, name='scalar')

    def _dense_(self, x, index=None):
        """ This function defines a dense op: y = x * w, where x shape [..., a, b], w shape [..., b, c],
        y shape [..., a, c]

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.matmul(x, w, name='dense')

    def _dense_t_(self, y, index=None):
        """ Transpose version of self._dense_

        :param y:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        return tf.matmul(y, w, transpose_b=True, name='dense_t')

    def _conv_(self, x, index=None):
        """ This function defines a conv op: y = x \otimes w, where x shape NCHW or NHWC, w shape kkhw,
        y shape NCHW or NHWC

        :param x:
        :param index: if index is provided, self.w is a list or tuple
        :return:
        """
        w = self.w if index is None else self.w[index]
        if self.sn_def['dilation'] > 1:
            return tf.nn.atrous_conv2d(
                x, w, rate=self.sn_def['dilation'], padding=self.sn_def['padding'], name='conv')
        else:
            return tf.nn.conv2d(
                x, w, strides=self.sn_def['strides'], padding=self.sn_def['padding'],
                data_format=self.sn_def['data_format'], name='conv')

    def _conv_t_(self, y, x_shape, index=None):
        """ Transpose version of self._conv_

        :param y:
        :param x_shape:
        :param index:
        :return:
        """
        w = self.w if index is None else self.w[index]
        if self.sn_def['dilation'] > 1:
            return tf.nn.atrous_conv2d_transpose(
                y, w, output_shape=x_shape, rate=self.sn_def['dilation'], padding=self.sn_def['padding'],
                name='conv_t')
        else:
            return tf.nn.conv2d_transpose(
                y, w, output_shape=x_shape, strides=self.sn_def['strides'], padding=self.sn_def['padding'],
                data_format=self.sn_def['data_format'], name='conv_t')

    def _l2_norm(self, x):
        if self.sn_def['op'] in {'cd'}:  # x shape [num_class, 1, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=2, keepdims=True)  # return [num_class, 1, 1]
        elif self.sn_def['op'] in {'dck'}:  # x shape [num_class, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=1, keepdims=True)  # return [num_class, 1]
        elif self.sn_def['op'] in {'cck', 'tcck'}:
            # x shape [num_class, num_in or num_out, H, W] or [num_class, H, W, num_in or num_out]
            # here i did not use tf.norm because axis cannot be (1, 2, 3)
            return tf.sqrt(
                tf.reduce_sum(tf.square(x), axis=(1, 2, 3), keepdims=True), name='norm')  # return [num_class, 1, 1, 1]
        elif self.sn_def['op'] in {'d', 'c', 'tc', 'project'}:
            # x shape [1, num_in or num_out], or [1, num_in or num_out, H, W] or [1, H, W, num_in or num_out]
            return tf.norm(x, ord='euclidean', axis=None)  # return scalar

    def _l2_normalize_(self, w):
        """

        :param w:
        :return:
        """
        return w / (self._l2_norm(w) + FLAGS.EPSI)

    def _power_iter_(self, x, step):
        """ This function does power iteration for one step

        :param x:
        :param step:
        :return:
        """
        y = self._l2_normalize_(self.forward(x))
        x_update = self._l2_normalize_(self.backward(y))
        sigma = self._l2_norm(self.forward(x))

        return sigma, x_update, step + 1

    def __call__(self, kernel, **kwargs):
        """ This function calculates spectral normalization for kernel

        :param kernel:
        :param kwargs:
        :return:
        """
        # check inputs
        if 'name_scope' in kwargs and kwargs['name_scope'] != self.name_scope:
            # different name_scope will initialize another SN process
            self.name_scope = kwargs['name_scope']
            self.name_in_err = self.scope_prefix + self.name_scope
            if self.is_initialized:
                warnings.warn(
                    '{}: a new SN process caused lost of links to the previous one.'.format(self.name_in_err))
                self.is_initialized = False
            self.use_u = None
        if 'num_iter' in kwargs:
            self.num_iter = kwargs['num_iter']
        if isinstance(kernel, (list, tuple)):
            # for dcd, cck, the kernel is a list of two kernels
            kernel_shape = [k.get_shape().as_list() for k in kernel]
        else:
            kernel_shape = kernel.get_shape().as_list()

        with tf.compat.v1.variable_scope(self.name_scope, reuse=tf.compat.v1.AUTO_REUSE):
            # In some cases, the spectral norm can be easily calculated.
            sigma = None
            if self.sn_def['op'] in {'d', 'project'} and 1 in kernel_shape:
                # for project op. kernel_shape = [num_class, num_in]
                sigma = tf.norm(kernel, ord='euclidean')
            elif self.sn_def['op'] in {'cd'}:
                if len(kernel_shape) == 2:  # equivalent to [num_class, num_in, 1]
                    sigma = tf.norm(kernel, ord='euclidean', axis=1, keepdims=True)
                elif kernel_shape[1] == 1 or kernel_shape[2] == 1:
                    sigma = tf.norm(kernel, ord='euclidean', axis=(1, 2), keepdims=True)
            elif self.sn_def['op'] in {'dcd'}:  # dense + conditional dense
                # kernel_cd [num_class, num_in, num_out]
                kernel_cd = tf.expand_dims(kernel[1], axis=2) if len(kernel_shape[1]) == 2 else kernel[1]
                kernel = tf.expand_dims(kernel[0], axis=0) + kernel_cd  # [num_class, num_in, num_out]
                if 1 in kernel_shape[0]:  # kernel_d shape [1, num_out] or [num_in, 1]
                    sigma = tf.norm(kernel, ord='euclidean', axis=(1, 2), keepdims=True)  # [num_class, 1, 1]
                else:  # convert dcd to cd
                    kernel_shape = kernel.get_shape().as_list()
                    self.sn_def['op'] = 'cd'
            elif self.sn_def['op'] in {'dck'}:  # dense * conditional scales
                if kernel_shape[0][1] == 1:
                    sigma = tf.norm(kernel[0], ord='euclidean') * tf.abs(kernel[1])  # [num_class, 1]

            # initialize a random input and calculate spectral norm
            if sigma is None:
                # decide the routine
                self.w = kernel
                self.kernel_shape = kernel_shape
                self._init_routine()
                # initialize sigma
                if self.sn_def['op'] in {'dck'}:
                    sigma_init = tf.zeros((self.kernel_shape[1][0], 1), dtype=tf.float32)
                elif self.sn_def['op'] in {'cd'}:  # for cd, the sigma is a [num_class, 1, 1]
                    sigma_init = tf.zeros((self.kernel_shape[0], 1, 1), dtype=tf.float32)
                elif self.sn_def['op'] in {'cck', 'tcck'}:
                    sigma_init = tf.zeros((self.kernel_shape[1][0], 1, 1, 1), dtype=tf.float32)
                else:
                    sigma_init = tf.constant(0.0, dtype=tf.float32)
                # do power iterations
                sigma, x_update, _ = tf.while_loop(
                    cond=lambda _1, _2, i: i < self.num_iter,
                    body=lambda _1, x, i: self._power_iter_(x, step=i),
                    loop_vars=(sigma_init, self.x, tf.constant(0, dtype=tf.int32)),
                    name='spectral_norm_while')
                # update the random input
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, tf.compat.v1.assign(self.x, x_update))

        return sigma

    def apply(self, kernel, **kwargs):
        return self.__call__(kernel, **kwargs)