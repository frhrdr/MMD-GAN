import numpy as np
import tensorflow as tf


def spatial_shape_after_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after conv layer.

    The formula is obtained from: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
    It should be note that current function assumes PS is done before conv

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
        if padding in ['same', 'SAME']:
            return np.int(np.ceil(input_spatial_shape / strides))
        else:
            return np.int(np.ceil((input_spatial_shape - (kernel_size - 1) * dilation) / strides))


def spatial_shape_after_transpose_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after conv layer.

    Since transpose conv is often used in upsampling, scale_factor is not used here.

    This function has not been fully tested, and may be wrong in some cases.

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_transpose_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
        if padding in ['same', 'SAME']:
            return np.int(input_spatial_shape * strides)
        else:
            return np.int(input_spatial_shape * strides + (kernel_size - 1) * dilation)


def get_batch_squared_dist(x_batch, y_batch=None, axis=1, mode='xx', name='squared_dist'):
    """ This function calculates squared pairwise distance for vectors under xi or between xi and yi
    where i refers to the samples in the batch

    :param x_batch: batch_size-a-b tensor
    :param y_batch: batch_size-c-d tensor
    :param axis: the axis to be considered as features; if axis==1, a=c; if axis=2, b=d
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :return: dist tensor(s)
    """
    # check inputs
    assert axis in [1, 2], 'axis has to be 1 or 2.'
    batch, a, b = x_batch.get_shape().as_list()
    if y_batch is not None:
        batch_y, c, d = y_batch.get_shape().as_list()
        assert batch == batch_y, 'Batch sizes do not match.'
        if axis == 1:
            assert a == c, 'Feature sizes do not match.'
        elif axis == 2:
            assert b == d, 'Feature sizes do not match.'
        if mode == 'xx':
            mode = 'xy'

    with tf.name_scope(name):
        if mode in {'xx', 'xxxyyy', 'xxxy'}:
            # xxt is batch-a-a if axis is 2 else batch-b-b
            xxt = tf.matmul(x_batch, tf.transpose(x_batch, [0, 2, 1])) \
                if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), x_batch)
            # dx is batch-a if axis is 2 else batch-b
            dx = tf.matrix_diag_part(xxt)
            dist_xx = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xxt + tf.expand_dims(dx, axis=1), 0.0)
            if mode == 'xx':
                return dist_xx
            elif mode == 'xxxy':
                # xyt is batch-a-c if axis is 2 else batch-b-d
                xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
                # dy is batch-c if axis is 2 else batch-d
                dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)

                return dist_xx, dist_xy
            elif mode == 'xxxyyy':
                # xyt is batch-a-c if axis is 2 else batch-b-d
                xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
                # yyt is batch-c-c if axis is 2 else batch-d-d
                yyt = tf.matmul(y_batch, tf.transpose(y_batch, [0, 2, 1])) \
                    if axis == 2 else tf.matmul(tf.transpose(y_batch, [0, 2, 1]), y_batch)
                # dy is batch-c if axis is 2 else batch-d
                dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
                dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)
                dist_yy = tf.maximum(tf.expand_dims(dy, axis=2) - 2.0 * yyt + tf.expand_dims(dy, axis=1), 0.0)

                return dist_xx, dist_xy, dist_yy

        elif mode == 'xy':
            # dx is batch-a if axis is 2 else batch-b
            dx = tf.reduce_sum(tf.multiply(x_batch, x_batch), axis=axis)
            # dy is batch-c if axis is 2 else batch-d
            dy = tf.reduce_sum(tf.multiply(y_batch, y_batch), axis=axis)
            # xyt is batch-a-c if axis is 2 else batch-b-d
            xyt = tf.matmul(x_batch, tf.transpose(y_batch, [0, 2, 1])) \
                if axis == 2 else tf.matmul(tf.transpose(x_batch, [0, 2, 1]), y_batch)
            dist_xy = tf.maximum(tf.expand_dims(dx, axis=2) - 2.0 * xyt + tf.expand_dims(dy, axis=1), 0.0)

            return dist_xy
        else:
            raise AttributeError('Mode {} not supported'.format(mode))