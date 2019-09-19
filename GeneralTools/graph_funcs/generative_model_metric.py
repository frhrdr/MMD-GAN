import numpy as np
import tensorflow as tf
from tensorflow.contrib import gan as tfgan

from GeneralTools.graph_funcs.my_session import MySession
from GeneralTools.math_funcs.graph_func_support import mean_cov_np, trace_sqrt_product_np
from GeneralTools.misc_fun import FLAGS


class GenerativeModelMetric(object):
    def __init__(self, image_format=None, model='v1', model_path=None):
        """ This class defines several metrics using pre-trained classifier inception v1.

        :param image_format:
        """
        if model_path is None:
            self.model = model
            if model == 'v1':
                self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(FLAGS.INCEPTION_V1)
            elif model == 'v3':
                self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(FLAGS.INCEPTION_V3)
            elif model in {'swd', 'ms_ssim', 'ssim'}:
                pass
            else:
                raise NotImplementedError('Model {} not implemented.'.format(model))
        else:
            self.model = 'custom'
            self.inception_graph_def = tfgan.eval.get_graph_def_from_disk(model_path)
        if image_format is None:
            self.image_format = FLAGS.IMAGE_FORMAT
        else:
            self.image_format = image_format

        # preserved for inception v3
        self._pool3_v3_ = None
        self._logits_v3_ = None

    def inception_v1_one_batch(self, image, output_tensor=None):
        """ This function runs the inception v1 model on images and give logits output.

        Note: if other layers of inception model is needed, change the output_tensor option in tfgan.eval.run_inception

        :param image:
        :param output_tensor:
        :return:
        """
        if output_tensor is None:
            output_tensor = ['logits:0', 'pool_3:0']

        image_size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        if self.image_format in {'channels_first', 'NCHW'}:
            image = tf.transpose(image, perm=(0, 2, 3, 1))
        if image.get_shape().as_list()[1] != image_size:
            image = tf.image.resize_bilinear(image, [image_size, image_size])

        # inception score uses the logits:0 while FID uses pool_3:0.
        return tfgan.eval.run_inception(
            image, graph_def=self.inception_graph_def, input_tensor='Mul:0', output_tensor=output_tensor)

    def inception_v1(self, images):
        """ This function runs the inception v1 model on images and give logits output.

        Note: if other layers of inception model is needed, change the output_tensor option in tfgan.eval.run_inception.
        Note: for large inputs, e.g. [10000, 64, 64, 3], it is better to run iterations containing this function.

        :param images:
        :return:
        """
        num_images = images.get_shape().as_list()[0]
        if num_images > 2500:
            raise MemoryError('The input is too big to possibly fit into memory. Consider using multiple runs.')
        if num_images >= 400:
            # Note: need to validate the code below

            # somehow tfgan.eval.classifier_score does not work properly when splitting the datasets.
            # The following code is inspired by:
            # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
            if num_images % 100 == 0:
                generated_images_list = tf.split(images, num_or_size_splits=num_images // 100, axis=0)
                logits, pool3 = tf.map_fn(
                    fn=self.inception_v1_one_batch,
                    elems=tf.stack(generated_images_list),
                    dtype=(tf.float32, tf.float32),
                    parallel_iterations=1,
                    back_prop=False,
                    swap_memory=True,
                    name='RunClassifier')
                logits = tf.concat(tf.unstack(logits), 0)
                pool3 = tf.concat(tf.unstack(pool3), 0)
            else:
                generated_images_list = tf.split(
                    images, num_or_size_splits=[100] * (num_images // 100) + [num_images % 100], axis=0)
                # tf.stack requires the dimension of tensor in list to be the same
                logits, pool3 = tf.map_fn(
                    fn=self.inception_v1_one_batch,
                    elems=tf.stack(generated_images_list[0:-1]),
                    dtype=(tf.float32, tf.float32),
                    parallel_iterations=1,
                    back_prop=False,
                    swap_memory=True,
                    name='RunClassifier')
                logits_last, pool3_last = self.inception_v1_one_batch(generated_images_list[-1])
                logits = tf.concat(tf.unstack(logits) + [logits_last], 0)
                pool3 = tf.concat(tf.unstack(pool3) + [pool3_last], 0)
        else:
            logits, pool3 = self.inception_v1_one_batch(images)

        return logits, pool3

    @staticmethod
    def inception_score_from_logits(logits):
        """ This function estimates the inception score from logits output by inception_v1

        :param logits:
        :return:
        """
        if type(logits) == np.ndarray:
            logits = tf.constant(logits, dtype=tf.float32)
        return tfgan.eval.classifier_score_from_logits(logits)

    @staticmethod
    def fid_from_pool3(x_pool3, y_pool3):
        """ This function estimates Fréchet inception distance from pool3 of inception model

        :param x_pool3:
        :param y_pool3:
        :return:
        """
        if type(x_pool3) == np.ndarray:
            x_pool3 = tf.constant(x_pool3, dtype=tf.float32)
        if type(y_pool3) == np.ndarray:
            y_pool3 = tf.constant(y_pool3, dtype=tf.float32)
        return tfgan.eval.frechet_classifier_distance_from_activations(x_pool3, y_pool3)

    @ staticmethod
    def my_fid_from_pool3(x_pool3_np, y_pool3_np):
        """ This function estimates Fréchet inception distance from pool3 of inception model.
        Different from fid_from_pool3, here pool3_np could be a list [mean, cov]

        :param x_pool3_np:
        :param y_pool3_np:
        :return:
        """
        # from scipy.linalg import sqrtm
        x_mean, x_cov = x_pool3_np if isinstance(x_pool3_np, (list, tuple)) else mean_cov_np(x_pool3_np)
        y_mean, y_cov = y_pool3_np if isinstance(y_pool3_np, (list, tuple)) else mean_cov_np(y_pool3_np)
        fid = np.sum((x_mean-y_mean) ** 2)+np.trace(x_cov)+np.trace(y_cov)-2.0*trace_sqrt_product_np(x_cov, y_cov)
        return fid
        # return np.sum((x_mean - y_mean) ** 2) + np.trace(x_cov + y_cov - 2.0 * sqrtm(np.dot(x_cov, y_cov)))

    def inception_score_and_fid_v1(self, x_batch, y_batch, num_batch=10, ckpt_folder=None, ckpt_file=None):
        """ This function calculates inception scores and FID based on inception v1.
        Note: batch_size * num_batch needs to be larger than 2048, otherwise the convariance matrix will be
        ill-conditioned.

        According to TensorFlow v1.7 (below), this is actually inception v3 model.
        Somehow the downloaded file says it's v1.
        code link: https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib \
        /gan/python/eval/python/classifier_metrics_impl.py

        Steps:
        1, the pool3 and logits are calculated for x_batch and y_batch with sess
        2, the pool3 and logits are passed to corresponding metrics

        :param ckpt_file:
        :param x_batch: tensor, one batch of x in range [-1, 1]
        :param y_batch: tensor, one batch of y in range [-1, 1]
        :param num_batch:
        :param ckpt_folder: check point folder
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        assert self.model == 'v1', 'GenerativeModelMetric is not initialized with model="v1".'
        assert ckpt_folder is not None, 'ckpt_folder must be provided.'

        x_logits, x_pool3 = self.inception_v1(x_batch)
        y_logits, y_pool3 = self.inception_v1(y_batch)

        with MySession(load_ckpt=True) as sess:
            inception_outputs = sess.run_m_times(
                [x_logits, y_logits, x_pool3, y_pool3],
                ckpt_folder=ckpt_folder, ckpt_file=ckpt_file,
                max_iter=num_batch, trace=True)

        # get logits and pool3
        x_logits_np = np.concatenate([inc[0] for inc in inception_outputs], axis=0)
        y_logits_np = np.concatenate([inc[1] for inc in inception_outputs], axis=0)
        x_pool3_np = np.concatenate([inc[2] for inc in inception_outputs], axis=0)
        y_pool3_np = np.concatenate([inc[3] for inc in inception_outputs], axis=0)
        FLAGS.print('logits calculated. Shape = {}.'.format(x_logits_np.shape))
        FLAGS.print('pool3 calculated. Shape = {}.'.format(x_pool3_np.shape))
        # calculate scores
        inc_x = self.inception_score_from_logits(x_logits_np)
        inc_y = self.inception_score_from_logits(y_logits_np)
        xp3_1, xp3_2 = np.split(x_pool3_np, indices_or_sections=2, axis=0)
        fid_xx = self.fid_from_pool3(xp3_1, xp3_2)
        fid_xy = self.fid_from_pool3(x_pool3_np, y_pool3_np)

        with MySession() as sess:
            scores = sess.run_once([inc_x, inc_y, fid_xx, fid_xy])

        return scores

    def sliced_wasserstein_distance(self, x_batch, y_batch, num_batch=128, ckpt_folder=None, ckpt_file=None):
        """ This function calculates the sliced wasserstein distance between real and fake images.

        This function does not work as expected, swd gives nan

        :param x_batch:
        :param y_batch:
        :param num_batch:
        :param ckpt_folder:
        :param ckpt_file:
        :return:
        """

        with MySession(load_ckpt=True) as sess:
            batches = sess.run_m_times(
                [x_batch, y_batch],
                ckpt_folder=ckpt_folder, ckpt_file=ckpt_file,
                max_iter=num_batch, trace=True)

        # get x_images and y_images
        x_images = (tf.constant(np.concatenate([batch[0] for batch in batches], axis=0)) + 1.0) * 128.5
        y_images = (tf.constant(np.concatenate([batch[1] for batch in batches], axis=0)) + 1.0) * 128.5

        if self.image_format in {'channels_first', 'NCHW'}:
            x_images = tf.transpose(x_images, perm=(0, 2, 3, 1))
            y_images = tf.transpose(y_images, perm=(0, 2, 3, 1))
        print('images obtained, shape: {}'.format(x_images.shape))

        # sliced_wasserstein_distance returns a list of tuples (distance_real, distance_fake)
        # for each level of the Laplacian pyramid from the highest resolution to the lowest
        swd = tfgan.eval.sliced_wasserstein_distance(
            x_images, y_images, patches_per_image=64, random_sampling_count=4, use_svd=True)
        with MySession() as sess:
            swd = sess.run_once(swd)

        return swd

    def ms_ssim(self, x_batch, y_batch, num_batch=128, ckpt_folder=None, ckpt_file=None, image_size=256):
        """ This function calculates the multiscale structural similarity between a pair of images.
        The image is downscaled four times; at each scale, a 11x11 filter is applied to extract patches.

        USE WITH CAUTION !!!
        1. This code was lost once and redone. Need to test on real datasets to verify it.
        2. This code can be improved to calculate pairwise ms-ssim using tf.image.ssim. tf.image.ssim_multicale is just
        tf.image.ssim with pool downsampling.

        :param x_batch:
        :param y_batch:
        :param num_batch:
        :param ckpt_folder:
        :param ckpt_file:
        :param image_size: ssim is defined on images of size at least 176
        :return:
        """

        # get x_images and y_images
        x_images = (x_batch + 1.0) * 128.5
        y_images = (y_batch + 1.0) * 128.5

        if self.image_format in {'channels_first', 'NCHW'}:
            x_images = tf.transpose(x_images, perm=(0, 2, 3, 1))
            y_images = tf.transpose(y_images, perm=(0, 2, 3, 1))
        if x_images.get_shape().as_list()[1] != 256:
            x_images = tf.image.resize_bilinear(x_images, [image_size, image_size])
            y_images = tf.image.resize_bilinear(y_images, [image_size, image_size])

        scores = tf.image.ssim_multiscale(x_images, y_images, max_val=255)  # scores in range [0, 1]

        with MySession(load_ckpt=True) as sess:
            scores = sess.run_m_times(
                scores,
                ckpt_folder=ckpt_folder, ckpt_file=ckpt_file,
                max_iter=num_batch, trace=True)

        ssim_score = np.mean(np.concatenate(scores, axis=0), axis=0)

        return ssim_score