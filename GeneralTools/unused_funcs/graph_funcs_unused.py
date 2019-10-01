
    # def intra_fid(self, ref_stat_filename, x_batch, num_batch=50, ckpt_folder=None, ckpt_file=None):
    #     """ This function calculates intra-fid and ms-ssim for data of one class
    #
    #     :param ref_stat_filename:
    #     :param x_batch:
    #     :param num_batch:
    #     :param ckpt_folder:
    #     :param ckpt_file:
    #     :return:
    #     """
    #     # get pool3 for all x
    #     x_pool3 = self.inception_v1_one_batch(x_batch, output_tensor='pool_3:0')
    #     with MySession(load_ckpt=True) as sess:
    #         inception_outputs = sess.run_m_times(
    #             x_pool3,
    #             ckpt_folder=ckpt_folder, ckpt_file=ckpt_file,
    #             max_iter=num_batch, trace=True)
    #     x_pool3_np = np.concatenate([inc for inc in inception_outputs], axis=0)
    #     # get stats of pool3 for all y
    #     ref_stat = np.load(os.path.join(FLAGS.DEFAULT_IN, ref_stat_filename + '.npz'))
    #     y_pool3_np = [ref_stat['mean'], ref_stat['cov']]
    #     # calculate fid
    #     fid = self.my_fid_from_pool3(x_pool3_np, y_pool3_np)
    #
    #     return fid

    # def _initialize_inception_v3_(self):
    #     """ This function adds inception v3 model to the graph and changes the tensor shape from [1, h, w, c]
    #     to [None, h, w, c] so that the inception 3 model can handle arbitrary input batch size.
    #
    #     Note: This function was obtained online. It did not work as expected.
    #
    #     :return:
    #     """
    #     # add inception graph to current graph
    #     with tf.gfile.FastGFile(FLAGS.INCEPTION_V3, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #         tf.import_graph_def(graph_def, name='')
    #
    #     # change the shape[0] of each tensor along the graph up to pool3_output
    #     with tf.Session() as sess:
    #         pool3_output = sess.graph.get_tensor_by_name('pool_3:0')
    #         ops = pool3_output.graph.get_operations()
    #         for op_idx, op in enumerate(ops):
    #             for o in op.outputs:
    #                 shape = o.get_shape().as_list()
    #                 if len(shape) > 0:
    #                     shape[0] = None
    #                 o.set_shape(shape)  # online resource uses o._shape = tf.TensorShape(shape), which did not work.
    #
    #         # define pool3 and logits
    #         # self._pool3_v3_ = tf.squeeze(pool3_output)  # squeeze remove dimensions of 1
    #         # print(sess.graph.get_tensor_by_name('Mul:0').get_shape().as_list())
    #         # print(self._pool3_v3_.get_shape().as_list())
    #         self._pool3_v3_ = tf.reshape(pool3_output, shape=[pool3_output.get_shape()[0], 2048])
    #         FLAGS.print(self._pool3_v3_.get_shape().as_list())
    #         weight = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    #         self._logits_v3_ = tf.matmul(self._pool3_v3_, weight)

    # def inception_v3(self, images, batch_size=100):
    #     """ This function runs the inception v3 model on images and give logits output.
    #
    #     Note: if other layers of inception model is needed, change the output_tensor option in
    #     self._initialize_inception_v3_
    #
    #     :param images:
    #     :type images: ndarray
    #     :param batch_size:
    #     :return:
    #     """
    #     # prepare
    #     if self.image_format in {'channels_first', 'NCHW'}:
    #         images = np.transpose(images, axes=(0, 2, 3, 1))
    #     images = images * 127.5 + 127.5  # rescale batch_image to [0, 255]
    #     num_images = images.shape[0]
    #     num_batches = int(math.ceil(num_images / batch_size))
    #
    #     # run iterations
    #     pool3 = []
    #     logits = []
    #     with tf.Session() as sess:
    #         for i in range(num_batches):
    #             batch_image = images[(i * batch_size):min((i + 1) * batch_size, num_images)]
    #             batch_pool3, batch_logits = sess.run(
    #                 [self._pool3_v3_, self._logits_v3_], feed_dict={'ExpandDims:0': batch_image})
    #             pool3.append(batch_pool3)
    #             logits.append(batch_logits)
    #         pool3 = np.concatenate(pool3, axis=0)
    #         logits = np.concatenate(logits, axis=0)
    #
    #     return logits, pool3

    # def inception_score_and_fid_v3(
    #         self, x_batch, y_batch, num_batch=10, inception_batch=100, ckpt_folder=None, ckpt_file=None):
    #     """ This function calculates inception scores and FID based on inception v1.
    #     Note: batch_size * num_batch needs to be larger than 2048, otherwise the convariance matrix will be
    #     ill-conditioned.
    #
    #     Steps:
    #     1. a large number of images are generated
    #     1, the pool3 and logits are calculated from numpy arrays x_images and y_images
    #     2, the pool3 and logits are passed to corresponding metrics
    #
    #     :param x_batch:
    #     :param y_batch:
    #     :param num_batch:
    #     :param inception_batch:
    #     :param ckpt_folder:
    #     :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
    #     :return:
    #     """
    #     assert self.model == 'v3', 'GenerativeModelMetric is not initialized with model="v3".'
    #     # initialize inception v3
    #     self._initialize_inception_v3_()
    #
    #     # generate x_batch, get logits and pool3
    #     with MySession(load_ckpt=True) as sess:
    #         x_image_list = sess.run_m_times(
    #             x_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, max_iter=num_batch, trace=True)
    #         x_images = np.concatenate(x_image_list, axis=0)
    #     FLAGS.print('x_image obtained, shape: {}'.format(x_images.shape))
    #     x_logits_np, x_pool3_np = self.inception_v3(x_images, batch_size=inception_batch)
    #     FLAGS.print('logits calculated. Shape = {}.'.format(x_logits_np.shape))
    #     FLAGS.print('pool3 calculated. Shape = {}.'.format(x_pool3_np.shape))
    #
    #     # generate y_batch, get logits and pool3
    #     with MySession(load_ckpt=True) as sess:
    #         y_image_list = sess.run_m_times(
    #             y_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file, max_iter=num_batch, trace=True)
    #         y_images = np.concatenate(y_image_list, axis=0)
    #     FLAGS.print('y_image obtained, shape: {}'.format(x_images.shape))
    #     y_logits_np, y_pool3_np = self.inception_v3(y_images, batch_size=inception_batch)
    #
    #     # calculate scores
    #     inc_x = self.inception_score_from_logits(x_logits_np)
    #     inc_y = self.inception_score_from_logits(y_logits_np)
    #     xp3_1, xp3_2 = np.split(x_pool3_np, indices_or_sections=2, axis=0)
    #     fid_xx = self.fid_from_pool3(xp3_1, xp3_2)
    #     fid_xy = self.fid_from_pool3(x_pool3_np, y_pool3_np)
    #
    #     with MySession() as sess:
    #         scores = sess.run_once([inc_x, inc_y, fid_xx, fid_xy])
    #
    #     return scores

    # def pairwise_ms_ssim(self, x_batch, num_batch=128, ckpt_folder=None, ckpt_file=None, image_size=256):
    #     """ This function calculates the pairwise multiscale structural similarity among a group of images.
    #     The image is downscaled four times; at each scale, a 11x11 filter is applied to extract patches.
    #
    #     :param x_batch:
    #     :param num_batch:
    #     :param ckpt_folder:
    #     :param ckpt_file:
    #     :param image_size:
    #     :return:
    #     """


# def _create_variable_(name, initializer, fan_size, trainable=True, weight_scale=1.0):
#     """ This function pins variables to cpu
#
#     tf.get_variable is used instead of tf.Variable, so that variable with the same name will not be re-initialized
#     """
#     # define initialization method
#     if initializer == 'zeros':
#         initializer_fun = tf.zeros(fan_size)
#     elif initializer == 'ones':
#         initializer_fun = tf.multiply(tf.ones(fan_size), weight_scale)
#     elif initializer == 'xavier':
#         initializer_fun = xavier_init(fan_size[0], fan_size[1], weight_scale=weight_scale)
#     elif initializer == 'normal_in':
#         initializer_fun = tf.random.normal(fan_size, mean=0.0, stddev=tf.divide(1.0, tf.sqrt(fan_size[0])),
#                                            dtype=tf.float32)
#     else:
#         raise AttributeError('Initialization method not supported.')
#
#     return tf.get_variable(name=name, initializer=initializer_fun, trainable=trainable)


# def create_variable(name, initializer, fan_size, trainable=True, weight_scale=1.0, pin_to_cpu=True):
#     """ This function pins variables to cpu
#
#     tf.get_variable is used instead of tf.Variable, so that variable with the same name will not be re-initialized
#     """
#     if pin_to_cpu:
#         with tf.device('/cpu:0'):
#             var = _create_variable_(name, initializer, fan_size, trainable, weight_scale)
#     else:
#         var = _create_variable_(name, initializer, fan_size, trainable, weight_scale)
#
#     return var


# def xavier_init(fan_in, fan_out, weight_scale=1.0):
#     """ Xavier initialization of network weights"""
#     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
#     high = tf.multiply(weight_scale, tf.sqrt(6.0 / tf.add(fan_in, fan_out)))
#
#     return tf.random_uniform((fan_in, fan_out), minval=tf.negative(high), maxval=high, dtype=tf.float32)


# class SynTower(object):
#     def __init__(self):
#         """ This class contains several methods to synchronize variables across towers
#
#         """
#         pass
#
#     @staticmethod
#     def average_grads(tower_grads):
#         """ This function averages the tower_grads
#
#         :param tower_grads: the list of gradients calculated from opt_op.compute_gradient
#         :return:
#         """
#         average_grads_list = []
#         for grad_var_tuple in zip(*tower_grads):
#             # a = zip(x,y) aggregates elements from each list into a = [(x0,y0),(x1,y1),...]
#             # b = zip(*a) unzip tuples in a. Let a2=list(a), b2=list(b), then b2[i][j]=a2[j][i]
#             # zip objects can only be unpack once using list(), tuple(), etc
#             # grad_var_tuple takes the form: ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1))
#             grads = []
#             for g, _ in grad_var_tuple:
#                 # first add one more dimension to g in the first dimension,
#                 # so that later we can add grads along the first dimension, aka, the tower dimension
#                 grads.append(tf.expand_dims(g, 0))  # grads now is a list of vectors
#             # concatenate grads along the first dimension so that it will become a matrix
#             grad_matrix = tf.concat(values=grads, axis=0)
#             # average grad
#             grad = tf.reduce_mean(grad_matrix, 0)
#             # get the name for this grad
#             var_name = grad_var_tuple[0][1]
#             # append the results
#             average_grads_list.append((grad, var_name))
#         return average_grads_list
#
#     @staticmethod
#     def average_var(tower_var):
#         """ This function averages the tower_var
#
#         :param tower_var: a zip of list of variables, zip([[a0, b0, c0, ...], [a1, b1, c1, ...]])
#         :return:
#         """
#         average_var_list = []
#         for var_tuple in zip(*tower_var):
#             # extract var from var_tuple
#             _vars = []
#             for v in var_tuple:
#                 _vars.append(tf.expand_dims(v, 0))
#             var_matrix = tf.concat(values=_vars, axis=0)
#             # average var
#             var_mean = tf.reduce_mean(var_matrix, 0)
#             # append the results
#             average_var_list.append(var_mean)
#         return average_var_list
#
#     @staticmethod
#     def stack_var(tower_var, axis=0):
#         """ This function stacks the tower_var
#
#         :param tower_var:
#         :param axis:
#         :return:
#         """
#         stack_var_list = []
#         for var_tuple in zip(*tower_var):
#             # extract var from var_tuple
#             var_matrix = tf.concat(values=list(var_tuple), axis=axis)
#             # append the results
#             stack_var_list.append(var_matrix)
#         return stack_var_list


# def average_tower_grads(tower_grads):
#     """ This function averages the tower_grads
#
#     Inputs:
#     tower_grads - a list of lists of (gradient, variable) tuples
#     """
#     average_grads = []
#     for grad_var_tuple in zip(*tower_grads):
#         # a = zip(x,y) aggregates elements from each list into a = [(x0,y0),(x1,y1),...]
#         # b = zip(*a) unzip tuples in a. Let a2=list(a), b2=list(b), then b2[i][j]=a2[j][i]
#         # zip objects can only be unpack once using list(), tuple(), etc
#         # grad_var_tuple takes the form: ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1))
#         grads = []
#         for g, _ in grad_var_tuple:
#             # first add one more dimension to g in the first dimension,
#             # so that later we can add grads along the first dimension, aka, the tower dimension
#             grads.append(tf.expand_dims(g, 0))  # grads now is a list of vectors
#         # concatenate grads along the first dimension so that it will become a matrix
#         grad = tf.concat(values=grads, axis=0)
#         # average grad
#         grad = tf.reduce_mean(grad, 0)
#         # get the name for this grad
#         var_name = grad_var_tuple[0][1]
#         # append the results
#         average_grads.append((grad, var_name))
#     return average_grads


# def print_tensor_in_ckpt(ckpt_folder, all_tensor_values=False, all_tensor_names=False):
#     """ This function print the list of tensors in checkpoint file.
#
#     Example:
#     from GeneralTools.graph_func import print_tensor_in_ckpt
#     ckpt_folder = '/home/richard/PycharmProjects/myNN/Results/cifar_ckpt/sngan_hinge_2e-4_nl'
#     print_tensor_in_ckpt(ckpt_folder)
#
#     :param ckpt_folder:
#     :param all_tensor_values: Boolean indicating whether to print the values of all tensors.
#     :param all_tensor_names: Boolean indicating whether to print all tensor names.
#     :return:
#     """
#     from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
#     if not isinstance(ckpt_folder, str):  # if list, use the name of the first file
#         ckpt_folder = ckpt_folder[0]
#
#     output_folder = os.path.join(FLAGS.DEFAULT_OUT, ckpt_folder)
#     print(output_folder)
#     ckpt = tf.train.get_checkpoint_state(output_folder)
#     print(ckpt)
#     print_tensors_in_checkpoint_file(
#         file_name=ckpt.model_checkpoint_path, tensor_name='',
#         all_tensors=all_tensor_values, all_tensor_names=all_tensor_names)


# def graph_configure(
#         initial_lr, global_step_name='global_step', lr_decay_steps=None,
#         end_lr=1e-7, optimizer='adam'):
#     """ This function configures global_step and optimizer
#
#     :param initial_lr:
#     :param global_step_name:
#     :param lr_decay_steps:
#     :param end_lr:
#     :param optimizer:
#     :return:
#     """
#     global_step = global_step_config(name=global_step_name)
#     learning_rate, opt_op = opt_config(initial_lr, lr_decay_steps, end_lr, optimizer, global_step)
#
#     return global_step, learning_rate, opt_op


# def data2sprite(
#         filename, image_size, mesh_num=None, if_invert=False,
#         num_threads=6, file_suffix='', image_transpose=False,
#         grey_scale=False, separate_channel=False, image_format='channels_last'):
#     """ This function reads data and writes them to sprite. Extra outputs are
#     greyscaled image and images in each RGB channel
#
#     :param filename: ['celebA_0']
#     :param image_size: [height, width, channels]
#     :param mesh_num: ()
#     :param if_invert:
#     :param num_threads:
#     :param file_suffix:
#     :param image_transpose: for dataset like MNIST, image needs to be transposed
#     :param grey_scale: if true, also plot the grey-scaled image
#     :param separate_channel: if true, also plot the red
#     :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
#     :return:
#     """
#     # check inputs
#     height, width, channels = image_size
#     data_dimension = np.prod(image_size, dtype=np.int32)
#     if mesh_num is None:
#         mesh_num = (10, 10)
#     batch_size = np.prod(mesh_num, dtype=np.int32)
#     if image_transpose:  # for dataset like MNIST, image needs to be transposed
#         perm = [0, 2, 1, 3]
#     else:
#         perm = None
#     if channels == 1:
#         grey_scale = False
#         separate_channel = False
#     # prepare folder
#     _, summary_folder, _ = prepare_folder(filename, sub_folder=file_suffix)
#
#     # read data
#     training_data = ReadTFRecords(filename, data_dimension, 0, num_threads=num_threads)
#     # training_data = PreloadGPU(filename, num_instance, self.D, num_threads=num_threads)
#     # convert matrix data to image tensor channels_first or channels_last format and scale them to [-1, 1]
#     training_data.shape2image(channels, height, width)
#
#     # build the network graph
#     with tf.Graph().as_default():
#         # get next batch
#         training_data.scheduler(batch_size=batch_size)
#         x_batch, _ = training_data.next_batch()
#         FLAGS.print('Graph configuration finished...')
#         # convert x_batch to channels_last format
#         if image_format == 'channels_first':
#             x_batch = np.transpose(x_batch, axes=(0, 2, 3, 1))
#
#         # calculate the value of x_batch and grey_scaled image
#         if grey_scale:
#             x_batch_gs = tf.image.rgb_to_grayscale(x_batch)
#             with MySession() as sess:  # loss is a list of tuples
#                 x_batch_value, x_batch_gs_value = sess.run_once([x_batch, x_batch_gs])
#         else:
#             with MySession() as sess:  # loss is a list of tuples
#                 x_batch_value = sess.run_once(x_batch)
#             x_batch_gs_value = None
#
#     # for dataset like MNIST, image needs to be transposed
#     if image_transpose:
#         x_batch_value = np.transpose(x_batch_value, axes=perm)
#     # write to files
#     write_sprite_wrapper(
#         x_batch_value, mesh_num, filename, file_folder=summary_folder,
#         file_index='_real', if_invert=if_invert, image_format='channels_last')
#     if grey_scale:
#         # for dataset like MNIST, image needs to be transposed
#         if image_transpose:
#             x_batch_gs_value = np.transpose(x_batch_gs_value, axes=perm)
#         # write to files
#         write_sprite_wrapper(
#             x_batch_gs_value, mesh_num, filename, file_folder=summary_folder,
#             file_index='_real_greyscale', if_invert=if_invert, image_format='channels_last')
#     if separate_channel:
#         channel_name = ['_R', '_G', '_B']
#         for i in range(channels):
#             write_sprite_wrapper(
#                 x_batch_value[:, :, :, i], mesh_num, filename, file_folder=summary_folder,
#                 file_index='_real' + channel_name[i], if_invert=if_invert, image_format='channels_last')


# class Fig(object):
#     """ This class uses following two packages for figure plotting:
#         import matplotlib.pyplot as plt
#         import plotly as py
#     """
#     def __init__(self, fig_def=None, sub_mode=False):
#         # change default figure setup
#         self.dict = {'grid': False, 'title': 'Figure', 'x_label': 'x', 'y_label': 'y'}
#         self._reset_fig_def_(fig_def)
#         self.sub_mode = sub_mode
#
#         # register plotly just in case
#         # py.tools.set_credentials_file(username=FLAGS.PLT_ACC, api_key=FLAGS.PLT_KEY)
#
#     def new_figure(self, *args, **kwargs):
#         if not self.sub_mode:
#             return plt.figure(*args, **kwargs)
#
#     def new_sub_figure(self, *args, **kwargs):
#         if self.sub_mode:
#             return plt.subplot(*args, **kwargs)
#
#     def show_figure(self, sub_mode=None):
#         if sub_mode is not None:
#             self.sub_mode = sub_mode
#         if not self.sub_mode:
#             plt.show()
#
#     def _reset_fig_def_(self, fig_def):
#         if fig_def is not None:
#             for key in fig_def:
#                 self.dict[key] = fig_def[key]
#
#     def _add_figure_labels_(self):
#         plt.grid(self.dict['grid'])
#         plt.title(self.dict['title'])
#         plt.xlabel(self.dict['x_label'])
#         plt.ylabel(self.dict['y_label'])
#
#     def hist(self, data_list, bins='auto', fig_def=None):
#         """ Histogram plot
#
#         :param data_list:
#         :param bins:
#         :param fig_def:
#         :return:
#         """
#         # check inputs
#         self._reset_fig_def_(fig_def)
#
#         # plot figure
#         self.new_figure()
#         plt.hist(data_list, bins)
#         self._add_figure_labels_()
#         # plt.colorbar()
#         self.show_figure()
#
#     def hist2d(self, x=None, x0=None, x1=None, bins=10, data_range=None, log_norm=False, fig_def=None):
#         """
#
#         :param x: either x or x0, x1 is given
#         :param x0:
#         :param x1:
#         :param bins:
#         :param data_range:
#         :param log_norm: if log normalization is used
#         :param fig_def:
#         :return:
#         """
#         from matplotlib.colors import LogNorm
#         # check inputs
#         self._reset_fig_def_(fig_def)
#         if x is not None:
#             x0 = x[:, 0]
#             x1 = x[:, 1]
#         if data_range is None:
#             data_range = [[-1.0, 1.0], [-1.0, 1.0]]
#         num_instances = x0.shape[0]
#         if num_instances > 200:
#             count_min = np.ceil(num_instances/bins/bins*0.05)  # bins under this value will not be displayed
#             print('hist2d; counts under {} will be ignored.'.format(count_min))
#         else:
#             count_min = None
#
#         # plot figure
#         self.new_figure()
#         if log_norm:
#             plt.hist2d(x0, x1, bins, range=data_range, norm=LogNorm(), cmin=count_min)
#         else:
#             plt.hist2d(x0, x1, bins, range=data_range, cmin=count_min)
#         self._add_figure_labels_()
#         plt.colorbar()
#         self.show_figure()
#
#     def plot(self, y, x=None, fig_def=None):
#         """ line plot
#
#         :param y:
#         :param x:
#         :param fig_def:
#         :return:
#         """
#         # check inputs
#         self._reset_fig_def_(fig_def)
#
#         # plot figure
#         self.new_figure()
#         if x is None:
#             plt.plot(y)
#         else:
#             plt.plot(x, y)
#         self._add_figure_labels_()
#         self.show_figure()
#
#     def scatter(self, x=None, x0=None, x1=None, fig_def=None):
#         """ scatter plot
#
#         :param x: The data is given either as x, a [N, 2] matrix or x0, x1, each a [N] vector
#         :param x0:
#         :param x1:
#         :param fig_def:
#         :return:
#         """
#         # check inputs
#         self._reset_fig_def_(fig_def)
#         if x is not None:
#             x0 = x[:, 0]
#             x1 = x[:, 1]
#
#         # plot figure
#         self.new_figure()
#         plt.scatter(x0, x1)
#         self._add_figure_labels_()
#         self.show_figure()
#
#     def group_scatter(self, data, labels, fig_def=None):
#         """ scatter plot with labels
#
#         :param data: either a tuple (data1, data2, ...) or list, or a matrix
#         :param labels: a tuple (label1, label2, ...) or list; its length either matches the length of data tuple or
#             the number of rows in data matrix
#         :param fig_def:
#         :return:
#         """
#         if isinstance(data, tuple):
#             assert isinstance(labels, tuple), 'if data is tuple, label must be tuple'
#             assert len(labels) == len(data), \
#                 'Length not match: len(labels)={} while len(data)={}'.format(len(labels), len(data))
#         else:  # data is a numpy array
#             unique_labels = tuple(np.unique(labels))
#             data_tuple = ()
#             for label in unique_labels:
#                 index = labels == label
#                 data_tuple = data_tuple + (data[index, :],)
#             data = data_tuple
#             labels = unique_labels
#
#         # plot
#         self._reset_fig_def_(fig_def)
#         fig = self.new_figure()
#         ax = fig.add_subplot(1, 1, 1)
#         for sub_data, label in zip(data, labels):
#             x = sub_data[:, 0]
#             y = sub_data[:, 1]
#             ax.scatter(x, y, label=label)
#
#         self._add_figure_labels_()
#         plt.legend(loc=0)
#         self.show_figure()
#
#     def text_scatter(self, data, texts, color_labels=None, fig_def=None):
#         """ scatter plot with texts
#
#         :param data: either a tuple (data1, data2, ...) or list, or a matrix.
#         :param texts: either a tuple (txt1, txt2, ...) or list; its length either matches the length of data tuple or
#             the number of rows in data matrix
#         :param color_labels: either a tuple (C1, C2, ...) or list; its length either matches the length of data tuple
#             or the number of rows in data matrix. If provided, label us used to decide the color of texts.
#         :param fig_def:
#         :return:
#         """
#         if isinstance(data, tuple):
#             assert isinstance(texts, tuple), 'if data is tuple, label must be tuple'
#             assert len(texts) == len(data), \
#                 'Length not match: len(texts)={} while len(data)={}'.format(len(texts), len(data))
#             if color_labels is not None:
#                 assert isinstance(color_labels, tuple), 'if data is tuple, colors must be tuple'
#                 assert len(color_labels) == len(data), \
#                     'Length not match: len(colors)={} while len(data)={}'.format(len(color_labels), len(data))
#         else:  # data is a numpy array. in this case, texts and color_labels must all be numpy array
#             if color_labels is None:  # one class
#                 data = (data,)
#                 texts = (texts,)
#                 color_labels = ('k',)
#             else:
#                 unique_colors = tuple(np.unique(color_labels))
#                 data_tuple = ()
#                 text_tuple = ()
#                 for color in unique_colors:
#                     index = color_labels == color
#                     data_tuple = data_tuple + (data[index, :],)
#                     text_tuple = text_tuple + (texts[index])
#                 data = data_tuple
#                 texts = text_tuple
#                 color_labels = unique_colors
#
#         if not isinstance(color_labels[0], str):
#             color_temp = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
#             color_labels = tuple([color_temp[color % 10] for color in color_labels])
#
#         # plot
#         self._reset_fig_def_(fig_def)
#         # fig = plt.figure(figsize=(9.6, 7.2))
#         fig = self.new_figure(figsize=(9.6, 7.2))
#         ax = fig.add_subplot(1, 1, 1)
#         for sub_data, sub_texts, color in zip(data, texts, color_labels):
#             ax.scatter(sub_data[:, 0], sub_data[:, 1], color='w')
#             for datum, text in zip(sub_data, sub_texts):
#                 # ax.text(datum[0], datum[1], s=text, color=color)
#                 ax.annotate(text, xy=(datum[0], datum[1]), color=color, ha='center', va='center', size='x-small')
#
#         self._add_figure_labels_()
#         self.show_figure()
#
#     def contour(self, z, x=None, y=None, custom_level=False, fig_def=None):
#         """ contour plot
#
#         :param z: the contour level, a [d, d] matrix
#         :param x:
#         :param y:
#         :param custom_level:
#         :param fig_def:
#         :return:
#         """
#         # check inputs
#         self._reset_fig_def_(fig_def)
#         # obtain levels
#         if custom_level:
#             z_max = np.percentile(z, q=99)
#             z_min = np.percentile(z, q=1)
#             levels = np.linspace(z_min, z_max, 10)
#         else:
#             levels = None
#
#         # plot figure
#         self.new_figure()
#         if levels is None:
#             if x is None or y is None:
#                 c_s = plt.contour(z)
#             else:
#                 c_s = plt.contour(x, y, z)
#         else:
#             if x is None or y is None:
#                 c_s = plt.contour(z, levels=levels)
#             else:
#                 c_s = plt.contour(x, y, z, levels=levels)
#         plt.clabel(c_s, inline=1, fontsize=10)
#         self._add_figure_labels_()
#         self.show_figure()
#
#     @staticmethod
#     def add_line(p1, p2, color='C0'):
#         """ This function adds a line to current plot without changing the bound of x or y axis
#
#         The default colors range from 'C0' to 'C9'.
#
#         :param p1:
#         :param p2:
#         :param color
#         :return:
#         """
#         import matplotlib.lines as ml
#
#         ax = plt.gca()
#         xl, xu = ax.get_xbound()
#
#         if p2[0] == p1[0]:
#             xl = xu = p1[0]
#             yl, yu = ax.get_ybound()
#         else:
#             yu = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xu - p1[0])
#             yl = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xl - p1[0])
#
#         line = ml.Line2D([xl, xu], [yl, yu], color=color)
#         ax.add_line(line)
#
#         return line


# def print_pb_to_event(model_path, event_folder):
#     """ This function print a pre-trained model to event_folder so that it can be viewed by tensorboard
#
#     :param model_path: for example, FLAGS.INCEPTION_V3
#     :param event_folder: for example, '/home/richard/PycharmProjects/myNN/Code/inception_v3/'
#     :return:
#     """
#     from Code.import_pb_to_tensorboard import import_to_tensorboard
#
#     import_to_tensorboard(model_path, event_folder)


# def imagenet_ref_stats():
#     """ The function calculates mean and cov of Inception pool3 for each class of the imagenet datasets
#
#     :return:
#     """
#     import sys
#
#     class_counts = np.genfromtxt(FLAGS.DEFAULT_IN + 'image_class_counts.txt', dtype=int)
#
#     start_time = time.time()
#     for class_id in range(1000):
#         num_images = class_counts[class_id, 1]
#         filename = 'imagenet_{:03d}'.format(class_id)
#
#         data = ReadTFRecords(filename, 128 * 128 * 3, 1, batch_size=num_images)
#         data.shape2image(3, 128, 128)
#         batch = data.next_batch(shuffle_data=False)
#         images = batch['x']
#         metric = GenerativeModelMetric()
#         if num_images % 100 == 0:
#             images_list = tf.split(images, num_or_size_splits=num_images // 100, axis=0)
#             pool3 = tf.map_fn(
#                 fn=lambda x: metric.inception_v1_one_batch(x, 'pool_3:0'),
#                 elems=tf.stack(images_list),
#                 dtype=tf.float32,
#                 parallel_iterations=1,
#                 back_prop=False,
#                 swap_memory=True,
#                 name='RunClassifier')
#             pool3 = tf.concat(tf.unstack(pool3), 0)
#         else:
#             images_list = tf.split(
#                 images, num_or_size_splits=[100] * (num_images // 100) + [num_images % 100], axis=0)
#             # tf.stack requires the dimension of tensor in list to be the same
#             pool3 = tf.map_fn(
#                 fn=lambda x: metric.inception_v1_one_batch(x, 'pool_3:0'),
#                 elems=tf.stack(images_list[0:-1]),
#                 dtype=tf.float32,
#                 parallel_iterations=1,
#                 back_prop=False,
#                 swap_memory=True,
#                 name='RunClassifier')
#             pool3_last = metric.inception_v1_one_batch(images_list[-1], 'pool_3:0')
#             pool3 = tf.concat(tf.unstack(pool3) + [pool3_last], 0)
#
#         with MySession() as sess:
#             pool3 = sess.run_once(pool3)
#
#         # save to file
#         mu, cov = mean_cov_np(pool3)
#         np.savez(os.path.join(FLAGS.DEFAULT_IN, filename + '.npz'), mean=mu, cov=cov)
#         # to load mean and cov, do: stat = np.load(os.path.join(FLAGS.DEFAULT_IN, filename+'.npz'))
#
#         if class_id % 50 == 0:
#             sys.stdout.write('\r {}/{} classes finished.'.format(class_id + 1, 1000))
#     duration = time.time() - start_time
#     print('\n All 1000 classes finished in {:.1f} seconds'.format(duration))
