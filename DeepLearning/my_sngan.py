""" This code builds a convolutional GAN.

"""
# check the summary:
# tensorboard --logdir="MMD-GAN/Data/celebA_0_log/test"
# check GPU workload
# nvidia-smi.exe

# default modules
import numpy as np
import tensorflow as tf
from collections import namedtuple

# helper functions
from GeneralTools.misc_fun import FLAGS
from GeneralTools.input_func import ReadTFRecords
from GeneralTools.graph_funcs.graph_func import prepare_folder, write_sprite_wrapper, \
    global_step_config, dis_gen_opt_config, embedding_image_wrapper, rollback
from GeneralTools.graph_funcs.generative_model_metric import GenerativeModelMetric
from GeneralTools.layer_func import Net, Routine
from GeneralTools.math_funcs.sn_gan_support import MeshCode
from GeneralTools.math_funcs.gan_losses import GANLoss
from dp_funcs.net_picker import NetPicker
from dp_funcs.dp_grads import dp_compute_grads

########################################################################
# from GeneralTools.misc_fun import FLAGS

"""
Class definition
"""


class SNGan(object):
    def __init__(self, architecture, num_class=0, loss_type='logistic', optimizer='adam', do_summary=True,
                 do_summary_image=True, num_summary_image=8, image_transpose=False, **kwargs):
        """ This function initializes a ladder adversarial network.

        :param architecture: a dictionary, see my_test_* for example
        :param num_class: number of classes in data
        :param loss_type: extra parameters to the model
        :param optimizer:
        :param do_summary:
        :param do_summary_image:
        :param num_summary_image:
        :param image_transpose:
        """
        # default parameters
        self.optimizer_type = ['sgd', 'momentum', 'adam', 'rmsprop']
        self.data_format = FLAGS.IMAGE_FORMAT
        # structure parameters
        self.architecture = architecture
        self.loss_type = loss_type
        self.optimizer = optimizer
        # input parameters
        self.num_class = num_class
        self.channels = self.architecture['input'][0][0]
        self.height = self.architecture['input'][0][1]
        self.width = self.architecture['input'][0][2]
        self.input_size = np.prod(self.architecture['input'][0], dtype=np.int32)
        self.code_size = self.architecture['code'][0][0]
        self.score_size = self.architecture['discriminator'][-1]['out']
        # control parameters
        self.do_summary = do_summary
        self.do_summary_image = do_summary_image
        self.num_summary_image = num_summary_image
        self.loss_tuple = None
        self.global_step = None
        self.step_per_epoch = None
        self.sample_same_class = False
        self.force_print = True
        # method parameters
        self.rep_weights = kwargs['rep_weights'] if 'rep_weights' in kwargs else [0.0, -1.0]
        self.penalty_weight = kwargs['mmd_g_scale'] if 'mmd_g_scale' in kwargs else 0.1

        if image_transpose:  # for dataset like MNIST, image needs to be transposed
            if FLAGS.IMAGE_FORMAT == 'channels_first':
                self.perm = [0, 1, 3, 2]
            elif FLAGS.IMAGE_FORMAT == 'channels_last':
                self.perm = [0, 2, 1, 3]
        else:
            self.perm = None

        # initialize network
        self.graph, self.Gen, self.Dis = None, None, None

        # initialize MoG
        self.mog_model, self.train_with_mog, self.data_batch = None, None, None
        self.repeat_for_mog = False

        # for random fourier kerrnel approximation in loss
        self.rff_spec = kwargs['rff_spec'] if 'rff_spec' in kwargs else None
        self.stop_snorm_grads = kwargs['stop_snorm_grads'] if 'stop_snorm_grads' in kwargs else None

    def register_mog(self, mog_model, train_with_mog, update_loss_type=False):
        self.mog_model = mog_model
        self.train_with_mog = train_with_mog
        self.repeat_for_mog = train_with_mog
        if update_loss_type is not False:
            self.loss_type = update_loss_type

    def init_net(self):
        """ This function initializes the network

        :return:
        """
        # initialize the generator network
        g_net = Net(self.architecture['generator'], 'gen', FLAGS.IMAGE_FORMAT, num_class=self.num_class)
        # define layer connections in generator
        # print('---------GENERATOR DEF')
        self.Gen = Routine(g_net)
        self.Gen.add_input_layers([64, self.code_size], [0])
        self.Gen.seq_links(list(range(g_net.num_layers)))
        self.Gen.add_output_layers([g_net.num_layers - 1])

        # initialize the discriminator network
        # print('---------DISCRIMINATOR DEF')
        d_net = Net(self.architecture['discriminator'], 'dis', FLAGS.IMAGE_FORMAT, num_class=self.num_class)

        if self.stop_snorm_grads:
            d_net.stop_all_snorm_grads()
        # define layer connections in discriminator
        self.Dis = Routine(d_net)
        self.Dis.add_input_layers([64] + list(self.architecture['input'][0]), [0])
        self.Dis.seq_links(list(range(d_net.num_layers)))
        self.Dis.add_output_layers([d_net.num_layers - 1])

        # print('-----------SNGAN INIT NET done')
        if self.mog_model is not None:
            self.mog_model.define_tfp_mog_vars(self.do_summary)



    ###################################################################
    def sample_codes(self, batch_size, code_x=None, code_y=None, name='codes'):
        """ This function prepares code_batch dictionary. If code_x is not provided, it is sampled
        from gaussian distribution. if code_y is not provided, it is sampled from uniform distribution.

        :param batch_size:
        :param code_x:
        :param code_y:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            if code_x is None:
                code_x = tf.random.normal(
                    [batch_size, self.code_size], mean=0.0, stddev=1.0, name='x', dtype=tf.float32)
            else:
                code_x = tf.identity(code_x, name='x')
                assert code_x.get_shape().as_list()[0] == batch_size, \
                    'Input code_x size {} does not match batch_size {}'.format(
                        code_x.get_shape().as_list()[0], batch_size)
                if code_x.dtype != tf.float32:
                    code_x = tf.cast(code_x, tf.float32)

            if self.num_class < 2:
                return {'x': code_x}
            else:
                if code_y is None:  # random sample data classes
                    code_y = tf.random_uniform(
                        [batch_size, 1], minval=0, maxval=self.num_class, name='y', dtype=tf.int32)
                elif isinstance(code_y, int):  # sample data from the same class
                    code_y = tf.constant(code_y, dtype=tf.int32, shape=[batch_size, 1], name='y')
                else:
                    code_y = tf.identity(code_y, name='y')
                    assert code_y.get_shape().as_list()[0] == batch_size, \
                        'Input code_x size {} does not match batch_size {}'.format(
                            code_y.get_shape().as_list()[0], batch_size)
                    if code_y.dtype != tf.int32:
                        code_y = tf.cast(code_y, tf.int32)

                return {'x': code_x, 'y': code_y}

    ###################################################################
    @staticmethod
    def concat_two_batches(batch1, batch2):
        """ This function concatenates two dictionaries.

        :param batch1:
        :param batch2:
        :return:
        """
        with tf.name_scope('concat_batch'):
            if 'y' in batch1 and isinstance(batch1['y'], tf.Tensor):
                return {'x': tf.concat([batch1['x'], batch2['x']], axis=0),
                        'y': tf.concat([batch1['y'], batch2['y']], axis=0)}
            else:
                return {'x': tf.concat([batch1['x'], batch2['x']], axis=0)}

    ###################################################################
    def _gpu_task(self, batch_size=64, is_training=False, data_batch=None,
                  opt_ops=None, code_batch=None, dp_spec=None):
        """ This function defines the task on a gpu

        :param batch_size:
        :param is_training:
        :param data_batch: dict. ['x'] is a 4-D tensor, either in channels_first or channels_last format
        :param opt_ops:
        :param code_batch:
        :param dp_spec
        :return:
        """
        if is_training:
            s_x, s_gen, s_mog, gen_batch = self.get_encodings(data_batch, batch_size, is_training)

            # loss function
            loss_ops = self.get_losses(s_gen, s_x, s_mog, batch_size, dp_spec)

            vars_dis = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "dis")
            vars_gen = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "gen")

            if dp_spec is not None:
                grads_list, loss_list = dp_compute_grads(loss_ops, dp_spec, vars_dis, vars_gen)
                loss_ops = self.loss_tuple(loss_list[0], loss_list[1])
                # print(grads_list[1])
            else:
                # compute gradient: grads is a list of (gradient, variable) tuples
                grads_dis = opt_ops.dis.compute_gradients(loss_ops.dis, var_list=vars_dis)
                grads_gen = opt_ops.gen.compute_gradients(loss_ops.gen, var_list=vars_gen)
                grads_list = [grads_dis, grads_gen]

            # summary op is always pinned to CPU
            # add summary to loss and intermediate variables
            if self.do_summary:
                tf.compat.v1.summary.histogram('x/x', data_batch['x'])
                tf.compat.v1.summary.histogram('x/x_gen', gen_batch['x'])
                tf.compat.v1.summary.histogram('x/sx', s_x)
                tf.compat.v1.summary.histogram('x/sg', s_gen)
                # g_x = tf.reshape(tf.gradients(s_x, data_batch['x'])[0], [batch_size, -1])
                # g_x_norm = tf.norm(g_x, ord=2, axis=1)
                # tf.compat.v1.summary.histogram('x/g_x_norm', g_x_norm)
                g_gen = tf.reshape(tf.gradients(s_gen, gen_batch['x'])[0], [batch_size, -1])
                g_gen_norm = tf.norm(g_gen, ord=2, axis=1)
                tf.compat.v1.summary.histogram('x/g_gen_norm', g_gen_norm)
                self.Gen.net.add_summary('kernel_norm')
                self.Dis.net.add_summary('kernel_norm')

            return grads_list, loss_ops
        else:
            if code_batch is None:
                code_batch = self.sample_codes(batch_size, name='code_te')
            # generate new images
            gen_batch = self.Gen(code_batch, is_training=is_training)
            assert isinstance(gen_batch, dict)
            return gen_batch

    def get_encodings(self, data_batch, batch_size, is_training):
        # sample new data, [batch_size*2, height, weight, channels]
        if self.sample_same_class:
            code_batch = self.sample_codes(batch_size, code_y=data_batch['y'], name='code_tr')
        else:
            code_batch = self.sample_codes(batch_size, name='code_tr')
        gen_batch = self.Gen(code_batch, is_training=is_training)

        dis_out = self.Dis(self.concat_two_batches(data_batch, gen_batch), is_training=True)
        s_x, s_gen = tf.split(dis_out['x'], num_or_size_splits=2, axis=0)

        s_mog = self.mog_model.sample_batch(batch_size) if self.mog_model is not None and self.train_with_mog else None
        return s_x, s_gen, s_mog, gen_batch

    def get_losses(self, s_gen, s_x, s_mog, batch_size, dp_spec):
        gan_losses = GANLoss(self.rff_spec, self.score_size, self.do_summary)
        if self.loss_type in {'rep', 'rmb'}:
            loss_gen, loss_dis = gan_losses.apply(s_gen, s_x, s_mog, self.loss_type, batch_size=batch_size,
                                                  d=self.score_size, rep_weights=self.rep_weights, dp_spec=dp_spec)
        else:
            loss_gen, loss_dis = gan_losses.apply(s_gen, s_x, s_mog, self.loss_type,
                                                  batch_size=batch_size, d=self.score_size, dp_spec=dp_spec)

        # form loss list
        # sigma = [layer.sigma for layer in self.Dis.net.layers]
        # kernel_norm = tf.squeeze(self.Dis.net.layers[-1].ops['kernel'].kernel_norm[1])
        self.loss_tuple = namedtuple('losses', ['dis', 'gen'])
        loss_ops = self.loss_tuple(loss_dis, loss_gen)
        return loss_ops

    def get_data_batch(self, filename, batch_size, file_repeat=1, num_threads=7, shuffle_file=False, name='data',
                       repeat_for_mog=False):
        """ This function reads image data

        :param filename:
        :param batch_size:
        :param file_repeat:
        :param num_threads:
        :param shuffle_file: bool, whether to shuffle the filename list
        :param name:
        :param repeat_for_mog:
        :return data_batch: a dictionary with key 'x' and optionally 'y'
        """
        with tf.name_scope(name):
            # read data
            num_labels = 0 if self.num_class < 2 else 1
            if num_labels == 0:
                self.sample_same_class = False

            training_data = ReadTFRecords(
                filename, self.input_size, num_labels=num_labels, x_dtype=tf.string, batch_size=batch_size,
                file_repeat=file_repeat, num_threads=num_threads, shuffle_file=shuffle_file)
            # training_data = PreloadGPU(filename, num_instance, self.D, num_threads=num_threads)
            # convert matrix data to image tensor and scale them to [-1, 1]
            training_data.shape2image(self.channels, self.height, self.width)
            data_batch = training_data.next_batch(self.sample_same_class, repeat_for_mog=repeat_for_mog)
            # convert x_combo to grey scale images
            # data_batch = tf.image.rgb_to_grayscale(data_batch)  # [batch_size, height, width, 1]
            # for dataset like MNIST, image needs to be transposed
            if self.perm is not None:
                data_batch['x'] = tf.transpose(data_batch['x'], perm=self.perm)

            self.data_batch = data_batch
        return data_batch

    ###################################################################
    def training(self, filename, agent, num_instance, lr_spec, end_lr=1e-7, max_step=None, batch_size=64,
                 sample_same_class=False, num_threads=7, gpu='/gpu:0', mog_model=None, dp_spec=None):
        """ This function defines the training process

        :param filename:
        :param agent:
        :param num_instance:
        :param lr_spec:
        :param end_lr:
        :param max_step:
        :type max_step: int
        :param batch_size:
        :param sample_same_class: bool, if at each iteration the data should be sampled from the same class
        :param num_threads:
        :param gpu: which gpu to use
        :param mog_model:
        :param dp_spec:
        :return:
        """
        self.step_per_epoch = np.floor(num_instance / batch_size).astype(np.int32)
        self.sample_same_class = sample_same_class
        if max_step >= self.step_per_epoch:
            from math import gcd
            file_repeat = int(batch_size / gcd(num_instance, batch_size)) if self.num_class < 2 \
                else int(batch_size / gcd(int(num_instance / self.num_class), batch_size))
            shuffle_file = False
        else:
            if isinstance(filename, str) or (isinstance(filename, (list, tuple)) and len(filename) == 1):
                raise AttributeError('max_step should be larger than step_per_epoch when there is a single file.')
            else:
                # for large dataset, the data are stored in multiple files. If all files cannot be visited
                # within max_step, consider shuffle the filename list every max_step
                file_repeat = 1
                shuffle_file = True

        FLAGS.print('Num Instance: {}; Num Class: {}; Batch: {}; File_repeat: {}'.format(
            num_instance, self.num_class, batch_size, file_repeat))

        # build the graph
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device(gpu):
            self.init_net()
            # get next batch
            data_batch = self.get_data_batch(filename, batch_size, file_repeat, num_threads, shuffle_file, 'data_tr',
                                             repeat_for_mog=self.repeat_for_mog)

            FLAGS.print('Shape of input batch: {}'.format(data_batch['x'].get_shape().as_list()))

            # setup training process
            # with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
            self.global_step = global_step_config()
            _, opt_ops = dis_gen_opt_config(lr_spec, end_lr=end_lr, optimizer=self.optimizer,
                                            global_step=self.global_step)
            # assign tasks
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):  # --------------------------- GPU TASK
                # calculate loss and gradients
                grads_list, loss_list = self._gpu_task(batch_size, True, data_batch, opt_ops, dp_spec=dp_spec)

            # apply the gradient
            op_list = self.apply_grads(opt_ops, grads_list, iu_spec=agent.imbalanced_update)

            # summary op is always pinned to CPU
            # add summary for all trainable variables
            if self.do_summary:
                for grads in grads_list:
                    for var_grad, var in grads:
                        var_name = var.name.replace(':', '_')
                        tf.compat.v1.summary.histogram('grad_' + var_name, var_grad)
                        tf.compat.v1.summary.histogram(var_name, var)
                summary_op = tf.compat.v1.summary.merge_all()
            else:
                summary_op = None
            # add summary for final image reconstruction
            if self.do_summary_image:
                tf.compat.v1.get_variable_scope().reuse_variables()
                summary_image_op = self.summary_image_sampling(data_batch)
            else:
                summary_image_op = None

            # run the session -----------------------------------------------------------GETTING CLOSER TO TRAINING LOOP
            FLAGS.print('loss_list name: {}.'.format(self.loss_tuple))
            agent.train(op_list, loss_list, self.global_step, max_step, self.step_per_epoch, summary_op,
                        summary_image_op, force_print=self.force_print, mog_model=mog_model)
            self.force_print = False  # force print at the first call

    def apply_grads(self, opt_ops, grads_list, iu_spec):
        step_dis, step_gen = False, False
        if iu_spec is None:
            step_dis = True
        elif isinstance(iu_spec, (list, tuple)):
            FLAGS.print('Imbalanced update used: dis per {} run and gen per {} run'.format(iu_spec[0], iu_spec[1]))
            if iu_spec[0] == 1:
                step_dis = True
            elif iu_spec[1] == 1:
                step_gen = True
            elif iu_spec[0] == -iu_spec[1]:
                # handle the alternating setting, since only one op is called at each step, both get the global step
                step_dis, step_gen = True, True
            else:
                raise AttributeError('One of the imbalanced_update must be 1 or they must alternate exactly.')
        elif isinstance(iu_spec, str):
            step_gen = True
        elif isinstance(iu_spec, NetPicker):
            # updates are laternating, so give globalstep to both (even though creating separate steps may make
            # sense in the long rung if either model is trained for a long time at once
            print('netpicker setting used in sngan')
            step_dis, step_gen = True, True
        else:
            raise AttributeError('Imbalanced_update not identified.')

        dis_op = opt_ops.dis.apply_gradients(grads_list[0], global_step=self.global_step if step_dis else None)
        gen_op = opt_ops.gen.apply_gradients(grads_list[1], global_step=self.global_step if step_gen else None)
        return [dis_op, gen_op]

    ###################################################################
    def summary_image_sampling(self, data_batch):
        """ This function randomly samples instances to compare with real samples.
        It returns a summary op.

        :param data_batch:
        :return:
        """
        # down sample x
        x_real = data_batch['x'][0:self.num_summary_image, :]
        # generate new images
        gen_batch = self._gpu_task(batch_size=self.num_summary_image, is_training=False)
        # do clipping
        x_gen = tf.clip_by_value(gen_batch['x'], clip_value_min=-1, clip_value_max=1)
        # tf.compat.v1.summary.image only accepts [batch_size, height, width, channels]
        if FLAGS.IMAGE_FORMAT == 'channels_first':
            x_real = tf.transpose(x_real, perm=(0, 2, 3, 1))
            x_gen = tf.transpose(x_gen, perm=(0, 2, 3, 1))
        # add summaries
        summaries_image = tf.compat.v1.summary.image('Ir', x_real, max_outputs=self.num_summary_image)
        summaries_gen = tf.compat.v1.summary.image('Ig', x_gen, max_outputs=self.num_summary_image)
        summary_image_op = tf.compat.v1.summary.merge([summaries_image, summaries_gen])

        return summary_image_op

    ###################################################################
    def eval_sampling(self, filename, sub_folder, mesh_num=None, mesh_mode=0, if_invert=False,
                      code_x=None, code_y=None, real_sample=False, sample_same_class=False,
                      get_dis_score=True, do_sprite=True,
                      do_embedding=False, ckpt_file=None, num_threads=7):
        """ This function randomly generates samples and writes them to sprite.

        :param sample_same_class:
        :param code_y:
        :param filename:
        :param sub_folder:
        :param mesh_num:
        :param if_invert:
        :param mesh_mode:
        :param code_x: if provided, z_batch will be used to generate images.
        :param num_threads:
        :param real_sample: True if real sample should also be obtained
        :param get_dis_score: bool, whether to calculate the scores from the discriminator
        :param do_sprite:
        :param do_embedding:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :return:
        """
        # prepare folder
        ckpt_folder, summary_folder, _ = prepare_folder(filename, sub_folder=sub_folder)
        # check inputs
        if mesh_num is None:
            mesh_num = (10, 10)
        elif code_x is not None:
            assert code_x.shape[0] == mesh_num[0] * mesh_num[1]
        batch_size = mesh_num[0] * mesh_num[1]
        if do_embedding is True:
            get_dis_score = True
            real_sample = True

        # build the network graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_net()

            # get real sample
            if real_sample:
                self.sample_same_class = sample_same_class
                data_batch = self.get_data_batch(filename, batch_size, num_threads=num_threads)
            else:
                data_batch = {'x': tf.constant(0)}

            # sample validation instances
            if code_x is None:
                code = MeshCode(self.code_size, mesh_num=mesh_num)
                code_x = code.get_batch(mesh_mode, name='code_x')
            if code_y is None and self.sample_same_class and 'y' in data_batch:
                code_y = data_batch['y']
            code_batch = self.sample_codes(batch_size, code_x, code_y, name='code_te')
            # generate new images
            gen_batch = self._gpu_task(code_batch=code_batch, is_training=False)
            # do clipping
            gen_batch['x'] = tf.clip_by_value(gen_batch['x'], clip_value_min=-1, clip_value_max=1)

            # get discriminator scores
            if get_dis_score and real_sample:
                dis_out = self.Dis(self.concat_two_batches(data_batch, gen_batch), is_training=False)
                s_x, s_gen = tf.split(dis_out['x'], num_or_size_splits=2, axis=0)
            else:
                s_x = tf.constant(0)
                s_gen = tf.constant(0)

            FLAGS.print('Graph configuration finished...')
            # calculate the value of x_gen
            var_list = [gen_batch['x'], data_batch['x'], s_x, s_gen]
            _temp, global_step_value = rollback(var_list, ckpt_folder, ckpt_file=ckpt_file)
            x_gen_value, x_real_value, s_x_value, s_gen_value = _temp

        # write to files
        if do_sprite:
            if real_sample:
                write_sprite_wrapper(
                    x_real_value, mesh_num, filename, file_folder=summary_folder,
                    file_index='_r_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                    if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)
            write_sprite_wrapper(
                x_gen_value, mesh_num, filename, file_folder=summary_folder,
                file_index='_g_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)

        # do visualization for code_value
        if do_embedding:
            # transpose image data if necessary
            if real_sample:
                x_as_image = np.transpose(x_real_value, axes=self.perm) if self.perm is not None else x_real_value
                x_gen_as_image = np.transpose(x_gen_value, axes=self.perm) if self.perm is not None else x_gen_value
                # concatenate real and generated images, codes and labels
                s_x_value = np.concatenate((s_x_value, s_gen_value), axis=0)
                x_as_image = np.concatenate((x_as_image, x_gen_as_image), axis=0)
                labels = np.concatenate(  # 1 for real, 0 for gen
                    (np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int)), axis=0)
                # embedding
                mesh_num = (mesh_num[0] * 2, mesh_num[1])
                embedding_image_wrapper(
                    s_x_value, filename, var_name='x_vs_xg', file_folder=summary_folder,
                    file_index='_x_vs_xg_' + sub_folder + '_' + str(global_step_value) + '_' + str(mesh_mode),
                    labels=labels, images=x_as_image, mesh_num=mesh_num,
                    if_invert=if_invert, image_format=FLAGS.IMAGE_FORMAT)

    def mdl_score(self, filename, sub_folder, batch_size, num_batch=10, model='v1', ckpt_file=None, num_threads=7,
                  grey_scale=False):
        """ This function calculates the scores for the real and generated samples

        :param filename:
        :param sub_folder:
        :param batch_size:
        :param num_batch:
        :param model: whether to use inception_v1 or inception_v3 (v3 is not working for now)
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param num_threads:
        :param grey_scale:
        :return:
        """
        # prepare folder
        ckpt_folder, summary_folder, _ = prepare_folder(filename, sub_folder=sub_folder)

        # build the network graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.init_net()
            self.sample_same_class = False
            data_batch = self.get_data_batch(filename, batch_size, num_threads=num_threads)

            # generate new images
            code_batch = self.sample_codes(batch_size)
            gen_batch = self._gpu_task(code_batch=code_batch, is_training=False)
            # do clipping
            gen_batch['x'] = tf.clip_by_value(gen_batch['x'], clip_value_min=-1, clip_value_max=1)

            metric = GenerativeModelMetric(model=model)
            if model == 'v1':
                d_b, g_b = data_batch['x'], gen_batch['x']
                if grey_scale:  # copy channel 3 times to fake RGB
                    d_b = tf.compat.v1.tile(d_b, [1, 3, 1, 1])
                    g_b = tf.compat.v1.tile(g_b, [1, 3, 1, 1])
                scores = metric.inception_score_and_fid_v1(
                    d_b, g_b, num_batch=num_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file)
            elif model == 'swd':  # swd gives nan somehow
                scores = metric.sliced_wasserstein_distance(
                    data_batch['x'], gen_batch['x'], num_batch=num_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file)
            elif model == 'ms_ssim':
                data1, data2 = tf.split(data_batch['x'], 2, axis=0)
                score_data = metric.ms_ssim(
                    data1, data2, num_batch=num_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file)
                print('MS-SSIM on real samples finished.')
                gen1, gen2 = tf.split(gen_batch['x'], 2, axis=0)
                score_gen = metric.ms_ssim(
                    gen1, gen2, num_batch=num_batch, ckpt_folder=ckpt_folder, ckpt_file=ckpt_file)
                scores = (score_data, score_gen)
            else:
                raise NotImplementedError('Model {} not implemented.'.format(model))

            return scores
