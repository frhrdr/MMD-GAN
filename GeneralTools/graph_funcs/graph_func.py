# import math
import os.path
import numpy as np
import tensorflow as tf
# import plotly as py
import warnings

from GeneralTools.misc_fun import FLAGS
from tf_privacy.optimizers.dp_optimizer_vectorized import VectorizedDPAdam


def prepare_folder(filename, sub_folder='', set_folder=True):
    """ This function prepares the folders

    :param filename:
    :param sub_folder:
    :param set_folder:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    ckpt_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_ckpt', sub_folder)
    if not os.path.exists(ckpt_folder) and set_folder:
        os.makedirs(ckpt_folder)
    summary_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_log', sub_folder)
    if not os.path.exists(summary_folder) and set_folder:
        os.makedirs(summary_folder)
    save_path = os.path.join(ckpt_folder, filename + '.ckpt')

    return ckpt_folder, summary_folder, save_path


def prepare_embedding_folder(summary_folder, filename, file_index=''):
    """ This function prepares the files for embedding

    :param summary_folder:
    :param filename:
    :param file_index:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    embedding_path = os.path.join(summary_folder, filename + file_index + '_embedding.ckpt')
    label_path = os.path.join(summary_folder, filename + file_index + '_label.tsv')
    sprite_path = os.path.join(summary_folder, filename + file_index + '.png')

    return embedding_path, label_path, sprite_path


def write_metadata(label_path, labels, names=None):
    """ This function writes raw_labels to file for embedding

    :param label_path: file name, e.g. '...\\metadata.tsv'
    :param labels: raw_labels
    :param names: interpretation for raw_labels, e.g. ['plane','auto','bird','cat']
    :return:
    """
    metadata_file = open(label_path, 'w')
    metadata_file.write('Name\tClass\n')
    if names is None:
        i = 0
        for label in labels:
            metadata_file.write('%06d\t%s\n' % (i, str(label)))
            i = i + 1
    else:
        for label in labels:
            metadata_file.write(names[label])
    metadata_file.close()


def write_sprite(sprite_path, images, mesh_num=None, if_invert=False):
    """ This function writes images to sprite image for embedding

    This function was taken from:
    https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

    The input image must be channels_last format.

    :param sprite_path: file name, e.g. '...\\a_sprite.png'
    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param mesh_num: nums of images in the row and column, must be a tuple
    :return:
    """
    if len(images.shape) == 3:  # if dimension of image is 3, extend it to 4
        images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))
    if images.shape[3] == 1:  # if last dimension is 1, extend it to 3
        images = np.tile(images, (1, 1, 1, 3))
    # scale image to range [0,1]
    images = images.astype(np.float32)
    image_min = np.min(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) - image_min).transpose((3, 0, 1, 2))
    image_max = np.max(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) / image_max).transpose((3, 0, 1, 2))
    if if_invert:
        images = 1 - images
    # check mesh_num
    if mesh_num is None:
        FLAGS.print('Mesh_num will be calculated as sqrt of batch_size')
        batch_size = images.shape[0]
        sprite_size = int(np.ceil(np.sqrt(batch_size)))
        mesh_num = (sprite_size, sprite_size)
        # add paddings if batch_size is not the square of sprite_size
        padding = ((0, sprite_size ** 2 - batch_size), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
        images = np.pad(images, padding, mode='constant', constant_values=0)
    elif isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    # Tile the individual thumbnails into an image
    new_shape = mesh_num + images.shape[1:]
    images = images.reshape(new_shape).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((mesh_num[0] * images.shape[1], mesh_num[1] * images.shape[3]) + images.shape[4:])
    images = (images * 255).astype(np.uint8)
    # save images to file
    # from scipy.misc import imsave
    # imsave(sprite_path, images)
    try:
        from imageio import imwrite
        imwrite(sprite_path, images)
    except:
        print('attempt to write image failed!')


def write_sprite_wrapper(
        images, mesh_num, filename, file_folder=None, file_index='',
        if_invert=False, image_format='channels_last'):
    """ This is a wrapper function for write_sprite.

    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num: mus tbe tuple (row_mesh, column_mesh)
    :param filename:
    :param file_folder:
    :param file_index:
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    if image_format in {'channels_first', 'NCHW'}:  # convert to [batch_size, height, width, channels]
        images = np.transpose(images, axes=(0, 2, 3, 1))
    # set up file location
    sprite_path = os.path.join(file_folder, filename + file_index + '.png')
    # write to files
    if os.path.isfile(sprite_path):
        warnings.warn('This file already exists: ' + sprite_path)
    else:
        write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)


def embedding_latent_code(
        latent_code, file_folder, embedding_path, var_name='codes',
        label_path=None, sprite_path=None, image_size=None):
    """ This function visualize latent_code using tSNE or PCA. The results can be viewed
    on tensorboard.

    :param latent_code: 2-D data
    :param file_folder:
    :param embedding_path:
    :param var_name:
    :param label_path:
    :param sprite_path:
    :param image_size:
    :return:
    """
    # register a session
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    # prepare a embedding variable
    # note this must be a variable, not a tensor
    embedding_var = tf.Variable(latent_code, name=var_name)
    sess.run(embedding_var.initializer)

    # configure the embedding
    from tensorflow.contrib.tensorboard.plugins import projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # add metadata (label) to embedding; comment out if no metadata
    if label_path is not None:
        embedding.metadata_path = label_path
    # add sprite image to embedding; comment out if no sprites
    if sprite_path is not None:
        embedding.sprite.image_path = sprite_path
        embedding.sprite.single_image_dim.extend(image_size)
    # finalize embedding setting
    embedding_writer = tf.summary.FileWriter(file_folder)
    projector.visualize_embeddings(embedding_writer, config)
    embedding_saver = tf.train.Saver([embedding_var], max_to_keep=1)
    embedding_saver.save(sess, embedding_path)
    # close all
    sess.close()


def embedding_image_wrapper(
        latent_code, filename, var_name='codes', file_folder=None, file_index='',
        labels=None, images=None, mesh_num=None, if_invert=False, image_format='channels_last'):
    """ This function is a wrapper function for embedding_image

    :param latent_code:
    :param filename:
    :param var_name:
    :param file_folder:
    :param file_index:
    :param labels:
    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num:
    :param if_invert:
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    # prepare folder
    embedding_path, label_path, sprite_path = prepare_embedding_folder(file_folder, filename, file_index)
    # write label to file if labels are given
    if labels is not None:
        if os.path.isfile(label_path):
            warnings.warn('Label file {} already exist.'.format(label_path))
        else:
            write_metadata(label_path, labels)
    else:
        label_path = None
    # write images to file if images are given
    if images is not None:
        # if image is in channels_first format, convert to channels_last
        if image_format == 'channels_first':
            images = np.transpose(images, axes=(0, 2, 3, 1))
        image_size = images.shape[1:3]  # [height, width]
        if os.path.isfile(sprite_path):
            warnings.warn('Sprite file {} already exist.'.format(sprite_path))
        else:
            write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)
    else:
        image_size = None
        sprite_path = None
    if os.path.isfile(embedding_path):
        warnings.warn('Embedding file {} already exist.'.format(embedding_path))
    else:
        embedding_latent_code(
            latent_code, file_folder, embedding_path, var_name=var_name,
            label_path=label_path, sprite_path=sprite_path, image_size=image_size)


def get_ckpt(ckpt_folder, ckpt_file=None):
    """ This function gets the ckpt states. In case an older ckpt file is needed, provide the name in ckpt_file

    :param ckpt_folder:
    :param ckpt_file:
    :return:
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    if ckpt_file is None:
        return ckpt
    else:
        index_file = os.path.join(ckpt_folder, ckpt_file+'.index')
        if os.path.isfile(index_file):
            ckpt.model_checkpoint_path = os.path.join(ckpt_folder, ckpt_file)
        else:
            raise FileExistsError('{} does not exist.'.format(index_file))

        return ckpt


def global_step_config(name='global_step'):
    """ This function is a wrapper for global step

    """
    global_step = tf.compat.v1.get_variable(
        name=name,
        shape=[],
        dtype=tf.int32,
        initializer=tf.constant_initializer(0),
        trainable=False)

    return global_step


def opt_config(
        initial_lr, lr_decay_steps=None, end_lr=1e-7,
        optimizer='adam', name_suffix='', global_step=None, target_step=1e5, dp_specs=None):
    """ This function configures optimizer.

    :param initial_lr:
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer:
    :param name_suffix:
    :param global_step:
    :param target_step:
    :return:
    """

    assert dp_specs is None or optimizer == 'adam'  # only implementing dp for adam for now

    if optimizer in ['SGD', 'sgd']:
        # sgd
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.GradientDescentOptimizer(
            learning_rate, name='GradientDescent'+name_suffix)
        FLAGS.print('GradientDescent Optimizer is used.')
    elif optimizer in ['Momentum', 'momentum']:
        # momentum
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.MomentumOptimizer(
            learning_rate, momentum=0.9, name='Momentum'+name_suffix)
        FLAGS.print('Momentum Optimizer is used.')
    elif optimizer in ['Adam', 'adam']:  # adam
        # Occasionally, adam optimizer may cause the objective to become nan in the first few steps
        # This is because at initialization, the gradients may be very big. Setting beta1 and beta2
        # close to 1 may prevent this.
        learning_rate = tf.constant(initial_lr)
        # opt_op = tf.train.AdamOptimizer(
        #     learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8, name='Adam'+name_suffix)
        if dp_specs is None:
            opt_op = tf.compat.v1.train.AdamOptimizer(
                learning_rate, beta1=0.5, beta2=0.999, epsilon=1e-8, name='Adam' + name_suffix)
            FLAGS.print('Adam Optimizer is used.')
        else:
            opt_op = VectorizedDPAdam(
                learning_rate, beta1=0.5, beta2=0.999, epsilon=1e-8, name='DPAdam' + name_suffix)
            FLAGS.print('DP-Adam Optimizer is used.')
    elif optimizer in ['RMSProp', 'rmsprop']:
        # RMSProp
        learning_rate = tf.constant(initial_lr)
        opt_op = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp'+name_suffix)
        FLAGS.print('RMSProp Optimizer is used.')
    else:
        raise AttributeError('Optimizer {} not supported.'.format(optimizer))

    return learning_rate, opt_op


def multi_opt_config(
        lr_list, lr_decay_steps=None, end_lr=1e-7,
        optimizer='adam', global_step=None, target_step=1e5):
    """ This function configures multiple optimizer

    :param lr_list: a list, e.g. [1e-4, 1e-3]
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer: a string, or a list same len as lr_multiplier
    :param global_step:
    :param target_step:
    :return:
    """
    num_opt = len(lr_list)
    if isinstance(optimizer, str):
        optimizer = [optimizer]
    # if one lr_multiplier is provided, configure one op
    # in this case, multi_opt_config is the same as opt_config
    if num_opt == 1:
        learning_rate, opt_op = opt_config(
            lr_list[0], lr_decay_steps, end_lr,
            optimizer[0], '', global_step, target_step)
    else:
        if len(optimizer) == 1:  # match the length of lr_multiplier
            optimizer = optimizer*num_opt
        # get a list of (lr, opt_op) tuple
        lr_opt_combo = [
            opt_config(
                lr_list[i], lr_decay_steps, end_lr,
                optimizer[i], '_'+str(i), global_step, target_step)
            for i in range(num_opt)]
        # separate lr and opt_op
        learning_rate = [lr_opt[0] for lr_opt in lr_opt_combo]
        opt_op = [lr_opt[1] for lr_opt in lr_opt_combo]

    return learning_rate, opt_op


def rollback(var_list, ckpt_folder, ckpt_file=None):
    """ This function provides a shortcut for reloading a model and calculating a list of variables

    :param var_list:
    :param ckpt_folder:
    :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
    :return:
    """
    global_step = global_step_config()
    # register a session
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    # initialization
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # load the training graph
    saver = tf.train.Saver(max_to_keep=2)
    ckpt = get_ckpt(ckpt_folder, ckpt_file=ckpt_file)
    if ckpt is None:
        raise FileNotFoundError('No ckpt Model found at {}.'.format(ckpt_folder))
    saver.restore(sess, ckpt.model_checkpoint_path)
    FLAGS.print('Model reloaded.')
    # run the session
    coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    var_value, global_step_value = sess.run([var_list, global_step])
    coord.request_stop()
    # coord.join(threads)
    sess.close()
    FLAGS.print('Variable calculated.')

    return var_value, global_step_value


