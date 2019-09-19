import tensorflow as tf

from GeneralTools.graph_func import prepare_folder
from GeneralTools.my_session import MySession
from GeneralTools.misc_fun import FLAGS


class Agent(object):
    def __init__(
            self, filename, sub_folder, load_ckpt=False, do_trace=False,
            do_save=True, debug_mode=False, debug_step=800, query_step=500,
            log_device=False, imbalanced_update=None, print_loss=True):
        """ Agent is a wrapper for the MySession class, used for training and evaluating complex model

        :param filename:
        :param sub_folder:
        :param load_ckpt:
        :param do_trace:
        :param do_save:
        :param debug_mode:
        :param log_device:
        :param query_step:
        :param imbalanced_update:
        """
        self.ckpt_folder, self.summary_folder, self.save_path = prepare_folder(filename, sub_folder=sub_folder)
        self.load_ckpt = load_ckpt
        self.do_trace = do_trace
        self.do_save = do_save
        self.debug = debug_mode
        self.debug_step = debug_step
        self.log_device = log_device
        self.query_step = query_step
        self.imbalanced_update = imbalanced_update
        self.print_loss = print_loss

    def train(
            self, op_list, loss_list, global_step, max_step=None, step_per_epoch=None,
            summary_op=None, summary_image_op=None, imbalanced_update=None, force_print=False):
        """ This method do the optimization process to minimizes loss_list

        :param op_list: [net0_op, net1_op, net2_op]
        :param loss_list: [loss0, loss1, loss2]
        :param global_step:
        :param max_step:
        :param step_per_epoch:
        :param summary_op:
        :param summary_image_op:
        :param imbalanced_update:
        :param force_print:
        :return:
        """
        # Check inputs
        if imbalanced_update is not None:
            self.imbalanced_update = imbalanced_update
        if self.imbalanced_update is not None:
            assert isinstance(self.imbalanced_update, (list, tuple, str)), \
                'Imbalanced_update must be a list, tuple or str.'

        if self.debug is None:
            # sess = tf.Session(config=tf.ConfigProto(
            #     allow_soft_placement=True,
            #     log_device_placement=False))
            writer = tf.summary.FileWriter(logdir=self.summary_folder, graph=tf.get_default_graph())
            writer.flush()
            # graph_protobuf = str(tf.get_default_graph().as_default())
            # with open(os.path.join(self.summary_folder, 'graph'), 'w') as f:
            #     f.write(graph_protobuf)
            FLAGS.print('Graph printed.')
        elif self.debug is True:
            FLAGS.print('Debug mode is on.')
            FLAGS.print('Remember to load ckpt to check variable values.')
            with MySession(self.do_save, self.do_trace, self.save_path, self.load_ckpt, self.log_device) as sess:
                sess.debug_mode(op_list, loss_list, global_step, summary_op, self.summary_folder, self.ckpt_folder,
                                max_step=self.debug_step, print_loss=self.print_loss, query_step=self.query_step,
                                imbalanced_update=self.imbalanced_update)
        elif self.debug is False:
            with MySession(self.do_save, self.do_trace, self.save_path, self.load_ckpt) as sess:
                sess.full_run(op_list, loss_list, max_step, step_per_epoch, global_step, summary_op, summary_image_op,
                              self.summary_folder, self.ckpt_folder, print_loss=self.print_loss,
                              query_step=self.query_step, imbalanced_update=self.imbalanced_update,
                              force_print=force_print)
        else:
            raise AttributeError('Current debug mode is not supported.')