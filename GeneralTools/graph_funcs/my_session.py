import json
import os.path
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from GeneralTools.graph_funcs.graph_func import get_ckpt
from GeneralTools.misc_fun import FLAGS
from dp_funcs.net_picker import NetPicker


class MySession(object):
    def __init__(
            self, do_save=False, do_trace=False, save_path=None,
            load_ckpt=False, log_device=False, ckpt_var_list=None):
        """ This class provides shortcuts for running sessions.
        It needs to be run under context managers. Example:
        with MySession() as sess:
            var1_value, var2_value = sess.run_once([var1, var2])

        :param do_save:
        :param do_trace:
        :param save_path:
        :param load_ckpt:
        :param log_device:
        :param ckpt_var_list: list of variables to save / restore
        """
        # somehow it gives error: "global_step does not exist or is not created from tf.get_variable".
        # self.global_step = global_step_config()
        self.log_device = log_device
        # register a session
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     allow_soft_placement=True,
        #     log_device_placement=log_device,
        #     gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device))
        # initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = None
        self.threads = None
        FLAGS.print('Graph initialization finished...')
        # configuration
        self.ckpt_var_list = ckpt_var_list
        if do_save:
            self.saver = self._get_saver_()
            self.save_path = save_path
        else:
            self.saver = None
            self.save_path = None
        self.summary_writer = None
        self.do_trace = do_trace
        self.load_ckpt = load_ckpt

    def __enter__(self):
        """ The enter method is called when "with" statement is used.

        :return:
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ The exit method is called when leaving the scope of "with" statement

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        FLAGS.print('Session finished.')
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.coord.request_stop()
        # self.coord.join(self.threads)
        self.sess.close()

    def _get_saver_(self):
        # create a saver to save all variables
        # Saver op should always be assigned to cpu, and it should be
        # created after all variables have been defined; otherwise, it
        # only save those variables already created.
        with tf.device('/cpu:0'):
            if self.ckpt_var_list is None:
                return tf.compat.v1.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
            else:
                return tf.compat.v1.train.Saver(var_list=self.ckpt_var_list, max_to_keep=2)

    def _load_ckpt_(self, ckpt_folder=None, ckpt_file=None, force_print=False):
        """ This function loads a checkpoint model

        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param force_print:
        :return:
        """
        if self.load_ckpt and (ckpt_folder is not None):
            ckpt = get_ckpt(ckpt_folder, ckpt_file=ckpt_file)
            if ckpt is None:
                FLAGS.print(
                    'No ckpt Model found at {}. Model training from scratch.'.format(ckpt_folder), force_print)
            else:
                if self.saver is None:
                    self.saver = self._get_saver_()
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                FLAGS.print('Model reloaded from {}.'.format(ckpt_folder), force_print)
        else:
            FLAGS.print('No ckpt model is loaded for current calculation.')

    def _check_thread_(self):
        """ This function initializes the coordinator and threads
        :return:
        """
        if self.threads is None:
            self.coord = tf.train.Coordinator()
            # self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def run_once(self, var_list, ckpt_folder=None, ckpt_file=None, ckpt_var_list=None, feed_dict=None, do_time=False):
        """ This functions calculates var_list.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :param do_time:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()

        if do_time:
            start_time = time.time()
            var_value = self.sess.run(var_list, feed_dict=feed_dict)
            FLAGS.print('Running session took {:.3f} sec.'.format(time.time() - start_time))
        else:
            var_value = self.sess.run(var_list, feed_dict=feed_dict)

        return var_value

    def run(self, *args, **kwargs):
        return self.run_once(*args, **kwargs)

    def run_m_times(
            self, var_list, ckpt_folder=None, ckpt_file=None, max_iter=10000,
            trace=False, ckpt_var_list=None, feed_dict=None):
        """ This functions calculates var_list for multiple iterations, as often done in
        Monte Carlo analysis.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param max_iter:
        :param trace: if True, keep all outputs of m iterations
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        start_time = time.time()
        if trace:
            var_value_list = []
            for i in range(max_iter):
                var_value, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
                var_value_list.append(var_value)
        else:
            for i in range(max_iter - 1):
                _, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
            var_value_list, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
        # global_step_value = self.sess.run([self.global_step])
        FLAGS.print('Calculation took {:.3f} sec.'.format(time.time() - start_time))
        return var_value_list

    @staticmethod
    def print_loss(loss_value, step=0, epoch=0):
        FLAGS.print('Epoch {}, global steps {}, loss_list {}'.format(
            epoch, step,
            ['{}'.format(['<{:.2f}>'.format(l_val) for l_val in l_value])
             if isinstance(l_value, (np.ndarray, list))
             else '<{:.3f}>'.format(l_value)
             for l_value in loss_value]))

    def save_model(self, global_step_value, summary_image_op):
        if self.saver is not None:
            self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
        if summary_image_op is not None:
            summary_image_str = self.sess.run(summary_image_op)
            self.summary_writer.add_summary(summary_image_str, global_step=global_step_value)

    def full_run(self, op_list, loss_list, max_step, step_per_epoch, global_step, summary_op=None,
                 summary_image_op=None, summary_folder=None, ckpt_folder=None, ckpt_file=None, print_loss=True,
                 query_step=500, imbalanced_update=None, force_print=False,
                 mog_model=None):
        """ This function run the session with all monitor functions.

        :param op_list: the first op in op_list runs every extra_steps when the rest run once.
        :param loss_list: the first loss is used to monitor the convergence
        :param max_step:
        :param step_per_epoch:
        :param global_step:
        :param summary_op:
        :param summary_image_op:
        :param summary_folder:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param print_loss:
        :param query_step:
        :param imbalanced_update: a list indicating the period to update each ops in op_list;
            the first op must have period = 1 as it updates the global step
        :param force_print:
        :param mog_model:
        :return:
        """
        # prepare writer
        if (summary_op is not None) or (summary_image_op is not None):
            self.summary_writer = tf.compat.v1.summary.FileWriter(summary_folder, self.sess.graph)
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file, force_print=force_print)
        # run the session
        self._check_thread_()
        extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        start_time = time.time()
        if imbalanced_update is None:  # ----------------------------------------------------- SIMULTANEOUS UPDATES HERE
            if mog_model is not None and mog_model.linked_gan.train_with_mog:
                mog_model.set_batch_encoding()

            global_step_value = None
            for step in range(max_step):

                # update MoG with current Dis params and current batch
                if mog_model is not None and mog_model.linked_gan.train_with_mog:
                    mog_model.update_by_batch(self.sess)

                    if mog_model.store_encodings:
                        if global_step_value is None:  # first iteration only
                            mog_model.store_encodings_and_params(self.sess, summary_folder, 0)

                        elif global_step_value % query_step == (query_step-1):
                            mog_model.store_encodings_and_params(self.sess, summary_folder, global_step_value)

                if not isinstance(loss_list, list):  # go from namedtuple to list
                    loss_list = list(loss_list)
                # update the model
                loss_value, _, _, global_step_value = self.sess.run(
                    [loss_list, op_list, extra_update_ops, global_step])
                # check if model produces nan outcome
                assert not any(np.isnan(loss_value)), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # maybe re-init mog after a few epochs, as it may have gotten lost given the rapid change of encodings
                if mog_model is not None and global_step_value == mog_model.re_init_at_step:
                    mog_model.init_np_mog()

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step-1) or global_step_value == 1:
                    if mog_model is not None and mog_model.means_summary_op is not None and summary_op is not None:
                        summary_str, summary_str_means = self.sess.run([summary_op, mog_model.means_summary_op])
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                        self.summary_writer.add_summary(summary_str_means, global_step=global_step_value)
                    elif summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                # save model at last step
                if step == max_step - 1:
                    self.save_model(global_step_value, summary_image_op)

        elif isinstance(imbalanced_update, (list, tuple, NetPicker)):  # <-------------------- ALTERNATING TRAINING HERE

            for step in range(max_step):  # <------------------------------------------------------ ACTUAL TRAINING LOOP
                # get update ops
                global_step_value = self.sess.run(global_step)

                if False and mog_model is not None and mog_model.linked_gan.train_with_mog:
                    if mog_model.time_to_update(global_step_value, imbalanced_update):
                        mog_model.update(self.sess)
                # IF STEP VALUE INDICATES TRAINING GENERATOR:
                # - collect all data encodings
                # - update MoG parameters
                # - proceed with training, sampling from updated MoG

                # in other places:
                # - predefine MoG distribution
                # - redefine generator loss through samples from the MoG

                update_ops = select_ops_to_update(op_list, global_step_value, imbalanced_update)  # <------ OP SELECTION

                # update the model
                loss_value, _, _ = self.sess.run([loss_list, update_ops, extra_update_ops])  # <---------- WEIGHT UPDATE
                # check if model produces nan outcome
                assert not any(np.isnan(loss_value)), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step - 1):
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                    # ------------------------------------------------------------ALSO TAKE MoG APPROXIMATION STATS HERE
                    if False and mog_model is not None and not mog_model.linked_gan.train_with_mog:
                        mog_model.test_mog_approx(self.sess)

                # save model at last step
                if step == max_step - 1:
                    self.save_model(global_step_value, summary_image_op)

        elif imbalanced_update == 'dynamic':
            # This case is used for sngan_mmd_rand_g only
            mmd_average = 0.0
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = op_list if \
                    global_step_value < 1000 or \
                    np.random.uniform(low=0.0, high=1.0) < 0.1 / np.maximum(mmd_average, 0.1) else \
                    op_list[1:]

                # update the model
                loss_value, _, _, global_step_value = self.sess.run([loss_list, update_ops, extra_update_ops])
                # check if model produces nan outcome
                assert not any(np.isnan(loss_value)), \
                    'Model diverged with loss = {} at step {}'.format(loss_value, step)

                # add summary and print loss every query step
                if global_step_value % query_step == (query_step - 1):
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                    if print_loss:
                        epoch = step // step_per_epoch
                        self.print_loss(loss_value, global_step_value, epoch)

                # save model at last step
                if step == max_step - 1:
                    self.save_model(global_step_value, summary_image_op)

        # calculate sess duration
        duration = time.time() - start_time
        FLAGS.print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))

    def abnormal_save(self, loss_value, global_step_value, summary_op):
        """ This function save the model in abnormal cases

        :param loss_value:
        :param global_step_value:
        :param summary_op:
        :return:
        """
        if any(np.isnan(loss_value)):
            # save the model
            if self.saver is not None:
                self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
            warnings.warn('Training Stopped due to nan in loss: {}.'.format(loss_value))
            return True
        elif any(np.greater(loss_value, 30000)):
            # save the model
            if self.saver is not None:
                self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
            # add summary
            if summary_op is not None:
                summary_str = self.sess.run(summary_op)
                self.summary_writer.add_summary(summary_str, global_step=global_step_value)
            warnings.warn('Training Stopped early as loss diverged.')
            return True
        else:
            return False

    def summary_and_save(self, summary_op, global_step_value, loss_value, step, max_step):
        if (summary_op is not None) and (global_step_value % 100 == 99):
            summary_str = self.sess.run(summary_op)
            # add summary and print out loss
            self.summary_writer.add_summary(summary_str, global_step=global_step_value)

        # in abnormal cases, save the model
        if self.abnormal_save(loss_value, global_step_value, summary_op):
            return 'break'

        if step == max_step - 1 and self.saver is not None:
            self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
        return None

    def do_imbalanced_update(self, step, max_step, loss_list, update_ops, extra_update_ops, run_options, run_metadata,
                             global_step_value, multi_runs_timeline):
        if self.do_trace and (step >= max_step - 5):
            # update the model in trace mode
            loss_value, _, _ = self.sess.run(
                [loss_list, update_ops, extra_update_ops],
                options=run_options, run_metadata=run_metadata)
            # add time line
            self.summary_writer.add_run_metadata(
                run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            chrome_trace = trace.generate_chrome_trace_format()
            multi_runs_timeline.update_timeline(chrome_trace)
        else:
            # update the model
            loss_value, _, _ = self.sess.run([loss_list, update_ops, extra_update_ops])
        return loss_value

    def debug_mode(self, op_list, loss_list, global_step, summary_op=None, summary_folder=None, ckpt_folder=None,
                   ckpt_file=None, max_step=200, print_loss=True, query_step=100, imbalanced_update=None):
        """ This function do tracing to debug the code. It will burn-in for 25 steps, then record
        the usage every 5 steps for 5 times.

        :param op_list:
        :param loss_list:
        :param global_step:
        :param summary_op:
        :param summary_folder:
        :param max_step:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param print_loss:
        :param query_step:
        :param imbalanced_update: a list indicating the period to update each ops in op_list;
            the first op must have period = 1 as it updates the global step
        :return:
        """
        if self.do_trace or (summary_op is not None):
            self.summary_writer = tf.compat.v1.summary.FileWriter(summary_folder, self.sess.graph)
        if self.do_trace:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            multi_runs_timeline = TimeLiner()
        else:
            run_options = None
            run_metadata = None
            multi_runs_timeline = None
        if query_step > max_step:
            query_step = np.minimum(max_step-1, 100)

        # run the session
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        # print(extra_update_ops)
        start_time = time.time()
        if imbalanced_update is None:
            for step in range(max_step):
                if self.do_trace and (step >= max_step - 5):
                    # update the model in trace mode
                    loss_value, _, global_step_value, _ = self.sess.run(
                        [loss_list, op_list, global_step, extra_update_ops],
                        options=run_options, run_metadata=run_metadata)
                    # add time line
                    self.summary_writer.add_run_metadata(
                        run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    chrome_trace = trace.generate_chrome_trace_format()
                    multi_runs_timeline.update_timeline(chrome_trace)
                else:
                    # update the model
                    loss_value, _, global_step_value, _ = self.sess.run(
                        [loss_list, op_list, global_step, extra_update_ops])

                # print(loss_value) and add summary
                if global_step_value % query_step == 1:  # at step 0, global step = 1
                    if print_loss:
                        self.print_loss(loss_value, global_step_value)
                    if summary_op is not None:
                        summary_str = self.sess.run(summary_op)
                        self.summary_writer.add_summary(summary_str, global_step=global_step_value)

                # in abnormal cases, save the model
                if self.abnormal_save(loss_value, global_step_value, summary_op):
                    break

                # save the mdl if for loop completes normally
                if step == max_step - 1 and self.saver is not None:
                    self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)

        elif isinstance(imbalanced_update, (list, tuple)):
            num_ops = len(op_list)
            assert len(imbalanced_update) == num_ops, 'Imbalanced_update length does not match ' \
                                                      'that of op_list. Expected {} got {}.'.format(
                num_ops, len(imbalanced_update))
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                # added function to take care of added negative option
                # update_ops = [op_list[i] for i in range(num_ops) if global_step_value % imbalanced_update[i] == 0]
                update_ops = select_ops_to_update(op_list, global_step_value, imbalanced_update)

                loss_value = self.do_imbalanced_update(step, max_step, loss_list, update_ops, extra_update_ops,
                                                       run_options, run_metadata, global_step_value,
                                                       multi_runs_timeline)

                # print(loss_value)
                if print_loss and (step % query_step == 0):
                    self.print_loss(loss_value, global_step_value)

                if self.summary_and_save(summary_op, global_step_value, loss_value, step, max_step) == 'break':
                    break

        elif isinstance(imbalanced_update, str) and imbalanced_update == 'dynamic':
            # This case is used for sngan_mmd_rand_g only
            mmd_average = 0.0
            for step in range(max_step):
                # get update ops
                global_step_value = self.sess.run(global_step)
                update_ops = op_list if \
                    global_step_value < 1000 or \
                    np.random.uniform(low=0.0, high=1.0) < 0.1 / np.maximum(mmd_average, 0.1) else \
                    op_list[1:]

                loss_value = self.do_imbalanced_update(step, max_step, loss_list, update_ops, extra_update_ops,
                                                       run_options, run_metadata, global_step_value,
                                                       multi_runs_timeline)

                # update mmd_average
                mmd_average = loss_value[2]

                # print(loss_value)
                if print_loss and (step % query_step == 0):
                    self.print_loss(loss_value, global_step_value)

                if self.summary_and_save(summary_op, global_step_value, loss_value, step, max_step) == 'break':
                    break

        # calculate sess duration
        duration = time.time() - start_time
        FLAGS.print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))
        # save tracing file
        if self.do_trace:
            trace_file = os.path.join(summary_folder, 'timeline.json')
            multi_runs_timeline.save(trace_file)


def select_ops_to_update(op_list, global_step_value, imbalanced_update):
    if isinstance(imbalanced_update, NetPicker):
        return imbalanced_update.pick_ops(op_list)
    elif not [k for k in imbalanced_update if k < 0]:  # no negative values -> select as usual
        update_ops = [op_list[i] for i in range(len(op_list)) if global_step_value % imbalanced_update[i] == 0]
    else:  # -> in addition, select all negative vals which don't divide the step count
        update_ops = []
        for idx, iup in enumerate(imbalanced_update):
            if iup > 0 and global_step_value % iup == 0:
                update_ops.append(op_list[idx])
            elif iup < 0 and global_step_value % iup != 0:
                update_ops.append(op_list[idx])
    return update_ops


class TimeLiner:
    def __init__(self):
        """ This class creates a timeline object that can be used to trace the timeline of
            multiple steps when called at each step.

        """
        self._timeline_dict = None

    ###################################################################
    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    ###################################################################
    def save(self, trace_file):
        with open(trace_file, 'w') as f:
            json.dump(self._timeline_dict, f)
