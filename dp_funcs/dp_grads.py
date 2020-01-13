# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified version of the compute_gradients function from the dp_optimizer at tensorflow_privacy/privacy/optimizers
# adapted to the RFF-MMD setting where we don't exactly do gradient clipping
# but clip and perturb parts of the gradient instead
# also removed all the bookkeeping including query objects. dp analysis is done separately
import tensorflow as tf
GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP
GATE_GRAPH = tf.compat.v1.train.Optimizer.GATE_GRAPH


def dp_rff_gradients(loss, var_list, l2_norm_clip, noise_factor, clip_by_layer_norm=False):
  nest = tf.contrib.framework.nest
  batch_size = loss.get_shape()[0]
  rff_dim = loss.get_shape()[1]

  def clip_global(record_as_list):
    return tf.clip_by_global_norm(record_as_list, l2_norm_clip)[0]

  def clip_layerwise(record_as_list):
    return [tf.clip_by_norm(k, n) for k, n in zip(record_as_list, l2_norm_clip)]

  clip_fun = clip_global if not clip_by_layer_norm else clip_layerwise

  def process_sample_loss(i, grad_acc):
    """Process one microbatch (record) with privacy helper."""
    grads_list = sample_grads(loss[i, :], var_list)
    # grads_list = zip(grads_n_vars)  # get grads
    # source: DPQuery.accumulate_record in gaussianquery.py
    # GaussianSumQuery.preprocess_record in dp_query.py

    grads_list = list(grads_list)
    record_as_list = nest.flatten(grads_list)  # flattening list. should already be flat after removing queries

    clipped_as_list = clip_fun(record_as_list)
    preprocessed_record = clipped_as_list

    # preprocessed_record, norm = tf.clip_by_global_norm(grads_list, l2_norm_clip)  # trying this simpler line for now

    return nest.map_structure(tf.add, grad_acc, preprocessed_record)  # add clipped sample grad to sum of grads

  def zeros_like(arg):
    try:
      arg = tf.convert_to_tensor(arg)  # pretty sure everything is a tensor at this point, try removing this...
    except TypeError:
      pass
    return tf.zeros([rff_dim] + list(arg.shape), arg.dtype)  # aggregate gradients have extra rff dimension in this case

  sample_state = nest.map_structure(zeros_like, var_list)

  def cond(i, _state):
    return tf.less(i, batch_size)

  def body(i, state):
    new_state = process_sample_loss(i, state)
    # with tf.control_dependencies(new_state):
    new_count = tf.add(i, 1)
    return [new_count, new_state]

  _, sample_state = tf.while_loop(cond=cond, body=body, loop_vars=[tf.constant(0), sample_state], name='dp_grads_while',
                                  parallel_iterations=100)

  # # unrolling microbatches does not seem like an option right now - may be worth revisiting after trying while loop
  # _, sample_state = tf.while_loop(cond=lambda i, _: tf.less(i, batch_size),
  #                                 body=lambda i, state: [tf.add(i, 1), process_sample_loss(i, state)],
  #                                 loop_vars=[tf.constant(0), sample_state], name='dp_grads_while',
  #                                 parallel_iterations=1)
  with tf.name_scope(None):  # return to root scope to avoid scope overlap
    tf.compat.v1.summary.scalar('DPSGD/grad_norm_avg_post_clip_global',
                                tf.linalg.global_norm(sample_state)/float(batch_size))
    for idx, grad in enumerate(sample_state):
      tf.compat.v1.summary.scalar(f'DPSGD/grad_norm_avg_post_clip_tensor_{idx}', tf.norm(grad)/float(batch_size))

  # grad_sums, global_state = dp_sum_query.get_noised_result(sample_state, global_state)
  def add_noise_global(grads):
    return nest.map_structure(lambda k: k + tf.random.normal(tf.shape(k), stddev=l2_norm_clip * noise_factor), grads)

  def add_noise_layerwise(grads):
    return [k + tf.random.normal(tf.shape(k), stddev=n * noise_factor) for k, n in zip(grads, l2_norm_clip)]

  final_grads = add_noise_global(sample_state) if not clip_by_layer_norm else add_noise_layerwise(sample_state)
  # normalization comes later

  return final_grads


def sample_grads(sample_loss, var_list):
  # all gradiend beloning to one sample (i.e. #RFF many)
  n_rff = sample_loss.get_shape()[0]
  grads = [single_grad(sample_loss[i], var_list) for i in range(n_rff)]
  # THIS IS WHERE WE COULD CLIP PER LOSS DIMENSION IF THAT SEEMS LIKE IT WILL BE USEFUL
  # if 1 % 1 > 1:  # don't do it for now
  #   made_up_clip = 2.
  #   grads = [tf.clip_by_global_norm(k, made_up_clip)[0] for k in grads]
  grad_stack = [tf.stack(k) for k in zip(*grads)]
  return grad_stack


def single_grad(loss, var_list):
  # compute a gradients for a single scalar loss associated with one sample
  # grads, _ = zip(*optimizer.compute_gradients(loss, var_list, gate_gradients=GATE_GRAPH))
  # print('-----------------------calling single grad')
  grads = tf.gradients(loss, var_list, gate_gradients=None)
  # fill up none gradients with zeros
  grads_list = [g if g is not None else tf.zeros_like(v) for (g, v) in zip(list(grads), var_list)]
  return grads_list


def compose_full_grads(fx_dp, fy, dfx_dp, dfy, batch_size):
  """
  once the data-dependent gradient components have been noised up, the full mmd gradient can be assembled
  :param fx_dp: noised up sum(f_w(x))
  :param fy: sum(f_w(y))
  :param dfx_dp: noised up d/dw sum(f_w(x))
  :param dfy: d/dw sum(f_w(y))
  :param batch_size: for normalization
  :return:
  """
  loss_diff_normed = -2 * (fx_dp - fy) / batch_size**2  # (rff_dims)

  def full_grad(x, y):
    # print('grad_x_shape', x.get_shape())  # (rff, w_dims)
    # print('grad_y_shape', y.get_shape())
    # print('loss_shape', loss_diff_normed.get_shape())  # (rff)
    # return tf.malmul(loss_diff_normed, x - y)  # dimensions are not clear here. target shape: (w_dims)
    ldn = loss_diff_normed
    for _ in range(len(x.get_shape()) - 1):
      ldn = tf.expand_dims(ldn, axis=-1)
    grad = tf.reduce_sum(ldn * (x - y), axis=0)

    # print('comp_grad_shape', grad.get_shape())
    return grad

  return tf.contrib.framework.nest.map_structure(full_grad, dfx_dp, dfy)


def release_loss_dis(loss_dis, l2_norm_clip, noise_factor):
  """
  clips each sample by l2 norm then adds and sums
  :param loss_dis:  shape (bs, rff)
  :param l2_norm_clip:
  :param noise_factor:
  :return:
  """
  with tf.name_scope(None):  # return to root scope to avoid scope overlap
    tf.compat.v1.summary.scalar('DPSGD/loss_norm_avg_pre_clip', tf.reduce_mean(tf.norm(loss_dis, axis=1)))

  loss_clip = tf.clip_by_norm(loss_dis, l2_norm_clip, axes=1)

  loss_sum = tf.reduce_sum(loss_clip, axis=0)
  return loss_sum + tf.random.normal(tf.shape(loss_sum), stddev=l2_norm_clip * noise_factor)


def loss_dis_from_rff(rff_dis_loss, rff_gen_loss, batch_size):
  """
  rff mmd^2 estimator = 1/n^2 || sum_i(f(x_i)) - sum_i(f(y_i))||_2^2
  :param rff_dis_loss: batch of rff data encodings (bs, rff)
  :param rff_gen_loss: sum over batch of gen encodings (rff)
  :param batch_size
  :return:
  """
  return -tf.reduce_sum((tf.reduce_sum(rff_dis_loss, axis=0) - rff_gen_loss)**2) / batch_size**2


def dp_compute_grads(loss_ops, dp_spec, vars_dis, vars_gen):
  """
  computes dp gradients for discriminator update with rff mmd estimator

  :param loss_ops:
  :param dp_spec:
  :param vars_dis
  :param vars_gen:
  :return: lists of gradient-variable pairs for generator and discriminator updates
  """
  batch_size = int(loss_ops.dis.fdat.get_shape()[0])
  # discriminator op
  # - compute the partial gradients
  grad_rff_dis_release = dp_rff_gradients(loss_ops.dis.fdat, vars_dis, dp_spec.grad_clip, dp_spec.grad_noise,
                                          dp_spec.clip_by_layer_norm)
  grad_rff_gen = sample_grads(loss_ops.dis.fgen, vars_dis)
  # - clip & perturb loss
  loss_rff_dis_release = release_loss_dis(loss_ops.dis.fdat, dp_spec.loss_clip, dp_spec.loss_noise)

  # get full grads, then couple with variables
  grads = compose_full_grads(fx_dp=loss_rff_dis_release, fy=loss_ops.dis.fgen,
                             dfx_dp=grad_rff_dis_release, dfy=grad_rff_gen, batch_size=batch_size)
  grads_n_vars_dis = list(zip(grads, vars_dis))

  # compute the actual discriminator loss for completion
  loss_dis = loss_dis_from_rff(loss_ops.dis.fdat, loss_ops.dis.fgen, batch_size)

  # generator op:
  # grads_n_vars_gen = opt_ops.gen.compute_gradients(loss_ops.gen, var_list=vars_gen)
  grads_n_vars_gen = list(zip(tf.gradients(loss_ops.gen, vars_gen), vars_gen))

  grads_list = [grads_n_vars_dis, grads_n_vars_gen]
  loss_list = [loss_dis, loss_ops.gen]
  return grads_list, loss_list
