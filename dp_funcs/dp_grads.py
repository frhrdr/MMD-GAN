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
GATE_OP = tf.train.Optimizer.GATE_OP
GATE_GRAPH = tf.train.Optimizer.GATE_GRAPH


def dp_rff_gradients(optimizer, loss, var_list, l2_norm_clip, noise_factor):
  nest = tf.contrib.framework.nest
  batch_size = loss.get_shape()[0]
  rff_dim = loss.get_shape()[1]

  def process_sample_loss(i, sample_state):
    """Process one microbatch (record) with privacy helper."""
    print('------------------------------------------------------calling process_sample_loss')
    grads_list = zip(sample_grads(loss[i, :], optimizer, var_list))  # get grads
    print('------------------------------------------------------called process_sample_loss')
    # source: DPQuery.accumulate_record in gaussianquery.py
    # GaussianSumQuery.preprocess_record in dp_query.py

    grads_list = list(grads_list)
    print('g', grads_list)
    record_as_list = nest.flatten(grads_list)  # flattening list. should already be flat after removing queries
    print('r', record_as_list)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    print('c', clipped_as_list)
    print('n', norm)
    preprocessed_record = clipped_as_list
    print('p', preprocessed_record)

    # preprocessed_record, norm = tf.clip_by_global_norm(grads_list, l2_norm_clip)  # trying this simpler line for now

    return nest.map_structure(tf.add, sample_state, preprocessed_record)  # add clipped sample grad to sum of grads

  def zeros_like(arg):
    try:
      arg = tf.convert_to_tensor(arg)  # pretty sure everything is a tensor at this point, try removing this...
    except TypeError:
      pass
    print([rff_dim] + list(arg.shape))
    return tf.zeros([rff_dim] + list(arg.shape), arg.dtype)  # aggregate gradients have extra rff dimension in this case

  sample_state = nest.map_structure(zeros_like, var_list)

  # unrolling microbatches does not seem like an option right now - may be worth revisiting after trying while loop
  _, sample_state = tf.while_loop(cond=lambda i, _: tf.less(i, batch_size),
                                  body=lambda i, state: [tf.add(i, 1), process_sample_loss(i, state)],
                                  loop_vars=[tf.constant(0), sample_state], name='dp_grads_while')

  # grad_sums, global_state = dp_sum_query.get_noised_result(sample_state, global_state)
  def add_noise(v):
    return v + tf.random_normal(tf.shape(v), stddev=noise_factor)

  final_grads = nest.map_structure(add_noise, sample_state)  # normalization comes later
  return final_grads


def single_grad(loss, optimizer, var_list):
  # compute a gradients for a single scalar loss associated with one sample
  # grads, _ = zip(*optimizer.compute_gradients(loss, var_list, gate_gradients=GATE_GRAPH))
  print('-----------------------calling single grad')
  grads = tf.gradients(loss, var_list, gate_gradients=GATE_GRAPH)
  # fill up none gradients with zeros
  grads_list = [g if g is not None else tf.zeros_like(v) for (g, v) in zip(list(grads), var_list)]
  return grads_list


def sample_grads(sample_loss, optimizer, var_list):
  # all gradiend beloning to one sample (i.e. #RFF many)
  print('-----------------------calling sample grads')
  n_rff = sample_loss.get_shape()[0]
  grads = [single_grad(sample_loss[i], optimizer, var_list) for i in range(n_rff)]
  print('-----------------------stacking sample grads')
  grad_stack = [tf.stack(k) for k in zip(*grads)]
  print('-----------------------returning sample grads')
  return grad_stack


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
  loss_diff_normed = (fx_dp - fy) / batch_size**2  # (rff_dims)

  def full_grad(x, y):
    print('grad_x_shape', x.get_shape())  # (rff, w_dims)
    print('grad_y_shape', y.get_shape())
    print('loss_shape', loss_diff_normed.get_shape())  # (rff)
    # return tf.malmul(loss_diff_normed, x - y)  # dimensions are not clear here. target shape: (w_dims)
    ldn = loss_diff_normed
    for _ in range(len(x.get_shape()) - 1):
      ldn = tf.expand_dims(ldn, axis=-1)
    grad = tf.reduce_sum(ldn * (x - y), axis=0)

    print('comp_grad_shape', grad.get_shape())
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
  loss_clip = tf.clip_by_norm(loss_dis, l2_norm_clip, axes=1)
  loss_sum = tf.reduce_sum(loss_clip, axis=0)
  return loss_sum + tf.random_normal(tf.shape(loss_sum), stddev=noise_factor)


def loss_dis_from_rff(rff_dis_loss, rff_gen_loss, batch_size):
  """
  rff mmd^2 estimator = 1/n^2 || sum_i(f(x_i)) - sum_i(f(y_i))||_2^2
  :param rff_dis_loss: batch of rff data encodings (bs, rff)
  :param rff_gen_loss: sum over batch of gen encodings (rff)
  :param batch_size
  :return:
  """
  return tf.reduce_sum((tf.reduce_sum(rff_dis_loss, axis=0) - rff_gen_loss)**2) / batch_size**2


def dp_compute_grads(loss_ops, opt_ops, dp_spec, vars_dis, vars_gen):
  """
  computes dp gradients for discriminator update with rff mmd estimator

  :param loss_ops:
  :param opt_ops: optimizer for gen and dis
  :param dp_spec:
  :param vars_dis
  :param vars_gen:
  :return: lists of gradient-variable pairs for generator and discriminator updates
  """
  batch_size = int(loss_ops.dis.fdat.get_shape()[0])
  # generator op:
  grads_n_vars_gen = opt_ops.gen.compute_gradients(loss_ops.gen, var_list=vars_gen)
  # discriminator op
  # - compute the partial gradients
  grad_rff_gen = sample_grads(loss_ops.dis.fgen, opt_ops.dis, vars_dis)
  grad_rff_dis_release = dp_rff_gradients(opt_ops.dis, loss_ops.dis.fdat, vars_dis,
                                          dp_spec['grad_clip'], dp_spec['grad_noise'])
  # - clip & perturb loss
  loss_rff_dis_release = release_loss_dis(loss_ops.dis.fdat, dp_spec['loss_clip'], dp_spec['loss_noise'])

  # get full grads, then couple with variables
  grads = compose_full_grads(fx_dp=loss_rff_dis_release, fy=loss_ops.dis.fgen,
                             dfx_dp=grad_rff_dis_release, dfy=grad_rff_gen, batch_size=batch_size)
  grads_n_vars_dis = list(zip(grads, vars_dis))

  # compute the actual discriminator loss for completion
  loss_dis = loss_dis_from_rff(loss_ops.dis.fdat, loss_ops.dis.fgen, batch_size)

  grads_list = [grads_n_vars_dis, grads_n_vars_gen]
  loss_list = [loss_dis, loss_ops.gen]
  return grads_list, loss_list
