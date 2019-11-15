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


def dp_rff_gradients(optimizer, loss, var_list, l2_norm_clip, noise_factor, aggregation_method=None):
  nest = tf.contrib.framework.nest
  batch_size = tf.shape(loss)[0]
  rff_dim = tf.shape(loss)[1]
  gate_gradients = tf.train.Optimizer.GATE_OP

  def process_sample_loss(i, sample_state):
    """Process one microbatch (record) with privacy helper."""

    grads_list = zip(sample_grads(loss[i, :], optimizer, var_list, gate_gradients, aggregation_method))  # get grads

    # source: DPQuery.accumulate_record in gaussianquery.py
    # GaussianSumQuery.preprocess_record in dp_query.py
    # record_as_list = nest.flatten(grads_list)  # flattening list. should already be flat after removing queries
    # clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    # preprocessed_record = nest.pack_sequence_as(grads_list, clipped_as_list)
    preprocessed_record, norm = tf.clip_by_global_norm(grads_list, l2_norm_clip)  # trying this simpler line for now

    return nest.map_structure(tf.add, sample_state, preprocessed_record)  # add clipped sample grad to sum of grads

  def zeros_like(arg):
    try:
      arg = tf.convert_to_tensor(arg)  # pretty sure everything is a tensor at this point, try removing this...
    except TypeError:
      pass
    return tf.zeros(rff_dim, arg.shape, arg.dtype)  # aggregate gradients have extra rff dimension in this case

  sample_state = nest.map_structure(zeros_like, var_list)

  # unrolling microbatches does not seem like an option right now - may be worth revisiting after trying while loop
  _, sample_state = tf.while_loop(cond=lambda i, _: tf.less(i, batch_size),
                                  body=lambda i, state: [tf.add(i, 1), process_sample_loss(i, state)],
                                  loop_vars=[tf.constant(0), sample_state])

  # grad_sums, global_state = dp_sum_query.get_noised_result(sample_state, global_state)
  def add_noise(v):
    return v + tf.random_normal(tf.shape(v), stddev=noise_factor)

  final_grads = nest.map_structure(add_noise, sample_state)  # normalization comes later
  return final_grads


def single_grad(loss, optimizer, var_list, gate_gradients, aggregation_method):
  # compute a gradients for a single scalar loss associated with one sample
  grads, _ = optimizer.compute_gradients(loss, var_list, gate_gradients, aggregation_method)
  # fill up none gradients with zeros
  grads_list = [g if g is not None else tf.zeros_like(v) for (g, v) in zip(list(grads), var_list)]
  return grads_list


def sample_grads(sample_loss, optimizer, var_list, gate_gradients, aggregation_method):
  # all gradiend beloning to one sample (i.e. #RFF many)
  n_rff = tf.shape(sample_loss)[0]
  grads = [single_grad(sample_loss[i], optimizer, var_list, gate_gradients, aggregation_method) for i in range(n_rff)]
  return [tf.stack(k) for k in zip(*grads)]


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
    print('grad_shape', x.get_shape())  # (rff, w_dims) ???
    print('loss_shape', loss_diff_normed.get_shape())
    return tf.matmul(loss_diff_normed, x - y)  # dimensions are not clear here. target shape: (w_dims)

  return tf.contrib.framework.nest.map_structure(full_grad, dfx_dp, dfy)


def release_loss_dis(loss_dis, l2_norm_clip, noise_factor):
  loss_clip = tf.clip_by_norm(loss_dis, l2_norm_clip)
  return loss_clip + tf.random_normal(tf.shape(loss_clip), stddev=noise_factor)


def dp_minimize(dp_loss_dis, loss_gen, opt_gen, opt_dis, var_list, gate_gradients,
                aggregation_method, l2_norm_clip, dp_noise, global_step):
  """
  computes dp gradients for discriminator update with rff mmd estimator

  :param dp_loss_dis:
  :param loss_gen:
  :param opt_gen: optimizer for generator
  :param opt_dis: optimizer for discriminator
  :param var_list:
  :param gate_gradients:
  :param aggregation_method:
  :param l2_norm_clip:
  :param dp_noise:
  :param global_step:
  :return: two optimization ops for generator and discriminator updates
  """
  # generator op:
  gen_opt_op = opt_gen.minimize(loss_gen, global_step=global_step, var_list=var_list,
                                gate_gradients=gate_gradients, aggregation_method=aggregation_method)

  # discriminator op
  # - compute the partial gradients
  grad_gen = sample_grads(dp_loss_dis['gen'], opt_dis, var_list, gate_gradients, aggregation_method)
  grad_dis_release = dp_rff_gradients(opt_dis, dp_loss_dis['dis'], var_list, l2_norm_clip['grad'], dp_noise['grad'],
                                      aggregation_method)
  # - clip & perturb loss
  loss_dis_release = release_loss_dis(dp_loss_dis['dis'], l2_norm_clip['loss'], dp_noise['loss'])

  # get full grads, then couple with variables
  grads = compose_full_grads(fx_dp=loss_dis_release, fy=dp_loss_dis['gen'],
                             dfx_dp=grad_dis_release, dfy=grad_gen, batch_size=tf.shape(loss_gen)[0])
  grads_n_vars = list(zip(grads, var_list))

  # get op
  dis_dp_opt_op = opt_dis.apply_gradients(grads_n_vars)
  return gen_opt_op, dis_dp_opt_op


# def dp_rff_gradients_microbatches(optimizer, loss, var_list, l2_norm_clip, dp_sigma, aggregation_method=None,
#                                   num_microbatches=None):
#   """
#   version of above function before microbatch option was removed for further simplification
#   may be used later to restore microbatch option
#   """
#
#   # Note: it would be closer to the correct i.i.d. sampling of records if
#   # we sampled each microbatch from the appropriate binomial distribution,
#   # although that still wouldn't be quite correct because it would be
#   # sampling from the dataset without replacement.
#   nest = tf.contrib.framework.nest
#
#   if num_microbatches is None:
#     num_microbatches = tf.shape(loss)[0]
#
#   microbatches_losses = tf.reshape(loss, shape=(num_microbatches, -1))
#   gate_gradients = tf.train.Optimizer.GATE_OP
#
#   def process_microbatch(i, sample_state):
#     """Process one microbatch (record) with privacy helper."""
#     grads, _ = zip(optimizer.compute_gradients(tf.reduce_mean(tf.gather(microbatches_losses, [i])),
#                                                var_list, gate_gradients, aggregation_method))
#
#     grads_list = [g if g is not None else tf.zeros_like(v) for (g, v) in zip(list(grads), var_list)]
#
#     # accumulate_record in DPQuery in gaussianquery.py
#     # preprocess_record in GaussianSumQuery in dp_query.py
#     record_as_list = nest.flatten(grads_list)
#     clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
#     preprocessed_record = nest.pack_sequence_as(grads_list, clipped_as_list)
#
#     # accumulate_preprocessed_record in SumAggregationDPQuery in dp_query.py
#     sample_state = nest.map_structure(tf.add, sample_state, preprocessed_record)
#     return sample_state
#
#   def zeros_like(arg):
#     try:
#       arg = tf.convert_to_tensor(arg)
#     except TypeError:
#       pass
#     return tf.zeros(arg.shape, arg.dtype)
#
#   sample_state = nest.map_structure(zeros_like, var_list)
#
#   # unrolling microbatches does not seem like an option right now - may be worth revisiting after trying while loop
#
#   # Use of while_loop here requires that sample_state be a nested structure of tensors.
#   # In general, we would prefer to allow it to be an arbitrary opaque type.
#   cond_fn = lambda i, _: tf.less(i, num_microbatches)
#   body_fn = lambda i, state: [tf.add(i, 1), process_microbatch(i, state)]  # pylint: disable=line-too-long
#   idx = tf.constant(0)
#   _, sample_state = tf.while_loop(cond_fn, body_fn, [idx, sample_state])
#
#   # grad_sums, global_state = dp_sum_query.get_noised_result(sample_state, global_state)
#   def add_noise(v):
#     return v + tf.random_normal(tf.shape(v), stddev=dp_sigma)
#
#   grad_sums = nest.map_structure(add_noise, sample_state)
#
#   def normalize(v):
#     return tf.truediv(v, tf.cast(num_microbatches, tf.float32))
#
#   final_grads = nest.map_structure(normalize, grad_sums)
#
#   return list(zip(final_grads, var_list))
