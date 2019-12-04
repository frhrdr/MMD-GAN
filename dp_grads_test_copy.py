# small test to check if dp gradient computation works on simple models
from dp_funcs.dp_grads import dp_rff_gradients, dp_compute_grads
import tensorflow as tf
import numpy as np
from collections import namedtuple


def test_dp_rff_grads():
  x = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  w = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  b = tf.Variable(np.ones(2).astype(np.float32))
  loss = x @ w + b  # 2d loss for 3 samples

  dp_grads = dp_rff_gradients(loss, [w, b], l2_norm_clip=tf.constant(10.), noise_factor=tf.constant(0.0))
  normal_grads = tf.gradients(loss, [w, b])
  sess = tf.Session()
  x_mat, dp_g_mat, nrm_g_mat = sess.run([x, dp_grads, normal_grads])
  print(x_mat)
  print('DP-grads')
  print(dp_g_mat)
  print('Normal-grads')
  print(nrm_g_mat)


def test_dp_compute_grads():

  dp_spec = {'loss_clip': 10.,
             'grad_clip': 10.,
             'loss_noise': 0.,
             'grad_noise': 0.}
  n_samples = 3
  d_in = 4
  d_out = 2

  x = tf.constant(np.random.randn(n_samples, d_in).astype(np.float32))  # 3 samples of dim 4
  w = tf.Variable(np.ones((d_in, d_out)).astype(np.float32))  # a mapping from din to dout
  b = tf.Variable(np.ones(d_out).astype(np.float32))
  loss1 = x @ w + b  # 2d loss for 3 samples

  y = tf.constant(np.random.randn(n_samples, d_in).astype(np.float32))  # 3 samples of dim 4
  v = tf.Variable(np.ones((d_in, d_out)).astype(np.float32))  # a mapping from din to dout
  d = tf.Variable(np.ones(d_out).astype(np.float32))
  loss2 = tf.reduce_sum(y @ v + d, axis=0)  # 2d loss sum over 3 samples

  full_loss = tf.norm(tf.reduce_sum(loss1, axis=0) - loss2)**2 / n_samples**2
  full_grad = tf.gradients(full_loss, [w, b, d])

  rff_loss = namedtuple('opt', ['fdat', 'fgen'])(loss1, loss2)
  loss_ops = namedtuple('loss', ['gen', 'dis'])(tf.reduce_sum(loss2), rff_loss)
  grads_list, loss_list = dp_compute_grads(loss_ops, dp_spec, vars_dis=[w, b, d], vars_gen=[y, v])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  l_dis, l_gen, l_full, l_1, l_2 = sess.run(loss_list + [full_loss, loss1, loss2])
  # print('loss components')
  # print(l_1, l_2)
  print('losses')
  print(l_dis, l_full)
  g_dis, g_gen, g_full = sess.run(grads_list + [full_grad])
  print('grads_compound')
  for g in g_dis:
    print(g[0])
  print('grads_direct')
  for g in g_full:
    print(g)

if __name__ == '__main__':
  test_dp_compute_grads()
  # test_dp_rff_grads()
