# small test to check if dp gradient computation works on simple models
from dp_funcs.dp_grads import dp_rff_gradients, dp_compute_grads
import tensorflow as tf
import numpy as np
from collections import namedtuple


def test_dp_rff_grads():
  optimizer = tf.train.AdamOptimizer()
  x = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  w = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  b = tf.Variable(np.ones(2).astype(np.float32))
  loss = x @ w + b  # 2d loss for 3 samples

  grads = dp_rff_gradients(optimizer, loss, [w, b], l2_norm_clip=tf.constant(10.), noise_factor=tf.constant(0.01))
  sess = tf.Session()
  x_mat, g_mat = sess.run([x, grads])
  print(x_mat)
  print(g_mat)


def test_dp_compute_grads():
  opt_gen = tf.train.AdamOptimizer()
  opt_dis = tf.train.AdamOptimizer()
  opt_ops = namedtuple('opt', ['gen', 'dis'])(opt_gen, opt_dis)

  dp_spec = {'loss_clip': 10.,
             'grad_clip': 10.,
             'loss_noise': 0.1,
             'grad_noise': 0.1}

  x = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  w = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  b = tf.Variable(np.ones(2).astype(np.float32))
  loss1 = x @ w + b  # 2d loss for 3 samples

  y = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  v = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  d = tf.Variable(np.ones(2).astype(np.float32))
  loss2 = tf.reduce_sum(y @ v + d, axis=0)  # 2d loss sum over 3 samples

  rff_loss = namedtuple('opt', ['fdat', 'fgen'])(loss1, loss2)
  loss_ops = namedtuple('loss', ['gen', 'dis'])(tf.reduce_sum(loss2), rff_loss)
  grads_list, loss_list = dp_compute_grads(loss_ops, opt_ops, dp_spec, vars_dis=[w, b, d], vars_gen=[y, v])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  l_dis, l_gen = sess.run(loss_list)
  print(l_dis, l_gen)
  g_dis, g_gen = sess.run(grads_list)
  print(g_dis)
  print(g_gen)


def test_dp_compute_grads_v2():
  opt_gen = tf.train.AdamOptimizer()
  opt_dis = tf.train.AdamOptimizer()
  opt_ops = namedtuple('opt', ['gen', 'dis'])(opt_gen, opt_dis)

  dp_spec = {'loss_clip': 10.,
             'grad_clip': 10.,
             'loss_noise': 0.1,
             'grad_noise': 0.1}

  x = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  w = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  b = tf.Variable(np.ones(2).astype(np.float32))
  loss1 = x @ w + b  # 2d loss for 3 samples

  y = tf.constant(np.random.randn(3, 4).astype(np.float32))  # 3 samples of dim 4
  v = tf.Variable(np.ones((4, 2)).astype(np.float32))  # a mapping from d=4 to d=2
  d = tf.Variable(np.ones(2).astype(np.float32))
  loss2 = tf.reduce_sum(y @ v + d, axis=0)  # 2d loss sum over 3 samples

  rff_loss = namedtuple('opt', ['fdat', 'fgen'])(loss1, loss2)
  loss_ops = namedtuple('loss', ['gen', 'dis'])(tf.reduce_sum(loss2), rff_loss)
  grads_list, loss_list = dp_compute_grads(loss_ops, opt_ops, dp_spec, vars_dis=[w, b, d], vars_gen=[y, v])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  l_dis, l_gen = sess.run(loss_list)
  print(l_dis, l_gen)
  g_dis, g_gen = sess.run(grads_list)
  print(g_dis)
  print(g_gen)


if __name__ == '__main__':
  test_dp_compute_grads()
