# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
# FLAGS.SPECTRAL_NORM_MODE = 'sn_paper'  # default, sn_paper
# FLAGS.WEIGHT_INITIALIZER = 'sn_paper'
from GeneralTools.graph_funcs.agent import Agent
from GeneralTools.run_args import parse_run_args, dataset_defaults
from dp_funcs.mog import EncodingMoG


def main(ar):
  tf.logging.set_verbosity(tf.logging.WARN)

  if ar.seed is not None:
    np.random.seed(ar.seed)
    tf.compat.v2.random.set_seed(ar.seed)

  assert ar.filename is not None
  FLAGS.DEFAULT_IN = FLAGS.DEFAULT_IN + '{}_NCHW/'.format(ar.dataset)
  from DeepLearning.my_sngan import SNGan

  num_instance, architecture, code_dim, act_k, d_enc = dataset_defaults(ar.dataset, ar.architecture_key)
  # debug_mode = False
  # optimizer = 'adam'
  num_class = 0 if ar.n_class is None else ar.n_class
  # end_lr = 1e-7
  # num_threads = 7
  # n_iterations = 1  # 8

  # random code to test model
  code_x = np.random.randn(400, code_dim).astype(np.float32)
  # to show the model improvements over iterations, consider save the random codes and use later
  # np.savetxt('MMD-GAN/z_128.txt', z_batch, fmt='%.6f', delimiter=',')
  # code_x = np.genfromtxt('MMD-GAN/z_128.txt', delimiter=',', dtype=np.float32)

  # a case
  lr_list = [ar.lr_dis, ar.lr_gen]  # [dis, gen]
  # rep - repulsive loss, rmb - repulsive loss with bounded rbf kernel
  # to test other losses, see GeneralTools/math_func/GANLoss
  # rep_weights = [0.0, -1.0]  # weights for e_kxy and -e_kyy, w[0]-w[1] must be 1

  if ar.loss_type in {'rep', 'rmb'}:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}_{:.1f}_{:.1f}'.format(
          ar.loss_type, lr_list[0], lr_list[1], act_k, ar.rep_weight_0, ar.rep_weight_1)
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear_{:.1f}_{:.1f}'.format(
  #         loss_type, lr_list[0], lr_list[1], rep_weights[0], rep_weights[1])
  else:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}'.format(ar.loss_type, lr_list[0], lr_list[1], act_k)
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])
  # sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])

  # imbalanced_update = None  # NetPicker(dis_steps=3, gen_steps=3)

  agent = Agent(
      ar.filename, sub_folder, load_ckpt=True, do_trace=False,
      do_save=True, debug_mode=ar.debug_mode, debug_step=ar.debug_step,
      query_step=ar.query_step, log_device=False, imbalanced_update=ar.imbalanced_update,
      print_loss=True)

  mdl = SNGan(
      architecture, num_class=num_class, loss_type=ar.loss_type,
      optimizer=ar.optimizer, do_summary=True, do_summary_image=True,
      num_summary_image=8, image_transpose=False)

  if ar.train_without_mog:
    mog_model = None
  else:
    mog_model = EncodingMoG(d_enc, ar.n_comp, linked_gan=mdl, np_mog=ar.mog_type,
                            n_data_samples=num_instance, enc_batch_size=200,
                            filename=ar.filename, cov_type=ar.cov_type,
                            fix_cov=ar.fix_cov, fix_pi=ar.fix_pi, re_init_at_step=ar.re_init_step,
                            decay_gamma=ar.decay_gamma, map_em=ar.map_em)
    mdl.register_mog(mog_model, train_with_mog=True, update_loss_type=False)

  grey_scale = ar.dataset in ['mnist', 'fashion']

  for i in range(ar.n_iterations):
      mdl.training(
          ar.filename, agent, num_instance,
          lr_list, end_lr=ar.lr_end, max_step=ar.save_per_step,
          batch_size=ar.batch_size, sample_same_class=ar.sample_same_class,
          num_threads=ar.n_threads, mog_model=mog_model)
      if ar.debug_mode is not None:
          _ = mdl.eval_sampling(
              ar.filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
              real_sample=False, do_embedding=False, do_sprite=True)
      if ar.compute_fid:  # v1 - inception score and fid, ms_ssim - MS-SSIM
          scores = mdl.mdl_score(ar.filename, sub_folder, ar.batch_size, num_batch=781,
                                 model='v1', grey_scale=grey_scale)
          print('Epoch {} with scores: {}'.format(i, scores))

  # if mog_model is not None:
  #   mog_model.save_loss_list('')
  print('Chunk of code finished.')


if __name__ == '__main__':
    main(parse_run_args())
