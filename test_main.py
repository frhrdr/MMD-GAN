import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
# FLAGS.SPECTRAL_NORM_MODE = 'sn_paper'  # default, sn_paper
# FLAGS.WEIGHT_INITIALIZER = 'sn_paper'
from GeneralTools.graph_funcs.agent import Agent
from GeneralTools.run_args import parse_run_args, dataset_defaults
from dp_funcs.mog import MoG


def main(args):

  if args.seed is not None:
    np.random.seed(args.seed)
    tf.compat.v2.random.set_seed(args.seed)

  assert args.filename is not None
  FLAGS.DEFAULT_IN = FLAGS.DEFAULT_IN + '{}_NCHW/'.format(args.dataset)
  from DeepLearning.my_sngan import SNGan

  num_instance, architecture, code_dim, act_k, d_enc = dataset_defaults(args.dataset, args.architecture_key)
  # debug_mode = False
  # optimizer = 'adam'
  save_per_step = 20000 if args.save_per_step is None else args.save_per_step
  num_class = 0 if args.n_class is None else args.n_class
  #end_lr = 1e-7
  # num_threads = 7
  # n_iterations = 1  # 8

  # random code to test model
  code_x = np.random.randn(400, code_dim).astype(np.float32)
  # to show the model improvements over iterations, consider save the random codes and use later
  # np.savetxt('MMD-GAN/z_128.txt', z_batch, fmt='%.6f', delimiter=',')
  # code_x = np.genfromtxt('MMD-GAN/z_128.txt', delimiter=',', dtype=np.float32)

  # a case
  lr_list = [args.lr_dis, args.lr_gen]  # [dis, gen]
  # rep - repulsive loss, rmb - repulsive loss with bounded rbf kernel
  # to test other losses, see GeneralTools/math_func/GANLoss
  # rep_weights = [0.0, -1.0]  # weights for e_kxy and -e_kyy, w[0]-w[1] must be 1

  if args.loss_type in {'rep', 'rmb'}:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}_{:.1f}_{:.1f}'.format(
          args.loss_type, lr_list[0], lr_list[1], act_k, args.rep_weight_0, args.rep_weight_1)
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear_{:.1f}_{:.1f}'.format(
  #         loss_type, lr_list[0], lr_list[1], rep_weights[0], rep_weights[1])
  else:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}'.format(args.loss_type, lr_list[0], lr_list[1], act_k)
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])
  # sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])

  # imbalanced_update = None  # NetPicker(dis_steps=3, gen_steps=3)

  agent = Agent(
      args.filename, sub_folder, load_ckpt=True, do_trace=False,
      do_save=True, debug_mode=args.debug_mode, debug_step=args.debug_step,
      query_step=args.query_step, log_device=False, imbalanced_update=args.imbalanced_update,
      print_loss=True)

  mdl = SNGan(
      architecture, num_class=num_class, loss_type=args.loss_type,
      optimizer=args.optimizer, do_summary=True, do_summary_image=True,
      num_summary_image=8, image_transpose=False)

  mog_model = MoG(n_dims=d_enc, n_clusters=args.n_clusters, max_iter=args.em_steps, linked_gan=mdl,
                  enc_batch_size=200, n_data_samples=num_instance,
                  filename=args.filename, cov_type=args.cov_type)
  mdl.register_mog(mog_model, train_with_mog=not args.train_without_mog, update_loss_type=False)
  # mdl.register_mog(mog_model)

  grey_scale = args.dataset in ['mnist', 'fashion']

  for i in range(args.n_iterations):
      mdl.training(
          args.filename, agent, num_instance,
          lr_list, end_lr=args.lr_end, max_step=save_per_step,
          batch_size=args.batch_size, sample_same_class=args.sample_same_class,
          num_threads=args.n_threads, mog_model=mog_model)
      if args.debug_mode is not None:
          _ = mdl.eval_sampling(
              args.filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
              real_sample=False, do_embedding=False, do_sprite=True)
      if args.compute_fid:  # v1 - inception score and fid, ms_ssim - MS-SSIM
          scores = mdl.mdl_score(args.filename, sub_folder, args.batch_size, num_batch=781,
                                 model='v1', grey_scale=grey_scale)
          print('Epoch {} with scores: {}'.format(i, scores))

  if mog_model is not None:
    mog_model.save_loss_list('')
  print('Chunk of code finished.')


if __name__ == '__main__':
    main(parse_run_args())
