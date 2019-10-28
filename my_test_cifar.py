import numpy as np
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = FLAGS.DEFAULT_IN + 'cifar_NCHW/'
# FLAGS.SPECTRAL_NORM_MODE = 'sn_paper'  # default, sn_paper
# FLAGS.WEIGHT_INITIALIZER = 'sn_paper'
from GeneralTools.graph_funcs.agent import Agent
from GeneralTools.architectures import cifar_default
from DeepLearning.my_sngan import SNGan
from dp_funcs.mog import MoG


def main():
  filename = 'cifar_logging5_diag'

  architecture, code_dim, act_k = cifar_default()
  debug_mode = False
  optimizer = 'adam'
  num_instance = 50000
  save_per_step = 25000  # 12500
  batch_size = 64
  num_class = 0
  end_lr = 1e-7
  num_threads = 7
  n_iterations = 1  # 8

  # random code to test model
  code_x = np.random.randn(400, 128).astype(np.float32)
  # to show the model improvements over iterations, consider save the random codes and use later
  # np.savetxt('MMD-GAN/z_128.txt', z_batch, fmt='%.6f', delimiter=',')
  # code_x = np.genfromtxt('MMD-GAN/z_128.txt', delimiter=',', dtype=np.float32)

  # a case
  lr_list = [5e-4, 2e-4]  # [dis, gen]
  loss_type = 'rep'
  # rep - repulsive loss, rmb - repulsive loss with bounded rbf kernel
  # to test other losses, see GeneralTools/math_func/GANLoss
  rep_weights = [0.0, -1.0]  # weights for e_kxy and -e_kyy, w[0]-w[1] must be 1

  sample_same_class = False
  if loss_type in {'rep', 'rmb'}:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}_{:.1f}_{:.1f}'.format(
          loss_type, lr_list[0], lr_list[1], act_k, rep_weights[0], rep_weights[1])
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear_{:.1f}_{:.1f}'.format(
  #         loss_type, lr_list[0], lr_list[1], rep_weights[0], rep_weights[1])
  else:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}'.format(loss_type, lr_list[0], lr_list[1], act_k)
  #     sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])
  # sub_folder = 'sngan_{}_{:.0e}_{:.0e}_gl1_linear'.format(loss_type, lr_list[0], lr_list[1])

  imbalanced_update = None
  # imbalanced_update = (-3, 3)  # order: (dis, gen)
  # imbalanced_update = NetPicker(dis_steps=3, gen_steps=3)

  agent = Agent(
      filename, sub_folder, load_ckpt=True, do_trace=False,
      do_save=True, debug_mode=debug_mode, debug_step=400,
      query_step=1000, log_device=False, imbalanced_update=imbalanced_update,
      print_loss=True)

  mdl = SNGan(
      architecture, num_class=num_class, loss_type=loss_type,
      optimizer=optimizer, do_summary=True, do_summary_image=True,
      num_summary_image=8, image_transpose=False)

  enc_batch_size = 200
  mog_model = MoG(n_dims=16, n_clusters=20, linked_gan=mdl,
                  enc_batch_size=enc_batch_size, n_data_samples=num_instance,
                  filename=filename, cov_type='diag')
  mdl.register_mog(mog_model, train_with_mog=False, update_loss_type=False)
  # mog_model = None

  for i in range(n_iterations):
      print('beginning iteration {}'.format(i))
      mdl.training(
          filename, agent, num_instance,
          lr_list, end_lr=end_lr, max_step=save_per_step,
          batch_size=batch_size, sample_same_class=sample_same_class, num_threads=num_threads, mog_model=mog_model)
      if debug_mode is not False:
          print('running eval_sampling')
          _ = mdl.eval_sampling(
              filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
              real_sample=False, do_embedding=False, do_sprite=True)
      if False and debug_mode is False:  # v1 - inception score and fid, ms_ssim - MS-SSIM
          print('running mdl_score')
          scores = mdl.mdl_score(filename, sub_folder, batch_size, num_batch=781, model='v1')
          print('Epoch {} with scores: {}'.format(i, scores))

  if mog_model is not None:
    mog_model.save_loss_list('{}_losses.npy'.format(filename))
  print('Chunk of code finished.')


if __name__ == '__main__':
    main()
