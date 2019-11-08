# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
# FLAGS.SPECTRAL_NORM_MODE = 'sn_paper'  # default, sn_paper
# FLAGS.WEIGHT_INITIALIZER = 'sn_paper'
from GeneralTools.graph_funcs.agent import Agent
from GeneralTools.run_args import parse_run_args, dataset_defaults
from dp_funcs.mog import EncodingMoG, default_mogs
from tf_privacy.analysis import privacy_ledger


def main(ar):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

  if ar.seed is not None:
    np.random.seed(ar.seed)
    tf.compat.v2.random.set_seed(ar.seed)

  assert ar.filename is not None
  FLAGS.DEFAULT_IN = FLAGS.DEFAULT_IN + '{}_NCHW/'.format(ar.dataset)
  from DeepLearning.my_sngan import SNGan

  n_data_samples, architecture, code_dim, act_k, d_enc = dataset_defaults(ar.dataset, ar.d_enc, ar.architecture_key)
  num_class = 0 if ar.n_class is None else ar.n_class
  code_x = np.random.randn(400, code_dim).astype(np.float32)

  lr_list = [ar.lr_dis, ar.lr_gen]  # [dis, gen]
  opt_list = [ar.optimizer_dis, ar.optimizer_gen]

  if ar.loss_type in {'rep', 'rmb'}:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}_{:.1f}_{:.1f}'.format(
          ar.loss_type, lr_list[0], lr_list[1], act_k, ar.rep_weight_0, ar.rep_weight_1)
  else:
      sub_folder = 'sngan_{}_{:.0e}_{:.0e}_k{:.3g}'.format(ar.loss_type, lr_list[0], lr_list[1], act_k)

  if ar.noise_multiplier is not None:
    dp_specs = {'l2_norm_clip': ar.l2_norm_clip,
                'noise_multiplier': ar.noise_multiplier,
                'num_microbatches': ar.num_microbatches,
                'ledger': privacy_ledger.PrivacyLedger(population_size=n_data_samples,
                                                       selection_probability=ar.batch_size / n_data_samples)}
  else:
    dp_specs = None

  agent = Agent(
      ar.filename, sub_folder, load_ckpt=True, do_trace=False,
      do_save=True, debug_mode=ar.debug_mode, debug_step=ar.debug_step,
      query_step=ar.query_step, log_device=False, imbalanced_update=ar.imbalanced_update,
      print_loss=True)

  mdl = SNGan(
      architecture, num_class=num_class, loss_type=ar.loss_type,
      optimizer=opt_list, do_summary=True, do_summary_image=True,
      num_summary_image=8, image_transpose=False)

  if ar.train_without_mog:
    mog_model = None
  else:
    np_mog = default_mogs(ar.mog_type, ar.n_comp, d_enc, ar.cov_type, ar.decay_gamma, ar.em_steps, ar.map_em,
                          ar.reg_covar)
    mog_model = EncodingMoG(d_enc, ar.n_comp, linked_gan=mdl, np_mog=np_mog, n_data_samples=n_data_samples,
                            enc_batch_size=200, filename=ar.filename, cov_type=ar.cov_type,
                            fix_cov=ar.fix_cov, fix_pi=ar.fix_pi, re_init_at_step=ar.re_init_step)
    mdl.register_mog(mog_model, train_with_mog=True, update_loss_type=False)

  grey_scale = ar.dataset in ['mnist', 'fashion']

  for i in range(ar.n_iterations):
      mdl.training(
          ar.filename, agent, n_data_samples,
          lr_list, end_lr=ar.lr_end, max_step=ar.save_per_step,
          batch_size=ar.batch_size, sample_same_class=ar.sample_same_class,
          num_threads=ar.n_threads, mog_model=mog_model, dp_specs=dp_specs)
      if ar.debug_mode is not None:
          _ = mdl.eval_sampling(
              ar.filename, sub_folder, mesh_num=(20, 20), mesh_mode=0, code_x=code_x,
              real_sample=False, do_embedding=False, do_sprite=True)
      if ar.compute_fid:  # v1 - inception score and fid, ms_ssim - MS-SSIM
          scores = mdl.mdl_score(ar.filename, sub_folder, ar.batch_size, num_batch=ar.n_fid_batches,
                                 model='v1', grey_scale=grey_scale)
          print('Epoch {} with scores: {}'.format(i, scores))

  # if mog_model is not None:
  #   mog_model.save_loss_list('')
  print('Chunk of code finished.')


if __name__ == '__main__':
    main(parse_run_args())
