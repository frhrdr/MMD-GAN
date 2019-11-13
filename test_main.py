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
from tf_privacy.analysis.rdp_accountant import compute_rdp_from_ledger, get_privacy_spent
from dp_funcs.rff_mmd_loss import rff_specs

def epoch_dp_analysis(samples, queries):
  samples = np.concatenate(samples)
  queries = np.concatenate(queries)
  orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
  formatted_ledger = privacy_ledger.format_ledger(samples, queries)
  rdp = compute_rdp_from_ledger(formatted_ledger, orders)
  eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
  print('For delta=1e-5, the current epsilon is: %.2f' % eps)


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
                                                       selection_probability=ar.batch_size / n_data_samples),
                'samples': [],
                'queries': []}

  else:
    dp_specs = None

  agent = Agent(ar.filename, sub_folder, load_ckpt=True, debug_mode=ar.debug_mode, debug_step=ar.debug_step,
                query_step=ar.query_step, imbalanced_update=ar.imbalanced_update)

  print(rff_specs(ar.rff_sigma, ar.rff_dims, ar.rff_const_noise, ar.rff_gen_loss))
  mdl = SNGan(architecture, num_class, ar.loss_type, opt_list,
              rff_specs=rff_specs(ar.rff_sigma, ar.rff_dims, ar.rff_const_noise, ar.rff_gen_loss))

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
      mdl.training(ar.filename, agent, n_data_samples, lr_list, ar.lr_end, ar.save_per_step, ar.batch_size,
                   ar.sample_same_class, ar.n_threads, mog_model=mog_model, dp_specs=dp_specs)
      if dp_specs is not None:
        epoch_dp_analysis(dp_specs['samples'], dp_specs['queries'])
      if ar.debug_mode is not None:
          mdl.eval_sampling(ar.filename, sub_folder, (20, 20), 0, code_x=code_x, do_sprite=True)
      if ar.compute_fid:  # v1 - inception score and fid, ms_ssim - MS-SSIM
          scores = mdl.mdl_score(ar.filename, sub_folder, ar.batch_size, ar.n_fid_batches, 'v1', grey_scale=grey_scale)
          print('Epoch {} with scores: {}'.format(i, scores))

  # if mog_model is not None:
  #   mog_model.save_loss_list('')
  print('Chunk of code finished.')


if __name__ == '__main__':
    main(parse_run_args())
