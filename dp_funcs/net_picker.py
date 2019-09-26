

class NetPicker:
  def __init__(self, dis_steps, gen_steps):
    self.dis_steps = dis_steps
    self.gen_steps = gen_steps
    self.train_dis = True
    self.steps_done = 0

  def do_mog_update(self):
    # only do mog update after having trained discriminator
    return not self.train_dis and self.steps_done == 0

  def pick_ops(self, op_list):
    assert len(op_list) == 2
    print('picking {} for training'.format('dis' if self.train_dis else 'gen'))
    ret_val = [op_list[0] if self.train_dis else op_list[1]]
    self.steps_done += 1
    if self.train_dis and self.steps_done == self.dis_steps:
      self.train_dis = False
      self.steps_done = 0
    elif not self.train_dis and self.steps_done == self.gen_steps:
      self.train_dis = True
      self.steps_done = 0
    return ret_val
