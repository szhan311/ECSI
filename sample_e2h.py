from sample import sample
import itertools

if __name__ == "__main__":
  
  #   # for baseline + heun sampler
  # config_dir = "./configs/config_e2h64.json"
  # num_samples = 1000 # 138567
  # train_batch_size = 256
  # batch_size = 200
  # route_scaling = 0
  # sampler = "heun" # heun, stoch
  # smooth = 0.
  # baseline = True
  # use_augment = False
  # rho = 7.
  # multiple = 10
  # # multiple
  # split_s = ["train"]
  # suffix_s = [""]
  # model_id_s = ["420000"]
  # pred_mode_s = ["vp"]
  # churn_step_ratio_s = [.33]
  # steps_s = [5]
  
  
  # for baseline
  config_dir = "./configs/config_e2h64.json"
  num_samples = 138567 # 138567
  train_batch_size = 256
  batch_size = 200
  route_scaling = 0
  sampler = "stoch" # heun, stoch
  smooth = 0.
  baseline = True
  use_augment = False
  rho = 0.6
  multiple = 1
  # multiple
  split_s = ["train"]
  suffix_s = [""]
  model_id_s = ["420000"]
  pred_mode_s = ["vp"]
  churn_step_ratio_s = [.3]
  steps_s = [5]
  

  
  for split, suffix, model_id, pred_mode, steps, churn_step_ratio in itertools.product(split_s, suffix_s, model_id_s, pred_mode_s, steps_s, churn_step_ratio_s):
    sample(
      config_dir=config_dir,
      pred_mode=pred_mode,
      model_id=model_id,
      churn_step_ratio=churn_step_ratio,
      num_samples=num_samples,
      train_batch_size = train_batch_size,
      batch_size=batch_size,
      route_scaling=route_scaling,
      suffix = suffix,
      steps=steps,
      split=split,
      baseline = baseline,
      sampler = sampler,
      smooth = smooth,
      use_augment = use_augment,
      rho = rho,
      multiple = multiple
    )