from sample import sample
import itertools

# CUDA_VISIBLE_DEVICES=0 sample_diode.py
# class args_diode64:

if __name__ == "__main__":
   # baseline
  config_dir = "./configs/config_diode256.json"
  train_batch_size = 64
  num_samples = 10
  batch_size = 10
  route_scaling = 0
  sampler = "stoch"
  smooth = 0.
  baseline = True
  use_augment = False
  rho = .8
  # multiple
  multiple = 1
  suffix_s = [""]
  split_s = ["train"]
  model_id_s = ["440000"]
  churn_step_ratio_s = [.3]
 
  pred_mode_s = ["vp"]
  steps_s = [20]
  
  
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