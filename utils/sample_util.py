import argparse
import os
from utils.config_default import model_and_diffusion_defaults
from sibm.diffusion import create_model_and_diffusion
from utils.train_utils import args_to_dict, get_time
from utils import dist_util
from datasets import load_data
from sibm.sampling import *
from sibm.route import *
from utils.nn import append_dims, mean_flat
from utils.plot_utils import show_img_grid
import accelerate
import k_diffusion as K
from utils import logger

def sample_small(pred_mode = 'vp', suffix = "", dataset = "edges2handbags", baseline = False, model_id = '100000', churn_step_ratio = 0.33, cov_xy = 0.125, batch_size = 64):
      '''
      pred_mode = 'vp' # vp, vpv1, vpv2
      suffix = "_lr_decay" # _aug_cov125, _lr1e-5, _lr_decay
      baseline = False
      dataset = "edges2handbags" # ["diode", "edges2handbags"]
      model_id = '100000'
      churn_step_ratio = 0.33
      cov_xy = 0.125
      batch_size = 64

      '''
      


      accelerator = accelerate.Accelerator()
      device = accelerator.device

      script_dir = os.getcwd()
      logger.configure(dir=f'{script_dir}/logs', format_strs=['log'], log_suffix=f"_{get_time()}")
      logger.log("Add Euler Step Ratio Decay: churn_step_ratio = sigmas[i] * churn_step_ratio")
      logger.log(f"pred_mode: {pred_mode}")
      logger.log(f"suffix: {suffix}")
      logger.log(f"baseline: {baseline}")
      logger.log(f"dataset: {dataset}")
      logger.log(f"model_id: {model_id}")
      logger.log(f"churn_step_ratio: {churn_step_ratio}")
      logger.log(f"cov_xy: {cov_xy}")
      logger.log(f"batch_size: {batch_size}")
      route = get_route(pred_mode)
      if dataset == 'edges2handbags':
            data_dir = f'{script_dir}/datasets/edges2handbags' 
            exp = f'e2h64_192d_{pred_mode}{suffix}'
            image_size = 64
            num_channels = 192
            num_res_blocks = 3
            model_path=f'{script_dir}/weights/e2h_ema_0.9999_420000.pt'
      elif dataset == 'diode':
            data_dir = f'{script_dir}/datasets/diode-normal-256'
            exp = f'diode256_256d_{pred_mode}{suffix}'
            image_size = 256
            num_channels = 256
            num_res_blocks = 2
            model_path=f'{script_dir}/weights/diode_ema_0.9999_440000.pt'

      if baseline == False:
            if int(model_id) % 100000 == 0:
                  model_path=f'{script_dir}/workdir/{exp}/ema_0.9999_{model_id}.pt'
            else:
                  model_path=f'{script_dir}/workdir/{exp}/freq_ema_0.9999_{model_id}.pt'

      img_path=f'{script_dir}/samples/{exp}/images_{model_id}.pt'

      args = argparse.Namespace(data_dir=data_dir,
                              dataset=dataset,
                              clip_denoised=True, # clip_denoised
                              num_samples=10000,
                              batch_size=batch_size,
                              sampler='heun',
                              split='train',
                              churn_step_ratio=churn_step_ratio,
                              rho=7.0,
                              steps=40, # 40
                              model_path=model_path,
                              exp=exp,
                              seed=42,
                              ts='',
                              upscale=False,
                              num_workers=2,
                              guidance=1.0,
                              sigma_data=0.5,
                              sigma_min=0.0001,
                              sigma_max=0.9999,
                              beta_d=2,
                              beta_min=0.1,
                              cov_xy=cov_xy,
                              image_size=image_size,
                              in_channels=3,
                              num_channels=num_channels,
                              num_res_blocks=num_res_blocks,
                              num_heads=4,
                              num_heads_upsample=-1,
                              num_head_channels=32,
                              unet_type='adm',
                              attention_resolutions='32,16,8',
                              channel_mult='',
                              dropout=0.1,
                              class_cond=False,
                              use_checkpoint=False,
                              use_scale_shift_norm=True,
                              resblock_updown=True,
                              use_fp16=True,
                              use_new_attention_order=False,
                              attention_type='flash',
                              learn_sigma=False,
                              condition_mode='concat',
                              pred_mode=pred_mode,
                              weight_schedule='bridge_karras')

      # load model
      print("load model ...")
      dist_util.setup_dist()
      model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()),
      )
      model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
      )
      if args.use_fp16:
            model.convert_to_fp16()
      model.eval()
      # device = dist_util.dev()
      # device =  th.device("cuda:4")
      model = model.to(device)
      # get conditional images
      all_images = []
      all_dataloaders = load_data(
            data_dir=args.data_dir,
            dataset=args.dataset,
            batch_size=args.batch_size,
            image_size=args.image_size,
            include_test=True,
            seed=args.seed,
            num_workers=args.num_workers,
      )
      if args.split =='train':
            dataloader = all_dataloaders[1]
      elif args.split == 'test':
            dataloader = all_dataloaders[2]
      else:
            raise NotImplementedError
      args.num_samples = len(dataloader.dataset)
      y1, y0, _ = next(iter(dataloader))
      y0 = y0.to(device) * 2 - 1
      y1 = y1.to(device) * 2 - 1
      model_kwargs = {'xT': y0}
      # model, train_dl = accelerator.prepare(model, dataloader)
      extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
      reals_features = K.evaluation.compute_features(accelerator, lambda x:y1, extractor, args.batch_size, args.batch_size)
      print("start sampling ...")
      sample, _, _ = karras_sample(
                  diffusion,
                  model,
                  y0,
                  None,
                  route,
                  steps=args.steps,
                  model_kwargs=model_kwargs,
                  device=device,
                  clip_denoised=args.clip_denoised,
                  sampler=args.sampler,
                  sigma_min=diffusion.sigma_min,
                  sigma_max=diffusion.sigma_max,
                  churn_step_ratio=args.churn_step_ratio,
                  rho=args.rho,
                  guidance=args.guidance
            )

      fakes_features = K.evaluation.compute_features(accelerator, lambda x:sample, extractor, args.batch_size, args.batch_size)

      fid = K.evaluation.fid(fakes_features, reals_features)
      logger.log(f"fid: {fid}")
      # print(f"fid: {fid}")
      return sample, fid