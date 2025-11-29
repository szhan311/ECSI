
import argparse
import torch.distributed as dist
from datasets import load_data
from datasets.augment import AugmentPipe
# from sibm.diffusion import create_model_and_diffusion
from sibm.diffusion import KarrasDenoiser
from sibm.model import create_model
from utils import dist_util, logger
from utils.resample import create_named_schedule_sampler
from utils.train_utils import get_workdir
from utils.train_loop import TrainLoop
from pathlib import Path
from glob import glob
import wandb
import json
import os
from dnnlib import EasyDict

def main():
      # args
      p = argparse.ArgumentParser(description="Arguments for model training and evaluation")
      p.add_argument('--config-dir', type=str, default='./configs/config_e2h32.json',
                   help='the configuration file')
      p.add_argument('--pred-mode', type=str, default='vp',
                   help='prediction model')
      p.add_argument('--weight-schedule', type=str, default='sibm',
                   help='weight schedule')
      p.add_argument('--suffix', type=str, default='',
                   help='suffix')
      p.add_argument('--smooth', type=float, default=0.,
                   help='smooth the base distribution')
      args = p.parse_args()
      
      # load config
      with open(args.config_dir, 'r') as file:
            config = json.load(file)
      model_config = EasyDict(config['model'])
      dataset_config = EasyDict(config['dataset'])
      diffusion_config = EasyDict(config['diffusion'])
      train_config = EasyDict(config['train'])
      sample_config = EasyDict(config['sample'])
      
      # update config
      aug = "_aug" if train_config.use_augment else ""
      smo = "" if args.smooth == 0 else f"_smooth{args.smooth}"
      exp = f'{dataset_config.exp}{model_config.image_size}_{train_config.batch_size}d_{args.pred_mode}{aug}{smo}{args.suffix}'
      diffusion_config.pred_mode = args.pred_mode
      diffusion_config.weight_schedule = args.weight_schedule
      diffusion_config.image_size = model_config.image_size
      diffusion_config.smooth = args.smooth
      
      print(f"dataset exp: {dataset_config.exp}")
      print(f"exp: {exp}")
      print(f"args: {args}")
      print(f"model config: {model_config}")
      print(f"dataset config: {dataset_config}")
      print(f"diffusion config: {diffusion_config}")
      print(f"train config: {train_config}")
      
      # create workdir
      workdir = get_workdir(exp)
      print(f"workdir: {workdir}")
      Path(workdir).mkdir(parents=True, exist_ok=True)
      dist_util.setup_dist()
      logger.configure(dir=workdir)
      
      # wandb initialize
      if dist.get_rank() == 0:
            name = exp if train_config.resume_checkpoint == "" else exp + '_resume'
            wandb.init(project=f"sibm_{dataset_config.exp}", group=exp, name=name, mode='online' if not train_config.debug else 'disabled')
            wandb.config.update({
                  "model": model_config,
                  "diffusion": diffusion_config,
                  "dataset": dataset_config,
                  "train": train_config,
                  "sample":sample_config,
                  "args": args
                  })
            logger.log("creating model and diffusion...")
      
      # resume_checkpoint
      if train_config.resume_checkpoint == "":
            model_ckpts = list(glob(f'{workdir}/*model*[0-9].*'))
            if len(model_ckpts) > 0:
                  max_ckpt = max(model_ckpts, key=lambda x: int(x.split('model_')[-1].split('.')[0]))
                  if os.path.exists(max_ckpt):
                        train_config.resume_checkpoint = max_ckpt
                  if dist.get_rank() == 0:
                        logger.log('Resuming from checkpoint: ', max_ckpt)
      
      # create model and diffusion
      model = create_model(**model_config)
      diffusion = KarrasDenoiser(**diffusion_config)
      model.to(dist_util.dev())
      print(dist_util.dev())
      if dist.get_rank() == 0:
            wandb.watch(model, log='all')
      
      # create schedule sampler
      schedule_sampler = create_named_schedule_sampler(train_config.schedule_sampler, diffusion)
      
      # use global_batch_size
      if train_config.batch_size == -1:
            batch_size = train_config.global_batch_size // dist.get_world_size()
            if train_config.global_batch_size % dist.get_world_size() != 0:
                  logger.log(
                        f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {train_config.global_batch_size}"
                  )
      else:
            batch_size = train_config.batch_size
      
      # load dataset
      if dist.get_rank() == 0:
            logger.log("creating data loader...")
      script_dir = os.getcwd()
      print(dataset_config)
      data, test_data = load_data(data_dir = f'{script_dir}{dataset_config.data_dir}', 
                                  dataset = dataset_config.dataset,
                                  batch_size = batch_size,
                                  image_size = model_config.image_size,
                                  num_workers=dataset_config.num_workers)
      
      # augmentation
      if train_config.use_augment:
            augment = AugmentPipe(
                  p=0.12,xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
      else:
            augment = None
      
      logger.log("training...")
      
      # train loop
      sample_config.generator = "determ"
      # sample_config.pred_mode = diffusion_config.pred_mode
      sample_config.sigma_min = diffusion_config.sigma_min
      sample_config.sigma_max = diffusion_config.sigma_max
      # sample_defaults = EasyDict(generator="determ")
      
      TrainLoop(
            model=model,
            diffusion=diffusion,
            train_data=data,
            test_data=test_data,
            batch_size=batch_size,
            microbatch=train_config.microbatch,
            lr=train_config.lr,
            ema_rate=train_config.ema_rate,
            log_interval=train_config.log_interval,
            test_interval=train_config.test_interval,
            save_interval=train_config.save_interval,
            save_interval_for_preemption=train_config.save_interval_for_preemption,
            resume_checkpoint=train_config.resume_checkpoint,
            workdir=workdir,
            use_fp16=model_config.use_fp16,
            fp16_scale_growth=train_config.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=train_config.weight_decay,
            lr_anneal_steps=train_config.lr_anneal_steps,
            augment_pipe=augment,
            pred_mode = diffusion_config.pred_mode,
            **sample_config
      ).run_loop()

if __name__ == "__main__":
      main()