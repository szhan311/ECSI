import argparse
import os
from utils import dist_util, compute_fid, get_time, save_img_grid, get_date, logger
from datasets import load_data
from tqdm import tqdm
from sibm.model import create_model
from sibm.diffusion import KarrasDenoiser
import k_diffusion as K
from utils.train_utils import get_workdir
from sibm.sampling import sample_loop, sample_loop_s
from sibm.route import *
import torch.distributed as dist
from utils import dist_util
from dnnlib import EasyDict
import json

def sample(config_dir,
           pred_mode,
           model_id,
           churn_step_ratio,
           num_samples,
           train_batch_size,
           batch_size,
           route_scaling,
           suffix,
           steps = 40,
           split="test",
           baseline = False,
           sampler = "heun",
           smooth = 0,
           use_augment = False,
           rho = 7.,
           multiple = 0):
      '''
      sample:
      sample image
      calculate fid
      
      '''
      # load config
      with open(config_dir, 'r') as file:
            config = json.load(file)
      model_config = EasyDict(config['model'])
      dataset_config = EasyDict(config['dataset'])
      diffusion_config = EasyDict(config['diffusion'])
      train_config = EasyDict(config['train'])
      sample_config = EasyDict(config['sample'])
      
      # update config
      train_config.use_augment = use_augment
      aug = "_aug" if use_augment else ""
      smo = "" if smooth == 0 else f"_smooth{smooth}"
      
      train_config.batch_size = train_batch_size
      exp = f'{dataset_config.exp}{model_config.image_size}_{train_config.batch_size}d_{pred_mode}{aug}{smo}{suffix}'
      diffusion_config.pred_mode = pred_mode
      diffusion_config.smooth = smooth
      diffusion_config.image_size = model_config.image_size
      
      sample_config.config_dir = config_dir
      sample_config.pred_mode = pred_mode
      sample_config.model_id = model_id
      sample_config.churn_step_ratio = churn_step_ratio
      sample_config.num_samples = num_samples
      sample_config.train_batch_size = train_batch_size
      sample_config.batch_size = batch_size
      sample_config.route_scaling = route_scaling
      sample_config.suffix = suffix
      sample_config.steps = steps
      sample_config.split = split
      sample_config.sampler = sampler
      sample_config.rho = rho
      
      
      
      # distibuted setup
      dist_util.setup_dist()
      device = dist_util.dev()
      world_size = dist.get_world_size()
      num_iters = (num_samples - 1) // (world_size * batch_size) + 1
      print(f"num_iters: {num_iters}")
      
      # Get script dir, data_dir, and save_img
      script_dir = os.getcwd()
      
      
      
      if baseline: 
            exp = f"{exp}_ddbm"
            diffusion_config.cov_xy = 0.
      
     
      # model path
      workdir = get_workdir(exp)
      if int(model_id) % train_config.save_interval == 0:
            model_path=f'{workdir}/ema_{train_config.ema_rate}_{model_id}.pt'
      else:
            model_path=f'{workdir}/freq_ema_{train_config.ema_rate}_{model_id}.pt'
      
      if baseline:
            if dataset_config.dataset == "edges2handbags":
                  model_path = "./weights/e2h_ema_0.9999_420000.pt"
            elif dataset_config.dataset == "diode":
                  model_path = "./weights/diode_ema_0.9999_440000.pt"
      if os.path.exists(model_path):
            print(f"load model from: {model_path}")
            logger.configure(dir=f'./logs/logs_{get_date()}', format_strs=['log'], log_suffix=f"_{get_time()}")
            logger.logkvs(model_config)
            logger.logkvs(diffusion_config)
            logger.logkvs(dataset_config)
            logger.logkvs(train_config)
            logger.logkvs(sample_config)
            logger.log(f"load model from: {model_path}")
      else:
            print(f"model path: {model_path} doesn't exist")
            return 0
      
      # save image path
      data_dir = f'{script_dir}{dataset_config.data_dir}' # data
      save_img_path = f'./samples/{exp}'
      print(f"data_dir: {data_dir}")
      os.makedirs(save_img_path, exist_ok=True)
      
      #######################################
      # load model and denoiser
      ######################################
      model = create_model(**model_config)
      diffusion = KarrasDenoiser(**diffusion_config)
      # print(f"model_config: {model_config}")
      # print(f"diffusion_config: {diffusion_config}")
      model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
      )
      if model_config.use_fp16:
            model.convert_to_fp16()
      model.eval()
      model = model.to(device)
      
      # print(diffusion_config)
      # k = 1
      # if k==1: return 0
      #######################################
      # load images
      ######################################
      print("Loading images...")
      # all_images = []
      all_dataloaders = load_data(
            data_dir=data_dir,
            dataset=dataset_config.dataset,
            batch_size=batch_size,
            image_size=model_config.image_size,
            include_test=True,
            seed=sample_config.seed,
            num_workers=sample_config.num_workers,
      )
      if split =='train':
            dataloader = all_dataloaders[1]
      elif split == 'test':
            dataloader = all_dataloaders[2]
      else:
            raise NotImplementedError
      print(f"Dataset size: {len(dataloader.dataset)}")
      
      # real_imgs = []
      # for i, data in tqdm(enumerate(dataloader), total=num_iters):
      #       if split =='train':
      #             y1, y0, index = data
      #       y0 = y0.to(device) * 2 - 1
      #       y1 = y1.to(device) * 2 - 1
      #       real_imgs.append(y1)
      # real_imgs = th.cat(real_imgs, axis=0)
      # real_imgs = real_imgs.reshape(-1, real_imgs.shape[-3], real_imgs.shape[-2], real_imgs.shape[-1])
      # real_imgs = real_imgs[:num_samples]
      # th.save(real_imgs, f"./samples/real_images/diode.pt")
      # br = True
      # if br:return 0
      #######################################
      # Sampling images
      # requirement: model, denoiser, dataloader, 
      ######################################
      print("Sampling ...")
      if multiple >= 2:
            Sample_imgs = []
            for i in range(multiple):
                  _, sample_imgs = sample_loop(diffusion,
                                          model,
                                          dataloader,
                                          num_samples,
                                          pred_mode,
                                          split,
                                          steps,
                                          sample_config.clip_denoised,
                                          sample_config.sampler,
                                          sample_config.sigma_min,
                                          sample_config.sigma_max,
                                          churn_step_ratio,
                                          route_scaling,
                                          sample_config.rho,
                                          sample_config.guidance,
                                          smooth)
                  Sample_imgs.append(sample_imgs)
            Sample_imgs = th.stack(Sample_imgs)
            print(f"Saving to {save_img_path}/model_{model_id[:-4]}_n_{num_samples}_esr_{churn_step_ratio}_{split}_{steps}_multiple_{multiple}.pt")
            out_path = f'{save_img_path}/model_{model_id[:-4]}_n_{num_samples}_esr_{churn_step_ratio}_{split}_{steps}_multiple_{multiple}'
            th.save(Sample_imgs, f"{out_path}.pt")
            return 0
      
      real_imgs, sample_imgs = sample_loop(diffusion,
                                          model,
                                          dataloader,
                                          num_samples,
                                          pred_mode,
                                          split,
                                          steps,
                                          sample_config.clip_denoised,
                                          sample_config.sampler,
                                          sample_config.sigma_min,
                                          sample_config.sigma_max,
                                          churn_step_ratio,
                                          route_scaling,
                                          sample_config.rho,
                                          sample_config.guidance,
                                          smooth)
      
      
                  
                  
                  
                  
      # calculate fid
      fid = compute_fid(real_imgs, sample_imgs, num_samples, batch_size, device)
      print(f"fid: {fid}")
      logger.log(f"fid: {fid}")
      
      # samples_ddbm = ((samples + 1) * 127.5).clip(0, 255)
      # samples_ddbm = samples_ddbm.transpose([0, 2, 3, 1])
      
      if dist.get_rank() == 0:
            print(f"The shape of the sample: {sample_imgs.shape}, number of the sample: {len(sample_imgs)}")
            print(f"The shape of the real images: {real_imgs.shape}, number of the sample: {len(real_imgs)}")
            
            print(f"Saving to {save_img_path}/model_{model_id[:-4]}_n_{num_samples}_esr_{churn_step_ratio}_{split}.pt")
            out_path = f'{save_img_path}/model_{model_id[:-4]}_n_{num_samples}_esr_{churn_step_ratio}_{split}_{steps}'
            # np.save(out_path, samples)
            th.save(sample_imgs, f"{out_path}.pt")
            save_img_grid(sample_imgs[:64], f"{out_path}_fid_{fid.item():.2f}.jpg")
            save_img_grid(real_imgs[:64], f"{save_img_path}/real_image_{split}.jpg")

      dist.barrier()
      print("Sampling complete")
      logger.log("Other parameters:")
      logger.dumpkvs()
      return 0



def sample_series(config_dir,
           pred_mode,
           model_id,
           churn_step_ratio,
           num_samples,
           train_batch_size,
           batch_size,
           route_scaling,
           suffix,
           steps = 40,
           split="test",
           baseline = False,
           sampler = "heun",
           smooth = 0,
           use_augment = False,
           sample_x0 = False,
           rho = 7.):
      '''
      sample:
      sample image
      calculate fid
      
      '''
      # load config
      with open(config_dir, 'r') as file:
            config = json.load(file)
      model_config = EasyDict(config['model'])
      dataset_config = EasyDict(config['dataset'])
      diffusion_config = EasyDict(config['diffusion'])
      train_config = EasyDict(config['train'])
      sample_config = EasyDict(config['sample'])
      
      # update config
      train_config.use_augment = use_augment
      aug = "_aug" if train_config.use_augment else ""
      smo = "" if smooth == 0 else f"_smooth{smooth}"
      
      train_config.batch_size = train_batch_size
      exp = f'{dataset_config.exp}{model_config.image_size}_{train_config.batch_size}d_{pred_mode}{aug}{smo}{suffix}'
      diffusion_config.pred_mode = pred_mode
      diffusion_config.image_size = model_config.image_size
      diffusion_config.smooth = smooth
      
      sample_config.config_dir = config_dir
      sample_config.pred_mode = pred_mode
      sample_config.model_id = model_id
      sample_config.churn_step_ratio = churn_step_ratio
      sample_config.num_samples = num_samples
      sample_config.train_batch_size = train_batch_size
      sample_config.batch_size = batch_size
      sample_config.route_scaling = route_scaling
      sample_config.suffix = suffix
      sample_config.steps = steps
      sample_config.split = split
      sample_config.sampler = sampler
      sample_config.rho = rho
      
      
      
      # distibuted setup
      dist_util.setup_dist()
      device = dist_util.dev()
      world_size = dist.get_world_size()
      num_iters = (num_samples - 1) // (world_size * batch_size) + 1
      print(f"num_iters: {num_iters}")
      
      # Get script dir, data_dir, and save_img
      script_dir = os.getcwd()
      
      
      if baseline: 
            exp = f"{exp}_ddbm"
            diffusion_config.cov_xy = 0.
      
      data_dir = f'{script_dir}{dataset_config.data_dir}' # data
      save_img_path = f'./samples/{exp}'
      print(f"data_dir: {data_dir}")
      os.makedirs(save_img_path, exist_ok=True)
      
      # model path
      workdir = get_workdir(exp)
      if int(model_id) % train_config.save_interval == 0:
            model_path=f'{workdir}/ema_{train_config.ema_rate}_{model_id}.pt'
      else:
            model_path=f'{workdir}/freq_ema_{train_config.ema_rate}_{model_id}.pt'
      
      if baseline:
            if dataset_config.dataset == "edges2handbags":
                  model_path = "./weights/e2h_ema_0.9999_420000.pt"
            elif dataset_config.dataset == "diode":
                  model_path = "./weights/diode_ema_0.9999_440000.pt"
      print(f"load model from: {model_path}")
      

      #######################################
      # load model and denoiser
      ######################################
      model = create_model(**model_config)
      diffusion = KarrasDenoiser(**diffusion_config)
      # print(f"model_config: {model_config}")
      # print(f"diffusion_config: {diffusion_config}")
      model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
      )
      if model_config.use_fp16:
            model.convert_to_fp16()
      model.eval()
      model = model.to(device)
      #######################################
      # load images
      ######################################
      print("Loading images...")
      # all_images = []
      all_dataloaders = load_data(
            data_dir=data_dir,
            dataset=dataset_config.dataset,
            batch_size=batch_size,
            image_size=model_config.image_size,
            include_test=True,
            seed=sample_config.seed,
            num_workers=sample_config.num_workers,
      )
      if split =='train':
            dataloader = all_dataloaders[1]
      elif split == 'test':
            dataloader = all_dataloaders[2]
      else:
            raise NotImplementedError
      print(f"Dataset size: {len(dataloader.dataset)}")
      
      #######################################
      # Sampling images
      # requirement: model, denoiser, dataloader, 
      ######################################
      print("Sampling ...")
      real_imgs, sample_imgs = sample_loop_s(diffusion,
                                          model,
                                          dataloader,
                                          num_samples,
                                          pred_mode,
                                          split,
                                          steps,
                                          sample_config.clip_denoised,
                                          sample_config.sampler,
                                          sample_config.sigma_min,
                                          sample_config.sigma_max,
                                          churn_step_ratio,
                                          route_scaling,
                                          sample_config.rho,
                                          sample_config.guidance,
                                          smooth,
                                          sample_x0)
      


      dist.barrier()
      print("Sampling complete")
      return real_imgs, sample_imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample images using a pretrained model.")
    parser.add_argument('--config-dir', type=str, default='./configs/config_e2h.json', help='Path to the config file.')
    parser.add_argument('--pred-mode', type=str, default='vpv2', help='Prediction mode.')
    parser.add_argument('--model-id', type=str, default=100000, help='Model ID.')
    parser.add_argument('--churn-step-ratio', type=float, default=0.33, help='Churn step ratio.')
    parser.add_argument('--smooth', type=float, default=0., help='smooth.')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples to generate.')
    parser.add_argument('--train-batch-size', type=int, default=192, help='Train batch size.')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size.')
    parser.add_argument('--route-scaling', default=0, help='route scaling.')
    parser.add_argument('--steps', type=int, default=40, help='Number of steps.')
    parser.add_argument('--split', type=str, default="test", help='Data split to use.')
    parser.add_argument('--suffix', type=str, default="", help='suffix.')
    parser.add_argument('--baseline', action='store_true', help='Include baseline for comparison')

    args = parser.parse_args()
    args = EasyDict(vars(args))
    sample(
        config_dir=args.config_dir,
        pred_mode=args.pred_mode,
        model_id=args.model_id,
        churn_step_ratio=args.churn_step_ratio,
        num_samples=args.num_samples,
        train_batch_size = args.train_batch_size,
        batch_size=args.batch_size,
        route_scaling=args.route_scaling,
        suffix = args.suffix,
        steps=args.steps,
        split=args.split,
        baseline=args.baseline,
        smooth = args.smooth
    )