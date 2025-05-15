import torch as th
import torch.distributed as dist
from utils import dist_util
from sibm.route import *
from tqdm import tqdm
from k_diffusion.sampling import get_sigmas_karras 
import k_diffusion as K
from utils.train_utils import FeatureDifferenceCalculator
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def get_scaling(k=0.):
    s = lambda t: th.where(t < 0.5, k * t + 1, k * (1 - t) + 1)
    s_deriv = lambda t: th.where(t < 0.5, k, -k)
    return s, s_deriv

def update_route(alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv):
    alpha_new = s * alpha
    beta_new = s * beta
    gamma_new = s * gamma
    alpha_deriv_new = s_deriv * alpha + s * alpha_deriv
    beta_deriv_new = s_deriv * beta + s * beta_deriv
    gamma_deriv_new = s_deriv * gamma + s * gamma_deriv
    return alpha_new, alpha_deriv_new, beta_new, beta_deriv_new, gamma_new, gamma_deriv_new
 
def to_d_sky(x, denoised, x_T, alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    score = (alpha * denoised + beta * x_T - x / s) / gamma ** 2
    score = score / s
    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = update_route(alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, s, s_deriv)
    if gamma == 0:
        return alpha_deriv * denoised + beta_deriv * x_T, 0
    # This is the implementation of the ODE sampler based on Corollary 1 and DDBM setting.
    elif stochastic is False:
        diffusion_term = alpha_deriv / alpha * x + (beta_deriv - alpha_deriv / alpha * beta) * x_T - (gamma * gamma_deriv - alpha_deriv / alpha * gamma ** 2) * score
        return diffusion_term, 0
    else:  
        diffusion_term = alpha_deriv / alpha * x + (beta_deriv - alpha_deriv / alpha * beta) * x_T - 2 * (gamma * gamma_deriv - alpha_deriv / alpha * gamma ** 2) * score
        drift_term = (2 * (gamma * gamma_deriv - alpha_deriv / alpha * (gamma ** 2))) ** 0.5
        return diffusion_term, drift_term


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    route,
    progress=False,
    callback=None,
    churn_step_ratio=0.,
    route_scaling = 0,
    smooth = 0
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    # x_T = x + smooth * th.randn_like(x)
    # if smooth != 0: x = x + smooth * th.randn_like(x)
    path = [x.detach().cpu()]
    x0_est = [x.detach().cpu()]
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0 # number of function evaluations
    
    
    ###########################################
    # s, s_deriv = get_scaling(k=route_scaling)
    s = lambda t: 1
    s_deriv = lambda t: 0
    # print(f"route_scaling: {route_scaling}")
    ###########################################
    
    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = route
    assert churn_step_ratio <= 1
    for j, i in enumerate(indices):
        if churn_step_ratio > 0:
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            denoised = denoiser(x / s(sigmas[i]), sigmas[i] * s_in, x_T)
            x0_est.append(denoised.detach().cpu())
            # diffusion_term, drift_term = to_d_sky(x, denoised, x_T, alpha(sigma_hat), alpha_deriv(sigma_hat), beta(sigma_hat), beta_deriv(sigma_hat), gamma(sigma_hat), gamma_deriv(sigma_hat), stochastic=True)
            diffusion_term, drift_term = to_d_sky(x, denoised, x_T, alpha(sigmas[i]), alpha_deriv(sigmas[i]), beta(sigmas[i]), beta_deriv(sigmas[i]), gamma(sigmas[i]), gamma_deriv(sigmas[i]), s(sigmas[i]), s_deriv(sigmas[i]), stochastic=True)
            dt = (sigma_hat - sigmas[i]) 
            x = x + diffusion_term * dt + th.randn_like(x) *((dt).abs() ** 0.5)*drift_term
            nfe += 1
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]

        # heun step
        if churn_step_ratio < 1:
            denoised = denoiser(x / s(sigma_hat), sigma_hat * s_in, x_T)
            x0_est.append(denoised.detach().cpu())
            d, _ = to_d_sky(x, denoised, x_T, alpha(sigma_hat), alpha_deriv(sigma_hat), beta(sigma_hat), beta_deriv(sigma_hat), gamma(sigma_hat), gamma_deriv(sigma_hat), s(sigma_hat), s_deriv(sigma_hat))
                
            nfe += 1
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = denoiser(x_2 / s(sigmas[i + 1]), sigmas[i + 1] * s_in, x_T)
                x0_est.append(denoised.detach().cpu())
                d_2, _ = to_d_sky(x, denoised_2, x_T, alpha(sigmas[i + 1]), alpha_deriv(sigmas[i + 1]), beta(sigmas[i + 1]), beta_deriv(sigmas[i + 1]), gamma(sigmas[i + 1]), gamma_deriv(sigmas[i + 1]), s(sigmas[i + 1]), s_deriv(sigmas[i + 1]))
                d_prime = (d + d_2) / 2
                # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
                x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
                nfe += 1
            path.append(x.detach().cpu())
    return x, path, x0_est

def to_d_stoch(x, x0_hat, x_T, alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv, epsilon):
    """Converts a denoiser output to a Karras ODE derivative."""
    # score = (alpha * x0_hat + beta * x_T - x) / gamma ** 2
    z_hat = (x - alpha * x0_hat - beta * x_T ) / gamma
    if gamma == 0:
        return alpha_deriv * x0_hat + beta_deriv * x_T, 0
    else:  
        # diffusion_term = alpha_deriv / alpha * x + (beta_deriv - alpha_deriv / alpha * beta) * x_T - (gamma * gamma_deriv - alpha_deriv / alpha * gamma ** 2) * score - epsilon * score
        diffusion_term = alpha_deriv * x0_hat + beta_deriv * x_T + ( gamma_deriv + epsilon / gamma) * z_hat
        drift_term =  (2 * epsilon) ** 0.5
        return diffusion_term, drift_term

def get_sigma(beta_d=2., beta_min=0.1, sigma_max=1.):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    # s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    snr = lambda t: 1 / sigma(t) ** 2
    return sigma, snr


       
        
        # scheme 2
        # if i >= len(indices) - 2:
        #         diffusion_term, drift_term = to_d_stoch(x, x0_hat, x_T_s, alpha(sigmas[i]), alpha_deriv(sigmas[i]), beta(sigmas[i]), beta_deriv(sigmas[i]), gamma(sigmas[i]), gamma_deriv(sigmas[i]), 0)
        #         x = x + diffusion_term * dt
                # x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s
        # else:
        #     z_hat = (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s) / gamma(sigmas[i])
        #     z_bar = th.randn_like(x)
        #     rho = (2 * epsilon(sigmas[i])) ** 0.5
        #     assert gamma[sigmas[i-1]] ** 2 - rho ** 2 > 0
        #     z = (gamma[sigmas[i-1]] ** 2 - rho ** 2) ** 0.5 * z_hat + rho * z_bar
        #     x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + z
        
@th.no_grad()
def sample_stoch(
    denoiser,
    x,
    sigmas,
    route,
    progress=False,
    callback=None,
    churn_step_ratio=0.,
    route_scaling = 0,
    smooth = 0
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    x = x + smooth * th.randn_like(x)
    x_T_s = x # smoothed x_T
    # path = [x.detach().cpu()]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0 # number of function evaluations
    
    ###########################################
    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = route
    epsilon = lambda t: churn_step_ratio * (gamma(t) * gamma_deriv(t) - alpha_deriv(t) / alpha(t) * gamma(t) ** 2)
    path = [x.detach().cpu()]
    x0_est = [x.detach().cpu()]
    for i in indices:
        x0_hat = denoiser(x, sigmas[i] * s_in, x_T)
        x0_est.append(x0_hat.detach().cpu())
        dt = (sigmas[i+1] - sigmas[i])  
        #  # DBIM
        # if i == 0:
        #     x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + gamma(sigmas[i+1]) * th.randn_like(x) 
        # else:
        #     x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + (gamma(sigmas[i+1]) / gamma(sigmas[i])) * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s)
        
        if i >= len(indices) - 2:
            x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + (gamma(sigmas[i+1]) / gamma(sigmas[i])) * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s)
        else:
            diffusion_term, drift_term = to_d_stoch(x, x0_hat, x_T_s, alpha(sigmas[i]), alpha_deriv(sigmas[i]), beta(sigmas[i]), beta_deriv(sigmas[i]), gamma(sigmas[i]), gamma_deriv(sigmas[i]), epsilon(sigmas[i]))
            x = x + diffusion_term * dt + th.randn_like(x) *((dt).abs() ** 0.5) * drift_term
        
        nfe += 1
        # print(f"nfe: {nfe}")
        path.append(x.detach().cpu())

    return x, path, x0_est


# @th.no_grad()
def sample_guidance(
    denoiser,
    x,
    sigmas,
    route,
    device,
    progress=False,
    callback=None,
    churn_step_ratio=0.,
    route_scaling = 0,
    smooth = 0
):
    x_T = x
    x = x + smooth * th.randn_like(x)
    x_T_s = x # smoothed x_T
    # path = [x.detach().cpu()]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0 # number of function evaluations
    ref_img_path = f'./samples/real_images/e2h.pt'
    # ref_imgs = th.load(ref_img_path)[:len(x)].to(device)
    ref_imgs = th.load(ref_img_path)[0].unsqueeze(0).repeat(len(x), 1, 1, 1).to(device)
    
    feature_extractor = th.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet18', 
                                     pretrained=True)
    
    # Remove final classification layer
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    
    # Initialize calculator
    calculator = FeatureDifferenceCalculator(feature_extractor, device=device)
    
    ###########################################
    alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv = route
    # epsilon = lambda t: churn_step_ratio * (gamma(t) * gamma_deriv(t) - alpha_deriv(t) / alpha(t) * gamma(t) ** 2)
    path = [x.detach().cpu()]
    x0_est = [x.detach().cpu()]
    for i in indices:
        with th.no_grad():
            x0_hat = denoiser(x, sigmas[i] * s_in, x_T)
        # x0_hat.requires_grad_()
        # residual = x0_hat - ref_imgs
        # residual_norm = th.linalg.norm(residual) ** 2
        # norm_grad = th.autograd.grad(outputs=residual_norm, inputs=x0_hat)[0]
        feature_diff = calculator.calculate_feature_difference(ref_imgs, x0_hat)
        # print("Feature difference shape:", feature_diff.shape)
        residual = calculator.calculate_residual(feature_diff)
        print("Residual:", th.mean(residual))
        gradients = calculator.calculate_gradient(x0_hat, ref_imgs)
        print("Gradients shape:", gradients.shape)
        
        if i<= 5:
            x0_hat = x0_hat - 0.2 * (len(indices) - i) / (len(indices) + 1)* gradients
        with th.no_grad():
            x0_est.append(x0_hat.detach().cpu())
            # dt = (sigmas[i+1] - sigmas[i])  
            #  # DBIM
            if i == 0:
                x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + gamma(sigmas[i+1]) * th.randn_like(x) 
            else:
                x = alpha(sigmas[i+1]) * x0_hat + beta(sigmas[i+1]) * x_T_s + (gamma(sigmas[i+1]) / gamma(sigmas[i])) * (x - alpha(sigmas[i]) * x0_hat - beta(sigmas[i]) * x_T_s)
            
            nfe += 1
            # print(f"nfe: {nfe}")
            path.append(x.detach().cpu())

    return x, path, x0_est

def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    route,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=1.00,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    route_scaling = 0,
    guidance=1,
    smooth = 0
):
    assert sampler in ["heun", "stoch", "guidance"]
    # sigmas = get_sigmas_karras(steps, 0.002, 1-1e-4, rho, device=device)
    # rho = 10.
    # sigma_min = 0.001
    # print(f"sigma_min: {sigma_min}")
    sigmas = get_sigmas_karras(steps, sigma_min+1e-4, sigma_max-1e-4, rho, device=device)

    sampler_args = dict(
            churn_step_ratio=churn_step_ratio,
            route_scaling = route_scaling,
            smooth = smooth
        )
    def denoiser(x_t, sigma, x_T=None):
        denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)

        if clip_denoised:
            denoised = denoised.clamp(-1, 1)

        return denoised
    if sampler == "heun":
        x_0, path, x0_est = sample_heun(
            denoiser,
            x_T,
            sigmas,
            route,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
    elif sampler == "stoch":
        x_0, path, x0_est = sample_stoch(
            denoiser,
            x_T,
            sigmas,
            route,
            progress=progress,
            callback=callback,
            **sampler_args,
        )
    elif sampler == "guidance":
        x_0, path, x0_est = sample_guidance(
            denoiser,
            x_T,
            sigmas,
            route,
            device,
            progress=progress,
            callback=callback,
            **sampler_args,
        )

    return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], [x.clamp(-1, 1) for x in x0_est]


def new_sigmas(sigmas):
    right = sigmas[:-1]/2
    right = right[:-1:2]
    left = 1 - th.flip(right, dims=[0])
    return th.concat([left, right, right.new_zeros([1])])


def sample_loop(diffusion,
                model,
                dataloader,
                num_samples,
                pred_mode,
                split,
                steps,
                clip_denoised,
                sampler,
                sigma_min,
                sigma_max,
                churn_step_ratio,
                route_scaling,
                rho,
                guidance,
                smooth=0):
      # distibuted setup
      dist_util.setup_dist()
      device = dist_util.dev()
      world_size = dist.get_world_size()
      batch_size = dataloader.batch_sampler.batch_size
      num_iters = (num_samples - 1) // (world_size * batch_size) + 1
      print(f"num_iters: {num_iters}")
      print("Sampling ...")
      # Get route
      route = get_route(pred_mode)
      
      sample_imgs = []
      real_imgs = []
      for i, data in tqdm(enumerate(dataloader), total=num_iters):
            if i >= num_iters: break
            # print(len(data))
            if split =='train':
                  y1, y0, index = data
            elif split == 'test':
                  y1, y0, index, _ = data
            if pred_mode.endswith("reverse"):
                  print("reverse")
                  y0, y1 = y1, y0
            y0 = y0.to(device) * 2 - 1
            y1 = y1.to(device) * 2 - 1
            index = index.to(device)
            model_kwargs = {'xT': y0}
            sample, _, _ = karras_sample(
                  diffusion,
                  model,
                  y0,
                  None,
                  route,
                  steps=steps,
                  model_kwargs=model_kwargs,
                  device=device,
                  clip_denoised=clip_denoised,
                  sampler=sampler,
                  sigma_min=sigma_min,
                  sigma_max=sigma_max,
                  churn_step_ratio=churn_step_ratio,
                  route_scaling = route_scaling,
                  rho=rho,
                  guidance=guidance,
                  smooth = smooth
            )
            # sample_imgs.append(sample)
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            if index is not None:
                  gathered_index = [th.zeros_like(index) for _ in range(dist.get_world_size())]
                  dist.all_gather(gathered_index, index)
                  gathered_samples = th.cat(gathered_samples)
                  gathered_index = th.cat(gathered_index)
                  gathered_samples = gathered_samples[th.argsort(gathered_index)]
            else:
                  gathered_samples = th.cat(gathered_samples)
            sample_imgs.append(gathered_samples)
            real_imgs.append(y1)
      
      sample_imgs = th.cat(sample_imgs, axis=0)
      sample_imgs = sample_imgs.reshape(-1, sample_imgs.shape[-3], sample_imgs.shape[-2], sample_imgs.shape[-1])
      sample_imgs = sample_imgs[:num_samples]
      
      real_imgs = th.cat(real_imgs, axis=0)
      real_imgs = real_imgs.reshape(-1, real_imgs.shape[-3], real_imgs.shape[-2], real_imgs.shape[-1])
      real_imgs = real_imgs[:num_samples]
      return real_imgs, sample_imgs
  
  
def sample_loop_s(diffusion,
                model,
                dataloader,
                num_samples,
                pred_mode,
                split,
                steps,
                clip_denoised,
                sampler,
                sigma_min,
                sigma_max,
                churn_step_ratio,
                route_scaling,
                rho,
                guidance,
                smooth=0,
                sample_x0 = False):
      # distibuted setup
      dist_util.setup_dist()
      device = dist_util.dev()
      world_size = dist.get_world_size()
      batch_size = dataloader.batch_sampler.batch_size
      num_iters = (num_samples - 1) // (world_size * batch_size) + 1
      print(f"num_iters: {num_iters}")
      print("Sampling ...")
      # Get route
      route = get_route(pred_mode)
      
      sample_imgs = []
      real_imgs = []
      for i, data in tqdm(enumerate(dataloader), total=num_iters):
            if i >= num_iters: break
            # print(len(data))
            if split =='train':
                  y1, y0, index = data
            elif split == 'test':
                  y1, y0, index, _ = data
            if pred_mode.endswith("reverse"):
                  y0, y1 = y1, y0
            y0 = y0.to(device) * 2 - 1
            y1 = y1.to(device) * 2 - 1
            index = index.to(device)
            model_kwargs = {'xT': y0}
            sample, path, x0_est = karras_sample(
                  diffusion,
                  model,
                  y0,
                  None,
                  route,
                  steps=steps,
                  model_kwargs=model_kwargs,
                  device=device,
                  clip_denoised=clip_denoised,
                  sampler=sampler,
                  sigma_min=sigma_min,
                  sigma_max=sigma_max,
                  churn_step_ratio=churn_step_ratio,
                  route_scaling = route_scaling,
                  rho=rho,
                  guidance=guidance,
                  smooth = smooth
            )
            # sample_imgs.append(sample)
            if sample_x0: path = x0_est
            path = th.stack(path)
            path = path.permute(1, 0, 2, 3, 4)
            sample_imgs.append(path)
            real_imgs.append(y1)
      
      sample_imgs = th.cat(sample_imgs, axis=0)
      sample_imgs = sample_imgs.reshape(-1, sample_imgs.shape[-4], sample_imgs.shape[-3], sample_imgs.shape[-2], sample_imgs.shape[-1])
      sample_imgs = sample_imgs[:num_samples]
      
      real_imgs = th.cat(real_imgs, axis=0)
      real_imgs = real_imgs.reshape(-1, real_imgs.shape[-3], real_imgs.shape[-2], real_imgs.shape[-1])
      real_imgs = real_imgs[:num_samples]
      return real_imgs, sample_imgs