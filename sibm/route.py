import torch as th
import numpy as np
from matplotlib import pyplot as plt

def get_route(pred_mode):
        if pred_mode == "vp":
            return VP_route()
        if pred_mode == "vpe":
            return VPE_route()
        if pred_mode == "vp_reverse":
            return VP_route()
        elif pred_mode.startswith("vpg1"): 
            return VPG_route(k=0.1)
        elif pred_mode.startswith("vpg2"): 
            return VPG_route(k=0.5)
        elif pred_mode.startswith("vpg3"): 
            return VPG_route(k=2.0)
        elif pred_mode.startswith("vpg4"): 
            return VPG_route(k=4.0)
        elif pred_mode.startswith("vpg5"): 
            return VPG_route(k=0.25)
        elif pred_mode.startswith("vpg6"): 
            return VPG_route(k=0.75)
        elif pred_mode.startswith("vpc1"): 
            return VPC_route(k=0.5)
        elif pred_mode.startswith("vpc2"): 
            return VPC_route(k=.7)
        elif pred_mode.startswith("vpc3"): 
            return VPC_route(k=1.0)
        elif pred_mode.startswith("vpc4"): 
            return VPC_route(k=2.0)
        elif pred_mode.startswith("vpc5"): 
            return VPC_route(k=4)
        elif pred_mode.startswith("vpc6"): 
            return VPC_route(k=1.6)
        elif pred_mode.startswith ("vpv1"): # VP varient 1
            return VPV1_route()
        elif pred_mode.startswith("vpv2"): # VP varient 2
            return VPV2_route()
        elif pred_mode.startswith("vpv3"): 
            return VPV3_route()
        elif pred_mode.startswith("vpv4"): 
            return VPV4_route()
        elif pred_mode.startswith("vpv5"): 
            return VPV2_route(beta_d=0.5)
        elif pred_mode.startswith("vpv6"): 
            return VPV2_route(beta_d=4)
        elif pred_mode.startswith("vpv7"): 
            return VPV2_route(beta_d=0.01)
        elif pred_mode.startswith("vpv8"): 
            return VPV2_route(beta_d=1.)
        elif pred_mode.startswith("ve"):
            return VE_route()
        elif pred_mode.startswith("sky"):
            return Sky_route(l = 1.6)
        elif pred_mode.startswith("exp"):
            return exp_route(k=2)
        elif pred_mode.startswith("linear1"):
            return linear_route(k=0.5)
        elif pred_mode.startswith("linear0"):
            return linear_route(k=0.)


def linear_route(k = 1):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: - th.ones_like(t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: k * (t * (1 - t))  ** 0.5
    gamma_deriv = lambda t: k * (1 - 2 * t) / (2 * (t * (1 - t))  ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv



def exp_route(k = 1):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: -th.ones_like(t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    
    gamma = lambda t: th.where(t < 0.5, th.e ** (k * t) - 1, th.e ** (k *(1 - t)) - 1)
    gamma_deriv = lambda t: th.where(t < 0.5, k * th.e ** (k * t), -k * th.e ** (k *(1 - t)))
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VP_route(beta_d=2., beta_min=0.1, sigma_max=1.):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    sigma_T = sigma(th.as_tensor(sigma_max))
    s_T = s(th.as_tensor(sigma_max))
    alpha =lambda t: s(t) * (1 - sigma(t) ** 2 / sigma_T**2)
    alpha_deriv = lambda t: s_deriv(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) - s(t) * 2 * sigma(t) * sigma_deriv(t) / sigma_T**2
    beta = lambda t: s(t) * sigma(t) ** 2 / (s_T  * sigma_T ** 2 )
    beta_deriv = lambda t: (s_deriv(t) * sigma(t) ** 2 + 2 * s(t) * sigma(t) * sigma_deriv(t)) / (s_T * sigma_T ** 2)
    gamma = lambda t: sigma(t) * s(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5
    gamma_deriv = lambda t: s(t) * ((sigma_deriv(t) * (sigma_T ** 2 - 2 * sigma(t) ** 2))/(sigma_T * (sigma_T**2 - sigma(t)**2)**0.5)) + s_deriv(t) * sigma(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPE_route(beta_d=2.07):
    # equivalant VP
    alpha = lambda t: 1 - t - 0.1 * t * (1 - t)
    alpha_deriv = lambda t: - th.ones_like(t) - 0.1 * (1 - 2 * t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: beta_d / 2 * (t * (1 - t))  ** 0.5
    gamma_deriv = lambda t: beta_d / 2 * (1 - 2 * t) / (2 * (t * (1 - t))  ** 0.5)
    
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPG_route(beta_d=2., beta_min=0.1, sigma_max=1., k = 1.):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    sigma_T = sigma(th.as_tensor(sigma_max))
    s_T = s(th.as_tensor(sigma_max))
    alpha =lambda t: s(t) * (1 - sigma(t) ** 2 / sigma_T**2)
    alpha_deriv = lambda t: s_deriv(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) - s(t) * 2 * sigma(t) * sigma_deriv(t) / sigma_T**2
    beta = lambda t: s(t) * sigma(t) ** 2 / (s_T  * sigma_T ** 2 )
    beta_deriv = lambda t: (s_deriv(t) * sigma(t) ** 2 + 2 * s(t) * sigma(t) * sigma_deriv(t)) / (s_T * sigma_T ** 2)
    gamma = lambda t: k * sigma(t) * s(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5
    gamma_deriv = lambda t:k * ( s(t) * ((sigma_deriv(t) * (sigma_T ** 2 - 2 * sigma(t) ** 2))/(sigma_T * (sigma_T**2 - sigma(t)**2)**0.5)) + s_deriv(t) * sigma(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPV1_route(beta_d=2., beta_min=0.1, sigma_max=1.):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    sigma_T = sigma(th.as_tensor(sigma_max))
    s_T = s(th.as_tensor(sigma_max))
    alpha =lambda t: s(t) * (1 - sigma(t) ** 2 / sigma_T**2)
    alpha_deriv = lambda t: s_deriv(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) - s(t) * 2 * sigma(t) * sigma_deriv(t) / sigma_T**2
    beta = lambda t: s(t) * sigma(t) ** 2 / (s_T  * sigma_T ** 2 )
    beta_deriv = lambda t: (s_deriv(t) * sigma(t) ** 2 + 2 * s(t) * sigma(t) * sigma_deriv(t)) / (s_T * sigma_T ** 2)
    gamma = lambda t: beta_d / 2 * (t * (1 - t))  ** 0.5
    gamma_deriv = lambda t: beta_d / 2 * (1 - 2 * t) / (2 * (t * (1 - t))  ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPC_route(beta_d=1., k =1, beta_min=0.1, sigma_max=1.):
    alpha = lambda t: 1 - t - 0.1 * t * (1 - t)
    alpha_deriv = lambda t: - th.ones_like(t) - 0.1 * (1 - 2 * t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: beta_d / 2 * (t ** k * (1 - t ** k))  ** 0.5
    gamma_deriv = lambda t: (beta_d * k * t**(k-1) * (1 - 2 * t**k)) / (4 * (t**k * (1 - t**k))**0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPV2_route(beta_d=2.):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: - th.ones_like(t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: beta_d / 2 * (t * (1 - t))  ** 0.5
    gamma_deriv = lambda t: beta_d / 2 * (1 - 2 * t) / (2 * (t * (1 - t))  ** 0.5)
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv



def VPV3_route(beta_d=2., beta_min=0.1, sigma_max=1.):
    sigma = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    sigma_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    s = lambda t: (1 + sigma(t) ** 2).rsqrt()
    s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    sigma_T = sigma(th.as_tensor(sigma_max))
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: - th.ones_like(t)
    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: sigma(t) * s(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5
    gamma_deriv = lambda t: s(t) * ((sigma_deriv(t) * (sigma_T ** 2 - 2 * sigma(t) ** 2))/(sigma_T * (sigma_T**2 - sigma(t)**2)**0.5)) + s_deriv(t) * sigma(t) * (1 - sigma(t) ** 2 / sigma_T ** 2) ** 0.5
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VPV4_route(beta_d=2., beta_min=0.1, sigma_max=1.):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: - th.ones_like(t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: beta_d / 2 * th.sin( th.pi * t)
    gamma_deriv = lambda t: beta_d / 2  * th.cos( th.pi * t) * th.pi
    return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def VE_route(sigma_max=1.):
      alpha = lambda t: 1 - (t ** 2) / (sigma_max ** 2)
      alpha_deriv = lambda t: - 2 * t / (sigma_max ** 2)

      beta = lambda t: (t ** 2) / (sigma_max ** 2)
      beta_deriv = lambda t: 2 * t / (sigma_max ** 2)

      gamma = lambda t: ((t ** 2) * alpha(t)).sqrt()
      gamma_deriv = lambda t: alpha(t).sqrt() + (alpha_deriv(t) * t / (2 * alpha(t).sqrt()))
      return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def Sky_route(l = 2, k = 2):
      alpha = lambda t: 1 - t ** l
      alpha_deriv = lambda t: - th.ones_like(t)

      beta = lambda t: t ** l
      beta_deriv = lambda t: th.ones_like(t)

      gamma = lambda t: (t ** k * (1 - t ** k))  ** 0.5
      gamma_deriv = lambda t: (k * t**(k-1) * (1 - 2 * t**k)) / (2 * (t**k * (1 - t**k))**0.5)
      return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


def Simple_route(k = 1):
      alpha = lambda t: 1 - (t ** 2) 
      alpha_deriv = lambda t: - 2 * t

      beta = lambda t: (t ** 2)
      beta_deriv = lambda t: 2 * t

      gamma = lambda t: k * (t ** 2 * (1 - t)) ** 0.5
      gamma_deriv = lambda t: k * ((1 - t) ** 0.5 - t / (2 * (1 - t) ** 0.5) )
      return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def Simple_route_2(k = 1):
      alpha = lambda t: th.cos( th.pi / 2 * t)
      alpha_deriv = lambda t: - th.sin (th.pi / 2 * t) * th.pi / 2

      beta = lambda t: 1 - th.cos( th.pi / 2 * t)
      beta_deriv = lambda t: th.sin( th.pi / 2 * t) * th.pi / 2

      gamma = lambda t: k * (t ** 2 * (1 - t)) ** 0.5
      gamma_deriv = lambda t: k * ((1 - t) ** 0.5 - t / (2 * (1 - t) ** 0.5) )
      return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv

def test_route_c2(k = 1):
      s = lambda t: th.cos( th.pi / 2 * t)
      s_deriv = lambda t: - th.sin (th.pi / 2 * t) * th.pi / 2
      
      sigma = lambda t: k * th.sin( th.pi * t)
      sigma_deriv = lambda t: k * th.cos( th.pi * t) * th.pi
      return s, s_deriv, sigma, sigma_deriv


if "__name__" == "__main__":
    Pred_mode = [ "vp", "sky"]
    t = th.linspace(0.000, 1-1e-4, 41)
    plt.figure(figsize=(12, 3), dpi=300)
    plt.subplot(1, 3, 1)
    for pred_mode in Pred_mode:
        alpha, _, beta, _, gamma, _ = get_route(pred_mode)
        plt.plot(t, alpha(t), label = pred_mode)
        plt.legend()
        plt.title("alpha(t)")
    plt.subplot(1, 3, 2)
    for pred_mode in Pred_mode:
        alpha, _, beta, _, gamma, _ = get_route(pred_mode)
        plt.plot(t, beta(t), label = pred_mode)
        plt.legend()
        plt.title("beta(t)")
    plt.subplot(1, 3, 3)
    for pred_mode in Pred_mode:
        alpha, _, beta, _, gamma, _ = get_route(pred_mode)
        plt.plot(t, gamma(t), label = pred_mode)
        plt.legend()
        plt.title("gamma(t)")