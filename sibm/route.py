import torch as th
import numpy as np
from matplotlib import pyplot as plt
import itertools

def get_route(pred_mode):
        if pred_mode.startswith("vp"):
            return VP_route()
        elif pred_mode.startswith("ve"):
            return VE_route()
        elif pred_mode.startswith("linear"):
            _, gamma_max = ["".join(g) for k, g in itertools.groupby(pred_mode, str.isalpha)]
            gamma_max = float(gamma_max)    
            return linear_route(gamma_max=gamma_max)
        else:
            return NotImplementedError


def linear_route(gamma_max = 1):
    alpha = lambda t: 1 - t
    alpha_deriv = lambda t: - th.ones_like(t)

    beta = lambda t: t
    beta_deriv = lambda t: th.ones_like(t)
    gamma = lambda t: gamma_max * 2 * (t * (1 - t))  ** 0.5
    gamma_deriv = lambda t: gamma_max  * 2 * (1 - 2 * t) / (2 * (t * (1 - t))  ** 0.5)
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



def VE_route(sigma_max=1.):
      alpha = lambda t: 1 - (t ** 2) / (sigma_max ** 2)
      alpha_deriv = lambda t: - 2 * t / (sigma_max ** 2)

      beta = lambda t: (t ** 2) / (sigma_max ** 2)
      beta_deriv = lambda t: 2 * t / (sigma_max ** 2)

      gamma = lambda t: ((t ** 2) * alpha(t)).sqrt()
      gamma_deriv = lambda t: alpha(t).sqrt() + (alpha_deriv(t) * t / (2 * alpha(t).sqrt()))
      return alpha, alpha_deriv, beta, beta_deriv, gamma, gamma_deriv


if "__name__" == "__main__":
    Pred_mode = [ "vp"]
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