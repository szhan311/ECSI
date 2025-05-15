import torch as th

from piq import LPIPS
from utils.nn import mean_flat, append_dims
from sibm.route import *

'''
KarrasDenoiser, and create_model_and_diffusion
'''

class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min = 1e-4,
        sigma_max = 1.0,
        cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
        rho=7.0,
        image_size=64,
        weight_schedule="", # "karras", "sibm"
        pred_mode="vp",
        loss_norm="lpips",
        smooth = 0):
        # for preconditioning
        self.sigma_0 = sigma_data
        self.sigma_1 = sigma_data + smooth
        self.sigma_01 = cov_xy
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max - 1e-4
        self.rho = rho
        self.smooth = smooth
        self.route = get_route(pred_mode)
        
        self.weight_schedule = weight_schedule # ["snr", "snr+1", "Karas", ...]
        self.pred_mode = pred_mode # ["VP", "VE"]
        self.loss_norm = loss_norm # ["lpips"]
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        
        self.num_timesteps = 40
        self.image_size = image_size

    def _get_bridge_scalings(self, t):
        alpha, _, beta, _, gamma, _ = self.route
        A = alpha(t) ** 2 * self.sigma_0 ** 2 + beta(t) ** 2 * self.sigma_1 ** 2 + 2 * alpha(t) * beta(t) * self.sigma_01 + gamma(t) ** 2
        c_in = 1 / A ** 0.5
        c_skip = (alpha(t) * self.sigma_0 ** 2 + beta(t) * self.sigma_01 ** 2) / A
        c_out = (beta(t) ** 2 * self.sigma_0 ** 2 * self.sigma_1 ** 2 - beta(t) ** 2 * self.sigma_01 ** 2 + gamma(t) ** 2 * self.sigma_0 ** 2) ** 0.5 * c_in
        # weight = 1 / c_out ** 2
        return c_in, c_skip, c_out
    
    def _get_weightings(self, t):
        if self.weight_schedule == "":
            weight = th.ones_like(t)
        elif self.weight_schedule == "snr":
            alpha, _, beta, _, gamma, _ = self.route
            weight = gamma(t) ** -2
        elif self.weight_schedule == "karras":
            alpha, _, beta, _, gamma, _ = self.route
            weight = gamma(t) ** -2 + self.sigma_0 ** -2
        elif self.weight_schedule == "sibm":
            alpha, _, beta, _, gamma, _ = self.route
            A = alpha(t) ** 2 * self.sigma_0 ** 2 + beta(t) ** 2 * self.sigma_1 ** 2 + 2 * alpha(t) * beta(t) * self.sigma_01 + gamma(t) ** 2
            weight = A / (beta(t) ** 2 * self.sigma_0 ** 2 * self.sigma_1 ** 2 - beta(t) ** 2 * self.sigma_01 ** 2 + gamma(t) ** 2 * self.sigma_0 ** 2)
        return weight
    
    def bridge_sample(self, x0, x1, t):
        x1 = x1 + self.smooth * th.randn_like(x1)
        t = append_dims(t, x0.ndim)
        z = th.randn_like(x0)
        alpha, _, beta, _, gamma, _ = self.route
        return alpha(t) * x0 + beta(t) * x1 + gamma(t) * z
    
    def denoise(self, model, x_t, t ,**model_kwargs):
        c_in, c_skip, c_out = [append_dims(x, x_t.ndim) for x in self._get_bridge_scalings(t)]
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44) # why?
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return denoised

    def training_score_losses(self, model, x_start, t, model_kwargs=None, noise=None, vae=None):
        assert model_kwargs is not None
        xT = model_kwargs['xT']
        if noise is None:
            noise = th.randn_like(x_start)
        t = th.minimum(t, th.ones_like(t) * self.sigma_max)
        terms = {}
        
        x_t = self.bridge_sample(x_start, xT, t)
        # denoise
        denoised = self.denoise(model, x_t, t, **model_kwargs)
        # calculate weights lambda(t)
        weights  = self._get_weightings(t)
        weights = append_dims((weights), x_start.ndim)
        
        
        alpha, _, beta, _, gamma, _ = self.route
        t = append_dims(t, x_start.ndim)
        cond_score = (alpha(t) * x_start + beta(t) * xT - x_t) / gamma(t) ** 2
        
        x0_hat = (denoised *  gamma(t) ** 2 + x_t - beta(t) * xT) / alpha(t)
        terms["xs_mse"] = mean_flat((x0_hat - x_start) ** 2)
        
        terms["mse"] = mean_flat((denoised - cond_score) ** 2)

        terms["loss"] = terms["mse"]

        return terms
    
    def training_bridge_losses(self, model, x_start, t, model_kwargs=None, noise=None, vae=None):
        assert model_kwargs is not None
        xT = model_kwargs['xT']
        if noise is None:
            noise = th.randn_like(x_start)
        t = th.minimum(t, th.ones_like(t) * self.sigma_max)
        terms = {}
        
        x_t = self.bridge_sample(x_start, xT, t)
        # denoise
        denoised = self.denoise(model, x_t, t, **model_kwargs)
        # calculate weights lambda(t)
        weights  = self._get_weightings(t)
        weights = append_dims((weights), x_start.ndim)

        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        terms["loss"] = terms["mse"]

        return terms




# create neural network
from utils.unet import UNetModel
from utils.edm_unet import SongUNet

NUM_CLASSES = 1000
def create_model(
    image_size,
    in_channels,
    num_channels,
    num_res_blocks,
    unet_type="adm",
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    attention_type='flash',
    condition_mode=None,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
   
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    
    if unet_type == 'adm':
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels*2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            attention_type=attention_type,
            condition_mode=condition_mode,
        )
    elif unet_type == 'edm':
        return SongUNet(
            img_resolution=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels*2),
            num_blocks=4,
            attn_resolutions=[16],
            dropout=dropout,
            channel_mult=channel_mult,
            channel_mult_noise=2,
            embedding_type='fourier',
            encoder_type='residual', 
            decoder_type='standard',
            resample_filter=[1,3,3,1],
        )
    else:
        raise ValueError(f"Unsupported unet type: {unet_type}")

def create_model_and_diffusion(
    image_size,
    in_channels,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    attention_type,
    condition_mode,
    pred_mode,
    weight_schedule,
    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    cov_xy=0.,
    unet_type='adm',
):
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        unet_type=unet_type,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
        condition_mode=condition_mode,
    )
    diffusion = KarrasDenoiser(
        sigma_data=sigma_data,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        # beta_d=beta_d,
        # beta_min=beta_min,
        cov_xy=cov_xy,
        image_size=image_size,
        weight_schedule=weight_schedule,
        pred_mode=pred_mode
    )
    return model, diffusion
