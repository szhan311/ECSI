
def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=1.0,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0.,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        unet_type='adm',
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        attention_type='flash',
        learn_sigma=False,
        condition_mode=None,
        pred_mode='ve',
        weight_schedule="karras",
    )
    return res

def sample_defaults():
    return dict(
        generator="determ",
        clip_denoised=True,
        sampler="euler",
        s_churn=0.0,
        s_tmin=0.002,
        s_tmax=80,
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )


def train_defaults():
    res = dict(
        data_dir="",
        dataset='edges2handbags',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=2,
        use_augment=False
    )
    return res

def test_defaults():
    res = dict(
        data_dir="datasets/edges2handbags", ## only used in bridge
        dataset='edges2handbags',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        split='train',
        churn_step_ratio=0.,
        rho=7.0,
        steps=40,
        model_path="models/e2h_ema_0.9999_420000.pt",
        exp="",
        seed=42,
        ts="",
        upscale=False,
        num_workers=2,
        guidance=1.,
    )
    return res