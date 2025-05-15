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