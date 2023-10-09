from .imagen import Unet3D
from pynvml import *
import torch
import torch.nn as nn

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

class Unet3D0409(nn.Module):
    def __init__(self):
        super(Unet3D0409, self).__init__()
        self.unet = Unet3D(
            dim=128,
            text_embed_dim = 512,
            attn_dim_head = 64,
            attn_heads = 8,
            ff_mult = 2., 
            cond_images_channels = 3,
            channels = 3,
            dim_mults = (1, 2, 4, 8),
            ff_time_token_shift = True,         # this would do a token shift along time axis, at the hidden layer within feedforwards - from successful use in RWKV (Peng et al), and other token shift video transformer works
            lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
            layer_attns = False,
            layer_attns_depth = 1,
            layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
            attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
            time_rel_pos_bias_depth = 2,
            time_causal_attn = True,
            layer_cross_attns = True,
            use_linear_attn = False,
            use_linear_cross_attn = False,
            cond_on_text = True,
            max_text_len = 32,
            memory_efficient = True,
            final_conv_kernel_size = 3,
            self_cond = False,
            combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
            pixel_shuffle_upsample = True,       # may address checkboard artifacts
            resize_mode = 'nearest'
            )

    def forward(self, x, t, x_cond, text_embed=None, **kwargs):
        x = self.unet(x, t, cond_images=x_cond, text_embeds=text_embed, **kwargs)
        return x
    
class BaseUnet64(Unet3D):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 160,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 4,
            ff_mult = 2.,
            memory_efficient = True,
            cond_images_channels=3,
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class Unet3D0410(nn.Module):
    def __init__(self):
        super(Unet3D0410, self).__init__()
        self.unet = BaseUnet64(
            time_causal_attn = False,
        )
    
    def forward(self, x, t, x_cond, text_embed=None, **kwargs):
        x = self.unet(x, t, cond_images=x_cond, text_embeds=text_embed, **kwargs)
        return x
    
    def forward_with_cond_scale(self, x, t, x_cond, text_embed=None, cond_scale=1.0, **kwargs):
        x = self.unet.forward_with_cond_scale(x, t, cond_images=x_cond, text_embeds=text_embed, cond_scale=cond_scale, **kwargs)
        return x

    
    


