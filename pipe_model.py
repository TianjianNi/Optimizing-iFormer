# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inception transformer implementation.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import warnings
from timm.layers.helpers import to_2tuple

from torch.distributed.pipeline.sync import Pipe

_logger = logging.getLogger(__name__)


# TODO: cleanup
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# default_cfgs = {
#     'iformer_224': _cfg(),
#     'iformer_384': _cfg(input_size=(3, 384, 384), crop_pct=1.0),
# }


default_cfgs = {
    'iformer_small': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small.pth'),
    'iformer_base': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base.pth'),
    'iformer_large': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large.pth'),
    'iformer_small_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_small_384.pth',
                              input_size=(3, 384, 384), crop_pct=1.0),
    'iformer_base_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_base_384.pth',
                             input_size=(3, 384, 384), crop_pct=1.0),
    'iformer_large_384': _cfg(url='https://huggingface.co/sail/dl2/resolve/main/iformer/iformer_large_384.pth',
                              input_size=(3, 384, 384), crop_pct=1.0),
}


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=16, stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x


class FirstPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, kernel_size=3, stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj1 = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.gelu1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        return x


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2

        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()

        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)

        hx = torch.cat((cx, px), dim=1)
        return hx


class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        return xa


class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim

        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  pool_size=pool_size, )

        self.conv_fuse = nn.Conv2d(low_dim + high_dim * 2, low_dim + high_dim * 2, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim * 2)
        self.proj = nn.Conv2d(low_dim + high_dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)

        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                 attn=Mixer,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                         attention_head=attention_head, pool_size=pool_size, )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            # print('use layer scale init value {}'.format(layer_scale_init_value))
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class PipeInceptionTransformer(nn.Module):
    # Forward pipelining for 4 GPUs
    # In official implementation, default num_classes was 1000
    def __init__(self,
                 dev0, dev1, dev2, dev3,
                 img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dims=None, depths=None,
                 num_heads=None, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 attention_heads=None,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 checkpoint_path=None,
                 **kwargs,
                 ):

        super().__init__()
        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)

        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2
        self.dev3 = dev3

        self.num_classes = num_classes

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.patch_embed = FirstPatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0]).to(self.dev0)
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[0])).to(self.dev0)
        self.blocks1 = nn.Sequential(*[
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, ).to(self.dev0)
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            # )
            for i in range(0, st2_idx)])

        self.patch_embed2 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1]).to(self.dev1)
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[1])).to(self.dev1)
        self.blocks2 = nn.Sequential(*[
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, ).to(self.dev1)
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, channel_layer_scale=channel_layer_scale,
            # )
            for i in range(st2_idx, st3_idx)])

        self.patch_embed3 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2]).to(self.dev2)
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[2])).to(self.dev2)
        self.blocks3 = nn.Sequential(*[
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            ).to(self.dev2)
            for i in range(st3_idx, st4_idx)])

        self.patch_embed4 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3]).to(self.dev3)
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[3])).to(self.dev3)
        self.blocks4 = nn.Sequential(*[
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            ).to(self.dev3)
            for i in range(st4_idx, depth)])

        self.norm = norm_layer(embed_dims[-1])
        # Classifier head(s)
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # set post block, for example, class attention layers

        self.init_weights(weight_init)

        self.to(dev0)
        self.patch_embed = self.patch_embed.to(dev0)
        self.pos_embed1 = self.pos_embed1.to(dev0)
        self.blocks1 = self.blocks1.to(dev0)
        self.patch_embed2 = self.patch_embed2.to(dev1)
        self.pos_embed2 = self.pos_embed2.to(dev1)
        self.blocks2 = self.blocks2.to(dev1)
        self.patch_embed3 = self.patch_embed3.to(dev2)
        self.pos_embed3 = self.pos_embed3.to(dev2)
        self.blocks3 = self.blocks3.to(dev2)
        self.patch_embed4 = self.patch_embed4.to(dev3)
        self.pos_embed4 = self.pos_embed4.to(dev3)
        self.blocks4 = self.blocks4.to(dev3)

    def init_weights(self, mode=''):
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)

        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def forward_features_1(self, x):
        x = x.to(self.dev0)
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W)
        x = self.blocks1(x)
        return x

    def forward_features_2(self, x):
        x = x.permute(0, 3, 1, 2).to(self.dev1)
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W)
        x = self.blocks2(x)
        return x

    def forward_features_3(self, x):
        x = x.permute(0, 3, 1, 2).to(self.dev2)
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W)
        x = self.blocks3(x)
        return x

    def forward_features_4(self, x):
        x = x.permute(0, 3, 1, 2).to(self.dev3)
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W)
        x = self.blocks4(x)
        x = x.flatten(1, 2)

        x = x.to(self.dev0)
        x = self.norm(x)
        return x.mean(1)

    def forward(self, x):
        # *** Not using ThreadPoolExecutor is actually faster ***
        split_size = 20
        x_split = torch.split(x, split_size, dim=0)
        x_split_iter = iter(x_split)
        outputs = []

        # we have four layers, so using x1, x2, x3 as intermediate results
        # x3 is output from the 3rd layer, cloest to the output and x1 is cloest to the input
        x1 = self.forward_features_1(next(x_split_iter))
        x2 = x3 = None

        while x1 != None or x2 != None or x3 != None:
            if x3 != None:
                outputs.append(self.forward_features_4(x3))
                x3 = None

            if x2 != None:
                x3 = self.forward_features_3(x2)
                x2 = None

            if x1 != None:
                x2 = self.forward_features_2(x1)
                try:
                    x1 = self.forward_features_1(next(x_split_iter))
                except:
                    x1 = None

        output = torch.cat(outputs, dim=0)
        final_output = self.head(output)

        return final_output

@register_model
def pipe_iformer_small(dev0, dev1, dev2, dev3, pretrained=False, **kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3, 3, 9, 3]
    embed_dims = [96, 192, 320, 384]
    num_heads = [3, 6, 10, 12]
    attention_heads = [1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3

    model = PipeInceptionTransformer(
                                 dev0 = dev0,
                                 dev1 = dev1,
                                 dev2 = dev2,
                                 dev3 = dev3,
                                 img_size=224,
                                 depths=depths,
                                 embed_dims=embed_dims,
                                 num_heads=num_heads,
                                 attention_heads=attention_heads,
                                 use_layer_scale=True, layer_scale_init_value=1e-6,
                                 **kwargs)
    model.default_cfg = default_cfgs['iformer_small']
    if pretrained:
        url = model.default_cfg['url']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
