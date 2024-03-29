# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import math
import timm.models.vision_transformer

# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=False, **kwargs):
#         super(VisionTransformer, self).__init__(**kwargs)

#         # Added by Samar, need default pos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
#                                             cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         self.global_pool = global_pool
#         if self.global_pool:
#             norm_layer = kwargs['norm_layer']
#             embed_dim = kwargs['embed_dim']
#             self.fc_norm = norm_layer(embed_dim)

#             del self.norm  # remove the original norm
        
#         self.fc=nn.Linear(self.embed_dim, 1000)

#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]
        
#         return outcome #shape : BxD
#     def forward(self,x):
        
#         return self.fc(self.forward_features(x))
        
        
     
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_size = kwargs['patch_size']
        self.in_c = kwargs['in_chans']
        embed_dim = kwargs['embed_dim']
        depth = kwargs['depth']
        dropout = kwargs['drop_rate']

        # Added by Samar, need default pos embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        
        self.decoder_pred = nn.Linear(embed_dim, self.patch_size ** 2 * self.in_c, bias=True)  # decoder to patch

       
    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, time):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (N, L+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        

        for i, blk in enumerate(self.blocks):
            x = blk(x)  # (N, L+1, D)

        x = self.norm(x)

        x = self.decoder_pred(x)  # (N, L+1, p^2 * C)

        # Remove cls token
        x = x[:, 1:, :]  # (N, L, p^2 * C)

        # Unpatchify
        x = self.unpatchify(x, self.patch_size, self.in_c)  # (N, C, H, W)

        # # Final conv
        # x = self.final_conv(x)  # (N, C, H, W)

        return x
class ViTFinetune(VisionTransformer):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.num_timesteps = num_timesteps
        # self.use_temb = use_temb
        # if not self.use_temb:
        #     del self.temb
        #     del self.temb_blocks

        norm_layer = nn.LayerNorm
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)
        self.fc=nn.Linear(embed_dim, kwargs['num_classes'])
        del self.decoder_pred
    

    def forward(self, x):
        print('in vit finetune')
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # (N, L+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)



        for i, blk in enumerate(self.blocks):
     
                x = blk(x)

        x = self.norm(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        x = self.fc_norm(x)  # (N, D)

        outcome = self.fc(x)  # (N, #classes)
        return outcome


def vit_base_patch16(**kwargs):
    print('in base vit')
    model = VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

# from functools import partial

# import torch
# import torch.nn as nn

# import timm.models.vision_transformer
# from timm.models.vision_transformer import PatchEmbed
# # from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

# def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
#     """
#     grid_size: int of the grid height and width
#     return:
#     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)

#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     if cls_token:
#         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
#     return pos_embed


# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0

#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb


# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     """
#     embed_dim: output dimension for each position
#     pos: a list of positions to be encoded: size (M,)
#     out: (M, D)
#     """
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float)
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega  # (D/2,)

#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

#     emb_sin = np.sin(out) # (M, D/2)
#     emb_cos = np.cos(out) # (M, D/2)

#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb

# class GroupChannelsVisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=False, channel_embed=256,
#                  channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)), **kwargs):
#         super().__init__(**kwargs)
#         img_size = 53
#         patch_size = 16
#         in_c = 3
#         embed_dim = 768

#         self.channel_groups = channel_groups

#         self.patch_embed = nn.ModuleList([PatchEmbed(img_size, patch_size, len(group), embed_dim)
#                                           for group in channel_groups])
#         # self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
#         num_patches = self.patch_embed[0].num_patches

#         # Positional and channel embed
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         num_groups = len(channel_groups)
#         self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
#         chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
#         self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

#         # Extra embedding for cls to fill embed_dim
#         self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
#         channel_cls_embed = torch.zeros((1, channel_embed))
#         self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

#         self.global_pool = global_pool
#         if self.global_pool:
#             norm_layer = kwargs['norm_layer']
#             embed_dim = kwargs['embed_dim']
#             self.fc_norm = norm_layer(embed_dim)

#             del self.norm  # remove the original norm

#     def forward_features(self, x):
#         b, c, h, w = x.shape

#         x_c_embed = []
#         for i, group in enumerate(self.channel_groups):
#             x_c = x[:, group, :, :]
#             x_c_embed.append(self.patch_embed[i](x_c))  # (N, L, D)

#         x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
#         _, G, L, D = x.shape

#         # add channel embed
#         channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
#         pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

#         # Channel embed same across (x,y) position, and pos embed same across channel (c)
#         channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, c, L, cD)
#         pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, c, L, pD)
#         pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)

#         # add pos embed w/o cls token
#         x = x + pos_channel  # (N, G, L, D)
#         x = x.view(b, -1, D)  # (N, G*L, D)

#         cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # (1, 1, D)
#         # stole cls_tokens impl from Phil Wang, thanks
#         cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)  # (N, 1 + c*L, D)
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]

#         return outcome
#     def forward(self,x):
        
#         return self.fc(self.forward_features(x))


# def vit_base_patch16(**kwargs):
#     model = GroupChannelsVisionTransformer(
#         channel_embed=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_large_patch16(**kwargs):
#     model = GroupChannelsVisionTransformer(
#         channel_embed=256, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = GroupChannelsVisionTransformer(
#         embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model