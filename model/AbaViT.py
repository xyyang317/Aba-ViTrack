import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from .layers import PatchEmbed
from timm.models.layers import Mlp, DropPath
from .utils import get_distribution_target, combine_tokens, recover_tokens
from torch.autograd import Variable


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None, masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask
        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_ACT(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4}

        self.index=index
        self.args = args

        if self.act_mode == 4:
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


    def forward_act(self, x, mask=None):

        bs, token, dim = x.shape

        if mask is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)))

        if self.act_mode==4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, 64, 192))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, 256, 192))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block_ACT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args, index=i, num_patches=self.patch_embed.num_patches+1)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.eps = 0.01
        for block in self.blocks:
            if args.act_mode == 1:
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        self.rho = None
        self.counter = None
        self.batch_cnt = 0

        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.rho_token_weight = None
        self.counter_token = None
        self.total_token_cnt = int((128/patch_size)**2 + (256/patch_size)**2) + self.num_tokens

        if args.distr_prior_alpha >0. :
            self.distr_target = torch.Tensor(get_distribution_target(standardized=True, target_depth=5)).cuda()
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        self.cat_mode = 'direct'

    def forward_features_act_token(self, z, x, t_mask=None, s_mask=None):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if not t_mask is None:
            t_mask = t_mask[:,0:t_mask.shape[1]:self.patch_size,0:t_mask.shape[2]:self.patch_size]
            s_mask = s_mask[:,0:s_mask.shape[1]:self.patch_size,0:s_mask.shape[2]:self.patch_size]
            t_mask1 = 1-t_mask
            s_mask1 = 1-s_mask
            t_mask = 1.5*t_mask1 + 1*t_mask
            s_mask = 1.5*s_mask1 + 1*s_mask
            t_mask = t_mask.view(t_mask.shape[0],-1)
            s_mask = s_mask.view(s_mask.shape[0],-1)

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        x = x + self.pos_embed_x
        z = z + self.pos_embed_z

        x = combine_tokens(z, x, mode=self.cat_mode)

        if not t_mask is None:
            self.rho_token_weight = combine_tokens(t_mask, s_mask, mode=self.cat_mode)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x)
        bs = x.size()[0]

        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        output = None
        out = x

        if self.args.distr_prior_alpha>0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1)
            block_output, h_lst = l.forward_act(out, 1.-mask_token.float())

            if self.args.distr_prior_alpha>0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()

            _, h_token = h_lst

            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            if i==len(self.blocks)-1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token

            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, t_mask=None, s_mask=None):
        if self.args.act_mode == 4:
            x, aux_dict = self.forward_features_act_token(z,x, t_mask=t_mask, s_mask=s_mask)
        else:
            print('Not implemented yet, please specify for token act.')
            exit()

        return x, aux_dict


from argparse import Namespace
__all__ = [
    'abavit_patch16_224'
]
def abavit_patch16_224():
    kwargs = {'num_classes': 1000, 'drop_rate': 0.0, 'drop_path_rate': 0.1}
    model_kwargs = {'act_mode': 4, 'gate_scale': 10.0, 'gate_center': 30.0,'distr_prior_alpha':0.01}
    model_kwargs = Namespace(**model_kwargs)

    kwargs['args'] = model_kwargs
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model








