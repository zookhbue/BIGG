import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LinearNorm, ConvNorm, DiffusionEmbedding, Mish, Attention
from .diffusion import GaussianDiffusion
from .loss import get_adversarial_losses_fn


class DiffGANHDT(nn.Module):
    """ DiffGAN-TTS """

    def __init__(self, args):
        super(DiffGANHDT, self).__init__()
        self.model = args.model
        self.diffusion = GaussianDiffusion(args)

    def forward(
        self,
        mels,
        coarse
    ):

        if self.model == "naive":
            (
                output,
                x_ts,
                x_t_prevs,
                x_t_prev_preds,
                diffusion_step,
                A,
                cls_pred
            ) = self.diffusion(
                mels,
                coarse
            )
        else:
            raise NotImplementedError

        return [
            output, A, cls_pred,
            (x_ts, x_t_prevs, x_t_prev_preds),
            diffusion_step]

    def _detach(self, p):
        return p.detach() if p is not None and self.model == "shallow" else p


class Discriminator(nn.Module):
    """  Discriminator """

    def __init__(self):
        super(Discriminator, self).__init__()

        N = 90
        t_dim = 187
        t_dim_v = [187, 94, 47, 24]
        self.scale=4
        kernel_sizes=[0, 3,3,3]
        strides = [0, 2,2,2]

        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    N,
                    N,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(1,self.scale)
            ]
        )

        self.att_block = nn.ModuleList(
            [Attention(t_dim_v[0], 1),
             Attention(t_dim_v[1], 1),
             Attention(t_dim_v[2], 1),
             Attention(t_dim_v[3], 1),]
        )

        self.linear_block1 = nn.ModuleList(
            [LinearNorm(t_dim_v[0], 1),
             LinearNorm(t_dim_v[1], 1),
             LinearNorm(t_dim_v[2], 1),
             LinearNorm(t_dim_v[3], 1), ]
        )

        self.linear_block2 = nn.ModuleList(
            [LinearNorm(N, 1),
             LinearNorm(N, 1),
             LinearNorm(N, 1),
             LinearNorm(N, 1), ]
        )

        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, t):

        x_t_prevs = x_t_prevs.transpose(1,2)
        cond_feats = []
        x=x_t_prevs
        for i in range(self.scale):

            if (i==0):
                x = self.att_block[i](x)
                tmp = self.linear_block1[i](x).squeeze()
                val =  torch.mean( F.sigmoid(tmp), dim=1 )
                down_x = x
            else:
                down_x = self.conv_block[i-1](x)
                down_x = self.att_block[i](down_x)
                tmp = self.linear_block1[i](down_x).squeeze()
                val = torch.mean(F.sigmoid(tmp), dim=1)
                x = down_x
            cond_feats.append(val)

        cond_feats = torch.cat(  (cond_feats[0].unsqueeze(0),
                                                              cond_feats[1].unsqueeze(0),
                                                              cond_feats[2].unsqueeze(0),
                                                              cond_feats[3].unsqueeze(0)), dim=0)
        cond_feats = torch.mean(cond_feats, dim=0).unsqueeze(1)

        return cond_feats
