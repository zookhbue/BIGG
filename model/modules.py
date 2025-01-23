import os
import json
import copy
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import nnls
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from .blocks import (
    Embedding,
    LayerNorm,
    LinearNorm,
    ConvNorm,
    BatchNorm1dTBC,
    EncSALayer,
    Mish,
    DiffusionEmbedding,
    ResidualBlock,
    Downsample,
    Upsample1,
    SeTeBlock,
    Attention1,
    Attention2,
    E2EBlock
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm="ln", ffn_padding="SAME", ffn_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size, num_heads, dropout=dropout,
            attention_dropout=0.0, relu_dropout=dropout,
            kernel_size=kernel_size,
            padding=ffn_padding,
            norm=norm, act=ffn_act)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, in_planes, d, num_classes=2):
        super(BrainNetCNN, self).__init__()
        self.in_planes = in_planes
        self.d = d

        self.cnn1 = torch.nn.Conv2d(in_planes, 8, (1, self.d), bias=True)
        self.cnn2 = torch.nn.Conv2d(in_planes, 8, (self.d, 1), bias=True)
        self.cnn3 = torch.nn.Conv2d(8, 16, (1, self.d), bias=True)
        self.cnn4 = torch.nn.Conv2d(8, 16, (self.d, 1), bias=True)

        self.E2N = torch.nn.Conv2d(16, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 64, (self.d, 1))
        self.dense1 = torch.nn.Linear(64, 128)
        self.dense2 = torch.nn.Linear(128, 10)
        self.dense3 = torch.nn.Linear(10, num_classes)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        e2econv1 = torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)
        out = F.leaky_relu(e2econv1, negative_slope=0.33)

        a = self.cnn3(out)
        b = self.cnn4(out)
        e2econv2 = torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)
        out = F.leaky_relu(e2econv2, negative_slope=0.33)

        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        # out = F.softmax(out, dim=1)

        return out

class Denoiser(nn.Module):
    """ Conditional Diffusion Denoiser """

    def __init__(self):
        super(Denoiser, self).__init__()

        self.resolution=4
        self.channels1 = [90,180,360,720,1440]
        self.t_dim_v1 = [187,94, 47,24,12]
        self.num_heads_v1 = [11, 2, 1, 4, 4]
        self.channels2 = [1440, 720, 360, 180, 90]
        self.t_dim_v2 = [12, 24, 47,94, 187]
        self.num_heads_v2 = [4, 4, 1, 2,11]

        self.Attention1 = Attention1(187)
        self.Attention2 = Attention2(187)
        self.ConvNorm = ConvNorm(90, 90, kernel_size=3, stride=1)
        self.LinearNorm = LinearNorm(187, 187)

        self.sample_layers_d = nn.ModuleList(
            [
                Downsample(self.channels1[i])
                for i in range(4)
            ]
        )
        self.residual_layers_d1 = nn.ModuleList(
            [
                SeTeBlock(self.t_dim_v1[i], self.num_heads_v1[i])
                for i in range(1,5)
            ]
        )
        self.residual_layers_d2 = nn.ModuleList(
            [
                SeTeBlock(self.t_dim_v1[i], self.num_heads_v1[i])
                for i in range(1, 5)
            ]
        )

        self.sample_layers_u = nn.ModuleList(
            [
                Upsample1(self.channels2[i])
                for i in range(4)
            ]
        )
        self.residual_layers_u1 = nn.ModuleList(
            [
                SeTeBlock(self.t_dim_v2[i], self.num_heads_v2[i])
                for i in range(1,4)
            ]
        )
        self.residual_layers_u2 = nn.ModuleList(
            [
                SeTeBlock(self.t_dim_v2[i], self.num_heads_v2[i])
                for i in range(1, 4)
            ]
        )

        self.diffusion_embedding = DiffusionEmbedding(self.t_dim_v1[0])
        self.mlp = nn.Sequential(
            LinearNorm(self.t_dim_v1[0], self.t_dim_v1[1]),
            Mish(),
            LinearNorm(self.t_dim_v1[1] , self.t_dim_v1[0])
        )
        self.BrainNetCNN = BrainNetCNN(1, 90, 2)


    def forward(self, mel, diffusion_step, conditioner, speaker_emb=1, mask=None):

        x = mel[:, 0]
        raw_x = x
        B,M,T = x.shape

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        x = x + diffusion_step.unsqueeze(1).repeat(1,M,1)
        for i in range(2):
            x1 = self.Attention1(x)
            x2 = self.Attention2(x1,conditioner)
            x3 = self.ConvNorm(x2) + x2
            x4 = self.LinearNorm(x3) + x3
            x = x4
        res0 = x4


        res1=[]
        x1 = x
        for i in range(self.resolution):
            x2 = self.sample_layers_d[i](x1)
            x3 = self.residual_layers_d1[i](x2)
            x4 = self.residual_layers_d1[i](x3)
            res1.append(x4)
            x1 = x4

        x1 = x4
        for i in range(self.resolution):
            x2 = self.sample_layers_u[i](x1, self.t_dim_v2[i+1])
            if i==3:
                x3 = x2
                x4 = x3
                x1 = x4
            else:
                x3 = self.residual_layers_u1[i](x2)
                x4 = self.residual_layers_u2[i](x3)
                x1 = x4 + res1[3-i-1]
        res2 = x1

        x0_pred = res0+res2
        mean_v = torch.mean(x0_pred, dim=[1,2]).unsqueeze(1).unsqueeze(1)
        max_v1, _ = torch.max(torch.abs(x0_pred), dim=1)
        max_v2, _ = torch.max(torch.abs(max_v1), dim=1)
        max_v2 = max_v2.unsqueeze(1).unsqueeze(1)
        x0_pred = (x0_pred - mean_v)/max_v2

        A = 1 - torch.eye(M).repeat(B,1,1)
        A = A.to(dtype=torch.float64)
        left = x0_pred.transpose(1, 2)
        right = raw_x.transpose(1,2)
        for i in range(B):
            for j in range(M):
                indx = list( range(M) )
                indx.remove(j)
                Aj, rnorm = nnls( left[i,:,indx].detach().numpy(),  right[i, :, j].detach().numpy() )
                A[i, indx, j] = torch.from_numpy(Aj).to(dtype=torch.float64)
        A = A.transpose(1,2)

        cls_pred = self.BrainNetCNN(A.unsqueeze(1).float())

        return x0_pred[:, None, :, :], A, cls_pred
