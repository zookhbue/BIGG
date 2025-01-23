import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial
from inspect import isfunction

from .modules import Denoiser

def get_noise_schedule_list(timesteps, min_beta=1e-4, max_beta=0.02):
    schedule_list = np.linspace(min_beta, max_beta, timesteps)
    return schedule_list

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class GaussianDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = args.model
        self.denoise_fn = Denoiser()
        self.mel_bins = 90

        betas = get_noise_schedule_list(timesteps=4)
  
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = "l1"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # in diffuse process q(x_t | x_0), compute the mean = sqrt_alphas_cumprod*x0; variance = 1. - self.alphas_cumprod;
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # x_0 -> x_t  reverse transform from equation q(x_t | x_0)
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute mean_t, variance_t, for q(x_{t-1} | x_t, x_0)
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # q(x_{t-1} | x_t, x_0)
    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device 
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # q( x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def diffuse_fn(self, x_start, t, noise=None):
        x_start = x_start.transpose(1, 2)[:, None, :, :]
        zero_idx = t < 0
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx]
        return out

    def forward(self, mel, cond, spk_emb=1, mel_mask=1, coarse_mel=None, clip_denoised=True):
        b, *_, device = *cond.shape, cond.device
        x_t = x_t_prev = x_t_prev_pred = t = None
        cond = cond.transpose(1, 2)
        self.cond = cond.detach()
        if mel is None:
            print("skip!\n")
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            # Diffusion
            x_t = self.diffuse_fn(mel, t) * mel_mask
            x_t_prev = self.diffuse_fn(mel, t - 1) * mel_mask

            # Predict x_{start}
            x_0_pred,A,cls_pred = self.denoise_fn(x_t, t, cond) * mel_mask
            if clip_denoised:
                x_0_pred.clamp_(-1., 1.)

            # Sample x_{t-1} using the posterior distribution
            if self.model != "shallow":
                x_start = x_0_pred
            else:
                x_start = self.norm_spec(coarse_mel)
                x_start = x_start.transpose(1, 2)[:, None, :, :]
            x_t_prev_pred = self.q_posterior_sample(x_start=x_start, x_t=x_t, t=t) * mel_mask

            x_0_pred = x_0_pred[:, 0].transpose(1, 2)
            x_t = x_t[:, 0].transpose(1, 2)
            x_t_prev = x_t_prev[:, 0].transpose(1, 2)
            x_t_prev_pred = x_t_prev_pred[:, 0].transpose(1, 2)
        return x_0_pred, x_t, x_t_prev, x_t_prev_pred, t,A,cls_pred


    def out2mel(self, x):
        return x
