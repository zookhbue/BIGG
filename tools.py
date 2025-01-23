import os
import torch
import numpy as np
from model.diffgantts import DiffGANTTS, JCUDiscriminator

# class ScheduledOptim:
#     """ A simple wrapper class for learning rate scheduling """
#
#     def __init__(self, model, current_step):
#
#         self._optimizer = torch.optim.Adam(
#             model.parameters(),
#             betas = [0.9, 0.98],
#             eps = 0.000000001,
#             weight_decay = 0.0,
#         )
#         self.n_warmup_steps = 20
#         self.anneal_steps = [100, 300, 500]
#         self.anneal_rate = 0.3
#         self.current_step = current_step
#         self.last_lr = self.init_lr = np.power(256, -0.5)
#
#     def get_last_lr(self):
#         return self.last_lr
#
#     def step(self):
#         lr = self._update_learning_rate()
#         self._optimizer.step()
#         return lr
#
#     def zero_grad(self):
#         # print(self.init_lr)
#         self._optimizer.zero_grad()
#
#     def load_state_dict(self, path):
#         self._optimizer.load_state_dict(path)
#
#     def _get_lr_scale(self):
#         lr = np.min(
#             [
#                 np.power(self.current_step, -0.5),
#                 np.power(self.n_warmup_steps, -1.5) * self.current_step,
#             ]
#         )
#         for s in self.anneal_steps:
#             if self.current_step > s:
#                 lr = lr * self.anneal_rate
#         return lr
#
#     def _update_learning_rate(self):
#         """ Learning rate scheduling per step """
#         self.current_step += 1
#         self.last_lr = lr = self.init_lr * self._get_lr_scale()
#
#         for param_group in self._optimizer.param_groups:
#             param_group["lr"] = lr
#         return lr


def get_model(args, device, train=False):

    epoch = 1
    model = DiffGANTTS(args).to(device)
    discriminator = JCUDiscriminator().to(device)
    if args.restore_epoch:
        ckpt_path = os.path.join(
            train_config["ckpt_path"],
            "{}.pth.tar".format(args.restore_epoch),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        epoch = int(ckpt["epoch"])
        model.load_state_dict(ckpt["G"])
        discriminator.load_state_dict(ckpt["D"])
  
    if train:
        init_lr_G = 0.001
        init_lr_D = 0.0002
        betas = [0.5, 0.9]
        gamma = 0.999
        # optG_fs2 = ScheduledOptim(model, args.restore_epoch)
        optG = torch.optim.Adam(model.parameters(), lr=init_lr_G, betas=betas)
        optD = torch.optim.Adam(discriminator.parameters(), lr=init_lr_D, betas=betas)
        sdlG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma)
        sdlD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma)
        if args.restore_epoch and args.restore_epoch != 600: # should be initialized when "shallow"
            # optG_fs2.load_state_dict(ckpt["optG_fs2"])
            optG.load_state_dict(ckpt["optG"])
            optD.load_state_dict(ckpt["optD"])
            sdlG.load_state_dict(ckpt["sdlG"])
            sdlD.load_state_dict(ckpt["sdlD"])
        model.train()
        discriminator.train()
        return model, discriminator, optG, optD, sdlG, sdlD, epoch
    model.eval()
    model.requires_grad_ = False
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_netG_params(model_kernel):
    return list(model_kernel.C.parameters()) \
        + list(model_kernel.Z.parameters()) \
        + list(model_kernel.G.parameters())

def get_netD_params(model_kernel):
    return model_kernel.D.parameters()

def to_device(data, device):

    if len(data) == 3:
        (mels, coarse, label_true) = data
        mels = mels.to(device)
        coarse = coarse.to(device)
        label_true = label_true.to(device)

        return (mels, coarse,label_true)
        