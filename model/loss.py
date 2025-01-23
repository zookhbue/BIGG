import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def get_lsgan_losses_fn():

    def jcu_loss_fn(logit_cond, label_fn, mask=None):
        cond_loss = F.mse_loss(logit_cond, label_fn(logit_cond), reduction="none" if mask is not None else "mean")
        cond_loss = (cond_loss * mask).sum() / mask.sum() if mask is not None else cond_loss
        return cond_loss

    def d_loss_fn(r_logit_cond, f_logit_cond, mask=None):
        r_loss = jcu_loss_fn(r_logit_cond, torch.ones_like, mask)
        f_loss = jcu_loss_fn(f_logit_cond, torch.zeros_like, mask)
        return r_loss, f_loss

    def g_loss_fn(f_logit_cond, mask=None):
        f_loss = jcu_loss_fn(f_logit_cond, torch.ones_like, mask)
        return f_loss

    return d_loss_fn, g_loss_fn



def get_gan_loss_fn(criterion=nn.BCELoss()):

    def d_loss_fn(r_logit_cond, f_logit_cond, mask=None):
        r_loss = criterion(r_logit_cond, torch.ones_like(r_logit_cond))
        f_loss = criterion(f_logit_cond, torch.zeros_like(f_logit_cond))
        return r_loss, f_loss

    def g_loss_fn(f_logit_cond, mask=None):
        f_loss = criterion(f_logit_cond, torch.ones_like(f_logit_cond))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'lsgan':
        return get_lsgan_losses_fn()
    else:
        return get_gan_loss_fn()


class DiffGANTTSLoss(nn.Module):
    """ DiffGAN-TTS Loss """

    def __init__(self, args):
        super(DiffGANTTSLoss, self).__init__()
        self.model = args.model
        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn("gan")

    def forward(self, model, inputs, predictions, A, cls_pred, coarse_mels=None, Ds=None):
        
        mel_targets, coarse, label_true = inputs
        mel_predictions= predictions

        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        recon_loss = mel_loss
        scp_loss = self.scp_loss(A, 1.9)
        cls_loss = self.cls_loss(label_true, cls_pred)

        return (
            recon_loss, scp_loss, cls_loss
        )

    def l1_loss(self, output, target):
        l1_loss = F.l1_loss(output, target)
        return l1_loss

    def scp_loss(self, A, lamb):
        return torch.mean(A)*lamb

    def cls_loss(self, label_true, label_pred):
        loss = nn.CrossEntropyLoss()
        return loss(label_pred, label_true)

