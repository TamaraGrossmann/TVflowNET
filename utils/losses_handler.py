import torch
import numpy as np
import sys

sys.path.append('..')
from utils.div_and_tv import TV


#######################################################################
# Define Loss Handler, this includes:
# - to_np: converting torch to numpy 
# - moving_average: a moving average used in plotting the loss
# - get_temporal_loss: calculation of the temporal loss (δusδts = div(φs))
# - get_tv_loss: calculation of tv loss (TV(us) = -<us , φs>)
# - get_normalisation_loss: calculation of normalisation loss (||φs||_{\infty} <= 1)
# - get_initial_loss: calculation of L2 error between initial image and us(0)
# - plot: plotting the results for loss and 5 example times
# - save_losses: save the loss values
# - __call__: calculate the total loss
#######################################################################

class LossesHandler:
    def __init__(self, tv_loss_scaling_factor=1e-3, minimal_norm_loss_scaling_factor=1e-4, ε=1e-2, losses=None):
        if losses is None:
            losses = {
                'temporal_loss': [],
                'tv_loss': [],
                'normalization_loss': [],
                'inital_loss': [],
                'minimal_norm_loss': [],
                'total_loss': [],
            }

        self.tv_loss_scaling_factor = tv_loss_scaling_factor
        self.minimal_norm_loss_scaling_factor = minimal_norm_loss_scaling_factor
        self.ε = ε
        self.losses = losses

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def get_temporal_loss(self, δusδts, div_φs, log=True):
        batch_size = δusδts.shape[0]
        temporal_loss = ((δusδts - div_φs) ** 2).view(batch_size, -1).mean(-1)
        temporal_loss = temporal_loss.mean()

        if log:
            self.losses['temporal_loss'].append(temporal_loss.item())
        return temporal_loss

    def get_tv_loss(self, us, div_φs, log=True):
        batch_size = us.shape[0]
        us_div_φs = (us * div_φs).view(batch_size, -1).sum(-1)
        tv = TV(us, ε=self.ε)
        assert np.allclose(us_div_φs.shape, tv.shape)

        tv_loss = ((us_div_φs + tv) ** 2).view(batch_size)  # .view(batch_size, -1).mean(-1)
        tv_loss = tv_loss.mean() * self.tv_loss_scaling_factor

        if log:
            self.losses['tv_loss'].append(tv_loss.item())
        return tv_loss

    def get_normalization_loss(self, φs, log=True):
        batch_size = φs.shape[0]
        normalization_loss = torch.relu(φs[:, :1] ** 2 + φs[:, 1:] ** 2 - 1)
        normalization_loss = normalization_loss.view(batch_size, -1).mean(-1)
        normalization_loss = normalization_loss.mean()

        if log:
            self.losses['normalization_loss'].append(normalization_loss.item())
        return normalization_loss

    def get_inital_loss(self, u0s, xs, log=True):
        batch_size = xs.shape[0]
        inital_loss = ((u0s - xs) ** 2).view(batch_size, -1).mean(-1)
        inital_loss = inital_loss.mean()

        if log:
            self.losses['inital_loss'].append(inital_loss.item())
        return inital_loss

    def get_minimal_norm_loss(self, div_φs, log=True):
        batch_size = div_φs.shape[0]
        minimal_norm_loss = ((div_φs) ** 2).view(batch_size, -1).mean(-1)
        minimal_norm_loss = minimal_norm_loss.mean()* self.minimal_norm_loss_scaling_factor

        if log:
            self.losses['minimal_norm_loss'].append(minimal_norm_loss.item())
        return minimal_norm_loss

    def save_losses(self, save_dir):
        torch.save(self.losses, save_dir)

    def __call__(self, xs, u0s, us, φs, δusδts, div_φs):
        temporal_loss = self.get_temporal_loss(δusδts, div_φs)
        tv_loss = self.get_tv_loss(us, div_φs)
        normalization_loss = self.get_normalization_loss(φs)
        inital_loss = self.get_inital_loss(u0s, xs)
        minimal_norm_loss = self.get_minimal_norm_loss(div_φs)

        total_loss = temporal_loss + tv_loss + normalization_loss + inital_loss + minimal_norm_loss
        self.losses['total_loss'].append(total_loss.item())

        return total_loss, temporal_loss, tv_loss, normalization_loss, inital_loss, minimal_norm_loss
