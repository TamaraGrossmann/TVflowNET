import torch
import numpy as np


#######################################################################
# Define gradient operators
#######################################################################
def du_dx(us, h=1., zero_out=True):
    assert us.shape[-2] > 3

    ux = torch.zeros_like(us)
    ux[:, :,  0:-1, :] = (us[:, :, 1:, :] - us[:, :, 0:-1, :]) / h
    ux[:, :,    -1, :] = 0

    if zero_out:
        ux[:, :,  0, :] = 0.
        ux[:, :, -1, :] = 0.

    return ux


def du_dy(us, h=1., zero_out=True):
    assert us.shape[-1] > 3

    uy = torch.zeros_like(us)
    uy[:, :, :,  0:-1] = (us[:, :, :, 1:] - us[:, :, :, 0:-1]) / h
    uy[:, :, :,    -1] = 0

    if zero_out:
        uy[:, :, :,  0] = 0.
        uy[:, :, :, -1] = 0.

    return uy


def du_dx_adj(u, h=1., zero_out=True):
    assert len(u.shape) == 4
    assert u.shape[2] > 2

    if zero_out:
        u = u.clone()
        u[:, :,  0, :] = 0.
        u[:, :, -1, :] = 0.

    du = torch.zeros_like(u)
    du[:, :, 1:-1, :] = (u[:, :, 1:-1, :] - u[:, :, 0:-2, :]) / h
    du[:, :,    0, :] =  u[:, : ,   0, :]
    du[:, :,   -1, :] = -u[:, :,   -2, :]

    return du


def du_dy_adj(u, h=1., zero_out=True):
    assert len(u.shape) == 4
    assert u.shape[3] > 2

    if zero_out:
        u = u.clone()
        u[:, :, :, 0] = 0.
        u[:, :, :, -1] = 0.

    du = torch.zeros_like(u)
    du[:, :, :, 1:-1] = (u[:, :, :, 1:-1] - u[:, :, :, 0:-2]) / h
    du[:, :, :,    0] =  u[:, :, :,    0]
    du[:, :, :,   -1] = -u[:, :, :,   -2]

    return du


#######################################################################
# Define divergence and total variation operators
#######################################################################

def div(us, h=1., zero_out=True):
    # divergence is adjoint to the gradient operator, i.e. we use the adjoint gradient to calculate the divergence
    assert len(us.shape) == 4, us.shape
    assert us.shape[1] == 2, us.shape

    δu0δx = du_dx_adj(us[:, :1], h=h, zero_out=zero_out)
    δu1δy = du_dy_adj(us[:, 1:], h=h, zero_out=zero_out)
    div = δu0δx + δu1δy

    assert div.shape[1] == 1
    return div


def TV(u, h=1., ε=0, zero_out=True, use_mean=False):
    assert len(u.shape) == 4, u.shape
    assert u.shape[1] == 1, u.shape

    δuδx = du_dx(u, h=h, zero_out=zero_out)
    δuδy = du_dy(u, h=h, zero_out=zero_out)
    norms = torch.sqrt(δuδx ** 2 + δuδy ** 2 + ε ** 2)

    tv_value = norms.view(norms.shape[0], -1)
    tv_value = tv_value.mean(-1) if use_mean else tv_value.sum(-1)

    assert len(tv_value.shape) == 1, tv_value.shape
    assert tv_value.shape[0] == u.shape[0], tv_value.shape
    assert torch.all(tv_value >= 0), tv_value
    assert torch.all(tv_value < np.inf), 'TV inf'
    return tv_value
