import torch
import numpy as np


#######################################################################
# Define function to get zero on the boundaries of the image
#######################################################################
def zero_out_edges(φs, edge_width):
    assert len(φs.shape) == 4, "_zero_out_edges len(shape) condition violated"
    assert φs.shape[1] == 2, "_zero_out_edges shape condition violated"
    φs[:, :,            :edge_width,            :          ] = 0.
    φs[:, :, -edge_width:          ,            :          ] = 0.
    φs[:, :,            :          ,            :edge_width] = 0.
    φs[:, :,            :          , -edge_width:          ] = 0.
    return φs


#######################################################################
# Define function to normalise φs
#######################################################################
def normalize_φs(φs, ε):
    assert not torch.any(torch.isnan(φs)), 'nans'
    norms = torch.sqrt(φs[:, :1] ** 2 + φs[:, 1:] ** 2 + ε ** 2)
    norms_clamped = torch.clamp(norms, 1, np.inf)
    assert torch.all(1 - 1e-3 <= norms_clamped), norms_clamped.min().item()
    φs = φs / norms_clamped
    return φs


#######################################################################
# Define function to count the number of parameters in the model (NN)
#######################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_np(x):
    return x.cpu().detach().numpy().astype(np.float32)