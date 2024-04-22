import torch
import numpy as np

def trajectory(tra,u0,device,batch_size,crop_size):
    if tra == 1:
        assert np.allclose(u0.shape, [1, 1, crop_size, crop_size]), f"{u0.shape=}"
        u0 = 5*u0.to(device) # *5 for comparison with Gilboa Code, contrast change influences the evolution time

        ts = torch.rand(batch_size-1, device=device).view(-1, 1)
        ts = torch.cat([torch.tensor([0.], device=device)[None],ts],dim=0)
        idx = torch.randperm(ts.nelement())
        ts = ts.view(-1)[idx].view(ts.size()) #Shuffle time

        u0s = torch.cat([u0] * len(ts), dim=0)
        assert np.allclose(u0s.shape, [batch_size, 1, crop_size, crop_size]), f"{u0s.shape=}"

    elif tra == 0:
        assert np.allclose(u0.shape, [batch_size, 1, crop_size, crop_size]), f"{u0.shape=}"
        u0s = 5*u0.to(device) 
        ts = torch.rand(batch_size,device=device).view(-1,1)

    else:
        print('Error in trajectory, either 1 (yes) or 0 (no).')

    return u0s, ts
