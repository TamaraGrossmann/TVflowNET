import time, glob
import json, os
import torch
import numpy as np

from paths import *

from utils.misc import get_np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def load_validation_dataset(n_time_points, device):   
    u0s = []
    GT_u = []
    GT_φ = []
    for i in range(200):#10
        ys = torch.load(os.path.join(GT_PATH,f'{i}_50.pt')) 
        GT_u.append(get_np(ys[:,:1]))
        GT_φ.append(get_np(ys[:,1:]))
        u0s.append(torch.cat([torch.tensor(GT_u[i][0][None]).to(device)] * n_time_points, dim=0))

    return u0s, GT_u, GT_φ

def get_psnrs_dataset(model, n_time_points):
    device = next(model.parameters()).device

    with torch.no_grad():
        u0s, GT_u, GT_φ = load_validation_dataset(
            n_time_points=n_time_points,
            device=device
        )

        ts = torch.linspace(0, 1, n_time_points).to(device).view(-1, 1)
        assert len(ts) == len(u0s[0]), f"{ts.shape=}, {u0s.shape=}"
        assert ts.shape[1] == 1, f"{ts.shape=}"
        assert np.allclose(u0s[0].shape[1:], [1, 96, 96]), f"{u0s.shape=}"

        psnr_us = []
        psnr_φs = []
        ssim_us = []
        ssim_φs = []
        model.eval()
        for i in range(len(u0s)): #10 len(u0s)
            _, us, φs, _, _= model(u0s[i], ts)
            psnr_us.append(peak_signal_noise_ratio(get_np(us),GT_u[i],data_range=5))
            psnr_φs.append(peak_signal_noise_ratio(get_np(φs),GT_φ[i],data_range=5))
            for j in range(us.shape[0]):
                ssim_us.append(structural_similarity(get_np(us)[j].squeeze(), GT_u[i][j].squeeze(), data_range=5))
                ssim_φs.append(structural_similarity(get_np(φs)[j,0], GT_φ[i][j,0], data_range=5))
                ssim_φs.append(structural_similarity(get_np(φs)[j,1], GT_φ[i][j,1], data_range=5))
        model.train()
    psnr_us_tot = np.mean(psnr_us)
    psnr_φs_tot = np.mean(psnr_φs)
    ssim_us_tot = np.mean(ssim_us)
    ssim_φs_tot = np.mean(ssim_φs)

    return psnr_us_tot, psnr_φs_tot, ssim_us_tot, ssim_φs_tot


def run_evaluation(
        model, optimizer, lr, save_dir,
        epoch, start_time_epoch, start_time, epochs_losses, best_psnr, best_epoch, best_time, loss_runner
):
    total_time = (time.time() - start_time)
    epoch_time = (time.time() - start_time_epoch)
    epoch_42 = []
    results = dict()
    results['Mean total training loss'] = loss_runner
    results['Mean temporal training loss'] = np.mean(epochs_losses["temporal"])
    results['Mean TV training loss'] = np.mean(epochs_losses["tv"])
    results['Mean normalisation training loss'] = np.mean(epochs_losses["normalisation"])
    results['Mean initial training loss'] = np.mean(epochs_losses["initial"])
    results['Mean minimal norm training loss'] = np.mean(epochs_losses["minimal_norm"])

    psnr_us, psnr_φs, ssim_us, ssim_φs = get_psnrs_dataset(model, 50)
    results['PSNR us'] = psnr_us
    results['PSNR φs'] = psnr_φs
    results['SSIM us'] = ssim_us
    results['SSIM φs'] = ssim_φs   

    results['Epoch'] = epoch
    results['Epoch Time'] = epoch_time
    results['Total Time'] = total_time

    if psnr_us >= 42:
        epoch_42.append(epoch)
        results['Epoch at which 42 is reached'] = epoch_42


    if psnr_us > best_psnr:
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            f"{save_dir}/model_lr_{lr}_best_psnr.pth"
        )
        best_psnr = psnr_us
        best_epoch = epoch
        best_time = total_time
        print('Better PSNR')

    results['best epoch'] = best_epoch
    results['best total time'] = best_time
    with open(f"{save_dir}/results_lr_{lr}_best_psnr.json", "w") as f:
            json.dump(results, f)
    print(f"Epoch {epoch} took {(time.time() - start_time_epoch)//60} minutes, Total time: {(time.time() - start_time)//60}")#
    print(json.dumps(results, indent=4))
    return best_psnr, best_epoch, best_time
    
