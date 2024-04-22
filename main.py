import os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.datasets import Food101

from paths import *
sys.path.insert(0,CODE_PATH)


from utils.losses_handler import LossesHandler
from utils.training import run_evaluation
from utils.trajectory import trajectory

lr = float(sys.argv[1])
assert 0 < lr, "You need to pass a positive learning rate."
relu = int(sys.argv[2])
assert 0 <= relu, "You need to pass a positive or 0 number of relu activations."
softplus = int(sys.argv[3])
assert 0 <= softplus, "You need to pass a positive or 0 number of softplus activations."
tra = int(sys.argv[4])
assert tra == 0 or tra == 1, "You need to pass 0 or 1 to decide if this is with (1) or without (0) trajectory code."
batch_size = int(sys.argv[5])
assert 0 < batch_size, "You need to pass a positive batch size, for tra = 1 it's the number of time instances, for tra = 0 it's the number of images."
mn_factor = float(sys.argv[6])
assert 0 < mn_factor, "You need to pass a positive minimum norm loss scaling factor."
epochs = int(sys.argv[7])
assert 0 < epochs, "You need to pass a positive number of epochs."
NN_select = int(sys.argv[8])
assert NN_select == 0 or NN_select ==1 or NN_select == 2, "Select 0 for Semi-ResNet, 1 for UNet and 2 for LGD"
use_norm = int(sys.argv[9])
assert use_norm == 0 or use_norm == 1, "Select either True=1 or False=0 for use_norm"
crop_size = int(sys.argv[10])
assert 0 < crop_size, "Select the size at which images are randomly cropped - int for square images"
crop_mixed = int(sys.argv[11])
assert crop_mixed == 0 or crop_mixed == 1, "Select 0 for training on single size and 1 for training on mixed image size"

if NN_select == 0:
    from models.Tensor_model import NN
    u_and_φ_model = NN(ε=1e-4,relu=relu,softplus=softplus)
elif NN_select ==1:
    from models.Unet_model import NN 
    u_and_φ_model = NN(ε=1e-4, scales=4, skip=3,channels=(32, 32, 64, 64, 128),crop_size=crop_size,use_sigmoid=False, use_norm=use_norm) #use_norm was true before
elif NN_select ==2:
    from models.LGD import NN 
    u_and_φ_model = NN(n_steps=5, dim_z=8, ε=1e-4, edge_width=1, crop_size=crop_size, use_norm=use_norm) #use_norm was true before

from models.Tensor_model_wrapper import ModelWrapper

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Model paramteter',count_parameters(u_and_φ_model))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #0
print(device)
print(f'Starting file with lr {lr}, trajectory {tra}, batch_size {batch_size}, and {relu} ReLU activations and {softplus} softplus activations.')
print(f'The minimum norm scaling factor is {mn_factor} and the training is run for {epochs} Epochs.')
print(f'Selected {NN_select}, where 0 is for Semi-ResNet (standard) and 1 for UNet and 2 for LGD')
print(f'Randomly cropping images to shape {crop_size}x{crop_size}, with mixing {crop_mixed}')
if use_norm:
    print('Using normalisation')

#######################################################################
# Load Datasets
#######################################################################
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop((crop_size,crop_size), padding=None, pad_if_needed=True, fill=0, padding_mode='reflect'),
    transforms.ToTensor()
])

dataset_train = Food101(root=DATA_PATH, split='train', download=True, transform=transform)
dataloader = DataLoader(dataset_train, batch_size=(1-tra)*batch_size + tra) #for tra = 0 this is the number of images to load = batch_size, for tra=1 we only need batch_size=1
print('Data is loaded')

#######################################################################
# Prepare saving
#######################################################################
save_dir = f'{CODE_PATH}/Results/ReLU_{relu}-Softplus_{softplus}_tests_16ch_{lr}_trajectory_{tra}_batch-size_{batch_size}_minimum-norm-factor_{mn_factor}_epochs_{epochs}_normalisation_{use_norm}_NN_{NN_select}_STL10_{crop_size}_mixing_{crop_mixed}' #Food101
os.makedirs(save_dir, exist_ok=True)

#######################################################################
# Initialise Model
#######################################################################
print('Initialise model')
model = ModelWrapper(u_and_φ_model, normalize_φ=True, edge_width=1).to(device)
losses_handler = LossesHandler(tv_loss_scaling_factor=1e-4, minimal_norm_loss_scaling_factor=mn_factor , ε=0.00001)

#######################################################################
# Run training for different learning rates
#######################################################################

print('Learning Rate', lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
start_time = time.time()

best_psnr = 0 #initialise best epoch saving
best_epoch = 0
best_time = 0
loss_runner = []

for epoch in range(epochs): 
    print(f'Starting epoch {epoch}.')
    start_time_epoch = time.time()
    model.train()

    epochs_losses = {
        "total": list(),
        "temporal": list(),
        "tv": list(),
        "normalisation": list(),
        "initial": list(),
        "minimal_norm": list()
    }

    for i, (u0, _) in enumerate(dataloader):
        if crop_mixed==1:
            crop_size = np.random.randint(32,256)
            trafo = transforms.RandomCrop((crop_size,crop_size), padding=None, pad_if_needed=True, fill=0, padding_mode='reflect')
            u0 = trafo(u0)
        assert np.allclose(u0.shape, [batch_size, 1, crop_size, crop_size]), f"{u0.shape=}, and loads {(1-tra)*batch_size + tra}"

        u0s,ts = trajectory(tra,u0,device,batch_size,crop_size)

        u0s_, us, φs, δusδts, div_φs = model(u0s, ts)

        total_loss, temporal_loss, tv_loss, normalization_loss, inital_loss, minimal_norm_loss = losses_handler(
            xs=u0s, u0s=u0s_, us=us, φs=φs, δusδts=δusδts, div_φs=div_φs
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        epochs_losses["total"].append(total_loss.item())
        epochs_losses["temporal"].append(temporal_loss.item())
        epochs_losses["tv"].append(tv_loss.item())
        epochs_losses["normalisation"].append(normalization_loss.item())
        epochs_losses["initial"].append(inital_loss.item())
        epochs_losses["minimal_norm"].append(minimal_norm_loss.item())
        

        if i > 625:
            break
    loss_runner.append(np.mean(epochs_losses['total']))
    best_psnr, best_epoch, best_time = run_evaluation(
        model, optimizer, lr, save_dir,
        epoch, start_time_epoch, start_time, epochs_losses, best_psnr, best_epoch, best_time, loss_runner
    )
