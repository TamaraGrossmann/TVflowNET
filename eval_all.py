import torch, os, json, glob, sys
from torchvision import transforms
import scipy.io as sio
import numpy as np
from torchvision.datasets import Food101
from timeit import default_timer as timer

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from paths import *
sys.path.insert(0,CODE_PATH)

from utils.misc import get_np
from models.Tensor_model_wrapper_eval_phi2 import ModelWrapper
# from utils.losses_handler import LossesHandler

from skimage import io

device = 0

#######################################################################
# Load Model
#######################################################################
model_locations = '/enter/model/location/here'


#######################################################################
# Declare testing set directories
#######################################################################
Food101_32 ='/enter/test/directory/here'
Food101_96 ='/enter/test/directory/here'
Food101_256 = '/enter/test/directory/here'

n_time_points = 50

#######################################################################
# Prepare saving
#######################################################################
save_dir = f'./Results/Evaluation/'
os.makedirs(save_dir, exist_ok=True)

#######################################################################
# Warm-up GPU
#######################################################################
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
dataset_test  = Food101(root=DATA_PATH, split='test',  download=True, transform=transform)

from models.Tensor_model import NN
u_and_φ_model = NN(ε=1e-4,relu=1,softplus=6)
model = ModelWrapper(u_and_φ_model, normalize_φ=True, edge_width=1).to(device)
checkpoint = torch.load('/enter/test/location/here/model_lr_0.0003_best_psnr.pth')
model.load_state_dict(checkpoint['state_dict'])
with torch.no_grad():
    model.eval()
    ts = torch.linspace(0,1,2).to(device).view(-1, 1)
    x = dataset_test.__getitem__(0)[0].to(device)
    x = 5*x.view(1, *x.shape)
    u0s = torch.cat([x] * len(ts), dim=0)
    us = model(u0s, ts)
    torch.cuda.synchronize()



#######################################################################
# Define Dataset Loader
#######################################################################
def load_test_dataset(img_dir,GT_dir,n_time_points,device,GT_format='Gilboa-GPU'):   
    u0s = []
    GT_u = []
    GT_φ = []
    if GT_format=='Joint_space-time':
        for i in range(len(img_dir)):
            ys = torch.load(GT_dir[i],map_location='cuda:0')
            GT_u.append(get_np(ys[:,:1])/5)
            GT_φ.append(get_np(ys[:,1:])/5)
            img = io.imread(img_dir[i], as_gray=True) 
            img = np.asarray(img)#/255
            u0s.append(torch.cat([torch.tensor(img[None], dtype=torch.float64).type(torch.FloatTensor)]*n_time_points,dim=0))

    elif GT_format=='Gilboa-GPU':
        for i in range(len(img_dir)):
            ys = np.load(GT_dir[i])
            GT_u.append(ys[:,None])
            img = io.imread(img_dir[i], as_gray=True) 
            img = np.asarray(img)#/255
            u0s.append(torch.cat([torch.tensor(img[None], dtype=torch.float64).type(torch.FloatTensor)]*n_time_points,dim=0))

    elif GT_format=='Gilboa-CPU':
        for i in range(len(img_dir)):
            ys = sio.loadmat(GT_dir[i])
            ys = ys['u']
            ys = torch.permute(torch.tensor(ys),(-1,0,1)).to(device)
            GT_u.append(get_np(ys[:,None]))
            img = io.imread(img_dir[i], as_gray=True) 
            img = np.asarray(img)#/255
            u0s.append(torch.cat([torch.tensor(img[None]).type(torch.FloatTensor)]*n_time_points,dim=0))
    
    else:
        print('Error, select as GT_format either Joint_space-time, Gilboa-GPU or Gilboa-CPU')

    return u0s, GT_u, GT_φ

#######################################################################
# Define PSNR and SSIM evaluation function
#######################################################################
def get_psnrs_dataset(model, n_time_points, img_dir, GT_dir_joint, GT_dir_GPU, GT_dir_CPU,split_size,all_GTs):
    device = next(model.parameters()).device

    results = dict({})
    results['Joint_space-time'], results['Gilboa-GPU'], results['Gilboa-CPU'] = dict({}),dict({}),dict({})

    with torch.no_grad():
        model.eval()

        u0s, GT_u_gpu, _ = load_test_dataset(
            img_dir=img_dir,
            GT_dir=GT_dir_GPU,
            n_time_points=n_time_points,
            device=device,
            GT_format='Gilboa-GPU'
        )
        ts = torch.linspace(0, 0.98, n_time_points).to(device).view(-1, 1)
        assert len(ts) == len(u0s[0]), f"{ts.shape}, {u0s.shape}"
        assert ts.shape[1] == 1, f"{ts.shape}"

        if all_GTs==1:
            _,  GT_u_joint, GT_φ_joint = load_test_dataset(
                img_dir=img_dir,
                GT_dir=GT_dir_joint,
                n_time_points=n_time_points,
                device=device,
                GT_format='Joint_space-time'
            )

            _, GT_u_cpu, _ = load_test_dataset(
                img_dir=img_dir,
                GT_dir=GT_dir_CPU,
                n_time_points=n_time_points,
                device=device,
                GT_format='Gilboa-CPU'
            )

        psnr_us_J,psnr_φs_J,ssim_us_J,ssim_φs_J = [],[],[],[]
        psnr_us_G,ssim_us_G = [],[]
        psnr_us_C,ssim_us_C = [],[]
        

        for i in range(len(u0s)):
            if split_size==1:
                us, φs= model(5*u0s[i][:,None].to(device), ts.to(device)) 
                us = get_np(us)/5
                φs = get_np(φs)/5
            elif split_size==2:
                us, φs= model(5*u0s[i][0:int(n_time_points/2),None].to(device), ts[0:int(n_time_points/2)].to(device)) 
                us1, φs1= model(5*u0s[i][int(n_time_points/2):,None].to(device), ts[int(n_time_points/2):].to(device)) #for disk images
                us = get_np(torch.cat((us,us1),dim=0))/5
                φs = get_np(torch.cat((φs,φs1),dim=0))/5
            elif split_size==3:
                us, φs= model(5*u0s[i][0:int(n_time_points/3),None].to(device), ts[0:int(n_time_points/3)].to(device)) 
                us1, φs1= model(5*u0s[i][int(n_time_points/3):int(2*n_time_points/3),None].to(device), ts[int(n_time_points/3):int(2*n_time_points/3)].to(device)) #for disk images
                us2, φs2= model(5*u0s[i][int(2*n_time_points/3):,None].to(device), ts[int(2*n_time_points/3):].to(device))

                us = get_np(torch.cat((us,us1,us2),dim=0))/5
                φs = get_np(torch.cat((φs,φs1,φs2),dim=0))/5

            psnr_us_G.append(peak_signal_noise_ratio(us.squeeze(),GT_u_gpu[i].squeeze(),data_range=1))
            
            if all_GTs==1:
                psnr_us_J.append(peak_signal_noise_ratio(us.squeeze(),GT_u_joint[i].squeeze(),data_range=1))
                psnr_φs_J.append(peak_signal_noise_ratio(φs.squeeze(),GT_φ_joint[i].squeeze(),data_range=1))
                psnr_us_C.append(peak_signal_noise_ratio(us.squeeze(),GT_u_cpu[i].squeeze(),data_range=1))

            for j in range(us.shape[0]):
                ssim_us_G.append(structural_similarity(us[j].squeeze(), GT_u_gpu[i][j].squeeze(), data_range=1))
                if all_GTs==1:
                    ssim_us_J.append(structural_similarity(us[j].squeeze(), GT_u_joint[i][j].squeeze(), data_range=1))
                    ssim_us_C.append(structural_similarity(us[j].squeeze(), GT_u_cpu[i][j].squeeze(), data_range=1))
                    ssim_φs_J.append(structural_similarity(φs[j,0], GT_φ_joint[i][j,0], data_range=1))
                    ssim_φs_J.append(structural_similarity(φs[j,1], GT_φ_joint[i][j,1], data_range=1))

    results['Joint_space-time']['psnr_us_mean'] = np.mean(psnr_us_J)
    results['Joint_space-time']['ssim_us_mean'] = np.mean(ssim_us_J)
    results['Joint_space-time']['psnr_us_std'] = np.std(psnr_us_J)
    results['Joint_space-time']['ssim_us_std'] = np.std(ssim_us_J)
    results['Joint_space-time']['psnr_phis_mean'] = np.mean(psnr_φs_J)
    results['Joint_space-time']['ssim_phis_mean'] = np.mean(ssim_φs_J)
    results['Joint_space-time']['psnr_phis_std'] = np.std(psnr_φs_J)
    results['Joint_space-time']['ssim_phis_std'] = np.std(ssim_φs_J)

    results['Gilboa-GPU']['psnr_us_mean'] = np.mean(psnr_us_G)
    results['Gilboa-GPU']['ssim_us_mean'] = np.mean(ssim_us_G)
    results['Gilboa-GPU']['psnr_us_std'] = np.std(psnr_us_G)
    results['Gilboa-GPU']['ssim_us_std'] = np.std(ssim_us_G)

    results['Gilboa-CPU']['psnr_us_mean'] = np.mean(psnr_us_C)
    results['Gilboa-CPU']['ssim_us_mean'] = np.mean(ssim_us_C)
    results['Gilboa-CPU']['psnr_us_std'] = np.std(psnr_us_C)
    results['Gilboa-CPU']['ssim_us_std'] = np.std(ssim_us_C)

    return results

#######################################################################
# Define run evaluation function
#######################################################################
def run_evaluation(img_dirs,joint_dir,model,n_time_points,split_size,all_GTs):
    img_dir = sorted(glob.glob(os.path.join(img_dirs,'*.png')))
    #'Joint_space-time_Food101-256px_2000ep_5e-3lr'
    GT_dir_GPU = sorted(glob.glob(os.path.join(img_dirs,'Gilboa-GPU/*_50.npy')))
    if os.path.exists(os.path.join(img_dirs,joint_dir)):
        GT_dir_joint = sorted(glob.glob(os.path.join(img_dirs,os.path.join(joint_dir,'*_50.pt'))))
        GT_dir_CPU = sorted(glob.glob(os.path.join(img_dirs,'Gilboa-CPU/*_50.mat')))
    else: 
        GT_dir_joint = []
        GT_dir_CPU = []
    results = get_psnrs_dataset(model, n_time_points, img_dir, GT_dir_joint, GT_dir_GPU, GT_dir_CPU, split_size,all_GTs)

    return results

#######################################################################
# Define run evaluation function over dataset for all NN
#######################################################################
def run_time_datasets(current_dir,joint_dir,n_time_points,split_size,all_GTs):
    results = dict({})
    results['free parameters'] = []
    results['Joint_space-time'], results['Gilboa-GPU'], results['Gilboa-CPU'] = dict({}),dict({}),dict({})
    results['Joint_space-time']['psnr us mean'], results['Joint_space-time']['psnr us std'], results['Joint_space-time']['ssim us mean'], results['Joint_space-time']['ssim us std'] = [],[],[],[]
    results['Joint_space-time']['psnr phis mean'], results['Joint_space-time']['psnr phis std'], results['Joint_space-time']['ssim phis mean'], results['Joint_space-time']['ssim phis std'] = [],[],[],[]
    results['Gilboa-GPU']['psnr us mean'], results['Gilboa-GPU']['psnr us std'], results['Gilboa-GPU']['ssim us mean'], results['Gilboa-GPU']['ssim us std'] = [],[],[],[]
    results['Gilboa-CPU']['psnr us mean'], results['Gilboa-CPU']['psnr us std'], results['Gilboa-CPU']['ssim us mean'], results['Gilboa-CPU']['ssim us std'] = [],[],[],[]
    
    # Semi-ResNet
    for i in range(0,4):
        from models.Tensor_model import NN
        u_and_φ_model = NN(ε=1e-4,relu=1,softplus=6)
        model = ModelWrapper(u_and_φ_model, normalize_φ=True, edge_width=1).to(device)
        checkpoint = torch.load(model_locations[i])
        model.load_state_dict(checkpoint['state_dict'])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = run_evaluation(current_dir,joint_dir,model,n_time_points,split_size,all_GTs)
        results['Joint_space-time']['psnr us mean'].append(data['Joint_space-time']['psnr_us_mean'])
        results['Joint_space-time']['psnr us std'].append(data['Joint_space-time']['psnr_us_std'])
        results['Joint_space-time']['ssim us mean'].append(data['Joint_space-time']['ssim_us_mean'])
        results['Joint_space-time']['ssim us std'].append(data['Joint_space-time']['ssim_us_std'])
        results['Joint_space-time']['psnr phis mean'].append(data['Joint_space-time']['psnr_phis_mean'])
        results['Joint_space-time']['psnr phis std'].append(data['Joint_space-time']['psnr_phis_std'])
        results['Joint_space-time']['ssim phis mean'].append(data['Joint_space-time']['ssim_phis_mean'])
        results['Joint_space-time']['ssim phis std'].append(data['Joint_space-time']['ssim_phis_std'])

        results['Gilboa-GPU']['psnr us mean'].append(data['Gilboa-GPU']['psnr_us_mean'])
        results['Gilboa-GPU']['psnr us std'].append(data['Gilboa-GPU']['psnr_us_std'])
        results['Gilboa-GPU']['ssim us mean'].append(data['Gilboa-GPU']['ssim_us_mean'])
        results['Gilboa-GPU']['ssim us std'].append(data['Gilboa-GPU']['ssim_us_std'])

        results['Gilboa-CPU']['psnr us mean'].append(data['Gilboa-CPU']['psnr_us_mean'])
        results['Gilboa-CPU']['psnr us std'].append(data['Gilboa-CPU']['psnr_us_std'])
        results['Gilboa-CPU']['ssim us mean'].append(data['Gilboa-CPU']['ssim_us_mean'])
        results['Gilboa-CPU']['ssim us std'].append(data['Gilboa-CPU']['ssim_us_std'])

        results['free parameters'].append(pytorch_total_params)

    # UNet
    for i in range(4,8):
        crop_size=2
        from models.Unet_model import NN 
        u_and_φ_model = NN(ε=1e-4, scales=4, skip=3,channels=(32, 32, 64, 64, 128),crop_size=crop_size,use_sigmoid=False, use_norm=0)
        model = ModelWrapper(u_and_φ_model, normalize_φ=True, edge_width=1).to(device)
        checkpoint = torch.load(model_locations[i])
        model.load_state_dict(checkpoint['state_dict'])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = run_evaluation(current_dir,joint_dir,model,n_time_points,split_size,all_GTs)
        results['Joint_space-time']['psnr us mean'].append(data['Joint_space-time']['psnr_us_mean'])
        results['Joint_space-time']['psnr us std'].append(data['Joint_space-time']['psnr_us_std'])
        results['Joint_space-time']['ssim us mean'].append(data['Joint_space-time']['ssim_us_mean'])
        results['Joint_space-time']['ssim us std'].append(data['Joint_space-time']['ssim_us_std'])
        results['Joint_space-time']['psnr phis mean'].append(data['Joint_space-time']['psnr_phis_mean'])
        results['Joint_space-time']['psnr phis std'].append(data['Joint_space-time']['psnr_phis_std'])
        results['Joint_space-time']['ssim phis mean'].append(data['Joint_space-time']['ssim_phis_mean'])
        results['Joint_space-time']['ssim phis std'].append(data['Joint_space-time']['ssim_phis_std'])

        results['Gilboa-GPU']['psnr us mean'].append(data['Gilboa-GPU']['psnr_us_mean'])
        results['Gilboa-GPU']['psnr us std'].append(data['Gilboa-GPU']['psnr_us_std'])
        results['Gilboa-GPU']['ssim us mean'].append(data['Gilboa-GPU']['ssim_us_mean'])
        results['Gilboa-GPU']['ssim us std'].append(data['Gilboa-GPU']['ssim_us_std'])

        results['Gilboa-CPU']['psnr us mean'].append(data['Gilboa-CPU']['psnr_us_mean'])
        results['Gilboa-CPU']['psnr us std'].append(data['Gilboa-CPU']['psnr_us_std'])
        results['Gilboa-CPU']['ssim us mean'].append(data['Gilboa-CPU']['ssim_us_mean'])
        results['Gilboa-CPU']['ssim us std'].append(data['Gilboa-CPU']['ssim_us_std'])
        results['free parameters'].append(pytorch_total_params)

    # LGD
    for i in range(8,12):
        crop_size=2 
        from models.LGD import NN 
        u_and_φ_model = NN(n_steps=5, dim_z=8, ε=1e-4, edge_width=1, crop_size=crop_size, use_norm=0) 
        model = ModelWrapper(u_and_φ_model, normalize_φ=True, edge_width=1).to(device)
        checkpoint = torch.load(model_locations[i])
        model.load_state_dict(checkpoint['state_dict'])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = run_evaluation(current_dir,joint_dir,model,n_time_points,split_size,all_GTs)
        results['Joint_space-time']['psnr us mean'].append(data['Joint_space-time']['psnr_us_mean'])
        results['Joint_space-time']['psnr us std'].append(data['Joint_space-time']['psnr_us_std'])
        results['Joint_space-time']['ssim us mean'].append(data['Joint_space-time']['ssim_us_mean'])
        results['Joint_space-time']['ssim us std'].append(data['Joint_space-time']['ssim_us_std'])
        results['Joint_space-time']['psnr phis mean'].append(data['Joint_space-time']['psnr_phis_mean'])
        results['Joint_space-time']['psnr phis std'].append(data['Joint_space-time']['psnr_phis_std'])
        results['Joint_space-time']['ssim phis mean'].append(data['Joint_space-time']['ssim_phis_mean'])
        results['Joint_space-time']['ssim phis std'].append(data['Joint_space-time']['ssim_phis_std'])

        results['Gilboa-GPU']['psnr us mean'].append(data['Gilboa-GPU']['psnr_us_mean'])
        results['Gilboa-GPU']['psnr us std'].append(data['Gilboa-GPU']['psnr_us_std'])
        results['Gilboa-GPU']['ssim us mean'].append(data['Gilboa-GPU']['ssim_us_mean'])
        results['Gilboa-GPU']['ssim us std'].append(data['Gilboa-GPU']['ssim_us_std'])

        results['Gilboa-CPU']['psnr us mean'].append(data['Gilboa-CPU']['psnr_us_mean'])
        results['Gilboa-CPU']['psnr us std'].append(data['Gilboa-CPU']['psnr_us_std'])
        results['Gilboa-CPU']['ssim us mean'].append(data['Gilboa-CPU']['ssim_us_mean'])
        results['Gilboa-CPU']['ssim us std'].append(data['Gilboa-CPU']['ssim_us_std'])
        results['free parameters'].append(pytorch_total_params)
    return results

#######################################################################
# Run evaluation
#######################################################################
current_dir = Food101_96
results = dict({})
results['joint space-time'], results['Gilboa GPU'], results['Gilboa CPU'],results['free parameters'] = dict({}),dict({}),dict({}), dict({})

data = run_time_datasets(Food101_32,'Joint_space-time_Food101-32px_2000ep_5e-3lr',n_time_points,1,1)
results['joint space-time']['Food101 32'] = data['Joint_space-time']
results['Gilboa GPU']['Food101 32'] = data['Gilboa-GPU']
results['Gilboa CPU']['Food101 32'] = data['Gilboa-CPU']
results['free parameters'] = data['free parameters']
with open(f"{save_dir}/Eval_all_NN_all_data_rerun_test.json", "w") as f:
    json.dump(results, f)
print(json.dumps(results, indent=4))

data = run_time_datasets(Food101_96,'Joint_space-time_Food101-96px_2000ep_5e-3lr',n_time_points,1,1)
results['joint space-time']['Food101 96'] = data['Joint_space-time']
results['Gilboa GPU']['Food101 96'] = data['Gilboa-GPU']
results['Gilboa CPU']['Food101 96'] = data['Gilboa-CPU']
with open(f"{save_dir}/Eval_all_NN_all_data.json", "w") as f:
    json.dump(results, f)
print(json.dumps(results, indent=4))

data = run_time_datasets(Food101_256,'Joint_space-time_Food101-256px_2000ep_5e-3lr',n_time_points,1,1)
results['joint space-time']['Food101 256'] = data['Joint_space-time']
results['Gilboa GPU']['Food101 256'] = data['Gilboa-GPU']
results['Gilboa CPU']['Food101 256'] = data['Gilboa-CPU']
with open(f"{save_dir}/Eval_all_NN_all_data.json", "w") as f:
    json.dump(results, f)
print(json.dumps(results, indent=4))







