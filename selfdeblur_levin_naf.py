import argparse
import os
import numpy as np
import cv2
import yaml
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
from copy import deepcopy

from networks.optims import setup_optimizers

from networks.archs import define_network

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='naf.yml', help='path to config yml file')
args = parser.parse_args()

# load config yml file
args.config = f'config/{args.config}'
with open(args.config) as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)

# visible gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['gpu'])
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

# data path, save path setting
opt['data_path'] = 'datasets/%s' % (opt['dataset'])
opt['save_path'] = 'results/%s/%s' % (opt['dataset'], opt['name'])

files_source = glob.glob(os.path.join(opt['data_path'], '*.png'))
files_source.sort()
save_path = opt['save_path']
os.makedirs(save_path, exist_ok=True)


# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'
    num_iter = opt['num_iter']
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('kernel1') != -1:
        opt['kernel_size'] = [17, 17]
    if imgname.find('kernel2') != -1:
        opt['kernel_size'] = [15, 15]
    if imgname.find('kernel3') != -1:
        opt['kernel_size'] = [13, 13]
    if imgname.find('kernel4') != -1:
        opt['kernel_size'] = [27, 27]
    if imgname.find('kernel5') != -1:
        opt['kernel_size'] = [11, 11]
    if imgname.find('kernel6') != -1:
        opt['kernel_size'] = [19, 19]
    if imgname.find('kernel7') != -1:
        opt['kernel_size'] = [21, 21]
    if imgname.find('kernel8') != -1:
        opt['kernel_size'] = [21, 21]

    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape
    print(imgname)
    # ######################################################################
    padh, padw = opt['kernel_size'][0]-1, opt['kernel_size'][1]-1
    opt['img_size'][0], opt['img_size'][1] = img_size[1]+padh, img_size[2]+padw

    '''
    x_net:
    '''
    input_depth = opt['network_x']['in_channel']

    net_input = get_noise(input_depth, INPUT, (opt['img_size'][0], opt['img_size'][1])).type(dtype)

    net = define_network(deepcopy(opt['network_x']))
    net = net.type(dtype)

    '''
    k_net:
    '''
    n_k = opt['network_k']['num_input_channels']
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    opt['network_k']['num_output_channels'] = opt['kernel_size'][0] * opt['kernel_size'][1]
    net_kernel = define_network(deepcopy(opt['network_k']))
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optim_params = [net.parameters(), net_kernel.parameters()]
    optimizer = setup_optimizers(optim_params, deepcopy(opt['train']['optim']))
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    ### start SelfDeblur
    img_frame_array = []
    ker_frame_array = []
    fps = 10
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)
    
        out_k_m = out_k.view(-1,1,opt['kernel_size'][0],opt['kernel_size'][1])
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        if step < 1000:     # TODO: 여기도 튜닝해야 함
            total_loss = mse(out_y,y) 
        else:
            total_loss = 1-ssim(out_y, y)

        total_loss.backward()
        optimizer.step()

        if (step+1) % opt['save_frequency'] == 0:

            save_path = os.path.join(opt['save_path'], '%s_x.png'%imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            out_x_np = np.clip(out_x_np * 255, 0, 255).astype(np.uint8)
            imsave(save_path, out_x_np)

            save_path = os.path.join(opt['save_path'], '%s_k.png'%imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            out_k_np = np.clip(out_k_np * 255, 0, 255).astype(np.uint8)
            imsave(save_path, out_k_np)

            torch.save(net, os.path.join(opt['save_path'], "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt['save_path'], "%s_knet.pth" % imgname))

            # Save video frames for x and k
            img_frame_array.append(out_x_np)
            ker_frame_array.append(out_k_np)
    
    # Save video
    out_img = cv2.VideoWriter(
        filename=os.path.join(opt['save_path'], '%s.avi' % imgname),
        fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
        fps=fps, 
        frameSize=(765, 765),
        isColor=False
    )

    for i in range(len(img_frame_array)):
        img = cv2.resize(img_frame_array[i], dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        ker = cv2.resize(ker_frame_array[i], dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        h, w = ker.shape
        img[:h+5, :w+5] = 255
        img[:h, :w] = ker
        out_img.write(img)
    out_img.release()
