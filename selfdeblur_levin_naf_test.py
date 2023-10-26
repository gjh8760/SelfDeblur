import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import yaml
import torch
import torch.optim
import warnings
from utils.common_utils import *
from copy import deepcopy

from networks.archs import define_network

from metrics.psnr_ssim import calculate_psnr


def imread(path):
    return np.asarray(Image.open(path))


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

# gt directory
gt_dir = opt['gt_dir']
gt_imgs = {f'im{i+1}': imread(osp.join(gt_dir, f'im{i+1}.png')) for i in range(4)}

# test directories
test_dirs = [osp.join('results', d) for d in opt['test_dirs']]

# Start test
for test_dir in test_dirs:
    
    print('test dir:', test_dir)

    mean_dict = {}
    mean_dict['total'] = []
    for i in range(4):
        mean_dict[f'im{i+1}'] = []
    for k in range(8):
        mean_dict[f'kernel{k+1}'] = []

    test_img_paths = sorted([osp.join(test_dir, d) for d in os.listdir(test_dir) if 'x.png' in d])
    for test_img_path in test_img_paths:
        i = osp.basename(test_img_path)[2]
        k = osp.basename(test_img_path)[10]
        test_img = imread(test_img_path)
        gt_img = gt_imgs[f'im{i}']
        psnr = calculate_psnr(test_img, gt_img, crop_border=0)

        print(osp.basename(test_img_path), ': %.3f' % psnr)
        mean_dict['total'].append(psnr)
        mean_dict[f'im{i}'].append(psnr)
        mean_dict[f'kernel{k}'].append(psnr)

    print('-'*50)

    for i in range(4):
        print(f'im{i+1} mean :', '%.3f' % np.mean(mean_dict[f'im{i+1}']))
    for k in range(8):
        print(f'kernel{k+1} mean :', '%.3f' % np.mean(mean_dict[f'kernel{k+1}']))
    print('total mean :', '%.3f' % np.mean(mean_dict['total']))
    print('='*50)
    

