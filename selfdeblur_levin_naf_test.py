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

from statistic.statistic_levin import comp_upto_shift


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

    mean_dict = {'total': {'psnr': [], 'ssim': []}}
    for i in range(4):
        mean_dict[f'im{i+1}'] = {'psnr': [], 'ssim': []}
    for k in range(8):
        mean_dict[f'kernel{k+1}'] = {'psnr': [], 'ssim': []}

    test_img_paths = sorted([osp.join(test_dir, d) for d in os.listdir(test_dir) if 'x.png' in d])
    for test_img_path in test_img_paths:
        i = osp.basename(test_img_path)[2]
        k = osp.basename(test_img_path)[10]
        test_img = imread(test_img_path)
        gt_img = gt_imgs[f'im{i}']
        
        # Get psnr and ssim from optimally shifted image
        psnr, ssim, test_img_shift = comp_upto_shift(test_img, gt_img, maxshift=5)

        print(osp.basename(test_img_path), '| PSNR: %.3f | SSIM: %.4f' % (psnr, ssim))
        mean_dict['total']['psnr'].append(psnr)
        mean_dict['total']['ssim'].append(ssim)
        mean_dict[f'im{i}']['psnr'].append(psnr)
        mean_dict[f'im{i}']['ssim'].append(ssim)
        mean_dict[f'kernel{k}']['psnr'].append(psnr)
        mean_dict[f'kernel{k}']['ssim'].append(ssim)

    print('-'*50)

    for i in range(4):
        psnr_mean = np.mean(mean_dict[f'im{i+1}']['psnr'])
        ssim_mean = np.mean(mean_dict[f'im{i+1}']['ssim'])
        print(f'im{i+1} mean', '| PSNR: %.3f | SSIM: %.4f' % (psnr_mean, ssim_mean))
    for k in range(8):
        psnr_mean = np.mean(mean_dict[f'kernel{k+1}']['psnr'])
        ssim_mean = np.mean(mean_dict[f'kernel{k+1}']['ssim'])
        print(f'kernel{k+1} mean', '| PSNR: %.3f | SSIM: %.4f' % (psnr_mean, ssim_mean))
    
    total_psnr_mean = np.mean(mean_dict['total']['psnr'])
    total_ssim_mean = np.mean(mean_dict['total']['ssim'])
    print('total mean', '| PSNR: %.3f | SSIM: %.4f' % (total_psnr_mean, total_ssim_mean))
    print('='*50)
    

