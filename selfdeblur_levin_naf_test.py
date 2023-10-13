import numpy as np
import os
import yaml
import argparse

from metrics.psnr_ssim import calculate_psnr, calculate_ssim


def str2bool(s):
    return True if s.lower() == 'true' else False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test.yml', help='config yml filename')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    args.config = os.path.join('config/test', args.config)
    with open(args.config) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    test_dirs = opt['test_dirs']
    gt_dir = opt['gt_dir']

    for test_dir in test_dirs:

        test_path = os.path.join('results/levin', test_dir)

    



    pass


if __name__ == '__main__':
    main()