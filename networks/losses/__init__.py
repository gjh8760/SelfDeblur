import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, x):
        x = x.squeeze()
        h_x = x.size()[0]
        w_x = x.size()[1]
        count_h = self._tensor_size(x[:, 1:])
        count_w = self._tensor_size(x[1:, :])
        w_tv = torch.pow((x[:, 1:]-x[:, :w_x-1]), 2).sum()
        h_tv = torch.pow((x[1:, :]-x[:h_x-1, :]), 2).sum()
        return 2. * (h_tv/count_h + w_tv/count_w)

    def _tensor_size(self, t):
        return t.size()[0] * t.size()[1]
    

def get_loss(output, gt, curr_step, loss_dict, opt):
    
    mse = loss_dict['mse']
    ssim = loss_dict['ssim']
    tv = loss_dict['tv']

    steps = opt['train']['loss']['steps']
    max_step = opt['num_iter']

    lambda_mse = opt['train']['loss']['lambda_mse']
    lambda_ssim = opt['train']['loss']['lambda_ssim']
    lambda_tv = opt['train']['loss']['lambda_tv']

    total_loss = 0.
    if steps == 0:
        total_loss += lambda_mse * mse(output, gt)
        total_loss += lambda_ssim * (1 - ssim(output, gt))
        total_loss += lambda_tv * tv(output)
        return total_loss
    else:
        steps.append(max_step)
        for n, step in enumerate(steps):
            if curr_step < step:
                total_loss += lambda_mse[n] * mse(output, gt)
                total_loss += lambda_ssim[n] * (1 - ssim(output, gt))
                total_loss += lambda_tv[n] * tv(output)
                return total_loss
