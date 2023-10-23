import torch


def _get_optimizers(type):
    if type == 'Adam':
        optim = torch.optim.Adam
    elif type == 'SGD':
        optim = torch.optim.SGD
    elif type == 'AdamW':
        optim = torch.optim.AdamW
    else:
        raise NotImplementedError(f'optimizer {type} is not supported yet.')
    return optim


def setup_optimizers(optim_params, opt):
    """
    Args:
        optim_params (list)
    """
    optim_type = opt['type']
    _optim = _get_optimizers(optim_type)
    opt.pop('type')
    ker_lr = opt.pop('ker_lr')
    
    optim_params_list = []
    optim_params_list.append({'params': optim_params[0]})
    optim_params_list.append({'params': optim_params[1], 'lr': ker_lr})
    optim = _optim(optim_params_list, **opt)

    return optim
