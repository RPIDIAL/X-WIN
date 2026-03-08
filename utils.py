import torch
import math
import numpy as np
import torch.distributed as dist


def setup_distributed(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def all_reduce_loss(loss):
    reduced_loss = loss.detach().clone()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss /= dist.get_world_size()
    return reduced_loss


def all_gather_tensor(tensor):
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def setup_paramgroup(encoder, predictor):
    param_group = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    return param_group
