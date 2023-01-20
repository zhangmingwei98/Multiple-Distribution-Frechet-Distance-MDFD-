#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author: Mingwei Zhang
@File:metrics.py
@Time:2021/12/01
'''
import os
import torch
import torchvision.utils as vutils
from utils.FID_score import calculate_fid_given_paths
from utils.RS_score import calculate_rs_given_paths

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def cal_fid_and_rs(args, net_G, path, device, channels=1):
    num_gen_image = 0
    while num_gen_image < args.num_gen_image:
        noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)
        samples = net_G(noise).detach().cpu()
        for sample in samples:
            if num_gen_image < args.num_gen_image:
                vutils.save_image(sample, os.path.join(path, "gen_samples",
                                                       "image_{}.png".format(num_gen_image)), normalize=True)
                num_gen_image = num_gen_image + 1

    fid = calculate_fid_given_paths(
        [args.test_data_path, os.path.join(path, "gen_samples")],
        args.batch_size, device, 2048, 0)
    rs = calculate_rs_given_paths(args, os.path.join(path, "gen_samples"), root_path='..', dataset= args.dataset, channels=channels)

    return fid, rs

    
if __name__ == '__main__':
    pass
