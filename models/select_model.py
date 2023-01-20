#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: select_model.py
@Time: 2022/09/30
"""

from pytorch_fid.inception import InceptionV3

def select_act_model(model_name, dims, device):
    "Selcet the model to calculate stastics"
    if model_name == "inceptionv3":

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
    elif model_name == "resnet":
        pass
    else:
        raise RuntimeError('Invalid model: %s' % model_name)
    return model


if __name__ == '__main__':
    pass
