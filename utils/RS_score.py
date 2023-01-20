#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File:RS_score.py
@Time:2022/03/26
"""
import torch
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
import sys, os
sys.path.append("../")
from models.ResNet import ResNet, BasicBlock
from data.dataset import load_data

def resnet_score(imgs, batch_size=64, splits=10, root_path="..", dataset='mnist', channels=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    resize -- if image size is smaller than 229, then resize it to 229
    splits -- number of splits, if splits are different, the inception score could be changing even using same data
    """
    # Set up dtype
    device = torch.device("cuda:0")  # you can change the index of cuda

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    # print('Creating data loader')
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], channels=channels).to(device)
    resnet_model.load_state_dict(torch.load(os.path.join(root_path,'utils/parameter/ResNet_{}.pth'.format(dataset))))
    resnet_model.eval()
    # up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

    def get_pred(x):
        # if resize:
        #     x = up(x)
        x = resnet_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    # print('Computing predictions using inception v3 model')
    preds = np.zeros((N, 10))

    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].to(device)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean KL Divergence
    # print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


#------------------- main function -------------------#
# example of torch dataset, you can produce your own dataset
# cifar = dset.CIFAR10(root='data/', download=True,
#                      transform=transforms.Compose([transforms.Resize(32),
#                                                    transforms.ToTensor(),
#                                                    transforms.Normalize(mean_inception, std_inception)
#                                                    ])
#                      )
def calculate_rs_given_paths(args, path, root_path='..', dataset='mnist', channels=1):
    train_data = load_data(args, path, subset="server")
    mean, std = resnet_score(train_data, splits=10, root_path=root_path, dataset=dataset, channels=channels)
    return mean


if __name__=='__main__':
    args = parse_arguments()
    train_data = load_data(args, os.path.join("/home/daipl/zmw/DNM_GAN/results/mnist/Ours_10_nodes_0.1_bias_0_run/server_Close_0", "retrain_dataset"), subset="server")
    mean, std = resnet_score(train_data, splits=10)
    print('IS is %.4f' % mean)