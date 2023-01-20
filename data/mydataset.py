#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: dataset_mnist.py.py
@Time: 2022/10/05
"""

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
import torchvision
import cv2

sys.path.append("../")
from src.config import *
from data.disturb import add_mask, add_sp_noise
from data.sampler import bias_sampler


# from data.sampling import sampler

class MyDataset(Dataset):
    def __init__(self, args, quantity=1000, subset='client', folder='./'):
        self.dataset = args.dataset
        self.data_path, self.subset = os.path.join(args.data_path, args.dataset), subset
        self.data_mean, self.data_std = args.data_mean, args.data_std
        self.clients, self.n_client, = args.clients, args.n_client
        self.input_size, self.nb_classes, self.channels = args.input_size, args.nb_classes, args.channels
        self.disturb_cate, self.alpha, self.disturb_fea, self.beta = args.disturb_cate, args.alpha, args.disturb_fea, args.beta
        self.quantity = quantity
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.data_mean, std=self.data_std)])

        if self.subset == 'construct':
            self.transform = None
            self.train_set, self.train_label = self.load_data(self.data_path)
            if self.disturb_cate == 'bias':
                self.train_set, self.train_label = bias_sampler(self.train_set, self.train_label, self.clients,
                                                                self.nb_classes, self.n_client, self.alpha,
                                                                self.quantity)
            elif self.disturb_cate == 'dirichlet':
                pass

            if self.disturb_fea == 'mask':
                opt_flag = args.n_client % 10 if args.n_client > 10 else args.n_client
                x = int(opt_flag % 3 / 3 * self.input_size)
                y = int(int(opt_flag / 3) / 3 * self.input_size)
                # print("the begin position of x and y : {} {}".format(x,y))
                self.train_set = add_mask(self.train_set, self.beta, x, y)
            elif self.disturb_fea == 'salt':
                self.beta = self.n_client * args.beta * 0.01
                print("the salt beta of client {} is {}".format(self.n_client, self.beta))
                self.train_set = add_sp_noise(self.train_set, self.beta)

        elif self.subset == 'node':
            # For compatibility with DNM_GAN
            train_set, train_label = self.load_data(args.data_path, self.data_name, self.label_name)
            # self.train_set, self.train_label = sampler(train_set, train_label, self.num_nodes, self.nb_classes, self.n_node, self.bias)
        elif self.subset == 'client':
            self.train_set, self.train_label = self.load_data_image_label(folder)
        elif self.subset == 'server':
            self.train_set, self.train_label = self.load_data_image_non_label(folder)

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_label[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    def load_data(self, data_folder):
        if self.dataset == 'mnist':
            data = torchvision.datasets.MNIST(root=data_folder, train=True, download=True)
            # print("mnist type", type(data.data)) torch.tensor
        elif self.dataset == 'fashionmnist':
            data = torchvision.datasets.FashionMNIST(root=data_folder, train=True, download=True)
        elif self.dataset == 'cifar10':
            data = torchvision.datasets.CIFAR10(root=data_folder, train=True, download=True)
            # print("cifar10 type",type(data.data)) numpy.narray

        data.train_label = np.array(data.targets)

        return data.data, data.train_label

    def load_data_image_non_label(self, data_folder):
        data_list = glob.glob(os.path.join(data_folder, '*'))
        for i, data_path in enumerate(data_list):
            if self.channels == 1:
                data = cv2.imread(data_path, 0)
            elif self.channels == 3:
                data = cv2.imread(data_path)
            if i == 0:
                x_train = data.reshape(1,  self.input_size, self.input_size)
            else:
                x_train = np.append(x_train, data.reshape(1, self.input_size, self.input_size), axis=0)
        y_train = np.zeros([i + 1])  # fake label
        return x_train, y_train

    def load_data_image_label(self, data_folder):
        data_list = glob.glob(os.path.join(data_folder, '*'))
        y_train = np.zeros([len(data_list)])
        for i, data_path in enumerate(data_list):
            label = int(data_path.split('/')[-1].split('_')[0])
            y_train[i] = label
            if self.channels == 1:
                data = cv2.imread(data_path, 0)
            elif self.channels == 3:
                data = cv2.imread(data_path)
            if i == 0:
                x_train = data.reshape(1, self.input_size, self.input_size)

            else:
                x_train = np.append(x_train, data.reshape(1, self.input_size, self.input_size), axis=0)

        # fake label
        return x_train, y_train


if __name__ == '__main__':
    args = parse_arguments()
    trainDataset = MyDataset(args,
                             folder="/home/daipl/zmw/MDFD/data/mnist/mnist_10_clients_0.4_bias_1_gaussblur_10000_0.05/client0",
                             subset='client')
    j = 0
    for i in range(10):
        num = 0
        for x in trainDataset.train_label:
            if x == i:
                num += 1
                j += 1

        print("num of {} = {}".format(i, num))
    print("all——{}".format(j))
    # print("len of train : {} data1 : {} ".format(len(trainDataset), trainDataset[1]))

    # trainDataset = MyMnistDataset(args, "../results/mnist/4_nodes_0.1_bias_2_run/server/retrain_dataset", 'server')

    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=10,
        shuffle=False,
    )
