#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: load_data.py
@Time: 2022/09/30
"""
import glob
import os.path
import random
import numpy as np
import torch
import torchvision.utils as vutils
import scipy.misc as misc
import cv2

from data.mydataset import MyDataset
from src.config import parse_arguments
from utils.utils_file import init_data_path

DATA_DIR = '../'


def load_data(args, data_path, subset):
    """ Data loader """
    dataset_argument(args)

    if subset == 'construct':
        data_path = init_data_path(args)
        if os.path.exists(data_path):
            print("find dataset", data_path)
            return None
        else:
            train_data = construct_dataset(args, data_path)
            return train_data
    elif subset == 'client':
        train_data = MyDataset(args, folder=data_path, subset='client')
        return train_data
    else:
        train_data = MyDataset(args, quantity=None, folder=data_path, subset=subset)

        return train_data


"""
def select_dataset(args, data_path, subset):
    if args.dataset == 'mnist':
        train_data = MyDataset(args, data_path, subset=subset)
        # datasets.MNIST(root=os.path.join(DATA_DIR, 'MNIST'), download=True,transform=transform_train)
    elif args.dataset == 'fashionmnist':
        # TODO
        pass
        # train_data = MyFashionMnistDataset(args, data_path, subset=subset)
    elif args.dataset == 'cifar10':
        # TODO
        pass
        # train_data = MyCIFAR10Dataset(args, data_path, subset=subset)
    else:
        raise NotImplementedError
    return train_data

"""


def dataset_argument(args):
    '''
    set the argument of mnist, fashionmnist, cifar10... dataset,
    :param args:
    :return:
    '''
    if args.dataset == 'mnist':
        args.data_mean, args.data_std = (0.5,), (0.5,)
        args.channels = 1
        args.nb_classes = 10
        args.test_data_path = "/home/daipl/zmw/DNM_GAN/data/MNIST/test"
    elif args.dataset == 'fashionmnist':
        args.data_mean, args.data_std = (0.5,), (0.5,)
        args.channels = 1
        args.nb_classes = 10
    elif args.dataset == 'cifar10':
        args.data_mean, args.data_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        args.input_size = 32
        args.channels = 3
        args.nb_classes = 10
        args.test_data_path = "/home/daipl/zmw/DNM_GAN/data/CIFAR10/test"


def construct_dataset(args, data_path):
    """ construct the dataset"""

    # Calculate the total data amount of each client
    quantity_each_client = []
    for i in range(args.clients):
        quantity_each_client.append(int((args.quantity_max * (100 - args.quantity_skew * i * 100)) / 100))
    random.shuffle(quantity_each_client)

    # Decide which clients disturbed feature
    # disturb_fea_each_client = []
    # clients_disturb = [i for i in range(args.clients)]
    # random.shuffle(clients_disturb)
    # clients_disturb = clients_disturb[0:int(args.clients * 0.5)]
    # for i in range(args.clients):
    #     if args.disturb_fea != 'null' and (i in clients_disturb):  # To control the probability
    #         disturb_fea_each_client.append(args.disturb_fea)
    #         args.disturb_fea_client = args.disturb_fea_client + '{}:{} '.format(i, args.disturb_fea)
    #     else:
    #         disturb_fea_each_client.append('null')
    #         args.disturb_fea_client = args.disturb_fea_client + '{}:null '.format(i)
    #
    # print("disturb fea client: " + args.disturb_fea_client)

    golbal_path = os.path.join(data_path, 'global')
    os.makedirs(golbal_path)

    global_data = []

    for i in range(args.clients):
        clients_path = os.path.join(data_path, 'client{}'.format(i))
        os.makedirs(clients_path)

        args.n_client = i
        quantity = quantity_each_client[i]
        # args.disturb_fea = disturb_fea_each_client[i]

        data = MyDataset(args, quantity=quantity, subset='construct')
        global_data.append(data)

        # calculate the index of data to sampling for global dataset
        global_data_flag = [j for j in range(len(data))]
        if int(args.num_gen_image / args.clients) < len(data):
            random.shuffle(global_data_flag)
            global_data_flag = global_data_flag[0:int(args.num_gen_image / args.clients)]

        for index, data_ in enumerate(data.train_set):
            # print(data_.shape)

            if not isinstance(data_, type(np.zeros(0))):
                data_ = data_.numpy()

            cv2.imwrite(os.path.join(clients_path, "{}_{}.png".format(data.train_label[index], index)), data_)
            if index in global_data_flag:
                cv2.imwrite(os.path.join(golbal_path, "{}_{}.png".format(data.train_label[index], index)), data_)
        # train_data = select_dataset(args, data_path, 'construct')

    return global_data


if __name__ == '__main__':
    args = parse_arguments()
    load_data(args, '', subset='construct')
    pass
