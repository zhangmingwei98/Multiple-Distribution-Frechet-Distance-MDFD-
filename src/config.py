#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: config.py
@Time: 2022/09/30
"""

import os, pickle
import argparse


def parse_arguments():
    '''
    define parser and add argements
    :return:args
    '''
    parser = argparse.ArgumentParser(description="node_configuration")

    # project parse  set in
    parser.add_argument('--dataset', '-data', type=str, default='fashionmnist', choices=['mnist', 'fashionmnist', 'cifar10'],
                        help='the train dataset')

    parser.add_argument('--run', '-run', type=str, default='0', help='index number of run')
    parser.add_argument('--method', '-method', type=str, default='Ours', help='name of method')

    parser.add_argument('--data_path', type=str, default='../data', help='local path of dataset')
    parser.add_argument('--test_data_path', type=str, default='', help='the path of test data')
    parser.add_argument('--input_size', type=int, default=28, help='imput size of the image')
    parser.add_argument('--nb_classes', type=int, default=10, help='classes of the dataset')
    parser.add_argument('--channels', type=int, default=1, help='chanels of image')
    parser.add_argument('--data_mean', help='mean of dataset, set in data/dataset.py')
    parser.add_argument('--data_std', help='std of dataset, set in data/dataset.py')

    parser.add_argument('--act_model', type=str, default='inceptionv3', choices=['inceptionv3', 'resnet'],
                        help='the model to calculate statistics')
    parser.add_argument('--dims', type=int, default=64, help='Dimensionality of features returned by Inception')

    parser.add_argument('--clients', type=int, default=10, help='the number of client')
    parser.add_argument('--disturb_cate', type=str, default='bias', choices=['bias', 'dirichlet'],
                        help='the method to disturb the distribution of categories skrew')
    parser.add_argument('--alpha', type=float, default=0.1, help='the degree to disturb the distribution of data skrew'
                                                                 'bias : 0.1~1')
    parser.add_argument('--disturb_fea', type=str, default='mask', choices=['null', 'salt', 'mask'],
                        help='the method to disturb the distribution of data feature skrew')
    parser.add_argument('--beta', type=float, default=12,
                        help='the degree to disturb the distribution of data feature skrew'
                             'salt : 0~10'
                             'mask : 2~18')
    parser.add_argument('--quantity_max', type=int, default=10000,
                        help='the degree to disturb the distribution of data')
    parser.add_argument('--quantity_skew', type=float, default=0,
                        help='the degree to disturb the distribution of data')
    parser.add_argument('--disturb_fea_client', type=str, default='', help='whether the client disturb feature')

    parser.add_argument('--n_client', '-n', type=int, default=0, help='the n-th  client/node')
    parser.add_argument('--gpu', default='0', help='number of gpus')
    parser.add_argument('--pid', type=str, help='the id of process')
    parser.add_argument('--workers', type=int, default=2, help='workers process')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=500, help='number of train epoch')
    parser.add_argument('--iters', type=int, default=100000, help='number of train iterations')
    parser.add_argument('--iter_k', type=int, default=5000, help='number of train iterations to communication')
    parser.add_argument('--iters_show', type=int, default=10000, help='number of train iterations')

    parser.add_argument('--iters_Classifier', type=int, default=100000, help='number of train classifier iterations')
    parser.add_argument('--iters_encoder', type=int, default=100000, help='the warm up epoch for auto encoder')

    parser.add_argument('--lr', type=float, default=0.0002, help='the init learning rate')
    parser.add_argument('--D_iters', type=int, default=3, help='number of discritic iters per epoch')
    parser.add_argument('--z_dim', type=int, default=100, help='dimensionality of nosise')
    parser.add_argument('--g_multiplier', type=int, default=32, help='channel multiplier of genetator')
    parser.add_argument('--d_multiplier', type=int, default=64, help='channel multiplier of diacrimination')
    parser.add_argument('--clamp', type=float, default=0.01, help='WGAN clip gradient')
    parser.add_argument('--gp', type=bool, default=False, help='whether use gradient penalty')
    parser.add_argument('--lambda_gp', type=float, default=10, help='gradient penalty lambda hyperparameter')

    parser.add_argument('--seed', type=int, default=8, help='the random seed')
    parser.add_argument('--save', type=str, default='node', help='the name of experiment')
    parser.add_argument('--save_dir', type=str, default='', help='the directory of saving experiment result')
    parser.add_argument('--save_node_dir', type=str, default='', help='the directory of saving node experiment result')
    parser.add_argument('--num_gen_image', type=int, default=10000, help='the number of image for claculate FID')

    args = parser.parse_args()

    return args


def save_config(args, path, logging):
    '''
    store the config
    '''

    config = vars(args)
    with open(os.path.join(path, 'params.pkl'), 'wb') as f_1:
        pickle.dump(config, f_1, protocol=2)
    with open(os.path.join(path, 'params.txt'), 'w') as f_2:
        for key, val in config.items():
            key_val = key + ' : ' + str(val)
            f_2.writelines(key_val)
            logging.info('{}'.format(key_val))
            # print(key_val)


if __name__ == '__main__':
    args = parse_arguments()
    pass
