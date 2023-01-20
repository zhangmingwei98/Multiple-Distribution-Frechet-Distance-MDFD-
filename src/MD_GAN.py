#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author: Mingwei Zhang
@File:MD_GAN.py
@Time:2022/03/14
'''

import sys, os, logging
sys.path.append('../')
import argparse
import pickle
import torch

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from models.WGAN import GeneratorWGAN, DiscriminatorWGAN, weight_init
from mdfd_score import calculate_mdfd_score_of_path

from utils.utils_file import set_random_seed, create_exp_dir, create_node_dir, set_logging, select_device, \
    count_parameters_in_MB, init_data_path, init_exp_name
from data.dataset import load_data
from utils.metrics import cal_fid_and_rs
from config import parse_arguments, save_config

RESULT_DIR = '../results'


def main(args):
    set_random_seed(int(args.seed * args.clients))

    # construct dataset
    dataset_path = init_data_path(args)

    # Set save directory
    exp_name = init_exp_name(args)
    args.save_dir = os.path.join(RESULT_DIR, args.dataset, exp_name)
    create_exp_dir(args.save_dir)
    create_node_dir(args.save_dir)

    set_logging(args.save_dir)
    # Configure devices
    device = select_device(args.gpu, batch_size=1)
    args.pid = os.getpid()

    # construct dataset
    load_data(args, '', subset='construct')
    client_data_path = []
    for i in range(args.clients):
        client_data_path.append(os.path.join(dataset_path, 'client{}'.format(i)))

    # model = select_act_model(args.model, args.channels, device)

    mdfd_value = calculate_mdfd_score_of_path(client_data_path, args.act_model, args.batch_size, args.dims, device)
    logging.info("mdfd value is {}".format(mdfd_value))

    train_datas = []
    train_queues = []
    train_queues_len = []
    for i in range(args.clients):
        args.n_node = i
        print("load data from: ", client_data_path[i])
        train_data = load_data(args, data_path=client_data_path[i], subset='client')
        train_datas.append(train_data)
        train_queue = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers)
        train_queues_len.append(len(train_queue))
        train_queues.append(train_queue)

    args.test_data_path = os.path.join(dataset_path, 'global')
    save_config(args, args.save_dir, logging)
    train_queues_len = min(train_queues_len)
    logging.info('Size of train dataset: {}'.format(train_queues_len))

    net_G = GeneratorWGAN(args).to(device).apply(weight_init)
    optimizer_G = optim.RMSprop(net_G.parameters(), lr=args.lr, alpha=0.9)
    logging.info('The size of net_G:%.2fMB' % count_parameters_in_MB(net_G))

    netD_list = []
    for id in range(args.clients):
        netD_list.append(DiscriminatorWGAN(args, id=id).to(device).apply(weight_init))
    logging.info('The size of net_D:%.2fMB' % count_parameters_in_MB(netD_list[0]))
    optimizerD_list = []
    for i in range(args.clients):
        optimizerD_list.append(optim.RMSprop(netD_list[i].parameters(), lr=args.lr, alpha=0.9))

    logging.info('## Begin training ##')

    def swap_parameter(net1, net2):
        dict1 = net1.state_dict()
        dict2 = net2.state_dict()
        net1.load_state_dict(dict2)
        net2.load_state_dict(dict1)
        # return net1,net2

    iter_ = 0

    pbar = tqdm(total=args.iters, ncols=80, leave=True, desc="train")
    while iter_ <= args.iters:
        if iter_ % args.iter_k == 0:
            swap_worker = np.random.randint(0, args.clients, args.clients)
            # logging.info("Swap Discriminators{}".format(swap_worker))
            for i in range(args.clients):
                swap_parameter(netD_list[i], netD_list[swap_worker[i]])
        if iter_ % train_queues_len == 0:
            train_iter = []
            for i in range(args.clients):
                train_iter.append(iter(train_queues[i]))

        # progress_bar = tqdm(enumerate(train_queue), total=len(train_queue))

        loss_fake_list = []
        errD_list = []
        for (i, net_D), optimizer_D in zip(enumerate(netD_list), optimizerD_list):
            # train D with real data
            net_D.zero_grad()
            data_real, target_real = train_iter[i].next()
            # if i==0:
            #     print(target_real)
            data_real = data_real.to(device)
            # .to(device)
            # target_real = target_real.to(device)
            batch_num = data_real.size(0)

            noise = torch.randn(batch_num, args.z_dim, 1, 1, device=device)
            data_fake = net_G(noise)

            D_real = net_D(data_real)
            D_fake = net_D(data_fake)

            D_real_loss = - torch.mean(D_real)
            D_fake_loss = torch.mean(D_fake)
            D_loss = D_real_loss + D_fake_loss

            D_loss.backward(retain_graph=True)
            optimizer_D.step()

            for parm in net_D.parameters():
                parm.data.clamp_(-args.clamp, args.clamp)

            # if (iters + 1) % args.D_iters == 0:
            loss_fake_list.append(D_fake_loss)
            errD_list.append(D_loss)

        # train Generator
        if iter_ % args.D_iters == 0:
            sum = 0
            for i in range(args.clients):
                sum = sum + loss_fake_list[i] * 0.9
            loss_fake_sum = (1 / args.clients) * sum

            G_loss = -loss_fake_sum
            net_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        D_loss_list = [round(x.item(), 2) for x in errD_list]

        if iter_ == 0 or iter_ % args.iters_show == 0:
            logging.info("iters_result: {} loss_G: {:.4f} loss_D: {} "
                         .format(iter_, G_loss, D_loss_list))

            fid, rs = cal_fid_and_rs(args, net_G, args.save_dir, device, channels=args.channels)
            logging.info("iters_result: {} FID: {:.4f} RS: {:.4f}".format(iter_, fid, rs))

            # plot the figure
            noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)
            samples = net_G(noise).detach().cpu()
            vutils.save_image(samples,
                              os.path.join(os.path.join(args.save_dir, "epoch_image"), f"iters_{iter_}.png"),
                              normalize=True)

            torch.save(net_D.state_dict(), os.path.join(args.save_dir, 'net_D.pth'.format(iter_)))
            torch.save(net_G.state_dict(), os.path.join(args.save_dir, 'net_G.pth'.format(iter_)))
            if iter_ != 0:
                pbar.update(args.iters_show)
        iter_ = iter_ + 1


if __name__ == '__main__':
    args = parse_arguments()
    args.method = 'MD_GAN'
    main(args)
