#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: FedGAN.py
@Time: 2022/10/20
"""
import sys,os
sys.path.append('../')

import argparse
import pickle
import torch
import logging
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
    alpha = 0.01

    set_random_seed(int(args.seed*args.clients))

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
        train_data = load_data(args, data_path=client_data_path[i], subset='client')
        train_datas.append(train_data)
        train_queue = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers)
        train_queues_len.append(len(train_queue))
        train_queues.append(train_queue)

    args.test_data_path = os.path.join(dataset_path, 'global')
    save_config(args, args.save_dir, logging)
    train_queues_len = min(train_queues_len)
    logging.info('Size of train dataset: {}'.format(train_queues_len))

    netG_list = []
    for id in range(args.clients):
        netG_list.append(GeneratorWGAN(args, id=id).to(device).apply(weight_init))
    logging.info('The size of net_G:%.2fMB' % count_parameters_in_MB(netG_list[0]))

    optimizerG_list = []
    for i in range(args.clients):
        optimizerG_list.append(optim.RMSprop(netG_list[i].parameters(), lr=args.lr, alpha=0.9))

    netD_list = []
    for id in range(args.clients):
        netD_list.append(DiscriminatorWGAN(args,id=id).to(device).apply(weight_init))
    logging.info('The size of net_D:%.2fMB' % count_parameters_in_MB(netD_list[0]))
    optimizerD_list = []
    for i in range(args.clients):
        optimizerD_list.append(optim.RMSprop(netD_list[i].parameters(), lr=args.lr, alpha=0.9))

    logging.info('## Begin training ##')

    iter_ = 0
    losses = []

    pbar = tqdm(total=args.iters, ncols=80, leave=True, desc="train")

    while iter_ <= args.iters:
        loss_fake_list = []
        errD_list = []
        fake_list = []
        parameters_D_diff = []
        for i in range(args.clients):
            parameters_D_diff.append([])
        if iter_ % train_queues_len == 0:
            train_iter = []
            for i in range(args.clients):
                train_iter.append(iter(train_queues[i]))

        for (i, net_D), optimizer_D in zip(enumerate(netD_list), optimizerD_list):
            #train D with real data
            net_D.zero_grad()
            data_real, target_real = train_iter[i].next()
            data_real = data_real.to(device)
            # .to(device)
            # target_real = target_real.to(device)
            batch_num = data_real.size(0)

            noise = torch.randn(batch_num, args.z_dim, 1, 1, device=device)
            data_fake = netG_list[i](noise)

            D_real = net_D(data_real)
            D_fake = net_D(data_fake)

            D_real_loss = - torch.mean(D_real)
            D_fake_loss = torch.mean(D_fake)
            D_loss = D_real_loss + D_fake_loss

            D_loss.backward(retain_graph=True)
            optimizer_D.step()

            for parm in net_D.parameters():
                parm.data.clamp_(-args.clamp, args.clamp)

            loss_fake_list.append(D_fake_loss)
            errD_list.append(D_loss)
            fake_list.append(data_fake)
        for (i, netD) in enumerate(netD_list):
                for (j, parm) in enumerate(netD.parameters()):
                    parameters_D_diff[i].append(parm.grad)
                    parm.data = parm.data + alpha * parameters_D_diff[i][j]

        if iter_ % args.D_iters == 0 and iter_ != 0:

            parameters_D_average = []
            for diff_0, diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_7, diff_8, diff_9 in zip(parameters_D_diff[0],
                                                                                                      parameters_D_diff[1],
                                                                                                      parameters_D_diff[2],
                                                                                                      parameters_D_diff[3],
                                                                                                      parameters_D_diff[4],
                                                                                                      parameters_D_diff[5],
                                                                                                      parameters_D_diff[6],
                                                                                                      parameters_D_diff[7],
                                                                                                      parameters_D_diff[8],
                                                                                                      parameters_D_diff[9]):
                parameters_D_average.append(
                    0.1 * (diff_0 + diff_1 + diff_2 + diff_3 + diff_4 + diff_5 + diff_6 + diff_7 + diff_8 + diff_9))

            for (i, netD) in enumerate(netD_list):
                for (j, parm) in enumerate(netD.parameters()):
                    parm.data = parm.data - alpha * parameters_D_average[j]

            errG_list = []
            parameters_G_diff = []
            for i in range(args.clients):
                parameters_G_diff.append([])

            for (i, netG), optimizerG in zip(enumerate(netG_list), optimizerG_list):
                loss_fake = torch.mean(netD_list[i](fake_list[i]))
                errG = - loss_fake
                netG.zero_grad()
                errG.backward()
                optimizerG.step()

                errG_list.append(errG)

            for (i, netG) in enumerate(netG_list):
                for j, parm in enumerate(netG.parameters()):
                    parameters_G_diff[i].append(parm.grad)
                    parm.data = parm.data + alpha * parameters_G_diff[i][j].detach()

            parameters_G_average = []
            for diff_0, diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_7, diff_8, diff_9 in zip(parameters_G_diff[0],
                                                                                                      parameters_G_diff[1],
                                                                                                      parameters_G_diff[2],
                                                                                                      parameters_G_diff[3],
                                                                                                      parameters_G_diff[4],
                                                                                                      parameters_G_diff[5],
                                                                                                      parameters_G_diff[6],
                                                                                                      parameters_G_diff[7],
                                                                                                      parameters_G_diff[8],
                                                                                                      parameters_G_diff[9]):
                parameters_G_average.append(
                    0.1 * (diff_0 + diff_1 + diff_2 + diff_3 + diff_4 + diff_5 + diff_6 + diff_7 + diff_8 + diff_9))

            for (i, netG) in enumerate(netG_list):
                for j, parm in enumerate(netG.parameters()):
                    parm.data = parm.data - alpha * parameters_G_average[j]

        # if args.gpu != -1:
        #     for err in errD_list:
        #         err = err.cpu()
        #     for err in errG_list:
        #         err = err.cpu()
        #
        # err_items = [err.item() for err in errD_list] + [err.item() for err in errG_list]
        # losses.append(tuple(err_items))

        if iter_ % args.iters_show == 0:
            # logging.info("iters_result: {} loss_G: {:.4f} loss_D: {} "
            #              .format(iter_, G_loss, D_loss_list))

            fid, rs = cal_fid_and_rs(args, netG_list[0], args.save_dir, device, channels=args.channels)
            logging.info("iters_result: {} FID: {:.4f} RS: {:.4f}".format(iter_, fid, rs))

            # plot the figure
            noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)
            samples = netG_list[0](noise).detach().cpu()
            vutils.save_image(samples,
                              os.path.join(os.path.join(args.save_dir, "epoch_image"), f"iters_{iter_}.png"),
                              normalize=True)

            torch.save(netD_list[0].state_dict(), os.path.join(args.save_dir, 'net_D.pth'.format(iter_)))
            torch.save(netG_list[0].state_dict(), os.path.join(args.save_dir, 'net_G.pth'.format(iter_)))
            if iter_ != 0:
                pbar.update(args.iters_show)
        iter_ = iter_ + 1

if __name__ == '__main__':
    args = parse_arguments()
    args.method = 'FedGAN'
    main(args)