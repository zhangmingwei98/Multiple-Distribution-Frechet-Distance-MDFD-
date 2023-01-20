#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author: Mingwei Zhang
@File:WGAN.py
@Time:2021/11/20
'''
import torch
import torch.nn as nn
import torch.autograd as autograd


class GeneratorWGAN(nn.Module):
    def __init__(self, args, id=0):
        super(GeneratorWGAN, self).__init__()
        self.id = id
        if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
            self.main = nn.Sequential(
                nn.ConvTranspose2d(args.z_dim, args.g_multiplier*8, 4 , 1, 0, bias=False),#4*4
                nn.BatchNorm2d(args.g_multiplier*8),
                nn.ReLU(True),

                # nn.ConvTranspose2d(args.g_multiplier*8,args.g_multiplier*4,4,2,1,bias=False),
                # nn.BatchNorm2d(args.g_multiplier*4),
                # nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier*8, args.g_multiplier*4, 4, 2, 1, bias=False),#8*8
                nn.BatchNorm2d(args.g_multiplier*4),
                nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier*4, args.g_multiplier*2, 2, 2, 1, bias=False),#14*14
                nn.BatchNorm2d(args.g_multiplier*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier*2, args.channels, 4, 2, 1, bias=False), #28*28
                nn.Tanh()
            )
        elif args.dataset == 'cifar10':
            self.main = nn.Sequential(
                nn.ConvTranspose2d(args.z_dim, args.g_multiplier * 8, 4, 1, 0, bias=False),  # 4*4
                nn.BatchNorm2d(args.g_multiplier * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier*8,args.g_multiplier*4,4,2,1,bias=False),  # 8*8
                nn.BatchNorm2d(args.g_multiplier*4),
                nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier * 4, args.g_multiplier * 2, 4, 2, 1, bias=False),  #
                nn.BatchNorm2d(args.g_multiplier * 2),
                nn.ReLU(True),

                # nn.ConvTranspose2d(args.g_multiplier * 2, args.g_multiplier, 2, 2, 1, bias=False),  # 14*14
                # nn.BatchNorm2d(args.g_multiplier),
                # nn.ReLU(True),

                nn.ConvTranspose2d(args.g_multiplier*2, args.channels, 4, 2, 1, bias=False),  # 32*32
                nn.Tanh()
            )

    def get_id(self):
        return self.id

    def forward(self, noise):
        output = self.main(noise)
        return output

class DiscriminatorWGAN(nn.Module):
    def __init__(self, args, id=0):
        super(DiscriminatorWGAN, self).__init__()
        self.id = id
        if args.dataset == 'mnist' or args.dataset == 'fashionmnist':
            self.main = nn.Sequential(
                nn.Conv2d(args.channels, args.d_multiplier, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier, args.d_multiplier * 2, 2, 2, 1, bias=False),
                nn.BatchNorm2d(args.d_multiplier * 2),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier * 2, args.d_multiplier * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.d_multiplier * 4),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier * 4, 1, 4, 1, 0, bias=False),
            )
        elif args.dataset == 'cifar10':
            self.main = nn.Sequential(
                nn.Conv2d(args.channels, args.d_multiplier, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier, args.d_multiplier * 2, 2, 2, 1, bias=False),
                nn.BatchNorm2d(args.d_multiplier * 2),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier * 2, args.d_multiplier * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.d_multiplier * 4),
                nn.LeakyReLU(0.2, inplace=False),

                nn.Conv2d(args.d_multiplier * 4, args.d_multiplier * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(args.d_multiplier * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(args.d_multiplier * 8, 1, 2, 1, 0, bias=False),
            )
    def get_id(self):
        return self.id

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

    def calc_gradient_penalty(self, real_data, fake_data, L_gp, device):
        '''
        compute gradient penalty term
        :param real_data:
        :param fake_data:
        :param L_gp:
        :param device:
        :return:
        '''

        batchsize = real_data.shape[0]
        real_data = real_data.to(device)
        fake_data = fake_data.to(device)
        alpha = torch.rand(batchsize, 1, real_data.shape[2], real_data.shape[2])
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * L_gp
        return gradient_penalty


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    pass
