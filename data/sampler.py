#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@Author: Mingwei Zhang
@File:sampling.py
@Time:2022/01/13
'''


import numpy as np
import random
from torchvision import datasets, transforms


def bias_sampler(train_set, train_label, num_nodes, nb_classes, n_node, bias, quantity):
    #Calculates the main classification of the current node
    data, label = [], []
    if num_nodes <= nb_classes:
        node_main_class = n_node
    else:
        if n_node <= num_nodes:
            node_main_class = n_node
        else:
            node_main_class = nb_classes % n_node

    # Sort out each category
    # quantity : the amount of all data

    num_main_set = int(bias * quantity)
    num_else_set = int((quantity - num_main_set) / (nb_classes - 1))

    data_one_class = train_set[train_label[:] == node_main_class]
    if len(data_one_class) >= num_main_set:
        data_one_class = random.choices(list(data_one_class), k=num_main_set)
        data.extend(data_one_class)
    else:
        data.extend(data_one_class)
        data.extend(random.choices(list(data_one_class), k=(num_main_set - len(data_one_class))))
        # raise IndexError("The amount of original data is insufficient.")

    label_one_class = []
    for i in range(num_main_set):
        label_one_class.append(node_main_class)

    label.extend(label_one_class)
    # print("data_one_class : {}".format(data_one_class.size()))

    for i in range(nb_classes):

        if i == node_main_class : continue
        else:
            data_one_class = train_set[train_label[:] == i]
            label_one_class = []
            for flag in (train_label[:] == i):
                if flag == True: label_one_class.append(i)

            len_label_one_class = len(label_one_class)
            if len_label_one_class < num_else_set:
                data.extend(data_one_class)
                data.extend(random.choices(list(data_one_class), k=(num_else_set-len_label_one_class)))
                label.extend(label_one_class)
                label.extend(random.choices(list(label_one_class), k=(num_else_set-len_label_one_class)))
            else:
                data.extend(random.sample(list(data_one_class), k=num_else_set))
                label.extend(random.sample(list(label_one_class), k=num_else_set))
    # print("data : {}".format(data[1]))
    return data, label

# def dirichlet_sampler(train_set, train_label, num_nodes, nb_classes, n_node):


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = bias_sampler(dataset_train.data, dataset_train.targets, 10, 10, 1, 0.2, 10000)