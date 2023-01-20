#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: Mingwei Zhang
@File: disturb.py
@Time: 2022/10/10
"""
import cv2
import numpy as np
import random


def add_mask(train_set, beta, x, y):
    """
    Add mask block
    :param train_set:
    :param beta:
    :param x:
    :param y:
    :return:
    """
    for data in train_set:
        mask_color = [255, 255, 255]

        if len(data.shape) == 3 and data.shape[0] == 3:
            x1, x2 = x, x + beta
            y1, y2 = y, y + beta
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    if ((i >= x1)    and (i <= x2) and (j >= y1) and (j <= y2)):
                        data[0][i][j] = mask_color[0]
                        data[1][i][j] = mask_color[1]
                        data[2][i][j] = mask_color[2]
        elif len(data.shape) == 2:
            x1, x2 = x, x + beta
            y1, y2 = y, y + beta
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if ((i >= x1) and (i <= x2) and (j >= y1) and (j <= y2)):
                        data[i][j] = mask_color[0]

    # mask_block = np.zeros((beta, beta), dtype="uint8")
    # cv2.rectangle(mask_block, (25, 25), (275, 275), 255, -1)
    # cv2.imshow("mask_block", mask_block)
    # cv2.imwrite('./mask_block.png', mask_block)

    return train_set


def add_sp_noise(train_set, beta):  # 添加椒盐噪声
    """
    Add salt and pepper noise
    :param train_set:
    :param beta:
    :return:
    """
    for data in train_set:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if random.random() < beta:
                    data[i][j] = 0 if random.random() < 0.5 else 255
                else:
                    data[i][j] = data[i][j]
    return train_set


if __name__ == '__main__':
    pass
