# -*- coding: utf-8 -*-
# @Time : 2023/1/10 13:41
# @Author : nlp_zzu
# @File : utils.py
# @File_Description :
import torch
from matplotlib import pyplot as plt


def plot_curve(data):
    """
    画loss曲线
    :param data: loss值
    :return:
    """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    """
    画图像识别结果
    :param img: 图片
    :param label: 标签
    :param name: 名称
    :return:
    """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}:{}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    """
    one_hot编码
    :param label: 标签
    :param depth: 编码长度
    :return:
    """
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
