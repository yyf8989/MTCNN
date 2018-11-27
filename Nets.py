#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-24 下午4:13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义PNet，训练时使用12*12的图片
class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.prenet = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(3, 10, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            # 第二个卷积层
            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            # 第三个卷积层
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU(),
        )
        # 分别输出两个卷积层的结果
        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        # print(x.size())
        x1 = self.prenet(x.cuda())
        # print(x.size())
        # 因为置信度要生成一个0到1的数，所以使用sigmoid输出
        cond = F.sigmoid(self.conv4_1(x1))
        # 因为offset要返回一个四维的输出，所以直接输出
        offset = self.conv4_2(x1)
        # print(cond.size())
        # print(offset.size())
        # print('***********************')
        # print(cond, offset)
        return cond, offset


# RNet训练，使用24*24的图片大小
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(3, 28, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            # 第二个卷积层
            nn.Conv2d(28, 48, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            # 第三个卷积层
            nn.Conv2d(48, 64, 2, 1),
        )
        # 输出第一个全连接
        self.mlp1 = nn.Linear(64*3*3, 128)
        self.PRelu1 = nn.PReLU()

        # 分成两部分输出
        # s生成置信度
        self.mlp2_1 = nn.Linear(128, 1)
        # 生成坐标点
        self.mlp2_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        # print(x.size())
        self.mlp_in = x.view(x.size(0), -1)
        # print(self.mlp_in.size())
        y = self.mlp1(self.mlp_in)
        y = self.PRelu1(y)
        label = F.sigmoid(self.mlp2_1(y))
        offset = self.mlp2_2(y)
        return label, offset


# ONet训练使用48*48的图片大小
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(3, 32, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            # 第二个卷积层
            nn.Conv2d(32, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            # 第三个卷积层
            nn.Conv2d(64, 64, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            # 第四个卷积层
            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU()
        )
        # 使用全连接进行输出
        self.mlp1 = nn.Linear(3*3*128, 256)
        # 使用全连接输出第一个置信度数据
        self.mlp2_1 = nn.Linear(256, 1)
        # 输出坐标值的位置
        self.mlp2_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        # print(x.size())
        self.mlp_in = x.view(x.size(0), -1)
        # print(self.mlp_in.size())
        y = self.mlp1(self.mlp_in)

        label = F.sigmoid(self.mlp2_1(y))
        offset = self.mlp2_2(y)
        return label, offset