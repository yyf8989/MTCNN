#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-26 上午11:00
"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


# 人脸的数据采样
class FaceDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        # 打开对应路径下的txt数据文件
        self.dataset.extend(open(os.path.join(path, 'positive.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'negative.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'part.txt')).readlines())

    def __getitem__(self, index):
        # 取出对应索引下的数据，数据结构是，文件名，置信度，x1,y1,x2,y2坐标偏移量
        strs = self.dataset[index].strip().split(" ")
        # 取出文件名，便于取出文件
        img_path = os.path.join(self.path, strs[0])
        # 取出置信度
        cond = torch.Tensor([int(strs[1])])
        # 取出偏移量
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        # 对图片数据进行归一化处理
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)
        img_data = img_data.permute(2, 0, 1)
        # 返回图片数据， 置信度，和偏移量
        # print(img_data.size())
        # print(cond.size())
        # print(offset.size())
        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset(r'/home/lievi/celeba_gen/12')
    print(dataset[0])
    pass
