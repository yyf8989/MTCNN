#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-26 上午11:23
"""
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from Sampling import FaceDataset


# 定义训练器
class Trainer:
    # 输入内容为网络类型， 保存路径， 数据集的路径
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()
        # 定义使用的损失函数
        # 因为置信度是概率问题，所以使用BCEloss
        self.cls_loss_fn = nn.BCELoss()
        # 因为偏移量是回归问题，所以使用均方差损失定义
        self.offset_loss_fn = nn.MSELoss()
        # 使用Adam优化器进行方向求导
        self.optimizer = optim.Adam(self.net.parameters())

        # 如果保存参数的路径存在就加载
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    # 定义训练函数方法
    def train(self):
        # 使用人脸数据集收集数据
        faceDataset = FaceDataset(self.dataset_path)
        # 取出数据
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=5)
        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                if self.isCuda:
                    # 图片数据
                    img_data_ = img_data_.cuda()
                    # 置信度
                    category_ = category_.cuda()
                    # 偏移量
                    offset_ = offset_.cuda()
                # 输出置信度和偏移量
                # print(img_data_)
                # print(category_)
                # print(offset_)
                _output_category, _output_offset = self.net(img_data_)
                # print(_output_category.size())
                # print(_output_offset.size())
                # 输出的置信度转换维度
                output_category = _output_category.view(-1, 1)
                # 因为要训练置信度，所以取出部分采样
                category_mask = torch.lt(category_, 2)
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                # print(output_category)
                cls_loss = self.cls_loss_fn(output_category, category)
                # 因为要训练偏移量，所以要取出负样本
                offset_mask = torch.gt(category_, 0)
                # print(offset_mask.size())
                offset_index = torch.nonzero(offset_mask)[:, 0]
                # print(offset_index)
                offset = offset_[offset_index]
                # print(offset.size())
                output_offset = _output_offset[offset_index]
                output_offset = output_offset.view(output_offset.size(0), -1)
                # print(output_offset.size())
                offset_loss = self.offset_loss_fn(output_offset, offset)

                loss = cls_loss + offset_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('loss', loss.cpu().data.numpy(),
                      'cls_loss:', cls_loss.cpu().data.numpy(),
                      'offset_loss:', offset_loss.cpu().data.numpy())

            torch.save(self.net.state_dict(), self.save_path)
            print('save success')
