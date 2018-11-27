#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-26 上午11:46
"""
import Nets
import Train

if __name__ == '__main__':
    net = Nets.ONet()

    Trainer = Train.Trainer(net, './param/onet.pt', r'/home/lievi/celeba_gen/48')
    Trainer.train()