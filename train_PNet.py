#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-26 上午11:22
"""
import Nets
import Train

if __name__ == '__main__':
    net = Nets.PNet()

    Trainer = Train.Trainer(net, './param/pnet.pt', r'/home/lievi/celeba_gen/12')
    Trainer.train()