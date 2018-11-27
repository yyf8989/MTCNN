#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-26 上午11:49
"""
import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from mtcnn_tool import utils
import Nets
from torchvision import transforms
import time
import signal

# 定义检测器
pnet_param = r"./param/pnet.pt"
rnet_param = r"./param/rnet.pt"
onet_param = r"./param/onet.pt"


class Detector:
    def __init__(self, pnet_param, rnet_param, onet_param, isCuda=True):
        self.isCuda = isCuda
        self.pnet = Nets.PNet()
        self.rnet = Nets.RNet()
        self.onet = Nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        # 加载网络参数
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        # 网络是测试
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        # 定义transform为ToTensor
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = (start_index[1].data.float() * stride) / scale
        # print(_x1.type())
        _y1 = (start_index[0].data.float() * stride) / scale
        _x2 = (start_index[1].data.float() * stride + side_len) / scale
        _y2 = (start_index[0].data.float() * stride + side_len) / scale

        ow = _x2 - _x1
        # print(ow.type())
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        # print(_offset.type())
        # print(_offset[0].type())
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        # print([x1, y1, x2, y2, cls])
        return [x1, y1, x2, y2, cls]

    def __pnet_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)
        scale = 1.
        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)
            _cls, _offset = self.pnet(img_data)
            # print(_cls)
            # print(_cls.max())
            # print('--/---------')
            # print(_cls, _offset)
            cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
            # 取出置信度大于0.6的
            # print(_cls[0].size())
            # print(_cls[0][0].size())
            # print(offset[0].size())
            # print('12344564567890987654345678')
            # print(cls, offset)
            # print(cls)
            idxs = torch.nonzero(torch.gt(cls, 0.6))
            # print(idxs)
            # print(idxs.size())
            for idx in idxs:
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _w = int(w * scale)
            # print('_w', _w)
            _h = int(h * scale)
            # print('_h', _h)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
            # print('min_side_len:', min_side_len)

        return utils.nms(np.array(boxes), 0.5)

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        # print(_pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.6)

        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        return utils.nms(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnet_boxes):

        _image_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _image_dataset.append(img_data)

        img_dataset = torch.stack(_image_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.95)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + ow * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.8, isMin=True)

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # print('PNET is OK')

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        # print('rNET is OK')

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([0])
        end_time = time.time()
        t_onet = end_time - start_time
        # print('oNET is OK')

        t_sum = t_pnet + t_rnet + t_onet

        print('total:{0} pnet:{1} rnet:{2} onet:{3}'.format(t_sum, t_pnet, t_rnet, t_onet))
        return onet_boxes


if __name__ == '__main__':
    image_file = r"/home/lievi/Pictures/015.png"
    detector = Detector(pnet_param, rnet_param, onet_param)
    with Image.open(image_file) as im:
        boxes = detector.detect(im)
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red')
        im.show()
