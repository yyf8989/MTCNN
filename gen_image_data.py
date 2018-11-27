#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
""" __author__ = "YYF" 
    __MTime__ = 18-11-24 下午4:49
"""
import os
from PIL import Image
import numpy as np
from mtcnn_tool import utils
import traceback
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 定义目标文件位置，内容为图片编号，x1, y1, 和框的长和宽
anno_src = r"/home/lievi/celeba/Anno/list_bbox_celeba.txt"
# 定义图片文件夹位置
img_dir = r"/home/lievi/celeba/img_celeba"

# 定义保存生成图片位置
save_path = r"/home/lievi/celeba_gen"
# 生成3种size的图片，分为12*12, 24*24, 48*48
for face_size in [12, 24, 48]:
    print('gen %i image' % face_size)
    # 将图片分成三个组，正样本，负样本，部分样本
    positive_image_dir = os.path.join(save_path, str(face_size), 'positive')
    negative_image_dir = os.path.join(save_path, str(face_size), 'negative')
    part_image_dir = os.path.join(save_path, str(face_size), 'part')

    # 查看是否文件位置文件夹存在，否则就利用os模块建立
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 分别针对不同的图片分别进行记录
    positive_anno_filename = os.path.join(save_path, str(face_size), 'positive.txt')
    negative_anno_filename = os.path.join(save_path, str(face_size), 'negative.txt')
    part_anno_filename = os.path.join(save_path, str(face_size), 'part.txt')

    # 计算所有的图片样本个数
    positive_count = 0
    negative_count = 0
    part_count = 0

    # 为了防止异常发生，使用try
    try:
        # 打开文件准备写入
        positive_anno_file = open(positive_anno_filename, 'w')
        negative_anno_file = open(negative_anno_filename, 'w')
        part_anno_file = open(part_anno_filename, 'w')
        # 利用迭代器，将原来数据取出
        for i, line in enumerate(open(anno_src)):
            # 由于目标数据的原因，跳过前两行提取数据
            if i < 2:
                continue
            try:
                # 将数据从 ['000001.jpg', '', '', '', '95', '', '71', '226', '313']
                # 转换成['000001.jpg', '95', '71', '226', '313']
                strs = line.strip().split(' ')
                strs = list(filter(bool, strs))
                # strs = line.split()
                # 取出文件名，处于strs第一索引的位置
                image_filename = strs[0].strip()

                # 将文件名和路径和经在一起，进入到图片文件夹中去找到对应数据
                image_file = os.path.join(img_dir, image_filename)
                # 打开对应的图片
                with Image.open(image_file) as img:
                    # 得到图片长宽
                    img_w, img_h = img.size
                    # 从原始的取出坐标，并计算出右下角的坐标点
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    # 如果原始数据中的框大小少于48或者坐标点为负，则跳过
                    if max(w, h) < 48 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    # 定义找到的目标框，使用二维是因为要和后面的很多框进行对比
                    boxes = [[x1, y1, x2, y2]]
                    # 找到中心点坐标
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    # 随意移动中心点20%的浮动，找出
                    for _ in range(5):
                        w_ = np.random.randint(int(-w * 0.2), int(w * 0.2))
                        h_ = np.random.randint(int(-h * 0.2), int(h * 0.2))
                        # 定义找到的中心点坐标
                        cx_ = cx + w_
                        cy_ = cy + h_
                        # 边长在最小边的0.8 和 最大边长的1.25之间取值
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        # 计算出移动中心点之后的左上角和右下角的坐标位置
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len
                        # 利用处理后的数据坐标对图片进行裁剪,先找出坐标特征框坐标数值列表
                        crop_box = np.array([x1_, y1_, x2_, y2_])
                        # 算出偏移量，加算公式 offset = （x1 - x1_) / side_len
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len
                        # 裁剪人脸框
                        face_crop = img.crop(crop_box)
                        # 然后将裁剪后的图片resize成我们需要的训练图片大小
                        face_resize = face_crop.resize((face_size, face_size))
                        # 计算出所得的iou
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        # 如果Iou大于0.65算作正样本
                        if iou > 0.65:
                            positive_anno_file.write(
                                'positive/{0}.jpg {1} {2} {3} {4} {5}\n'.format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                            # flush()方法是用来把文件从内存buffer（缓冲区）中强制刷新到硬盘中，同时清空缓冲区
                            positive_anno_file.flush()
                            # 将resize之后的图片放在positive的文件夹中
                            face_resize.save(os.path.join(positive_image_dir, '{0}.jpg'.format(positive_count)))
                            # 计数
                            positive_count += 1
                        # 如果iou大于0.4则算做部分样本
                        elif iou > 0.4:
                            part_anno_file.write(
                                'part/{0}.jpg {1} {2} {3} {4} {5}\n'.format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, '{0}.jpg'.format(part_count)))
                            part_count += 1
                        # 如果iou小于0.3算作负样本
                        elif iou < 0.3:
                            negative_anno_file.write(
                                'negative/{0}.jpg {1} 0 0 0 0\n'.format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, '{0}.jpg'.format(negative_count)))
                            negative_count += 1

                        _boxes = np.array(boxes)
                    # 制作负样本，随机取样，因为上面制作的样本绝大多数为正样本和部分样本
                    for i in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                        if np.max(utils.iou(crop_box, _boxes)) < 0.3:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  #抗锯齿设置
                            negative_anno_file.write('negative/{0}.jpg {1} 0 0 0 0 \n'.format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, '{0}.jpg'.format(negative_count)))
                            negative_count += 1
            except Exception as e:
                # 打印异常
                traceback.print_exc()
    finally:
        # 关闭文件
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

