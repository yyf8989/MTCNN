import numpy as np


# 定义IOU（交并比）计算公式， 传入真实框和其他移动后的框
def iou(box, boxes, isMin=False):
    # 计算原始真实框的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    # 计算移动后的框的面积，这里计算的是矩阵
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 找到两个框的内部点计算交集
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])
    # 然后找到交集区域的长和宽，有的框没有交集那么相差可能为负，所以需要使用0来规整数据
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    # 计算交集的面积
    inter_area = w * h
    # 两种计算方法：1是交并比等于交集除以并集，2是交集除以最小的面积
    if isMin:
        ovr_area = np.true_divide(inter_area, np.minimum(boxes_area, box_area))
    else:
        ovr_area = np.true_divide(inter_area, (boxes_area + box_area - inter_area))
    # 返回交并比，也就是IOU
    return ovr_area


# 定义NMS，筛选不符合标准的线框
def nms(boxes, thresh=0.3, isMin=False):
    # 如果照片里面没有框数据了，就返回空列表
    if boxes.shape[0] == 0:
        return np.array([])
    # 以计算出的iou从大到小排列
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    r_boxes = []
    # 如果框的有1个以上就进行对比
    while _boxes.shape[0] > 1:
        # 取出最大的框
        a_box = _boxes[0]
        # 剩下的框分别和之前的进行比对
        b_boxes = _boxes[1:]
        # 先将最大iou的框添加到保留框的列表中
        r_boxes.append(a_box)
        # 保留iou 小于0.3的，说明这个框和目前比对的不是同一个框，去除交集较多的框
        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        _boxes = b_boxes[index]
        # _boxes = b_boxes[iou(a_box, b_boxes, isMin) < thresh]
    # 如果保留的框数量大于0,则添加iou最大的那个框
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    # 将这些框堆叠在一起
    return np.stack(r_boxes)


# 定义将图片框变为正方形的工具
def convert_to_square(bbox):
    # 先将数据copy
    square_bbox = bbox.copy()
    # 如果数据框内没有框，则返回空列表
    if bbox.shape[0] == 0:
        return np.array([])
    # 计算出框的长宽
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    # 找出最大的那个边
    max_side = np.maximum(h, w)
    # 计算正方形的左上角的点
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    # 返回到正方形的列表
    return square_bbox


# 图片处理
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


if __name__ == '__main__':
    a = np.array([1, 1, 11, 11])
    bs = np.array([[1, 1, 10, 10], [11, 11, 20, 20]])
    print(iou(a, bs))
