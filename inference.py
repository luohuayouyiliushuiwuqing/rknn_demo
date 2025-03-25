# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo 
@File    ：inference.py
@Author  ：高筱六和栾昊六
"""
import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # 对图像进行缩放和填充操作，同时满足步长倍数的约束
    # 输入参数:
    # im: 输入的图像
    # new_shape: 新的图像尺寸，可以是整数或元组
    # color: 填充颜色，默认为黑色

    # 获取当前图像的高度和宽度
    shape = im.shape[:2]  # current shape [height, width]
    # print(im.shape)

    # 如果 new_shape 是整数，将其转换为 (new_shape, new_shape) 的元组形式
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例，取高度和宽度缩放比例中的较小值,这样可以保证图像在缩放过程中不会变形
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算宽度和高度的缩放比例
    ratio = r, r  # width, height ratios

    # 计算缩放后的未填充图像的宽度和高度
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算需要填充的宽度和高度
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # 将填充量平均分配到图像的两侧
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 如果当前图像的尺寸和缩放后的未填充图像尺寸不一致，则进行缩放操作
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 计算图像顶部和底部的填充量
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算图像左侧和右侧的填充量
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 使用 cv2.copyMakeBorder 函数在图像周围添加边框进行填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 返回处理后的图像、缩放比例和填充量
    return im, ratio, (dw, dh)


def filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH=None):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    # candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores, NMS_THRESH=None):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position, IMG_SIZE=None):
    # 获取特征图的高度和宽度
    grid_h, grid_w = position.shape[2:4]

    # 生成网格坐标 (col: 列索引, row: 行索引)
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))

    # 重新调整形状，以便广播计算
    col = col.reshape(1, 1, grid_h, grid_w)  # (1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)  # (1, 1, grid_h, grid_w)

    # 合并列坐标和行坐标，形成网格坐标
    grid = np.concatenate((col, row), axis=1)  # (1, 2, grid_h, grid_w)

    # 计算步长（从特征图尺度转换到输入图像尺度）
    # stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    stride = np.array([IMG_SIZE[0] // grid_h, IMG_SIZE[1] // grid_w]).reshape(1, 2, 1, 1)

    # 通过 DFL（Distribution Focal Loss）方法解码 position 偏移量
    position = dfl(position)  # (batch_size, 4, grid_h, grid_w)

    # 计算左上角坐标 (x1, y1)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]

    # 计算右下角坐标 (x2, y2)
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]

    # 计算最终边界框的坐标 (x1, y1, x2, y2)
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)  # (batch_size, 4, grid_h, grid_w)

    return xyxy


def post_process(input_data, img_sz=None, nms=None, obj_conf=None):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 2
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i], IMG_SIZE=img_sz))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, OBJ_THRESH=obj_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, NMS_THRESH=nms)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores
