# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo 
@File    ：postprocess.py
@Author  ：高筱六和栾昊六
"""
import math

import cv2


def center_bbox(left, top, right, bottom, img_size=(640, 480)):
    """ 计算目标框的新位置，使其居中 """
    box_w = right - left
    box_h = bottom - top

    new_center_x = img_size[0] // 2
    new_center_y = img_size[1] // 2

    new_left = round(new_center_x - box_w / 2)
    new_top = round(new_center_y - box_h / 2)
    new_right = round(new_center_x + box_w / 2)
    new_bottom = round(new_center_y + box_h / 2)

    # 确保坐标不超出图像边界
    new_left = max(0, min(img_size[0] - 1, new_left))
    new_top = max(0, min(img_size[1] - 1, new_top))
    new_right = max(0, min(img_size[0] - 1, new_right))
    new_bottom = max(0, min(img_size[1] - 1, new_bottom))

    return new_left, new_top, new_right, new_bottom


def angle_dff(height_dff):
    short_side = height_dff * 38 / 158
    long_side = 200
    # 计算正切值
    tan_value = short_side / long_side
    # 计算弧度值
    radian_angle = math.atan(tan_value)
    # 将弧度值转换为角度值
    angle = math.degrees(radian_angle)

    # angle=height_dff*39/180

    return angle


def draw_one(image, box, score, cl, dw, dh, r, CLASSES):
    img_h, img_w = image.shape[:2]  # 获取图像尺寸

    # 获取原始目标框
    left, top, right, bottom = box

    # 进行缩放变换
    left = int((left - dw) / r[0])
    top = int((top - dh) / r[0])
    right = int((right - dw) / r[0])
    bottom = int((bottom - dh) / r[0])

    # 确保坐标不会超出边界
    left = max(0, min(img_w - 1, left))
    top = max(0, min(img_h - 1, top))
    right = max(0, min(img_w - 1, right))
    bottom = max(0, min(img_h - 1, bottom))

    # 画原始目标框
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)

    # 添加类别和置信度信息
    label = f'{CLASSES[cl]} {score:.2f}'
    # print(label)
    cv2.putText(image, label, (left, max(10, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    return left, top, right, bottom


def draw(image, boxes, scores, classes, dw, dh, r, CLASSES):
    # anchor = []
    img_h, img_w = image.shape[:2]  # 获取图像尺寸

    for box, score, cl in zip(boxes, scores, classes):
        if score < 0.5:  # 过滤低置信度目标
            continue

        # 获取原始目标框
        left, top, right, bottom = box

        # 进行缩放变换
        left = int((left - dw) / r[0])
        top = int((top - dh) / r[0])
        right = int((right - dw) / r[0])
        bottom = int((bottom - dh) / r[0])

        # 确保坐标不会超出边界
        left = max(0, min(img_w - 1, left))
        top = max(0, min(img_h - 1, top))
        right = max(0, min(img_w - 1, right))
        bottom = max(0, min(img_h - 1, bottom))

        # 画原始目标框
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        # cv2.circle(image, ((left + right) // 2, (top + bottom) // 2), 1, (0, 255, 255), 3)
        # print((left + right) // 2, (top + bottom) // 2)
        # 添加类别和置信度信息
        label = f'{CLASSES[cl]} {score:.2f}'
        # print(label)
        cv2.putText(image, label, (left, max(10, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # 计算居中的目标框位置
        new_left, new_top, new_right, new_bottom = center_bbox(left, top, right, bottom, (img_w, img_h))

        # # 复制原始目标框区域
        # roi = image[top:bottom, left:right].copy()
        #
        # # 计算新框尺寸
        # new_h = new_bottom - new_top
        # new_w = new_right - new_left

        # # 调整原始区域大小以适应新框
        # if (new_w, new_h) != roi.shape[:2]:
        #     roi_resized = cv2.resize(roi, (new_w, new_h))
        # else:
        #     roi_resized = roi

        # 粘贴到新位置
        # image[new_top:new_bottom, new_left:new_right] = roi_resized

        # 画新目标框
        # cv2.rectangle(image, (new_left, new_top), (new_right, new_bottom), (0, 255, 0), 2)

        # 画新目标框的中心点
        # cv2.circle(image, ((new_left + new_right) // 2, (new_top + new_bottom) // 2), 10, (0, 255, 255), 3)

        # 记录新框坐标
        # anchor.append([new_left, new_top, new_right, new_bottom])

        left, top, right, bottom = box

    return new_left, new_top, new_right, new_bottom, left, top, right, bottom
