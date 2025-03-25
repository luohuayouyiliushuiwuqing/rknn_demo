import time

import math

from inference import letterbox, post_process
from rknn_executor import RKNN_model_container
from holder_serial import SerialCommunicator
import cv2
import socket
import struct
import pickle
import multiprocessing as mp


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
    long_side = 170
    # 计算正切值
    tan_value = short_side / long_side
    # 计算弧度值
    radian_angle = math.atan(tan_value)
    # 将弧度值转换为角度值
    angle = math.degrees(radian_angle)

    # angle=height_dff*39/180

    return angle


def draw(image, boxes, scores, classes, dw, dh, r, CLASSES):
    # anchor = []
    img_h, img_w = image.shape[:2]  # 获取图像尺寸

    for box, score, cl in zip(boxes, scores, classes):
        # if score < 0.5:  # 过滤低置信度目标
        #     continue

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
        # cv2.circle(image, ((left + right) // 2, (top + bottom) // 2), 1, (0, 255, 255), 3)
        # print((left + right) // 2, (top + bottom) // 2)
        # 添加类别和置信度信息
        label = f'{CLASSES[cl]} {score:.2f}'
        print(label)
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


# 配置参数
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 921600

VIDEO_SOURCE = "/dev/video11"
SERVER_IP = "192.168.254.110"
SERVER_PORT = 8888

MODEL_PATH = "models/test/n_small1_rknn.rknn"
TARGET = "rk3588"
DEVICE_ID = None

IMG_SIZE = (640, 480)
CLASSES = ("Drone", "")
OBJ_THRESH = 0.25
NMS_THRESH = 0.1


def init_serial(port=SERIAL_PORT, baudrate=BAUDRATE, bytesize=8, stopbits=1, parity="N"):
    """初始化串口通信"""
    ser = SerialCommunicator(port, baudrate, bytesize, stopbits, parity)
    ser.open_serial()
    ser.send_data("0")
    ser.send_data("*90")  # 发送初始角度
    return ser


def init_tcp_client(server_ip=SERVER_IP, server_port=SERVER_PORT):
    """初始化 TCP 连接"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    return client_socket


def init_video(video_source=VIDEO_SOURCE, width=640, height=480, fps=60):
    """初始化视频流"""
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def init_model(model_path=MODEL_PATH, target=TARGET, device_id=DEVICE_ID):
    """加载模型"""
    return RKNN_model_container(model_path, target, device_id)


def send_frame(client_socket, frame):
    """发送图像到服务器"""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = pickle.dumps(buffer)
    size = struct.pack(">L", len(data))
    client_socket.sendall(size + data)


def main():
    """主函数"""
    # 初始化组件
    ser = init_serial()
    client_socket = init_tcp_client()
    cap = init_video()
    model = init_model()

    old_angle = 90
    frame_count = 0
    lost_frame = 0
    start_time = time.time()

    last_send_time = 0  # 记录上次发送的时间
    SEND_INTERVAL = 0.2  # 200ms 的最小间隔

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # 预处理 & 推理
            img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            outputs = model.run(inputs=[img])

            boxes, classes, scores = post_process(outputs, img_sz=IMG_SIZE, nms=NMS_THRESH, obj_conf=OBJ_THRESH)

            if boxes is not None:
                new_left, new_top, new_right, new_bottom, left, top, right, bottom = draw(
                    frame, boxes, scores, classes, dw, dh, ratio, CLASSES=CLASSES
                )

                # if frame_count % 10 == 0:
                #     height_diff = new_bottom - bottom
                #     if height_diff < 50:
                #         continue
                #     deflection_angle = angle_dff(height_diff)
                #     angle = deflection_angle + 90
                #
                #     # 仅当角度变化大于等于 1° 时才发送
                #     if abs(angle - old_angle) >= 1:
                #         ser.send_data(f"*{angle}")
                #         print(f"Sent Angle: *{angle} Δ={deflection_angle}")
                #         old_angle = angle
            else:
                lost_frame += 1
                print("Lost frame: ", frame_count)

                # if frame_count % 10 == 0:
                #     height_diff = new_bottom - bottom
                #     if height_diff < 50:
                #         continue  # 变化过小，忽略
                #
                #     deflection_angle = angle_dff(height_diff)
                #     angle = deflection_angle + 90
                #     angle_diff = abs(angle - old_angle)
                #
                #     # 获取当前时间
                #     current_time = time.time()
                #
                #     # 1. 角度变化大于等于 5° 时，立即发送
                #     # 2. 角度变化 1°-5° 且时间间隔足够时，发送
                #     if angle_diff >= 5 or (angle_diff >= 1 and current_time - last_send_time >= SEND_INTERVAL):
                #         ser.send_data(f"*{angle}")
                #         print(f"Sent Angle: *{angle}, Δ: {angle_diff}°")
                #         old_angle = angle
                #         last_send_time = current_time  # 更新上次发送时间
                #     else:
                #         continue

            # 计算 FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # 显示 FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.circle(frame, (320, 240), 1, (0, 0, 255), 4)

            # 发送数据到服务器
            send_frame(client_socket, frame)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 释放资源
        cap.release()
        client_socket.close()
        model.release()
        ser.close_serial()


if __name__ == "__main__":
    main()
