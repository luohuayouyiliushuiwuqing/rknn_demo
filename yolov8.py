import time

import numpy as np

from inference import letterbox, post_process
from postprocess import angle_dff, draw_one
from rknn_executor import RKNN_model_container
from holder_serial import SerialCommunicator
import cv2
import socket
import struct
import pickle

# 配置参数
SERIAL_PORT = "/dev/ttyS0"
BAUDRATE = 115200

VIDEO_SOURCE = "/dev/video13"
# VIDEO_SOURCE = "/root/yhj_demo/videos/005.h264"
SERVER_IP = "192.168.1.14"
SERVER_PORT = 8888

MODEL_PATH = "models/v8n/export_rknn640.rknn"
TARGET = "rk3588"
DEVICE_ID = None

IMG_SIZE = (640, 480)
CLASSES = ("Drone", "")
OBJ_THRESH = 0.45
NMS_THRESH = 0.1


def init_serial(port=SERIAL_PORT, baudrate=BAUDRATE, bytesize=8, stopbits=1, parity="N"):
    """初始化串口通信"""
    ser = SerialCommunicator(port, baudrate, bytesize, stopbits, parity)
    ser.open_serial()
    # ser.send_data("0")
    ser.send_data("#000P1500T0025!")  # 发送初始角度
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


def send_data_thread(ser, data):
    ser.send_data(data)


def main():
    """主函数"""
    # 初始化组件
    ser = init_serial()
    client_socket = init_tcp_client()
    cap = init_video()
    model = init_model()

    frame_count = 0
    lost_frame = 0
    start_time = time.time()

    pwm_out = 1500

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("no frame")
                break

            # 预处理 & 推理
            img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 480))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            outputs = model.run(inputs=[img])

            boxes, classes, scores = post_process(outputs, img_sz=IMG_SIZE, nms=NMS_THRESH, obj_conf=OBJ_THRESH)
            if boxes is not None:
                # 找到置信度最高的检测框的索引
                max_conf_index = np.argmax(scores)  # 使用 NumPy 的 argmax 方法

                # 提取置信度最高的检测框的坐标和类别
                highest_conf_box = boxes[max_conf_index]
                highest_conf_class = classes[max_conf_index]
                highest_conf_score = scores[max_conf_index]

                # 绘制置信度最高的检测框
                left, top, right, bottom = draw_one(
                    frame, highest_conf_box, highest_conf_score, highest_conf_class, dw, dh, ratio, CLASSES=CLASSES
                )

                # if frame_count % 1 == 0:
                width_center = (right + left) / 2
                height_center = (bottom + top) / 2

                img_height_center = frame.shape[0] / 2
                img_width_center = frame.shape[1] / 2

                height_diff = img_height_center - height_center

                # 根据检测框高度差计算目标偏移角度（注意：这里函数 angle_dff 需要根据实际情况确定转换规则）
                deflection_angle = - round(angle_dff(height_diff), 3)

                # 计算出目标角度（假设图像中心对应 90°）
                angle = round(deflection_angle + 90, 3)

                # 计算此次应调整的 pwm 增量（比例系数可根据实际情况调节）
                pwm_adjustment = deflection_angle * 2
                # 采用低通滤波平滑 pwm 输出，避免调整幅度过大
                new_pwm = int(pwm_out + pwm_adjustment)
                # 限制 pwm 范围
                new_pwm = max(min(new_pwm, 2500), 500)

                # 当角度变化较大且 pwm 有变化时再发送数据（这里设定角度变化至少 1°）
                if abs(deflection_angle) > 1:
                    pwm_data = f"#000P{new_pwm:04d}T0025!"
                    print(f"Sent: {pwm_data} angle:{angle} Δ={deflection_angle}")
                    pwm_out = new_pwm
                    # 开启新线程发送数据，防止阻塞主线程
                    # threading.Thread(target=send_data_thread, args=(ser, pwm_data)).start()
            else:
                lost_frame += 1
                # print("Lost frame: ", lost_frame)

            # 计算 FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # 显示 FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
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
