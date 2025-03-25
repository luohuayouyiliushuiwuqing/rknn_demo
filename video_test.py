# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo 
@File    ：video_test.py
@Author  ：高筱六和栾昊六
"""
import cv2

# 摄像头设备路径
camera_device = "/dev/video11"

# # 定义视频捕获设备（0 表示默认摄像头）
cap = cv2.VideoCapture(camera_device)

# 设置摄像头的分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 设置视频编码器和输出文件
# 使用 XVID 编码器，保存为 AVI 格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

count = 0

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("开始录制视频，按 'q' 键停止...")
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 写入帧到输出文件
    out.write(frame)

    # 显示帧
    # cv2.imshow('frame', frame)
    count += 1
    # 按 'q' 键退出循环
    if count ==100:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()