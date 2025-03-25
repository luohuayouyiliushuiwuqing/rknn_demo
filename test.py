# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo 
@File    ：test.py
@Author  ：高筱六和栾昊六
"""




# def serial_process(command_queue):
#     """串口进程，控制云台角度"""
#     ser = SerialCommunicator(SERIAL_PORT, BAUDRATE, 8, 1, "N")
#     ser.open_serial()
#     ser.send_data("0")
#     ser.send_data("*90")
#
#     old_angle = 0
#     while True:
#         angle = command_queue.get()
#         if angle is None:  # 结束信号
#             break
#         if angle != old_angle:
#             ser.send_data(f"*{angle}")
#             print(f"Sent Angle: *{angle}")
#             old_angle = angle
#
#     ser.close_serial()
#
#
# def tcp_process(frame_queue):
#     """TCP 进程，向服务器发送数据"""
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect((SERVER_IP, SERVER_PORT))
#
#     while True:
#         frame = frame_queue.get()
#         if frame is None:  # 结束信号
#             break
#         _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#         data = pickle.dumps(buffer)
#         size = struct.pack(">L", len(data))
#         client_socket.sendall(size + data)
#
#     client_socket.close()
#
#
# def video_process(command_queue, frame_queue):
#     """视频处理进程"""
#     cap = cv2.VideoCapture(VIDEO_SOURCE)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 60)
#
#     model = RKNN_model_container(MODEL_PATH, target="rk3588", device_id=None)
#
#     frame_count = 0
#     start_time = time.time()
#
#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#
#             # 预处理 & 推理
#             img, ratio, (dw, dh) = letterbox(frame, new_shape=IMG_SIZE)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             outputs = model.run(inputs=[img])
#
#             boxes, classes, scores = post_process(outputs, img_sz=IMG_SIZE, nms=NMS_THRESH, obj_conf=OBJ_THRESH)
#
#             if boxes is not None:
#                 new_left, new_top, new_right, new_bottom, left, top, right, bottom = draw(
#                     frame, boxes, scores, classes, dw, dh, ratio, CLASSES=CLASSES
#                 )
#
#                 if frame_count % 60 == 0:
#                     height_diff = new_bottom - bottom
#                     angle = int(angle_dff(height_diff)) + 90
#                     command_queue.put(angle)  # 发送角度到串口进程
#
#             # 计算 FPS
#             frame_count += 1
#             elapsed_time = time.time() - start_time
#             fps = frame_count / elapsed_time if elapsed_time > 0 else 0
#
#             # 显示 FPS
#             cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#             cv2.circle(frame, (320, 240), 1, (0, 0, 255), 4)
#
#             frame_queue.put(frame)  # 发送帧到 TCP 进程
#
#     except Exception as e:
#         print(f"Error in video process: {e}")
#
#     finally:
#         cap.release()
#         model.release()
#         command_queue.put(None)  # 发送结束信号
#         frame_queue.put(None)  # 发送结束信号
#
#
# def main():
#     """主函数，创建并启动多个进程"""
#     command_queue = mp.Queue()
#     frame_queue = mp.Queue()
#
#     serial_proc = mp.Process(target=serial_process, args=(command_queue,))
#     tcp_proc = mp.Process(target=tcp_process, args=(frame_queue,))
#     video_proc = mp.Process(target=video_process, args=(command_queue, frame_queue))
#
#     serial_proc.start()
#     tcp_proc.start()
#     video_proc.start()
#
#     video_proc.join()
#     serial_proc.join()
#     tcp_proc.join()
#
#
# if __name__ == "__main__":
#     main()
