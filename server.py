# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo
@File    ：server.py
@Author  ：高筱六和栾昊六
"""
# import cv2
# import socket
# import struct
# import pickle
#
# host_ip = "0.0.0.0"
# port = 8888
#
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((host_ip, port))
# server_socket.listen(5)
#
# print(f"等待连接：{host_ip}:{port}")
# conn, addr = server_socket.accept()
# print(f"连接成功：{addr}")
#
# conn.settimeout(5.0)  # 设置超时 5 秒
#
# data = b""
# payload_size = struct.calcsize(">L")
#
# while True:
#     try:
#         while len(data) < payload_size:
#             packet = conn.recv(4096)
#             if not packet:
#                 print("客户端断开连接")
#                 break
#             data += packet
#
#         if len(data) < payload_size:
#             break  # 防止继续执行
#
#         packed_msg_size = data[:payload_size]
#         data = data[payload_size:]
#         msg_size = struct.unpack(">L", packed_msg_size)[0]
#
#         while len(data) < msg_size:
#             packet = conn.recv(4096)
#             if not packet:
#                 print("客户端断开连接")
#                 break
#             data += packet
#
#         if len(data) < msg_size:
#             break
#
#         frame_data = data[:msg_size]
#         data = data[msg_size:]
#
#         frame = pickle.loads(frame_data)
#         frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
#
#         cv2.imshow("video", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     except socket.timeout:
#         print("等待超时，无数据输入")
#         break
#     except Exception as e:
#         print(f"发生错误：{e}")
#         break
#
# conn.close()
# server_socket.close()
# cv2.destroyAllWindows()

import cv2
import socket
import struct
import pickle

host_ip = "0.0.0.0"
port = 8888

# 创建服务器套接字并绑定地址
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, port))
server_socket.listen(5)

print(f"服务器启动，等待连接：{host_ip}:{port}")

while True:
    # 接受客户端连接
    conn, addr = server_socket.accept()
    print(f"客户端连接成功：{addr}")

    conn.settimeout(5.0)  # 设置超时 5 秒

    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        try:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print("客户端断开连接")
                    break
                data += packet

            if len(data) < payload_size:
                break  # 防止继续执行

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    print("客户端断开连接")
                    break
                data += packet

            if len(data) < msg_size:
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            center_x, center_y = 320, 240
            cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), 4)
            cv2.line(frame, (center_x - 320, center_y), (center_x + 320, center_y), (0, 255, 0), 1)
            cv2.line(frame, (center_x, center_y - 240), (center_x, center_y + 240), (0, 255, 0), 1)
            frame = cv2.resize(frame, (960, 720))
            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except socket.timeout:
            print("等待超时，无数据输入")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            break

    # 关闭当前连接
    conn.close()
    print("客户端连接已关闭，等待新的连接...")
    cv2.destroyAllWindows()
# 关闭服务器套接字
server_socket.close()

