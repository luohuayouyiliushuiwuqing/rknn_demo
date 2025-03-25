# coding=utf-8
# vim:set fileencoding=utf-8:
"""
@Project ：rknn_demo 
@File    ：holder_serial.py
@Author  ：高筱六和栾昊六
"""
import time

import serial  # 导入串口通信库


class SerialCommunicator:
    def __init__(self, port, baudrate, bytesize, stopbits, parity):
        """
        初始化串口配置
        :param port: 串口号
        :param baudrate: 波特率
        :param bytesize: 数据位
        :param stopbits: 停止位
        :param parity: 奇偶校验位
        """
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.bytesize = bytesize
        self.ser.stopbits = stopbits
        self.ser.parity = parity

    def open_serial(self):
        """
        打开串口
        :return: 打开的串口对象，如果打开失败则返回 None
        """
        try:
            self.ser.open()
            print("串口打开成功！")
            return self.ser
        except serial.SerialException as e:
            print(f"串口打开失败！错误信息: {e}")
            return None

    def send_data(self, data):
        """
        向串口发送数据
        :param data: 要发送的数据
        :return: 发送是否成功
        """
        try:
            self.ser.write(data.encode("utf-8"))
            # print(f"=========> Sent {data}")
            return True
        except serial.SerialException as e:
            print(f"数据发送失败！错误信息: {e}")
            return False

    def close_serial(self):
        """
        关闭串口
        """
        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    port = '/dev/ttyUSB0'
    baudrate = 921600
    bytesize = 8
    stopbits = 1
    parity = "N"

    ser = SerialCommunicator(port, baudrate, bytesize, stopbits, parity)
    ser.open_serial()
    if ser:
        ser.send_data("0")
        # time.sleep(5)
        print("end")
        # ser.send_data("*80")
        # time.sleep(5)
        # ser.send_data("*90")
        # time.sleep(5)
