# 引入包
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from datetime import datetime, date
import os, sys
import sumolib
import time
import socket
import struct
import select
import time
import ctypes
import json


# -----引入地址------
sumo_path = "D:/Sumo"
project_path = "C:/Users/T470/PycharmProjects/pythonSumoRL/RLIntersectionWithSumo"
cfg_path = "C:/Users/T470/PycharmProjects/pythonSumoRL/RLIntersectionWithSumo/sumo.cfg"
# ----------------------------------------#

# ---------------------#
sys.path.append(sumo_path)
sys.path.append(sumo_path + "tools")

# 引入地址导入模块
sys.path.append(sumo_path + "/tools/xml")
import traci
from sumolib import checkBinary

RemoteVehiclePosition = {}
RemoteVehicleSpeed = {}
RemoteVehicleYawAngle = {}
SimulationStep = 0

def RVData(id):
    RemoteVehiclePosition[id] = traci.vehicle.getPosition(id)
    RemoteVehicleSpeed[id] = traci.vehicle.getSpeed(id)
    RemoteVehicleYawAngle[id] = traci.vehicle.getAngle(id)


class Server:

    Remotes = []
    server_socket = None
    client_sockets = []
    Step = 0

    # 定义结构体
    class FordRLModelHeader(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("tag", ctypes.c_uint16),
            ("type", ctypes.c_uint16),
            ("cmd", ctypes.c_uint16),
            ("from_id", ctypes.c_uint16),
            ("callback_id", ctypes.c_uint32),
            ("payload_size", ctypes.c_uint32)
        ]

    @staticmethod
    def receive_data(client_socket):
        recv_buf = bytearray(2048)
        recv_buf_data_len = 0
        temp_buf = bytearray(1024)
        recv_count = client_socket.recv_into(temp_buf)
        print("server_recv_count:", recv_count)
        if recv_count > 0:
            recv_buf[recv_buf_data_len:recv_buf_data_len + recv_count] = temp_buf[:recv_count]
            recv_buf_data_len += recv_count
            header = Server.FordRLModelHeader.from_buffer(recv_buf)
            payload_size = 0
            if header.type != 3:
                payload_size = header.payload_size
            # print('payload_size:',payload_size)

            while recv_buf_data_len >= (ctypes.sizeof(Server.FordRLModelHeader) + payload_size):
                header = Server.FordRLModelHeader.from_buffer(recv_buf)
                if header.type != 3:
                    payload_size = header.payload_size
                else:
                    payload_size = 0
                # print("server_recv_cmd:", header.cmd)
                if header.cmd == 0x20:
                    payload_data = recv_buf[
                                   ctypes.sizeof(Server.FordRLModelHeader):ctypes.sizeof(
                                       Server.FordRLModelHeader) + payload_size]
                    payload_dict = json.loads(payload_data)
                    # Process the payload data here
                    print("Server Received Payload dict:", payload_dict)
                    # print(int(payload_dict['TickId']))
                    Server.Remotes = [payload_dict['Action']]
                    print('Remotes:', Server.Remotes)

                recv_buf_data_len -= (ctypes.sizeof(Server.FordRLModelHeader) + payload_size)
                temp_buf = bytearray(1024)
                temp_buf[:recv_buf_data_len] = recv_buf[ctypes.sizeof(Server.FordRLModelHeader) +
                                                        payload_size:recv_buf_data_len +
                                                        ctypes.sizeof(Server.FordRLModelHeader) + payload_size]
                recv_buf = temp_buf
                return True


    @staticmethod
    def send_data(client_socket):

        data = {
                    "TickId": int(SimulationStep),
                    "RemoteVehicles": [{
                            "Id": int('1'),
                            "X": RemoteVehiclePosition['1'][0],
                            "Y": RemoteVehiclePosition['1'][1],
                            "YawAngle": RemoteVehicleYawAngle['1'],
                            "Speed": RemoteVehicleSpeed['1']
                        },
                        {
                            "Id": int('2'),
                            "X": RemoteVehiclePosition['2'][0],
                            "Y": RemoteVehiclePosition['2'][1],
                            "YawAngle": RemoteVehicleYawAngle['2'],
                            "Speed": RemoteVehicleSpeed['2']
                        }
                    ]
                }

        payload_json = json.dumps(data)

        header = Server.FordRLModelHeader(
            tag=0xabba,
            type=0,
            cmd=0x10,
            from_id=0,
            callback_id=0,
            payload_size=len(payload_json)
        )

        header_bytes = ctypes.string_at ( ctypes.addressof ( header ), ctypes.sizeof ( header ) )
        payload_send = payload_json.encode('utf-8')
        total_data = header_bytes + payload_send

        # 发送数据
        client_socket.sendall(total_data)

        # print
        print("server send data:", payload_json)


def run(step):

    # 步长控制
    vehicle_id0 = '0'
    vehicle_id1 = '1'
    vehicle_id2 = '2'
    vehicle_id3 = '3'
    vehicle_id4 = '4'
    vehicle_id5 = '5'
    vehicle_id6 = '6'
    vehicle_id7 = '7'
    vehicle_id8 = '8'
    vehicle_id9 = '9'
    vehicle_id10 = '10'

    # 匹配RL与SUMO的主车的航向角
    Server.Remotes[0]['YawAngle'] = 180/math.pi*Server.Remotes[0]['YawAngle']
    if Server.Remotes[0]['YawAngle'] < 0:
        Server.Remotes[0]['YawAngle'] = Server.Remotes[0]['YawAngle'] + 360
    # SUMO的主车的移动
    traci.vehicle.moveToXY('0', '-E3', '1', x=Server.Remotes[0]['X'], y=Server.Remotes[0]['Y'],
                           angle=Server.Remotes[0]['YawAngle'], keepRoute=2)
    # SUMO中远车的初始位置及状态的设置
    traci.vehicle.moveToXY('10', '-E3', 1, 2, -24, 0)  # 停止的车辆与主车同道
    if step == 0:
        # 运动车辆与主车同方向
        traci.vehicle.moveToXY('1', '-E3', 0, 6, -115, 0)
        traci.vehicle.moveToXY('2', '-E3', 0, 6, -100, 0)
        # 运动车辆与主车方向垂直
        traci.vehicle.moveToXY('3', '-E0', 1, 40, 2, -90)
        traci.vehicle.moveToXY('4', '-E2', 1, -54, -2, 90)
        traci.vehicle.moveToXY('5', '-E0', 0, 80, 6, -90)
        traci.vehicle.moveToXY('6', '-E2', 0, -100, -6, 90)
        # 运动车辆与主车方向相反
        traci.vehicle.moveToXY('7', '-E1', 1, -2, 30, 180)
        traci.vehicle.moveToXY('8', '-E1', 0, -6, 60, 180)
        traci.vehicle.moveToXY('9', '-E1', 0, -6, 106, 180)
        # 设置速度：运动车辆与主车同方向
        traci.vehicle.setSpeed(vehicle_id1, 2)
        traci.vehicle.setSpeed(vehicle_id2, 8)
        # 设置速度：运动车辆与主车方向垂直
        traci.vehicle.setSpeed(vehicle_id3, 6)
        traci.vehicle.setSpeed(vehicle_id4, 8)
        traci.vehicle.setSpeed(vehicle_id5, 6)
        traci.vehicle.setSpeed(vehicle_id6, 8)
        # 设置速度：运动车辆与主车方向相反
        traci.vehicle.setSpeed(vehicle_id7, 8)
        traci.vehicle.setSpeed(vehicle_id8, 8)
        traci.vehicle.setSpeed(vehicle_id9, 4)

    # SUMO中模拟一步
    traci.simulationStep()
    # SUMO中更新远车位置
    if vehicle_id1 in traci.vehicle.getIDList():
        RVData(vehicle_id1)
        print('Step：', step, "X:", RemoteVehiclePosition['1'][0],
                            "Y:", RemoteVehiclePosition['1'][1],
                            "YawAngle:", RemoteVehicleYawAngle['1'],
                            "Speed:", RemoteVehicleSpeed['1'])
    if vehicle_id2 in traci.vehicle.getIDList():
        RVData(vehicle_id2)

if __name__ == "__main__":

    N_STATES = 200

    # 是否打开gui True为打开 False为关闭
    gui = True
    if gui == 1:
        sumoBinary = checkBinary(sumo_path + "/bin/sumo-gui")
    else:
        sumoBinary = checkBinary(sumo_path + "/bin/sumo")

    # -----配置文件cfg打开以及输出xml格式统计数据
    sumoCmd = [sumoBinary, "-c", cfg_path, '--tripinfo-output', project_path + "/tripinfo2_TEST.xml",
               '--duration-log.statistics']

    server_address = ('', 9030)  # 监听所有可用的接口
    server_socket = socket.socket ( socket.AF_INET, socket.SOCK_STREAM )
    server_socket.bind ( server_address )
    server_socket.listen ( 5 )

    print ( "Server started. Waiting for connections..." )

    # ----开始---
    traci.start(sumoCmd)
    while SimulationStep <= N_STATES:
        # 监听可读事件
        readable, _, _ = select.select ( [server_socket] + Server.client_sockets, [], [] )
        # 接受新连接
        for sock in readable:
            if sock is server_socket:
                # 接受新连接
                client_socket, client_address = sock.accept ()
                Server.client_sockets.append ( client_socket )
                Server.empty_data_count = 0  # 初始化空数据计数器
                print ( "New client connected:", client_address )
            else:
                # 处理已连接客户端发送的数据
                received_data = Server.receive_data ( sock )
                if received_data:
                    run ( SimulationStep )
                    Server.send_data ( sock )
                    SimulationStep += 1
                    Server.empty_data_count = 0  # 重置空数据计数器
                else:
                    # 空数据计数器增加
                    Server.empty_data_count += 1
                    if Server.empty_data_count >= 10:
                        # 容忍达到 10 个连续空数据，断开连接
                        print ( "Client closed the connection:", sock.getpeername () )
                        sock.close ()
                        Server.client_sockets.remove ( sock )
                        print ( "Server closed" )

    sys.stdout.flush()
    traci.close()
    server_socket.close ()

