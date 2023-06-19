import socket
import struct
import select
import time
import ctypes
import json
from readcsv import *

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
    def start_server():
        server_address = ('', 9030)  # 监听所有可用的接口
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(server_address)
        server_socket.listen(5)

        print("Server started. Waiting for connections...")

        while True:
            # 监听可读事件
            readable, _, _ = select.select([server_socket] + Server.client_sockets, [], [])

            for sock in readable:
                if sock is server_socket:
                    # 接受新连接
                    client_socket, client_address = sock.accept()
                    Server.client_sockets.append(client_socket)
                    Server.Step = 0
                    print("New client connected:", client_address)
                else:
                    # 处理已连接客户端发送的数据
                    received_data = Server.receive_data(sock)
                    Server.send_data(sock, Server.Step)
                    Server.Step += 1


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

            while recv_buf_data_len >= (ctypes.sizeof(Server.FordRLModelHeader) + payload_size):
                header = Server.FordRLModelHeader.from_buffer(recv_buf)
                if header.type != 3:
                    payload_size = header.payload_size
                else:
                    payload_size = 0
                print("server_recv_cmd:", header.cmd)
                if header.cmd == 0x20:
                    payload_data = recv_buf[
                                   ctypes.sizeof(Server.FordRLModelHeader):ctypes.sizeof(
                                       Server.FordRLModelHeader) + payload_size]
                    payload_dict = json.loads(payload_data)
                    # Process the payload data here
                    print("Server Received Payload dict:", payload_dict)
                    print(int(payload_dict['TickId']))

                recv_buf_data_len -= (ctypes.sizeof(Server.FordRLModelHeader) + payload_size)
                temp_buf = bytearray(1024)
                temp_buf[:recv_buf_data_len] = recv_buf[ctypes.sizeof(Server.FordRLModelHeader) +
                                                        payload_size:recv_buf_data_len +
                                                        ctypes.sizeof(Server.FordRLModelHeader) + payload_size]
                recv_buf = temp_buf

                print("recv_buf_data_len:", recv_buf_data_len)


    @staticmethod
    def send_data(client_socket, TickId):

        RV_data = float_csv_data[TickId]

        data = {
                    "TickId": int(RV_data[0]),
                    "RemoteVehicles": [{
                            "Id": int(RV_data[1]),
                            "X": RV_data[2],
                            "Y": RV_data[3],
                            "YawAngle": RV_data[4],
                            "Speed": RV_data[5]
                        },
                        {
                            "Id": int(RV_data[6]),
                            "X": RV_data[7],
                            "Y": RV_data[8],
                            "YawAngle": RV_data[9],
                            "Speed": RV_data[10]
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

        # 发送数据包头部
        client_socket.sendall(header)

        # 发送 payload 数据
        client_socket.sendall(payload_json.encode('utf-8'))

        # print
        print("server send data:", payload_json)


# 启动服务器
Server.start_server()






