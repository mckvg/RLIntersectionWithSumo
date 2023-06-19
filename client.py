import socket
import struct
import select
import time
import ctypes
import json

class client():

    Remotes = []
    client_socket = None
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

    # 接收数据函数
    @staticmethod
    def receive_data(socket_num):
        recv_buf = bytearray(2048)
        temp_buf = bytearray(1024)
        recv_buf_data_len = 0

        # while True:
        # l_fdset = select.select([socket_num], [], [], 0)[0]
        # if socket_num in l_fdset:
        temp_buf = bytearray(1024)
        recv_count = socket_num.recv_into(temp_buf)
        print("client_recv_count:", recv_count)
        if recv_count > 0:
            recv_buf[recv_buf_data_len:recv_buf_data_len + recv_count] = temp_buf[:recv_count]
            recv_buf_data_len += recv_count
            header = client.FordRLModelHeader.from_buffer(recv_buf)
            payload_size = 0
            if header.type != 3:
                payload_size = header.payload_size

            while recv_buf_data_len >= (ctypes.sizeof(client.FordRLModelHeader) + payload_size):
                header = client.FordRLModelHeader.from_buffer(recv_buf)
                if header.type != 3:
                    payload_size = header.payload_size
                else:
                    payload_size = 0
                print("client_recv_cmd:", header.cmd)
                if header.cmd == 0x10:
                    payload_data = recv_buf[
                                   ctypes.sizeof(client.FordRLModelHeader):ctypes.sizeof(
                                       client.FordRLModelHeader) + payload_size]
                    payload_dict = json.loads(payload_data)
                    # Process the payload data here
                    print("Client Received Payload dict:", payload_dict)
                    for item in payload_dict['RemoteVehicles']:
                        existing_item = next((r for r in client.Remotes if r['Id'] == item['Id']), None)
                        if existing_item:
                            existing_item.update(item)
                        else:
                            client.Remotes.append(item)
                    # print('Remotes:', client.Remotes)

                recv_buf_data_len -= (ctypes.sizeof(client.FordRLModelHeader) + payload_size)
                temp_buf = bytearray(1024)
                temp_buf[:recv_buf_data_len] = recv_buf[ctypes.sizeof(client.FordRLModelHeader) +
                                                        payload_size:recv_buf_data_len +
                                                                     ctypes.sizeof(
                                                                         client.FordRLModelHeader) + payload_size]
                recv_buf = temp_buf


    # 发送数据函数
    @staticmethod
    def send_data(socket_num, data):

        # 将数据转换为JSON格式
        payload = {
            "TickId": data['TickId'],
            "Action": {
                "X": data['X'],
                "Y": data['Y'],
                "YawAngle": data['YawAngle']
            }
        }
        payload_json = json.dumps(payload)

        # 构建数据包头部
        header = client.FordRLModelHeader(
            tag=0xabba,
            type=0,
            cmd=0x20,
            from_id=0,
            callback_id=0,
            payload_size=len(payload_json)
        )

        header_bytes = ctypes.string_at ( ctypes.addressof ( header ), ctypes.sizeof ( header ) )
        payload_send = payload_json.encode('utf-8')
        total_data = header_bytes + payload_send

        # 发送数据
        socket_num.sendall(total_data)

        # print
        print("client send data:", payload_json)

    # TCP客户端
    @staticmethod
    def tcp_client():
        server_address = ('127.0.0.1', 9030)  # 替换为实际的服务器地址
        # try:
        # 建立TCP套接字
        client.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.client_socket.connect(server_address)
        print("Connected to server")
