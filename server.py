import socket
import struct
import select
import time
import json
import ctypes


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


def recv_message(client_socket):
    recv_buf = bytearray(2048)
    temp_buf = bytearray(1024)
    recv_buf_data_len = 0

    while True:
        l_fdset = select.select([client_socket], [], [], 0)[0]
        if client_socket in l_fdset:
            temp_buf = bytearray(1024)
            recv_count = client_socket.recv_into(temp_buf)
            if recv_count > 0:
                recv_buf[recv_buf_data_len:recv_buf_data_len + recv_count] = temp_buf[:recv_count]
                recv_buf_data_len += recv_count

                # Process complete messages
                while True:
                    # Find the start index of a message
                    start_index = recv_buf.find(b'\xba\xab')
                    if start_index == -1:
                        break

                    # Find the end index of the message
                    if recv_buf_data_len - start_index < ctypes.sizeof(FordRLModelHeader):
                        break

                    header_data = recv_buf[start_index:start_index + ctypes.sizeof(FordRLModelHeader)]
                    header = FordRLModelHeader.from_buffer_copy(header_data)
                    payload_size = header.payload_size

                    if recv_buf_data_len - start_index < ctypes.sizeof(FordRLModelHeader) + payload_size:
                        break

                    end_index = start_index + ctypes.sizeof(FordRLModelHeader) + payload_size
                    message = recv_buf[start_index:end_index]
                    print("Received message:", message)

                    # Process the received message here
                    payload_data = recv_buf[start_index + ctypes.sizeof(FordRLModelHeader):end_index]
                    payload_json = payload_data.decode('utf-8')
                    payload_dict = json.loads(payload_json)
                    print("Payload dict:", payload_dict)

                    # Clear the processed message from the buffer
                    temp_buf = bytearray(1024)
                    temp_buf[:recv_buf_data_len - end_index] = recv_buf[end_index:recv_buf_data_len]
                    recv_buf = temp_buf
                    recv_buf_data_len -= end_index

        time.sleep(0.5)


def send_message(client_socket, message):
    sent_count = client_socket.send(message)
    print("Sent:", sent_count, "bytes")


# TCP server
def tcp_server():
    server_address = ('127.0.0.1', 8080)  # Server address and port
    try:
        # Create TCP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(server_address)
        server_socket.listen(1)  # Listen for client connections

        print("Waiting for client to connect...")
        client_socket, client_address = server_socket.accept()
        print("Client connected:", client_address)

        while True:
            # Receive messages from the client
            recv_message(client_socket)

            # Send a message to the client
            message = bytearray.fromhex(
                "ba ab 00 00 10 00 00 00 00 00 00 00 bd 00 00 00 7b 0a 09 22 52 65 6d 6f 74 65 56 65 68 69 63 6c 65 73 22 3a 09 5b 7b 0a 09 09 09 22 49 64 22 3a 09 31 2c 0a 09 09 09 22 58 22 3a 09 32 2c 0a 09 09 09 22 59 22 3a 09 2d 31 32 30 2c 0a 09 09 09 22 59 61 77 41 6e 67 6c 65 22 3a 09 30 2c 0a 09 09 09 22 53 70 65 65 64 22 3a 09 32 0a 09 09 7d 2c 20 7b 0a 09 09 09 22 49 64 22 3a 09 32 2c 0a 09 09 09 22 58 22 3a 09 31 30 2c 0a 09 09 09 22 59 22 3a 09 31 38 2c 0a 09 09 09 22 59 61 77 41 6e 67 6c 65 22 3a 09 35 30 2c 0a 09 09 09 22 53 70 65 65 64 22 3a 09 34 0a 09 09 7d 5d 2c 0a 09 22 54 69 63 6b 49 64 22 3a 09 32 0a 7d ")

            if client_socket is not None:
                send_message(client_socket, message)

            time.sleep(1)  # Control the sending frequency, here it sends once per second

    except Exception as e:
        print("Error:", e)

    finally:
        if client_socket is not None:
            client_socket.close()
        server_socket.close()


# Run TCP server
tcp_server()

