import socket
import struct
import select
import time
import json
import ctypes
import threading

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

        # Create receive thread
        recv_thread = threading.Thread(target=recv_message, args=(client_socket,))
        recv_thread.start()

        while True:
            # Send the message to the client
            message = bytearray.fromhex("ba ab 00 00 20 00 00 00 00 00 00 00 48 00 00 00 7b 0a 09 22 41 63 74 69 6f 6e 22 3a 09 7b 0a 09 09 22 58 22 3a 09 32 2c 0a 09 09 22 59 22 3a 09 2d 31 31 38 2c 0a 09 09 22 59 61 77 41 6e 67 6c 65 22 3a 09 30 0a 09 7d 2c 0a 09 22 54 69 63 6b 49 64 22 3a 09 32 0a 7d")
            send_message(client_socket, message)

            time.sleep(1)  # Control the sending frequency, here it sends once per second

    except Exception as e:
        print("Error:", e)

    finally:
        server_socket.close()

# Run TCP server
tcp_server()
