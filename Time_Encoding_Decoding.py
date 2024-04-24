import numpy as np
import datetime

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def time_encoding(data):
    sin_minutes, cos_minutes = year_minutes_encoding(data[0], data[1], data[2], data[3], data[4])
    sin_ms, cos_ms = current_ms_encoding(data[5], data[6])
    return sin_minutes, cos_minutes, sin_ms, cos_ms

def year_minutes_encoding(year, month, day, hour, minute):
    # 获取当前时间
    current_time = datetime.datetime(year, month, day, hour, minute)
    # 计算当前时间距离年初的分钟数
    year_start = datetime.datetime(year, 1, 1)
    minutes_passed = (current_time - year_start).total_seconds() // 60
    if is_leap_year(year):
        normalized_minutes = minutes_passed / (366 * 24 * 60)
    else:
        normalized_minutes = minutes_passed / (365 * 24 * 60)
    # 对分钟数进行归一化到[0, 1)范围内
    # 使用三角函数编码
    sin_minutes = np.sin(2 * np.pi * normalized_minutes)
    cos_minutes = np.cos(2 * np.pi * normalized_minutes)
    return sin_minutes, cos_minutes

def current_ms_encoding(second, millisecond):
    # 计算当前秒已经过去的毫秒数
    total_ms = second * 1000 + millisecond
    # 对毫秒数进行归一化到[0, 1)范围内
    normalized_ms = total_ms / (60 * 1000)
    # 使用三角函数编码
    sin_ms = np.sin(2 * np.pi * normalized_ms)
    cos_ms = np.cos(2 * np.pi * normalized_ms)
    return sin_ms, cos_ms

def time_decoding(data, year):
    decoded_time = year_minutes_decoding(data[0], data[1], year)
    decoded_second, decoded_millisecond = current_ms_decoding(data[2], data[3])
    return decoded_time, decoded_second, decoded_millisecond

def year_minutes_decoding(sin_minutes, cos_minutes, year):
    # 使用反三角函数解码分钟数
    normalized_minutes = np.arctan2(sin_minutes, cos_minutes) / (2 * np.pi)
    if normalized_minutes < 0:
        normalized_minutes += 1
    # 计算对应的分钟数
    if is_leap_year(year):
        days_in_year = 366
    else:
        days_in_year = 365
    total_minutes = normalized_minutes * (days_in_year * 24 * 60)

    # 将张量转换为标量值
    total_minutes_scalar = total_minutes.item()

    # 计算对应的日期时间
    year_start = datetime.datetime(year, 1, 1)
    decoded_time = year_start + datetime.timedelta(minutes=total_minutes_scalar)
    # 对超过一年的部分进行修正
    if decoded_time.year != year:
        remaining_minutes = total_minutes - (days_in_year * 24 * 60)
        decoded_time = year_start.replace(year=decoded_time.year) + datetime.timedelta(minutes=remaining_minutes)
    return decoded_time

def current_ms_decoding(sin_ms, cos_ms):
    # 使用反三角函数解码毫秒数
    normalized_ms = np.arctan2(sin_ms, cos_ms) / (2 * np.pi)
    if normalized_ms < 0:
        normalized_ms += 1
    # 计算对应的毫秒数
    total_ms = normalized_ms * (60 * 1000)
    # 计算对应的秒和毫秒
    decoded_second = int(total_ms // 1000)
    decoded_millisecond = int(total_ms % 1000)
    return decoded_second, decoded_millisecond

# # 输入时间
# year = 2024
# month = 1
# day = 31
# hour = 23
# minute = 59
# second = 59
# millisecond = 345
#
# # 真值时间
# true_year = 2024
# true_month = 2
# true_day = 1
# true_hour = 0
# true_minute = 0
# true_second = 0
# true_millisecond = 776
#
# # 编码输入时间
# sin_minutes_input, cos_minutes_input = year_minutes_encoding(year, month, day, hour, minute)
# sin_ms_input, cos_ms_input = current_ms_encoding(second, millisecond)
#
# # 编码真值时间
# sin_minutes_true, cos_minutes_true = year_minutes_encoding(true_year, true_month, true_day, true_hour, true_minute)
# sin_ms_true, cos_ms_true = current_ms_encoding(true_second, true_millisecond)
#
# # 解码输出时间
# decoded_minutes = year_minutes_decoding(sin_minutes_input, cos_minutes_input, year)
# decoded_second, decoded_millisecond = current_ms_decoding(sin_ms_input, cos_ms_input)
#
# # 解码真值时间
# decoded_minutes_true = year_minutes_decoding(sin_minutes_true, cos_minutes_true, true_year)
# decoded_second_true, decoded_millisecond_true = current_ms_decoding(sin_ms_true, cos_ms_true)
#
# # 对比编码解码结果
# print("Input Encoding (Year Minutes Sin):", sin_minutes_input)
# print("Input Encoding (Year Minutes Cos):", cos_minutes_input)
# print("Input Encoding (Current Milliseconds Sin):", sin_ms_input)
# print("Input Encoding (Current Milliseconds Cos):", cos_ms_input)
# #
# print("\nTrue Encoding (Year Minutes Sin):", sin_minutes_true)
# print("True Encoding (Year Minutes Cos):", cos_minutes_true)
# print("True Encoding (Current Milliseconds Sin):", sin_ms_true)
# print("True Encoding (Current Milliseconds Cos):", cos_ms_true)
#
# print("\nDecoded Year Minutes:", decoded_minutes)
# print("Decoded Current Time:")
# print("Second:", decoded_second)
# print("Millisecond:", decoded_millisecond)
#
# print("\nDecoded True Year Minutes:", decoded_minutes_true)
# print("Decoded True Current Time:")
# print("True Second:", decoded_second_true)
# print("True Millisecond:", decoded_millisecond_true)