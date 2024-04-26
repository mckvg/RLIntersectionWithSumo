import datetime



def datatime_to_moy_timestamp(vehicle_posterior_data):
    # 转换每个时间戳的前5个元素为分钟数
    minutes_passed_list = [
        (datetime.datetime(*timestamp[:5]) - datetime.datetime(timestamp[0], 1, 1)).total_seconds() // 60
        for timestamp in vehicle_posterior_data
    ]

    # 转换每个时间戳的后两个元素为总毫秒数
    total_ms_list = [
        timestamp[-2] * 1000 + timestamp[-1]
        for timestamp in vehicle_posterior_data
    ]

    # 将分钟数和总毫秒数组合成子列表，并将所有子列表组合成一个列表
    vehicle_posterior_list = [
        [minutes, total_ms]
        for minutes, total_ms in zip(minutes_passed_list, total_ms_list)
    ]

    return vehicle_posterior_list

import datetime

def moy_timestamp_to_datetime(vehicle_posterior_list, year):
    output_matrix = []

    for timestamp in vehicle_posterior_list:
        # 解析分钟数和毫秒数
        minutes = timestamp[0]
        total_ms = timestamp[1]

        # 计算日期时间部分
        datetime_obj = datetime.datetime(1, 1, 1) + datetime.timedelta(minutes=minutes)

        # 提取年、月、日、小时、分钟、秒和毫秒
        decoded_time_list = [year, datetime_obj.month, datetime_obj.day,
                             datetime_obj.hour, datetime_obj.minute, total_ms // 1000, total_ms % 1000]

        # 添加到输出矩阵中
        output_matrix.append(decoded_time_list)

    return output_matrix

def moy_timestamp_to_timestamp(vehicle_moy_timestamp):
    output_matrix=[]

    for moy_timestamp in vehicle_moy_timestamp:
        minutes_ms = moy_timestamp[0]*60*1000
        total_ms = minutes_ms+moy_timestamp[1]
        output_matrix.append(total_ms)

    return output_matrix

def timestamp_to_moy_timestamp(vehicle_timestamp):
    output_matrix=[]

    for timestamp in vehicle_timestamp:
        minutes = timestamp // (60*1000)
        ms = timestamp % (60*1000)
        total = [minutes, ms]
        output_matrix.append(total)

    return output_matrix
