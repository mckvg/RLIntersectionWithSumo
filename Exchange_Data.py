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

