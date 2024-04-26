import numpy as np
from Time_Data import cloud_SPAT_intervals, vehicle_posterior_intervals
from Time_Data import vehicle_posterior_data, cloud_SPAT_data, cloud_SPAT_verified_data
from Regular_Latency_BP_Network import inference_end_datetime
from Exchange_Data import datatime_to_moy_timestamp, moy_timestamp_to_datetime,\
    moy_timestamp_to_timestamp, timestamp_to_moy_timestamp
import datetime


# 执行最小二乘法拟合
coefficients = np.polyfit(cloud_SPAT_intervals, vehicle_posterior_intervals, 1)

# 提取斜率和截距
slope = coefficients[0]
intercept = coefficients[1]

# 打印拟合结果
print("斜率:", slope)
print("截距:", intercept)

# irregular latency数据集
# vehicle_posterior_list = datatime_to_moy_timestamp(vehicle_posterior_data)
# could_SPAT = cloud_SPAT_data

# 结合 regular BP network的后验预测值
vehicle_posterior_list = datatime_to_moy_timestamp(inference_end_datetime)
cloud_SPAT = datatime_to_moy_timestamp(cloud_SPAT_verified_data)

print("Vehicle   Posterior:", inference_end_datetime)
print("Cloud verified SPAT:", cloud_SPAT_verified_data)

vehicle_posterior_list_timestamp = moy_timestamp_to_timestamp(vehicle_posterior_list)
cloud_SPAT_timestamp = moy_timestamp_to_timestamp(cloud_SPAT)

# 利用车辆时间与云端时间的时间间隔对车辆时间的后验进行检验
for i in range(1, len(vehicle_posterior_list_timestamp)):
    interval_vehicle = abs(vehicle_posterior_list_timestamp[i]-vehicle_posterior_list_timestamp[i-1])
    interval_cloud = abs(cloud_SPAT_timestamp[i]-cloud_SPAT_timestamp[i-1])
    if  abs(interval_vehicle-interval_cloud) > 300:
        interval_revised_vehicle = (interval_cloud - intercept) / slope
        vehicle_pre = vehicle_posterior_list_timestamp[i-1]
        vehicle_current = vehicle_pre + interval_revised_vehicle
        vehicle_posterior_list_timestamp[i] = vehicle_current

vehicle_posterior_revised_moy_timestamp = timestamp_to_moy_timestamp(vehicle_posterior_list_timestamp)

final_vehicle_data = moy_timestamp_to_datetime(vehicle_posterior_revised_moy_timestamp, cloud_SPAT_verified_data[0][0])
print("Vehicle     Revised:", final_vehicle_data)
