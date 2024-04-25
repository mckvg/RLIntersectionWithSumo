import numpy as np
from Time_Data import cloud_SPAT_intervals, vehicle_posterior_intervals
from Time_Data import vehicle_posterior_data, cloud_SPAT_data, cloud_SPAT_verified_data
from Regular_Latency_BP_Network import inference_final
from Exchange_Data import datatime_to_moy_timestamp, moy_timestamp_to_datetime
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
vehicle_posterior_list = datatime_to_moy_timestamp(inference_final)
cloud_SPAT = cloud_SPAT_verified_data

print("Vehicle   Posterior:", inference_final)
print("Cloud verified SPAT:", cloud_SPAT)

# 逻辑有问题处*
for i in range(1, len(vehicle_posterior_list)):
    interval_vehicle = abs((vehicle_posterior_list[i][0]-vehicle_posterior_list[i-1][0])*60*1000+
           abs((vehicle_posterior_list[i][1]-vehicle_posterior_list[i-1][1])))
    interval_cloud = abs((cloud_SPAT[i][0]-cloud_SPAT[i-1][0])*60*1000+
           abs((cloud_SPAT[i][1]-cloud_SPAT[i-1][1])))
    if  abs(interval_vehicle-interval_cloud) > 200:
        interval_revised_vehicle = (interval_cloud - intercept) / slope
        # print(interval_revised_vehicle) 有问题
        vehicle_pre = vehicle_posterior_list[i-1][0]*60*1000+vehicle_posterior_list[i-1][1]
        vehicle_current = vehicle_pre + interval_revised_vehicle
        vehicle_posterior_list[i][0] = vehicle_current // (60*1000)
        vehicle_posterior_list[i][1] = vehicle_current % (60*1000)

final_vehicle_data = moy_timestamp_to_datetime(vehicle_posterior_list, cloud_SPAT[0][0])
print("Vehicle     Revised:", final_vehicle_data)
