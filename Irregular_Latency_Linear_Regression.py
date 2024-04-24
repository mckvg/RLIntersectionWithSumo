import numpy as np
from Time_Data import cloud_SPAT_intervals, vehicle_posterior_intervals
from Time_Data import vehicle_posterior_data, cloud_SPAT
from Exchange_Data import datatime_to_moy_timestamp
import datetime


# 执行最小二乘法拟合
coefficients = np.polyfit(cloud_SPAT_intervals, vehicle_posterior_intervals, 1)

# 提取斜率和截距
slope = coefficients[0]
intercept = coefficients[1]

# 打印拟合结果
print("斜率:", slope)
print("截距:", intercept)

vehicle_posterior_list = datatime_to_moy_timestamp(vehicle_posterior_data)

print("Vehicle Posterior:", vehicle_posterior_list)
print("Cloud        SPAT:", cloud_SPAT)

for i in range(1, len(vehicle_posterior_list)):
    interval_vehicle = abs((vehicle_posterior_list[i][0]-vehicle_posterior_list[i-1][0])*60*1000+
           abs((vehicle_posterior_list[i][1]-vehicle_posterior_list[i-1][1])))
    interval_cloud = abs((cloud_SPAT[i][0]-cloud_SPAT[i-1][0])*60*1000+
           abs((cloud_SPAT[i][1]-cloud_SPAT[i-1][1])))
    if  abs(interval_vehicle-interval_cloud) > 200:
        interval_revised_vehicle = (interval_cloud - intercept) / slope
        vehicle_pre = vehicle_posterior_list[i-1][0]*60*1000+vehicle_posterior_list[i-1][1]
        vehicle_current = vehicle_pre + interval_revised_vehicle
        vehicle_posterior_list[i][0] = vehicle_current // (60*1000)
        vehicle_posterior_list[i][1] = vehicle_current % (60*1000)

print("Vehicle   Revised:", vehicle_posterior_list)
