import numpy as np
import torch
from torch import nn
from Time_Encoding_Decoding import time_encoding, time_decoding, time_encoding_matrix, time_decoding_matrix
from Exchange_Data import datatime_to_moy_timestamp, moy_timestamp_to_timestamp,\
    timestamp_to_moy_timestamp, moy_timestamp_to_datetime
from Time_Data import vehicle_data, cloud_data, vehicle_inference_data

def z_score_normalize(data):
    mean = torch.mean(data, dim=0)  # 计算每列的平均值
    epsilon = 1e-8  # 很小的正则化项
    std = torch.std(data, dim=0) + epsilon  # 计算每列的标准差并添加正则化项
    normalized_data = (data - mean) / std  # 均值-标准差标准化
    return normalized_data, mean, std

def inverse_z_score_normalize(normalized_data, mean, std):
    # 检查标准差是否为0
    zero_std_mask = (std == 0)

    # 对标准差为0的特征进行处理
    original_data = torch.where(zero_std_mask, normalized_data, normalized_data * std + mean)
    return original_data


raw_data = time_encoding_matrix(vehicle_data)
raw_label = time_encoding_matrix(cloud_data)
tensor_raw_data = torch.tensor(raw_data, dtype=torch.float64)
tensor_raw_label = torch.tensor(raw_label, dtype=torch.float64)

# 沿着指定维度标准化数据（这里假设数据在维度1上）
# nn.functional.normalize标准化
# normalized_data = nn.functional.normalize(tensor_raw_data, p=1, dim=1)
# normalized_label = nn.functional.normalize(tensor_raw_label, p=1, dim=1)
# z_score_标准化
normalized_data, data_mean, data_std = z_score_normalize(tensor_raw_data)
normalized_label, label_mean, label_std = z_score_normalize(tensor_raw_label)

bp = nn.Sequential(nn.Linear(4, 10), nn.Tanh(), nn.Linear(10, 6), nn.Tanh(), nn.Linear(6, 4), nn.Tanh()).double()
Loss = nn.MSELoss()
optim = torch.optim.SGD(params=bp.parameters(), lr=0.1,
                        # weight_decay=0.1
                        )  # 参数更新方法，随机梯度下降

for i in range(100):
    yp = bp(normalized_data)  # 前向传递的预测值
    loss = Loss(yp, normalized_label)  # 预测值与实际值的差别
    optim.zero_grad()
    loss.backward()  # 反向传递
    optim.step()  # 更新参数

# 准备推断数据
inference_data = time_encoding_matrix(vehicle_inference_data)  # 假设inference_vehicle_data为待推断的车辆数据
tensor_inference_data = torch.tensor(inference_data, dtype=torch.float64)

print("vehicle_inference_data:", vehicle_inference_data)
# z_score_标准化
normalized_inference_data, inference_data_mean, inference_data_std = z_score_normalize(tensor_inference_data)
# nn.functional.normalize标准化
# normalized_inference_data = nn.functional.normalize(tensor_inference_data, p=1, dim=1)
# inference_data_mean = torch.mean(normalized_inference_data, dim=0)
# inference_data_std = torch.std(normalized_inference_data, dim=0)

# 使用模型进行预测
with torch.no_grad():  # 禁用梯度计算，因为在推断阶段不需要计算梯度
    inference_result = bp(normalized_inference_data)

# 对标准化后的数据进行逆操作
# z_score_去标准化
original_data = inverse_z_score_normalize(inference_result, inference_data_mean, inference_data_std)
# nn.functional.normalize去标准化
# original_data = inference_result * inference_data_std + inference_data_mean

# 使用解码函数预测结果
inference_final = time_decoding_matrix(original_data, cloud_data[0][0])

# 打印推断结果
print("Inference Result:", inference_final)

vehicle_inference_data_moy_timestamp = datatime_to_moy_timestamp(vehicle_inference_data)
inference_final_moy_timestamp = datatime_to_moy_timestamp(inference_final)

vehicle_inference_data_timestamp = moy_timestamp_to_timestamp(vehicle_inference_data_moy_timestamp)
inference_final_timestamp = moy_timestamp_to_timestamp(inference_final_moy_timestamp)

# 将车辆时间后验与车辆先验时间进行对比，若差值（延迟）大于1s（通信时间延迟），则后验不可信，返回先验数据
inference_end_timestamp = []
for i in range(len(vehicle_inference_data_timestamp)):
    if abs(vehicle_inference_data_timestamp[i]-inference_final_timestamp[i])>1000:
        inference_end_timestamp.append(vehicle_inference_data_timestamp[i])
    else:
        inference_end_timestamp.append(inference_final_timestamp[i])

inference_end_moy_timestamp = timestamp_to_moy_timestamp(inference_end_timestamp)

inference_end_datetime = moy_timestamp_to_datetime(inference_end_moy_timestamp, vehicle_inference_data[0][0])

print("inference_end_datetime:", inference_end_datetime)

# print("data:")
# for row in tensor_raw_data:
#     print(["{:.7f}".format(item) for item in row])
#
# print("\nlabel:")
# for row in tensor_raw_label:
#     print(["{:.7f}".format(item) for item in row])
#
# print("Normalized data:")
# for row in normalized_data:
#     print(["{:.7f}".format(item) for item in row])
#
# print("\nNormalized label:")
# for row in normalized_label:
#     print(["{:.7f}".format(item) for item in row])
