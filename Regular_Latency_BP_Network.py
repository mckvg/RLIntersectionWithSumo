import numpy as np
import torch
from torch import nn
from Time_Encoding_Decoding import time_encoding, time_decoding
from Time_Data import vehicle_data, cloud_data

data = time_encoding(vehicle_data)
label = time_encoding(cloud_data)
data = torch.tensor(data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.float32)

bp = nn.Sequential(nn.Linear(4, 10), nn.Tanh(), nn.Linear(10, 6), nn.Tanh(), nn.Linear(6, 4), nn.Tanh())
Loss = nn.MSELoss()
optim = torch.optim.SGD(params=bp.parameters(), lr=0.1)  # 参数更新方法，随机梯度下降

for i in range(100):
    yp = bp(data)  # 前向传递的预测值
    loss = Loss(yp, label)  # 预测值与实际值的差别
    optim.zero_grad()
    loss.backward()  # 反向传递
    optim.step()  # 更新参数

# 准备推断数据
inference_data = time_encoding(vehicle_data)  # 假设inference_vehicle_data为待推断的车辆数据
inference_data = torch.tensor(inference_data, dtype=torch.float32)

# 使用模型进行预测
with torch.no_grad():  # 禁用梯度计算，因为在推断阶段不需要计算梯度
    inference_result = bp(inference_data)

# 使用解码函数预测结果
inference_final = time_decoding(inference_result, cloud_data[0])

# 打印推断结果
print("Inference Result:", inference_final)
