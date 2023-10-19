
from math import tan, sin, cos, sqrt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
class UKF_Prediction:

    def fx(self, x, dt, v):
        x[2] = x[2]
        x[0] += sin(x[2]) * v * dt
        x[1] += cos(x[1]) * v * dt
        return x

    def hx(self, x):
        return x

    def initUKF(self, ukf, vehicle_x, vehicle_y, vehicle_yaw_angle):
       # 初始化状态
       ukf.x = np.array([vehicle_x, vehicle_y, vehicle_yaw_angle])
       # 初始化协方差
       ukf.P = np.diag([0.1, 0.1, 0.05])
       # 测量噪声协方差
       ukf.R = np.diag([0.1**2, 0.1**2, 0.05**2])
       # 系统噪声协方差
       ukf.Q = np.eye(3) * 0.0001



