
from math import tan, sin, cos, sqrt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
class UKF_CL_Prediction:

    def fx(self, x, dt):
        x[0] = x[0]
        x[1] = x[1]
        return x

    def hx(self, x):
        return x

    def initUKF(self, ukf, vehicle_x, vehicle_y):
       # 初始化状态
       ukf.x = np.array([vehicle_x, vehicle_y])
       # 初始化协方差
       ukf.P = np.diag([0.1, 0.1])
       # 测量噪声协方差
       ukf.R = np.diag([0.1**2, 0.1**2])
       # 系统噪声协方差
       ukf.Q = np.eye(2) * 0.0001



