
from math import tan, sin, cos, sqrt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
class UKF_Turn_Prediction:

    def fx(self, x, dt, dangle):
        vx = x[1]
        vy = x[3]
        x[4] = dangle
        x[1] = vx * cos(dangle) + vy * sin(dangle)
        x[3] = vy * cos(dangle) - vx * sin(dangle)
        x[0] += x[1] * dt
        x[2] += x[3] * dt

        return x


    def hx(self, x):
        return x

    def initUKF(self, ukf, vehicle_x, vehicle_velocity_x,
                vehicle_y, vehicle_velocity_y, vehicle_dangle):
       # 初始化状态
       ukf.x = np.array([vehicle_x, vehicle_velocity_x,
                         vehicle_y, vehicle_velocity_y, vehicle_dangle])
       # 初始化协方差
       ukf.P = np.diag([0.1, 0.05, 0.1, 0.05, 0.01])
       # 测量噪声协方差
       ukf.R = np.diag([0.1**2, 0.05**2, 0.1**2, 0.05**2, 0.01**2])
       # 系统噪声协方差
       ukf.Q = np.eye(5) * 0.0001



