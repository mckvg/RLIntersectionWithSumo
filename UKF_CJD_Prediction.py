
from math import tan, sin, cos, sqrt, atan2
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
class UKF_CJD_Prediction:

    def __init__(self):
        self.jerk = -4

    def reset(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.angle = atan2(self.vx, self.vy)

    def fx(self, x, dt):
        x[1] += self.vx + 0.5 * self.jerk * sin(self.angle) * dt
        x[0] += x[1] * dt
        x[3] += self.vy + 0.5 * self.jerk * cos(self.angle) * dt
        x[2] += x[3] * dt
        return x

    def hx(self, x):
        return x

    def initUKF(self, ukf, vehicle_x, vehicle_velocity_x, vehicle_y, vehicle_velocity_y):
       # 初始化状态
       ukf.x = np.array([vehicle_x, vehicle_velocity_x, vehicle_y, vehicle_velocity_y])
       # 初始化协方差
       ukf.P = np.diag([0.1, 0.05, 0.1, 0.05])
       # 测量噪声协方差
       ukf.R = np.diag([0.1**2, 0.05**2, 0.1**2, 0.05**2])
       # 系统噪声协方差
       ukf.Q = np.eye(4) * 0.0001



