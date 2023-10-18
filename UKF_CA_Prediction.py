
from math import tan, sin, cos, sqrt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
class UKF_CA_Prediction:

    def fx(self, x, dt, ax, ay):
        x[2] = ax
        x[1] += x[2] * dt
        x[0] += x[1] * dt
        x[5] = ay
        x[4] += x[5] * dt
        x[3] += x[4] * dt
        return x

    # def fx(self):
    #     self.predicted_yaw_angle = 2 * self.vehicle_yaw_angle - self.vehicle_pre_yaw_angle
    #     self.predicted_x += sin(self.predicted_yaw_angle) * self.vehicle_velocity
    #     self.predicted_y += cos(self.predicted_yaw_angle) * self.vehicle_velocity
    #     return self.predicted_x, self.predicted_y, self.predicted_yaw_angle

    def hx(self, x):
        return x

    def initUKF(self, ukf, vehicle_x, vehicle_velocity_x, vehicle_acceleration_x,
                vehicle_y, vehicle_velocity_y, vehicle_acceleration_y):
       # 初始化状态
       ukf.x = np.array([vehicle_x, vehicle_velocity_x, vehicle_acceleration_x,
                         vehicle_y, vehicle_velocity_y, vehicle_acceleration_y])
       # 初始化协方差
       ukf.P = np.diag([0.1, 0.05, 0.01, 0.1, 0.05, 0.01])
       # 测量噪声协方差
       ukf.R = np.diag([0.1**2, 0.05**2, 0.01**2, 0.1**2, 0.05**2, 0.01**2])
       # 系统噪声协方差
       ukf.Q = np.eye(6) * 0.0001



