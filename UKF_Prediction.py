
from math import tan, sin, cos, sqrt

class UKF_Prediction:

    def __init__(self, vehicle_pre_x, vehicle_pre_y, vehicle_pre_yaw_angle,
                 vehicle_x, vehicle_y, vehicle_yaw_angle, vehicle_velocity):
        self.vehicle_pre_x = vehicle_pre_x
        self.vehicle_pre_y = vehicle_pre_y
        self.vehicle_pre_yaw_angle = vehicle_pre_yaw_angle
        self.vehicle_x = vehicle_x
        self.vehicle_y = vehicle_y
        self.vehicle_yaw_angle = vehicle_yaw_angle
        self.vehicle_velocity = vehicle_velocity


