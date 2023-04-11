
###########################
#
#   @author mckvg
#   @author olin322
#
###########################

import math

from CONSTANTS import min_positionx
from CONSTANTS import max_positionx
from CONSTANTS import min_positiony
from CONSTANTS import max_positiony

# 指引agent前进行驶的Track的类,获取前沿坐标
class Track:

    def __init__(self, size):
        self.size = size
        self.judgment_str = ''
        # straight: positive_y: y+; positive_x: x+;
        self.straight_frontier_coord_positive_x = min_positionx
        self.straight_frontier_coord_positive_y = min_positiony
        # straight: negative_y: y-; negative_x: x-;
        self.straight_frontier_coord_negative_x = max_positionx
        self.straight_frontier_coord_negative_y = max_positiony
        # turn left y+: 或者说更新最大角度的极坐标值，逆时针转向，角度增大，角度从0到pi/2
        self.steering_frontier_polar_coord_positive_radius = 0.0
        self.steering_frontier_polar_coord_positive_angle = 0.0
        # turn right y+: 或者说更新最小角度的极坐标值，顺时针转向，角度减小，角度从pi到pi/2
        self.steering_frontier_polar_coord_negative_radius = 0.0
        self.steering_frontier_polar_coord_negative_angle = math.pi

    #  根据汽车在十字路口内的位置和交叉口方向选择，更新车辆在十字路口内的track的细分阶段
    def IntersectionJudgment(self, intersection_steering_choice: int):
        # intersection_steering_choice: 2: default; 0: straight_y+; -1: turn left_x-; 1: turn right_x+.
        if intersection_steering_choice == 0:
            self.judgment_str = 'intersection_straight_y+'
        elif intersection_steering_choice == -1:
            self.judgment_str = 'intersection_left_y+'
        elif intersection_steering_choice == 1:
            self.judgment_str = 'intersection_right_y+'

    # 返回智能体在十字路口内的之前动作的最大最小坐标值
    def UpdatePreExtremum(self, vehicle_max_pre_x_or_max_pre_polar_radius, vehicle_max_pre_y_or_max_pre_polar_angle,
            vehicle_min_pre_x_or_min_pre_polar_radius, vehicle_min_pre_y_or_min_pre_polar_angle,
            vehicle_pre_x_or_pre_polar_radius, vehicle_pre_y_or_pre_polar_angle,):

        if self.judgment_str == 'intersection_straight_y+':
            # 更新之前动作的最大的y值
            if vehicle_max_pre_y_or_max_pre_polar_angle <= vehicle_pre_y_or_pre_polar_angle:
                vehicle_max_pre_y_or_max_pre_polar_angle = vehicle_pre_y_or_pre_polar_angle
        elif self.judgment_str == 'intersection_left_y+':
            # 更新之前动作最大角度的极坐标值
            if vehicle_max_pre_y_or_max_pre_polar_angle <= vehicle_pre_y_or_pre_polar_angle:
                vehicle_max_pre_y_or_max_pre_polar_angle = vehicle_pre_y_or_pre_polar_angle
                vehicle_max_pre_x_or_max_pre_polar_radius = vehicle_pre_x_or_pre_polar_radius
        elif self.judgment_str == 'intersection_right_y+':
            # 更新之前动作最小角度的极坐标值
            if vehicle_min_pre_y_or_min_pre_polar_angle >= vehicle_pre_y_or_pre_polar_angle:
                vehicle_min_pre_y_or_min_pre_polar_angle = vehicle_pre_y_or_pre_polar_angle
                vehicle_min_pre_x_or_min_pre_polar_radius = vehicle_pre_x_or_pre_polar_radius

        # 返回智能体之前动作的最大最小坐标值或极坐标值
        return vehicle_max_pre_x_or_max_pre_polar_radius, vehicle_max_pre_y_or_max_pre_polar_angle, \
               vehicle_min_pre_x_or_min_pre_polar_radius, vehicle_min_pre_y_or_min_pre_polar_angle

    #  如果车辆track属于直道阶段，更新智能体当前track起始的前沿坐标
    def StraightFrontier(self, vehicle_x, vehicle_y, judgment_str: str):
        if judgment_str == 'straight_y+':
            # 更新智能体当前track起始的前沿坐标最大的y值
            if self.straight_frontier_coord_positive_y <= vehicle_y:
                self.straight_frontier_coord_positive_y = vehicle_y
        elif judgment_str == 'straight_x+':
            # 更新智能体当前track起始的前沿坐标最大的x值
            if self.straight_frontier_coord_positive_x <= vehicle_x:
                self.straight_frontier_coord_positive_x = vehicle_x
        elif judgment_str == 'straight_y-':
            # 更新智能体当前track起始的前沿坐标最小的y值
            if self.straight_frontier_coord_negative_y >= vehicle_y:
                self.straight_frontier_coord_negative_y = vehicle_y
        elif judgment_str == 'straight_x-':
            # 更新智能体当前track起始的前沿坐标最小的x值
            if self.straight_frontier_coord_negative_x >= vehicle_x:
                self.straight_frontier_coord_negative_x = vehicle_x

    # 如果车辆track属于交叉口阶段，更新智能体当前track起始的前沿坐标
    def IntersectionFrontier(self, vehicle_x_or_radius, vehicle_y_or_angle, judgment_str: str):
        # 如果路口内状态为y+直道直行
        if judgment_str == 'intersection_straight_y+':
            # 更新智能体当前track起始的前沿坐标最大的y值
            if self.straight_frontier_coord_positive_y <= vehicle_y_or_angle:
                self.straight_frontier_coord_positive_y = vehicle_y_or_angle
        # 如果路口内状态为y+左转
        elif judgment_str == 'intersection_left_y+':
            # 更新智能体当前track起始前沿最大角度的极坐标值
            if self.steering_frontier_polar_coord_positive_angle <= vehicle_y_or_angle:
                self.steering_frontier_polar_coord_positive_angle = vehicle_y_or_angle
                self.steering_frontier_polar_coord_positive_radius = vehicle_x_or_radius
        # 如果路口内状态为y+右转
        elif judgment_str == 'intersection_right_y+':
            # 更新智能体当前track起始前沿最小角度的极坐标值
            if self.steering_frontier_polar_coord_negative_angle >= vehicle_y_or_angle:
                self.steering_frontier_polar_coord_negative_angle = vehicle_y_or_angle
                self.steering_frontier_polar_coord_negative_radius = vehicle_x_or_radius
