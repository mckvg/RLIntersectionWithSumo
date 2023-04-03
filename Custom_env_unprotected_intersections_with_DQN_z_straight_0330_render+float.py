from typing import List, Union, Any

import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

import gym
from gym import spaces
from gym.spaces.space import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DDPG, HerReplayBuffer, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import torch as th
import math
import os

os.environ['KMP_DUPLICATE_LIB_DK'] = 'True'

SCALE = 2
SIZE = 501.0  # 游戏区域的大小
# noinspection PyTypeChecker
MIN_COORD = float(-(SIZE - 1.0) / 2)
# noinspection PyTypeChecker
MAX_COORD = float((SIZE - 1.0) / 2)
VEHICLE_LENGTH = 5.0 * SCALE
VEHICLE_WIDTH = 1.8 * SCALE

# 通过车辆质心的速度方向向量与质心与vertex0连线的夹角
VEHICLE_ANGLE = math.atan(VEHICLE_WIDTH/VEHICLE_LENGTH)
# 车辆对角线长度
VEHICLE_DIAGONAL = math.sqrt(VEHICLE_WIDTH**2+VEHICLE_LENGTH**2)

VEHICLE_HALF_SIZE = 10
VEHICLE_ILLUSTRATION_HALF_SIZE = 1 * SCALE
SINGLE_LANE_WIDTH = 4.0 * SCALE
SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS = 40.0
STRIKE_LENGTH = 10
# straight length with 4 lanes with 2 lanes space
STRAIGHT_LENGTH = float(((SIZE - 1.0) - 6 * SINGLE_LANE_WIDTH) / 2)
# half intersection size(width or length) with 4 lanes with 2 lanes space
INTERSECTION_HALF_SIZE = 3 * SINGLE_LANE_WIDTH
REVERSE_DRIVING_LENGTH = 30.0

min_positionx = MIN_COORD + VEHICLE_HALF_SIZE
max_positionx = MAX_COORD - VEHICLE_HALF_SIZE
min_positiony = MIN_COORD + VEHICLE_HALF_SIZE
max_positiony = MAX_COORD - VEHICLE_HALF_SIZE


# 智能体的类，有其 状态信息 和 动作函数
class Cube:

    def __init__(self, size):  # 初始化状态
        self.size = size
        self.x = float(SINGLE_LANE_WIDTH / 2)
        self.y = MIN_COORD + 2 * VEHICLE_HALF_SIZE
        self.pre_x = self.x
        self.pre_y = self.y
        self.next1_x = self.x
        self.next1_y = self.y
        self.next2_x = self.x
        self.next2_y = self.y
        self.velocity = 2.0 * SCALE
        self.acceleration = 0.0 * SCALE
        self.yaw_angle = 0.0

        self.polar_radius = 0.0
        self.polar_angle = 0.0
        self.polar_radius_min_edge = 0.0
        self.polar_radius_max_edge = 0.0

        self.pre_polar_radius = 0.0
        self.pre_polar_angle = 0.0

        self.next1_polar_radius = 0.0
        self.next1_polar_angle = 0.0
        self.next1_polar_radius_min_edge = 0.0
        self.next1_polar_radius_max_edge = 0.0

        self.next2_polar_radius = 0.0
        self.next2_polar_angle = 0.0
        self.next2_polar_radius_min_edge = 0.0
        self.next2_polar_radius_max_edge = 0.0

        # 车头朝向正北(yaw_angle = 0),vertex0为左前点，1为右前点，2为右后点，3为左后点，按顺时针方向。
        self.vertex0x = self.x + VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex0y = self.y + VEHICLE_DIAGONAL/2 * math.cos(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex1x = self.x + VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle+VEHICLE_ANGLE)
        self.vertex1y = self.y + VEHICLE_DIAGONAL / 2 * math.cos(self.yaw_angle + VEHICLE_ANGLE)
        self.vertex2x = -self.vertex0x
        self.vertex2y = -self.vertex0y
        self.vertex3x = -self.vertex1x
        self.vertex3y = -self.vertex1y

        self.edge0 = self.x - VEHICLE_HALF_SIZE
        self.edge1 = self.y + VEHICLE_HALF_SIZE
        self.edge2 = self.x + VEHICLE_HALF_SIZE
        self.edge3 = self.y - VEHICLE_HALF_SIZE

        self.next1_edge0 = self.next1_x - VEHICLE_HALF_SIZE
        self.next1_edge1 = self.next1_y + VEHICLE_HALF_SIZE
        self.next1_edge2 = self.next1_x + VEHICLE_HALF_SIZE
        self.next1_edge3 = self.next1_y - VEHICLE_HALF_SIZE

        self.next2_edge0 = self.next2_x - VEHICLE_HALF_SIZE
        self.next2_edge1 = self.next2_y + VEHICLE_HALF_SIZE
        self.next2_edge2 = self.next2_x + VEHICLE_HALF_SIZE
        self.next2_edge3 = self.next2_y - VEHICLE_HALF_SIZE

        self.move_step = 0
        self.max_pre_x = min_positionx
        self.max_pre_y = min_positiony
        self.min_pre_x = max_positionx
        self.min_pre_y = max_positiony
        self.max_pre_polar_radius = 0.0
        self.max_pre_polar_angle = 0.0
        self.min_pre_polar_radius = 0.0
        self.min_pre_polar_angle = math.pi
        # y+方向为yaw_angle = 0, 顺时针方向为角度正方向，在yaw_angle= [-pi,pi], yaw_angle大于0为右转。

        self.state = ''
        self.pre_state = ''
        self.intersection_steering_choice = 2  # 2: default; 0: straight_y+; -1: turn left_x-; 1: turn right_x+.
        self.distance_to_off_road = np.array([self.edge0-0, self.edge2-INTERSECTION_HALF_SIZE], dtype=np.float32)
        # self.intersection_judgment = False

    def __str__(self):
        return f'{(self.x, self.y)},{self.yaw_angle},{self.velocity},{self.acceleration},{self.state}'

    def collision(self, other):
        flag = 0
        if abs(self.y - other.y) <= (2 * VEHICLE_HALF_SIZE) and abs(other.x - self.x) <= (2 * VEHICLE_HALF_SIZE) \
                and other.y != max_positiony and other.x != max_positionx and other.x != min_positionx:
            flag = 1
        else:
            flag = 0
        return flag

    # 玩家做动作
    def action(self, choice):
        power = 1
        min_action_acceleration = -4.5 * SCALE
        center_action_acceleration = 0.0 * SCALE
        max_action_acceleration = 2.6 * SCALE
        min_action_steering = -math.pi/6
        max_action_steering = math.pi/6
        min_speed = 0.0 * SCALE
        max_speed = 55.55 * SCALE
        action0 = 1.0 * SCALE
        action1 = -1.0 * SCALE
        action2 = min_action_steering / 2  # turn left pi/12
        action3 = max_action_steering / 2  # turn right pi/12

        self.pre_x = self.x
        self.pre_y = self.y
        self.move_step += 1
        if choice == 0:
            self.acceleration = 0.0
            self.acceleration_gas = 0.0
            self.acceleration_break = 0.0
            self.yaw_angle = self.yaw_angle
        elif choice == 1:
            self.acceleration_gas = action0 * power
            self.acceleration = self.acceleration_gas
            self.acceleration_break = 0.0
            self.yaw_angle = self.yaw_angle
        elif choice == 2:
            self.acceleration_break = action1 * power
            self.acceleration = self.acceleration_break
            self.acceleration_gas = 0.0
            self.yaw_angle = self.yaw_angle
        elif choice == 3:
            steer_left = action2
            self.yaw_angle += steer_left
            self.acceleration = 0.0
            self.acceleration_gas = 0.0
            self.acceleration_break = 0.0
        elif choice == 4:
            steer_right = action3
            self.yaw_angle += steer_right
            self.acceleration = 0.0
            self.acceleration_gas = 0.0
            self.acceleration_break = 0.0
        # force_gas = min(max(action0, center_action_acceleration), max_action_acceleration)
        # force_break = min(max(action1, min_action_acceleration), center_action_acceleration)
        # self.acceleration_gas = force_gas * power
        # self.acceleration_break = force_break * power
        # self.acceleration = self.acceleration_gas + self.acceleration_break
        self.velocity += self.acceleration

        if self.velocity > max_speed:
            self.velocity = max_speed
        if self.velocity < min_speed:
            self.velocity = min_speed

        # steer = min(max(action2, min_action_steering), max_action_steering)
        # self.yaw_angle += steer
        if self.yaw_angle > math.pi:
            self.yaw_angle = self.yaw_angle - 2*math.pi
        elif self.yaw_angle <= -math.pi:
            self.yaw_angle = 2*math.pi + self.yaw_angle
        # 转化为x坐标
        self.x += math.sin(self.yaw_angle) * self.velocity
        self.next1_x = self.x + (math.sin(self.yaw_angle) * self.velocity)
        self.next2_x = self.x + 2 * (math.sin(self.yaw_angle) * self.velocity)

        if self.x > max_positionx:
            self.x = max_positionx
        if self.x < min_positionx:
            self.x = min_positionx
        if self.next1_x > max_positionx:
            self.next1_x = max_positionx
        if self.next1_x < min_positionx:
            self.next1_x = min_positionx
        if self.next2_x > max_positionx:
            self.next2_x = max_positionx
        if self.next2_x < min_positionx:
            self.next2_x = min_positionx

        # 转化为y坐标
        self.y += (math.cos(self.yaw_angle) * self.velocity)
        self.next1_y = self.y + (math.cos(self.yaw_angle) * self.velocity)
        self.next2_y = self.y + 2 * (math.cos(self.yaw_angle) * self.velocity)

        if self.y > max_positiony:
            self.y = max_positiony
        if self.y < min_positiony:
            self.y = min_positiony
        if self.next1_y > max_positiony:
            self.next1_y = max_positiony
        if self.next1_y < min_positiony:
            self.next1_y = min_positiony
        if self.next2_y > max_positiony:
            self.next2_y = max_positiony
        if self.next2_y < min_positiony:
            self.next2_y = min_positiony

        # 更新车辆四个顶点的坐标



        # 0点方向起始，顺时针；以此为：上，右，下，左
        self.edge0 = self.x - VEHICLE_HALF_SIZE
        self.edge1 = self.y + VEHICLE_HALF_SIZE
        self.edge2 = self.x + VEHICLE_HALF_SIZE
        self.edge3 = self.y - VEHICLE_HALF_SIZE

        self.next1_edge0 = self.next1_x - VEHICLE_HALF_SIZE
        self.next1_edge1 = self.next1_y + VEHICLE_HALF_SIZE
        self.next1_edge2 = self.next1_x + VEHICLE_HALF_SIZE
        self.next1_edge3 = self.next1_y - VEHICLE_HALF_SIZE

        self.next2_edge0 = self.next2_x - VEHICLE_HALF_SIZE
        self.next2_edge1 = self.next2_y + VEHICLE_HALF_SIZE
        self.next2_edge2 = self.next2_x + VEHICLE_HALF_SIZE
        self.next2_edge3 = self.next2_y - VEHICLE_HALF_SIZE

        return np.array([self.acceleration_gas, self.acceleration_break, self.yaw_angle], dtype=np.float32)

    # 判断智能体处在什么阶段，更新self.state, 并更新直道的之前动作的最大最小坐标值,除了’intersection‘
    def judgment(self):
        if (-INTERSECTION_HALF_SIZE <= self.x <= INTERSECTION_HALF_SIZE) and \
                (-INTERSECTION_HALF_SIZE <= self.y <= INTERSECTION_HALF_SIZE):
            self.state = 'intersection'
        elif (MIN_COORD < self.y < -INTERSECTION_HALF_SIZE and 0 < self.x < INTERSECTION_HALF_SIZE) or \
                (INTERSECTION_HALF_SIZE < self.y < MAX_COORD and 0 < self.x < INTERSECTION_HALF_SIZE):
            self.state = 'straight_y+'
        elif (MIN_COORD < self.y < -INTERSECTION_HALF_SIZE and -INTERSECTION_HALF_SIZE < self.x < 0) or \
                (INTERSECTION_HALF_SIZE < self.y < MAX_COORD and -INTERSECTION_HALF_SIZE < self.x < 0):
            self.state = 'straight_y-'
        elif (MIN_COORD < self.x < -INTERSECTION_HALF_SIZE and -INTERSECTION_HALF_SIZE < self.y < 0) or \
                (INTERSECTION_HALF_SIZE < self.x < MAX_COORD and -INTERSECTION_HALF_SIZE < self.y < 0):
            self.state = 'straight_x+'
        elif (MIN_COORD < self.x < -INTERSECTION_HALF_SIZE and 0 < self.y < INTERSECTION_HALF_SIZE) or \
                (INTERSECTION_HALF_SIZE < self.x < MAX_COORD and 0 < self.y < INTERSECTION_HALF_SIZE):
            self.state = 'straight_x-'

        if (-INTERSECTION_HALF_SIZE <= self.pre_x <= INTERSECTION_HALF_SIZE) and \
                (-INTERSECTION_HALF_SIZE <= self.pre_y <= INTERSECTION_HALF_SIZE):
            self.pre_state = 'intersection'
        elif (MIN_COORD < self.pre_y < -INTERSECTION_HALF_SIZE and 0 < self.pre_x < INTERSECTION_HALF_SIZE) or \
                (INTERSECTION_HALF_SIZE < self.pre_y < MAX_COORD and 0 < self.pre_x < INTERSECTION_HALF_SIZE):
            self.pre_state = 'straight_y+'
        elif (MIN_COORD < self.pre_y < -INTERSECTION_HALF_SIZE and -INTERSECTION_HALF_SIZE < self.pre_x < 0) or \
                (INTERSECTION_HALF_SIZE < self.pre_y < MAX_COORD and -INTERSECTION_HALF_SIZE < self.pre_x < 0):
            self.pre_state = 'straight_y-'
        elif (MIN_COORD < self.pre_x < -INTERSECTION_HALF_SIZE and -INTERSECTION_HALF_SIZE < self.pre_y < 0) or \
                (INTERSECTION_HALF_SIZE < self.pre_x < MAX_COORD and -INTERSECTION_HALF_SIZE < self.pre_y < 0):
            self.pre_state = 'straight_x+'
        elif (MIN_COORD < self.pre_x < -INTERSECTION_HALF_SIZE and 0 < self.pre_y < INTERSECTION_HALF_SIZE) or \
                (INTERSECTION_HALF_SIZE < self.pre_x < MAX_COORD and 0 < self.pre_y < INTERSECTION_HALF_SIZE):
            self.pre_state = 'straight_x-'

    # 更新直道的之前动作的最大最小坐标值, 除了’intersection‘，根据state
    def UpdatePreExtremeValue(self):
        if self.state == 'straight_y+':
            # 更新之前动作的最大y值
            if self.max_pre_y <= self.pre_y:
               self.max_pre_y = self.pre_y
        elif self.state == 'straight_y-':
            # 更新之前动作的最小y值
            if self.min_pre_y >= self.pre_y:
                self.min_pre_y = self.pre_y
        elif self.state == 'straight_x+':
            # 更新之前动作的最大x值
            if self.max_pre_x <= self.pre_x:
                self.max_pre_x = self.pre_x
        elif self.state == 'straight_x-':
            # 更新之前动作的最小x值
            if self.min_pre_x >= self.pre_x:
                self.min_pre_x = self.pre_x

    # 智能体坐标及前一步坐标换成极坐标值: y+左转以（-INTERSECTION_HALF_SIZE,-INTERSECTION_HALF_SIZE）为原点，
    #                                    x+方向（水平向右）为角度为0的正方向，角度方向逆时针为正; angle[0,pi/2]
    #                               y+右转以（INTERSECTION_HALF_SIZE,-INTERSECTION_HALF_SIZE）为原点，
    #                                     x+方向（水平向右）为角度为0的正方向，角度方向逆时针为正; angle[pi/2,pi]
    def PolarCoord(self, judgment_str: str):

        x_prime = 0
        next1_x_prime = 0
        next2_x_prime = 0
        pre_x_prime = 0

        if judgment_str == 'intersection_left_y+':
            x_prime = self.x + INTERSECTION_HALF_SIZE
            next1_x_prime = self.next1_x + INTERSECTION_HALF_SIZE
            next2_x_prime = self.next2_x + INTERSECTION_HALF_SIZE
            pre_x_prime = self.pre_x + INTERSECTION_HALF_SIZE
        elif judgment_str == 'intersection_right_y+':
            x_prime = self.x - INTERSECTION_HALF_SIZE
            pre_x_prime = self.pre_x - INTERSECTION_HALF_SIZE
            next1_x_prime = self.next1_x - INTERSECTION_HALF_SIZE
            next2_x_prime = self.next2_x - INTERSECTION_HALF_SIZE

        y_prime = self.y + INTERSECTION_HALF_SIZE
        next1_y_prime = self.next1_y + INTERSECTION_HALF_SIZE
        next2_y_prime = self.next2_y + INTERSECTION_HALF_SIZE
        pre_y_prime = self.pre_y + INTERSECTION_HALF_SIZE

        self.polar_radius = math.sqrt(x_prime ** 2 + y_prime ** 2)
        self.polar_radius_min_edge = self.polar_radius - VEHICLE_HALF_SIZE
        self.polar_radius_max_edge = self.polar_radius + VEHICLE_HALF_SIZE

        self.next1_polar_radius = math.sqrt(next1_x_prime ** 2 + next1_y_prime ** 2)
        self.next1_polar_radius_min_edge = self.next1_polar_radius - VEHICLE_HALF_SIZE
        self.next1_polar_radius_max_edge = self.next1_polar_radius + VEHICLE_HALF_SIZE

        self.next2_polar_radius = math.sqrt(next2_x_prime ** 2 + next2_y_prime ** 2)
        self.next2_polar_radius_min_edge = self.next2_polar_radius - VEHICLE_HALF_SIZE
        self.next2_polar_radius_max_edge = self.next2_polar_radius + VEHICLE_HALF_SIZE

        self.pre_polar_radius = math.sqrt(pre_x_prime ** 2 + pre_y_prime ** 2)
        if x_prime == 0:
            self.polar_angle = math.pi / 2
        else:
            self.polar_angle = math.atan(y_prime / x_prime)
        if next1_x_prime == 0:
            self.next1_polar_angle = math.pi / 2
        else:
            self.next1_polar_angle = math.atan(next1_y_prime / next1_x_prime)
        if next2_x_prime == 0:
            self.next2_polar_angle = math.pi / 2
        else:
            self.next2_polar_angle = math.atan(next2_y_prime / next2_x_prime)
        if pre_x_prime == 0:
            self.pre_polar_angle = math.pi / 2
        else:
            self.pre_polar_angle = math.atan(pre_y_prime / pre_x_prime)


# 前车的类 有其 状态信息 和 行动
class Forward_Cube_First:

    def __init__(self, size):  # 初始化状态
        self.size = size
        self.x = float(SINGLE_LANE_WIDTH / 2 + SINGLE_LANE_WIDTH)
        self.y = MIN_COORD + 2 * VEHICLE_HALF_SIZE
        self.velocity = 0.0 * SCALE
        self.acceleration = 0.0
        self.move_step = 0
        self.yaw_angle = 0.0

    def __str__(self):
        return f'{(self.x, self.y)},{self.velocity},{self.acceleration}'

    def move(self):
        min_acceleration = -4.5 * SCALE
        max_acceleration = 2.6 * SCALE
        min_speed = 0.0 * SCALE
        max_speed = 55.55 * SCALE
        mid_speed = 2.0 * SCALE

        self.move_step += 1

        self.velocity = mid_speed

        if self.velocity < min_speed:
            self.velocity = min_speed
        if self.velocity > max_speed:
            self.velocity = max_speed

        self.y += self.velocity

        if self.y > max_positiony:
            self.y = max_positiony
        if self.y < min_positiony:
            self.y = min_positiony



# 前车的类 有其 状态信息 和 行动
class Forward_Cube_Second:

    def __init__(self, size):  # 初始化状态
        self.size = size
        self.x = float(SINGLE_LANE_WIDTH / 2 + SINGLE_LANE_WIDTH)
        self.y = MIN_COORD + 10 * VEHICLE_HALF_SIZE
        self.velocity = 0.0 * SCALE
        self.acceleration = 0.0 * SCALE
        self.move_step = 0
        self.yaw_angle = 0.0

    def __str__(self):
        return f'{(self.x, self.y)},{self.velocity},{self.acceleration}'

    def move(self):
        min_acceleration = -4.5 * SCALE
        max_acceleration = 2.6 * SCALE
        min_speed = 0.0 * SCALE
        max_speed = 55.55 * SCALE
        mid_speed = 8.0 * SCALE

        self.move_step += 1

        self.velocity = mid_speed

        if self.velocity < min_speed:
            self.velocity = min_speed
        if self.velocity > max_speed:
            self.velocity = max_speed

        self.y += self.velocity

        if self.y > max_positiony:
            self.y = max_positiony
        if self.y < min_positiony:
            self.y = min_positiony

# 信号灯及停止线的类
class Signal_Light_Stop_Line:

    def __init__(self, size):
        self.size = size
        # 停止线中心位置
        self.x = SINGLE_LANE_WIDTH
        self.y = MIN_COORD + STRAIGHT_LENGTH
        # 车祸区域中心位置
        self.danger_x = SINGLE_LANE_WIDTH/2
        self.danger_y = -INTERSECTION_HALF_SIZE + SINGLE_LANE_WIDTH/2
        self.signal = 0  # red: 0  green: 1  yellow: 2
        # 红 20s 绿 15s 黄 5s 依次变化/
        self.countdown_red = 20
        self.countdown_green = 15
        self.countdown_yellow = 5
        self.countdown_phase1 = self.countdown_red
        self.countdown_phase12 = self.countdown_red + self.countdown_green
        self.totalphasetime = self.countdown_red + self.countdown_yellow + self.countdown_green
        self.move_step = 0
        self.loop_step = 0
        self.countdown = 0

    def __str__(self):
        return f'{(self.x, self.y)},{self.signal}{self.countdown}'

    def signalphase(self, step: int):  # 红 20s 绿 15s 黄 5s 依次变化
        if math.floor(step / self.totalphasetime) > 0 and step % self.totalphasetime == 0:
            self.loop_step = self.totalphasetime
        else:
            self.loop_step = step % self.totalphasetime

        if 0 < self.loop_step <= self.countdown_phase1:
            self.signal = 0
        elif self.countdown_phase1 < self.loop_step <= self.countdown_phase12:
            self.signal = 1
        elif self.countdown_phase12 < self.loop_step <= self.totalphasetime:
            self.signal = 2

        return self.signal

    def signalcountdown(self, phase: int):
        self.move_step += 1
        if phase == 0:
            self.countdown = self.countdown_phase1 - self.loop_step
        elif phase == 1:
            self.countdown = self.countdown_phase12 - self.loop_step
        elif phase == 2:
            self.countdown = self.totalphasetime - self.loop_step

        return self.countdown


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


# 环境类
class envCube(gym.Env):
    RETURN_IMAGE = False
    RED_VIOLATION = False
    COLLISION = False
    GREEN_PASSING = False
    JUDGEMENT_IN_ROAD = True
    NEXT_1_IN_ROAD = True
    NEXT_2_IN_ROAD = True
    EXCEED_MAX_STEP = False

    d = {
        1: (255, 0, 0),  # blue
        2: (0, 255, 0),  # green
        3: (255, 255, 255),  # white
        4: (0, 0, 255),  # red
        5: (0, 255, 255)  # yellow
    }

    # 设定4个部分的颜色分别是蓝、绿、白、红
    VEHICLE_OTHER_N = 1
    VEHICLE_N = 2
    LINE_N = 3
    RED_LIGHT_N = 4
    YELLOW_LIGHT_N = 5
    GREEN_LIGHT_N = 2

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.min_action_acceleration = -4.5 * SCALE
        self.center_action_acceleration = 0.0 * SCALE
        self.max_action_acceleration = 2.6 * SCALE
        self.steering_space = 3
        self.min_action_steering = -math.pi/6  # left
        self.max_action_steering = math.pi/6  # right
        self.min_speed = 0.0 * SCALE
        self.max_speed = 55.55 * SCALE
        self.goal_min_positionx = min_positionx
        self.goal_max_positionx = max_positionx
        # straight goal
        self.goal_straight_positiony = max_positiony

        self.mid_goal_front_positiony = np.arange(
            min_positiony+2*VEHICLE_HALF_SIZE, -INTERSECTION_HALF_SIZE,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_front_space_y = self.mid_goal_front_positiony.shape[0]

        self.mid_goal_intersection_positiony = np.arange(
            -INTERSECTION_HALF_SIZE+SINGLE_LANE_WIDTH/2, INTERSECTION_HALF_SIZE+SINGLE_LANE_WIDTH,
            SINGLE_LANE_WIDTH, dtype=np.float32)
        self.mid_goal_intersection_space_y = self.mid_goal_intersection_positiony.shape[0]

        self.mid_goal_rear_positiony = np.arange(
            INTERSECTION_HALF_SIZE+SINGLE_LANE_WIDTH/2, max_positiony,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_rear_space_y = self.mid_goal_rear_positiony.shape[0]

        # turn left goal
        self.goal_turn_left_positionx = min_positionx

        self.mid_goal_turn_left_angle = np.arange(
            math.pi/16, 5*math.pi/8, math.pi/8, dtype=np.float32)
        self.mid_goal_turn_left_space_angle = self.mid_goal_turn_left_angle.shape[0]

        self.mid_goal_rear_positionx = np.arange(
            -INTERSECTION_HALF_SIZE-SINGLE_LANE_WIDTH/2, min_positionx,
            -SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_rear_space_x = self.mid_goal_rear_positionx.shape[0]

        # turn right goal
        self.goal_turn_right_positionx = max_positionx

        self.mid_goal_turn_right_angle = np.arange(
            15*math.pi/16, 3*math.pi/8, -math.pi/8, dtype=np.float32)
        self.mid_goal_turn_right_space_angle = self.mid_goal_turn_right_angle.shape[0]

        self.mid_goal_front_positionx = np.arange(
            INTERSECTION_HALF_SIZE + SINGLE_LANE_WIDTH/2, max_positionx,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_front_space_x = self.mid_goal_front_positionx.shape[0]

        self.judgement_space = 2
        self.risk_space = 4
        self.ACTION_SPACE_VALUES = 5
        # self.signal_light_phase_space_values = 3
        # self.signal_light_count_down_space_values = 60

        # self.low_action_state = np.array(
        #     [self.center_action_acceleration,
        #      self.center_action_acceleration,
        #      self.min_action_steering/self.max_action_steering,
        #      ], dtype=np.float32
        # )
        # self.high_action_state = np.array(
        #     [self.max_action_acceleration/self.max_action_acceleration,
        #      self.max_action_acceleration / self.max_action_acceleration,
        #      self.max_action_steering / self.max_action_steering,
        #      ], dtype=np.float32
        # )

        self.low_state = np.array(
            [self.min_speed,
             self.min_action_steering/self.max_action_steering], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_speed,
             self.max_action_steering/self.max_action_steering], dtype=np.float32
        )

        super(envCube, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(self.ACTION_SPACE_VALUES)
        # 0: do noting; 1: gas, 2: brake, 3: turn left 4: turn right

        self.observation_space = Dict(
            {
                'agent_position':  Box(MIN_COORD, MAX_COORD, shape=(2,), dtype=np.float32),
                'agent_speed_yawangle': Box(low=self.low_state, high=self.high_state, dtype=np.float32),
                'relative_position_2_danger': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),
                'first_other_vehicle_position': Box(MIN_COORD, MAX_COORD, shape=(2,), dtype=np.float32),
                'first_other_vehicle_relative_position': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),
                'first_other_vehicle_speed': Box(self.min_speed, self.max_speed, shape=(1,), dtype=np.float32),
                'second_other_vehicle_position': Box(MIN_COORD, MAX_COORD, shape=(2,), dtype=np.float32),
                'second_other_vehicle_relative_position': Box(2*MIN_COORD-1, 2*MAX_COORD+1, shape=(2,), dtype=np.float32),
                'second_other_vehicle_speed': Box(self.min_speed, self.max_speed, shape=(1,), dtype=np.float32),
                'relative_distance_2_goal': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(1,), dtype=np.float32),
                'judgement_in_road': Discrete(self.judgement_space),
                'risk_off_road': Discrete(self.risk_space),
                'distance_2_nearest_off_road': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),

                # 'distance_2_center_line': Discrete(MAX_COORD),
                # 'stop_line_position': Box(MIN_COORD, MAX_COORD, shape=(2,), dtype=np.int32),
                # 'signal_light_phase_countdown': MultiDiscrete(
                #             [self.signal_light_phase_space_values, self.signal_light_count_down_space_values])
            }
        )

    # 玩家做动作
    def step(self, action):
        self.episode_step += 1
        # agent_action0 = action[0] * self.max_action_acceleration   # gas
        # agent_action1 = action[1] * self.min_action_acceleration   # break
        # agent_action2 = action[2] * self.max_action_steering
        real_action = self.vehicle.action(action)
        self.first_other_vehicle.move()
        self.second_other_vehicle.move()
        # phase = self.signal_stop.signalphase(self.episode_step)
        # countdown = self.signal_stop.signalcountdown(phase)

        # 判断智能体处在什么阶段(其中十字路口中的统一模糊判断为intersection)
        self.vehicle.judgment()
        # 更新直道的之前动作的最大最小坐标值(intersection中之前动作不更新)
        self.vehicle.UpdatePreExtremeValue()

        reward = 0.0

        terminated = False
        break_out = False

        self.NEXT_1_IN_ROAD = True
        self.NEXT_2_IN_ROAD = True
        self.Time_to_off_road = 3

        # 进入逆行车道或者道路外，终止训练，并给予-500分惩罚
        if self.vehicle.edge3 <= MIN_COORD + STRAIGHT_LENGTH:
            if self.vehicle.edge0 <= 0 or self.vehicle.edge2 >= INTERSECTION_HALF_SIZE:
                self.JUDGEMENT_IN_ROAD = False
        elif -INTERSECTION_HALF_SIZE < self.vehicle.edge3 and self.vehicle.edge1 < 0:
            if self.vehicle.edge0 <= -INTERSECTION_HALF_SIZE:
                self.JUDGEMENT_IN_ROAD = False
        elif self.vehicle.edge1 >= 0 and self.vehicle.edge3 <= 0:
            if self.vehicle.edge0 <= -INTERSECTION_HALF_SIZE or self.vehicle.edge2 >= INTERSECTION_HALF_SIZE:
                self.JUDGEMENT_IN_ROAD = False
        elif self.vehicle.edge3 > 0 and self.vehicle.edge1 < INTERSECTION_HALF_SIZE:
            if self.vehicle.edge2 >= INTERSECTION_HALF_SIZE:
                self.JUDGEMENT_IN_ROAD = False
        elif self.vehicle.edge1 >= MAX_COORD - STRAIGHT_LENGTH:
            if self.vehicle.edge0 <= 0 or self.vehicle.edge2 >= INTERSECTION_HALF_SIZE:
                self.JUDGEMENT_IN_ROAD = False

        # 按当前路径下一步进入逆行车道或者道路外
        if self.vehicle.next1_edge3 <= MIN_COORD + STRAIGHT_LENGTH:
            if self.vehicle.next1_edge0 <= 0 or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_1_IN_ROAD = False
        elif -INTERSECTION_HALF_SIZE < self.vehicle.next1_edge3 and self.vehicle.next1_edge1 < 0:
            if self.vehicle.next1_edge0 <= -INTERSECTION_HALF_SIZE:
                self.NEXT_1_IN_ROAD = False
        elif self.vehicle.next1_edge1 >= 0 and self.vehicle.next1_edge3 <= 0:
            if self.vehicle.next1_edge0 <= -INTERSECTION_HALF_SIZE or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_1_IN_ROAD = False
        elif self.vehicle.next1_edge3 > 0 and self.vehicle.next1_edge1 < INTERSECTION_HALF_SIZE:
            if self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_1_IN_ROAD = False
        elif self.vehicle.next1_edge1 >= MAX_COORD - STRAIGHT_LENGTH:
            if self.vehicle.next1_edge0 <= 0 or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_1_IN_ROAD = False

        # 按当前路径下两步进入逆行车道或者道路外
        if self.vehicle.next2_edge3 <= MIN_COORD + STRAIGHT_LENGTH:
            if self.vehicle.next2_edge0 <= 0 or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_2_IN_ROAD = False
        elif -INTERSECTION_HALF_SIZE < self.vehicle.next2_edge3 and self.vehicle.next2_edge1 < 0:
            if self.vehicle.next2_edge0 <= -INTERSECTION_HALF_SIZE:
                self.NEXT_2_IN_ROAD = False
        elif self.vehicle.next2_edge1 >= 0 and self.vehicle.next2_edge3 <= 0:
            if self.vehicle.next2_edge0 <= -INTERSECTION_HALF_SIZE or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_2_IN_ROAD = False
        elif self.vehicle.next2_edge3 > 0 and self.vehicle.next2_edge1 < INTERSECTION_HALF_SIZE:
            if self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_2_IN_ROAD = False
        elif self.vehicle.next2_edge1 >= MAX_COORD - STRAIGHT_LENGTH:
            if self.vehicle.next2_edge0 <= 0 or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
                self.NEXT_2_IN_ROAD = False

        # 主车进入到车祸区域
        if -VEHICLE_HALF_SIZE <= self.vehicle.x <= SINGLE_LANE_WIDTH + VEHICLE_HALF_SIZE and \
                -VEHICLE_HALF_SIZE - INTERSECTION_HALF_SIZE <= self.vehicle.y <= VEHICLE_HALF_SIZE - SINGLE_LANE_WIDTH:
            self.JUDGEMENT_IN_ROAD = False
        # 主车下一步进入到车祸区域
        if -VEHICLE_HALF_SIZE <= self.vehicle.next1_x <= SINGLE_LANE_WIDTH + VEHICLE_HALF_SIZE and \
                -VEHICLE_HALF_SIZE - INTERSECTION_HALF_SIZE <= self.vehicle.next1_y <= VEHICLE_HALF_SIZE - SINGLE_LANE_WIDTH:
            self.NEXT_1_IN_ROAD = False
        # 主车下两步进入到车祸区域
        if -VEHICLE_HALF_SIZE <= self.vehicle.next2_x <= SINGLE_LANE_WIDTH + VEHICLE_HALF_SIZE and \
                -VEHICLE_HALF_SIZE - INTERSECTION_HALF_SIZE <= self.vehicle.next2_y <= VEHICLE_HALF_SIZE - SINGLE_LANE_WIDTH:
            self.NEXT_2_IN_ROAD = False

        # 两车相撞，终止训练，并给予-500分惩罚
        if self.vehicle.collision(self.first_other_vehicle) == 0 and self.vehicle.collision(self.second_other_vehicle) == 0:
            self.COLLISION = False
        if (self.vehicle.collision(self.first_other_vehicle) == 1 and self.COLLISION == False) or \
                (self.vehicle.collision(self.second_other_vehicle) == 1 and self.COLLISION == False):
            self.COLLISION = True
            break_out = True

        # # 闯红灯，终止训练，并给予-500分惩罚
        # if ((self.vehicle.pre_y <= self.signal_stop.y - VEHICLE_HALF_SIZE <= self.vehicle.y)
        #     or (self.signal_stop.y + 2 * VEHICLE_HALF_SIZE >= self.vehicle.edge1 >= self.signal_stop.y)) \
        #         and 2 * SINGLE_LANE_WIDTH > self.vehicle.x > 0 and phase == 0 and self.RED_VIOLATION == False:
        #     self.RED_VIOLATION = True
        #     break_out = True

        # # 绿灯时第一次通过，给予500分奖励
        # if ((self.vehicle.pre_y <= self.signal_stop.y - VEHICLE_HALF_SIZE <= self.vehicle.y)
        #     or (self.signal_stop.y + 2 * VEHICLE_HALF_SIZE >= self.vehicle.edge1 >= self.signal_stop.y)) \
        #         and 2 * SINGLE_LANE_WIDTH > self.vehicle.x > 0 and phase == 1 and self.GREEN_PASSING == False:
        #     self.GREEN_PASSING = True
        #     reward += 500

        # 如果车在直行,找到track前沿坐标，在智能体前方生成与车道方向平行的track，track覆盖所有车道，并且track带有奖赏，每直行1米获得1分奖赏
        if self.vehicle.state == 'straight_y+':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_positive_y - self.vehicle.max_pre_y) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle_track.straight_frontier_coord_positive_y - self.vehicle.y > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle_track.straight_frontier_coord_positive_y - self.vehicle.y) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = self.vehicle.edge0 - 0 if self.vehicle.edge0 > 0 else 0
            self.vehicle.distance_to_off_road[1] = self.vehicle.edge2 - INTERSECTION_HALF_SIZE \
                if self.vehicle.edge2 < INTERSECTION_HALF_SIZE else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.edge3 <= MIN_COORD:
                self.JUDGEMENT_IN_ROAD = False
            if self.vehicle.next1_edge3 <= MIN_COORD:
                self.NEXT_1_IN_ROAD = False
            if self.vehicle.next2_edge3 <= MIN_COORD:
                self.NEXT_2_IN_ROAD = False
            # 如果第一次或再次进入直行道，更新阶段目标空间和计数器
            if self.vehicle.pre_state != 'straight_y+' and self.vehicle.y < 0:
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_y), dtype=np.int32)
            elif self.vehicle.pre_state != 'straight_y+' and self.vehicle.y > 0:
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_rear_space_y), dtype=np.int32)
            # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
            if self.vehicle.y < 0:
                num = self.CIRCLE_COUNT
                if num < self.mid_goal_front_space_y:
                    if self.vehicle.y < self.mid_goal_front_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0:
                        self.distance_2_goal = abs(self.vehicle.y - self.mid_goal_front_positiony[num])
                    if self.vehicle.y >= self.mid_goal_front_positiony[num]:
                        self.CIRCLE_COUNT = num + 1
                    if self.vehicle.y >= self.mid_goal_front_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                0 <= self.vehicle.edge0 and self.vehicle.edge2 <= INTERSECTION_HALF_SIZE:
                        reward += 200.0
                        self.ARRIVE_AT_MID_GOAL[num] = 1
                        self.distance_2_goal = 0
            # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
            if self.vehicle.y > 0:
                num = self.CIRCLE_COUNT
                if num < self.mid_goal_rear_space_y:
                    if self.vehicle.y < self.mid_goal_rear_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0:
                        self.distance_2_goal = abs(self.vehicle.y - self.mid_goal_rear_positiony[num])
                    if self.vehicle.y >= self.mid_goal_rear_positiony[num]:
                        self.CIRCLE_COUNT = num + 1
                    if self.vehicle.y >= self.mid_goal_rear_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                0 <= self.vehicle.edge0 and self.vehicle.edge2 <= INTERSECTION_HALF_SIZE:
                        reward += 200.0
                        self.ARRIVE_AT_MID_GOAL[num] = 1
                        self.distance_2_goal = 0
                if self.mid_goal_rear_positiony[self.mid_goal_rear_space_y - 1] <= self.vehicle.y <= self.goal_straight_positiony \
                        and num >= self.mid_goal_rear_space_y:
                    self.distance_2_goal = abs(self.vehicle.y - self.goal_straight_positiony)

        elif self.vehicle.state == 'straight_x+':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.max_pre_x) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.x > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.x) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = 0 - self.vehicle.edge1 - 0 if self.vehicle.edge1 > 0 else 0
            self.vehicle.distance_to_off_road[1] = INTERSECTION_HALF_SIZE - self.vehicle.edge3 \
                if self.vehicle.edge3 < INTERSECTION_HALF_SIZE else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.edge0 <= MIN_COORD:
                self.JUDGEMENT_IN_ROAD = False
            if self.vehicle.next1_edge0 <= MIN_COORD:
                self.NEXT_1_IN_ROAD = False
            if self.vehicle.next2_edge0 <= MIN_COORD:
                self.NEXT_2_IN_ROAD = False
            # 如果第一次或再次进入直行道，更新阶段目标空间和计数器
            if self.vehicle.pre_state != 'straight_x+':
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_x), dtype=np.int32)
            # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
            num = self.CIRCLE_COUNT
            if num < self.mid_goal_front_space_x:
                if self.vehicle.x < self.mid_goal_front_positionx[num] and \
                        self.ARRIVE_AT_MID_GOAL[num] == 0:
                    self.distance_2_goal = abs(self.vehicle.x - self.mid_goal_front_positionx[num])
                if self.vehicle.x >= self.mid_goal_front_positionx[num]:
                    self.CIRCLE_COUNT = num + 1
                if self.vehicle.x >= self.mid_goal_front_positionx[num] and \
                        self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                        0 >= self.vehicle.edge1 and self.vehicle.edge3 >= -INTERSECTION_HALF_SIZE:
                    reward += 200.0
                    self.ARRIVE_AT_MID_GOAL[num] = 1
                    self.distance_2_goal = 0
            if self.mid_goal_front_positionx[self.mid_goal_front_space_x-1] <= self.vehicle.x <= self.goal_turn_right_positionx \
                    and num >= self.mid_goal_front_space_x:
                self.distance_2_goal = abs(self.vehicle.x - self.goal_turn_right_positionx)

        elif self.vehicle.state == 'straight_y-':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_negative_y - self.vehicle.min_pre_y) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle.y - self.vehicle_track.straight_frontier_coord_negative_y > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle.y - self.vehicle_track.straight_frontier_coord_negative_y) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = 0 - self.vehicle.edge2 if self.vehicle.edge2 > 0 else 0
            self.vehicle.distance_to_off_road[1] = INTERSECTION_HALF_SIZE - self.vehicle.edge0 \
                if self.vehicle.edge0 < INTERSECTION_HALF_SIZE else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.edge1 >= MAX_COORD:
                self.JUDGEMENT_IN_ROAD = False
            if self.vehicle.next1_edge1 >= MAX_COORD:
                self.NEXT_1_IN_ROAD = False
            if self.vehicle.next2_edge1 >= MAX_COORD:
                self.NEXT_2_IN_ROAD = False

        elif self.vehicle.state == 'straight_x-':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_negative_x - self.vehicle.min_pre_x) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle.x - self.vehicle_track.straight_frontier_coord_negative_x > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle.x - self.vehicle_track.straight_frontier_coord_negative_x) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = self.vehicle.edge3 - 0 if self.vehicle.edge3 > 0 else 0
            self.vehicle.distance_to_off_road[1] = self.vehicle.edge1 - INTERSECTION_HALF_SIZE \
                if self.vehicle.edge1 < INTERSECTION_HALF_SIZE else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.edge2 >= MAX_COORD:
                self.JUDGEMENT_IN_ROAD = False
            if self.vehicle.next1_edge2 >= MAX_COORD:
                self.NEXT_1_IN_ROAD = False
            if self.vehicle.next2_edge2 >= MAX_COORD:
                self.NEXT_2_IN_ROAD = False
            # 如果第一次或再次进入直行道，更新阶段目标空间和计数器
            if self.vehicle.pre_state != 'straight_x-':
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_rear_space_x), dtype=np.int32)
            # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
            num = self.CIRCLE_COUNT
            if num < self.mid_goal_rear_space_x:
                if self.vehicle.x > self.mid_goal_rear_positionx[num] and \
                        self.ARRIVE_AT_MID_GOAL[num] == 0:
                    self.distance_2_goal = abs(self.vehicle.x - self.mid_goal_rear_positionx[num])
                if self.vehicle.x <= self.mid_goal_rear_positionx[num]:
                    self.CIRCLE_COUNT = num + 1
                if self.vehicle.x <= self.mid_goal_rear_positionx[num] and \
                        self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                        0 <= self.vehicle.edge3 and self.vehicle.edge1 <= INTERSECTION_HALF_SIZE:
                    reward += 200.0
                    self.ARRIVE_AT_MID_GOAL[num] = 1
                    self.distance_2_goal = 0
            if self.mid_goal_rear_positionx[self.mid_goal_rear_space_x-1] >= self.vehicle.x >= self.goal_turn_left_positionx \
                    and num >= self.mid_goal_rear_space_x:
                self.distance_2_goal = abs(self.vehicle.x - self.goal_turn_left_positionx)

        # 十字路口内选择直行
        self.vehicle.intersection_steering_choice = 0  # 0: straight; -1: turn left;  1: turn right

        # 进入十字路口范围内，根据intersection_steering_choice，绘制track路线图，带有rewards，转弯部分利用极坐标系绘制
        if self.vehicle.state == 'intersection':
            # 更新车辆在十字路口内的track的细分阶段
            self.vehicle_track.IntersectionJudgment(self.vehicle.intersection_steering_choice)
            # 如果路口内状态为直道直行_y+：
            if self.vehicle_track.judgment_str == 'intersection_straight_y+':
                # 更新智能体在十字路口内的直行y+的之前动作的最大最小坐标值
                self.vehicle.max_pre_x, self.vehicle.max_pre_y, self.vehicle.min_pre_x, self.vehicle.min_pre_y = \
                    self.vehicle_track.UpdatePreExtremum(self.vehicle.max_pre_x, self.vehicle.max_pre_y,
                                                         self.vehicle.min_pre_x, self.vehicle.min_pre_y,
                                                         self.vehicle.pre_x, self.vehicle.pre_y)
                # 更新智能体当前track的直道前沿坐标值
                self.vehicle_track.IntersectionFrontier(self.vehicle.x, self.vehicle.y, self.vehicle_track.judgment_str)
                # 如果智能体在十字路口内的直行范围内，更新rewards
                if 0 <= self.vehicle.edge0 and self.vehicle.edge2 <= INTERSECTION_HALF_SIZE:
                    reward += abs(self.vehicle_track.straight_frontier_coord_positive_y - self.vehicle.max_pre_y) * 1
                # 计算智能体距off-road最近距离
                self.vehicle.distance_to_off_road[0] = self.vehicle.edge0 - 0 if self.vehicle.edge0 > 0 else 0
                self.vehicle.distance_to_off_road[1] = self.vehicle.edge2 - INTERSECTION_HALF_SIZE \
                    if self.vehicle.edge2 < INTERSECTION_HALF_SIZE else 0
                # 制定道路范围内边界
                if self.vehicle.edge0 < 0 or self.vehicle.edge2 > INTERSECTION_HALF_SIZE:
                    self.JUDGEMENT_IN_ROAD = False
                if self.vehicle.next1_edge0 < 0 or self.vehicle.next1_edge2 > INTERSECTION_HALF_SIZE:
                    self.NEXT_1_IN_ROAD = False
                if self.vehicle.next2_edge0 < 0 or self.vehicle.next2_edge2 > INTERSECTION_HALF_SIZE:
                    self.NEXT_2_IN_ROAD = False
                # 如果第一次进入交叉口，更新阶段目标空间和计数器
                if self.vehicle.pre_state != 'intersection':
                    self.CIRCLE_COUNT = 0
                    self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_intersection_space_y), dtype=np.int32)
                # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                num = self.CIRCLE_COUNT
                if num < self.mid_goal_intersection_space_y:
                    if self.vehicle.y < self.mid_goal_intersection_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0:
                        self.distance_2_goal = abs(self.vehicle.y - self.mid_goal_intersection_positiony[num])
                    if self.vehicle.y >= self.mid_goal_intersection_positiony[num]:
                        self.CIRCLE_COUNT = num + 1
                    if self.vehicle.y >= self.mid_goal_intersection_positiony[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                            0 <= self.vehicle.edge0 and self.vehicle.edge2 <= INTERSECTION_HALF_SIZE:
                        reward += 200.0
                        self.ARRIVE_AT_MID_GOAL[num] = 1
                        self.distance_2_goal = 0

            # 如果路口内状态为左转_y+
            elif self.vehicle_track.judgment_str == 'intersection_left_y+':
                # 将智能体的直角坐标转化为极坐标
                self.vehicle.PolarCoord(self.vehicle_track.judgment_str)
                # 更新智能体在十字路口内的左转_y+的之前动作的最大最小极坐标值
                self.vehicle.max_pre_polar_radius, self.vehicle.max_pre_polar_angle, \
                self.vehicle.min_pre_polar_radius, self.vehicle.min_pre_polar_angle = \
                    self.vehicle_track.UpdatePreExtremum(self.vehicle.max_pre_polar_radius, self.vehicle.max_pre_polar_angle,
                                                         self.vehicle.min_pre_polar_radius, self.vehicle.min_pre_polar_angle,
                                                         self.vehicle.pre_polar_radius, self.vehicle.pre_polar_angle)
                # 更新智能体当前track的左转_y+前沿极坐标值
                self.vehicle_track.IntersectionFrontier(self.vehicle.polar_radius, self.vehicle.polar_angle,
                                                        self.vehicle_track.judgment_str)
                # 如果第一次进入交叉口，更新阶段目标空间和计数器
                if self.vehicle.pre_state != 'intersection':
                    self.CIRCLE_COUNT = 0
                    self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_turn_left_space_angle), dtype=np.int32)
                # 如果智能体在十字路口内的左转_y+范围内，分三个阶段，更新rewards，并计算off-road最近距离，并指定道路范围边界。
                # 第1阶段：angle = [0, atan1/2], min_radius = INTERSECTION_HALF_SIZE, max_radius = 2*INTERSECTION_HALF_SIZE/cos(angle)
                if 0 <= self.vehicle.polar_angle < math.atan(1.0 / 2.0):
                    if INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - INTERSECTION_HALF_SIZE \
                        if self.vehicle.polar_radius_min_edge > INTERSECTION_HALF_SIZE else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle) \
                        if self.vehicle.polar_radius_max_edge < 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle):
                        self.JUDGEMENT_IN_ROAD = False
                    if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next1_polar_radius_max_edge > 2 * INTERSECTION_HALF_SIZE/math.cos(self.vehicle.next1_polar_angle):
                        self.NEXT_1_IN_ROAD = False
                    if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next2_polar_radius_max_edge > 2 * INTERSECTION_HALF_SIZE/math.cos(self.vehicle.next2_polar_angle):
                        self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                                self.vehicle.polar_radius_max_edge <= 2 * INTERSECTION_HALF_SIZE / math.cos(self.vehicle.polar_angle):
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0
                # 第2阶段：angle = (atan1/2, atan2), min_radius = INTERSECTION_HALF_SIZE, max_radius = INTERSECTION_HALF_SIZE*sqrt(5.0)
                elif math.atan(1.0 / 2.0) <= self.vehicle.polar_angle <= math.atan(2.0):
                    if INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - INTERSECTION_HALF_SIZE \
                        if self.vehicle.polar_radius_min_edge > INTERSECTION_HALF_SIZE else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - INTERSECTION_HALF_SIZE * math.sqrt ( 5.0 ) \
                        if self.vehicle.polar_radius_max_edge < INTERSECTION_HALF_SIZE * math.sqrt(5.0) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        self.JUDGEMENT_IN_ROAD = False
                    if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next1_polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        self.NEXT_1_IN_ROAD = False
                    if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next2_polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                                self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE * math.sqrt ( 5.0 ):
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0
                # 第3阶段：angle = (atan2, pi/2], min_radius = INTERSECTION_HALF_SIZE, max_radius = 2*INTERSECTION_HALF_SIZE/sin(angle)
                elif math.atan(2.0) < self.vehicle.polar_angle <= math.pi/2:
                    if INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - INTERSECTION_HALF_SIZE \
                        if self.vehicle.polar_radius_min_edge > INTERSECTION_HALF_SIZE else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle) \
                        if self.vehicle.polar_radius_max_edge < 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle):
                        self.JUDGEMENT_IN_ROAD = False
                    if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next1_polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.next1_polar_angle):
                        self.NEXT_1_IN_ROAD = False
                    if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                            self.vehicle.next2_polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.next2_polar_angle):
                        self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                INTERSECTION_HALF_SIZE <= self.vehicle.polar_radius_min_edge and \
                                self.vehicle.polar_radius_max_edge <= 2 * INTERSECTION_HALF_SIZE / math.sin(self.vehicle.polar_angle):
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0

            # 如果路口内状态为右转_y+
            elif self.vehicle_track.judgment_str == 'intersection_right_y+':
                # 将智能体的直角坐标转化为极坐标
                self.vehicle.PolarCoord(self.vehicle_track.judgment_str)
                # 更新智能体在十字路口内的右转_y+的之前动作的最大最小极坐标值
                self.vehicle.max_pre_polar_radius, self.vehicle.max_pre_polar_angle, \
                self.vehicle.min_pre_polar_radius, self.vehicle.min_pre_polar_angle = \
                    self.vehicle_track.UpdatePreExtremum(self.vehicle.max_pre_polar_radius, self.vehicle.max_pre_polar_angle,
                                                         self.vehicle.min_pre_polar_radius, self.vehicle.min_pre_polar_angle,
                                                         self.vehicle.pre_polar_radius, self.vehicle.pre_polar_angle)
                # 更新智能体当前track的右转_y+前沿极坐标值
                self.vehicle_track.IntersectionFrontier(self.vehicle.polar_radius, self.vehicle.polar_angle,
                                                        self.vehicle_track.judgment_str)
                # 如果智能体在十字路口内的右转_y+范围内，更新rewards
                if 0 <= self.vehicle.polar_radius_min_edge and self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE:
                    reward += abs(self.vehicle_track.steering_frontier_polar_coord_negative_angle - self.vehicle.min_pre_polar_angle) * 100.0
                # 计算智能体距off-road最近距离
                self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - 0 \
                    if self.vehicle.polar_radius_min_edge > 0 else 0
                self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - INTERSECTION_HALF_SIZE \
                    if self.vehicle.polar_radius_max_edge < INTERSECTION_HALF_SIZE else 0
                # 制定道路范围内边界
                if self.vehicle.polar_radius_min_edge < 0 or \
                        self.vehicle.polar_radius_max_edge > INTERSECTION_HALF_SIZE:
                    self.JUDGEMENT_IN_ROAD = False
                if self.vehicle.next1_polar_radius_min_edge < 0 or \
                        self.vehicle.next1_polar_radius_max_edge > INTERSECTION_HALF_SIZE:
                    self.NEXT_1_IN_ROAD = False
                if self.vehicle.next2_polar_radius_min_edge < 0 or \
                        self.vehicle.next2_polar_radius_max_edge > INTERSECTION_HALF_SIZE:
                    self.NEXT_2_IN_ROAD = False
                # 如果第一次进入交叉口，更新阶段目标空间和计数器
                if self.vehicle.pre_state != 'intersection':
                    self.CIRCLE_COUNT = 0
                    self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_turn_right_space_angle), dtype=np.int32)
                # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                num = self.CIRCLE_COUNT
                if num < self.mid_goal_turn_right_space_angle:
                    if self.vehicle.polar_angle > self.mid_goal_turn_right_angle[num] and \
                            self.ARRIVE_AT_MID_GOAL[num] == 0:
                        self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_right_angle[num]) * 100
                    if self.vehicle.polar_angle <= self.mid_goal_turn_right_angle[num]:
                        self.CIRCLE_COUNT = num + 1
                    if self.vehicle.polar_angle <= self.mid_goal_turn_right_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                            0 <= self.vehicle.polar_radius_min_edge and self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE:
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0

        # 如果主车进入到禁止进入区域，break-out.并根据主车下两步的路径更新Time_to_off_road
        if self.NEXT_2_IN_ROAD == False:
            self.Time_to_off_road = 2
        if self.NEXT_1_IN_ROAD == False:
            self.Time_to_off_road = 1
        if self.JUDGEMENT_IN_ROAD == False:
            self.Time_to_off_road = 0
            break_out = True

        # 成功驶出十字路口范围，到达目标，给予1000分奖赏
        if self.vehicle.intersection_steering_choice == -1:  # turn left
            self.goal = bool(self.vehicle.x <= self.goal_min_positionx and self.vehicle.velocity > 0)
        elif self.vehicle.intersection_steering_choice == 0:  # straight
            self.goal = bool(self.vehicle.y >= self.goal_straight_positiony and self.vehicle.velocity > 0)
        elif self.vehicle.intersection_steering_choice == 1:  # turn right
            self.goal = bool(self.vehicle.x >= self.goal_max_positionx and self.vehicle.velocity > 0)

        # 如果超过200步，未通过路口，(未完成goal），break_out
        if self.episode_step > 200:
            self.EXCEED_MAX_STEP = True
            break_out = True

        # 油门及刹车动作的惩罚
        reward -= math.pow((real_action[0]+real_action[1]), 2) * 0.1

        if break_out:
            reward += -500.0
            terminated = True

        if self.goal:
            reward += 1000.0
            terminated = True

        self.vehicle_position = np.array(
            [self.vehicle.x, self.vehicle.y], dtype=np.float32)
        self.vehicle_state = np.array(
            [self.vehicle.velocity,
             self.vehicle.yaw_angle/math.pi], dtype=np.float32)
        self.relative_position_2_danger = np.array(
            [self.vehicle.x - self.signal_stop.danger_x, self.vehicle.y - self.signal_stop.danger_y], dtype=np.float32)
        self.first_other_vehicle_position = np.array(
            [self.first_other_vehicle.x, self.first_other_vehicle.y], dtype=np.float32)
        self.first_other_vehicle_relative_position = np.array(
            [self.vehicle.x-self.first_other_vehicle.x, self.vehicle.y-self.first_other_vehicle.y], dtype=np.float32)
        self.first_other_vehicle_speed = np.array([self.first_other_vehicle.velocity], dtype=np.float32)
        self.second_other_vehicle_position = np.array(
            [self.second_other_vehicle.x, self.second_other_vehicle.y], dtype=np.float32)
        self.second_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.second_other_vehicle.x, self.vehicle.y - self.second_other_vehicle.y], dtype=np.float32)
        self.second_other_vehicle_speed = np.array([self.second_other_vehicle.velocity], dtype=np.float32)
        self.relative_distance_to_goal = np.array([self.distance_2_goal], dtype=np.float32)

        # self.signal_stop_position = np.array([self.signal_stop.x, self.signal_stop.y], dtype=np.int32)
        # self.signal_stop_state = np.array([phase, countdown], dtype=np.int32)

        print(self.episode_step, ':', 'action:', action, self.vehicle_position, self.vehicle_state, self.relative_distance_to_goal,
              self.JUDGEMENT_IN_ROAD, self.Time_to_off_road, self.vehicle.distance_to_off_road)
        if reward < -100 or reward > 100:
          print(reward)

        self.state: dict = (
            {
                'agent_position': self.vehicle_position,
                'agent_speed_yawangle': self.vehicle_state,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_position': self.first_other_vehicle_position,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_position': self.second_other_vehicle_position,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'risk_off_road': self.Time_to_off_road,
                'distance_2_nearest_off_road': self.vehicle.distance_to_off_road,
                # 'stop_line_position': self.signal_stop_position,
                # 'signal_light_phase_countdown': self.signal_stop_state
            }
        )

        info = {}

        return self.state, reward, terminated, info

    # 重置环境-整局游戏结束之后。可做初始化函数，智能体和观测
    def reset(self):
        self.vehicle = Cube(SIZE)
        self.first_other_vehicle = Forward_Cube_First(SIZE)
        self.second_other_vehicle = Forward_Cube_Second(SIZE)
        self.signal_stop = Signal_Light_Stop_Line(SIZE)
        self.vehicle_track = Track(SIZE)

        self.vehicle_position = np.array(
            [self.vehicle.x, self.vehicle.y], dtype=np.float32)
        self.vehicle_state = np.array(
            [self.vehicle.velocity,
             self.vehicle.yaw_angle/math.pi], dtype=np.float32)
        self.relative_position_2_danger = np.array(
            [self.vehicle.x - self.signal_stop.danger_x, self.vehicle.y - self.signal_stop.danger_y], dtype=np.float32)
        self.first_other_vehicle_position = np.array(
            [self.first_other_vehicle.x, self.first_other_vehicle.y], dtype=np.float32)
        self.first_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.first_other_vehicle.x, self.vehicle.y - self.first_other_vehicle.y], dtype=np.float32)
        self.first_other_vehicle_speed = np.array([self.first_other_vehicle.velocity], dtype=np.float32)
        self.second_other_vehicle_position = np.array(
            [self.second_other_vehicle.x, self.second_other_vehicle.y], dtype=np.float32)
        self.second_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.second_other_vehicle.x, self.vehicle.y - self.second_other_vehicle.y], dtype=np.float32)
        self.second_other_vehicle_speed = np.array([self.second_other_vehicle.velocity], dtype=np.float32)
        self.relative_distance_to_goal = np.array([abs(self.vehicle.y - self.mid_goal_front_positiony[0])], dtype=np.float32)

        # self.signal_stop_position = np.array([self.signal_stop.x, self.signal_stop.y], dtype=np.int32)
        # self.signal_stop_state = np.array([self.signal_stop.signal, self.signal_stop.countdown], dtype=np.int64)

        self.episode_step = 0

        self.RED_VIOLATION = False
        self.COLLISION = False
        self.GREEN_PASSING = False
        self.JUDGEMENT_IN_ROAD = True
        self.NEXT_1_IN_ROAD = True
        self.NEXT_2_IN_ROAD = True
        self.EXCEED_MAX_STEP = False
        self.Time_to_off_road = 3
        self.CIRCLE_COUNT = 0
        self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_y), dtype=np.int32)

        # print(self.episode_step, ':', self.vehicle_position, self.vehicle_state, self.relative_distance_to_goal,
        #       self.JUDGEMENT_IN_ROAD, self.Time_to_off_road, self.vehicle.distance_to_off_road)

        self.state: dict = (
            {
                'agent_position': self.vehicle_position,
                'agent_speed_yawangle': self.vehicle_state,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_position': self.first_other_vehicle_position,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_position': self.second_other_vehicle_position,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'risk_off_road': self.Time_to_off_road,
                'distance_2_nearest_off_road': self.vehicle.distance_to_off_road,
                # 'stop_line_position': self.signal_stop_position,
                # 'signal_light_phase_countdown': self.signal_stop_state
            }
        )

        return self.state

    # 多媒体演示
    def render(self, mode="human"):
        img = self.get_image()
        cv2.imshow('test', np.array(img))
        if self.goal or self.EXCEED_MAX_STEP:
            cv2.waitKey(1500)
        else:
            cv2.waitKey(100)

    def get_image(self):
        INT_SIZE = int(round(SIZE, 0))
        env = np.zeros((INT_SIZE, INT_SIZE, 3), dtype=np.uint8)

        # 画出十字路口框架及车道
        INT_MAX_COORD = int(round(MAX_COORD, 0))
        INT_STRAIGHT_LENGTH = int(round(STRAIGHT_LENGTH,0))
        INT_SINGLE_LANE_WIDTH = int(round(SINGLE_LANE_WIDTH,0))
        INT_INTERSECTION_HALF_SIZE = int(round(INTERSECTION_HALF_SIZE,0))
        INT_VEHICLE_HALF_SIZE = int(round(VEHICLE_HALF_SIZE, 0))
        # 十字路口框架
        for x1 in range(0, INT_STRAIGHT_LENGTH):
            for y1 in range(0, INT_STRAIGHT_LENGTH):
                env[x1][y1] = self.d[self.LINE_N]
                env[INT_SIZE - x1 - 1][y1] = self.d[self.LINE_N]
                env[x1][INT_SIZE - y1 - 1] = self.d[self.LINE_N]
                env[INT_SIZE - x1 - 1][INT_SIZE - y1 - 1] = self.d[self.LINE_N]
        for x111 in range(0, INT_STRAIGHT_LENGTH):
            for y111 in range(INT_STRAIGHT_LENGTH, INT_STRAIGHT_LENGTH + INT_SINGLE_LANE_WIDTH):
                env[x111][y111] = self.d[self.LINE_N]
                env[y111][x111] = self.d[self.LINE_N]
                env[INT_SIZE - x111 - 1][y111] = self.d[self.LINE_N]
                env[INT_SIZE - y111 - 1][x111] = self.d[self.LINE_N]
                env[x111][INT_SIZE - y111 - 1] = self.d[self.LINE_N]
                env[y111][INT_SIZE - x111 - 1] = self.d[self.LINE_N]
                env[INT_SIZE - x111 - 1][INT_SIZE - y111 - 1] = self.d[self.LINE_N]
                env[INT_SIZE - y111 - 1][INT_SIZE - x111 - 1] = self.d[self.LINE_N]
        # 车道线
        for x2 in range(INT_STRAIGHT_LENGTH, INT_STRAIGHT_LENGTH + 4 * INT_SINGLE_LANE_WIDTH + 1, INT_INTERSECTION_HALF_SIZE):
            for y2 in range(0, INT_STRAIGHT_LENGTH):
                env[x2][y2] = self.d[self.LINE_N]
                env[y2][x2] = self.d[self.LINE_N]
                env[x2][INT_SIZE - y2 - 1] = self.d[self.LINE_N]
                env[INT_SIZE - y2 - 1][x2] = self.d[self.LINE_N]
        # 画出道中央虚线
        for x7 in range(INT_STRAIGHT_LENGTH + 2 * INT_SINGLE_LANE_WIDTH, INT_STRAIGHT_LENGTH + 4 * INT_SINGLE_LANE_WIDTH + 1,
                        2 * INT_SINGLE_LANE_WIDTH):
            for y7 in range(0, 22*STRIKE_LENGTH, 3*STRIKE_LENGTH):
                for z7 in range(y7, y7+STRIKE_LENGTH):
                    env[x7][z7] = self.d[self.LINE_N]
                    env[z7][x7] = self.d[self.LINE_N]
                    env[x7][INT_SIZE - z7 - 1] = self.d[self.LINE_N]
                    env[INT_SIZE - z7 - 1][x7] = self.d[self.LINE_N]

        # 画出智能体及前车的位置
        v_position: np.ndarry = [0, 0]
        INT_VEHICLE_Y = int(round(self.vehicle.y, 0))
        INT_VEHICLE_X = int(round(self.vehicle.x, 0))
        INT_FIRST_OTHER_VEHICLE_Y = int(round(self.first_other_vehicle.y, 0))
        INT_FIRST_OTHER_VEHICLE_X = int(round(self.first_other_vehicle.x, 0))
        INT_SECOND_OTHER_VEHICLE_Y = int(round(self.second_other_vehicle.y, 0))
        INT_SECOND_OTHER_VEHICLE_X = int(round(self.second_other_vehicle.x, 0))
        v_position[0] = -INT_VEHICLE_Y + INT_MAX_COORD
        v_position[1] = INT_VEHICLE_X + INT_MAX_COORD
        first_other_vehicle_v_position: np.ndarry = [0, 0]
        first_other_vehicle_v_position[0] = -INT_FIRST_OTHER_VEHICLE_Y + INT_MAX_COORD
        first_other_vehicle_v_position[1] = INT_FIRST_OTHER_VEHICLE_X + INT_MAX_COORD
        second_other_vehicle_v_position: np.ndarry = [0, 0]
        second_other_vehicle_v_position[0] = -INT_SECOND_OTHER_VEHICLE_Y + INT_MAX_COORD
        second_other_vehicle_v_position[1] = INT_SECOND_OTHER_VEHICLE_X + INT_MAX_COORD

        for x3 in range(-VEHICLE_ILLUSTRATION_HALF_SIZE, VEHICLE_ILLUSTRATION_HALF_SIZE):
            for y3 in range(-VEHICLE_ILLUSTRATION_HALF_SIZE, VEHICLE_ILLUSTRATION_HALF_SIZE):
                env[v_position[0] + x3][v_position[1] + y3] = self.d[self.VEHICLE_N]
                if first_other_vehicle_v_position[1] < INT_MAX_COORD + max_positiony:
                    env[first_other_vehicle_v_position[0] + x3][first_other_vehicle_v_position[1] + y3] = self.d[
                          self.VEHICLE_OTHER_N]
                if second_other_vehicle_v_position[1] < INT_MAX_COORD + max_positiony:
                    env[second_other_vehicle_v_position[0] + x3][second_other_vehicle_v_position[1] + y3] = self.d[
                          self.VEHICLE_OTHER_N]
        for x5 in range(0, int(2.0 * VEHICLE_ILLUSTRATION_HALF_SIZE)):
            x = v_position[0] - int(x5 * math.cos(self.vehicle.yaw_angle))
            y = v_position[1] + int(x5 * math.sin(self.vehicle.yaw_angle))
            if x >= INT_SIZE:
                x = INT_SIZE - 1
            if y >= INT_SIZE:
                y = INT_SIZE - 1
            env[x][y] = self.d[self.RED_LIGHT_N]

        # 画出车祸区域
        for x6 in range(INT_MAX_COORD + INT_SINGLE_LANE_WIDTH, INT_MAX_COORD + 2 * INT_SINGLE_LANE_WIDTH):
            for y6 in range(INT_MAX_COORD, INT_MAX_COORD + INT_SINGLE_LANE_WIDTH):
                env[x6][y6] = self.d[self.RED_LIGHT_N]

        # # 画出车道的信号灯及停止线
        # ss_position: np.ndarry = [0, 0]
        # ss_position[0] = self.signal_stop.x + MAX_COORD
        # ss_position[1] = self.signal_stop.y + MAX_COORD
        #
        # if self.signal_stop.signalphase(self.episode_step) == 0:
        #     for x4 in range(ss_position[0] - SINGLE_LANE_WIDTH, ss_position[0] + SINGLE_LANE_WIDTH):
        #         env[x4][ss_position[1]] = self.d[self.RED_LIGHT_N]
        # elif self.signal_stop.signalphase(self.episode_step) == 1:
        #     for x4 in range(ss_position[0] - SINGLE_LANE_WIDTH, ss_position[0] + SINGLE_LANE_WIDTH):
        #         env[x4][ss_position[1]] = self.d[self.GREEN_LIGHT_N]
        # elif self.signal_stop.signalphase(self.episode_step) == 2:
        #     for x4 in range(ss_position[0] - SINGLE_LANE_WIDTH, ss_position[0] + SINGLE_LANE_WIDTH):
        #         env[x4][ss_position[1]] = self.d[self.YELLOW_LIGHT_N]

        img = Image.fromarray(env, 'RGB')
        return img


env = envCube()
check_env(env)

# # Instantiate the agent
# model = DQN(
#     "MultiInputPolicy",
#     env,
#     verbose=1,
#     tensorboard_log='./logs',
#     learning_rate=1e-4,
# )
# print(model.policy)
#
# # Callback
# eval_callback = EvalCallback(env, best_model_save_path="./logs/BestModel_straight1/",
#                              log_path="./logs/BestModel_straight1/", eval_freq=500,
#                              deterministic=True, render=False)
#
# # Train the agent and display a progress bar
# model.learn(
#     total_timesteps=int(5e6),
#     tb_log_name='Unprotected_intersection_with_collision_area_straight_DQN_5M_call',
#     progress_bar=True,
#     callback=eval_callback
# )
#
# # Save the agent
# model.save("Unprotected_intersection_with_collision_area_straight_DQN_5M_call")
# del model  # delete trained model to demonstrate loading
#
# # Load the trained agent
# # NOTE: if you have loading issue, you can pass `print_system_info=True`
# # to compare the system on which the model was trained vs the current one
# model = DQN.load("Unprotected_intersection_with_collision_area_straight_DQN_5M_call", env=env)
#
# print(model.policy)

#
# model = DQN.load("./logs/BestModel_straight1/best_model", env=env)

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(
#     model,
#     model.get_env(),
#     deterministic = True,
# #     render = True,
# #     n_eval_episodes=10,
# # )
# # print(mean_reward, std_reward)
#
# # best_model = A2C.load("./logs/BestModel/best_model.zip",env=env)
# #
# # best_mean_reward, best_std_reward = evaluate_policy(
# #     best_model,
# #     best_model.get_env(),
# #     deterministic = True,
# #     render = True,
# #     n_eval_episodes=10,
# # )
# # print(best_mean_reward, best_std_reward)

eposides = 100
for ep in range(eposides):
    obs = env.reset()
    done = False
    rewards = 0
    while not done:
        # action = env.action_space.sample()
        action = 0
        # action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.render()
        rewards += reward
        # if reward < -100 or reward > 100:
        #   print(reward)

    print(rewards)

# tensorboard --logdir ./logs
