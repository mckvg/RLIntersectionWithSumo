
###########################
#
#   @author mckvg
#   @author olin322
#
###########################



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

from Cube import Cube
from Forward_Cube_First import Forward_Cube_First
from Forward_Cube_Second import Forward_Cube_Second
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
from Track import Track
import Rectangle
from Rectangle import Rectangle, Rectangle_List_Off_Road, Rectangle_List_Reverse, Rectangle_List_Crash_Area

from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD 
from CONSTANTS import MAX_COORD 
from CONSTANTS import VEHICLE_LENGTH 
from CONSTANTS import VEHICLE_WIDTH 
from CONSTANTS import VEHICLE_ANGLE 
from CONSTANTS import VEHICLE_DIAGONAL
# from CONSTANTS import VEHICLE_HALF_SIZE
from CONSTANTS import VEHICLE_ILLUSTRATION_YAW_ANGLE_SIZE
from CONSTANTS import SINGLE_LANE_WIDTH
from CONSTANTS import SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS
from CONSTANTS import STRIKE_LENGTH
from CONSTANTS import STRAIGHT_LENGTH 
from CONSTANTS import INTERSECTION_HALF_SIZE
from CONSTANTS import REVERSE_DRIVING_LENGTH
from CONSTANTS import min_positionx
from CONSTANTS import max_positionx
from CONSTANTS import min_positiony
from CONSTANTS import max_positiony






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
    OCCUPIED_MID_LANE_LINE = False

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
        self.max_speed = 12.00 * SCALE
        self.goal_min_positionx = min_positionx
        self.goal_max_positionx = max_positionx
        # straight goal
        self.goal_straight_positiony = max_positiony

        self.mid_goal_front_positiony = np.arange(
            min_positiony + 35, -SINGLE_LANE_WIDTH,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_front_space_y = self.mid_goal_front_positiony.shape[0]

        self.mid_goal_intersection_positiony = np.arange(
            -INTERSECTION_HALF_SIZE / 2, 2 * INTERSECTION_HALF_SIZE,
            INTERSECTION_HALF_SIZE, dtype=np.float32)
        self.mid_goal_intersection_space_y = self.mid_goal_intersection_positiony.shape[0]

        self.mid_goal_rear_positiony = np.arange(
            INTERSECTION_HALF_SIZE + 16, max_positiony - 5,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_rear_space_y = self.mid_goal_rear_positiony.shape[0]

        # turn left goal
        self.goal_turn_left_positionx = min_positionx

        self.mid_goal_turn_left_angle = np.arange(
            math.pi/16, 5*math.pi/8, math.pi/8, dtype=np.float32)
        self.mid_goal_turn_left_space_angle = self.mid_goal_turn_left_angle.shape[0]

        self.mid_goal_rear_positionx = np.arange(
            -INTERSECTION_HALF_SIZE - 16, min_positionx + 5,
            -SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_rear_space_x = self.mid_goal_rear_positionx.shape[0]

        # turn right goal
        self.goal_turn_right_positionx = max_positionx

        self.mid_goal_turn_right_angle = np.arange(
            15*math.pi/16, 3*math.pi/8, -math.pi/8, dtype=np.float32)
        self.mid_goal_turn_right_space_angle = self.mid_goal_turn_right_angle.shape[0]

        self.mid_goal_front_positionx = np.arange(
            INTERSECTION_HALF_SIZE + 16, max_positionx - 5,
            SEPARATE_STRAIGHT_DISTANCE_OF_REWARDS, dtype=np.float32)
        self.mid_goal_front_space_x = self.mid_goal_front_positionx.shape[0]

        self.judgement_space = 2
        # self.risk_space = 4
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
                'distance_2_mid_lane_line': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(1,), dtype=np.float32),
                # 'risk_off_road': Discrete(self.risk_space),
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

        # 前两步判断出初始车辆状态，利用初始状态辨别出逆行区域
        if self.episode_step == 1:
            self.vehicle.init_state = self.vehicle.state

        self.rectangle_list_reverse.judgement(self.vehicle.init_state)

        # 更新直道的之前动作的最大最小坐标值(intersection中之前动作不更新)
        self.vehicle.UpdatePreExtremeValue()

        reward = 0.0

        terminated = False
        break_out = False
        self.OCCUPIED_MID_LANE_LINE = False

        # self.NEXT_1_IN_ROAD = True
        # self.NEXT_2_IN_ROAD = True
        # self.Time_to_off_road = 3

        # 进入道路外，终止训练，并给予-500分惩罚
        for rec in range(self.rectangle_list_off_road.RectangleList.size):
            if ((self.vehicle.max_vertex_x < self.rectangle_list_off_road.RectangleList[0][rec].min_x) or
                    (self.vehicle.min_vertex_x > self.rectangle_list_off_road.RectangleList[0][rec].max_x) or
                    (self.vehicle.max_vertex_y < self.rectangle_list_off_road.RectangleList[0][rec].min_y) or
                    (self.vehicle.min_vertex_y > self.rectangle_list_off_road.RectangleList[0][rec].max_y)):
                self.JUDGEMENT_IN_ROAD = True
            else:
                self.JUDGEMENT_IN_ROAD = False
                break
        # 进入逆行车道，终止训练，并给予-500分惩罚
        if self.JUDGEMENT_IN_ROAD == True:
            for rec1 in range(self.rectangle_list_reverse.RectangleList.size):
                if ((self.vehicle.max_vertex_x < self.rectangle_list_reverse.RectangleList[0][rec1].min_x) or
                        (self.vehicle.min_vertex_x > self.rectangle_list_reverse.RectangleList[0][rec1].max_x) or
                        (self.vehicle.max_vertex_y < self.rectangle_list_reverse.RectangleList[0][rec1].min_y) or
                        (self.vehicle.min_vertex_y > self.rectangle_list_reverse.RectangleList[0][rec1].max_y)):
                    self.JUDGEMENT_IN_ROAD = True
                else:
                    self.JUDGEMENT_IN_ROAD = False
                    break
        elif self.JUDGEMENT_IN_ROAD == False:
            self.JUDGEMENT_IN_ROAD = False


        # # 按当前路径下一步进入逆行车道或者道路外
        # if self.vehicle.next1_edge3 <= MIN_COORD + STRAIGHT_LENGTH:
        #     if self.vehicle.next1_edge0 <= 0 or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_1_IN_ROAD = False
        # elif -INTERSECTION_HALF_SIZE < self.vehicle.next1_edge3 and self.vehicle.next1_edge1 < 0:
        #     if self.vehicle.next1_edge0 <= -INTERSECTION_HALF_SIZE:
        #         self.NEXT_1_IN_ROAD = False
        # elif self.vehicle.next1_edge1 >= 0 and self.vehicle.next1_edge3 <= 0:
        #     if self.vehicle.next1_edge0 <= -INTERSECTION_HALF_SIZE or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_1_IN_ROAD = False
        # elif self.vehicle.next1_edge3 > 0 and self.vehicle.next1_edge1 < INTERSECTION_HALF_SIZE:
        #     if self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_1_IN_ROAD = False
        # elif self.vehicle.next1_edge1 >= MAX_COORD - STRAIGHT_LENGTH:
        #     if self.vehicle.next1_edge0 <= 0 or self.vehicle.next1_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_1_IN_ROAD = False
        #
        # # 按当前路径下两步进入逆行车道或者道路外
        # if self.vehicle.next2_edge3 <= MIN_COORD + STRAIGHT_LENGTH:
        #     if self.vehicle.next2_edge0 <= 0 or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_2_IN_ROAD = False
        # elif -INTERSECTION_HALF_SIZE < self.vehicle.next2_edge3 and self.vehicle.next2_edge1 < 0:
        #     if self.vehicle.next2_edge0 <= -INTERSECTION_HALF_SIZE:
        #         self.NEXT_2_IN_ROAD = False
        # elif self.vehicle.next2_edge1 >= 0 and self.vehicle.next2_edge3 <= 0:
        #     if self.vehicle.next2_edge0 <= -INTERSECTION_HALF_SIZE or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_2_IN_ROAD = False
        # elif self.vehicle.next2_edge3 > 0 and self.vehicle.next2_edge1 < INTERSECTION_HALF_SIZE:
        #     if self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_2_IN_ROAD = False
        # elif self.vehicle.next2_edge1 >= MAX_COORD - STRAIGHT_LENGTH:
        #     if self.vehicle.next2_edge0 <= 0 or self.vehicle.next2_edge2 >= INTERSECTION_HALF_SIZE:
        #         self.NEXT_2_IN_ROAD = False

        # # 主车进入到车祸区域，终止训练，并给予-500分惩罚
        # if self.JUDGEMENT_IN_ROAD == True:
        #     for rec in range(self.rectangle_list_crash_area.RectangleList.size):
        #         if ((self.vehicle.max_vertex_x < self.rectangle_list_crash_area.RectangleList[0][rec].min_x) or
        #                 (self.vehicle.min_vertex_x > self.rectangle_list_crash_area.RectangleList[0][rec].max_x) or
        #                 (self.vehicle.max_vertex_y < self.rectangle_list_crash_area.RectangleList[0][rec].min_y) or
        #                 (self.vehicle.min_vertex_y > self.rectangle_list_crash_area.RectangleList[0][rec].max_y)):
        #             self.JUDGEMENT_IN_ROAD = True
        #         else:
        #             self.JUDGEMENT_IN_ROAD = False
        #             break
        # elif self.JUDGEMENT_IN_ROAD == False:
        #     self.JUDGEMENT_IN_ROAD = False


        # # 主车下一步进入到车祸区域
        # if -VEHICLE_HALF_SIZE <= self.vehicle.next1_x <= SINGLE_LANE_WIDTH + VEHICLE_HALF_SIZE and \
        #         -VEHICLE_HALF_SIZE - INTERSECTION_HALF_SIZE <= self.vehicle.next1_y <= VEHICLE_HALF_SIZE - SINGLE_LANE_WIDTH:
        #     self.NEXT_1_IN_ROAD = False
        # # 主车下两步进入到车祸区域
        # if -VEHICLE_HALF_SIZE <= self.vehicle.next2_x <= SINGLE_LANE_WIDTH + VEHICLE_HALF_SIZE and \
        #         -VEHICLE_HALF_SIZE - INTERSECTION_HALF_SIZE <= self.vehicle.next2_y <= VEHICLE_HALF_SIZE - SINGLE_LANE_WIDTH:
        #     self.NEXT_2_IN_ROAD = False

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
            self.vehicle.distance_to_off_road[0] = self.vehicle.min_vertex_x - 0 if self.vehicle.min_vertex_x > 0 else 0
            self.vehicle.distance_to_off_road[1] = self.vehicle.max_vertex_x - 2 * SINGLE_LANE_WIDTH \
                if self.vehicle.max_vertex_x < 2 * SINGLE_LANE_WIDTH else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.min_vertex_y <= MIN_COORD:
                self.JUDGEMENT_IN_ROAD = False
            # if self.vehicle.next1_edge3 <= MIN_COORD:
            #     self.NEXT_1_IN_ROAD = False
            # if self.vehicle.next2_edge3 <= MIN_COORD:
            #     self.NEXT_2_IN_ROAD = False
            # 如果第一次或再次进入直行道，更新阶段目标空间和计数器
            if self.episode_step == 1 and self.vehicle.y < 0:
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_y), dtype=np.int32)
            elif self.vehicle.pre_state != 'straight_y+' and self.vehicle.y > 0:
                self.CIRCLE_COUNT = 0
                self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_rear_space_y), dtype=np.int32)
            # 计算front阶段目标的相对位置，并获得阶段reward和distance_2_goal
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
                                0 <= self.vehicle.min_vertex_x and self.vehicle.max_vertex_x <= 2*SINGLE_LANE_WIDTH:
                        reward += 200.0
                        self.ARRIVE_AT_MID_GOAL[num] = 1
                        self.distance_2_goal = 0
            # 计算rear阶段目标的相对位置，并获得阶段reward和distance_2_goal
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
                                0 <= self.vehicle.min_vertex_x and self.vehicle.max_vertex_x <= 2*SINGLE_LANE_WIDTH:
                        reward += 200.0
                        self.ARRIVE_AT_MID_GOAL[num] = 1
                        self.distance_2_goal = 0
                if self.mid_goal_rear_positiony[self.mid_goal_rear_space_y - 1] <= self.vehicle.y <= self.goal_straight_positiony \
                        and num >= self.mid_goal_rear_space_y:
                    self.distance_2_goal = abs(self.vehicle.y - self.goal_straight_positiony)
            # 计算智能体距mid-road-line的距离
            self.vehicle.distance_to_mid_lane_line = abs(self.vehicle.x - SINGLE_LANE_WIDTH)
            # 如果车辆范围与中线有交点，给予一个惩罚
            if self.vehicle.min_vertex_x < SINGLE_LANE_WIDTH < self.vehicle.max_vertex_x:
                self.OCCUPIED_MID_LANE_LINE = True

        elif self.vehicle.state == 'straight_x+':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.max_pre_x) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.x > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle_track.straight_frontier_coord_positive_x - self.vehicle.x) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = 0 - self.vehicle.max_vertex_y if self.vehicle.max_vertex_y > 0 else 0
            self.vehicle.distance_to_off_road[1] = 2*SINGLE_LANE_WIDTH - self.vehicle.min_vertex_y \
                if self.vehicle.min_vertex_y < 2*SINGLE_LANE_WIDTH else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.min_vertex_x <= MIN_COORD:
                self.JUDGEMENT_IN_ROAD = False
            # if self.vehicle.next1_edge0 <= MIN_COORD:
            #     self.NEXT_1_IN_ROAD = False
            # if self.vehicle.next2_edge0 <= MIN_COORD:
            #     self.NEXT_2_IN_ROAD = False
            # 如果再次进入直行道，更新阶段目标空间和计数器
            if self.vehicle.pre_state != 'straight_x+' and self.vehicle.x > 0:
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
                        0 >= self.vehicle.max_vertex_y and self.vehicle.min_vertex_y >= -2*SINGLE_LANE_WIDTH:
                    reward += 200.0
                    self.ARRIVE_AT_MID_GOAL[num] = 1
                    self.distance_2_goal = 0
            if self.mid_goal_front_positionx[self.mid_goal_front_space_x-1] <= self.vehicle.x <= self.goal_turn_right_positionx \
                    and num >= self.mid_goal_front_space_x:
                self.distance_2_goal = abs(self.vehicle.x - self.goal_turn_right_positionx)
            # 计算智能体距mid-road-line的距离
            self.vehicle.distance_to_mid_lane_line = abs(self.vehicle.y + SINGLE_LANE_WIDTH)
            # 如果车辆范围与中线有交点，给予一个惩罚
            if self.vehicle.min_vertex_y < -SINGLE_LANE_WIDTH < self.vehicle.max_vertex_y:
                self.OCCUPIED_MID_LANE_LINE = True
        #
        # elif self.vehicle.state == 'straight_y-':
        #     # 更新智能体当前track的直道前沿坐标值
        #     self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
        #     # 前进track的奖励
        #     reward += abs(self.vehicle_track.straight_frontier_coord_negative_y - self.vehicle.min_pre_y) * 1
        #     # 如果逆行，逆行的惩罚
        #     if self.vehicle.y - self.vehicle_track.straight_frontier_coord_negative_y > REVERSE_DRIVING_LENGTH:
        #         reward -= abs(self.vehicle.y - self.vehicle_track.straight_frontier_coord_negative_y) * 1
        #     # 计算智能体距off-road最近距离
        #     self.vehicle.distance_to_off_road[0] = 0 - self.vehicle.edge2 if self.vehicle.edge2 > 0 else 0
        #     self.vehicle.distance_to_off_road[1] = INTERSECTION_HALF_SIZE - self.vehicle.edge0 \
        #         if self.vehicle.edge0 < INTERSECTION_HALF_SIZE else 0
        #     # 制定出发时车辆不能逆行回到出生点的边界
        #     if self.vehicle.edge1 >= MAX_COORD:
        #         self.JUDGEMENT_IN_ROAD = False
        #     if self.vehicle.next1_edge1 >= MAX_COORD:
        #         self.NEXT_1_IN_ROAD = False
        #     if self.vehicle.next2_edge1 >= MAX_COORD:
        #         self.NEXT_2_IN_ROAD = False

        elif self.vehicle.state == 'straight_x-':
            # 更新智能体当前track的直道前沿坐标值
            self.vehicle_track.StraightFrontier(self.vehicle.x, self.vehicle.y, self.vehicle.state)
            # 前进track的奖励
            reward += abs(self.vehicle_track.straight_frontier_coord_negative_x - self.vehicle.min_pre_x) * 1
            # 如果逆行，逆行的惩罚
            if self.vehicle.x - self.vehicle_track.straight_frontier_coord_negative_x > REVERSE_DRIVING_LENGTH:
                reward -= abs(self.vehicle.x - self.vehicle_track.straight_frontier_coord_negative_x) * 1
            # 计算智能体距off-road最近距离
            self.vehicle.distance_to_off_road[0] = self.vehicle.min_vertex_y - 0 if self.vehicle.min_vertex_y > 0 else 0
            self.vehicle.distance_to_off_road[1] = self.vehicle.max_vertex_y - 2*SINGLE_LANE_WIDTH \
                if self.vehicle.max_vertex_y < 2*SINGLE_LANE_WIDTH else 0
            # 制定出发时车辆不能逆行回到出生点的边界
            if self.vehicle.max_vertex_x >= MAX_COORD:
                self.JUDGEMENT_IN_ROAD = False
            # if self.vehicle.next1_edge2 >= MAX_COORD:
            #     self.NEXT_1_IN_ROAD = False
            # if self.vehicle.next2_edge2 >= MAX_COORD:
            #     self.NEXT_2_IN_ROAD = False
            # 如果再次进入直行道，更新阶段目标空间和计数器
            if self.vehicle.pre_state != 'straight_x-' and self.vehicle.x < 0:
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
                        0 <= self.vehicle.min_vertex_y and self.vehicle.max_vertex_y <= 2*SINGLE_LANE_WIDTH:
                    reward += 200.0
                    self.ARRIVE_AT_MID_GOAL[num] = 1
                    self.distance_2_goal = 0
            if self.mid_goal_rear_positionx[self.mid_goal_rear_space_x-1] >= self.vehicle.x >= self.goal_turn_left_positionx \
                    and num >= self.mid_goal_rear_space_x:
                self.distance_2_goal = abs(self.vehicle.x - self.goal_turn_left_positionx)
            # 计算智能体距mid-road-line的距离
            self.vehicle.distance_to_mid_lane_line = abs(self.vehicle.y - SINGLE_LANE_WIDTH)
            # 如果车辆范围与中线有交点，给予一个惩罚
            if self.vehicle.min_vertex_y < SINGLE_LANE_WIDTH < self.vehicle.max_vertex_y:
                self.OCCUPIED_MID_LANE_LINE = True

        # 十字路口内选择
        self.vehicle.intersection_steering_choice = -1  # 0: straight; -1: turn left;  1: turn right

        # 进入十字路口范围内，根据intersection_steering_choice，绘制track路线图，带有rewards，转弯部分利用极坐标系绘制
        if self.vehicle.state == 'intersection':
            # 更新车辆在十字路口内的track的细分阶段
            self.vehicle_track.IntersectionJudgment(self.vehicle.intersection_steering_choice)
            # 更新车辆距离中线的距离(定义为最佳远离距离)
            self.vehicle.distance_to_mid_lane_line = SINGLE_LANE_WIDTH / 2
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
                if 0 <= self.vehicle.min_vertex_x and self.vehicle.max_vertex_x <= 2*SINGLE_LANE_WIDTH:
                    reward += abs(self.vehicle_track.straight_frontier_coord_positive_y - self.vehicle.max_pre_y) * 1
                # 计算智能体距off-road最近距离
                self.vehicle.distance_to_off_road[0] = self.vehicle.min_vertex_x - 0 if self.vehicle.min_vertex_x > 0 else 0
                self.vehicle.distance_to_off_road[1] = self.vehicle.max_vertex_x - 2*SINGLE_LANE_WIDTH \
                    if self.vehicle.max_vertex_x < 2*SINGLE_LANE_WIDTH else 0
                # 制定道路范围内边界
                if self.vehicle.min_vertex_x < 0 or self.vehicle.max_vertex_x > 2*SINGLE_LANE_WIDTH:
                    self.JUDGEMENT_IN_ROAD = False
                # if self.vehicle.next1_edge0 < 0 or self.vehicle.next1_edge2 > INTERSECTION_HALF_SIZE:
                #     self.NEXT_1_IN_ROAD = False
                # if self.vehicle.next2_edge0 < 0 or self.vehicle.next2_edge2 > INTERSECTION_HALF_SIZE:
                #     self.NEXT_2_IN_ROAD = False
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
                            0 <= self.vehicle.min_vertex_x and self.vehicle.max_vertex_x <= 2*SINGLE_LANE_WIDTH:
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
                # 第1阶段：angle = [0, atan1/2], min_radius = SINGLE_LANE_WIDTH, max_radius = 2*INTERSECTION_HALF_SIZE/cos(angle)
                if 0 <= self.vehicle.polar_angle < math.atan(1.0 / 2.0):
                    if SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - SINGLE_LANE_WIDTH \
                        if self.vehicle.polar_radius_min_edge > SINGLE_LANE_WIDTH else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle) \
                        if self.vehicle.polar_radius_max_edge < 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < SINGLE_LANE_WIDTH or \
                            self.vehicle.polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.cos(self.vehicle.polar_angle):
                        self.JUDGEMENT_IN_ROAD = False
                    # if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next1_polar_radius_max_edge > 2 * INTERSECTION_HALF_SIZE/math.cos(self.vehicle.next1_polar_angle):
                    #     self.NEXT_1_IN_ROAD = False
                    # if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next2_polar_radius_max_edge > 2 * INTERSECTION_HALF_SIZE/math.cos(self.vehicle.next2_polar_angle):
                    #     self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
                                self.vehicle.polar_radius_max_edge <= 2 * INTERSECTION_HALF_SIZE / math.cos(self.vehicle.polar_angle):
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0
                # 第2阶段：angle = (atan1/2, atan2), min_radius = SINGLE_LANE_WIDTH, max_radius = INTERSECTION_HALF_SIZE*sqrt(5.0)
                elif math.atan(1.0 / 2.0) <= self.vehicle.polar_angle <= math.atan(2.0):
                    if SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - SINGLE_LANE_WIDTH \
                        if self.vehicle.polar_radius_min_edge > SINGLE_LANE_WIDTH else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - INTERSECTION_HALF_SIZE * math.sqrt ( 5.0 ) \
                        if self.vehicle.polar_radius_max_edge < INTERSECTION_HALF_SIZE * math.sqrt(5.0) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < SINGLE_LANE_WIDTH or \
                            self.vehicle.polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                        self.JUDGEMENT_IN_ROAD = False
                    # if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next1_polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                    #     self.NEXT_1_IN_ROAD = False
                    # if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next2_polar_radius_max_edge > INTERSECTION_HALF_SIZE * math.sqrt(5.0):
                    #     self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
                                self.vehicle.polar_radius_max_edge <= INTERSECTION_HALF_SIZE * math.sqrt ( 5.0 ):
                            reward += 200.0
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0
                # 第3阶段：angle = (atan2, pi/2], min_radius = SINGLE_LANE_WIDTH, max_radius = 2*INTERSECTION_HALF_SIZE/sin(angle)
                elif math.atan(2.0) < self.vehicle.polar_angle <= math.pi/2:
                    if SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
                            self.vehicle.polar_radius_max_edge <= 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle):
                        reward += abs(
                            self.vehicle_track.steering_frontier_polar_coord_positive_angle - self.vehicle.max_pre_polar_angle ) * 100.0
                    # 计算智能体距off-road最近距离
                    self.vehicle.distance_to_off_road[0] = self.vehicle.polar_radius_min_edge - SINGLE_LANE_WIDTH \
                        if self.vehicle.polar_radius_min_edge > SINGLE_LANE_WIDTH else 0
                    self.vehicle.distance_to_off_road[1] = self.vehicle.polar_radius_max_edge - 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle) \
                        if self.vehicle.polar_radius_max_edge < 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle) else 0
                    # 制定道路范围内边界
                    if self.vehicle.polar_radius_min_edge < SINGLE_LANE_WIDTH or \
                            self.vehicle.polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.polar_angle):
                        self.JUDGEMENT_IN_ROAD = False
                    # if self.vehicle.next1_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next1_polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.next1_polar_angle):
                    #     self.NEXT_1_IN_ROAD = False
                    # if self.vehicle.next2_polar_radius_min_edge < INTERSECTION_HALF_SIZE or \
                    #         self.vehicle.next2_polar_radius_max_edge > 2*INTERSECTION_HALF_SIZE/math.sin(self.vehicle.next2_polar_angle):
                    #     self.NEXT_2_IN_ROAD = False
                    # 计算与阶段目标的相对位置，并获得阶段reward和distance_2_goal
                    num = self.CIRCLE_COUNT
                    if num < self.mid_goal_turn_left_space_angle:
                        if self.vehicle.polar_angle < self.mid_goal_turn_left_angle[num] and \
                                self.ARRIVE_AT_MID_GOAL[num] == 0:
                            self.distance_2_goal = abs(self.vehicle.polar_angle - self.mid_goal_turn_left_angle[num]) * 100
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num]:
                            self.CIRCLE_COUNT = num + 1
                        if self.vehicle.polar_angle >= self.mid_goal_turn_left_angle[num] and self.ARRIVE_AT_MID_GOAL[num] == 0 and \
                                SINGLE_LANE_WIDTH <= self.vehicle.polar_radius_min_edge and \
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
                # if self.vehicle.next1_polar_radius_min_edge < 0 or \
                #         self.vehicle.next1_polar_radius_max_edge > INTERSECTION_HALF_SIZE:
                #     self.NEXT_1_IN_ROAD = False
                # if self.vehicle.next2_polar_radius_min_edge < 0 or \
                #         self.vehicle.next2_polar_radius_max_edge > INTERSECTION_HALF_SIZE:
                #     self.NEXT_2_IN_ROAD = False
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
        # if self.NEXT_2_IN_ROAD == False:
        #     self.Time_to_off_road = 2
        # if self.NEXT_1_IN_ROAD == False:
        #     self.Time_to_off_road = 1
        if self.JUDGEMENT_IN_ROAD == False:
            # self.Time_to_off_road = 0
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
        reward -= math.pow((real_action[0]+real_action[1]), 2) * 1

        #如果车辆压到中线，返回一个小惩罚
        if self.OCCUPIED_MID_LANE_LINE == True:
            reward -= 30

        if break_out:
            reward += -500.0
            terminated = True

        if self.goal:
            reward += 1000.0
            terminated = True

        self.vehicle_position = np.array(
            [self.vehicle.x, self.vehicle.y], dtype=np.float32)
        self.vehicle_speed_yawangle = np.array(
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
        self.distance_to_mid_lane_line = np.array([self.vehicle.distance_to_mid_lane_line], dtype=np.float32)

        # self.signal_stop_position = np.array([self.signal_stop.x, self.signal_stop.y], dtype=np.int32)
        # self.signal_stop_state = np.array([phase, countdown], dtype=np.int32)

        print(self.episode_step, ':', 'Action:', action, ';',
              'Position:', self.vehicle_position, ';', 'Speed_YawAngle:', self.vehicle_speed_yawangle, ';',
              # 'Relative_distance_to_goal:', self.relative_distance_to_goal, ';',
              'vehicle_state:', self.vehicle.state, ';',
              # 'Judgement_in_road:', self.JUDGEMENT_IN_ROAD, ';',
              # 'Distance_to_off_road:', self.vehicle.distance_to_off_road, ';',
              'Distance_to_mid_lane_line', self.vehicle.distance_to_mid_lane_line, ';',
              # 'Polar_radius:', self.vehicle.polar_radius, ';',
              # 'Polar_angle:', self.vehicle.polar_angle, ';'
              'reward:', reward
              )
        if reward < -100 or reward > 100:
          print(reward)

        self.state: dict = (
            {
                'agent_position': self.vehicle_position,
                'agent_speed_yawangle': self.vehicle_speed_yawangle,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_position': self.first_other_vehicle_position,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_position': self.second_other_vehicle_position,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'distance_2_mid_lane_line': self.distance_to_mid_lane_line,
                # 'risk_off_road': self.Time_to_off_road,
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
        self.rectangle_list_off_road = Rectangle_List_Off_Road()
        self.rectangle_list_reverse = Rectangle_List_Reverse()
        self.rectangle_list_crash_area = Rectangle_List_Crash_Area()

        self.vehicle_position = np.array(
            [self.vehicle.x, self.vehicle.y], dtype=np.float32)
        self.vehicle_speed_yawangle = np.array(
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
        self.distance_to_mid_lane_line = np.array([self.vehicle.distance_to_mid_lane_line], dtype=np.float32)

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
        self.OCCUPIED_MID_LANE_LINE = False
        # self.Time_to_off_road = 3
        self.CIRCLE_COUNT = 0
        self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_y), dtype=np.int32)

        # print(self.episode_step, ':', self.vehicle_position, self.vehicle_state, self.relative_distance_to_goal,
        #       self.JUDGEMENT_IN_ROAD, self.Time_to_off_road, self.vehicle.distance_to_off_road)

        self.state: dict = (
            {
                'agent_position': self.vehicle_position,
                'agent_speed_yawangle': self.vehicle_speed_yawangle,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_position': self.first_other_vehicle_position,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_position': self.second_other_vehicle_position,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'distance_2_mid_lane_line': self.distance_to_mid_lane_line,
                # 'risk_off_road': self.Time_to_off_road,
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
        # INT_VEHICLE_HALF_SIZE = int(round(VEHICLE_HALF_SIZE, 0))
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
        INT_VEHICLE_Y = int(round(self.vehicle.y, 0))
        INT_VEHICLE_X = int(round(self.vehicle.x, 0))
        INT_VEHICLE_MAX_VERTEX_X = int(round(self.vehicle.max_vertex_x, 0))
        INT_VEHICLE_MIN_VERTEX_X = int(round(self.vehicle.min_vertex_x, 0))
        INT_VEHICLE_MAX_VERTEX_Y = int(round(self.vehicle.max_vertex_y, 0))
        INT_VEHICLE_MIN_VERTEX_Y = int(round(self.vehicle.min_vertex_y, 0))


        INT_FIRST_OTHER_VEHICLE_Y = int(round(self.first_other_vehicle.y, 0))
        INT_FIRST_OTHER_VEHICLE_X = int(round(self.first_other_vehicle.x, 0))
        INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_X = int(round(self.first_other_vehicle.max_vertex_x, 0))
        INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_X = int(round(self.first_other_vehicle.min_vertex_x, 0))
        INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_Y = int(round(self.first_other_vehicle.max_vertex_y, 0))
        INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_Y = int(round(self.first_other_vehicle.min_vertex_y, 0))

        INT_SECOND_OTHER_VEHICLE_Y = int(round(self.second_other_vehicle.y, 0))
        INT_SECOND_OTHER_VEHICLE_X = int(round(self.second_other_vehicle.x, 0))
        INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_X = int(round(self.second_other_vehicle.max_vertex_x, 0))
        INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_X = int(round(self.second_other_vehicle.min_vertex_x, 0))
        INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_Y = int(round(self.second_other_vehicle.max_vertex_y, 0))
        INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_Y = int(round(self.second_other_vehicle.min_vertex_y, 0))

        for x3 in range(INT_VEHICLE_MIN_VERTEX_X + INT_MAX_COORD, INT_VEHICLE_MAX_VERTEX_X + INT_MAX_COORD):
            for y3 in range(INT_VEHICLE_MIN_VERTEX_Y + INT_MAX_COORD, INT_VEHICLE_MAX_VERTEX_Y + INT_MAX_COORD):
                env[-y3][x3] = self.d[self.VEHICLE_N]
        if INT_FIRST_OTHER_VEHICLE_Y < INT_MAX_COORD + max_positiony:
            for x13 in range(INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_X + INT_MAX_COORD, INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_X + INT_MAX_COORD):
                for y13 in range(INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_Y + INT_MAX_COORD, INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_Y + INT_MAX_COORD):
                    env[-y13][x13] = self.d[self.VEHICLE_OTHER_N]
        if INT_SECOND_OTHER_VEHICLE_Y < INT_MAX_COORD + max_positiony:
            for x23 in range(INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_X + INT_MAX_COORD, INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_X + INT_MAX_COORD):
                for y23 in range(INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_Y + INT_MAX_COORD, INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_Y + INT_MAX_COORD):
                    env[-y23][x23] = self.d[self.VEHICLE_OTHER_N]

        for x5 in range(0, int(2.0 * VEHICLE_ILLUSTRATION_YAW_ANGLE_SIZE)):
            x = INT_MAX_COORD - INT_VEHICLE_Y - int(x5 * math.cos(self.vehicle.yaw_angle))
            y = INT_MAX_COORD + INT_VEHICLE_X + int(x5 * math.sin(self.vehicle.yaw_angle))
            if x >= INT_SIZE:
                x = INT_SIZE - 1
            if y >= INT_SIZE:
                y = INT_SIZE - 1
            env[x][y] = self.d[self.RED_LIGHT_N]

        # # 画出车祸区域
        # for x6 in range(INT_MAX_COORD + INT_SINGLE_LANE_WIDTH, INT_MAX_COORD + 2 * INT_SINGLE_LANE_WIDTH):
        #     for y6 in range(INT_MAX_COORD, INT_MAX_COORD + INT_SINGLE_LANE_WIDTH):
        #         env[x6][y6] = self.d[self.RED_LIGHT_N]

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
