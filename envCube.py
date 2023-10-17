
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
from Remote_Cube import Remote_Cube
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
from Track import Track
import Rectangle
from Rectangle import Rectangle, Rectangle_List_Off_Road, Rectangle_List_Reverse, Rectangle_List_Crash_Area
from Rectangle_Show import Rectangle_Show, Rectangle_List_Off_Road_Show, Rectangle_List_Reverse_Show, Rectangle_List_Crash_Area_Show
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from UKF_Prediction import UKF_Prediction

from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD 
from CONSTANTS import MAX_COORD, SEPARATE_SIZE, GRAY_SPACE
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
from CONSTANTS import NUMBER_REMOTE_VEHICLES
from CONSTANTS import PI
from Image_Map_Observation import Image_Map_Observation
from Image_Map_Observation_Show import Image_Map_Observation_Show
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
# from readcsv import float_csv_data
from client import *


# 环境类
class envCube(gym.Env):

    TCPClient = client()
    TCPClient.tcp_client()
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

        self.low_steering_state = np.array([self.min_action_steering/self.max_action_steering], dtype=np.float32)
        self.high_steering_state = np.array([self.max_action_steering/self.max_action_steering], dtype=np.float32)

        super(envCube, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(self.ACTION_SPACE_VALUES)
        # 0: do noting; 1: gas, 2: brake, 3: turn left 4: turn right

        self.observation_space = Dict(
            {
                'agent_speed': Box(low=self.min_speed, high=self.max_speed, shape=(1,), dtype=np.float32),
                'agent_yawangle': Box(low=self.low_steering_state, high=self.high_steering_state, dtype=np.float32),
                'relative_position_2_danger': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),
                'first_other_vehicle_relative_position': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),
                'first_other_vehicle_speed': Box(self.min_speed, self.max_speed, shape=(1,), dtype=np.float32),
                'second_other_vehicle_relative_position': Box(2*MIN_COORD-1, 2*MAX_COORD+1, shape=(2,), dtype=np.float32),
                'second_other_vehicle_speed': Box(self.min_speed, self.max_speed, shape=(1,), dtype=np.float32),
                'relative_distance_2_goal': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(1,), dtype=np.float32),
                'judgement_in_road': Discrete(self.judgement_space),
                'distance_2_mid_lane_line': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(1,), dtype=np.float32),
                'distance_2_nearest_off_road': Box(2*MIN_COORD-1.0, 2*MAX_COORD+1.0, shape=(2,), dtype=np.float32),
                'bird_eye_view_gray_image': Box(low=0, high=GRAY_SPACE, shape=(int(SEPARATE_SIZE), int(SEPARATE_SIZE), 1), dtype=np.uint8),

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

        data = {
            'TickId': self.episode_step,
            'X': self.vehicle.x / SCALE,
            'Y': self.vehicle.y / SCALE,
            'YawAngle': self.vehicle.yaw_angle
        }
        # while True:
        #     step = 0
        #     self.TCPClient.send_data (self.TCPClient.client_socket,data)
        #     self.TCPClient.receive_data ( self.TCPClient.client_socket )
        #     if step >= 3:
        #         break
        #     # 等待一段时间后继续循环
        #     step += 1
        #     time.sleep(1)
        self.TCPClient.send_data(self.TCPClient.client_socket, data)
        self.TCPClient.receive_data(self.TCPClient.client_socket)

        if self.TCPClient.Remotes:
            for i in range(NUMBER_REMOTE_VEHICLES):
                self.remote_vehicles[i].x = self.TCPClient.Remotes[i]['X'] * SCALE
                self.remote_vehicles[i].y = self.TCPClient.Remotes[i]['Y'] * SCALE
                if self.TCPClient.Remotes[i]['YawAngle'] > 180:
                    self.TCPClient.Remotes[i]['YawAngle'] = self.TCPClient.Remotes[i]['YawAngle'] - 360
                self.remote_vehicles[i].yaw_angle = self.TCPClient.Remotes[i]['YawAngle']/180*PI
                self.remote_vehicles[i].velocity = self.TCPClient.Remotes[i]['Speed'] * SCALE



        # # 从训练表格中更新远车状态
        # self.RV_data = float_csv_data[self.episode_step]
        # self.first_other_vehicle.x = self.RV_data[2] * SCALE
        # self.first_other_vehicle.y = self.RV_data[3] * SCALE
        # self.first_other_vehicle.yaw_angle = self.RV_data[4]
        # self.first_other_vehicle.velocity = self.RV_data[5] * SCALE
        #
        # self.second_other_vehicle.x = self.RV_data[7] * SCALE
        # self.second_other_vehicle.y = self.RV_data[8] * SCALE
        # self.second_other_vehicle.yaw_angle = self.RV_data[9]
        # self.second_other_vehicle.velocity = self.RV_data[10] * SCALE
        for i in range(NUMBER_REMOTE_VEHICLES):
            self.remote_vehicles[i].move()

        # phase = self.signal_stop.signalphase(self.episode_step)
        # countdown = self.signal_stop.signalcountdown(phase)

        # 判断智能体处在什么阶段(其中十字路口中的统一模糊判断为intersection)
        self.vehicle.judgment()

        # 前两步判断出初始车辆状态，利用初始状态辨别出逆行区域
        if self.episode_step == 1:
            self.vehicle.init_state = self.vehicle.state

        self.rectangle_list_reverse.judgement(self.vehicle.init_state)
        self.rectangle_list_reverse_show.judgement(self.vehicle.init_state)

        # observation里画出逆行区域
        self.image_map_observation.reverse(self.rectangle_list_reverse.RectangleList)
        self.image_map_observation_show.reverse(self.rectangle_list_reverse_show.RectangleList_Show)

        # observation里画出远车范围

        for i in range(2):  # 处理前两辆车辆
            self.image_map_observation.RemoteVehicle(
                self.remote_vehicles[i].max_vertex_x, self.remote_vehicles[i].min_vertex_x,
                self.remote_vehicles[i].max_vertex_y, self.remote_vehicles[i].min_vertex_y,
                self.remote_vehicles[i].pre_max_vertex_x, self.remote_vehicles[i].pre_min_vertex_x,
                self.remote_vehicles[i].pre_max_vertex_y, self.remote_vehicles[i].pre_min_vertex_y
            )

        # observation show 里画出远车范围
        for i in range(NUMBER_REMOTE_VEHICLES):
            self.image_map_observation_show.RemoteVehicle(
                self.remote_vehicles[i].max_vertex_x, self.remote_vehicles[i].min_vertex_x,
                self.remote_vehicles[i].max_vertex_y, self.remote_vehicles[i].min_vertex_y,
                self.remote_vehicles[i].pre_max_vertex_x, self.remote_vehicles[i].pre_min_vertex_x,
                self.remote_vehicles[i].pre_max_vertex_y, self.remote_vehicles[i].pre_min_vertex_y
            )

        # observation里画出智能体范围
        self.image_map_observation.agent(self.vehicle.max_vertex_x, self.vehicle.min_vertex_x,
                                         self.vehicle.max_vertex_y, self.vehicle.min_vertex_y,
                                         self.vehicle.pre_max_vertex_x, self.vehicle.pre_min_vertex_x,
                                         self.vehicle.pre_max_vertex_y, self.vehicle.pre_min_vertex_y)

        self.image_map_observation_show.agent(self.vehicle.max_vertex_x, self.vehicle.min_vertex_x,
                                         self.vehicle.max_vertex_y, self.vehicle.min_vertex_y,
                                         self.vehicle.pre_max_vertex_x, self.vehicle.pre_min_vertex_x,
                                         self.vehicle.pre_max_vertex_y, self.vehicle.pre_min_vertex_y)

        # observation画出以主车为中心的灰度图区域
        self.image_map_observation.Separate_Map(self.vehicle.max_vertex_x, self.vehicle.min_vertex_x,
                                               self.vehicle.max_vertex_y, self.vehicle.min_vertex_y)
        self.image_map_observation_show.Separate_Map(self.vehicle.max_vertex_x, self.vehicle.min_vertex_x,
                                               self.vehicle.max_vertex_y, self.vehicle.min_vertex_y)

        # 信号灯信息在灰度图表示
        if self.signal_light_stop_line.countdown_phase1 > self.episode_step:
            self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x1,
                                                      self.signal_light_stop_line.y1, True)
            # self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x2,
            #                                           self.signal_light_stop_line.y2, True)
        elif self.episode_step == self.signal_light_stop_line.countdown_phase1:
            self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x1,
                                                      self.signal_light_stop_line.y1, False)
            # self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x2,
            #                                           self.signal_light_stop_line.y2, False)
        # if self.signal_light_stop_line.countdown_phase1 <= self.episode_step < self.signal_light_stop_line.countdown_phase123:
        #     self.image_map_observation_show.StopLineY(self.signal_light_stop_line.x3,
        #                                               self.signal_light_stop_line.y3, True)
        #     self.image_map_observation_show.StopLineY(self.signal_light_stop_line.x4,
        #                                               self.signal_light_stop_line.y4, True)
        # elif self.episode_step == self.signal_light_stop_line.countdown_phase123:
        #     self.image_map_observation_show.StopLineY(self.signal_light_stop_line.x3,
        #                                               self.signal_light_stop_line.y3, False)
        #     self.image_map_observation_show.StopLineY(self.signal_light_stop_line.x4,
        #                                               self.signal_light_stop_line.y4, False)
        if self.signal_light_stop_line.countdown_phase123 <= self.episode_step < self.signal_light_stop_line.countdown_phase1231:
            self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x1,
                                                      self.signal_light_stop_line.y1, True)
            # self.image_map_observation_show.StopLineX(self.signal_light_stop_line.x2,
            #                                           self.signal_light_stop_line.y2, True)

        # UKF prediction

        if self.episode_step == 1:
            for i in range(NUMBER_REMOTE_VEHICLES):
                self.ukf[i].x_prior[0] = self.remote_vehicles[i].x
                self.ukf[i].x_prior[1] = self.remote_vehicles[i].y
                self.ukf[i].x_prior[2] = self.remote_vehicles[i].yaw_angle
                self.ukf_prediction[i].initUKF(self.ukf[i], self.remote_vehicles[i].x, self.remote_vehicles[i].y,
                                        self.remote_vehicles[i].yaw_angle)

                self.predicted_remote_vehicles[i].x = self.ukf[i].x_prior[0]
                self.predicted_remote_vehicles[i].y = self.ukf[i].x_prior[1]
                self.predicted_remote_vehicles[i].yaw_angle = self.ukf[i].x_prior[2]
                self.predicted_remote_vehicles[i].move()

        # self.predicted_remote_vehicles[0].x = self.remote_vehicles[0].x
        # self.predicted_remote_vehicles[0].y = self.remote_vehicles[0].y
        # self.predicted_remote_vehicles[0].yaw_angle = self.remote_vehicles[0].yaw_angle
        # self.predicted_remote_vehicles[0].velocity = self.remote_vehicles[0].velocity
        #
        # self.predicted_remote_vehicles[0].move()

        if self.episode_step != 1:
            for i in range(NUMBER_REMOTE_VEHICLES):
                self.ukf[i].update(z=([self.remote_vehicles[i].x,self.remote_vehicles[i].y,
                                    self.remote_vehicles[i].yaw_angle]))
        for i in range(NUMBER_REMOTE_VEHICLES):
            self.ukf[i].predict(v=self.remote_vehicles[i].velocity)

        for i in range(NUMBER_REMOTE_VEHICLES):
            self.predicted_remote_vehicles[i].x = self.ukf[i].x_prior[0]
            self.predicted_remote_vehicles[i].y = self.ukf[i].x_prior[1]
            self.predicted_remote_vehicles[i].yaw_angle = self.ukf[i].x_prior[2]
            self.predicted_remote_vehicles[i].move()

        #  prediction show
        self.predicted_image_map_observation_show.reverse(self.rectangle_list_reverse_show.RectangleList_Show)
        self.predicted_image_map_observation_show.offroad()
        for i in range(NUMBER_REMOTE_VEHICLES):
            self.predicted_image_map_observation_show.RemoteVehicle(
                self.predicted_remote_vehicles[i].max_vertex_x, self.predicted_remote_vehicles[i].min_vertex_x,
                self.predicted_remote_vehicles[i].max_vertex_y, self.predicted_remote_vehicles[i].min_vertex_y,
                self.predicted_remote_vehicles[i].pre_max_vertex_x, self.predicted_remote_vehicles[i].pre_min_vertex_x,
                self.predicted_remote_vehicles[i].pre_max_vertex_y, self.predicted_remote_vehicles[i].pre_min_vertex_y)



        # 更新直道的之前动作的最大最小坐标值(intersection中之前动作不更新)
        self.vehicle.UpdatePreExtremeValue()

        reward = 0.0

        terminated = False
        break_out = False
        self.OCCUPIED_MID_LANE_LINE = False

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

        # 主车进入到车祸区域，终止训练，并给予-500分惩罚
        if self.JUDGEMENT_IN_ROAD == True:
            for rec in range(self.rectangle_list_crash_area.RectangleList.size):
                if ((self.vehicle.max_vertex_x < self.rectangle_list_crash_area.RectangleList[0][rec].min_x) or
                        (self.vehicle.min_vertex_x > self.rectangle_list_crash_area.RectangleList[0][rec].max_x) or
                        (self.vehicle.max_vertex_y < self.rectangle_list_crash_area.RectangleList[0][rec].min_y) or
                        (self.vehicle.min_vertex_y > self.rectangle_list_crash_area.RectangleList[0][rec].max_y)):
                    self.JUDGEMENT_IN_ROAD = True
                else:
                    self.JUDGEMENT_IN_ROAD = False
                    break
        elif self.JUDGEMENT_IN_ROAD == False:
            self.JUDGEMENT_IN_ROAD = False

        # 两车相撞，终止训练，并给予-500分惩罚
        if self.vehicle.collision(self.remote_vehicles[0]) == 0 and self.vehicle.collision(self.remote_vehicles[1]) == 0:
            self.COLLISION = False
        if (self.vehicle.collision(self.remote_vehicles[0]) == 1 and self.COLLISION == False) or \
                (self.vehicle.collision(self.remote_vehicles[1]) == 1 and self.COLLISION == False):
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

        # 选择目的地 = intersection_steering_choice：
        self.vehicle.intersection_steering_choice = -1  # 0: straight; -1: turn left;  1: turn right
        # 根据目的地，限定车辆的行驶范围
        if self.vehicle.intersection_steering_choice == -1:
            if self.vehicle.y > INTERSECTION_HALF_SIZE or self.vehicle.x > INTERSECTION_HALF_SIZE:
                break_out = True
        elif self.vehicle.intersection_steering_choice == 0:
            if self.vehicle.x > INTERSECTION_HALF_SIZE or self.vehicle.x < -INTERSECTION_HALF_SIZE:
                break_out = True
        elif self.vehicle.intersection_steering_choice == 1:
            if self.vehicle.y > INTERSECTION_HALF_SIZE or self.vehicle.x < -INTERSECTION_HALF_SIZE:
                break_out = True

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
                        # reward与steps联系起来
                        reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                        # reward与steps联系起来
                        reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                    # reward与steps联系起来
                    reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                    # reward与steps联系起来
                    reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                        # reward与steps联系起来
                        reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                            # reward与steps联系起来
                            reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                            # reward与steps联系起来
                            reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                            # reward与steps联系起来
                            reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
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
                            # reward与steps联系起来
                            reward += (200.0 - self.episode_step) * math.sqrt(num + 1)
                            self.ARRIVE_AT_MID_GOAL[num] = 1
                            self.distance_2_goal = 0

        # 如果主车进入到禁止进入区域，break-out.
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
            reward += -2.0

        if break_out:
            reward += -1000.0
            terminated = True

        if self.goal:
            reward += 1000.0
            terminated = True

        if terminated == True:
            self.TCPClient.client_socket.close()
            print("client closed")


        self.vehicle_speed = np.array(
            [self.vehicle.velocity], dtype=np.float32)
        self.vehicle_yawangle = np.array(
            [self.vehicle.yaw_angle/math.pi], dtype=np.float32)
        self.relative_position_2_danger = np.array(
            [self.vehicle.x - self.signal_stop.danger_x, self.vehicle.y - self.signal_stop.danger_y], dtype=np.float32)
        self.first_other_vehicle_relative_position = np.array(
            [self.vehicle.x-self.remote_vehicles[0].x, self.vehicle.y-self.remote_vehicles[0].y], dtype=np.float32)
        self.first_other_vehicle_speed = np.array([self.remote_vehicles[0].velocity], dtype=np.float32)
        self.second_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.remote_vehicles[1].x, self.vehicle.y - self.remote_vehicles[1].y], dtype=np.float32)
        self.second_other_vehicle_speed = np.array([self.remote_vehicles[1].velocity], dtype=np.float32)
        self.relative_distance_to_goal = np.array([self.distance_2_goal], dtype=np.float32)
        self.distance_to_mid_lane_line = np.array([self.vehicle.distance_to_mid_lane_line], dtype=np.float32)

        # self.signal_stop_position = np.array([self.signal_stop.x, self.signal_stop.y], dtype=np.int32)
        # self.signal_stop_state = np.array([phase, countdown], dtype=np.int32)

        print('episode_step:', self.episode_step, ':', 'Action:', action, ';',
              'Position:', self.vehicle.x, self.vehicle.y, ';',
              'first_other_vehicle:', self.remote_vehicles[0].x, self.remote_vehicles[0].y, ';'
              'second_other_vehicle:', self.remote_vehicles[1].x, self.remote_vehicles[1].y, ';'
              # 'Speed:', self.vehicle_speed, ';', 'YawAngle:', self.vehicle_yawangle, ';',
              # 'Relative_distance_to_goal:', self.relative_distance_to_goal, ';',
              # 'vehicle_state:', self.vehicle.state, ';',
              # 'intersection_steering_choice:', self.vehicle.intersection_steering_choice,
              # 'track_judgment:', self.vehicle_track.judgment_str,
              # 'Judgement_in_road:', self.JUDGEMENT_IN_ROAD, ';',
              # 'Distance_to_off_road:', self.vehicle.distance_to_off_road, ';',
              # 'Distance_to_mid_lane_line', self.vehicle.distance_to_mid_lane_line, ';',
              # 'Polar_radius:', self.vehicle.polar_radius, ';',
              # 'Polar_angle:', self.vehicle.polar_angle, ';'
              'reward:', reward
              )
        # if reward < -100 or reward > 100:
        #   print(reward)

        self.state: dict = (
            {
                'agent_speed': self.vehicle_speed,
                'agent_yawangle': self.vehicle_yawangle,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'distance_2_mid_lane_line': self.distance_to_mid_lane_line,
                'distance_2_nearest_off_road': self.vehicle.distance_to_off_road,
                'bird_eye_view_gray_image': self.image_map_observation.separate_map,
                # 'stop_line_position': self.signal_stop_position,
                # 'signal_light_phase_countdown': self.signal_stop_state
            }
        )

        info = {}

        return self.state, reward, terminated, info

    # 重置环境-整局游戏结束之后。可做初始化函数，智能体和观测
    def reset(self):
        self.vehicle = Cube(SIZE)
        # self.first_other_vehicle = Remote_Cube(SIZE)
        # self.second_other_vehicle = Remote_Cube(SIZE)
        self.remote_vehicles = [Remote_Cube(SIZE) for _ in range(NUMBER_REMOTE_VEHICLES)]
        self.predicted_remote_vehicles = [Remote_Cube(SIZE) for _ in range(NUMBER_REMOTE_VEHICLES)]
        self.signal_stop = Signal_Light_Stop_Line()
        self.vehicle_track = Track(SIZE)
        self.rectangle_list_off_road = Rectangle_List_Off_Road()
        self.rectangle_list_reverse = Rectangle_List_Reverse()
        self.rectangle_list_crash_area = Rectangle_List_Crash_Area()
        self.image_map_observation = Image_Map_Observation()
        self.rectangle_list_off_road_show = Rectangle_List_Off_Road_Show()
        self.rectangle_list_reverse_show = Rectangle_List_Reverse_Show()
        self.rectangle_list_crash_area_show = Rectangle_List_Crash_Area_Show()
        self.image_map_observation_show = Image_Map_Observation_Show()
        self.signal_light_stop_line = Signal_Light_Stop_Line()
        self.predicted_image_map_observation_show = Image_Map_Observation_Show()


        print('reset')

        data = {
            'TickId': 0,
            'X': self.vehicle.x/SCALE,
            'Y': self.vehicle.y/SCALE,
            'YawAngle': self.vehicle.yaw_angle
        }
        # while True:
        #     step = 0
        #     self.TCPClient.send_data ( self.TCPClient.client_socket, data )
        #     self.TCPClient.receive_data ( self.TCPClient.client_socket )
        #     if step >= 3:
        #         break
        #     # 等待一段时间后继续循环
        #     step += 1
        #     time.sleep(1)

        self.TCPClient.send_data(self.TCPClient.client_socket, data)
        self.TCPClient.receive_data ( self.TCPClient.client_socket)
        # if self.TCPClient.Remotes:
        #
        #     self.remote_vehicles[0].x = self.TCPClient.Remotes[0]['X'] * SCALE
        #     self.remote_vehicles[0].y = self.TCPClient.Remotes[0]['Y'] * SCALE
        #     self.remote_vehicles[0].yaw_angle = self.TCPClient.Remotes[0]['YawAngle']
        #     self.remote_vehicles[0].velocity = self.TCPClient.Remotes[0]['Speed'] * SCALE
        #
        #     self.remote_vehicles[1].x = self.TCPClient.Remotes[1]['X'] * SCALE
        #     self.remote_vehicles[1].y = self.TCPClient.Remotes[1]['Y'] * SCALE
        #     self.remote_vehicles[1].yaw_angle = self.TCPClient.Remotes[1]['YawAngle']
        #     self.remote_vehicles[1].velocity = self.TCPClient.Remotes[1]['Speed'] * SCALE

        if self.TCPClient.Remotes:
            for i in range(NUMBER_REMOTE_VEHICLES):
                self.remote_vehicles[i].x = self.TCPClient.Remotes[i]['X'] * SCALE
                self.remote_vehicles[i].y = self.TCPClient.Remotes[i]['Y'] * SCALE
                if self.TCPClient.Remotes[i]['YawAngle'] > 180:
                    self.TCPClient.Remotes[i]['YawAngle'] = self.TCPClient.Remotes[i]['YawAngle'] - 360
                self.remote_vehicles[i].yaw_angle = self.TCPClient.Remotes[i]['YawAngle']/180*PI
                self.remote_vehicles[i].velocity = self.TCPClient.Remotes[i]['Speed'] * SCALE


        self.vehicle_speed = np.array(
            [self.vehicle.velocity], dtype=np.float32)
        self.vehicle_yawangle = np.array(
            [self.vehicle.yaw_angle/math.pi], dtype=np.float32)
        self.relative_position_2_danger = np.array(
            [self.vehicle.x - self.signal_stop.danger_x, self.vehicle.y - self.signal_stop.danger_y], dtype=np.float32)
        self.first_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.remote_vehicles[0].x, self.vehicle.y - self.remote_vehicles[0].y], dtype=np.float32)
        self.first_other_vehicle_speed = np.array([self.remote_vehicles[0].velocity], dtype=np.float32)
        self.second_other_vehicle_relative_position = np.array(
            [self.vehicle.x - self.remote_vehicles[1].x, self.vehicle.y - self.remote_vehicles[1].y], dtype=np.float32)
        self.second_other_vehicle_speed = np.array([self.remote_vehicles[1].velocity], dtype=np.float32)
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
        self.CIRCLE_COUNT = 0
        self.ARRIVE_AT_MID_GOAL = np.zeros((self.mid_goal_front_space_y), dtype=np.int32)

        for i in range(NUMBER_REMOTE_VEHICLES):
            self.predicted_remote_vehicles[i].x = self.remote_vehicles[i].x
            self.predicted_remote_vehicles[i].y = self.remote_vehicles[i].y
            self.predicted_remote_vehicles[i].yaw_angle = self.remote_vehicles[i].yaw_angle
            self.predicted_remote_vehicles[i].velocity = self.remote_vehicles[i].velocity

        self.ukf_prediction = [UKF_Prediction() for _ in range(NUMBER_REMOTE_VEHICLES)]
        self.points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0)

        # self.ukf = [[None] for _ in range(NUMBER_REMOTE_VEHICLES)]
        #
        # for i in range(NUMBER_REMOTE_VEHICLES):
        #     self.ukf[i] = UnscentedKalmanFilter(dim_x=3, dim_z=3, dt=1, fx=self.ukf_prediction[i].fx,
        #                                         hx=self.ukf_prediction[i].hx, points=self.points)

        # for i in range(NUMBER_REMOTE_VEHICLES):
        #     self.ukf[i].x_prior[0] = self.remote_vehicles[i].x
        #     self.ukf[i].x_prior[1] = self.remote_vehicles[i].y
        #     self.ukf[i].x_prior[2] = self.remote_vehicles[i].yaw_angle

        self.ukf = []  # 创建一个空列表来存储 UnscentedKalmanFilter 对象

        for i in range(NUMBER_REMOTE_VEHICLES):
            ukf_instance = UnscentedKalmanFilter(dim_x=3, dim_z=3, dt=1, fx=self.ukf_prediction[i].fx,
                                                 hx=self.ukf_prediction[i].hx, points=self.points)
            ukf_instance.x_prior[0] = self.remote_vehicles[i].x
            ukf_instance.x_prior[1] = self.remote_vehicles[i].y
            ukf_instance.x_prior[2] = self.remote_vehicles[i].yaw_angle
            self.ukf.append(ukf_instance)

        for i in range(NUMBER_REMOTE_VEHICLES):
            self.ukf_prediction[i].initUKF(self.ukf[i], self.remote_vehicles[i].x, self.remote_vehicles[i].y,
                                    self.remote_vehicles[i].yaw_angle)

        # print(self.episode_step, ':', self.vehicle_position, self.vehicle_state, self.relative_distance_to_goal,
        #       self.JUDGEMENT_IN_ROAD, self.Time_to_off_road, self.vehicle.distance_to_off_road)

        print('episode_step:', self.episode_step, ':',
              'Position:', self.vehicle.x, self.vehicle.y, ';',
              'first_other_vehicle:', self.remote_vehicles[0].x, self.remote_vehicles[0].y, ';',
              'second_other_vehicle:', self.remote_vehicles[1].x, self.remote_vehicles[1].y, ';'
              )

        self.state: dict = (
            {
                'agent_speed': self.vehicle_speed,
                'agent_yawangle': self.vehicle_yawangle,
                'relative_position_2_danger': self.relative_position_2_danger,
                'first_other_vehicle_relative_position': self.first_other_vehicle_relative_position,
                'first_other_vehicle_speed': self.first_other_vehicle_speed,
                'second_other_vehicle_relative_position': self.second_other_vehicle_relative_position,
                'second_other_vehicle_speed': self.second_other_vehicle_speed,
                'relative_distance_2_goal': self.relative_distance_to_goal,
                'judgement_in_road': self.JUDGEMENT_IN_ROAD,
                'distance_2_mid_lane_line': self.distance_to_mid_lane_line,
                'distance_2_nearest_off_road': self.vehicle.distance_to_off_road,
                'bird_eye_view_gray_image': self.image_map_observation.separate_map,
                # 'stop_line_position': self.signal_stop_position,
                # 'signal_light_phase_countdown': self.signal_stop_state
            }
        )

        return self.state

    # 多媒体演示
    def render(self, mode="human"):
        img, img1, img2, img3 = self.get_image()
        # cv2.imshow('0', np.array(img))
        cv2.imshow('1', np.array(img1))
        cv2.imshow('2', np.array(img2))
        cv2.imshow('3', np.array(img3))
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


        INT_FIRST_OTHER_VEHICLE_Y = int(round(self.remote_vehicles[0].y, 0))
        INT_FIRST_OTHER_VEHICLE_X = int(round(self.remote_vehicles[0].x, 0))
        INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_X = int(round(self.remote_vehicles[0].max_vertex_x, 0))
        INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_X = int(round(self.remote_vehicles[0].min_vertex_x, 0))
        INT_FIRST_OTHER_VEHICLE_MAX_VERTEX_Y = int(round(self.remote_vehicles[0].max_vertex_y, 0))
        INT_FIRST_OTHER_VEHICLE_MIN_VERTEX_Y = int(round(self.remote_vehicles[0].min_vertex_y, 0))

        INT_SECOND_OTHER_VEHICLE_Y = int(round(self.remote_vehicles[1].y, 0))
        INT_SECOND_OTHER_VEHICLE_X = int(round(self.remote_vehicles[1].x, 0))
        INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_X = int(round(self.remote_vehicles[1].max_vertex_x, 0))
        INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_X = int(round(self.remote_vehicles[1].min_vertex_x, 0))
        INT_SECOND_OTHER_VEHICLE_MAX_VERTEX_Y = int(round(self.remote_vehicles[1].max_vertex_y, 0))
        INT_SECOND_OTHER_VEHICLE_MIN_VERTEX_Y = int(round(self.remote_vehicles[1].min_vertex_y, 0))

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

        for x5 in range(0, int(4.0 * VEHICLE_ILLUSTRATION_YAW_ANGLE_SIZE)):
            x = INT_MAX_COORD - INT_VEHICLE_Y - int(x5 * math.cos(self.vehicle.yaw_angle)) + 1
            y = INT_MAX_COORD + INT_VEHICLE_X + int(x5 * math.sin(self.vehicle.yaw_angle))
            if x >= INT_SIZE:
                x = INT_SIZE - 1
            if y >= INT_SIZE:
                y = INT_SIZE - 1
            env[x][y] = self.d[self.RED_LIGHT_N]

        # 画出车祸区域
        for x6 in range(INT_MAX_COORD + 3 * INT_SINGLE_LANE_WIDTH, INT_MAX_COORD + 5 * INT_SINGLE_LANE_WIDTH):
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

        # img = Image.fromarray(env, 'RGB')
        # return img
        self.image_map_observation_show.illustration()
        env1 = self.image_map_observation_show.illustration_whole_map
        self.image_map_observation_show.illustrationagent()
        env2 = self.image_map_observation_show.illustration_separate_map
        self.predicted_image_map_observation_show.illustration()
        env3 = self.predicted_image_map_observation_show.illustration_whole_map

        img = Image.fromarray(env, 'RGB')
        img1 = Image.fromarray(env1, 'L')
        img2 = Image.fromarray(env2, 'L')
        img3 = Image.fromarray(env3, 'L')
        return img, img1, img2, img3
