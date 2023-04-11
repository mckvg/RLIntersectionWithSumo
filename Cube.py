
###########################
#
#   @author mckvg
#   @author olin322
#
###########################


import math
import numpy as np

from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD 
from CONSTANTS import MAX_COORD 
from CONSTANTS import VEHICLE_LENGTH 
from CONSTANTS import VEHICLE_WIDTH 
from CONSTANTS import VEHICLE_ANGLE 
from CONSTANTS import VEHICLE_DIAGONAL
from CONSTANTS import VEHICLE_HALF_SIZE
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
        self.vertex2x = self.x - VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex2y = self.y - VEHICLE_DIAGONAL/2 * math.cos(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex3x = self.x - VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle+VEHICLE_ANGLE)
        self.vertex3y = self.y - VEHICLE_DIAGONAL / 2 * math.cos(self.yaw_angle + VEHICLE_ANGLE)

        self.max_vertex_x = max(self.vertex0x, self.vertex1x, self.vertex2x, self.vertex3x)
        self.min_vertex_x = min(self.vertex0x, self.vertex1x, self.vertex2x, self.vertex3x)
        self.max_vertex_y = max(self.vertex0y, self.vertex1y, self.vertex2y, self.vertex3y)
        self.min_vertex_y = min(self.vertex0y, self.vertex1y, self.vertex2y, self.vertex3y)

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
        self.init_state = ''
        self.pre_state = ''
        self.intersection_steering_choice = 2  # 2: default; 0: straight_y+; -1: turn left_x-; 1: turn right_x+.
        self.distance_to_off_road = np.array([self.edge0-0, self.edge2-INTERSECTION_HALF_SIZE], dtype=np.float32)
        # self.intersection_judgment = False

    def __str__(self):
        return f'{(self.x, self.y)},{self.yaw_angle},{self.velocity},{self.acceleration},{self.state}'

    def collision(self, other):
        flag = 0
        if ((self.max_vertex_x < other.min_vertex_x) or (self.min_vertex_x > other.max_vertex_x) or
            (self.max_vertex_y < other.min_vertex_y) or (self.min_vertex_y > other.max_vertex_y) or
            other.y == max_positiony or other.x == max_positionx or other.x == min_positionx):

            flag = 0
        else:
            flag = 1
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

        # 更新车辆四周的坐标，车头朝向正北(yaw_angle = 0),vertex0为左前点，1为右前点，2为右后点，3为左后点，按顺时针方向。
        self.vertex0x = self.x + VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex0y = self.y + VEHICLE_DIAGONAL/2 * math.cos(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex1x = self.x + VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle+VEHICLE_ANGLE)
        self.vertex1y = self.y + VEHICLE_DIAGONAL / 2 * math.cos(self.yaw_angle + VEHICLE_ANGLE)
        self.vertex2x = self.x - VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex2y = self.y - VEHICLE_DIAGONAL/2 * math.cos(self.yaw_angle-VEHICLE_ANGLE)
        self.vertex3x = self.x - VEHICLE_DIAGONAL/2 * math.sin(self.yaw_angle+VEHICLE_ANGLE)
        self.vertex3y = self.y - VEHICLE_DIAGONAL / 2 * math.cos(self.yaw_angle + VEHICLE_ANGLE)

        self.max_vertex_x = max(self.vertex0x, self.vertex1x, self.vertex2x, self.vertex3x)
        self.min_vertex_x = min(self.vertex0x, self.vertex1x, self.vertex2x, self.vertex3x)
        self.max_vertex_y = max(self.vertex0y, self.vertex1y, self.vertex2y, self.vertex3y)
        self.min_vertex_y = min(self.vertex0y, self.vertex1y, self.vertex2y, self.vertex3y)

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
        self.polar_radius_min_edge = self.polar_radius - VEHICLE_WIDTH / 2
        self.polar_radius_max_edge = self.polar_radius + VEHICLE_WIDTH / 2

        self.next1_polar_radius = math.sqrt(next1_x_prime ** 2 + next1_y_prime ** 2)
        self.next1_polar_radius_min_edge = self.next1_polar_radius - VEHICLE_WIDTH / 2
        self.next1_polar_radius_max_edge = self.next1_polar_radius + VEHICLE_WIDTH / 2

        self.next2_polar_radius = math.sqrt(next2_x_prime ** 2 + next2_y_prime ** 2)
        self.next2_polar_radius_min_edge = self.next2_polar_radius - VEHICLE_WIDTH / 2
        self.next2_polar_radius_max_edge = self.next2_polar_radius + VEHICLE_WIDTH / 2

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