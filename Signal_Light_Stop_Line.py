
###########################
#
#   @author mckvg
#   @author olin322
#
###########################
import math

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

# 信号灯及停止线的类
class Signal_Light_Stop_Line:

    def __init__(self):
        # 停止线中心位置
        self.x1 = 3*SINGLE_LANE_WIDTH/2
        self.y1 = -3*SINGLE_LANE_WIDTH + SINGLE_LANE_WIDTH/4
        self.x2 = -3*SINGLE_LANE_WIDTH/2
        self.y2 = 3*SINGLE_LANE_WIDTH
        self.x3 = 3*SINGLE_LANE_WIDTH
        self.y3 = 3*SINGLE_LANE_WIDTH/2
        self.x4 = -3*SINGLE_LANE_WIDTH
        self.y4 = -3*SINGLE_LANE_WIDTH/2
        # 车祸区域中心位置
        self.danger_x = SINGLE_LANE_WIDTH/2
        self.danger_y = -INTERSECTION_HALF_SIZE + SINGLE_LANE_WIDTH/2
        self.signal = 0  # red: 0  green: 1  yellow: 2
        # 红 9s 绿 26s 黄 2s 依次变化/
        self.countdown_red = 9
        self.countdown_green = 26
        self.countdown_yellow = 2
        self.countdown_phase1 = self.countdown_red
        self.countdown_phase12 = self.countdown_red + self.countdown_green
        self.countdown_phase123 = self.countdown_red + self.countdown_yellow + self.countdown_green
        self.countdown_phase1231 = self.countdown_phase123 + self.countdown_phase1
        self.move_step = 0
        self.loop_step = 0
        self.countdown = 0

    def __str__(self):
        return f'{(self.x, self.y)},{self.signal}{self.countdown}'

    def signalphase(self, step: int):  # 红 9s 绿 26s 黄 2s 依次变化
        if math.floor(step / self.countdown_phase123) > 0 and step % self.countdown_phase123 == 0:
            self.loop_step = self.countdown_phase123
        else:
            self.loop_step = step % self.countdown_phase123

        if 0 < self.loop_step <= self.countdown_phase1:
            self.signal = 0
        elif self.countdown_phase1 < self.loop_step <= self.countdown_phase12:
            self.signal = 1
        elif self.countdown_phase12 < self.loop_step <= self.countdown_phase123:
            self.signal = 2

        return self.signal

    def signalcountdown(self, phase: int):
        self.move_step += 1
        if phase == 0:
            self.countdown = self.countdown_phase1 - self.loop_step
        elif phase == 1:
            self.countdown = self.countdown_phase12 - self.loop_step
        elif phase == 2:
            self.countdown = self.countdown_phase123 - self.loop_step

        return self.countdown