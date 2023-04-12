
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