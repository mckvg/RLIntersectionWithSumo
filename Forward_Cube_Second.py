
###########################
#
#   @author mckvg
#   @author olin322
#
###########################

from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD 
from CONSTANTS import MAX_COORD 
from CONSTANTS import VEHICLE_HALF_SIZE
from CONSTANTS import SINGLE_LANE_WIDTH
from CONSTANTS import min_positionx
from CONSTANTS import max_positionx
from CONSTANTS import min_positiony
from CONSTANTS import max_positiony


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
