
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
# from CONSTANTS import VEHICLE_HALF_SIZE
from CONSTANTS import SINGLE_LANE_WIDTH
from CONSTANTS import min_positionx
from CONSTANTS import max_positionx
from CONSTANTS import min_positiony
from CONSTANTS import max_positiony
from CONSTANTS import VEHICLE_LENGTH
from CONSTANTS import VEHICLE_DIAGONAL, VEHICLE_ANGLE

# 前车的类 有其 状态信息 和 行动
class Forward_Cube_Second:

    def __init__(self, size):  # 初始化状态
        self.size = size
        self.x = float(SINGLE_LANE_WIDTH / 2 + SINGLE_LANE_WIDTH)
        self.y = MIN_COORD + 5 * VEHICLE_LENGTH
        self.pre_x = self.x
        self.pre_y = self.y
        self.velocity = 0.0 * SCALE
        self.acceleration = 0.0 * SCALE
        self.move_step = 0
        self.yaw_angle = 0.0
        self.pre_yaw_angle = self.yaw_angle

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

        self.pre_max_vertex_x = self.max_vertex_x
        self.pre_min_vertex_x = self.min_vertex_x
        self.pre_max_vertex_y = self.max_vertex_y
        self.pre_min_vertex_y = self.min_vertex_y

    def __str__(self):
        return f'{(self.x, self.y)},{self.velocity},{self.acceleration}'

    def move(self):
        min_acceleration = -4.5 * SCALE
        max_acceleration = 2.6 * SCALE
        min_speed = 0.0 * SCALE
        max_speed = 12.00 * SCALE
        mid_speed = 8.0 * SCALE

        self.move_step += 1

        # self.pre_x = self.x
        # self.pre_y = self.y
        # self.pre_yaw_angle = self.yaw_angle

        self.pre_max_vertex_x = self.max_vertex_x
        self.pre_min_vertex_x = self.min_vertex_x
        self.pre_max_vertex_y = self.max_vertex_y
        self.pre_min_vertex_y = self.min_vertex_y

        # self.velocity = mid_speed
        #
        if self.velocity < min_speed:
            self.velocity = min_speed
        if self.velocity > max_speed:
            self.velocity = max_speed
        #
        # self.y += self.velocity

        if self.y > max_positiony:
            self.y = max_positiony
        if self.y < min_positiony:
            self.y = min_positiony

        # 更新车辆边界点的坐标，车头朝向正北(yaw_angle = 0),vertex0为左前点，1为右前点，2为右后点，3为左后点，按顺时针方向。
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
