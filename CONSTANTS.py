import math	

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