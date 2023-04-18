from Cube import Cube
from Forward_Cube_First import Forward_Cube_First
from Forward_Cube_Second import Forward_Cube_Second
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
from Track import Track
from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD
from CONSTANTS import MAX_COORD, SEPARATE_SIZE
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
import numpy as np

# Rectangle
class Rectangle:

    def __init__(self,max_x,min_x,max_y,min_y ):
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y

# Rectangle of Off-road
class Rectangle_List_Off_Road:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList = np.ndarray(shape=(1, 8), dtype=Rectangle)

        self.RectangleList[0][0] = Rectangle(-2*SINGLE_LANE_WIDTH, MIN_COORD, MAX_COORD, INTERSECTION_HALF_SIZE)
        self.RectangleList[0][1] = Rectangle(MAX_COORD, 2*SINGLE_LANE_WIDTH, MAX_COORD, INTERSECTION_HALF_SIZE)
        self.RectangleList[0][2] = Rectangle(MAX_COORD, 2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE, MIN_COORD)
        self.RectangleList[0][3] = Rectangle(-2*SINGLE_LANE_WIDTH, MIN_COORD, -INTERSECTION_HALF_SIZE, MIN_COORD)
        self.RectangleList[0][4] = Rectangle(-INTERSECTION_HALF_SIZE, MIN_COORD, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH)
        self.RectangleList[0][5] = Rectangle(MAX_COORD, INTERSECTION_HALF_SIZE, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH)
        self.RectangleList[0][6] = Rectangle(MAX_COORD, INTERSECTION_HALF_SIZE, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE)
        self.RectangleList[0][7] = Rectangle(-INTERSECTION_HALF_SIZE, MIN_COORD, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE)

# Rectangle of Reverse
# 根据车辆状态定义逆行区域
class Rectangle_List_Reverse:

    def __init__(self):
        self.RectangleList = np.ndarray(shape=(1, 4), dtype=Rectangle)

    def judgement(self, state):
        # max_x, min_x, max_y, min_y
        if state == 'straight_y+':
            self.RectangleList[0][0] = Rectangle(0, -2*SINGLE_LANE_WIDTH, MAX_COORD, INTERSECTION_HALF_SIZE)
            self.RectangleList[0][1] = Rectangle(MAX_COORD, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH, 0)
            self.RectangleList[0][2] = Rectangle(0, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE, MIN_COORD)
            self.RectangleList[0][3] = Rectangle(-INTERSECTION_HALF_SIZE, MIN_COORD, 0, -2*SINGLE_LANE_WIDTH)

class Rectangle_List_Crash_Area:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList = np.ndarray(shape=(1, 1), dtype=Rectangle)

        self.RectangleList[0][0] = Rectangle(SINGLE_LANE_WIDTH, 0, -3*SINGLE_LANE_WIDTH, -5*SINGLE_LANE_WIDTH)

class Rectangle_List_Expand_Area:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList = np.ndarray(shape=(1, 4), dtype=Rectangle)
        INT_SEPARATE_HALF_SIZE = int(round(SEPARATE_SIZE/2,0))
        self.RectangleList[0][0] = Rectangle(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MAX_COORD))
        self.RectangleList[0][1] = Rectangle(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MAX_COORD),
                                             int(MAX_COORD), int(MIN_COORD))
        self.RectangleList[0][2] = Rectangle(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MIN_COORD), int(MIN_COORD)-INT_SEPARATE_HALF_SIZE)
        self.RectangleList[0][3] = Rectangle(int(MIN_COORD), int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MAX_COORD), int(MIN_COORD))

