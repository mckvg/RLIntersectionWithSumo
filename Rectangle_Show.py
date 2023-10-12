from Cube import Cube
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
from Track import Track
from CONSTANTS import SCALE
from CONSTANTS import SIZE
from CONSTANTS import MIN_COORD
from CONSTANTS import MAX_COORD, SEPARATE_SHOW_SIZE
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
class Rectangle_Show:

    def __init__(self,max_x,min_x,max_y,min_y ):
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y

# Rectangle of Off-road
class Rectangle_List_Off_Road_Show:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList_Show = np.ndarray(shape=(1, 8), dtype=Rectangle_Show)

        self.RectangleList_Show[0][0] = Rectangle_Show(-2*SINGLE_LANE_WIDTH, MIN_COORD, MAX_COORD, INTERSECTION_HALF_SIZE)
        self.RectangleList_Show[0][1] = Rectangle_Show(MAX_COORD, 2*SINGLE_LANE_WIDTH, MAX_COORD, INTERSECTION_HALF_SIZE)
        self.RectangleList_Show[0][2] = Rectangle_Show(MAX_COORD, 2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE, MIN_COORD)
        self.RectangleList_Show[0][3] = Rectangle_Show(-2*SINGLE_LANE_WIDTH, MIN_COORD, -INTERSECTION_HALF_SIZE, MIN_COORD)
        self.RectangleList_Show[0][4] = Rectangle_Show(-INTERSECTION_HALF_SIZE, MIN_COORD, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH)
        self.RectangleList_Show[0][5] = Rectangle_Show(MAX_COORD, INTERSECTION_HALF_SIZE, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH)
        self.RectangleList_Show[0][6] = Rectangle_Show(MAX_COORD, INTERSECTION_HALF_SIZE, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE)
        self.RectangleList_Show[0][7] = Rectangle_Show(-INTERSECTION_HALF_SIZE, MIN_COORD, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE)

# Rectangle of Reverse
# 根据车辆状态定义逆行区域
class Rectangle_List_Reverse_Show:

    def __init__(self):
        self.RectangleList_Show = np.ndarray(shape=(1, 4), dtype=Rectangle_Show)

    def judgement(self, state):
        # max_x, min_x, max_y, min_y
        if state == 'straight_y+':
            self.RectangleList_Show[0][0] = Rectangle_Show(0, -2*SINGLE_LANE_WIDTH, MAX_COORD, INTERSECTION_HALF_SIZE)
            self.RectangleList_Show[0][1] = Rectangle_Show(MAX_COORD, INTERSECTION_HALF_SIZE, 2*SINGLE_LANE_WIDTH, 0)
            self.RectangleList_Show[0][2] = Rectangle_Show(0, -2*SINGLE_LANE_WIDTH, -INTERSECTION_HALF_SIZE, MIN_COORD)
            self.RectangleList_Show[0][3] = Rectangle_Show(-INTERSECTION_HALF_SIZE, MIN_COORD, 0, -2*SINGLE_LANE_WIDTH)

class Rectangle_List_Crash_Area_Show:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList_Show = np.ndarray(shape=(1, 1), dtype=Rectangle_Show)

        self.RectangleList_Show[0][0] = Rectangle_Show(SINGLE_LANE_WIDTH, 0, -3*SINGLE_LANE_WIDTH, -5*SINGLE_LANE_WIDTH)

class Rectangle_List_Expand_Area_Show:

    def __init__(self):
        # max_x, min_x, max_y, min_y
        self.RectangleList_Show = np.ndarray(shape=(1, 4), dtype=Rectangle_Show)
        INT_SEPARATE_HALF_SIZE = int(round(SEPARATE_SHOW_SIZE/2,0))
        self.RectangleList_Show[0][0] = Rectangle_Show(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MAX_COORD))
        self.RectangleList_Show[0][1] = Rectangle_Show(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MAX_COORD),
                                             int(MAX_COORD), int(MIN_COORD))
        self.RectangleList_Show[0][2] = Rectangle_Show(int(MAX_COORD)+INT_SEPARATE_HALF_SIZE, int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MIN_COORD), int(MIN_COORD)-INT_SEPARATE_HALF_SIZE)
        self.RectangleList_Show[0][3] = Rectangle_Show(int(MIN_COORD), int(MIN_COORD)-INT_SEPARATE_HALF_SIZE,
                                             int(MAX_COORD), int(MIN_COORD))

