from Cube import Cube
from Forward_Cube_First import Forward_Cube_First
from Forward_Cube_Second import Forward_Cube_Second
from Signal_Light_Stop_Line import Signal_Light_Stop_Line
from Track import Track
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
import numpy as np
from Rectangle import Rectangle_List_Reverse, Rectangle_List_Crash_Area, Rectangle_List_Off_Road, Rectangle
import Cube

d = {
    1: (255),  # white
    2: (10),  # black
    3: (100), # agent
    4: (200), # remote vehicle
}

road = 1
block = 2
agent = 3
remote_vehicle = 4



class Image_Map_Observation:

    def __init__(self):
        INT_SIZE = int(round(SIZE, 0))
        INT_SEPARATE = 64
        self.whole_map = np.zeros((INT_SIZE, INT_SIZE), dtype=np.uint8)
        self.illustration_whole_map = np.zeros((INT_SIZE, INT_SIZE), dtype=np.uint8)
        self.separate_map = np.zeros((INT_SEPARATE,INT_SEPARATE),dtype=np.uint8)
        self.rectangle_list_off_road = Rectangle_List_Off_Road()
        self.rectangle_list_crash_area = Rectangle_List_Crash_Area()

        # 根据Rectangle_List_Off_Road类进行填图，填色为block
        for rec in range(self.rectangle_list_off_road.RectangleList.size):
            for x1 in range(int(self.rectangle_list_off_road.RectangleList[0][rec].min_x),
                            int(self.rectangle_list_off_road.RectangleList[0][rec].max_x) + 1, 1):
                for y1 in range(int(self.rectangle_list_off_road.RectangleList[0][rec].min_y),
                                int(self.rectangle_list_off_road.RectangleList[0][rec].max_y + 1), 1):
                    self.whole_map[x1][y1] = d[block]

        # 根据Rectangle_List_Crash_Area类进行填图，填色为block
        for rec in range(self.rectangle_list_crash_area.RectangleList.size):
            for x2 in range(int(self.rectangle_list_crash_area.RectangleList[0][rec].min_x),
                            int(self.rectangle_list_crash_area.RectangleList[0][rec].max_x) + 1, 1):
                for y2 in range(int(self.rectangle_list_crash_area.RectangleList[0][rec].min_y),
                                int(self.rectangle_list_crash_area.RectangleList[0][rec].max_y + 1), 1):
                    self.whole_map[x2][y2] = d[block]

        # 将剩余区域填色为road
        for x3 in range(int(MIN_COORD),int(MAX_COORD)+1,1 ):
            for y3 in range(int(MIN_COORD),int(MAX_COORD)+1,1 ):
                if self.whole_map[x3][y3] != d[block]:
                    self.whole_map[x3][y3] = d[road]

    # 在主程序判定完reverse area:(self.rectangle_list_reverse.judgement(self.vehicle.init_state))之后，调用此函数，对reverse area进行填充
    def reverse(self, rectangle_list_reverse_rectanglelist):
        # 根据Rectangle_List_Reverse类进行填图，填色为block
        for rec in range(rectangle_list_reverse_rectanglelist.size):
            for x4 in range(int(rectangle_list_reverse_rectanglelist[0][rec].min_x),
                            int(rectangle_list_reverse_rectanglelist[0][rec].max_x) + 1, 1):
                for y4 in range(int(rectangle_list_reverse_rectanglelist[0][rec].min_y),
                                int(rectangle_list_reverse_rectanglelist[0][rec].max_y + 1), 1):
                    self.whole_map[x4][y4] = d[block]

    # 对智能体的位置区域填色为agent
    def agent(self, vehicle_max_vertex_x, vehicle_min_vertex_x, vehicle_max_vertex_y, vehicle_min_vertex_y,
              vehicle_pre_max_vertex_x, vehicle_pre_min_vertex_x,vehicle_pre_max_vertex_y,vehicle_pre_min_vertex_y):
        int_vehicle_max_vertex_x = int(round(vehicle_max_vertex_x, 0))
        int_vehicle_min_vertex_x = int(round(vehicle_min_vertex_x, 0))
        int_vehicle_max_vertex_y = int(round(vehicle_max_vertex_y, 0))
        int_vehicle_min_vertex_y = int(round(vehicle_min_vertex_y, 0))
        int_vehicle_pre_max_vertex_x = int(round(vehicle_pre_max_vertex_x, 0))
        int_vehicle_pre_min_vertex_x = int(round(vehicle_pre_min_vertex_x, 0))
        int_vehicle_pre_max_vertex_y = int(round(vehicle_pre_max_vertex_y, 0))
        int_vehicle_pre_min_vertex_y = int(round(vehicle_pre_min_vertex_y, 0))
        for x55 in range(int_vehicle_pre_min_vertex_x, int_vehicle_pre_max_vertex_x + 1, 1):
            for y55 in range(int_vehicle_pre_min_vertex_y, int_vehicle_pre_max_vertex_y + 1, 1):
                self.whole_map[x55][y55] = d[road]
        for x5 in range(int_vehicle_min_vertex_x, int_vehicle_max_vertex_x + 1, 1):
            for y5 in range(int_vehicle_min_vertex_y, int_vehicle_max_vertex_y + 1, 1):
                self.whole_map[x5][y5] = d[agent]


    # 对远车的位置区域填色为remote_vehicle
    def RemoteVehicle(self, remote_vehicle_max_vertex_x, remote_vehicle_min_vertex_x,
                      remote_vehicle_max_vertex_y, remote_vehicle_min_vertex_y,
                      remote_vehicle_pre_max_vertex_x, remote_vehicle_pre_min_vertex_x,
                      remote_vehicle_pre_max_vertex_y, remote_vehicle_pre_min_vertex_y):
        int_remote_vehicle_max_vertex_x = int(round(remote_vehicle_max_vertex_x, 0))
        int_remote_vehicle_min_vertex_x = int(round(remote_vehicle_min_vertex_x, 0))
        int_remote_vehicle_max_vertex_y = int(round(remote_vehicle_max_vertex_y, 0))
        int_remote_vehicle_min_vertex_y = int(round(remote_vehicle_min_vertex_y, 0))
        int_remote_vehicle_pre_max_vertex_x = int(round(remote_vehicle_pre_max_vertex_x, 0))
        int_remote_vehicle_pre_min_vertex_x = int(round(remote_vehicle_pre_min_vertex_x, 0))
        int_remote_vehicle_pre_max_vertex_y = int(round(remote_vehicle_pre_max_vertex_y, 0))
        int_remote_vehicle_pre_min_vertex_y = int(round(remote_vehicle_pre_min_vertex_y, 0))
        for x66 in range(int_remote_vehicle_pre_min_vertex_x, int_remote_vehicle_pre_max_vertex_x + 1, 1):
            for y66 in range(int_remote_vehicle_pre_min_vertex_y, int_remote_vehicle_pre_max_vertex_y + 1, 1):
                self.whole_map[x66][y66] = d[road]
        for x6 in range(int_remote_vehicle_min_vertex_x, int_remote_vehicle_max_vertex_x + 1, 1):
            for y6 in range(int_remote_vehicle_min_vertex_y, int_remote_vehicle_max_vertex_y + 1, 1):
                self.whole_map[x6][y6] = d[remote_vehicle]

    def illustration(self, INT_SIZE):
        for x7 in range(INT_SIZE):
            for y7 in range(INT_SIZE):
                self.illustration_whole_map[-y7][x7] = self.whole_map[x7-int(MAX_COORD)][y7-int(MAX_COORD)]


    # def SeparateMap(self):

