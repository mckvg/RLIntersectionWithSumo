# 引入包
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from datetime import datetime, date
import os, sys
import sumolib
import time

# -----引入地址------
sumo_path = "D:/Sumo"
project_path = "D:/pythonSumoRL/SumoRL"
cfg_path = "D:/pythonSumoRL/SumoRL/sumo.cfg"
# ----------------------------------------#

# ---------------------#
sys.path.append(sumo_path)
sys.path.append(sumo_path + "tools")

# 引入地址导入模块
sys.path.append(sumo_path + "/tools/xml")
import traci
from sumolib import checkBinary

RemoteVehiclePosition = {}
RemoteVehicleSpeed = {}
RemoteVehicleYawAngle = {}

def RVData(id):
    RemoteVehiclePosition[id] = traci.vehicle.getPosition(id)
    RemoteVehicleSpeed[id] = traci.vehicle.getSpeed(id)
    RemoteVehicleYawAngle[id] = traci.vehicle.getAngle(id)

def run(maxstep):

    for step in range(0, maxstep):

        if step == 0:
            traci.vehicle.moveToXY('0', '-E3', 1, 2, -115, 0)
            traci.vehicle.moveToXY('1', '-E3', 0, 6, -115, 0)
            traci.vehicle.moveToXY('2', '-E3', 0, 6, -100, 0)
        # 步长控制
        vehicle_id0 = '0'
        vehicle_id1 = '1'
        vehicle_id2 = '2'
        if vehicle_id0 in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(vehicle_id0, 4)
        if vehicle_id1 in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(vehicle_id1, 2)
            RVData(vehicle_id1)
        if vehicle_id2 in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(vehicle_id2, 8)
            RVData(vehicle_id2)

        print(RemoteVehiclePosition)
        print(RemoteVehiclePosition['1'])
        print(RemoteVehiclePosition['1'][0])

        traci.simulationStep(step + 1)

    sys.stdout.flush()
    traci.close()


if __name__ == "__main__":

    N_STATES = 200

    # 是否打开gui True为打开 False为关闭
    gui = True
    if gui == 1:
        sumoBinary = checkBinary(sumo_path + "/bin/sumo-gui")
    else:
        sumoBinary = checkBinary(sumo_path + "/bin/sumo")

    # -----配置文件cfg打开以及输出xml格式统计数据
    sumoCmd = [sumoBinary, "-c", cfg_path, '--tripinfo-output', project_path + "/tripinfo2_TEST.xml",
               '--duration-log.statistics']

    # ----开始---
    traci.start(sumoCmd)
    run(N_STATES)

