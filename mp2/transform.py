
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha_i = arm.getArmAngle()[0]
    beta_i = arm.getArmAngle()[1]
    a_limit = arm.getArmLimit()[0]
    b_limit = arm.getArmLimit()[1]

    rows = int((a_limit[1] - a_limit[0])/granularity + 1)
    cols = int((b_limit[1] - b_limit[0])/granularity + 1)
    shape = tuple((rows, cols))

    maze_map = np.full(shape, SPACE_CHAR)
   
    for idxs in np.ndindex(shape):
        angles = idxToAngle(idxs, tuple((a_limit[0], b_limit[0])), granularity)

        arm.setArmAngle(angles)

        if angles[0] == alpha_i and angles[1] == beta_i:
            #print('reached start')
            maze_map[idxs] = START_CHAR

        elif doesArmTouchObjects(arm.getArmPosDist(), obstacles) or not isArmWithinWindow(arm.getArmPos(), window):
            maze_map[idxs] = WALL_CHAR

        elif not doesArmTipTouchGoals(arm.getEnd(), goals) and doesArmTouchObjects(arm.getArmPosDist(), goals, True):
            #print('True')
            maze_map[idxs] = WALL_CHAR

        elif doesArmTipTouchGoals(arm.getEnd(), goals):
            #print('obj')
            maze_map[idxs] = OBJECTIVE_CHAR


    return Maze(maze_map, tuple((a_limit[0], b_limit[0])), granularity)
