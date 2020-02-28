import search as search_student
import geometry as geometry_student
import transform as transform_student
import maze as maze_student

from arm import Arm
from const import *
import time
import configparser
import copy
import math

def build_maze_basic(configfile, map_name):
	config = configparser.ConfigParser()

    # set 'tests/data/' to you config directory
	config.read('tests/data/' + configfile)

	window = eval(config.get(map_name, 'Window'))
	armBase = eval(config.get(map_name, 'ArmBase'))
	armLinks = eval(config.get(map_name, 'ArmLinks'))

	arm1 = Arm(armBase, armLinks)
	obstacles = eval(config.get(map_name, 'Obstacles'))
	goals = eval(config.get(map_name, 'Goals'))

	return arm1, goals, obstacles, window

# modify configfile to the path of actual config file
# To test if your extra credit code run with autograder, put it in the same folder with your code and run debug.py. If all goes well, the program should print out the path found.
configfile, map_name, granularity = "test_config_extra.txt", "Test1", 10
# configfile, map_name, granularity = "test_config.txt", "Test2", 5

arm, goals, obstacles, window = build_maze_basic(configfile, map_name)
arm_student = copy.deepcopy(arm)
student_maze = transform_student.transformToMaze(
    arm_student, goals, obstacles, window, granularity
)

student_path = search_student.search(student_maze, "bfs")

print(student_path)
