# mp2.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/04/2018

"""
This file contains the main application that is run for this MP.
"""

import pygame
import sys
import argparse
import configparser
import copy

from pygame.locals import *
from arm import Arm
from transform import transformToMaze
from search import search
from const import *
from util import *
from geometry import *

class Application:

    def __init__(self, configfile, map_name, human=True, fps=DEFAULT_FPS):
        self.running = False
        self.displaySurface = None
        self.config = configparser.ConfigParser()
        self.config.read(configfile)
        self.fps = fps
        self.__human = human
        self.clock = pygame.time.Clock()   
        self.trajectory = []   

        # Parse config file
        self.windowTitle = "CS440 MP2 Robotic Arm"
        self.window = eval(self.config.get(map_name, 'Window'))

        armBase = eval(self.config.get(map_name, 'ArmBase'))
        armLinks = eval(self.config.get(map_name, 'ArmLinks'))
        self.armLimits = [(0, 0), (0, 0), (0, 0)]
        for i in range(len(armLinks)):
            self.armLimits[i] = armLinks[i][-1]
        self.arm = Arm(armBase, armLinks)

        self.obstacles = eval(self.config.get(map_name, 'Obstacles'))
        self.goals = eval(self.config.get(map_name, 'Goals'))


    # Initializes the pygame context and certain properties of the maze
    def initialize(self):
        
        pygame.init()
        self.displaySurface = pygame.display.set_mode((self.window[0], self.window[1]), pygame.HWSURFACE)
        self.displaySurface.fill(WHITE)
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)
        self.running = True

    # Once the application is initiated, execute is in charge of drawing the game and dealing with the game loop
    def execute(self, searchMethod, granularity, trajectory, saveImage, saveMaze):        
        self.initialize()
        if not self.running:
            print("Program init failed")
            raise SystemExit
        
        currAngle = [0, 0, 0]
        for i in range(len(self.arm.getArmAngle())):
            currAngle[i] = self.arm.getArmAngle()[i]
        self.gameLoop()        

        if not self.__human:
            print("Transforming a map configuration to a maze...")
            maze = transformToMaze(self.arm, self.goals, self.obstacles, self.window, granularity)
            print("Done!")
            print("Searching the path...")
            path = search(maze, searchMethod)
            if path is None:
                print("No path found!")
            else:
                for i in range(len(path)):
                    self.arm.setArmAngle(path[i])
                    if (trajectory > 0) and (i % trajectory == 0):
                        self.trajectory.append(self.arm.getArmPos())
                    self.gameLoop()
                print("Done!")
                self.drawTrajectory()

        while self.running:
            pygame.event.pump()            
            keys = pygame.key.get_pressed()
                        
            if (keys[K_ESCAPE]):
                self.running = False                

            if self.__human:                
                alpha, beta, gamma = currAngle                
                if (keys[K_z]):                    
                    alpha += granularity if isValueInBetween(self.armLimits[ALPHA], alpha+granularity) else 0

                if (keys[K_x]):                    
                    alpha -= granularity if isValueInBetween(self.armLimits[ALPHA], alpha-granularity) else 0

                if (keys[K_a]):                    
                    beta += granularity if isValueInBetween(self.armLimits[BETA], beta+granularity) else 0

                if (keys[K_s]):                    
                    beta -= granularity if isValueInBetween(self.armLimits[BETA], beta-granularity) else 0

                if (keys[K_q]):                    
                    gamma += granularity if isValueInBetween(self.armLimits[GAMMA], gamma+granularity) else 0

                if (keys[K_w]):                    
                    gamma -= granularity if isValueInBetween(self.armLimits[GAMMA], gamma-granularity) else 0

                newAngle = (alpha, beta, gamma)                
                tempArm = copy.deepcopy(self.arm)
                tempArm.setArmAngle(newAngle)
                armEnd = tempArm.getEnd()
                armPos = tempArm.getArmPos()
                armPosDist = tempArm.getArmPosDist()

                print("doesArmTouchObjects", doesArmTouchObjects(armPosDist, self.obstacles))

                # print(armPosDist)
                
                if doesArmTouchObjects(armPosDist, self.obstacles) or not isArmWithinWindow(armPos, self.window):
                    continue

                if not doesArmTipTouchGoals(armEnd, self.goals) and doesArmTouchObjects(armPosDist, self.goals, isGoal=True):
                    continue
                
                self.arm.setArmAngle(newAngle)
                self.gameLoop()
                currAngle = copy.deepcopy(newAngle)

                if doesArmTipTouchGoals(armEnd, self.goals):
                    self.gameLoop()
                    print("SUCCESS")
                    raise SystemExit


        if saveImage:
            pygame.image.save(self.displaySurface, saveImage)

        if saveMaze and not self.__human:
            maze.saveToFile(saveMaze)
            

    def gameLoop(self):
        self.clock.tick(self.fps)
        self.displaySurface.fill(WHITE)
        self.drawTrajectory()
        self.drawArm()
        self.drawObstacles()
        self.drawGoal()
        pygame.display.flip()
      

    def drawTrajectory(self):
        cnt = 1
        for armPos in self.trajectory:
            x = (255 - 255/len(self.trajectory)*cnt)
            color = (x, x, x)
            cnt += 1
            for i in range(len(armPos)):
                pygame.draw.line(self.displaySurface, color, armPos[i][0], armPos[i][1], ARM_LINKS_WIDTH[i])  


    def drawArm(self):
        armPos = self.arm.getArmPos()
        for i in range(len(armPos)):
            pygame.draw.line(self.displaySurface, BLACK, armPos[i][0], armPos[i][1], ARM_LINKS_WIDTH[i])  


    def drawObstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.circle(self.displaySurface, RED, (obstacle[0], obstacle[1]), obstacle[2])


    def drawGoal(self):
        for goal in self.goals:
            pygame.draw.circle(self.displaySurface, BLUE, (goal[0], goal[1]), goal[2])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CS440 MP2 Robotic Arm')
    
    parser.add_argument('--config', dest="configfile", type=str, default = "test_config.txt",
                        help='configuration filename - default BasicMap')
    parser.add_argument('--map', dest="map_name", type=str, default = "BasicMap",
                        help='configuration filename - default BasicMap')
    parser.add_argument('--method', dest="search", type=str, default = "bfs", 
                        choices = ["bfs"],
                        help='search method - default bfs')
    parser.add_argument('--human', default = False, action = "store_true",
                        help='flag for human playable - default False')
    parser.add_argument('--fps', dest="fps", type=int, default = DEFAULT_FPS,
                        help='fps for the display - default '+str(DEFAULT_FPS))
    parser.add_argument('--granularity', dest="granularity", type=int, default = DEFAULT_GRANULARITY,
                        help='degree granularity - default '+str(DEFAULT_GRANULARITY))
    parser.add_argument('--trajectory', dest="trajectory", type=int, default = 0, 
                        help='leave footprint of rotation trajectory in every x moves - default 0')
    parser.add_argument('--save-image', dest="saveImage", type=str, default = None, 
                        help='save output to image file - default not saved')
    parser.add_argument('--save-maze', dest="saveMaze", type=str, default = None, 
                        help='save the contructed maze to maze file - default not saved')
    
    args = parser.parse_args()
    app = Application(args.configfile, args.map_name, args.human, args.fps)
    app.execute(args.search, args.granularity, args.trajectory, args.saveImage, args.saveMaze)