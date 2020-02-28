# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    end_x = start[0] + int(length*np.cos(np.radians(angle)))
    end_y = start[1] - int(length*np.sin(np.radians(angle)))
    return (end_x, end_y)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    for each_arm in armPosDist:
        v0 = np.array(each_arm[0])
        v1 = np.array(each_arm[1])
        v01 = v1-v0
        pad = each_arm[2]
        if isGoal == True:
            pad = 0

        for obj in objects:
            q = np.array((obj[0], obj[1]))
            r = obj[2]
            v0q = q - v0 

            #projection scalar of v0q onto v01
            v0q_01 = np.dot(v0q, v01)

            if v0q_01 > np.dot(v01,v01):
                v1q = q - v1
                len_v1q = np.sqrt(np.dot(v1q, v1q))
                if len_v1q - (r + pad) <= 0:
                    return True

            elif v0q_01 >= 0 and np.dot(v01, v01) - v0q_01 >= 0:
                v1q = v0q - ((v0q_01)*(v01/(v01[0]**2 + v01[1]**2)))
                len_v1q = np.sqrt(v1q[0]**2 + v1q[1]**2)

                if len_v1q - (r + pad) <= 0:
                    return True

            elif v0q_01 < 0:
                v1q = q - v0
                len_v1q = np.sqrt(np.dot(v1q, v1q))
                if len_v1q - (r + pad) <= 0:
                    return True

    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tick touches any goal. False if not.
    """
    for g in goals:
        d = np.sqrt((armEnd[0]-g[0])**2 + (armEnd[1]-g[1])**2) 
        if d <= g[2]:
            return True

    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    wx = window[0]
    wy = window[1]

    for a in armPos:
        a0 = a[0]
        a1 = a[1]
        if (a0[0] > wx or a0[0] < 0) or (a0[1] > wy or a0[1] < 0) or (a1[0] > wx or a1[0] < 0) or (a1[1] > wy or a1[1] < 0):
            return False 
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
