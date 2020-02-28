# armPart.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the ArmLink class
"""

from geometry import *

class ArmLink:
    def __init__(self, base, length, angle, distance=0):
        # This angle is absolute angle, not alpha/beta/gamma
        self.__base = base
        self.__length = length        
        self.__angle = angle
        self.__distance = distance

    def setBase(self, base):
        self.__base = base                

    def setAngle(self, angle):
        # This angle is absolute angle, not alpha or beta or gamma        
        self.__angle = angle             

    def getBase(self):
        return self.__base

    def getLength(self):
        return self.__length

    def getAngle(self):
        return self.__angle

    def getDistance(self):
        return self.__distance

    def computeEnd(self):
        """This function computes the end position of this arm link for the given angle.
           Note that the angle here is counter-clockwise from the x-axis. 
        """        
        self.__end = computeCoordinate(self.__base, self.__length, self.__angle)

    def getEnd(self):
        self.computeEnd()
        return self.__end  