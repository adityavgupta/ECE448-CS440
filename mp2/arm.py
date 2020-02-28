# arm.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the Arm class
"""

from const import *
from armLink import ArmLink

class Arm:
    def __init__(self, armBasePos, armLinkSpec):

        if len(armLinkSpec) > MAX_NUM_OF_ART_LINKS:
            print("Maximum number of arm links is %d" % (MAX_NUM_OF_ART_LINKS))
            raise SystemExit

        self.__armLinks = []
        self.__armRelativeAngle = []
        self.__armLimit = []

        base = armBasePos
        totalRelativeAngle = 0
        for i in range(len(armLinkSpec)):
            length, relativeAngle, distance, limit = armLinkSpec[i]
            if relativeAngle < min(limit) or relativeAngle > max(limit):
                print("The given relativeAngle is not in available range. Set to minimum.")
                relativeAngle = min(limit)
            self.__armLimit.append(limit)
            self.__armRelativeAngle.append(relativeAngle)
            totalRelativeAngle += relativeAngle
            armLink = ArmLink(base, length, totalRelativeAngle % 360, distance)
            self.__armLinks.append(armLink)
            base = armLink.getEnd()        


    def getBase(self):
        """This function returns (x, y) of the arm base
        """
        return self.__armLinks[0].getBase()

    def getEnd(self):
        """This function returns (x, y) of the arm tip
        """
        return self.__armLinks[-1].getEnd()

    def getArmPos(self):
        """This function returns (start, end) of all arm links
           For example, if there are two arm links, the return value would be '
           [ [(x1, y1), (x2, y2)], 
             [(x2, y2), (x3, y3)] ]
        """
        info = []
        for armLink in self.__armLinks:
            info.append((armLink.getBase(), armLink.getEnd()))
        return info
    
    def getArmPosDist(self):
        """This function returns (start, end) of all arm links with the padding distance of the arm
           For example, if there are two arm links, the return value would be '
           [ [(x1, y1), (x2, y2), distance], 
             [(x2, y2), (x3, y3), distance] ]
        """
        info = [(armLink.getBase(), armLink.getEnd(), armLink.getDistance()) for armLink in self.__armLinks]
        return info

    def getArmAngle(self):
        """This function returns relative angles of all arm links.
           If there are two arm links, the return value would be (alpha, beta) 
        """
        return self.__armRelativeAngle

    def getArmLimit(self):        
        """This function returns (min angle, max angle) of all arm links
        """
        return self.__armLimit

    def getNumArmLinks(self):
        """This function returns the number of arm links of this arm
        """
        return len(self.__armLinks)

    def setArmAngle(self, angles):    
        """This function sets angles(alpha, beta, gamma) for all arm links
        """
        angles = angles[:self.getNumArmLinks()]

        for i in range(len(angles)):
            if angles[i] < min(self.__armLimit[i]) or angles[i] > max(self.__armLimit[i]):
                return False

        self.__armRelativeAngle = angles
        totalAngle = 0
        base = self.getBase()
        for i in range(len(self.__armRelativeAngle)):
            totalAngle += self.__armRelativeAngle[i]
            self.__armLinks[i].setAngle(totalAngle % 360)
            self.__armLinks[i].setBase(base)
            base = self.__armLinks[i].getEnd()

        return True
