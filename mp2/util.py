# util.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) 
#            Krishna Harsha (kk20@illinois.edu) on 09/12/2018

"""
This file contains helper functions that helps other modules, 
"""

# Transform between angles (alpha, beta, gamma) and array index
def angleToIdx(angles, offsets, granularity):
    result = []
    for i in range(len(angles)):
        result.append(int((angles[i]-offsets[i]) / granularity))
    return tuple(result)

def idxToAngle(index, offsets, granularity):
    result = []
    for i in range(len(index)):
        result.append(int((index[i]*granularity)+offsets[i]))
    return tuple(result)

def isValueInBetween(valueRange, target):
    if target < min(valueRange) or target > max(valueRange):
        return False
    else:
        return True