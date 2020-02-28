# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
import queue as q

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    return_path = []

    s = maze.getStart()

    # if start is the goal
    if maze.isObjective(s[0], s[1]):
        return_path.append(s)
        return return_path

    # queue for the bfs
    queue = []
    queue.append(s)

    # set to keep track of visited
    visited = set()
    visited.add(s)

    # a map to keep track of the previous aka parent node
    prev = {}

    # bfs traversal
    while queue:
        s = queue.pop(0)
        if maze.isObjective(s[0], s[1]):
            return_path = [s]
            while return_path[-1] != maze.getStart():
                return_path.append(prev[return_path[-1]])
            return_path.reverse()
            return return_path

        neighbors = maze.getNeighbors(s[0], s[1])

        for i in neighbors:
            if i not in visited and i not in queue:
                prev[i] = s
                queue.append(i)
                visited.add(i)
