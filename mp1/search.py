# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
import queue as q
import itertools
from itertools import combinations
from itertools import permutations
import sys
import copy
from copy import deepcopy
import heapq

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    # the path to return
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


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    p_queue = q.PriorityQueue()
    start = maze.getStart()

    objectives = maze.getObjectives()
    s_node = (manhattan_heuristic(start, objectives),start)
    visited = set()
    visited.add(start)
    prev = {}
    p_queue.put(s_node)

    while p_queue:
        s = p_queue.get()
        s_pos = s[1]

        if s_pos in objectives:
            goal = s_pos
            return path(start, goal, prev)
        neighbors = maze.getNeighbors(s_pos[0], s_pos[1])

        for i in neighbors:
            if i not in visited:
                prev[i] = s_pos
                new_node = (manhattan_heuristic(i, objectives)+len(path(start, s_pos, prev)),i)
                p_queue.put(new_node)
                visited.add(s_pos)


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # For a general idea for this part I looked at this stackoverflow link: https://stackoverflow.com/questions/28570079/a-search-with-multiple-goals-python
    # TODO: Write your code here

    #start = maze.getStart()
    #goals = maze.getObjectives()


    #final_paths = []

    #combinations = list(itertools.permutations(goals, len(goals)));
    #for combo in combinations:
    #    t_list = [start]+list(combo)
    #    path = []
    #    for i in range(1, len(t_list)):
    #        c_maze = deepcopy(maze)
    #        c_maze.setStart(t_list[i-1])
    #        c_maze.setObjectives([t_list[i]])
    #        temp = astar(c_maze)
    #        del temp[0]
    #        path += temp
    #    final_paths.append(path)
    #f = min(final_paths, key=len)
    #f = [start]+f
    #return f

    return astar_multi(maze)



def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goals = maze.getObjectives()
    paths = []
    p_queue = q.PriorityQueue()

    for i in range(len(goals)):
        for j in range(i+1, len(goals)):
            paths.append((i, j))

    len_map = {}
    for path in paths:
        c_maze = deepcopy(maze)
        c_maze.setStart(goals[path[0]])
        c_maze.setObjectives([goals[path[1]]])
        dist = len(astar(c_maze))
        len_map[path] = dist-1

    # a map of the previous or parent nodes
    prev = {(start, tuple(goals)):None}

    # the priority queue gets (f, distance to current node(g), (current node, remaing goals))
    s_node = (mst_heuristic(start, tuple(goals), len_map, goals)+0, 0, (start, tuple(goals)))
    p_queue.put(s_node)

    cur_node_dst_map = {s_node[2]:0}

    while p_queue:
        cur = p_queue.get()
        cur_pos = cur[2][0]
        if (len(cur[2][1]) == 0):
            return getPath(cur[2], prev)

        neighbors = maze.getNeighbors(cur_pos[0], cur_pos[1])
        for n in neighbors:
            goals_from_node = tuple(goals_to_get(n, cur[2][1]))
            dst_node = (n, goals_from_node)
            if dst_node in cur_node_dst_map and cur_node_dst_map[dst_node] <= cur_node_dst_map[cur[2]]+1:
                continue
            #update distance map of the node
            cur_node_dst_map[dst_node] = cur_node_dst_map[cur[2]]+1
            #update node's parent
            prev[dst_node] = cur[2]

            #update p_queue: this part is borrowed from the textbook Fig. 3.26
            old_f = cur[0]
            new_f = cur_node_dst_map[dst_node]+mst_heuristic(n, goals_from_node, len_map, goals)
            new_f = max(old_f, new_f)

            new_node = (new_f, cur_node_dst_map[dst_node], dst_node)
            p_queue.put(new_node)



def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goals = maze.getObjectives()
    paths = []
    p_queue = q.PriorityQueue()

    for i in range(len(goals)):
        for j in range(i+1, len(goals)):
            paths.append((i, j))

    len_map = {}
    for path in paths:
        c_maze = deepcopy(maze)
        c_maze.setStart(goals[path[0]])
        c_maze.setObjectives([goals[path[1]]])
        dist = len(astar(c_maze))
        len_map[path] = dist

    # a map of the previous or parent nodes
    prev = {(start, tuple(goals)):None}

    # the priority queue gets (f, distance to current node(g), (current node, remaing goals))
    s_node = (mst_heuristic(start, tuple(goals), len_map, goals)+0, 0, (start, tuple(goals)))
    p_queue.put(s_node)

    cur_node_dst_map = {s_node[2]:0}

    while p_queue:
        cur = p_queue.get()
        cur_pos = cur[2][0]
        if (len(cur[2][1]) == 0):
            return getPath(cur[2], prev)

        neighbors = maze.getNeighbors(cur_pos[0], cur_pos[1])
        for n in neighbors:
            goals_from_node = tuple(goals_to_get(n, cur[2][1]))
            dst_node = (n, goals_from_node)
            if dst_node in cur_node_dst_map and cur_node_dst_map[dst_node] <= cur_node_dst_map[cur[2]]+1:
                continue
            #update distance map of the node
            cur_node_dst_map[dst_node] = cur_node_dst_map[cur[2]]+1
            #update node's parent
            prev[dst_node] = cur[2]

            #update p_queue: this part is borrowed from the textbook Fig. 3.26
            old_f = cur[0]
            new_f = cur_node_dst_map[dst_node]+mst_heuristic(n, goals_from_node, len_map, goals)
            #new_f = max(old_f, new_f)

            new_node = (new_f, cur_node_dst_map[dst_node], dst_node)
            p_queue.put(new_node)

# extra functions used by me
def manhattan_dst(start, end):
    dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
    return dist

def manhattan_heuristic(node, objectives):
    # set the min value to a large number
    min = 9223372036854775807
    for i in objectives:
        h = manhattan_dst(node, i)
        min = h if h < min else min
    return min

def path(start, end, prev_map):
    return_path = [end]
    while return_path[-1] != start:
        return_path.append(prev_map[return_path[-1]])
    return_path.reverse()
    return return_path

def getPath(cur, prev_map):
    p = []
    while cur != None:
        p.append(cur[0])
        cur = prev_map[cur]
    p.reverse()
    return p

def goals_to_get(node, goals):
    ret = []
    for i in goals:
        if node != i:
            ret.append(i)
    return ret

# heuristic for part 3
def mst_heuristic(node, goals, lmap, objectives):
    if len(goals) == 0:
        return 0
    result = 0
    cur_v = [objectives.index(goals[0])]
    vertices = []
    for i in range(1, len(goals)):
        vertices.append(objectives.index(goals[i]))
    while len(cur_v) != len(goals):
        min_paths = []
        for cv in cur_v:
            min_nv = sys.maxsize
            min_n = None
            for vert in vertices:
                if vert < cv:
                    edge = (vert, cv)
                else:
                    edge = (cv, vert)
                if lmap[edge] < min_nv:
                    min_nv = lmap[edge]
                    min_n = vert
            min_paths.append((min_nv, min_n))
        min_p = min(min_paths)
        vertices.remove(min_p[1])
        result += min_p[0]
        cur_v.append(min_p[1])
    l = []
    for x in goals:
        l.append(manhattan_dst(node, x))
    return result + min(l)
