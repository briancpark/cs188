# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent, Actions, Directions
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=0)]

# Compute spent in getAction costs 1000 points per second, but planning in
# registerInitialState is free as long as it finishes within the 1s startup
# alarm. So MyAgent plans the whole food tour up front and only replays it.
PLANNING_BUDGET = 0.5

class MyAgent(Agent):
    """
    Plans a short open tour over every food pellet at startup (nearest
    neighbor, then 2-opt and or-opt), then replays the resulting actions.
    """

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """
        if self.plan:
            return self.plan.pop()
        return Directions.STOP

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """
        self.plan = []

    def registerInitialState(self, state):
        deadline = time.time() + PLANNING_BUDGET
        walls = state.getWalls()
        start = state.getPacmanPosition(self.index)
        foods = state.getFood().asList()
        if not foods:
            return

        # Flatten the open cells so BFS works on small integer ids.
        cells = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        cell_id = {c: i for i, c in enumerate(cells)}
        neighbors = []
        for x, y in cells:
            neighbors.append([cell_id[n] for n in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)) if n in cell_id])

        nodes = [cell_id[start]] + [cell_id[f] for f in foods]
        n = len(nodes)
        dist = [self.bfs(src, neighbors, len(cells)) for src in nodes]
        d = [[dist[i][nodes[j]] for j in range(n)] for i in range(n)]

        order = self.nearestNeighborTour(d, n)
        order = self.improveTour(order, d, deadline)
        self.plan = self.buildActions(order, nodes, cells, neighbors, foods, start)
        self.plan.reverse()

    def bfs(self, src, neighbors, num_cells):
        dist = [-1] * num_cells
        dist[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                du = dist[u] + 1
                for v in neighbors[u]:
                    if dist[v] < 0:
                        dist[v] = du
                        nxt.append(v)
            frontier = nxt
        return dist

    def nearestNeighborTour(self, d, n):
        order = [0]
        unvisited = set(range(1, n))
        while unvisited:
            row = d[order[-1]]
            nxt = min(unvisited, key=row.__getitem__)
            unvisited.remove(nxt)
            order.append(nxt)
        return order

    def improveTour(self, order, d, deadline):
        """
        Open-path 2-opt and or-opt with node 0 (the start) fixed in front.
        Stops early once the planning deadline passes.
        """
        n = len(order)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 2-opt: reverse order[i..j]. The edge after j may not exist.
            for i in range(1, n - 1):
                a, b = order[i - 1], order[i]
                da = d[a]
                for j in range(i + 1, n):
                    c = order[j]
                    if j + 1 < n:
                        e = order[j + 1]
                        delta = da[c] + d[b][e] - da[b] - d[c][e]
                    else:
                        delta = da[c] - da[b]
                    if delta < 0:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        b = order[i]
                        improved = True
                if time.time() >= deadline:
                    return order
            # or-opt: move a segment of 1-3 nodes to a better position.
            for seg in (1, 2, 3):
                i = 1
                while i + seg <= n:
                    prev, first, last = order[i - 1], order[i], order[i + seg - 1]
                    nxt = order[i + seg] if i + seg < n else None
                    removed = d[prev][first] + (d[last][nxt] - d[prev][nxt] if nxt is not None else 0)
                    rest = order[:i] + order[i + seg:]
                    segment = order[i:i + seg]
                    best, best_k, best_rev = 0, None, False
                    for k in range(1, len(rest) + 1):
                        p = rest[k - 1]
                        q = rest[k] if k < len(rest) else None
                        base = d[p][q] if q is not None else 0
                        for rev, (s, t) in ((False, (first, last)), (True, (last, first))):
                            added = d[p][s] + (d[t][q] if q is not None else 0) - base
                            if added - removed < best:
                                best, best_k, best_rev = added - removed, k, rev
                    if best_k is not None:
                        if best_rev:
                            segment.reverse()
                        order = rest[:best_k] + segment + rest[best_k:]
                        improved = True
                    i += 1
                    if time.time() >= deadline:
                        return order
        return order

    def buildActions(self, order, nodes, cells, neighbors, foods, start):
        """
        Walk the tour along shortest paths, skipping any target already
        eaten on the way to an earlier one.
        """
        remaining = set(nodes[1:])
        actions = []
        cur = nodes[0]
        for idx in order[1:]:
            target = nodes[idx]
            if target not in remaining:
                continue
            for v in self.shortestPath(cur, target, neighbors):
                actions.append(Actions.vectorToDirection((cells[v][0] - cells[cur][0], cells[v][1] - cells[cur][1])))
                remaining.discard(v)
                cur = v
        return actions

    def shortestPath(self, src, dst, neighbors):
        parent = {src: None}
        frontier = [src]
        while dst not in parent:
            nxt = []
            for u in frontier:
                for v in neighbors[u]:
                    if v not in parent:
                        parent[v] = u
                        nxt.append(v)
            frontier = nxt
        path = []
        while dst != src:
            path.append(dst)
            dst = parent[dst]
        path.reverse()
        return path

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
        return search.uniformCostSearch(problem)

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y] == True

