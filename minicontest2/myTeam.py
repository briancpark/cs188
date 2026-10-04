# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
from game import Directions, Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class PlanningAgent(CaptureAgent):
  """
  Shared machinery: the maze graph, our side of the border, dead-end pockets,
  and a BFS that only expands cells passing a safety test.

  Enemy positions are always visible in this contest (the sonar code in
  capture.py is disabled), so agents plan against exact ghost positions.
  """

  # Ghosts farther than this are ignored when planning a path: we replan
  # every move, so a ghost that starts chasing gets avoided once it is close.
  SAFETY_RADIUS = 3

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    walls = gameState.getWalls()
    self.width = walls.width
    self.neighbors = {}
    for x in range(walls.width):
      for y in range(walls.height):
        if not walls[x][y]:
          self.neighbors[(x, y)] = [(nx, ny) for nx, ny in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y))
                                    if not walls[nx][ny]]
    homeX = self.width // 2 - 1 if self.red else self.width // 2
    self.border = [(homeX, y) for y in range(walls.height) if not walls[homeX][y]]
    self.computePockets()

  def computePockets(self):
    """
    Peels dead ends off the maze. For each peeled cell, records the cell where
    its pocket joins the rest of the maze and how far inside the pocket it is.
    A pacman inside a pocket can only escape through that entrance.
    """
    degree = {c: len(n) for c, n in self.neighbors.items()}
    removed = set()
    stack = [c for c, d in degree.items() if d <= 1]
    while stack:
      c = stack.pop()
      if c in removed:
        continue
      removed.add(c)
      for n in self.neighbors[c]:
        if n not in removed:
          degree[n] -= 1
          if degree[n] == 1:
            stack.append(n)
    self.pocketEntrance = {}
    self.pocketDepth = {}
    frontier = []
    for c in removed:
      for n in self.neighbors[c]:
        if n not in removed and c not in self.pocketEntrance:
          self.pocketEntrance[c] = n
          self.pocketDepth[c] = 1
          frontier.append(c)
    while frontier:
      nxt = []
      for c in frontier:
        for n in self.neighbors[c]:
          if n in removed and n not in self.pocketEntrance:
            self.pocketEntrance[n] = self.pocketEntrance[c]
            self.pocketDepth[n] = self.pocketDepth[c] + 1
            nxt.append(n)
      frontier = nxt

  def onOurSide(self, pos):
    return (pos[0] < self.width // 2) == self.red

  def enemies(self, gameState):
    return [(i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)
            if gameState.getAgentPosition(i) is not None]

  def dangerousGhosts(self, gameState, horizon=0):
    """Positions of enemy ghosts that can still eat us `horizon` moves from now."""
    return [s.getPosition() for _, s in self.enemies(gameState)
            if not s.isPacman and s.scaredTimer <= horizon + 2]

  def bfs(self, start, allowed):
    """Returns (steps, parent) for every cell reachable through allowed(cell, step)."""
    steps = {start: 0}
    parent = {start: None}
    frontier = [start]
    while frontier:
      nxt = []
      for c in frontier:
        t = steps[c] + 1
        for n in self.neighbors[c]:
          if n not in steps and allowed(n, t):
            steps[n] = t
            parent[n] = c
            nxt.append(n)
      frontier = nxt
    return steps, parent

  def firstMove(self, start, target, parent):
    cell = target
    while parent[cell] != start:
      cell = parent[cell]
    return Actions.vectorToDirection((cell[0] - start[0], cell[1] - start[1]))

  def moveToward(self, gameState, start, targets, steps, parent):
    """Action along the shortest allowed path to the closest target, or None."""
    reachable = [t for t in targets if t in steps and t != start]
    if not reachable:
      return None
    target = min(reachable, key=lambda t: steps[t])
    action = self.firstMove(start, target, parent)
    if action in gameState.getLegalActions(self.index):
      return action
    return None

  def safeCells(self, gameState):
    """
    BFS predicate for a pacman: a cell on the enemy side is unsafe if a ghost
    that is still dangerous then could get there by the time we do.
    """
    ghosts = [(s.getPosition(), s.scaredTimer) for _, s in self.enemies(gameState) if not s.isPacman]
    dist = self.getMazeDistance

    def allowed(cell, t):
      if self.onOurSide(cell):
        return True
      for g, scared in ghosts:
        if scared <= t + 2 and dist(g, cell) <= min(t + 1, self.SAFETY_RADIUS):
          return False
      return True
    return allowed

  def escapeIsSafe(self, cell, t, ghosts):
    """Whether we can get back out of the pocket containing `cell` before a ghost blocks it."""
    if cell not in self.pocketEntrance:
      return True
    entrance = self.pocketEntrance[cell]
    exitTime = t + self.pocketDepth[cell]
    return all(self.getMazeDistance(g, entrance) > exitTime + 1 for g in ghosts)

  def chase(self, gameState, myPos, targetPos):
    """Step toward an enemy pacman on our side, staying on our side."""
    steps, parent = self.bfs(myPos, lambda c, t: self.onOurSide(c))
    return self.moveToward(gameState, myPos, [targetPos], steps, parent)


class OffensiveAgent(PlanningAgent):
  """
  Raids enemy food along paths no ghost can intercept, skips pockets a ghost
  could seal off, and banks food when threatened, loaded, or out of time.
  """

  CARRY_LIMIT = 6

  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # While still a ghost at home, pick off invaders that wander close.
    if not myState.isPacman and myState.scaredTimer == 0:
      invaders = [s.getPosition() for _, s in self.enemies(gameState) if s.isPacman]
      close = [p for p in invaders if self.getMazeDistance(myPos, p) <= 4]
      if close:
        action = self.chase(gameState, myPos, min(close, key=lambda p: self.getMazeDistance(myPos, p)))
        if action:
          return action
    return self.raid(gameState)

  def raid(self, gameState, mustReturn=False):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    carrying = myState.numCarrying
    foods = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    ghosts = self.dangerousGhosts(gameState)
    movesLeft = gameState.data.timeleft // 4
    homeDist = min(self.getMazeDistance(myPos, b) for b in self.border)
    nearestGhost = min([self.getMazeDistance(myPos, g) for g in ghosts] or [99])

    steps, parent = self.bfs(myPos, self.safeCells(gameState))

    goHome = carrying > 0 and (
      mustReturn
      or len(foods) <= 2
      or movesLeft <= homeDist + 4
      or carrying >= self.CARRY_LIMIT
      or (nearestGhost <= 5 and carrying >= 2))
    if goHome:
      action = self.moveToward(gameState, myPos, self.border, steps, parent)
      if action:
        return action

    if len(foods) > 2 and movesLeft > homeDist + 4:
      targets = [f for f in foods if f in steps and self.escapeIsSafe(f, steps[f], ghosts)]
      # Leave food that a raiding teammate is closer to.
      for i in self.getTeam(gameState):
        mate = gameState.getAgentState(i)
        if i != self.index and mate.isPacman:
          matePos = mate.getPosition()
          split = [f for f in targets if steps[f] <= self.getMazeDistance(matePos, f)]
          targets = split or targets
      if nearestGhost <= 6:
        targets += [c for c in capsules if c in steps]
      action = self.moveToward(gameState, myPos, targets, steps, parent)
      if action:
        return action

    # Nothing safe to eat. Bank what we have, otherwise probe the border at
    # the point farthest from the enemy ghosts.
    if carrying > 0 or myState.isPacman:
      action = self.moveToward(gameState, myPos, self.border + capsules, steps, parent)
      if action:
        return action
      return self.flee(gameState, myPos, ghosts)
    return self.probe(gameState, myPos, ghosts, steps, parent)

  def probe(self, gameState, myPos, ghosts, steps, parent):
    def openness(b):
      return min([self.getMazeDistance(g, b) for g in ghosts] or [99]) - steps.get(b, 99) / 2.0
    target = max(self.border, key=openness)
    if target == myPos:
      return Directions.STOP
    return self.moveToward(gameState, myPos, [target], steps, parent) or Directions.STOP

  def flee(self, gameState, myPos, ghosts):
    def score(action):
      pos = Actions.getSuccessor(myPos, action)
      pos = (int(pos[0]), int(pos[1]))
      return min([self.getMazeDistance(g, pos) for g in ghosts] or [99]) * 10 \
          - min(self.getMazeDistance(pos, b) for b in self.border)
    return max(gameState.getLegalActions(self.index), key=score)


class DefensiveAgent(OffensiveAgent):
  """
  Hunts invaders on our side, heading home first if it is out raiding. When
  no enemy is close enough to invade before it could get back, it joins the
  raid so a lone enemy defender has two attackers to cover.
  """

  RAID_MARGIN = 3

  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    invaders = [s.getPosition() for _, s in self.enemies(gameState) if s.isPacman]

    if myState.isPacman:
      if invaders or not self.canRaid(gameState, myPos):
        return self.raid(gameState, mustReturn=True) if myState.numCarrying else self.goHome(gameState, myPos)
      return self.raid(gameState)

    if invaders:
      return self.defend(gameState, myPos, myState, invaders)
    if self.canRaid(gameState, myPos):
      return self.raid(gameState)
    return self.guard(gameState, myPos)

  def canRaid(self, gameState, myPos):
    """True while every enemy is farther from our border than we are from home."""
    myHome = min(self.getMazeDistance(myPos, b) for b in self.border)
    for _, s in self.enemies(gameState):
      enemyToBorder = min(self.getMazeDistance(s.getPosition(), b) for b in self.border)
      if enemyToBorder <= myHome + self.RAID_MARGIN:
        return False
    return True

  def goHome(self, gameState, myPos):
    steps, parent = self.bfs(myPos, self.safeCells(gameState))
    return self.moveToward(gameState, myPos, self.border, steps, parent) \
        or self.flee(gameState, myPos, self.dangerousGhosts(gameState))

  def defend(self, gameState, myPos, myState, invaders):
    steps, parent = self.bfs(myPos, lambda c, t: self.onOurSide(c))
    target = min(invaders, key=lambda p: self.getMazeDistance(myPos, p))
    if myState.scaredTimer > 0:
      # A scared ghost dies on contact: trail at distance 2 instead.
      ring = [c for c in steps if self.getMazeDistance(c, target) == 2]
      action = self.moveToward(gameState, myPos, ring, steps, parent)
      return action or self.keepAway(gameState, myPos, target)
    return self.moveToward(gameState, myPos, [target], steps, parent) or Directions.STOP

  def guard(self, gameState, myPos):
    """Stand on the border opposite the enemy closest to crossing."""
    steps, parent = self.bfs(myPos, lambda c, t: self.onOurSide(c))
    enemyPositions = [s.getPosition() for _, s in self.enemies(gameState)]
    threat = min(enemyPositions, key=lambda p: min(self.getMazeDistance(p, b) for b in self.border))
    guard = min(self.border, key=lambda b: (self.getMazeDistance(threat, b), self.getMazeDistance(myPos, b)))
    if guard == myPos:
      return Directions.STOP
    return self.moveToward(gameState, myPos, [guard], steps, parent) or Directions.STOP

  def keepAway(self, gameState, myPos, target):
    def score(action):
      pos = Actions.getSuccessor(myPos, action)
      return self.getMazeDistance((int(pos[0]), int(pos[1])), target)
    return max(gameState.getLegalActions(self.index), key=score)
