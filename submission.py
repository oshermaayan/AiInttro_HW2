import random, util, math
from game import Agent, Directions
from util import manhattanDistance
from pacman import GameState

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    #return scoreEvaluationFunction(successorGameState)
    #Use the better evaluation function instead
    return betterEvaluationFunction(successorGameState)

#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """
    # TODO: Maybe encourage finishing areas with little food first?
    # Todo: play with coeffs
    # Todo: change distances from constants to grid-size dependent variables
    #
    vicinityDistance = 3
    eps = 10e-4
    pacmanPosition = gameState.getPacmanPosition()
    evalValue = 0.0
    evalValue += gameState.getScore() ### Change scales of coeffs


    #Walls related
    wallsGrid = gameState.getWalls().data
    wallGridRows = len(wallsGrid)  # currentFoodGrid.height
    wallGridCols = len(wallsGrid[0])
    surroundingWallNum = 0
    for row in range(wallGridRows):
        for col in range(wallGridCols):
            if wallsGrid[row][col]==True:
                #Square has wall
                foodPos = (row,col)
                if manhattanDistance(pacmanPosition,foodPos) <= 1:
                    surroundingWallNum += 1

    #if surroundingWallNum == 3:
        #Surrounded by walls
        #evalValue += -100

    #Food-related parameters
    numOfNearFood = 0
    currentFoodGrid = gameState.getFood().data
    gridRows = len(currentFoodGrid)#currentFoodGrid.height
    gridCols = len(currentFoodGrid[0])

    for row in range(gridRows):
        for col in range(gridCols):
            if currentFoodGrid[row][col]==True:
                #Square has food
                foodPos = (row,col)
                if manhattanDistance(pacmanPosition,foodPos) <= vicinityDistance:
                        numOfNearFood += 1


    if gameState.getNumFood() < 5:
        #Encourage moving towards food towards the end of the game
            evalValue += 10*numOfNearFood
    else:
        evalValue += numOfNearFood

    evalValue -= gameState.getNumFood()#= 1/(gameState.getNumFood()+eps) # Number of food items left

    #Ghost-related information
    nearThreatGhostsNum = 0
    nearThreatGhostsScore = 0.0
    nearScaredGhostsNum = 0
    nearScaredGhostsScore = 0.0

    for ghostPos, ghostState in zip(gameState.getGhostPositions(), gameState.getGhostStates()):
        distanceFromGhost = manhattanDistance(pacmanPosition, ghostPos)
        if distanceFromGhost <= vicinityDistance:
            if ghostState.scaredTimer > 0:
                nearScaredGhostsNum += 1
                nearScaredGhostsScore += 1/(distanceFromGhost + eps) # maybe remove and use num instead?
            else:
                #Ghost is a threat to Pacman :O
                nearThreatGhostsNum +=1
                nearThreatGhostsScore += 1/(distanceFromGhost + eps)

    evalValue -= 100*nearThreatGhostsScore ### Play with coefficent 100/10000/...
    evalValue += nearScaredGhostsScore

    # Capsules information
    capsulesPositions = gameState.getCapsules()
    nearCapsulesNum = 0

    if nearThreatGhostsNum > 0:
        for capsulePos in capsulesPositions:
            if manhattanDistance(pacmanPosition,capsulePos) < vicinityDistance:
                nearCapsulesNum +=1

    evalValue += 100*nearCapsulesNum

    return evalValue

#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

  def isFinalState(self, gameState):
    pacmanLegalAction = gameState.getLegalActions()
    return gameState.isLose() or gameState.isWin() or len(pacmanLegalAction)==0

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """
    """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
        The stop direction

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.getScore():
        Returns the score corresponding to the current state of the game

        gameState.isWin():
        Returns True if it's a winning state

        gameState.isLose():
        Returns True if it's a losing state

        self.depth:
        The depth to which search should continue

        """

    def getAction(self, gameState):
        '''Returns one of the following actions: North, South, East, West, Stop'''
        # Call recursive auxilary function
        # start with Pacman - agent #0
        return self.getActionAux(gameState, self.index, self.depth)


    def getActionAux(self, gameState, agent, depth):
        if self.isFinalState(gameState):
            return gameState.getScore()
        if depth == 0:
            return self.evaluationFunction(gameState)

        numOfAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agent)
        nextAgent = (agent + 1) % numOfAgents

        if agent == self.index:
            #Pacman's turn
            # Initializing values
            bestMaxScore = -math.inf
            wantedMove = Directions.STOP

            ###actions = [action for action in legalActions]
            nextStates = [gameState.generateSuccessor(agent, action) for action in legalActions]
            scores = [self.getActionAux(state, nextAgent, depth) for state in nextStates]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            wantedMove = legalActions[chosenIndex]

            # If we're at the root of the game tree - returned the preferred move
            # else - return the score
            if depth == self.depth:
                return wantedMove
            else:
                return bestScore
        else:
            #Ghost (min player)
            bestMinScore = math.inf # best score for the min_agent is the lowest score
            for action in legalActions:
                nextState = gameState.generateSuccessor(agent, action)
                if nextAgent == self.index:
                    #This is the last ghost's turn, next turn is Pacman's
                    if depth == 1: ### maybe 1?
                        #Next states are leaves (we've reached the maximum depth)
                        score = self.evaluationFunction(nextState)
                    else:
                        score = self.getActionAux(nextState, nextAgent, depth - 1)
                else:
                    score = self.getActionAux(nextState, nextAgent, depth)

                bestMinScore = min(bestMinScore, score)
            return bestMinScore



######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    '''Returns one of the following actions: North, South, East, West, Stop'''
    # Call recursive auxilary function
    # start with Pacman - agent #0
    return self.getActionAux(gameState, self.index, self.depth, -math.inf, math.inf)


  def getActionAux(self, gameState, agent, depth, alpha, beta):
        if self.isFinalState(gameState):
            return gameState.getScore()

        if depth == 0:
            return self.evaluationFunction(gameState)

        numOfAgents = gameState.getNumAgents()
        nextAgent = (agent + 1) % numOfAgents
        legalActions = gameState.getLegalActions(agent)

        if agent == self.index:
            #Pacman's turn
            # Initializing values
            bestMaxScore = -math.inf
            wantedMove = Directions.STOP
            wantedMoves=[]
            for action in legalActions:
                nextState = gameState.generateSuccessor(agent,action)
                '''Note: depth is decreased at Ghost's turn'''
                ###Osher - notice the last line
                score = self.getActionAux(nextState, nextAgent, depth, alpha, beta)
                if score > bestMaxScore:
                    bestMaxScore = score
                    wantedMoves = [action]
                if score == bestMaxScore:
                    # Enable move selection from several moves with the best score
                    wantedMoves.append(action)
                alpha = max(alpha, bestMaxScore)
                if alpha >= beta: ## Osher: changed from alpha >= beta
                    ### Osher: check if this is the check we need - does Beta hold the right value?
                    return math.inf
            # If we're at the root of the game tree - returned the preferred move
            # else - return the score
            if depth == self.depth:
                bestIndices = [index for index in range(len(wantedMoves))]
                return wantedMoves[random.choice(bestIndices)]
            else:
                return bestMaxScore
        else:
            #Ghost (min player)
            bestMinScore = math.inf # best score for the min_agent is the lowest score
            changed = False
            for action in legalActions:
                nextState = gameState.generateSuccessor(agent, action)
                if nextAgent == self.index:
                    #This is the last ghost's turn, next turn is Pacman's
                    if depth == 1: ### maybe 1?
                        #Next states are leaves (we've reached the maximum depth)
                        score = self.evaluationFunction(nextState)
                    else:
                        score = self.getActionAux(nextState, nextAgent, depth - 1, alpha, beta)
                else:
                    score = self.getActionAux(nextState, nextAgent, depth, alpha, beta)
                if score != -math.inf:
                    bestMinScore = min(bestMinScore, score)
                    changed = True
                    beta = min(bestMinScore, beta)
                if alpha >= beta: ### Osher: changed from alpha >= beta
                    ### Same check as for max agent
                    return -math.inf
            if not changed:
                ### Osher: why do we need this if statement?
                return -math.inf
            return bestMinScore

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """
    return self.getActionAux(gameState, self.index, self.depth)

  def getActionAux(self, gameState, agent, depth):
        if self.isFinalState(gameState):
            return gameState.getScore()

        if depth == 0:
            return self.evaluationFunction(gameState)

        numOfAgents = gameState.getNumAgents()
        nextAgent = (agent + 1) % numOfAgents
        legalActions = gameState.getLegalActions(agent)

        ###actions = [action for action in legalActions]
        nextStates = [gameState.generateSuccessor(agent, action) for action in legalActions]

        if agent == self.index:
            # Pacman's turn
            # Initializing values
            bestMaxScore = -math.inf
            wantedMove = Directions.STOP

            scores = [self.getActionAux(state, nextAgent, depth) for state in nextStates]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            wantedMove = legalActions[chosenIndex]

            # If we're at the root of the game tree - returned the preferred move
            # else - return the score
            if depth == self.depth:
                return wantedMove
            else:
                return bestScore
        else:
            # Ghost (min player) - randomGhost
            totalScore = 0  # best score for the min_agent is the lowest score
            for state in nextStates:
                if nextAgent == self.index:
                    # This is the last ghost's turn, next turn is Pacman's
                    if depth == 1:  ### maybe 1?
                        # Next states are leaves (we've reached the maximum depth)
                        totalScore += self.evaluationFunction(state)
                    else:
                        totalScore += self.getActionAux(state, nextAgent, depth - 1)
                else:
                    totalScore += self.getActionAux(state, nextAgent, depth)
            assert len(nextStates) != 0 ### Just for sanity check
            return totalScore/len(nextStates)


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



