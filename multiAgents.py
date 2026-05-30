# multiAgents.py
# --------------
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

from game import Agent
from copy import deepcopy
from pacman import GameState
from game import Grid
from util import Queue
import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random
import util
random.seed(42)  # For reproducibility


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        ####################################################################
        # Daniel y Crespo
        ####################################################################
        self.heatmap = None
        self.frame = 0

        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()

        ####################################################################
        # Daniel y Crespo
        ####################################################################
        layout = state.getWalls()
        height = layout.height
        width = layout.width

        # We create the heatmap only one time, as it is fixed
        if self.heatmap is None:
            ghost_positions = []
            for g in ghost_states:
                x, y = g.getPosition()
                ghost_positions.append((int(x), int(y)))
            self.heatmap = createHeatMap(layout, ghost_positions)

        # We make it local to add our heuristics
        heatmap = deepcopy(self.heatmap)

        # Factor 1: Distancia a la comida más cercana
        # Replaced manhattan distances by a sense of smell as the former would
        # bypass walls and trick pacman
        if food:
            if not PacmanSmell(pacman_pos, layout, food, 5, 100, heatmap):
                # If there is no more food left in the smell radius,
                # we enable a long-range search, similar to the original factor
                # but amping up the effect.
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos)
                                        for food_pos in food)
                score += 150 / (min_food_distance + 1)

        # Factor 2: Proximidad a fantasmas
        # Replaced by a heat footprint that follows the ghosts and better
        # represent the dangerous level relative to pacman using a bfs strategy
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, ir a por él
                score += 50 / (ghost_distance + 1)

            # We activate it moments before ghost activation to avoid
            # pacman running into them when they are going to return back
            elif ghost_state.scaredTimer > 37:
                updateGhostHeatMap(ghost_pos, layout, 3, 1000, heatmap)

            else:
                # Si no está asustado, evitarlo
                updateGhostHeatMap(ghost_pos, layout, 3, 1000, heatmap)

        # HeatMap
        # We implement the heatmap, containing both the amplified versions,
        # onto the score
        x, y = int(pacman_pos[0]), int(pacman_pos[1])
        score += heatmap[x][y]

        heatmap[x, y] = 0
        for x in range(width):
            for y in range(height):
                if heatmap[x, y] == -99999:
                    heatmap[x, y] = 9
        print(heatmap, end=f"\n==============={self.frame}==================\n\n")
        self.frame += 1

        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                #############################################################
                # Daniel y Crespo
                #############################################################
                if action == Directions.STOP:
                    score -= probabilities[i] * 200

                neural_score += probabilities[i] * 100

        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]


# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)

############################################################################
# Daniel y Crespo
############################################################################


def collectIntersections(layout: Grid) -> list[tuple[int, int]]:
    """
    Returns a list with the position of all the intersections in the map
    An intersection being a path with 3 or more movement possibilities
    """
    height = layout.height
    width = layout.width

    intersections = []

    for y in range(height):
        for x in range(width):
            # If there is a wall, skip
            if layout[x][y]:
                continue

            neighbors = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = x + dx
                ny = y + dy
                # The layout is cornered by walls, so this function is never
                # going to exceed the limits
                if not layout[nx][ny]:
                    neighbors += 1

            if neighbors >= 3:
                intersections.append((x, y))
    return intersections


def initialCage(ghost_positions, heatmap, layout):
    """
    Locates the initial cage or home of the ghost. Every classical pacman
    map has a box or cage where the ghosts start and does not contain food
    and generally is a death sentence.
    """
    CAGE_POINTS = -99999
    queue = Queue()
    visited = set()

    # We locate ghost position
    for x, y in ghost_positions:
        x, y = int(x), int(y)

        queue.push((x, y))
        visited.add((x, y))
        heatmap[x][y] = CAGE_POINTS

    while not queue.isEmpty():
        x, y = queue.pop()

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy

            if layout[nx][ny]:
                continue

            if (nx, ny) in visited:
                continue

            is_exit = False

            for Dx, Dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                Nx = nx + Dx
                Ny = ny + Dy

                # If the neighbour of nx, ny is touching the exterior,
                # we do not expand further
                if heatmap[Nx][Ny] > 1 and heatmap[Nx][Ny] > CAGE_POINTS:
                    is_exit = True
                    break

            visited.add((nx, ny))

            # No expanding or scoring outside house
            if not is_exit:
                heatmap[nx][ny] = CAGE_POINTS
                queue.push((nx, ny))

    return heatmap


def createHeatMap(layout: Grid, ghost_position: list[tuple]):
    """
    Creates a heatmap with the level of danger of the tiles, this level being
    calculated by how far the tile is from an intersection. The intersections
    count as safe places because you can always scape from ghosts.
    """
    width = layout.width
    height = layout.height
    intersections = collectIntersections(layout)

    # We implement a BFS from every intersection
    heatmap = np.full((width, height), -1)
    queue = Queue()

    for x, y in intersections:
        heatmap[x][y] = 1  # 1 == safe
        queue.push((x, y, 1))  # x, y, distanceToIntersection

    while not queue.isEmpty():
        x, y, dist = queue.pop()
        ndist = dist + 1

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy

            if not layout[nx][ny] and heatmap[nx][ny] == -1:
                heatmap[nx][ny] = ndist
                queue.push((nx, ny, ndist))

    heatmap = initialCage(ghost_position, heatmap, layout)

    return heatmap


def circle(cx: int, cy: int, radio: int) -> list[tuple]:
    """
    Collects the points that form a circle with a certain radius and center
    """
    points = []
    r2 = radio**2

    for x in range(cx - radio, cx + radio + 1):
        for y in range(cy - radio, cy + radio + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                points.append((x, y))

    return points


def bfsSmell(layout, max_dist, food, px, py) -> list[tuple]:
    """
    Returns the shortest path between a piece of food and pacman's position
    """

    fx, fy = int(food[0]), int(food[1])
    queue = Queue()
    queue.push((fx, fy, 0, [(fx, fy, 0)]))
    visited = {(fx, fy)}
    
    while not queue.isEmpty():
        x, y, d, p = queue.pop()
        nd = d + 1

        if nd > 2*max_dist:
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy

            if not layout[nx][ny] and (nx, ny) not in visited:
                if (nx, ny) == (px, py):
                    return p + [(nx, ny, nd)]
                queue.push((nx, ny, nd, p + [(nx, ny, nd)]))
            visited.add((nx, ny))

    return []


def PacmanSmell(p_pos: tuple, layout: Grid, food_pos: list,
                max_dist: int, fruit_points: int, smell_hm):

    """
    Updates the heatmap to incorporate the smell of the closest food using
    a bfs strategy for every food in the circle and then selecting the best
    score for each cell

    This hard-sets the heatmap to the score, so this must be the first
    heuristic to apply
    """
    px, py = int(p_pos[0]), int(p_pos[1])

    food_dict = {}

    smell_area = circle(px, py, max_dist)
    for p in smell_area:
        if p in food_pos:
            bfs_path = bfsSmell(layout, max_dist, p, px, py)
            length = len(bfs_path)
            for x, y, d in bfs_path:
                points = fruit_points * (length - d) / length
                food_dict[(x, y)] = (points if (x, y) not in food_dict
                                     else max(points, food_dict[(x, y)]))

    for pos, points in food_dict.items():
        smell_hm[pos] = points
    
    return True if food_dict else False


def updateGhostHeatMap(g_pos: tuple, layout: Grid, max_dist: int,
                       scare_points: int, ghost_hm):

    """
    Updates the heatmap to add the heat footprint of the ghosts. This footprint
    is just a bfs starting at the ghost position.

    This is a soft addition to the heatmap, so it can be done at any point.
    """
    gx, gy = int(g_pos[0]), int(g_pos[1])

    queue = Queue()

    queue.push((gx, gy, 0))
    ghost_hm[gx][gy] -= scare_points
    visited = {(gx, gy)}

    while not queue.isEmpty():
        x, y, d = queue.pop()

        nd = d + 1
        if nd >= max_dist:
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy

            if not layout[nx][ny] and (nx, ny) not in visited:
                center_distance = int(manhattanDistance((nx, ny), (gx, gy)))
                ghost_hm[nx][ny] -= scare_points / (center_distance + 1)
                queue.push((nx, ny, nd))
 
            visited.add((nx, ny))
