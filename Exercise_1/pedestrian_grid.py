import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import itertools
import operator


NEIGHBOURS = [(-1,-1),(-1,0),(-1,1),
             (0,-1),        (0,1),
             (1,-1), (1,0), (1,1)]
NEIGHBOURS = [(-1,0),(0,-1),(0,1),(1,0)]
NEIGHBOURS = {i: np.array(n) for i, n in enumerate(NEIGHBOURS)}

class PedestrianGrid():
    '''
    Main class for simulating the crowd
    '''

    def __init__(self, grid = None,
                       n=None,
                       m=None, 
                       pedestrians= None, 
                       target = None, 
                       obstacles = None):
        '''
        Initializes the grid on which to simulate the crowd movements as a numpy array with
        free space (coded as 0), pedestrians (coded as 1), targets (coded as 2) and
        obstacles (coded as 3).


        :param n: (int) Width of the grid
        :param m: (int) Height of the grid
        :param pedestrians: (list) List of tuples (int, int) with coordinates for the pedestrians
        :param targets: (list) List of tuples (int, int) with coordinates for the targets
        :param obstacles: (list) List of tuples (int, int) with coordinates for the obstacles
        '''
        
        if grid is not None:
            self.grid = grid
            self.size = grid.shape
            self.pedestrians = list(np.argwhere(grid==1))
            obstacles = tuple(np.argwhere(grid==3))
            self.obstacles = obstacles if len(obstacles) else None
            target = np.argwhere(grid==2)
            if len(target) > 1: 
                raise ValueError("excatly one target required")
            else:
                self.target = target[0]
            
        else:
            try:
                self.size = n, m
                self.pedestrians = [np.array(p) for p in pedestrians]
                self.target = np.array(target)
                self.obstacles = [np.array(o) for o in obstacles]
                self.grid = self.__generate_grid()
            except NameError as e:
                print("Wrong!")
                raise e
        self.target_neighbours = [tuple((self.target + n)) for n in NEIGHBOURS.values()]

    def __generate_grid(self):
        '''
        Initialize the grid and mark the cells accordingly
        '''

        tmp_grid = np.zeros((self.size))

        for pedestrian in self.pedestrians:
            tmp_grid[pedestrian] = 1 

        for target in self.targets:
            tmp_grid[target] = 2
        
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                tmp_grid[obstacle] = 3
            
        return tmp_grid

    def __check_obstacle(self, pedestrian):
        ''' 
        Checks whether a pedestrian stands on an obstacle or not. Returns True if true and False if false.

        :param pedestrian: (tuple) Coordinates for a pedestrian (int, int).
        '''
        if self.obstacles is None:
            return False

        else:
            for obstacle in self.obstacles:
                if obstacle[0] == pedestrian[0] and obstacle[1] == pedestrian[1]:
                    return True

        return False
        
    def move_pedestrians(self, pix, move):
        ''' 
        Moves the pedestrians in accordance with the moving_instructions.

        :param moving_instructions: (list) List of tuples (int, int) containing how much the corresponding 
                                    pedestrian is to move. 
                                    Example: moving_instructions[0] = (1,1) would move pedestrians[0]
                                    one cell in each direction.
        '''

        pedestrian = self.pedestrians[pix]

        self.grid[tuple(pedestrian)] = 0
        new_pedestrian = pedestrian + move
        if new_pedestrian[0] >= self.size[0] and new_pedestrian[1] >= self.size[1]:
            self.grid[tuple(pedestrian)] = 1
            print(f'Can\'t move pedestrian {pix} outside the grid or on an obstacle, therefore kept at original position')
        else:
            self.pedestrians[pix] = new_pedestrian
            self.grid[tuple(new_pedestrian)] = 1
            

    def plot_grid(self):
        '''
        Plots the grid
        '''

        if self.grid.max() == 3:
            colors = ['white', 'red', 'lightblue', 'black']
            label = ['Free space', 'Pedestrian', 'Target', 'Obstacle']
        else:
            colors = ['white', 'red', 'lightblue']
            label = ['Free space', 'Pedestrian', 'Target']

        plt.pcolormesh(self.grid, cmap=LinearSegmentedColormap.from_list('', colors))
        legend_elements = [Patch(facecolor=color, edgecolor='black') for color in colors]
        plt.legend(handles = legend_elements, labels = label, loc="upper left", bbox_to_anchor=[1.02, 1])

        plt.yticks(size = self.size[0], fontsize = 'medium')
        plt.xticks(size = self.size[1], fontsize = 'medium')
        plt.show() 

    def evolve(self, method='basic'):


        a = 2
        for pix, pedestrian in enumerate(self.pedestrians):
            if tuple(pedestrian) in self.target_neighbours:
                continue
            neighbours_cost = self.cost_function(pedestrian, method)
            goto_neigbour = NEIGHBOURS[min(neighbours_cost.keys(), key=neighbours_cost.__getitem__)]
            move = goto_neigbour - pedestrian
            move = goto_neigbour
            self.move_pedestrians(pix, move)
    
    def cost_function(self, pedestrian, method="basic"):
        
        if method=='basic':
            method = self.basic_cost
        elif method=="dijkstra":
            method = self.dijkstra   

        neighbours_cost = {}
        for i, neighbour in NEIGHBOURS.items():
            neighbours_cost[i] = method(pedestrian, neighbour)
        
        return neighbours_cost
    
    def basic_cost(self, pedestrian, neighbour):
        return np.linalg.norm(self.target - 
                              (pedestrian + neighbour))
           
        
    def dijkstra(self):
        pass

    def simulate(self, max_steps = 100):
        for i in range(max_steps):
            self.evolve()
