import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import itertools
import operator

class PedestrianGrid():
    '''
    Main class for simulating the crowd
    '''

    def __init__(self, n, m, pedestrians, targets, obstacles = None):
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
        
        self.size = n, m
        self.pedestrians = pedestrians
        self.targets = targets
        self.obstacles = obstacles
        self.grid = self.__generate_grid()

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

    def move_pedestrians(self, moving_instructions):
        ''' 
        Moves the pedestrians in accordance with the moving_instructions.

        :param moving_instructions: (list) List of tuples (int, int) containing how much the corresponding 
                                    pedestrian is to move. 
                                    Example: moving_instructions[0] = (1,1) would move pedestrians[0]
                                    one cell in each direction.
        '''

        if len(moving_instructions) is not len(self.pedestrians):
            print(f'Error in length of the moving instructions, which has to match number of pedestrians. Length of list is {len(moving_instructions)}, expected {len(self.pedestrians)}')

        else:
            for i in range(len(moving_instructions)):
                move = moving_instructions[i]
                pedestrian = self.pedestrians[i]

                self.grid[pedestrian] = 0
                new_pedestrian = tuple(map(operator.add, move, pedestrian))
                if new_pedestrian[0] < self.size[0] and new_pedestrian[1] < self.size[1]:
                    self.pedestrians[i] = new_pedestrian
                    self.grid[new_pedestrian] = 1
                else:
                    self.grid[pedestrian] = 1
                    print(f'Can\'t move pedestrian {i} outside the grid, therefore kept at original position')

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

    def simulation(self):
        pass

    def dijkstra(self):
        pass
