import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
import operator

class PedestrianGrid():

    def __init__(self, n, m, pedestrians, targets, obstacles = None):
        self.size = n, m
        self.pedestrians = pedestrians
        self.targets = targets
        self.obstacles = obstacles
        self.grid = self.__generate_grid()


    def __generate_grid(self):
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
        if len(moving_instructions) is not len(self.pedestrians):
            print(f'Error in length of the moving instructions, which has to match number of pedestrians. Length of list is {len(moving_instructions)}, expected {len(self.pedestrians)}')

        else:
            for i in range(len(moving_instructions)):
                move = moving_instructions[i]
                pedestrian = self.pedestrians[i]

                self.grid[pedestrian] = 0
                pedestrian = tuple(map(operator.add, move, pedestrian))
                self.pedestrians[i] = pedestrian
                self.grid[pedestrian] = 1


    def plot_grid(self):
        colormap = colors.ListedColormap(['white', 'green', 'black', 'lightblue'])
        plt.pcolor(self.grid, cmap=colormap)
        plt.yticks(size = self.size[0], fontsize = 'medium')
        plt.xticks(size = self.size[1], fontsize = 'medium')
        plt.show() 

    def simulation(self):
        pass

    def dijkstra(self):
        pass
