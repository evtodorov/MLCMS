# -*- coding: utf-8 -*-
"""
Grid functionality

Created on Sun Apr 18 10:05:34 2021

@author: 
"""

class Grid(object):
    """
    Main class for the simulation grid
    """
    
    def __init__(self, grid):
        """
        Initialize a grid from a numpy array

        Parameters
        ----------
        grid : numpy.array dtype=int
            A 2D numpy array containings:
                0 for a blank cell
                1 for a pedestrian cells
                2 for the target cell
                3 for an obstacle cell
        Returns
        -------
        Grid object

        """
        self.grid = grid
        self.size = grid.shape
        self.location = [10,0]
    def update(self):
        #temporary for testing
        self.grid[tuple(self.location)] = 0
        self.location[1] += 1
        self.grid[tuple(self.location)] = 1