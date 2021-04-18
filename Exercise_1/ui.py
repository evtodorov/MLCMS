# -*- coding: utf-8 -*-
"""
User interface and graphic interaction

Created on Sun Apr 18 09:59:46 2021

@author: etodorov
"""
import os, glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from grid import Grid

#constants
INIT_CONDITIONS_PATTERN = "./ICs/*.txt"
RESULT_DIR = "./results"

class UI(object):
    """
    Main User Interface object
    """
    def __init__(self):
        """
        Initialize a User interface object

        Parameters
        ----------
        None.
        
        Returns
        -------
        UI object

        """
        self.intro()
        gridarray = self.get_init_condition()
        self.grid = Grid(gridarray)
        self.show()
        self.exit()
        
    def intro(self):
        """
        Print introduction to the command line interface
        """
        print("Crowd simulation based on cellular automata\n\n")
        print("\tThe domain is a rectangular grid to be provided by the user\n"
              "matching the following definitions:\n"
              "\t\t0\tEmpty cell\n"
              "\t\t1\tPedestrian\n"
              "\t\t2\tTarget (one per file)\n"
              "\t\t3\tObstacle")

    def get_init_condition(self, ic_pattern=INIT_CONDITIONS_PATTERN):
        """
        Establish the initial condition from a choice of files
        
        Parameters
        ----------
        ic_pattern (optional) - Glob pattern of initial condition files
        
        Returns
        -------
        Numpy array matching the requirements for a grid
        """
        print("Choose initial conditions:\n"
              "The following initial conditions match ", ic_pattern,
              ":\nPick one of the options below:\n"
              "\n"
              "\t0\tChange the directory")
        filelist = glob.glob(ic_pattern)
        for fn, fname in enumerate(filelist):
            print(f"\t{fn+1}\t{fname}")
        print('\tQ\tTo exit the program')
        
        loop = True
        while loop:
            n = input("Number of your choice followed by Enter:\n")
            if n.lower()=="q":
                self.exit()
            try:
                n = int(n)
            except ValueError:
                n = -1
            if n >= 0 and n <= len(filelist):
                loop = False
            else:
                print("Invalid choice. Please choose again.")
        
        if n==0:
            new_pattern = input("Input the new IC path glob pattern:\n")
            return self.get_init_condition(new_pattern)
        else:
            try:
                self.icname, _ = os.path.splitext(
                                    os.path.basename(filelist[n-1]))
                gridarray = self.file2grid(filelist[n-1])
            except ValueError as e:
                print("Error reading the file:\n\t", str(e))
                return self.get_init_condition(ic_pattern)
            return gridarray
    
    def file2grid(self, fname):
        """
        Convert a file to a grid, checking the requirements
        
        Parameters
        ----------
        fname - file path of the file with initial conditions
        
        Returns
        -------
        Numpy array matching the requirements for a grid
        """
        gridarray = np.genfromtxt(fname, dtype=int, delimiter=1)
        if (gridarray==1).sum() < 1:
            raise ValueError("No pedestrians found in " + fname)
        if (gridarray==2).sum() != 1:
            raise ValueError("Exactly one target is required in " + fname)
        if gridarray.min() < 0 or gridarray.max() > 3:
            raise ValueError("Only values 0, 1, 2, 3 are allowed.")
        return gridarray
        
    def show(self):
        """
        Show the animatied simulation
        """
        self.fig = plt.figure()
        self.ax = plt.gca()
        if self.grid.grid.max() == 3:
            self.colormap = colors.ListedColormap(['white','green', 'yellow', 'red'])
        else:
            self.colormap = colors.ListedColormap(['white','green', 'yellow'])
        
        self.ax.set_xlim((0,self.grid.size[0]))
        self.ax.set_ylim((0,self.grid.size[1]))
        
        self.im = plt.imshow(self.grid.grid, cmap=self.colormap)
        ani = FuncAnimation(self.fig, self.draw)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ani.save(os.path.join(RESULT_DIR, f"{self.icname}_{timestamp}.gif"),
                 writer = PillowWriter(fps=10))                             
        plt.show()
    
    def draw(self,i):
        """
        Draw the next frame of the animation
        """
        self.grid.update()
        self.im.set_data(self.grid.grid)
        return self.im,
    
    def exit(self):
        """ 
        Terminate the program
        """
        raise SystemExit()
        
if __name__ == "__main__":
    myUI = UI()