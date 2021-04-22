# -*- coding: utf-8 -*-
"""
User interface and graphic interaction

Created on Sun Apr 18 09:59:46 2021

@author: etodorov
"""
import os, io, glob
from datetime import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter
from pedestrian_grid import PedestrianGrid

#constants
INIT_CONDITIONS_PATTERN = "./ICs/*.txt"
RESULT_DIR = "./results"

class UI(object):
    """
    Main User Interface object
    """
    def __init__(self, max_frames =  100):
        """
        Initialize a default User interface object

        :param max_frames: (int)
            Maximum frames in the simulation
        """
        self.animation = {}
        self.max_frames = max_frames
        self.animation["frames_per_second"] = 10
        self.animation["time_steps_per_frame"] = 1
        self.animation["save"] = True
        self.animation["result_dir"] = RESULT_DIR
        
        self.simulation = {}
        self.simulation["cell_size"] = 1.0 #m
        self.simulation["time_step"] = 1.0 #s
        self.simulation["duration"] = float(max_frames)
        self.simulation["population"] = "clones"
        self.simulation["seed"] = 42
        self.simulation["algorithm"] = "basic"
        
    
    def start_CLI(self):
        '''
        Start the command-line interface to choose file from a folder
        '''
        self.intro()
        gridarray = self.get_init_condition()
        self.grid = PedestrianGrid(gridarray, configs=self.simulation)
        self.animate(self.animation['frames_per_second'])
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
        Establish configuration and initial condition from a choice of files
        
        :param ic_pattern: (string)
            (optional) - Glob pattern of initial condition files
        
        :return gridarray: (np.array)
            Initial condition for a PedestrianGrid
        :return options: (dict)
            Configuration options from file
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
            except (IOError, ValueError, yaml.YAMLError) as e:
                print("Error reading the file:\n\t", str(e))
                raise e
                return self.get_init_condition(ic_pattern)
            return gridarray
    
    def file2grid(self, fname):
        """
        Convert a file to a PedestrianGrid, checking the requirements 
        
        :param fname: (string)
            file path of the file with initial conditions
        
        :return: (np.array)
            Initial condition for a PedestrianGrid
        """
        with open(fname, 'r') as stream:
            config = yaml.safe_load(stream)
        initial_condition = self.parse_config(config)
        gridarray = np.genfromtxt(initial_condition, dtype=int, delimiter=1)
        if (gridarray==1).sum() < 1:
            raise ValueError("No pedestrians found in " + fname)
        if (gridarray==2).sum() != 1:
            raise ValueError("Exactly one target is required in " + fname)
        if gridarray.min() < 0 or gridarray.max() > 3:
            raise ValueError("Only values 0, 1, 2, 3 are allowed.")
        return gridarray
    
    def parse_config(self, config):
        """
        Apply configuration parameters from file
        
        :param config: (dict)
            Dictionary of parameters. If any are missing, defaults are used.
        
        :return: (StringIO)
            Initial conditions string to be further parsed
        """
        animold = self.animation
        try:
            animnew = config["animation"]
        except (KeyError, TypeError):
            animnew = {}
            print("No animation found - defaults will be used")
        for key, val in animold.items():
            try:#basic presence and type check
                if animnew[key] is None:
                    raise KeyError;
                animold[key] = type(animold[key])(animnew[key])
            except KeyError:
                print(f"No animation.{key} found - default {animold[key]}")
            except (ValueError, TypeError):
                print(f"Error with animation.{key} - default {animold[key]}")
        
        simold = self.simulation
        try:
            simnew = config["simulation"]
        except (KeyError, TypeError):
            raise ValueError("No simulation.initial_condition found!")
            
        for key, val in simold.items():
            try:#basic presence and type check
                if simnew[key] is None:
                    raise KeyError;
                simold[key] = type(simold[key])(simnew[key])
            except KeyError:
                print(f"No simulation.{key} found - default {simold[key]}")
            except (ValueError, TypeError):
                print(f"Error with simulation.{key} - default {simold[key]}")
            
        try: 
            initial_condition = simnew["initial_condition"]
        except KeyError:
            raise ValueError("The file must contain initial condition!")
        
        return io.StringIO(initial_condition)
    
    def plot(self, grid):
        """
        Plot the underlying object of a pedestrian grid

        :param grid: (np.array)
            PedestrianGrid.grid array to be plotted

        :return (fig, ax, im): (tuple)
            Plotted figure, axis and image objects
        """
        fig = plt.figure(figsize=(self.grid.size[0]/10+2,
                                  self.grid.size[1]/10+1))
        ax = plt.gca()
        if np.max(grid) == 3:
            self.colormap = colors.ListedColormap(['white','green', 'yellow', 'red'])
        else:
            self.colormap = colors.ListedColormap(['white','green', 'yellow'])
        
        ax.time_text = ax.text(0.05, 1.01, '', 
                                         transform=ax.transAxes)
        litems = [Patch(facecolor=c, edgecolor='black') 
                      for c in self.colormap.colors]
        plt.legend(handles = litems, 
                   labels = ["Empty",
                             "Pedestrian",
                             "Target",
                             "Obstacle"][:len(self.colormap.colors)], 
                   loc="upper left", 
                   bbox_to_anchor=[1.02, 1])

        im = plt.imshow(grid, cmap=self.colormap)

        plt.tight_layout()
        return fig, ax, im
    
    def animate(self, fps=10):
        """
        Create an animatied simulation and save it to .gif
        
        :param fps:
            Frames per second of animation
        """
        self.fig, self.ax, self.im = self.plot(self.grid.grid)
        ani = FuncAnimation(self.fig, 
                            self.draw,
                            fargs = (self.animation['time_steps_per_frame'],),
                            frames=self.max_frames)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if self.animation["save"]:
            ani.save(os.path.join(self.animation["result_dir"],
                                  f"{self.icname}_{timestamp}.gif"),
                     writer = PillowWriter(fps=fps))  
        else:
            plt.show()                           
        
    def draw(self,i, tpf=1):
        """
        Draw the next frame of the animation
        
        :param i: (int)
            Frame number
        :param tpf: (int)
            Frames to run before drawing
            
        :return im: (plt.Image)
        :return ax.time_text: (plt.Text)
        """
        if tpf==1:
            self.grid.evolve()
        else:
            self.grid.simulate(tpf)
            
        self.im.set_data(self.grid.grid)
        dt = self.simulation['time_step']
        self.ax.time_text.set_text(f"frame = {i}; t = {i*tpf*dt} s" )
        return self.im, self.ax.time_text
    
    def exit(self):
        """ 
        Terminate the program
        """
        raise SystemExit()
        
if __name__ == "__main__":
    myUI = UI(60)
    myUI.start_CLI()