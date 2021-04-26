import numpy as np
from dijkstra import Dijkstra_path
from clock import Clock
# TODO:
#   - Implement time_step
#   - Implement cell_size
#   = Implement population speeds
#   - Implement the measurements
#Constants
NEIGHBOURS = [(-1,-1),(-1, 0),(-1, 1),
              ( 0,-1),( 0, 0),( 0, 1),
              ( 1,-1),( 1, 0),( 1, 1)]

NEIGHBOURS = {i: np.array(n) for i, n in enumerate(NEIGHBOURS)}

class PedestrianGrid():
    '''
    Main class for simulating the crowd
    '''

    def __init__(self, grid = None,
                       n = None,
                       m = None, 
                       pedestrians = None, 
                       target = None, 
                       obstacles = None,
                       configs = None):
        '''
        Initializes the grid on which to simulate the crowd movements as a 
        numpy array with:
            0   free space
            1   pedestrian
            2   target (should be only 1)
            3   obstacles.

        :param grid: (np.array) 
            Initialized grid
        :param n: (int) 
            Width of the grid
        :param m: (int) 
            Height of the grid
        :param pedestrians: (list) 
            List of tuples (int, int) with coordinates for the pedestrians
        :param targets: (list) 
            List of tuples (int, int) with coordinates for the targets
        :param obstacles: (list) 
            List of tuples (int, int) with coordinates for the obstacles
        :param configs: (dict)
            Dictionary of simulation parameters to overwrite the defaults
        
        Usage 1:
        pg = PedestrianGrid(grid=grid)
        Usage 2:
        pg = PedestrianGrid(n=50, m=50, 
                            pedestrians=pedestrians,
                            target=target,
                            obstacles = obstacles)
        '''
        from ui import UI
        mytempui = UI()
        for k,v in mytempui.simulation.items():
            self.__setattr__(k, v)
            
        if type(configs) is dict: 
            for k, v in configs.items():
                self.__setattr__(k, v)
        
        if grid is not None:
            self.grid = grid
            self.size = grid.shape
            self.pedestrians = list(np.argwhere(grid==1))

            obstacles = tuple(np.argwhere(grid==3))
            self.obstacles = obstacles if len(obstacles) else None
            target = np.argwhere(grid==2)
            if len(target) > 1: 
                raise ValueError("Excatly one target required.")
            else:
                self.target = target[0]
        else:
            try:
                self.size = n, m
                self.pedestrians = [np.array(p) for p in pedestrians]
                self.target = np.array(target)
                self.obstacles = [np.array(o) for o in obstacles] if obstacles is not None else None
                self.grid = self._generate_grid()
            except NameError as e:
                print("Not enough input parameters for PedestrianGrid.")
                raise e
        self.target_neighbours = [tuple((self.target + n)) 
                                  for n in NEIGHBOURS.values()]
        self.speeds = self.generate_speeds(self.population,
                                           self.seed)
        self.time_credits = np.zeros(len(self.pedestrians))
        self.planned_paths = [None]*len(self.pedestrians)
        self.moves_done = [0]*len(self.pedestrians)
        if self.clocks is not None:
            self.clock_list = [ [Clock(cell_size=self.cell_size, **clock) 
                                 for clock in self.clocks]
                            for i in range(len(self.pedestrians))]
        self.time = 0
    
    def _generate_grid(self):
        '''
        Initialize the grid and mark the cells accordingly
        
        :return: (np.array)
        '''
        tmp_grid = np.zeros((self.size))
        
        for pedestrian in self.pedestrians:
            tmp_grid[tuple(pedestrian)] = 1 

        for target in self.target:
            tmp_grid[tuple(target)] = 2
        
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                tmp_grid[tuple(obstacle)] = 3
            
        return tmp_grid
    
    def generate_speeds(self, population='clones', seed=42):
        ''' 
        Generate a distribution of walking speeds
        
        :param population: (string) {'clones', 'spread', 'similar'}
            Chose how spread-out the population will be
        :param seed: (int)
            Seed for the random number generator
        :return: (np.array)
            Walking speed per pedestrian
        '''
        # Mean free walking speed for men is 1,41 [m/s] and women 1,27 [m/s].
        # Rought estimation of the std for the plot is that it holds std = 0.25
        # Should end up with the same
        mean_speed = (1.41+1.27)/2
        np.random.seed(seed)
        if population == 'clones':
            return mean_speed*np.ones(len(self.pedestrians)) 
        elif population == 'similar':
            np.random.seed(seed)
            return np.random.normal(loc=mean_speed, scale=0.05, size=len(self.pedestrians))
        elif population == 'spread':
            np.random.seed(seed)
            return np.random.normal(loc=mean_speed, scale=0.25, size=len(self.pedestrians))
        else:
            print(f"Unknown population parameter {population}, 'clones' will "
                  "be used instead")
            return self.generate_speeds('clones', seed)
    
    def move_pedestrian(self, pix, move):
        ''' 
        Move a pedestrian in accordance with the moving_instructions.
        
        :param pix: (int)
            Index of the pedestrian in self.pedestrians
        :param move: (np.array([int,int])) 
            Distance traveled from the pedestrian in one update
        '''
        # Only make a step if sufficient time credit is available [9] Sec. 3
        # dtau = lambda / v
        dtau = np.linalg.norm(move*self.cell_size)/self.speeds[pix]
        pedestrian = self.pedestrians[pix]
        new_pedestrian = pedestrian + move
        if self.time_credits[pix] >= dtau and \
           self.grid[new_pedestrian[0], new_pedestrian[1]] == 0:
            self.time_credits[pix] -= dtau
            self.grid[pedestrian[0], pedestrian[1]] = 0
            
            if new_pedestrian[0] >= self.size[0] and \
               new_pedestrian[1] >= self.size[1]:
                self.grid[pedestrian[0], pedestrian[1]] = 1
                print(f'Can\'t move pedestrian {pix} outside the grid or on an obstacle, therefore kept at original position')
            else:
                self.pedestrians[pix] = new_pedestrian
                self.grid[new_pedestrian[0], new_pedestrian[1]] = 1
                self.moves_done[pix] += 1

    def evolve(self, method=None):
        """
        Advance the position of all pedestrians by one timestep
        
        :param method: (string) 
            Method to use to compute the evolution. Available methods:
                "basic" (default)   Compute Euclidean distance at neighbours
                "dijkstra"          Compute path based on Dijsktra's algorithm
        """
        self.time += self.time_step
        if method is None:
            method = self.algorithm
        for pix, pedestrian in enumerate(self.pedestrians):
            self.time_credits[pix] += self.time_step
            if tuple(pedestrian) in self.target_neighbours:
                continue
            
            if method=="basic":
                move = self.basic_cost(pedestrian)
            elif method=="dijkstra":
                move = self.dijkstra(pix)
            else:
                print("No such algorithm!")
                move = np.array((0,0))
            
            self.move_pedestrian(pix, move)
            
            if self.clocks is not None:
                for clock in self.clock_list[pix]:
                    if clock.check(self.pedestrians[pix], self.time):
                        clock.tick(self.time_step)
    
    def basic_cost(self, pedestrian):
        """
        Find the best move based on Euclidean distance between each neigbour
        and the target
        
        :param pedestrian: (np.array([int, int]))
            Index in the grid of the current pedestrian
        :param neighbour: (np.array([int, int]))
            Relative position of the neighbour wrt to the pedestrian
            
        :return: (np.array([int, int]))
            move relative to the current cell
        """
        huge_cost = 10e6
        
        
        neighbours_cost = {}
        for i, neighbour in NEIGHBOURS.items():
            possible_move = pedestrian + neighbour
            
            # get the cost as the Euclidean distance
            cost = np.linalg.norm(self.target - (possible_move))   
            # if the neighbour cell is not empty, you can't go there

            # If possible_move lies outisde the grid we get an error
            try:
                if self.grid[tuple(possible_move)] > 0 and \
                np.linalg.norm(neighbour) != 0:
                    cost += huge_cost
            except IndexError:
                continue


            neighbours_cost[i] = cost

        goto_neigbour = NEIGHBOURS[min(neighbours_cost.keys(), 
                                  key=neighbours_cost.__getitem__)]
        return goto_neigbour
           
        
    def dijkstra(self, pix):
        """
        Find the best move based on the precomputed path to the target
        
        :param pedestrian: (np.array([int, int]))
            Index in the grid of the current pedestrian
            
        :return: (np.array([int, int]))
            move relative t
        """
        if self.planned_paths[pix] is None:
        # Plan the path based on Dijsktra if it's not planned yet
            self.planned_paths[pix] = Dijkstra_path(self.grid, 
                                                    self.pedestrians[pix],
                                                    self.target)
            
        try:

            path = self.planned_paths[pix].path
            moves_along_the_path = self.moves_done[pix] + 1
            if self.grid[tuple(path[moves_along_the_path])] == 0:
                # if the cell is empty, move to it
                dijsktra_cell = np.array(path[moves_along_the_path])
                move = dijsktra_cell - self.pedestrians[pix] 
            elif self.grid[tuple(path[moves_along_the_path])] == 2:
                raise IndexError #move (0,0) = wait here
            else:
                # otherwise recalculate the path to take
                self.planned_paths[pix] = Dijkstra_path(self.grid, 
                                                    self.pedestrians[pix],
                                                    self.target)
                raise IndexError #move (0,0) = wait here
        except IndexError:
            #No more moves from Dijkstra
            move = np.array((0,0)) #move (0,0) = wait here
            
        return move

    def simulate(self, max_steps = 100):
        """
        Execute multiple steps of the simulation
        
        :param max_steps: (int)
            Steps to execute
        """
        for i in range(max_steps):
            self.evolve()
