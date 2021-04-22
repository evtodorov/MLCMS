import numpy as np

#Constants
NEIGHBOURS = [(-1,-1),(-1, 0),(-1, 1),
              ( 0,-1),        ( 0, 1),
              ( 1,-1),( 1, 0),( 1, 1)]

NEIGHBOURS = {i: n for i, n in enumerate(NEIGHBOURS)}

class PedestrianGrid():
    '''
    Main class for simulating the crowd
    '''

    def __init__(self, grid = None,
                       n = None,
                       m = None, 
                       pedestrians = None, 
                       target = None, 
                       obstacles = None):
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
        
        Usage 1:
        pg = PedestrianGrid(grid=grid)
        Usage 2:
        pg = PedestrianGrid(n=50, m=50, 
                            pedestrians=pedestrians,
                            target=target,
                            obstacles = obstacles)
        '''
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
                self.grid = self.__generate_grid()
            except NameError as e:
                print("Not enough input parameters for PedestrianGrid.")
                raise e
        self.target_neighbours = [self.target + n 
                                  for n in NEIGHBOURS.values()]
        self.speeds = np.ones(len(self.pedestrians)) #TODO: set speed != 1
        self.time_credits = np.zeros(len(self.pedestrians))

    def __generate_grid(self):
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

    def __check_obstacle(self, pedestrian):
        ''' 
        Checks whether a pedestrian stands on an obstacle or not. 
        Returns True if true and False if false.

        :param pedestrian: (tuple) 
            Coordinates for a pedestrian (int, int).
            
        :return: (bool)
        '''
        if self.obstacles is None:
            return False

        else:
            for obstacle in self.obstacles:
                if obstacle[0] == pedestrian[0] and obstacle[1] == pedestrian[1]:
                    return True

        return False
        
    def move_pedestrian(self, pix, move):
        ''' 
        Move a pedestrian in accordance with the moving_instructions.
        
        :param pix: (int)
            Index of the pedestrian in self.pedestrians
        :param move: (np.array([int,int])) 
            Distance traveled from the pedestrian in one update
        '''
        # Only make a step if sufficient time credit is available [9] Sec. 3
        dtau = np.linalg.norm(move)/self.speeds[pix] # dtau = lambda / v
        if self.time_credits[pix] >= dtau:
            self.time_credits[pix] -= dtau
            pedestrian = self.pedestrians[pix]
            
            self.grid[pedestrian[0], pedestrian[1]] = 0
            new_pedestrian = pedestrian + move
            if new_pedestrian[0] >= self.size[0] and \
               new_pedestrian[1] >= self.size[1]:
                self.grid[pedestrian[0], pedestrian[1]] = 1
                print(f'Can\'t move pedestrian {pix} outside the grid or on an obstacle, therefore kept at original position')
            else:
                self.pedestrians[pix] = new_pedestrian
                self.grid[new_pedestrian[0], new_pedestrian[1]] = 1

    def evolve(self, method='basic'):
        """
        Advance the position of all pedestrians by one timestep
        
        :param method: (string) 
            Method to use to compute the evolution. Available methods:
                "basic" (default)   Compute Euclidean distance at neighbours
                "dijkstra"          Compute path based on Dijsktra's algorithm
        """
        for pix, pedestrian in enumerate(self.pedestrians):
            self.time_credits[pix] += 1
            #if tuple(pedestrian) in self.target_neighbours:
            for target_neighbour in self.target_neighbours:
                if tuple(pedestrian) != tuple(target_neighbour[0]):
                    continue
            neighbours_cost = self.cost_function(pedestrian, method)
            goto_neigbour = NEIGHBOURS[min(neighbours_cost.keys(), 
                                           key=neighbours_cost.__getitem__)]
            move = goto_neigbour - pedestrian
            move = goto_neigbour
            self.move_pedestrian(pix, move)
    
    def cost_function(self, pedestrian, method="basic"):
        """
        Compute the cost function for each neighbouring cell
        
        :param pedestrian: (np.array([int, int]))
            Index in the grid of the current pedestrian
        :param method: (string) 
            Method to use to compute the evolution. Available methods:
                "basic" (default)   Compute Euclidean distance at neighbours
                "dijkstra"          Compute path based on Dijsktra's algorithm
        
        :return: (dict)
        """
        if method=='basic':
            method = self.basic_cost
        elif method=="dijkstra":
            method = self.dijkstra   

        neighbours_cost = {}
        for i, neighbour in NEIGHBOURS.items():
            neighbours_cost[i] = method(pedestrian, neighbour)
        
        return neighbours_cost
    
    def basic_cost(self, pedestrian, neighbour):
        """
        Compute the basic cost based on Euclidean distance between a neigbour
        and the target
        
        :param pedestrian: (np.array([int, int]))
            Index in the grid of the current pedestrian
        :param neighbour: (np.array([int, int]))
            Relative position of the neighbour wrt to the pedestrian
            
        :return: (np.array([int, int]))
        """
        return np.linalg.norm(self.target - 
                              (pedestrian + neighbour))
           
        
    def dijkstra(self):
    #TODO:
        pass

    def simulate(self, max_steps = 100):
        """
        Execute multiple steps of the simulation
        
        :param max_steps: (int)
            Steps to execute
        """
        for i in range(max_steps):
            self.evolve()
