# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:06:32 2021

@author: etodorov
"""

class Clock(object):
    """
    Class used for measurements in a cell grid
    
    Typical usage:
    >>> c = Clock((0,0),(10,10),'{time:.2f},{speed:.2f}')
    >>> for cell in moves:
    >>>     if c.check(cell):
    >>>         c.tick(0.1)
    >>> print(c.report())
    """
    def __init__(self, top_left, bottom_right, cell_size, report_configs):
        '''
        Create a clock for a rectangular clocking area.

        :param top_left: ((int, int))
            Top left cell position defining the clocking area
        :param bottom_right: ((int, int))
            Bottom right cell position defining the clocking area
        :param cell_size: (float)
            Cell size (used to convert grid units to meters)
        :param report_congifs: (string)
            Format string configuring the report of the clock. The defined 
            parameters to report are:
                {time}      - time elapsed with the clock active [s]
                {entry}     - etnry point of the clock (cell Ids)
                {exit}      - exit point of the clock  (cells Ids)
                {distance}  - distance between the entry and exit points [m]
                {speed}     - average speed between entry and exit points [m/s]
                {finished}  - did the clock stop*
        *if a measurement is not finished (e.g. the target is in the measured
        area or the pedestiran is stuck), the measurements will not be reliable
        '''
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.report_configs = report_configs
        self.cell_size = cell_size
        self.timer = 0
        self.started = False
        self.finished = False
        self.entry = None
        self.exit = None
        self.last_checked_cell = None
    
    def check(self, cell):
        '''
        Check if a cell belongs to a clocking area and handle starting/stopping
        the clock.

        :param cell: ((int, int))
            Top left cell position defining the clocking area
        :return: (bool)
            True if the cell is in the clocking area, False otherwise
        '''
        self.last_checked_cell = cell
        clocking =  cell[0] >= self.top_left[0] and \
                    cell[1] >= self.top_left[1] and \
                    cell[0] <= self.bottom_right[0] and \
                    cell[1] <= self.bottom_right[1]
        if not self.started and clocking:
            self.started = True
            self.entry = cell
        if self.started and not self.finished and not clocking:
            self.finished = True
            self.exit = cell
        if self.started and self.finished and clocking:
            #restart timer
            self.time = 0
            self.finished = False
            self.entry = cell
            self.exit = None
        return clocking
    
    def tick(self, dt):
        '''
        Increment the clock. (User needs to check for the clokcing area first
        using Clock.check(cell))
        
        :param dt: (float)
            Time increment
        '''
        self.timer += dt
    
    def report(self):
        '''
        Report the clock measurements
        
        :return: (string)
            Clock measurement report based on report_configs in the consturctor
        '''
        exit_point = self.exit if self.exit is not None \
                               else self.last_checked_cell
        distance = ((exit_point[0]-self.entry[0])**2 + \
                    (exit_point[1]-self.entry[1])**2)**0.5*self.cell_size
        try:
            speed = distance/self.timer
        except ZeroDivisionError:
            speed = 0
        return self.report_configs.format(speed=speed,
                                          distance = distance,
                                          time = self.timer,
                                          entry = self.entry,
                                          exit = exit_point,
                                          finished = self.finished)
    
# Test
if __name__ == "__main__":
    c = Clock((1,1),(10,10),1,'{time:.2f},{speed:.2f},{entry},{exit},{distance}')
    moves = [(i+2,i) for i in range(20)]
    for cell in moves:
        if c.check(cell):
            c.tick(0.2)
    print(c.report())
            