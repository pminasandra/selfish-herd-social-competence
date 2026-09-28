# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides class SelfishHerd, a parallelisable way to instantiate and run
simulations.
"""

import pickle

import numpy as np

import movement
import voronoi

def _unique_elements(list_):
    """
    Returns a list of unique elements in a list.
    """
    s = set(list_)
    return s


class SelfishHerd:
    """
    Args:
        n (int): number of selfish agents
        depth_of_reasoning (int or array-like): how deep they should anticipate others'
                    behaviours.
        init_locs (np.array, n*2): initial_locations of agents.
    """

    def __init__(self,
                    n,
                    depth_of_reasoning,
                    init_locs):

        self.n = n
        self.depth = depth_of_reasoning
        self.init_locs = init_locs.copy()
        self.records = init_locs.copy()
        self.records = self.records[:,:,np.newaxis] #make 3d array


    def run(self, t):
        """
        Run the model for time t.
        Args:
            t (int): how many iterations to update the model.
        """

        for timestep in range(t):
            locs = self.records.copy()[:,:,-1]
            vor = voronoi.get_bounded_voronoi(locs)

            if isinstance(self.depth, int):
                if self.depth >= 0:#cognitive, recursive anticipation
                    next_locs = movement.recursive_reasoning(locs, vor, self.depth,
                                                                locs)
                else:#momentum based, conventional anticipation
                    if timestep == 0:
                        next_locs = movement.momentum_based_anticipatory_reasoning(locs,
                                        None)
                    else:
                        prev_locs = self.records.copy()[:,:,-2]
                        next_locs = movement.momentum_based_anticipatory_reasoning(locs,
                                        prev_locs)
            else:#is an array-like
                if _unique_elements(self.depth) == set([0, 1]):
                    next_locs = movement.recursive_reasoning(locs, vor, self.depth,
                                                                locs)
                elif _unique_elements(self.depth) == set([0, config.MU]):

                else:#not supported.
                    raise ValueError(f"Herd specified depths for members:\
{_unique_elements(self.depth)}. This combination is not currently\
supported.")

            self.records = np.dstack((self.records, next_locs))


    def savedata(self, filename):
        """
        Pickle-dumps the records to given filename
        """

        with open(filename, "wb") as file_obj:
            pickle.dump(self.records, file_obj)


    def __str__(self):
        return f"SelfishHerd object with {self.n} individuals and {self.records.shape[2]} rows of data."

    def __repr__(self):
        return self.__str__()
