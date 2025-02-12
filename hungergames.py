# Pranav Minasandra
# pminasandra.github.io
# 11 Feb 2025

import glob
from os.path import join as joinpath
import uuid

import numpy as np

import config
import measurements
import selfishherd
import utilities

def hungergame(init_locs, num_smart):
    """
    Sets up and runs an individual contest, starting a smart selfish herd
    of n individuals, of which the first num_smart are d_1 and the rest
    are d_0.

    Args:
        init_locs (n×2 array-like): initial locations of agents
        num_smart (int): how many d1 individuals 
    """

    num_inds = init_locs.shape[0]
    depths = np.zeros(num_inds).astype(int)
    depths[:num_smart] += 1

    herd = selfishherd.SelfishHerd(num_inds, depths, init_locs)
    uname = str(uuid.uuid4())
    fname = joinpath(config.DATA, "HungerGames",
                f"{num_inds}-n{num_smart}-{uname}.pkl")

    return herd, fname


def hungergames(popsize, num_smart, num_instances):
    """
    *GENERATOR*
    Wrapper around hungergame(...)
    Args:
        popsize (int): population size
        num_smart (int): number of d_1 inds
        num_instances (int): how many simulations are needed
    """

    for i in range(num_instances):
        init_locs = np.random.uniform(size=(popsize, 2))
        herd, fname = hungergame(init_locs, num_smart)

        yield herd, fname


def _hungergames_files_for(popsize, num_smart):
    datadir = joinpath(config.DATA, "HungerGames")
    fformat = f"{popsize}-n{num_smart}-*.pkl"

    files = glob.glob(joinpath(datadir, fformat))

    return list(files)

def _read_hungergames_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# following analyses will be done after initial randomness
# let's say we will look at t=150 to t=250

def extract_groupsizes(dataset, num_smart):
    """
    For a given dataset containing d_0 and d_1 individuals,  returns TGS
    fo d_0 and d_1 individuals separately.

    Args:
        dataset (array-like, n×2×t): location data across time.
        num_smart (int): starting from index 0, how many d_1 individuals.

    Returns:
        np.array: (tgs_d_0, tgs_d_1)
    """
    dataset = dataset.copy()[:,:,config.HUNGERGAMES_TIME_LIMS[0]:
                            config.HUNGERGAMES_TIME_LIMS[1]]


    ttotal = dataset.shape[2]
    values_across_time = []
    for t in range(0, ttotal, 20):
        data_sub = dataset[:,:,t]
        group_ids = measurements.dbscan(data_sub)

        group_sizes_per_ind = np.zeros(group_ids.shape)
        group_labels, group_sizes = np.unique(group_ids, return_counts=True)
        
        for group in group_ids.unique():
            index = group_labels == group #mask
            assert sum(index) == 1 # can only be one with this label
            size = group_sizes[index][0]

            group_sizes_per_ind[group_ids == group] = size

        mean_d0 = group_sizes_per_ind[num_smart:].mean()
        mean_d1 = group_sizes_per_ind[:num_smart].mean()

        values_across_time.append([mean_d0, mean_d1])

    values_across_time = np.array(values_across_time)

    return values_across_time.mean(axis=0)
