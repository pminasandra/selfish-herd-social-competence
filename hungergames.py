# Pranav Minasandra
# pminasandra.github.io
# 11 Feb 2025

import glob
from os.path import join as joinpath
import pickle
import uuid

import matplotlib.pyplot as plt
import numpy as np

import config
import measurements
import selfishherd
import utilities
import voronoi

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

def extract_groupsizes(dataset, rel_indices):
    """
    For a given dataset containing focal and nonfocal individuals,  returns TGS
    for focal and nonfocal individuals separately.

    Args:
        dataset (array-like, n×2×t): location data across time.
        rel_indices (array-like): indices of focal individuals

    Returns:
        tuple of floats: (tgs_nonfocal, tgs_focal)
    """
    dataset = dataset.copy()[:,:,config.HUNGERGAMES_TIME_LIMS[0]:
                            config.HUNGERGAMES_TIME_LIMS[1]]


    ttotal = dataset.shape[2]
    values_across_time = []
    non_indices = list(range(dataset.shape[0]))
    non_indices = [j for j in non_indices if j not in rel_indices]
    for t in range(0, ttotal, 20):
        data_sub = dataset[:,:,t]
        group_ids = measurements.dbscan(data_sub)

        group_sizes_per_ind = np.zeros(group_ids.shape)
        group_labels, group_sizes = np.unique(group_ids, return_counts=True)
        
        for group in np.unique(group_ids):
            index = group_labels == group #mask
            assert sum(index) == 1 # can only be one with this label
            size = group_sizes[index][0]

            group_sizes_per_ind[group_ids == group] = size

        mean_d0 = group_sizes_per_ind[non_indices].mean()
        mean_d1 = group_sizes_per_ind[rel_indices].mean()

        values_across_time.append([mean_d0, mean_d1])

    values_across_time = np.array(values_across_time)

    results =  values_across_time.mean(axis=0)
    return results[0], results[1]

def extract_areas(dataset, num_smart):
    """
    For a given dataset containing d_0 and d_1 individuals,  returns mean Voronoi areas
    for d_0 and d_1 individuals separately.

    Args:
        dataset (array-like, n×2×t): location data across time.
        num_smart (int): starting from index 0, how many d_1 individuals.

    Returns:
        tuple of floats: (area_d_0, area_d_1)
    """
    dataset = dataset.copy()[:,:,config.HUNGERGAMES_TIME_LIMS[0]:
                            config.HUNGERGAMES_TIME_LIMS[1]]


    ttotal = dataset.shape[2]
    values_across_time = []
    non_indices = list(range(dataset.shape[0]))
    non_indices = [j for j in non_indices if j not in rel_indices]
    for t in range(0, ttotal, 20):
        data_sub = dataset[:,:,t]
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        area_d0 = np.log(areas[non_indices]).mean()
        area_d1 = np.log(areas[rel_indices]).mean()
        values_across_time.append([area_d0, area_d1])

    values_across_time = np.array(values_across_time)

    results =  values_across_time.mean(axis=0)
    return results[0], results[1]

def compute_metric(all_datasets, rel_indices, metricfunc):
    """
    For a given metric func (of the type of extract_areas and
    extract_groupsizes), computes the function for all available data
    and computes a metric for the hypothesis that focals > nonfocals.
    Args:
        all_datasets (list)
        rel_indices (list): list of indices of focal individuals
        metricfunc (func): extract_groupsizes or extract_areas
    Returns:
        fraction of dataset cases where focal > nonfocal
    """
    metricbases = [metricfunc(data, rel_indices)\
                    for data in all_datasets]
    metricdiff = metricbases[:,1] - metricbases[:,0]
    return sum(metricdiff>0)/len(metricdiff)

def permutation(all_datasets, rel_indices, metricfunc):
    """
    For a given metric func (of the type of extract_areas and
    extract_groupsizes), computes the function for all available data
    and performs one permutation using non-focal individuals.
    Args:
        all_datasets (list)
        rel_indices (list): list of indices of focal individuals
        metricfunc (func): extract_groupsizes or extract_areas
    """
    
    avail_indices = list(range(all_datasets[0].shape[0]))
    avail_indices = [j for j in avail_indices if j not in rel_indices]

    fake_indices = np.random.choice(avail_indices, len(rel_indices),
                        replace=False)
    return compute_metric(all_datasets, fake_indices, metricfunc)

def permutations(all_datasets, rel_indices, metricfunc, num_perms=10_000):
    """
    *GENERATOR* on permutations
    """
    for i in range(num_perms):
        yield permutation(all_datasets, rel_indices, metricfunc)

def extract_all_data():
    """
    Runs above analyses on all simulated hungergames data.
    """

    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in config.POP_S_SMART_GUYS_HG[popsize]:
            files = _hungergames_files_for(popsize, num_smart)
            gsizes, areas = [], []
            for file_ in files:
                data = _read_hungergames_data(file_)
                group_s_d0, group_s_d1 = extract_groupsizes(data, num_smart)
                area_d0, area_d1 = extract_areas(data, num_smart)

                gsizes.append([group_s_d0, group_s_d1])
                areas.append([area_d0, area_d1])
            gsizes = np.array(gsizes)
            areas = np.array(areas)

            print(popsize, num_smart)
            areadiff = areas[:,0] - areas[:,1]
            print(sum(areadiff>0)/len(areadiff))
        print()

extract_all_data()
