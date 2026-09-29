# Pranav Minasandra
# pminasandra.github.io
# 11 Feb 2025

import glob
from os.path import join as joinpath
import multiprocessing as mp
import pickle
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError


import config
import measurements
import selfishherd
import utilities
import voronoi

def hungergame(init_locs, num_smart,
                only_momentum_anticipation=False,
                reverse=False):
    """
    Sets up an individual contest, starting a smart selfish herd
    of n individuals, of which the first num_smart are d_1 and the rest
    are d_0.

    Args:
        init_locs (n×2 array-like): initial locations of agents
        num_smart (int): how many d1 individuals
        only_momentum_anticipation (bool): whether agents use d_1 or d_\mu anticipation.
            True for d_\mu.
        reverse (bool): whether roles of d_0 and d_1/d_\mu agents should be swapped.
    Returns:
        selfishherd.SelfishHerd,
        fname (str)
    """

    num_inds = init_locs.shape[0]
    depths = np.zeros(num_inds).astype(int)
    if not only_momentum_anticipation:
        if not reverse:
            depths[:num_smart] = 1
        else:
            depths[num_smart:] = 1
    else:
        if not reverse:
            depths[:num_smart] = config.MU
        else:
            depths[num_smart:] = config.MU

    herd = selfishherd.SelfishHerd(num_inds, depths, init_locs)
    uname = str(uuid.uuid4())
    ftag = "embedded"
    if only_momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    fname = joinpath(config.DATA, "HungerGames",
                f"{ftag}-{revtag}-{num_inds}-n{num_smart}-{uname}.pkl")

    return herd, fname


def hungergames(popsize, num_smart, num_instances,
                    only_momentum_anticipation=False,
                    reverse=False):
    """
    *GENERATOR*
    Wrapper around hungergame(...)
    Args:
        popsize (int): population size
        num_smart (int): number of d_1 inds
        num_instances (int): how many simulations are needed
        only_momentum_anticipation (bool): whether agents use d_1 or d_\mu anticipation.
            True for d_\mu.
        reverse (bool): whether roles of d_0 and d_1/d_\mu agents should be swapped.
    """

    for i in range(num_instances):
        init_locs = np.random.uniform(size=(popsize, 2))
        herd, fname = hungergame(init_locs, num_smart,
                    only_momentum_anticipation=only_momentum_anticipation,
                    reverse=reverse)

        yield herd, fname


def runmodel(herd, filename):
    """
    parallelization helper function
    """
    np.random.seed()
    herd.run(config.TMAX)
    herd.savedata(filename)


def simulate_all_hungergames():
    """
    *WRAPPER*
    Runs all simulations needed for the paper.
    """
    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in config.POP_S_SMART_GUYS_HG[popsize]:
            # First the normal hunger-games
#            print(f"Ordinary hunger-games for d1 invading d0. {popsize=}")
#            contests = hungergames(popsize, num_smart, num_instances=config.NUM_REPEATS,
#                        only_momentum_anticipation=False,
#                        reverse=False)
#
#            pool = mp.Pool()
#            pool.starmap(runmodel, contests)
#            pool.close()
#            pool.join()
#            del pool
#
#            # Then with the roles reversed
#            print(f"Reversed hunger-games for d1 invading d0. {popsize=}")
#            contests = hungergames(popsize, num_smart, num_instances=config.NUM_REPEATS,
#                        only_momentum_anticipation=False,
#                        reverse=True)
#
#            pool = mp.Pool()
#            pool.starmap(runmodel, contests)
#            pool.close()
#            pool.join()
#            del pool

            # Then with momentun only
            print(f"Ordinary hunger-games for d_\mu invading d0. {popsize=}")
            contests = hungergames(popsize, num_smart, num_instances=config.NUM_REPEATS,
                        only_momentum_anticipation=True,
                        reverse=False)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool

            # Then momentum only + roled reversed
            print(f"Reversed hunger-games for d_\mu invading d0. {popsize=}")
            contests = hungergames(popsize, num_smart, num_instances=config.NUM_REPEATS,
                        only_momentum_anticipation=True,
                        reverse=True)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool



def _hungergames_files_for(popsize, num_smart,
                    only_momentum_anticipation=False,
                    reverse=False):

    ftag = "embedded"
    if only_momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    datadir = joinpath(config.DATA, "HungerGames")
    fformat = f"{ftag}-{revtag}-{popsize}-n{num_smart}-*.pkl"

    files = glob.glob(joinpath(datadir, fformat))

    return list(files)

def _read_hungergames_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# following analyses will be done after initial randomness
# let's say we will look at t=150 to t=250

def extract_areas(dataset, rel_indices):
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

        area_d0 = -np.log(areas[non_indices]).mean()
        area_d1 = -np.log(areas[rel_indices]).mean()
#        area_d0 = np.log(areas[non_indices]).mean()
#        area_d1 = np.log(areas[rel_indices]).mean()
        # NOTE: -np.log is chosen because hypothesis testing
        # functions below test for focal > non-focal, whereas 
        # area_focal < area_non-focal is our hypothesis.
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
    metricbases = np.array(metricbases)
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

def permutations(all_datasets, rel_indices, metricfunc, num_perms=1000):
    """
    *GENERATOR* on permutations
    """
    print()
    for i in range(num_perms):
        print(f"Permutation {i+1} of {num_perms}", end="\033[K\r")
        yield permutation(all_datasets, rel_indices, metricfunc)

def run_data_analysis(only_momentum_anticipation=False, reverse=False):
    """
    Runs above analyses on simulated hungergames data.
    """

    colnames = ["popsize", "num_smart",
                    "true_area_metric", "area_p_val"]
    df = []

    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in [5]: #NOTE: CAN CHANGE AS YOU LIKE
            print(f"Analysing n={popsize}, d_1={num_smart}.")
            files = _hungergames_files_for(popsize, num_smart,
                                    only_momentum_anticipation,
                                    reverse)
            alldata = [_read_hungergames_data(file_) for file_ in files]

            rel_indices = list(range(0, num_smart))

# area data analyses
            true_area_metric = compute_metric(alldata, rel_indices,
                                                extract_areas)
            print("true_area_metric:", true_area_metric)
            permuted_data = []
            for p in permutations(alldata, rel_indices, extract_areas, num_perms=5000):
                permuted_data.append(p)
            permuted_data = np.array(permuted_data)
            fig, ax = plt.subplots()
            ax.hist(permuted_data, 75)
            ax.axvline(true_area_metric, color="red")
            print(f"Out of {len(permuted_data)} sims, {sum(permuted_data >= true_area_metric)} were served.")
            ax.set_xlabel("Proportion of sims with smaller domains of danger")
            utilities.saveimg(fig, f"stat_test_area_{popsize}")
            print()
            area_p_val = sum(permuted_data >= true_area_metric)/len(permuted_data)
            print("area_p_val:", area_p_val)

            df.append([popsize, num_smart,
                        true_area_metric, area_p_val])

    ftag = "embedded"
    if only_momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    import pandas as pd
    df = pd.DataFrame(df, columns=colnames)
    df.to_csv(joinpath(config.DATA, f"{ftag}-{revtag}-hungergames-results.csv"), index=False)
