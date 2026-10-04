# Pranav Minasandra
# pminasandra.github.io
# 11 Feb 2025
# Fully overhauled and re-written based on reviewer suggestions
# on 3 Ocr 2026

import glob
import os
from os.path import join as joinpath
import multiprocessing as mp
import pickle
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import ttest_1samp

import config
import measurements
import selfishherd
import utilities
import voronoi

def uniform_points_in_circle(n, center=(0, 0), radius=1):
    """
    Uniformly sample n points inside a circle.
    """
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(np.random.uniform(0, 1, n))

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack((x, y))


def hungergame(init_locs, num_smart,
                momentum_anticipation=False,
                reverse=False):
    """
    Sets up an individual contest, starting a smart selfish herd
    of n individuals, of which the first num_smart are d_1 and the rest
    are d_0.

    Args:
        init_locs (n×2 array-like): initial locations of agents
        num_smart (int): how many d1 individuals
        momentum_anticipation (bool): whether agents use d_1 or d_\mu anticipation.
            True for d_\mu.
        reverse (bool): whether roles of d_0 and d_1/d_\mu agents should be swapped.
    Returns:
        selfishherd.SelfishHerd,
        fname (str)
    """

    num_inds = init_locs.shape[0]
    depths = np.zeros(num_inds).astype(int)
    if not momentum_anticipation:
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

    gpsize = num_inds

    ftag = "embedded"
    if momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    tgtdir = joinpath(config.DATA, "HungerGames", f"{gpsize}")
    os.makedirs(tgtdir, exist_ok=True)
    fname = joinpath(tgtdir,
                f"{ftag}-{revtag}-{num_inds}-n{num_smart}-{uname}.pkl")

    return herd, fname


def hungergames(gpsize, num_smart, num_instances,
                    momentum_anticipation=False,
                    reverse=False):
    """
    *GENERATOR*
    Wrapper around hungergame(...)
    Args:
        gpsize (int): group size
        num_smart (int): number of d_1 inds
        num_instances (int): how many simulations are needed
        momentum_anticipation (bool): whether agents use d_1 or d_\mu anticipation.
            True for d_\mu.
        reverse (bool): whether roles of d_0 and d_1/d_\mu agents should be swapped.
    """

    radius = np.sqrt(1e-3 / np.pi)#assuming initial area of 1e-3 units.
    for i in range(num_instances):
        init_locs = uniform_points_in_circle(gpsize, center=(0.5, 0.5), radius=radius)
        herd, fname = hungergame(init_locs, num_smart,
                    momentum_anticipation=momentum_anticipation,
                    reverse=reverse)

        yield herd, fname


def runmodel(herd, filename):
    """
    parallelization helper function
    """
    np.random.seed()
    herd.run(config.HUNGERGAMES_TMAX)
    herd.savedata(filename)


def simulate_all_hungergames():
    """
    *WRAPPER*
    Runs all simulations needed for the paper.
    """
    for gpsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in config.POP_S_SMART_GUYS_HG[gpsize]:

            # First the normal hunger-games
            print(f"Ordinary hunger-games for d1 invading d0. {gpsize=}")
            contests = hungergames(gpsize, num_smart, num_instances=config.NUM_REPEATS,
                        momentum_anticipation=False,
                        reverse=False)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool
            del contests

            # Then with the roles reversed
            print(f"Reversed hunger-games for d1 invading d0. {gpsize=}")
            contests = hungergames(gpsize, num_smart, num_instances=config.NUM_REPEATS,
                        momentum_anticipation=False,
                        reverse=True)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool
            del contests

            # Then with momentum only, for d\mu invading d0
            print(f"Ordinary hunger-games for d_\mu invading d0. {gpsize=}")
            contests = hungergames(gpsize, num_smart, num_instances=config.NUM_REPEATS,
                        momentum_anticipation=True,
                        reverse=False)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool
            del contests

            # Then momentum only + roles reversed, d0 invading d\mu
            print(f"Reversed hunger-games for d_\mu invading d0. {gpsize=}")
            contests = hungergames(gpsize, num_smart, num_instances=config.NUM_REPEATS,
                        momentum_anticipation=True,
                        reverse=True)

            pool = mp.Pool()
            pool.starmap(runmodel, contests)
            pool.close()
            pool.join()
            del pool
            del contests



def _hungergames_files_for(gpsize, num_smart,
                    momentum_anticipation=False,
                    reverse=False):

    ftag = "embedded"
    if momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    datadir = joinpath(config.DATA, "HungerGames", f"{gpsize}")
    fformat = f"{ftag}-{revtag}-{gpsize}-n{num_smart}-*.pkl"

    files = glob.glob(joinpath(datadir, fformat))

    return list(files)

def _read_hungergames_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def area_difference_metric(a_invader, a_resident):
    """
    Specific metric to be compared for both these areas.
    """
    mean_a_inv = np.log(a_invader.mean())
    mean_a_res = np.log(a_resident.mean())
    print(f"{mean_a_inv=}, {mean_a_res=}")
    return mean_a_inv - mean_a_res

def extract_area_diff_metric(dataset, rel_indices):
    """
    For a given dataset containing d_0 and d_1/d_\mu individuals, stores the area
    difference score of Voronoi polygons for invader (rel_indices) vs resident
    individuals.

    Args:
        dataset (array-like, n×2×t): location data across time.
        rel_indices (array-like, index): which indices represent invader agents.

    Returns:
        tuple of floats: (area_d_0, area_d_1)
    """
    dataset = dataset.copy()[:,:,config.HUNGERGAMES_TIME_LIMS[0]:
                            config.HUNGERGAMES_TIME_LIMS[1]]


    ttotal = dataset.shape[2]
    values_across_time = []
    non_indices = list(range(dataset.shape[0]))
    non_indices = [j for j in non_indices if j not in rel_indices]

    for t in range(0, ttotal, config.HUNGERGAMES_T_SAMPLE_EVERY):
        data_sub = dataset[:,:,t]
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        area_resident = areas[non_indices]
        area_invader = areas[rel_indices]

        values_across_time.append(area_difference_metric(area_invader, area_resident))

    values_across_time = np.array(values_across_time)

    return values_across_time.mean()

def run_data_analysis_on(momentum_anticipation=False, reverse=False):
    """
    Runs above analyses on simulated hungergames data.
    """

    colnames = ["gpsize", "num_smart",
                    "mean_val", "area_p_val"]
    df = []

    ftag = "embedded"
    if momentum_anticipation:
        ftag = "momentum"

    revtag = "noreverse"
    if reverse:
        revtag = "reverse"

    import matplotlib.pyplot as plt
    for gpsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in config.POP_S_SMART_GUYS_HG[gpsize]: #NOTE: CAN CHANGE AS YOU LIKE
            print(f"Analysing n={gpsize}, n_invader={num_smart}.")

            # read in all relevant files
            files = _hungergames_files_for(gpsize, num_smart,
                                    momentum_anticipation,
                                    reverse)
            alldata = [_read_hungergames_data(file_) for file_ in files]
            rel_indices = list(range(0, num_smart))

            # compute area-difference metric
            metric_values = []
            for dataset in alldata:
                metric_values.append(extract_area_diff_metric(dataset, rel_indices))

            plt.hist(metric_values, 100)
            # Now the stats: H0: stat >= 0; H1: stat < 0
            stat_results = ttest_1samp(metric_values, popmean=0, nan_policy='omit', alternative='less')

            df.append([gpsize, num_smart,
                        stat_results.statistic, stat_results.pvalue])
        plt.show()
        plt.clf(); plt.cla()
    df = pd.DataFrame(df, columns=colnames)
    df.to_csv(joinpath(config.DATA, f"{ftag}-{revtag}-hungergames-results.csv"), index=False)


def run_all_analyses():
    """
    Define all hunger-games related analyses.
    """

    print("d1 invading population of d0")
    run_data_analysis_on(momentum_anticipation=False, reverse=False)
    print()

    print("d0 invading population of d1")
    run_data_analysis_on(momentum_anticipation=False, reverse=True)
    print()

    print("d\\mu invading population of d0")
    run_data_analysis_on(momentum_anticipation=True, reverse=False)
    print()

    print("d0 invading population of d\\mu")
    run_data_analysis_on(momentum_anticipation=True, reverse=True)
    print()
