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

def extract_areas(dataset):
    """
    Compute Voronoi areas once for all sampled time points.

    Args:
        dataset (array-like, n×2×t): location data across time.

    Returns:
        np.ndarray (t_sampled × n): Voronoi area of each individual
        at each sampled time point.
    """
    dataset = dataset.copy()[
        :, :,
        config.HUNGERGAMES_TIME_LIMS[0]:
        config.HUNGERGAMES_TIME_LIMS[1]
    ]

    areas_across_time = []

    for t in range(
        0,
        dataset.shape[2],
        config.HUNGERGAMES_T_SAMPLE_EVERY
    ):
        data_sub = dataset[:, :, t]

        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        areas_across_time.append(areas)

    return np.asarray(areas_across_time)


def u_metric(areas, rel_indices):
    """
    Mean normalized Mann-Whitney U across time.

    Args:
        areas (array-like, t × n): Voronoi areas.
        rel_indices: indices of special individuals.

    Returns:
        float: mean U across time.

    U > 0.5 means special individuals tend to have smaller areas.
    """
    rel_indices = np.asarray(rel_indices)

    is_special = np.zeros(areas.shape[1], dtype=bool)
    is_special[rel_indices] = True

    special = areas[:, is_special]
    resident = areas[:, ~is_special]

    # Shape: time × n_special × n_resident
    comparisons = (
        special[:, :, None] < resident[:, None, :]
    )

    ties = (
        special[:, :, None] == resident[:, None, :]
    )

    u_by_time = (
        comparisons.sum(axis=(1, 2))
        + 0.5 * ties.sum(axis=(1, 2))
    ) / (special.shape[1] * resident.shape[1])

    return u_by_time.mean()


def run_data_analysis_on(momentum_anticipation=False, reverse=False,
                         n_permutations=config.HUNGERGAMES_NUM_PERM, seed=None):
    """
    Rank-based permutation analysis of Voronoi areas.

    H0: special identity is unrelated to Voronoi-area ranking.
    H1: special individuals tend to have smaller Voronoi areas.
    """
    colnames = ["gpsize", "num_smart", "stat", "area_p_val"]
    df = []

    rng = np.random.default_rng(seed)

    ftag = "momentum" if momentum_anticipation else "embedded"
    revtag = "reverse" if reverse else "noreverse"

    for gpsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in config.POP_S_SMART_GUYS_HG[gpsize]:

            print(f"Analysing n={gpsize}, n_invader={num_smart}.")

            files = _hungergames_files_for(
                gpsize,
                num_smart,
                momentum_anticipation,
                reverse
            )

            alldata = [
                _read_hungergames_data(file_)
                for file_ in files
            ]

            # ----------------------------------------------------------
            # Expensive part: compute Voronoi areas ONCE.
            # ----------------------------------------------------------
            allareas = [
                extract_areas(dataset)
                for dataset in alldata
            ]

            # ----------------------------------------------------------
            # Observed statistic
            # ----------------------------------------------------------
            rel_indices = np.arange(num_smart)

            observed_u = [
                u_metric(areas, rel_indices)
                for areas in allareas
            ]

            stat = np.mean(observed_u)

            # ----------------------------------------------------------
            # Permutation null
            # ----------------------------------------------------------
            permuted_stats = np.empty(n_permutations)

            for p in range(n_permutations):
                permuted_u = []

                for areas in allareas:

                    # New random special identities for this simulation.
                    # These identities remain fixed across all its timepoints.
                    perm_indices = rng.choice(
                        gpsize,
                        size=num_smart,
                        replace=False
                    )

                    permuted_u.append(
                        u_metric(areas, perm_indices)
                    )

                permuted_stats[p] = np.mean(permuted_u)

            # One-sided: large U = special individuals have smaller areas.
            p_value = (
                1 + np.sum(permuted_stats >= stat)
            ) / (n_permutations + 1)

            df.append([
                gpsize,
                num_smart,
                stat,
                p_value
            ])

    df = pd.DataFrame(df, columns=colnames)

    df.to_csv(
        joinpath(
            config.DATA,
            f"{ftag}-{revtag}-hungergames-results.csv"
        ),
        index=False
    )


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
