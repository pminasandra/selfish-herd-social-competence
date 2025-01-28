# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides methods to detect groups of animals, and to identify key variables
such as number of groups formed and time to formation of groups.
"""

import glob
from os.path import join as joinpath
import os
import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import config
import voronoi

def _files_for(pop_size, depth):
    dir_ = joinpath(config.DATA, str(pop_size), "d"+str(depth))
    if not os.path.exists(dir_):
        return []
    else:
        files_ = glob.glob(joinpath(dir_, f"{pop_size}-{depth}-*.pkl"))
        files_.sort()
        return files_

def _read_data(filename):
    with open(filename, "rb") as file_obj:
        return pickle.load(file_obj)

def dbscan(positions, eps=0.005):
    """
    Given an iterable of individual positions, returns DBSCAN labels for each.
    Args:
        positions (iterable, typically np.ndarray with shape (n, 2)).
        eps (float): Distance threshold for DBSCANning
    Returns:
        np.array with cluster labels
    """

    pos = positions.copy()

    scanner = DBSCAN(eps=eps, min_samples=1)
    clusters = scanner.fit(pos)

    return clusters.labels_


def group_sizes(labels):
    """
    Extracts group sizes from given labels.
    """

    group_labels, group_sizes = np.unique(labels, return_counts=True)
    return group_sizes


def typical_group_size(group_sizes):
    """
    Computes typical group size.
    Jarman, P. J. 1974. The social organization of antelope in relation to their ecology.
    Behaviour, 48, 215e268.
    Args:
        labels (np.array -like), output from dbscan(...)
    Returns:
        float
    """

    return (group_sizes**2).sum()/group_sizes.sum()


def average_typical_group_size(data, timerange, eps=0.005):
    """
    Computes typical group-size averaged across time from data.
    """
    all_tgs = []
    for t in timerange:
        data_sub = data[:,:,t]
        group_sizes_extract = group_sizes(dbscan(data_sub, eps=eps))

        tgs = typical_group_size(group_sizes_extract)
        all_tgs.append(tgs)

    return np.array(all_tgs).mean()


def gen_row_of_g_sizes(positions, timerange, eps=0.005):
    all_tgs_row = []
    for t in timerange:
        data_sub = positions[:,:,t].copy()
        group_sizes_extract = group_sizes(dbscan(data_sub, eps=eps))

        tgs = typical_group_size(group_sizes_extract)
        all_tgs_row.append(tgs)
    return all_tgs_row

def gen_row_of_g_areas(positions, timerange):
    all_areas_row = []
    for t in timerange:
        data_sub = positions[:,:,t].copy()
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        all_areas_row.append(np.median(areas))

    return all_areas_row

def gen_row_of_g_area_vars(positions, timerange):
    all_areas_row = []
    for t in timerange:
        data_sub = positions[:,:,t].copy()
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        all_areas_row.append(np.var(areas))

    return all_areas_row

def make_tgs_csv_for(pop_size, depth, timerange, eps=0.005):
    all_files = _files_for(pop_size, depth)
    rows = []
    for file_ in all_files:
        data = _read_data(file_)
        uname = "-".join(file_[:-len(".pkl")].split("-")[2:])
        rows.append([uname] + gen_row_of_g_sizes(data, timerange, eps=eps))

    col_labels =["uname"] + [f"t{time}" for time in timerange]

    df = pd.DataFrame(rows, columns=col_labels)
    return df

def make_area_csv_for(pop_size, depth, timerange, eps=0.005):
    all_files = _files_for(pop_size, depth)
    rows = []
    for file_ in all_files:
        data = _read_data(file_)
        uname = "-".join(file_[:-len(".pkl")].split("-")[2:])
        rows.append([uname] + gen_row_of_g_areas(data, timerange))

    col_labels =["uname"] + [f"t{time}" for time in timerange]

    df = pd.DataFrame(rows, columns=col_labels)
    return df

def make_areavar_csv_for(pop_size, depth, timerange, eps=0.005):
    all_files = _files_for(pop_size, depth)
    rows = []
    for file_ in all_files:
        data = _read_data(file_)
        uname = "-".join(file_[:-len(".pkl")].split("-")[2:])
        rows.append([uname] + gen_row_of_g_area_vars(data, timerange))

    col_labels =["uname"] + [f"t{time}" for time in timerange]

    df = pd.DataFrame(rows, columns=col_labels)
    return df

if __name__ == "__main__":
    group_metrics = []
    if not config.ANALYSE_DATA:
        print("config.ANALYSE_DATA was False. Exiting.")
        quit()
    timerange = range(0, 501, 20)
    os.makedirs(joinpath(config.DATA, "Results"), exist_ok=True)
    for pop_size in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            tgt_file = joinpath(config.DATA, "Results",
                                    f"var-area-{pop_size}-d{depth}.csv")
            data = make_areavar_csv_for(pop_size, depth, timerange, eps=0.02)
            data.to_csv(tgt_file, index=False)
