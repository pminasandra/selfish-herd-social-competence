# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides methods to detect groups of animals, and to identify key variables
such as number of groups formed and time to formation of groups.
"""

import glob
from os.path import join as joinpath
import os.path
import pickle

import numpy as np
from sklearn.cluster import DBSCAN

import config

def _files_for(pop_size, depth):
    dir_ = joinpath(config.DATA, str(pop_size), "d"+str(depth))
    if not os.path.exists(dir_):
        return []
    else:
        return glob.glob(joinpath(dir_, f"{pop_size}-{depth}-*.pkl"))

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

if __name__ == "__main__":
    group_metrics = []
    for depth in [0, 1, 2, 3]:
        filenames = _files_for(25, depth)
        all_data = [_read_data(file_) for file_ in filenames]

        group_metrics_sub = [average_typical_group_size(data, range(200,501,50))
                                for data in all_data]

        print(depth, "-->", np.array(group_metrics_sub).mean())

    print(group_metrics)
