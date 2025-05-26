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

def group_touches_edge(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array indicating whether each point is in a group
    that touches the edge of the unit square.

    Args:
    - data: np.ndarray of shape (n, 2)
    - labels: np.ndarray of shape (n,), cluster labels (e.g., from DBSCAN)

    Returns:
    - np.ndarray of shape (n,), dtype=bool
    """
    edge_threshold = 0.01
    result = np.zeros(data.shape[0], dtype=bool)

    for label in np.unique(labels):
        if label == -1:
            # Skip noise points
            continue
        group_indices = np.where(labels == label)[0]
        group_points = data[group_indices]

        # Check if any point in the group touches the edge
        on_edge = (
            (group_points[:, 0] < edge_threshold) |
            (group_points[:, 0] > 1 - edge_threshold) |
            (group_points[:, 1] < edge_threshold) |
            (group_points[:, 1] > 1 - edge_threshold)
        )
        if np.any(on_edge):
            result[group_indices] = True

    return result

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

        all_areas_row.append(np.var(np.log(areas)))

    return all_areas_row

def gen_row_of_g_speeds(positions, timerange):
    all_speeds_row = []
    for t in timerange:
        data_sub = positions[:,:,t].copy()
        xs, ys = data_sub[:,0], data_sub[:,1]
        near_left = xs < 0.01 + 0.005
        near_right = xs > 0.99 - 0.005
        near_bottom = ys < 0.01 + 0.005
        near_top = ys > 0.99 - 0.005

        near_edge = near_left | near_right | near_bottom | near_top

        data_sub = data_sub[~near_edge]
        vels = 0.1*(positions[~near_edge,:,t+10] - positions[~near_edge,:,t])
        speeds = np.sqrt((vels**2).sum(axis=1))

        all_speeds_row.append(np.mean(speeds))

    return all_speeds_row

def gen_row_of_g_edgeeffects(positions, timerange):
    all_ee_row = []
    pcount = positions.shape[0]
    for t in timerange:
        data_sub = positions[:,:,t].copy()
        xs, ys = data_sub[:,0], data_sub[:,1]
        near_left = xs < 0.01 + 0.005
        near_right = xs > 0.99 - 0.005
        near_bottom = ys < 0.01 + 0.005
        near_top = ys > 0.99 - 0.005

        near_edge = near_left | near_right | near_bottom | near_top

        all_ee_row.append(np.sum(near_edge)/pcount)

    return all_ee_row

def extract_velocities_exclude_edge(
    data: np.ndarray,
    T_REL_MIN: int = 40,
    T_REL_MAX: int = 200,
    dbscan_fn = dbscan
) -> list:
    """
    Extracts velocities from an n×2×t trajectory array for the timepoints
    T_REL_MIN to T_REL_MAX (inclusive) in steps of 20, using dt=1,
    while excluding individuals in groups that touch the edge of the unit square.

    Parameters:
    - data: np.ndarray of shape (n, 2, 500), trajectories
    - T_REL_MIN: int, starting timepoint
    - T_REL_MAX: int, ending timepoint
    - dbscan_fn: function that takes (n, 2) array and returns (n,) cluster labels

    Returns:
    - List of velocity arrays (n_kept, 2), one for each selected timepoint
    """
    assert dbscan_fn is not None, "Must provide a dbscan function"

    # Use positions at T_REL_MIN to define clusters
    positions = data[:, :, T_REL_MIN]  # shape: (n, 2)
    labels = dbscan_fn(positions)

    # Identify individuals in edge-touching groups
    edge_mask = group_touches_edge(positions, labels)

    # Select only non-edge individuals
    keep_mask = ~edge_mask

    velocities = []
    for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
        if t + 1 >= data.shape[2]:
            break
        vel = data[keep_mask, :, t + 1] - data[keep_mask, :, t]
        velocities.extend(list(np.sqrt((vel**2).sum(axis=1))))

    return velocities

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

def make_speed_csv_for(pop_size, depth, timerange, eps=0.005):
    all_files = _files_for(pop_size, depth)
    rows = []
    for file_ in all_files:
        data = _read_data(file_)
        uname = "-".join(file_[:-len(".pkl")].split("-")[2:])
        rows.append([uname] + gen_row_of_g_speeds(data, timerange))

    col_labels =["uname"] + [f"t{time}" for time in timerange]

    df = pd.DataFrame(rows, columns=col_labels)
    return df

def make_edgeeffect_csv_for(pop_size, depth, timerange, eps=0.005):
    all_files = _files_for(pop_size, depth)
    rows = []
    for file_ in all_files:
        data = _read_data(file_)
        uname = "-".join(file_[:-len(".pkl")].split("-")[2:])
        rows.append([uname] + gen_row_of_g_edgeeffects(data, timerange))

    col_labels =["uname"] + [f"t{time}" for time in timerange]

    df = pd.DataFrame(rows, columns=col_labels)
    return df

if __name__ == "__main__":
    group_metrics = []
    if not config.ANALYSE_DATA:
        print("config.ANALYSE_DATA was False. Exiting.")
        quit()
    timerange = range(0, 481, 20)
    os.makedirs(joinpath(config.DATA, "Results"), exist_ok=True)
    for pop_size in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            tgt_file = joinpath(config.DATA, "Results",
                                    f"speed-{pop_size}-d{depth}.csv")
            data = make_speed_csv_for(pop_size, depth, timerange, eps=0.02)
            data.to_csv(tgt_file, index=False)
