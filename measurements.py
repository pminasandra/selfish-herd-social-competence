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


def extract_all_group_sizes(data: np.ndarray, T_REL_MIN=40, T_REL_MAX=200, dbscan_fn=dbscan) -> list:
    """
    Collects all group sizes between t=T_REL_MIN and t=T_REL_MAX (inclusive)
    in steps of 20 using DBSCAN clustering.

    Parameters:
    - data: np.ndarray of shape (n, 2, 500)
    - dbscan_fn: function (n, 2) -> (n,) cluster labels

    Returns:
    - List[int]: group sizes across all selected timepoints
    """
    assert dbscan_fn is not None
    group_sizes_all = []

    for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
        if t >= data.shape[2]:
            break

        positions = data[:, :, t]  # shape: (n, 2)
        labels = dbscan_fn(positions)
        sizes = group_sizes(labels)
        tgs = typical_group_size(sizes)

        group_sizes_all.append(tgs)
    return group_sizes_all


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

def extract_log_voronoi_areas(data: np.ndarray, T_REL_MIN=40, T_REL_MAX=200) -> list:
    """
    Extracts log Voronoi cell areas for each individual between
    T_REL_MIN and T_REL_MAX (inclusive, step 20).

    Parameters:
    - data: np.ndarray of shape (n, 2, 500)
    
    Returns:
    - List[float]: flattened list of log-areas
    """
    log_areas = []

    for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
        if t >= data.shape[2]:
            break

        data_sub = data[:, :, t].copy()  # shape: (n, 2)
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)  # shape: (n,)

        # Take log of each area (safely)
        log_areas.extend(np.log(areas + 1e-10).tolist())  # small epsilon to avoid log(0)

    return log_areas

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
    dt = 5

    # Use positions at T_REL_MIN to define clusters
    positions = data[:, :, T_REL_MIN]  # shape: (n, 2)
    labels = dbscan_fn(positions)

    # Identify individuals in edge-touching groups
    edge_mask = group_touches_edge(positions, labels)

    # Select only non-edge individuals
    keep_mask = ~edge_mask

    velocities = []
    for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
        if t + dt >= data.shape[2]:
            break
        vel = (data[keep_mask, :, t + dt] - data[keep_mask, :, t])/dt
        velocities.extend(list(np.sqrt((vel**2).sum(axis=1))))

    return velocities

def extract_polarisations_exclude_edge(data: np.ndarray, T_REL_MIN=40, T_REL_MAX=200, dbscan_fn=dbscan) -> list:
    """
    Computes polarisation for each group (excluding edge-touching and singleton groups).

    Returns:
    - List[float]: Each group's polarisation repeated k times (group size k, k > 1)
    """
    assert dbscan_fn is not None

    positions_ref = data[:, :, T_REL_MIN]
    labels_ref = dbscan_fn(positions_ref)
    edge_mask = group_touches_edge(positions_ref, labels_ref)
    keep_mask = ~edge_mask

    polarisations = []

    for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
        if t + 1 >= data.shape[2]:
            break

        positions_t = data[:, :, t]
        positions_t1 = data[:, :, t + 1]

        pos = positions_t[keep_mask]
        pos1 = positions_t1[keep_mask]

        labels = dbscan_fn(pos)

        for label in np.unique(labels):
            if label == -1:
                continue  # skip noise

            group_idx = np.where(labels == label)[0]
            if len(group_idx) <= 1:
                continue  # skip singleton groups

            group_vel = pos1[group_idx] - pos[group_idx]
            norms = np.linalg.norm(group_vel, axis=1, keepdims=True)

            valid = norms[:, 0] > 1e-8
            unit_vecs = np.zeros_like(group_vel)
            unit_vecs[valid] = group_vel[valid] / norms[valid]

            mean_vec = np.mean(unit_vecs[valid], axis=0)
            polarisation = np.linalg.norm(mean_vec)

            k = len(group_idx)
            polarisations.extend([polarisation] * k)

    return polarisations

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
