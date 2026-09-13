# Pranav Minasandra
# 11 Sep 2026
# pminasandra.github.io

"""
Potential mechanistic explanation for why groups form.
"""

import multiprocessing as mp
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import measurements
from selfishherd import SelfishHerd as herd
import utilities

# PROGRAM FLOW
RUN_SIMS = False
ANALYSE_OUTPUT = True

depths_tested = [0, 1, 2, 3, config.MU]
pop_sizes_tested = [50]
n_repeats = 100
init_intergroup_dist = 0.3
init_group_spread = 0.01
t_max = 500

fusion_exp_dir = config.DATA / "Fusion_Exp"

def setup_collision_test(n):
    """
    Create initial locations of individuals ready for a group collision test.

    Args:
        n (int): how many individuals exist.

    Returns:
        np.array (n×2): locations of individuals.
    """
    n_left = n // 2
    n_right = n - n_left

    left_center = np.array([0.5 - init_intergroup_dist/2, 0.5])
    right_center = np.array([0.5 + init_intergroup_dist/2, 0.5])

    left = left_center + np.random.uniform(
        -init_group_spread / 2,
        init_group_spread / 2,
        size=(n_left, 2)
    )

    right = right_center + np.random.uniform(
        -init_group_spread / 2,
        init_group_spread / 2,
        size=(n_right, 2)
    )

    positions = np.vstack((left, right))
    positions = np.clip(positions, 0.01, 0.99)

    return positions


def collision_trial(n, depth, t_max):
    """
    Sets up and runs collision trial for specified time period.
    Args:
        n (int): how many individuals exist.
        depth (int): depth of reasoning of these individuals.
        t_max (int): how many timesteps to simulate
    Returns:
        np.array (n×2×t_max)
    """
    init_locs = setup_collision_test(n)
    trial_herd = herd(n, depth, init_locs)

    trial_herd.run(t_max)

    return trial_herd.records


def _mp_helper(pop_size, depth, t_max):
    np.random.seed()
    trajs = collision_trial(pop_size, depth, t_max)
    recs = measurements.extract_all_group_areas(trajs, 0, t_max, 10)
    recs = pd.DataFrame(recs, columns=["group_size", "area", "exterior", "timestamp"])

    tgt_dir = fusion_exp_dir / f"{pop_size}" / f"d{depth}"
    tgt_dir.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(tgt_dir / f"{pop_size}-{depth}-{uuid.uuid4()}.parquet")
    print(pop_size, depth, t_max, "done!")
    

def _load_fusion_files_for(pop_size, depth):
    fdir = fusion_exp_dir / f"{pop_size}" / f"d{depth}"
    dfs = [pd.read_parquet(f) for f in fdir.glob("*.parquet")]
    return dfs


def make_plot(dfs_by_d):
    plot_data = pd.concat([
        pd.concat(dfs, keys=range(len(dfs)), names=["replicate"])
          .reset_index(level="replicate")
          .assign(d=d)
        for d, dfs in dfs_by_d.items()
    ])

# Rename only for display
    plot_data["d"] = plot_data["d"].replace({"d-1": r"$d_\mu$"})

    fig, ax = plt.subplots()

    sns.lineplot(
        data=plot_data,
        x="timestamp",
        y="group_size",
        hue="d",
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
    )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Average group size")
    ax.set_ylim((15, 60))
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    if RUN_SIMS:
        params = [
                            (pop_size, depth, t_max) for n in range(n_repeats)
                        for depth in depths_tested
                    for pop_size in pop_sizes_tested]
        print(params)

        pool = mp.Pool()
        pool.starmap(_mp_helper, params)
        pool.close()
        pool.join()

    if ANALYSE_OUTPUT:
        dfs_by_d = {}
        for pop_size in pop_sizes_tested:
            for depth in depths_tested:
                dfs_by_d[f"d{depth}"] = _load_fusion_files_for(pop_size, depth)

            fig, ax = make_plot(dfs_by_d)
            utilities.saveimg(fig, f"fusion_graph_{pop_size}")
            plt.close(fig)
