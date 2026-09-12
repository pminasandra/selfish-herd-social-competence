# Pranav Minasandra
# 11 Sep 2026
# pminasandra.github.io

"""
Potential mechanistic explanation for why groups form.
"""

import multiprocessing as mp
import uuid

import numpy as np
import pandas as pd

import config
import measurements
from selfishherd import SelfishHerd as herd
import utilities

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
    trajs = collision_trial(pop_size, depth, t_max)
    recs = measurements.extract_all_group_areas(trajs, 0, t_max, 10)
    recs = pd.DataFrame(recs, columns=["group_size", "area", "exterior", "timestamp"])

    tgt_dir = fusion_exp_dir / f"{pop_size}" / f"d{depth}"
    tgt_dir.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(tgt_dir / f"{pop_size}-{depth}-{uuid.uuid4()}.parquet")
    print(pop_size, depth, t_max, "done!")
    

if __name__ == "__main__":
    params = [
                        (pop_size, depth, t_max) for n in range(n_repeats)
                    for depth in depths_tested
                for pop_size in pop_sizes_tested]
    print(params)

    pool = mp.Pool()
    pool.starmap(_mp_helper, params)
    pool.close()
    pool.join()
