# Pranav Minasandra and Cecilia Baldoni
# 27 Jan, 2025
# pminsandra.github.io

import os.path
from os.path import join as joinpath

from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import measurements
import utilities

RESULTS_DIR = joinpath(config.DATA, "Results")

trange = np.array(range(0, 481, 20))#(0, 481, 20) for speed etc
tcolnames = [f"t{i}" for i in trange]

keep_only = trange>100

def load_gsdata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"{popsize}-d{dval}.csv"))

def load_areadata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"areas-{popsize}-d{dval}.csv"))
    
def load_areavardata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"var-area-{popsize}-d{dval}.csv"))

def load_speeddata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"speed-{popsize}-d{dval}.csv"))

def load_eedata_for(popsize, dval):
    return pd.read_csv(joinpath(RESULTS_DIR, f"ee-{popsize}-d{dval}.csv"))

def get_avg_val(df):
    dfs = df[tcolnames]

    return dfs.mean(axis=0).to_numpy()

def get_cis_for(df, ulim=0.975, llim=0.025):
    dfs = df[tcolnames]

    dfs = dfs.to_numpy()
    res = np.quantile(dfs, (ulim, llim),
                method='closest_observation',
                axis=0)

    return res

def speed_violinplot():
    """
    Makes a violinplot of speed as a function of depth of reasoning
    """
    records = []

    for pop_size in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            files = measurements._files_for(pop_size, depth)
            for f in files:
                try:
                    data = measurements._read_data(f)  # Should be n×2×500 array
                    speeds = measurements.extract_velocities_exclude_edge(data)
                    for s in speeds:
                        records.append({
                            "speed": s,
                            "depth": depth,
                            "pop_size": pop_size
                        })
                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")

    df = pd.DataFrame.from_records(records)

    # Plot
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    g = sns.violinplot(data=df, x="pop_size", y="speed", hue="depth", dodge=True, ax=ax, inner='box', palette="pastel")
    ax.set_xlabel("Population size")
    ax.set_ylabel("Speed")
    ax.legend(title="Depth of reasoning", fontsize="small", title_fontsize="small")
    utilities.saveimg(fig, "speed-violinplot")

if __name__ == "__main__":
    speed_violinplot()
