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

def make_violinplot(fn, ydesc, fig=None, ax=None, palette="pastel"):
    """
    Makes a violinplot of speed as a function of depth of reasoning
    Args:
        fn: what function to use to compute data for plotting
        ydesc: str description of wth you are measuring
    """
    records = []

    for pop_size in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            files = measurements._files_for(pop_size, depth)
            for f in files:
                data = measurements._read_data(f)  # Should be n×2×500 array
                points = fn(data)
                for s in points:
                    records.append({
                        ydesc: s,
                        "depth": depth,
                        "pop_size": pop_size
                    })

    df = pd.DataFrame.from_records(records)

    # Plot

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    g = sns.violinplot(data=df, x="pop_size", y=ydesc, hue="depth",
                        dodge=True, ax=ax, inner='box', palette=palette,
                        bw_method=0.05
                    )
    ax.set_xlabel("Population size")
    ax.set_ylabel(ydesc)
#    ax.legend(title="Depth of reasoning", fontsize="small", title_fontsize="small")
    ax.legend(
        title="Depth of reasoning",
        fontsize="x-small",
        title_fontsize="x-small",
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0.5
    )

    return fig, ax

def compare_gpsize_area_relation():
    plot_data = {}
    for pop_size in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            if depth not in plot_data:
                plot_data[depth] = []
            files = measurements._files_for(pop_size, depth)

            for f in files:
                data = measurements._read_data(f)  # Should be n×2×500 array

                size_areas = measurements.extract_all_group_areas(data)
                plot_data[depth].extend(size_areas)

    fig, ax = plt.subplots()
    dfs = []
    for depth in config.ANALYSE_DEPTHS:
        xs = [point[0] for point in plot_data[depth]]
        ys = [point[1] for point in plot_data[depth]]

        df = pd.DataFrame({"Group Size": xs, "Group Area": ys})
        df = df[df["Group Size"] < 35] #for better plotting
        df.loc[:, "Depth"] = f"$d_{depth}$"
        dfs.append(df)

    df = pd.concat(dfs)
    sns.pointplot(data=df,
        x="Group Size",
        y="Group Area",
        hue="Depth",
        estimator=np.median,
        linestyle=None,
        errorbar=("pi", 50),
        ax=ax,
        markersize=3,
        err_kws={"alpha": 0.4, "linewidth": 0.8},
        alpha= 0.6,
        dodge=0.3
    )

    ax.set_yscale("log")
    ax.set_xticks(ax.get_xticks()[2::5])
    utilities.saveimg(fig, "gpsize_area_relation")
            
                


if __name__ == "__main__":
#    fig, ax = plt.subplots(figsize=(11.45, 4.921))
#    make_violinplot(measurements.extract_all_group_sizes, ydesc="group size", fig=fig, ax=ax, palette="pastel")
#    utilities.saveimg(fig, "vplot-group-sizes")
    compare_gpsize_area_relation()
