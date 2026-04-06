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
            print(f"Analysing {ydesc} for {pop_size=} {depth=}")
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
            print(f"gpsize area centrality analysis {pop_size=} {depth=}")
            if depth not in plot_data:
                plot_data[depth] = []
            files = measurements._files_for(pop_size, depth)

            for f in files:
                data = measurements._read_data(f)  # n×2×500 array

                size_areas = measurements.extract_all_group_areas(data)
                # each entry: [group_size, hull_area, exterior_prop]
                plot_data[depth].extend(size_areas)

    # Build one combined DataFrame with area + exterior proportion
    dfs = []
    for depth in config.ANALYSE_DEPTHS:
        xs  = [point[0] for point in plot_data[depth]]
        ysA = [point[1] for point in plot_data[depth]]  # area
        ysP = [point[2] for point in plot_data[depth]]  # exterior proportion

        df = pd.DataFrame(
            {
                "Group Size": xs,
                "Group Area": ysA,
                "Proportion on Exterior": ysP,
            }
        )

        # limit very large group sizes for plotting
        df = df[df["Group Size"] < 35]

        df.loc[:, "Depth"] = f"$d_{depth}$"
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # ---------- Plot 1: area vs group size (as before) ----------
    fig1, ax1 = plt.subplots()
    sns.pointplot(
        data=df,
        x="Group Size",
        y="Group Area",
        hue="Depth",
        estimator=np.median,
        linestyle=None,
        errorbar=("pi", 50),
        ax=ax1,
        markersize=3,
        err_kws={"alpha": 0.4, "linewidth": 0.8},
        dodge=0.3,
        alpha=0.6
    )

    ax1.set_yscale("log")
    # keep every 5th tick, starting around size 5 (adjust as needed)
    ax1.set_xticks(ax1.get_xticks()[2::5])
    utilities.saveimg(fig1, "gpsize_area_relation")

    # ---------- Plot 2: exterior proportion vs group size ----------
    # only from group size 5 onwards
    df_prop = df[df["Group Size"] >= 5]

    fig2, ax2 = plt.subplots()
    sns.pointplot(
        data=df_prop,
        x="Group Size",
        y="Proportion on Exterior",
        hue="Depth",
        estimator=np.mean,
        linestyle=None,
        errorbar=("ci", 95),
        ax=ax2,
        markersize=3,
        err_kws={"alpha": 0.4, "linewidth": 0.8},
        dodge=0.3,
        alpha=0.6,
    )

    ax2.set_ylabel("Proportion on Group Exterior")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(ax2.get_xticks()[2::5])
    utilities.saveimg(fig2, "gpsize_exterior_prop_relation")


def plot_group_size_ccdfs_displot(
    T_REL_MIN: int = 40,
    T_REL_MAX: int = 200,
    dbscan_fn=measurements.dbscan,
):
    """
    Plot complementary empirical CDFs of group sizes.

    For each population size in config.ANALYSE_POP_SIZES and each depth in
    config.ANALYSE_DEPTHS:

      - Cluster individuals at timepoints t in [T_REL_MIN, T_REL_MAX] (step 20)
        using `dbscan_fn`.
      - For each DBSCAN cluster (label != -1), record its size and whether
        that group touches the edge of the unit square (via group_touches_edge).
      - Treat noise points (label == -1) as groups of size 1, and classify
        them as edge-adjacent or not based on their own position.
      - Aggregate all group sizes in a long DataFrame.

    Then use seaborn.displot(kind="ecdf", complementary=True) to plot the
    complementary ECDF (CCDF) of group sizes, with:

        - 2 rows: groups that *do not* touch the edge (top),
                  groups that *do* touch the edge (bottom)
        - columns: one per population size
        - hue: depth (config.ANALYSE_DEPTHS)
    """
    assert dbscan_fn is not None

    rows = []
    edge_threshold = 0.02  # consistent with group_touches_edge

    for pop in config.ANALYSE_POP_SIZES:
        for depth in config.ANALYSE_DEPTHS:
            print(f"Collecting group sizes: n={pop}, depth={depth}")
            files = measurements._files_for(pop, depth)
            for f in files:
                data = measurements._read_data(f)  # shape: (N, 2, T)
                N, _, T = data.shape

                for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
                    if t >= T:
                        break

                    positions = data[:, :, t]  # (N, 2)
                    labels = dbscan_fn(positions)
                    touches = measurements.group_touches_edge(positions, labels)  # (N,) bool

                    # clustered groups
                    for label in np.unique(labels):
                        if label == -1:
                            continue  # skip noise here

                        group_idx = np.where(labels == label)[0]
                        size = group_idx.size
                        if size == 0:
                            continue

                        edge_adj = np.any(touches[group_idx])
                        edge_status = "Yes" if edge_adj else "No"

                        rows.append(
                            {
                                "Group size": size,
                                "Population size": pop,
                                "Depth": f"$d_{depth}$",
                                "Edge-Adjacent": edge_status,
                            }
                        )

                    # noise points as singleton groups (size 1)
                    noise_idx = np.where(labels == -1)[0]
                    if noise_idx.size > 0:
                        pts = positions[noise_idx]
                        on_edge = (
                            (pts[:, 0] < edge_threshold)
                            | (pts[:, 0] > 1 - edge_threshold)
                            | (pts[:, 1] < edge_threshold)
                            | (pts[:, 1] > 1 - edge_threshold)
                        )
                        for is_edge in on_edge:
                            edge_status = "Yes" if is_edge else "No"
                            rows.append(
                                {
                                    "Group size": 1,
                                    "Population size": pop,
                                    "Depth": f"$d_{depth}$",
                                    "Edge-Adjacent": edge_status,
                                }
                            )

    df_plot = pd.DataFrame(rows)
    print(df_plot[df_plot["Edge-Adjacent"] == "Yes"].shape, df_plot.shape)

    # --- displot: ECDF + complementary → CCDF ---
    g = sns.displot(
        data=df_plot,
        x="Group size",
        hue="Depth",
        row="Population size",
        col="Edge-Adjacent",
        kind="ecdf",
        complementary=True,  # CCDF: P(size >= s)
        palette="colorblind",
        height=3,
        log_scale=(False, True),
        aspect=1.2,
    )

    g.set_titles("")
    g.set(ylim=(0, 1.0))
    g.fig.subplots_adjust(top=0.9)

    utilities.saveimg(g.fig, "group_size_ccdf_edge_vs_noedge_displot")


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(11.45, 4.921))
    make_violinplot(measurements.extract_polarisations_exclude_edge, ydesc="polarisation", fig=fig, ax=ax, palette="pastel")
    utilities.saveimg(fig, "vplot-polarisations")
    compare_gpsize_area_relation()
    plot_group_size_ccdfs_displot()
