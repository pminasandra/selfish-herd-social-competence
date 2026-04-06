# Pranav Minasandra
# pminasandra.github.io
# 11 Feb 2025

import glob
from os.path import join as joinpath
import pickle
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError


import config
import measurements
import selfishherd
import utilities
import voronoi

def hungergame(init_locs, num_smart):
    """
    Sets up an individual contest, starting a smart selfish herd
    of n individuals, of which the first num_smart are d_1 and the rest
    are d_0.

    Args:
        init_locs (n×2 array-like): initial locations of agents
        num_smart (int): how many d1 individuals
    Returns:
        selfishherd.SelfishHerd,
        fname (str)
    """

    num_inds = init_locs.shape[0]
    depths = np.zeros(num_inds).astype(int)
    depths[:num_smart] += 1

    herd = selfishherd.SelfishHerd(num_inds, depths, init_locs)
    uname = str(uuid.uuid4())
    fname = joinpath(config.DATA, "HungerGames",
                f"{num_inds}-n{num_smart}-{uname}.pkl")

    return herd, fname


def hungergames(popsize, num_smart, num_instances):
    """
    *GENERATOR*
    Wrapper around hungergame(...)
    Args:
        popsize (int): population size
        num_smart (int): number of d_1 inds
        num_instances (int): how many simulations are needed
    """

    for i in range(num_instances):
        init_locs = np.random.uniform(size=(popsize, 2))
        herd, fname = hungergame(init_locs, num_smart)

        yield herd, fname


def _hungergames_files_for(popsize, num_smart):
    datadir = joinpath(config.DATA, "HungerGames")
    fformat = f"{popsize}-n{num_smart}-*.pkl"

    files = glob.glob(joinpath(datadir, fformat))

    return list(files)

def _read_hungergames_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# following analyses will be done after initial randomness
# let's say we will look at t=150 to t=250

def extract_areas(dataset, rel_indices):
    """
    For a given dataset containing d_0 and d_1 individuals,  returns mean Voronoi areas
    for d_0 and d_1 individuals separately.

    Args:
        dataset (array-like, n×2×t): location data across time.
        num_smart (int): starting from index 0, how many d_1 individuals.

    Returns:
        tuple of floats: (area_d_0, area_d_1)
    """
    dataset = dataset.copy()[:,:,config.HUNGERGAMES_TIME_LIMS[0]:
                            config.HUNGERGAMES_TIME_LIMS[1]]


    ttotal = dataset.shape[2]
    values_across_time = []
    non_indices = list(range(dataset.shape[0]))
    non_indices = [j for j in non_indices if j not in rel_indices]
    for t in range(0, ttotal, 20):
        data_sub = dataset[:,:,t]
        vor = voronoi.get_bounded_voronoi(data_sub)
        areas = voronoi.get_areas(data_sub, vor)

        area_d0 = -np.log(areas[non_indices]).mean()
        area_d1 = -np.log(areas[rel_indices]).mean()
#        area_d0 = np.log(areas[non_indices]).mean()
#        area_d1 = np.log(areas[rel_indices]).mean()
        # NOTE: -np.log is chosen because hypothesis testing
        # functions below test for focal > non-focal, whereas 
        # area_focal < area_non-focal is our hypothesis.
        values_across_time.append([area_d0, area_d1])

    values_across_time = np.array(values_across_time)

    results =  values_across_time.mean(axis=0)
    return results[0], results[1]

def compute_metric(all_datasets, rel_indices, metricfunc):
    """
    For a given metric func (of the type of extract_areas and
    extract_groupsizes), computes the function for all available data
    and computes a metric for the hypothesis that focals > nonfocals.
    Args:
        all_datasets (list)
        rel_indices (list): list of indices of focal individuals
        metricfunc (func): extract_groupsizes or extract_areas
    Returns:
        fraction of dataset cases where focal > nonfocal
    """
    metricbases = [metricfunc(data, rel_indices)\
                    for data in all_datasets]
    metricbases = np.array(metricbases)
    metricdiff = metricbases[:,1] - metricbases[:,0]
    return sum(metricdiff>0)/len(metricdiff)

def permutation(all_datasets, rel_indices, metricfunc):
    """
    For a given metric func (of the type of extract_areas and
    extract_groupsizes), computes the function for all available data
    and performs one permutation using non-focal individuals.
    Args:
        all_datasets (list)
        rel_indices (list): list of indices of focal individuals
        metricfunc (func): extract_groupsizes or extract_areas
    """
    
    avail_indices = list(range(all_datasets[0].shape[0]))
    avail_indices = [j for j in avail_indices if j not in rel_indices]

    fake_indices = np.random.choice(avail_indices, len(rel_indices),
                        replace=False)
    return compute_metric(all_datasets, fake_indices, metricfunc)

def permutations(all_datasets, rel_indices, metricfunc, num_perms=1000):
    """
    *GENERATOR* on permutations
    """
    print()
    for i in range(num_perms):
        print(f"Permutation {i+1} of {num_perms}", end="\033[K\r")
        yield permutation(all_datasets, rel_indices, metricfunc)

def run_data_analysis():
    """
    Runs above analyses on simulated hungergames data.
    """

    colnames = ["popsize", "num_smart",
                    "true_area_metric", "area_p_val"]
    df = []

    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in [5]: #NOTE: CAN CHANGE AS YOU LIKE
            print(f"Analysing n={popsize}, d_1={num_smart}.")
            files = _hungergames_files_for(popsize, num_smart)
            alldata = [_read_hungergames_data(file_) for file_ in files]

            rel_indices = list(range(0, num_smart))

# area data analyses
            true_area_metric = compute_metric(alldata, rel_indices,
                                                extract_areas)
            print("true_area_metric:", true_area_metric)
            permuted_data = []
            for p in permutations(alldata, rel_indices, extract_areas, num_perms=5000):
                permuted_data.append(p)
            permuted_data = np.array(permuted_data)
            fig, ax = plt.subplots()
            ax.hist(permuted_data, 75)
            ax.axvline(true_area_metric, color="red")
            print(f"Out of {len(permuted_data)} sims, {sum(permuted_data >= true_area_metric)} were served.")
            ax.set_xlabel("Proportion of sims with smaller domains of danger")
            utilities.saveimg(fig, f"stat_test_area_{popsize}")
            print()
            area_p_val = sum(permuted_data >= true_area_metric)/len(permuted_data)
            print("area_p_val:", area_p_val)

            df.append([popsize, num_smart,
                        true_area_metric, area_p_val])

    import pandas as pd
    df = pd.DataFrame(df, columns=colnames)
    df.to_csv(joinpath(config.DATA, "hungergames-results.csv"), index=False)


def violinplot_tgs_and_area_by_pop():
    records = []
    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in [5]:  # Adjust if needed
            files = _hungergames_files_for(popsize, num_smart)
            for f in files:
                try:
                    data = _read_hungergames_data(f)
                    rel_indices = list(range(num_smart))

                    tgs_d0, tgs_d1 = extract_groupsizes(data, rel_indices)
                    area_d0, area_d1 = extract_areas(data, rel_indices)

                    records.extend([
                        {"metric": "TGS", "type": "d_0", "value": tgs_d0, "pop_size": popsize},
                        {"metric": "TGS", "type": "d_1", "value": tgs_d1, "pop_size": popsize},
                        {"metric": "Area", "type": "d_0", "value": area_d0, "pop_size": popsize},
                        {"metric": "Area", "type": "d_1", "value": area_d1, "pop_size": popsize},
                    ])
                except Exception as e:
                    print(f"Skipping {f}: {e}")

    df = pd.DataFrame(records)

    # Plot setup
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    sns.set(style="whitegrid")

    for i, metric in enumerate(["TGS", "Area"]):
        ax = axs[i]
        sns.violinplot(
            data=df[df["metric"] == metric],
            x="pop_size", y="value", hue="type",
            ax=ax, inner="box", palette="pastel", cut=0, dodge=True
        )
        ax.set_title(metric)
        ax.set_xlabel("Population size")
        ax.set_ylabel(metric)
        ax.set_ylabel("")
        ax.legend(title="Type", labels=["$d_0$", "$d_1$"], fontsize="small", title_fontsize="small")

    plt.tight_layout()
    utilities.saveimg(fig, "vplot-hungergames")

def run_surroundedness_analysis(
    T_REL_MIN: int = 40,
    T_REL_MAX: int = 200,
    dbscan_fn=measurements.dbscan,
):
    """
    For each pop size and num_smart (currently [5]), load all hungergames data,
    subsample the same timestamps as in the area/size analyses, compute
    circumpolar-variance surroundedness for each individual, and collect
    surroundedness values separately for smart (IDs [0:num_smart]) and
    non-smart (IDs [num_smart:]) individuals.

    For each pop_size, produces a violin plot comparing the distributions of
    surroundedness for smart vs non-smart individuals.
    """
    dfs = []
    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in [5]:  # can extend this list later
            print(f"Analysing surroundedness: n={popsize}, d_1={num_smart}.")
            files = _hungergames_files_for(popsize, num_smart)
            alldata = [_read_hungergames_data(file_) for file_ in files]  # list of (N, 2, T)

            smart_surrounded = []
            nonsmart_surrounded = []

            for data in alldata:
                # data: (N, 2, T)
                N, _, T = data.shape

                # time subsampling as in extract_all_group_areas / extract_all_group_sizes
                for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
                    if t >= T:
                        break

                    positions_t = data[:, :, t]  # (N, 2)

                    # compute surroundedness at this time
                    surr = measurements.compute_surroundedness(positions_t, dbscan_fn=dbscan_fn)  # (N,)

                    # split smart vs non-smart, filter NaNs
                    smart_idx = np.arange(num_smart)
                    nonsmart_idx = np.arange(num_smart, N)

                    smart_vals = surr[smart_idx]
                    nonsmart_vals = surr[nonsmart_idx]

                    smart_vals = smart_vals[~np.isnan(smart_vals)]
                    nonsmart_vals = nonsmart_vals[~np.isnan(nonsmart_vals)]

                    if smart_vals.size > 0:
                        smart_surrounded.extend(smart_vals.tolist())
                    if nonsmart_vals.size > 0:
                        nonsmart_surrounded.extend(nonsmart_vals.tolist())

            # build DataFrame for this popsize / num_smart
            df = pd.DataFrame(
                {
                    "Surroundedness": smart_surrounded + nonsmart_surrounded,
                    "Type": (["Smart"] * len(smart_surrounded))
                            + (["Non-smart"] * len(nonsmart_surrounded)),
                }
            )
            df.loc[:, "popsize"] = popsize
            df.loc[:, "num_smart"] = num_smart
            dfs.append(df)

    df = pd.concat(dfs)
    print(df, "\n", df.shape)
    # violin plot comparing smart vs non-smart for this popsize
    fig, ax = plt.subplots()
#    sns.stripplot(data=df, x="Type", y="Surroundedness", hue="Type",
#    alpha=1.0, jitter=0.05, size=0.4, legend=False)
#    print("swarmplot done")
#    sns.boxplot(data=df, x="Surroundedness", y="popsize", hue="Type", width=0.4, gap=0)
#    print("boxplot done")
    sns.violinplot(
        data=df,
        x="popsize",
        y="Surroundedness",
        hue="Type",
        cut=0,
        inner="quartile",
        ax=ax,
        split=True
    )
#        ax.set_title(f"Surroundedness: n={popsize}, d_1={num_smart}")
    utilities.saveimg(fig, f"surroundedness_hungergames")
    print("figure saved")


def run_interior_probability_analysis(
    T_REL_MIN: int = 40,
    T_REL_MAX: int = 200,
    dbscan_fn=measurements.dbscan,
):
    """
    For each population size and num_smart (currently [5]), load all hungergames
    data, subsample timestamps as in earlier analyses, and estimate the
    probability that smart vs non-smart individuals are in the *group interior*.

    Interior is defined per timepoint as:
        - DBSCAN group of the focal,
        - group size >= 4 (otherwise hull has no interior),
        - group does NOT touch the edge (via group_touches_edge),
        - focal individual is NOT a vertex of that group's convex hull.

    For each popsize, this function:
        - tallies interior vs total eligible appearances for smart and non-smart,
        - estimates p_interior for each type,
        - and produces a simple bar plot (Smart vs Non-smart).
    """

    results = []  # to store summary numbers per (popsize, num_smart)

    for popsize in config.POP_S_SMART_GUYS_HG:
        for num_smart in [5]:  # extend this list if needed
            print(f"Analysing interior probability: n={popsize}, d_1={num_smart}.")
            files = _hungergames_files_for(popsize, num_smart)
            alldata = [_read_hungergames_data(file_) for file_ in files]  # list of (N, 2, T)

            smart_interior = 0
            smart_total = 0
            nonsmart_interior = 0
            nonsmart_total = 0

            for data in alldata:
                N, _, T = data.shape
                smart_idx = np.arange(num_smart)
                nonsmart_idx = np.arange(num_smart, N)

                # time subsampling as before
                for t in range(T_REL_MIN, T_REL_MAX + 1, 20):
                    if t >= T:
                        break

                    positions_t = data[:, :, t]  # (N, 2)
                    labels = dbscan_fn(positions_t)
                    touches = measurements.group_touches_edge(positions_t, labels)

                    # loop over groups
                    for label in np.unique(labels):
                        if label == -1:
                            continue  # noise

                        group_indices = np.where(labels == label)[0]
                        group_size = group_indices.size

                        if group_size < 10:
                            continue

                        # skip groups that touch the arena edge
                        if np.any(touches[group_indices]):
                            continue

                        pts = positions_t[group_indices]

                        try:
                            hull = ConvexHull(pts)
                        except QhullError:
                            # degenerate geometry: no well-defined hull interior
                            continue

                        hull_local = np.unique(hull.vertices)
                        hull_global = group_indices[hull_local]

                        # interior indices = group minus hull vertices
                        interior_global = np.setdiff1d(group_indices, hull_global, assume_unique=True)
                        if interior_global.size == 0:
                            # no interior individuals in this group
                            continue

                        # smart / non-smart counts within this group
                        smart_in_group = np.intersect1d(group_indices, smart_idx, assume_unique=True)
                        nonsmart_in_group = np.intersect1d(group_indices, nonsmart_idx, assume_unique=True)

                        # eligible appearances: all group members of each type
                        smart_total += smart_in_group.size
                        nonsmart_total += nonsmart_in_group.size

                        # interior appearances: intersection with interior set
                        smart_interior += np.intersect1d(interior_global, smart_in_group, assume_unique=True).size
                        nonsmart_interior += np.intersect1d(interior_global, nonsmart_in_group, assume_unique=True).size

            # estimate probabilities (guard against division by zero)
            p_smart = smart_interior / smart_total if smart_total > 0 else np.nan
            p_nonsmart = nonsmart_interior / nonsmart_total if nonsmart_total > 0 else np.nan

            print(f"Smart interior: {smart_interior}/{smart_total} -> p={p_smart:.3f}")
            print(f"Non-smart interior: {nonsmart_interior}/{nonsmart_total} -> p={p_nonsmart:.3f}")

            results.append({
                "popsize": popsize,
                "num_smart": num_smart,
                "p_smart_interior": p_smart,
                "p_nonsmart_interior": p_nonsmart,
                "smart_interior": smart_interior,
                "smart_total": smart_total,
                "nonsmart_interior": nonsmart_interior,
                "nonsmart_total": nonsmart_total,
            })

            # --- simple bar plot for this (popsize, num_smart) ---
            fig, ax = plt.subplots()
            ax.bar(
                ["Smart", "Non-smart"],
                [p_smart, p_nonsmart],
                color=["tab:blue", "tab:orange"],
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel("P(Interior | eligible)")
            ax.set_title(f"Interior probability: n={popsize}, d_1={num_smart}")
            utilities.saveimg(fig, f"interior_prob_n{popsize}_d{num_smart}")

    # optional: save summary table
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        joinpath(config.DATA, "hungergames-interior-probabilities.csv"),
        index=False,
    )


if __name__ == "__main__":
    #run_data_analysis()
    #violinplot_tgs_and_area_by_pop()
    run_surroundedness_analysis()
    run_interior_probability_analysis()
