# Pranav Minasandra and Cecilia Baldoni
# Dec 13, 2024
# pminasandra.github.io

import datetime as dt
import glob
from os.path import join as joinpath
from os.path import basename
import multiprocessing as mp
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt

import analyses
import config
import hungergames
import measurements
import selfishherd
import utilities

def runmodel(herd, filename):
    """
    parallelization helper function
    """
    np.random.seed()
    herd.run(config.TMAX)
    herd.savedata(filename)

def plot_edge_proportion(data):
    """
    Plot mean ± standard error of the proportion of population at edge.

    Args:
        data (dict): Maps integers to DataFrames with columns
                     'uname', 't0', 't20', ..., 't500'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_order = [0, 1, 2, 3, -1]

    for key in plot_order:
        if key not in data:
            continue

        df = data[key]

        cols = sorted(
            (c for c in df.columns if c.startswith("t") and c[1:].isdigit()),
            key=lambda c: int(c[1:])
        )
        times = np.array([int(c[1:]) for c in cols])

        mean = df[cols].mean(axis=0).to_numpy()
        sem = df[cols].sem(axis=0).to_numpy()

        label = r"$d_\mu$" if key == -1 else rf"$d_{{{key}}}$"

        line, = ax.plot(times, mean, label=label, linewidth=2)
        ax.fill_between(
            times, mean - sem, mean + sem,
            color=line.get_color(), alpha=0.2
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Proportion of population at edge")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    return fig, ax

if __name__ == "__main__":
    POP_SIZES = list(config.POP_S_DOR.keys())

    if config.RUN_SIMS:
        depth_dirs = []
        for pop_size in POP_SIZES:
            depth_dirs.extend([joinpath(config.DATA, str(pop_size),
                                f"d{depth}")\
                            for depth in config.POP_S_DOR[pop_size]])
        [os.makedirs(dir_, exist_ok=True) for dir_ in depth_dirs]

        for pop_size in POP_SIZES:
            print("Working on pop_size", pop_size)
            # if some data already exists, account for that
            existing_files = measurements._files_for(pop_size, 0)
            if len(list(existing_files)) == 0:
                inits = [np.random.uniform(size=(pop_size, 2))\
                            for i in range(config.NUM_REPEATS)]
                init_names = [str(uuid.uuid4())\
                                for i in range(config.NUM_REPEATS)]
            else:
                print("Already found", len(list(existing_files)), "files.")
                inits = [measurements._read_data(filename)[:,:,0]\
                            for filename in existing_files]
                init_names = ["-".join(basename(f)[:-len(".pkl")].split("-")[2:])\
                                for f in existing_files]

            for depth in config.POP_S_DOR[pop_size]:
                print(dt.datetime.now(), "Depth of reasoning:", depth)
                herds = [selfishherd.SelfishHerd(pop_size, depth, loc) for loc\
                            in inits]
                filenames = [joinpath(config.DATA, str(pop_size), f"d{depth}",
                                    f"{pop_size}-{depth}-{uname}.pkl")\
                                    for uname in init_names]
                args = zip(herds, filenames)
                
                pool = mp.Pool()
                pool.starmap(runmodel, args)
                pool.close()
                pool.join()
            print()

    if config.CONDUCT_HUNGERGAMES:
        os.makedirs(joinpath(config.DATA, "HungerGames"), exist_ok=True)
        hungergames.simulate_all_hungergames()


    if config.ANALYSE_HUNGERGAMES:
        hungergames.run_all_analyses()


    if config.ANALYSE_DATA:
        group_metrics = []
        timerange = range(0, 501, 20)
        os.makedirs(joinpath(config.DATA, "Results"), exist_ok=True)
        for pop_size in config.ANALYSE_POP_SIZES:
            all_edge_data = {}
            for depth in config.ANALYSE_DEPTHS:
                tgs_file = joinpath(config.DATA, "Results",
                                        f"{pop_size}-d{depth}.csv")
                area_file = joinpath(config.DATA, "Results",
                                        f"areas-{pop_size}-d{depth}.csv")
                
                edge_data = measurements.make_edgeeffect_csv_for(pop_size, depth,
                                    timerange, eps=0.02)
                edge_data.to_csv(tgs_file, index=False)
                all_edge_data[depth] = edge_data
            fig, ax = plot_edge_proportion(all_edge_data)
            utilities.saveimg(fig, f"edge-effect-graph-{pop_size}")


# TODO: plot edge data again
#        fig, ax = plt.subplots(figsize=(11.45, 4.921))
#        analyses.make_violinplot(measurements.extract_polarisations_exclude_edge,
#                                    ydesc="polarisation",
#                                    fig=fig,
#                                    ax=ax,
#                                    palette="pastel"
#                                )
#        utilities.saveimg(fig, "vplot-polarisations")
#        analyses.compare_gpsize_area_relation()
#        analyses.plot_group_size_ccdfs_displot()
