# Pranav Minasandra
# Dec 16, 2024
# pminasandra.github.io

import random
from os.path import join as joinpath
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import config
import measurements
import voronoi

matplotlib.use("TkAgg")

def animate_data(datasets, tmin=0, tmax=None, delay=None):

    if tmax is None:
        tmax = datasets[0].shape[2]
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))

    scs = []
    locs = []
    for data in datasets:
        loc = data[:,:,tmin]
        locs.append(loc)
        scs.append(ax.scatter(loc[:,0], loc[:,1], s=0.3, alpha=0.6))

    if delay is None:
        delay = 0
    def update(i):
        if i <= delay*1000/50:
            return ax,
        print(i, end="\033[K\r")
#            return 0
        for j in range(len(datasets)):
            locs[j] = datasets[j][:,:,tmin + i - int(delay*1000/50)]
#        vor = voronoi.get_bounded_voronoi(locs)
#        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.3, 
#                        line_alpha=0.2, show_points=False)

        for sc, loc in zip(scs, locs):
            sc.set_offsets(loc)
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.axvline(0, linestyle="dotted", linewidth=0.3)
        ax.axvline(1, linestyle="dotted", linewidth=0.3)
        ax.axhline(0, linestyle="dotted", linewidth=0.3)
        ax.axhline(1, linestyle="dotted", linewidth=0.3)

        return ax,

    ani = FuncAnimation(fig, update,
                            frames=tmax-tmin+int(delay*1000/50),
                            interval=50,
                            blit=False)
    return ani

def animate_area_hists(datasets, tmin=0, tmax=None, delay=None):

    if tmax is None:
        tmax = len(datasets[0])
    fig, ax = plt.subplots(1, len(datasets), sharey=True, figsize=(4.0, 4.0), dpi=300)
    for i, a in enumerate(ax):
        a.set_xscale('log')
        a.set_xlabel('Area of Voronoi polygons')
        a.set_title(f'$d_{i}$')

    bins = np.logspace(np.log10(1e-8), np.log10(1),  50)
    hists = []
    j = 0
    for data in datasets:
        hists.append(ax[j].hist(data[0], bins=bins))
        ax[j].set_xlim((1e-8, 1))
        ax[j].set_ylim((0, 350))
        j += 1

    if delay is None:
        delay = 0
    def update(i):
        if i <= delay*1000/50:
            return ax,
        print(i, end="\033[K\r")
        for j in range(len(datasets)):
            hists[j] = datasets[j][i - int(delay*1000/50)]

        for hist_dat, a in zip(hists, ax):
            a.cla()
            a.hist(hist_dat, bins=bins)
            a.set_xlim((1e-8, 1))
            a.set_ylim((0, 350))
        for i, a in enumerate(ax):
            a.set_xscale('log')
            a.set_xlabel('Area of Voronoi polygons')
            a.set_title(f'$d_{i}$')

        return ax,

    ani = FuncAnimation(fig, update,
                    frames=tmax-tmin+int(delay*1000/50), 
                    interval=50, 
                    blit=False)
    return ani

def save_animation(anim, name):
    anim_dir = joinpath(config.FIGURES, "animations")
    os.makedirs(anim_dir, exist_ok=True)

    anim.save(joinpath(anim_dir, name), writer="ffmpeg")

def _extract_areas(dataset):

    areas = []
    _, _, k = dataset.shape
    for i in range(k):
        dataslice = dataset[:,:,i]
        vor = voronoi.get_bounded_voronoi(dataslice)
        areasslice = voronoi.get_areas(dataslice, vor)
        areas.append(areasslice)

    return areas

def _joinlists(lol1, lol2):
    lol_res = []
    for x, y in zip(lol1, lol2):
        lol_res.append(list(x)+list(y))

    return lol_res

if __name__ == "__main__":
    area_datasets = [[], []]
    for j in random.sample(range(500), 50):
        data = []
        for i in [0, 1]:
            tgt_file = measurements._files_for(50, i)
            tgt_file = list(tgt_file)[j]
            tgt_file = measurements._read_data(tgt_file)
            data.append(tgt_file)

        areas_data = []
        for dataset in data:
            areas_data.append(_extract_areas(dataset))


        if len(area_datasets[0]) == 0:
            area_datasets[0] = areas_data[0]
            area_datasets[1] = areas_data[1]
        else:
#            print(len(area_datasets[0][0]))
# this joining is fully improper
# plan this properly
            area_datasets[0] = _joinlists(area_datasets[0], areas_data[0])
            area_datasets[1] = _joinlists(area_datasets[1], areas_data[1])

    print("Initiating animations")
    anim = animate_area_hists(area_datasets, delay=0.5)
    save_animation(anim, f"hist_50.gif")
