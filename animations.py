# Pranav Minasandra
# Dec 16, 2024
# pminasandra.github.io

from os.path import join as joinpath
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import config
import measurements

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
        loc = data[:,:,0]
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

    ani = FuncAnimation(fig, update, frames=tmax-tmin+int(delay*1000/50), interval=50, blit=False)
    return ani

def save_animation(anim, name):
    anim_dir = joinpath(config.FIGURES, "animations")
    os.makedirs(anim_dir, exist_ok=True)

    anim.save(joinpath(anim_dir, name), writer="ffmpeg")

if __name__ == "__main__":
    data = []
    for i in [0, 1]:
        tgt_file = measurements._files_for(75, i)
        tgt_file = list(tgt_file)[50]
        tgt_file = measurements._read_data(tgt_file)
        data.append(tgt_file)

    anim = animate_data(data, tmax=100, delay=1)
    save_animation(anim, f"movement_75.gif")
