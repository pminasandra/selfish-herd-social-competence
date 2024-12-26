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

def animate_data(data, tmin=0, tmax=None):

    if tmax is None:
        tmax = data.shape[2]
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))

    locs = data[:,:,0]
    sc = ax.scatter(locs[:,0], locs[:,1], s=0.4, color="black")

    def update(i):
        print(i, end="\033[K\r")
#            return 0
        locs = data[:,:,tmin+i]
#        vor = voronoi.get_bounded_voronoi(locs)
#        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.3, 
#                        line_alpha=0.2, show_points=False)

        sc.set_offsets(locs)
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        ax.axvline(0, linestyle="dotted", linewidth=0.3)
        ax.axvline(1, linestyle="dotted", linewidth=0.3)
        ax.axhline(0, linestyle="dotted", linewidth=0.3)
        ax.axhline(1, linestyle="dotted", linewidth=0.3)

        return ax,

    ani = FuncAnimation(fig, update, frames=tmax-tmin, interval=30, blit=False)
    return ani

def save_animation(anim, name):
    anim_dir = joinpath(config.FIGURES, "animations")
    os.makedirs(anim_dir, exist_ok=True)

    anim.save(joinpath(anim_dir, name), writer="ffmpeg")

if __name__ == "__main__":
    for i in [0, 1, 2]:
        tgt_file = measurements._files_for(100, i)
        tgt_file = list(tgt_file)[20]
        tgt_file = measurements._read_data(tgt_file)

        anim = animate_data(tgt_file, tmax=100)
        save_animation(anim, f"movement_100_d{i}.gif")
