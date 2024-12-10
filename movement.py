# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import voronoi_plot_2d

import config
import voronoi

"""
Provides functions that determine movement rules for selfish agents. 
Includes simple gradient descent and recursive reasoning.
"""


def gradient_for_id(id_, locations, vor, areas):
    """
    Computes raw gradient of voronoi area for one individual.
    Args:
        id_ (int): index of individual in question
        locations (np.array, n×2)
        vor (scipy.spatial.Voronoi object)
        areas (np.array): output from voronoi.get_areas(...)
    Returns:
        np.array (1×2): gradient of area for id_.
    """
    area_guy = areas[id_]

    # find d/dx
    new_locs = locations.copy()
    new_locs[id_, 0] += config.GRAD_DESC_DX

    vor_new = voronoi.get_bounded_voronoi(new_locs)
    areas_new = voronoi.get_areas(new_locs, vor_new)
    ddx_area = -(area_guy - areas_new[id_])/config.GRAD_DESC_DX

    # find d/dy
    new_locs = locations.copy()
    new_locs[id_, 1] += config.GRAD_DESC_DY

    vor_new = voronoi.get_bounded_voronoi(new_locs)
    areas_new = voronoi.get_areas(new_locs, vor_new)
    ddy_area = -(area_guy - areas_new[id_])/config.GRAD_DESC_DY

    return np.array([ddx_area, ddy_area])


def capped_grad(id_, locations, vor, areas):
    """
    Computes capped gradient of voronoi area for one individual.
    Args:
        id_ (int): index of individual in question
        locations (np.array, n×2)
        vor (scipy.spatial.Voronoi object)
        areas (np.array): output from voronoi.get_areas(...)
    Returns:
        np.array (1×2): capped gradient of area for id_.
    """

    raw_grad = gradient_for_id(id_, locations, vor, areas)
    norm = (raw_grad[0]**2 + raw_grad[1]**2)**0.5

    if norm > config.GRAD_DESC_MAX_STEP_SIZE:
        raw_grad *= (config.GRAD_DESC_MAX_STEP_SIZE / norm)

    return raw_grad

def everyone_do_grad_descent(locations, vor):
    """
    Performs one iteration of gradient descent with all individuals.
    Args:
        locations (np.array, n×2)
        vor (scipy.spatial.Voronoi object)
    Returns:
        np.array, new locations, same shape as locations
    """

    areas = voronoi.get_areas(locations, vor)
    new_locs = []
    for id_ in range(len(locations)):
        movement = -capped_grad(id_, locations, vor, areas)*config.GRAD_DESC_MULTPL_FACTOR
        new_loc = locations[id_, :] + movement
        new_loc[0] = max(0, new_loc[0])
        new_loc[0] = min(1, new_loc[0])
        new_loc[1] = max(0, new_loc[1])
        new_loc[1] = min(1, new_loc[1])
        new_locs.append(new_loc)

    return np.array(new_locs)


if __name__ == "__main__":
    locs = np.random.uniform(size=(100, 2))
    vor = voronoi.get_bounded_voronoi(locs)

    fig, ax = plt.subplots()
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))

    def update(i):
        global locs
        global vor
        ax.clear()
        locs = everyone_do_grad_descent(locs, vor)
        vor = voronoi.get_bounded_voronoi(locs)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.3, 
                        line_alpha=0.2, show_points=False)

        ax.scatter(locs[:, 0], locs[:, 1], s=0.4)
        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
#        ax.axvline(0, linestyle="dotted", linewidth=0.3)
#        ax.axvline(1, linestyle="dotted", linewidth=0.3)
#        ax.axhline(0, linestyle="dotted", linewidth=0.3)
#        ax.axhline(1, linestyle="dotted", linewidth=0.3)

    ani = FuncAnimation(fig, update, frames=2000, interval=30)
    ani.save("movement.mp4", writer="ffmpeg")
