# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

import random

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
    flip = random.random()

    new_locs = locations.copy()
    if flip > 0.5:
        sign = -1.0
    else:
        sign = 1.0

    new_locs[id_, 0] += sign*config.GRAD_DESC_DX

    vor_new = voronoi.get_bounded_voronoi(new_locs)
    areas_new = voronoi.get_areas(new_locs, vor_new)
    ddx_area = -(area_guy - areas_new[id_])/config.GRAD_DESC_DX*sign

    # find d/dy
    new_locs = locations.copy()
    new_locs[id_, 1] += sign*config.GRAD_DESC_DY

    vor_new = voronoi.get_bounded_voronoi(new_locs)
    areas_new = voronoi.get_areas(new_locs, vor_new)
    ddy_area = -(area_guy - areas_new[id_])/config.GRAD_DESC_DY*sign

    return np.array([ddx_area, ddy_area])


def capped_grad(id_, locations, vor, areas):
    """
    Computes capped gradient of voronoi area for one individual.
    Args:
        id_ (int): index of individual in question
        locations (np.array, n*2)
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
        locations (np.array, n*2)
        vor (scipy.spatial.Voronoi object)
    Returns:
        np.array, new locations, same shape as locations
    """

    areas = voronoi.get_areas(locations, vor)
    new_locs = []
    for id_ in range(len(locations)):
        movement = -capped_grad(id_, locations, vor,
                        areas)*config.GRAD_DESC_MULTPL_FACTOR
        new_loc = locations[id_, :] + movement
        new_loc[0] = max(0.01, new_loc[0])
        new_loc[0] = min(0.99, new_loc[0])
        new_loc[1] = max(0.01, new_loc[1])
        new_loc[1] = min(0.99, new_loc[1])
        new_locs.append(new_loc)

    return np.array(new_locs)


def one_guy_one_recursion(id_, my_loc, vor, predicted_locations):
    predicted_locations_with_me = predicted_locations.copy()
    predicted_locations_with_me[id_] = my_loc

    new_vor = voronoi.get_bounded_voronoi(predicted_locations_with_me)
    areas_new = voronoi.get_areas(predicted_locations_with_me, new_vor)
    movement = -capped_grad(id_, predicted_locations_with_me, 
                        new_vor, areas_new)*config.GRAD_DESC_MULTPL_FACTOR

    new_loc = my_loc + movement
    new_loc[0] = max(0.01, new_loc[0])
    new_loc[0] = min(0.99, new_loc[0])
    new_loc[1] = max(0.01, new_loc[1])
    new_loc[1] = min(0.99, new_loc[1])

    return new_loc

def group_one_recursion(locations, vor):
    predicted_locations = everyone_do_grad_descent(locations, vor)
    new_locs = []
    for id_ in range(len(locations)):
        my_loc = locations[id_, :]
        new_loc = one_guy_one_recursion(id_, my_loc, vor, predicted_locations)

        new_locs.append(new_loc)

    return np.array(new_locs)


def recursive_reasoning(locations, vor, desired_depth,
                        orig_locations, curr_depth=0):
    """
    Performs movement decisions with theory of mind for a desired depth of
    reasoning.
    Args:
        locations (np.array, n*2)
        vor (scipy.spatial.Voronoi object)
        desired_depth (int or array-like): how many recursions each animal will do.
        orig_locations (np.array, n*2): original locations without ANY
        modifications.
    Returns:
        np.array, new locations, same shape as locations
    """
# desired_depth == 0 -> normal gradient descent
    if desired_depth.max() == 0:
        return everyone_do_grad_descent(locations, vor)

    if isinstance(desired_depth, int):
        desired_depth = np.ones(locations.shape[0])*desired_depth

    if desired_depth.max() == curr_depth:
        return locations
    # first do one recursion
    new_locs = everyone_do_grad_descent(locations, vor)
    new_updated_locs = []
    for id_ in range(len(orig_locations)):
# then ask if everyone else were in these new locations,
# where should I go?
        if desired_depth[id_] < curr_depth:
            new_updated_locs.append(locations[id_])
        else:
            new_locs_with_me = new_locs.copy()
            new_locs_with_me[id_] = orig_locations[id_]

            new_vor = voronoi.get_bounded_voronoi(new_locs_with_me)
            areas_new = voronoi.get_areas(new_locs_with_me, new_vor)
            my_movement = -capped_grad(id_, new_locs_with_me, new_vor, areas_new)*\
                                config.GRAD_DESC_MULTPL_FACTOR
            my_new_loc = new_locs_with_me[id_] + my_movement
            my_new_loc[0] = max(0.01, my_new_loc[0])
            my_new_loc[0] = min(0.99, my_new_loc[0])
            my_new_loc[1] = max(0.01, my_new_loc[1])
            my_new_loc[1] = min(0.99, my_new_loc[1])
            new_updated_locs.append(my_new_loc)

# after everyone has asked this question, store their new
# movement decisions. Recurse on these new choices.
    new_updated_locs = np.array(new_updated_locs)
    new_vor = voronoi.get_bounded_voronoi(new_updated_locs)
    return recursive_reasoning(new_updated_locs, new_vor, desired_depth,
                                orig_locations, curr_depth=curr_depth+1)


if __name__ == "__main__":
    quit()
#    locs = np.random.uniform(size=(30, 2))
#    vor = voronoi.get_bounded_voronoi(locs)
#
#    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=200)
#    ax.set_xlim((-0.1, 1.1))
#    ax.set_ylim((-0.1, 1.1))
#    sc = ax.scatter(locs[:, 0], locs[:, 1], s=0.4)
#
#    def update(i):
#        print(i, end="\033[K\r")
#        global locs
#        global vor
#        global sc
#        locs = recursive_reasoning(locs, vor, 2, locs)
#        vor = voronoi.get_bounded_voronoi(locs)
#        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=0.3, 
#                        line_alpha=0.2, show_points=False)
#
#        sc.set_offsets(locs)
#        ax.set_xlim((-0.1, 1.1))
#        ax.set_ylim((-0.1, 1.1))
#        ax.axvline(0, linestyle="dotted", linewidth=0.3)
#        ax.axvline(1, linestyle="dotted", linewidth=0.3)
#        ax.axhline(0, linestyle="dotted", linewidth=0.3)
#        ax.axhline(1, linestyle="dotted", linewidth=0.3)
#
#    ani = FuncAnimation(fig, update, frames=200, interval=30)
#    plt.show()
#    ani.save("movement_d2.gif", writer="ffmpeg")
