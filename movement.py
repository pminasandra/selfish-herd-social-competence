# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides functions that determine movement rules for selfish agents. 
Includes simple gradient descent and recursive reasoning.
"""

import random

import numpy as np

import config
import voronoi

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

    # We first choose one direction in which to step, for grad computation.
    # This helps reduce computational load while preserving similar results.
    flip = random.random()
    if flip > 0.5:
        sign = -1.0
    else:
        sign = 1.0

    # find d/dx
    new_locs = locations.copy()
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

        # bound to inside of unit square:
        new_loc[0] = max(0.01, new_loc[0])
        new_loc[0] = min(0.99, new_loc[0])
        new_loc[1] = max(0.01, new_loc[1])
        new_loc[1] = min(0.99, new_loc[1])

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
    if isinstance(desired_depth, int):
        desired_depth = np.ones(locations.shape[0])*desired_depth

    # desired_depth == 0 -> normal gradient descent
    if desired_depth.max() == 0:
        return everyone_do_grad_descent(locations, vor)

    # if recursion has reached desired_depth:
    if desired_depth.max() == curr_depth:
        return locations


    # first do one recursion
    new_locs = everyone_do_grad_descent(locations, vor)
    new_updated_locs = []

    # everyone then asks themselves one question:
    for id_ in range(len(orig_locations)):
    # 'if everyone else were in these new locations,
    # where should I go?'

        # if current individual doesn't operate at or above current depth,
        # she doesn't update anything anymore
        if desired_depth[id_] < curr_depth:
            new_updated_locs.append(locations[id_])
        else:
            new_locs_with_me = new_locs.copy()
            new_locs_with_me[id_] = orig_locations[id_]#i.e., everyone updated but me.

            # then do the whole gradient descent business
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


def momentum_based_anticipatory_reasoning(locations, locations_before=None):
    """
    $\mu$ model. Performs anticipation not by using embedded models, instead uses
    movement from last step to extrapolate future positions.
    Args:
        locations (np.array, n×2)
        locations_before (np.array, n×2 | None): locations in last iteration. None if
            this is the first iteration.
    Returns:
        np.array, new locations, same shape as locations
    """

    if locations_before is None:
        future_locations = locations
    else:
        future_locations = locations + (locations - locations_before)

    # bound to inside of unit square:
    future_locations = np.clip(future_locations, 0.01, 0.99)

    orig_locations = locations.copy()

    new_updated_locs = []
    # everyone then asks themselves one question:
    for id_ in range(len(orig_locations)):
    # 'if everyone else were in these new locations,
    # where should I go?'

        future_locations_with_me = future_locations.copy()
        future_locations_with_me[id_] = orig_locations[id_]#i.e., everyone updated but me.

        # then do the whole gradient descent business
        new_vor = voronoi.get_bounded_voronoi(future_locations_with_me)
        areas_new = voronoi.get_areas(future_locations_with_me, new_vor)
        my_movement = -capped_grad(id_, future_locations_with_me, new_vor, areas_new)*\
                            config.GRAD_DESC_MULTPL_FACTOR
        my_future_location = future_locations_with_me[id_] + my_movement
        my_future_location[0] = max(0.01, my_future_location[0])
        my_future_location[0] = min(0.99, my_future_location[0])
        my_future_location[1] = max(0.01, my_future_location[1])
        my_future_location[1] = min(0.99, my_future_location[1])
        new_updated_locs.append(my_future_location)

    # after everyone has asked this question, store and return their new
    # movement decisions.
    return np.array(new_updated_locs)

if __name__ == "__main__":
    pass
