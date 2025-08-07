# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides methods to construct Voronoi polygons for groups of simulated animals
and compute their areas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def polygon_area(vertices):
    """Calculate the area of a polygon given its vertices."""
    x, y = vertices[:, 0], vertices[:, 1]

    # shoe-lace formula follows
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# The next five functions will be used to construct dummy points by mirroring real
# points about the edges of the unique square

def _mirror_bottom(locations):
    new_locs = locations.copy()
    new_locs[:, 1] *= -1.0
    return new_locs

def _mirror_top(locations):
    new_locs = locations.copy()
    new_locs[:, 1] *= -1.0
    new_locs[:, 1] += 2.0
    return new_locs

def _mirror_left(locations):
    new_locs = locations.copy()
    new_locs[:, 0] *= -1.0
    return new_locs

def _mirror_right(locations):
    new_locs = locations.copy()
    new_locs[:, 0] *= -1.0
    new_locs[:, 0] += 2.0
    return new_locs

def _mirror_unit_sq(locations):
    mirrored_locs = []
    mirrored_locs.append(_mirror_left(locations))
    mirrored_locs.append(_mirror_right(locations))
    mirrored_locs.append(_mirror_top(locations))
    mirrored_locs.append(_mirror_bottom(locations))

    mirrored_locs = np.vstack(mirrored_locs)
    return mirrored_locs


# Following functions generate voronoi tesselations with bounds
# and compute the appropriate areas

def get_bounded_voronoi(locations):
    """
    Uses scipy.spatial.Voronoi to compute voronoi tesselation AFTER
    first mirroring points about unit sq.

    Args:
        locations (np.ndarray, n×2)
    Returns:
        scipy.spatial._qhull.Voronoi
    """
    mirr_locs = _mirror_unit_sq(locations)
    pseudolocs = np.vstack((locations, mirr_locs))

    return Voronoi(pseudolocs)

def get_areas(locations, voronoi):
    """
    Computes the areas of voronoi polygons for REAL points.
    Args:
        locations (np.ndarray, n×2)
        voronoi: typically output from get_bounded_voronoi(...)
    Returns:
        np.array (len(locations)): areas of polygons
    Raises:
        ValueError (if an infinite polygon is somehow seen)
    """
    num_loc = locations.shape[0]
    poly_areas = []
    curr_loc = 0
    for region_idx in voronoi.point_region:
        if curr_loc >= num_loc:
            return np.array(poly_areas)
        region = voronoi.regions[region_idx]
        if -1 in region:
            raise ValueError("somehow encountered an infinite Voronoi polygon!")
        polygon = voronoi.vertices[region]
        poly_areas.append(polygon_area(polygon))
        curr_loc += 1

if __name__ == "__main__":
    locs = np.random.uniform(size=(10, 2))

    vor  = get_bounded_voronoi(locs)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, linewidth=0.2, show_points=False)

    areas = get_areas(locs, vor)
    ax.scatter(locs[:, 0], locs[:, 1], s=0.4, color="black")
    ax.axvline(0, linestyle="dotted", linewidth=0.3)
    ax.axvline(1, linestyle="dotted", linewidth=0.3)
    ax.axhline(0, linestyle="dotted", linewidth=0.3)
    ax.axhline(1, linestyle="dotted", linewidth=0.3)
    plt.show()
