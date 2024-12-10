# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

import numpy as np
import matplotlib.pyplot as plt

"""
Provides methods to construct Voronoi polygons for groups of simulated animals
and compute their areas.
"""

def mirror_bottom(locations):
    new_locs = locations.copy()
    new_locs[:, 1] *= -1.0
    return new_locs

def mirror_top(locations):
    new_locs = locations.copy()
    new_locs[:, 1] *= -1.0
    new_locs[:, 1] += 2.0
    return new_locs

def mirror_left(locations):
    new_locs = locations.copy()
    new_locs[:, 0] *= -1.0
    return new_locs

def mirror_right(locations):
    new_locs = locations.copy()
    new_locs[:, 0] *= -1.0
    new_locs[:, 0] += 2.0
    return new_locs

if __name__ == "__main__":
    locs = np.random.uniform(size=(20, 2))
    bottoms = mirror_bottom(locs)
    tops = mirror_top(locs)
    lefts = mirror_left(locs)
    rights = mirror_right(locs)

    plt.scatter(locs[:, 0], locs[:, 1], color="black", s=0.3)
    plt.scatter(bottoms[:, 0], bottoms[:, 1], color="gray", s=0.3)
    plt.scatter(tops[:, 0], tops[:, 1], color="gray", s=0.3)
    plt.scatter(lefts[:, 0], lefts[:, 1], color="gray", s=0.3)
    plt.scatter(rights[:, 0], rights[:, 1], color="gray", s=0.3)

    plt.axvline(0, linestyle="dotted")
    plt.axvline(1, linestyle="dotted")
    plt.axhline(0, linestyle="dotted")
    plt.axhline(1, linestyle="dotted")
    plt.show()
