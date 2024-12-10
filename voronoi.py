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

def mirror_unit_sq(locations):
    mirrored_locs = []
    mirrored_locs.append(mirror_left(locations))
    mirrored_locs.append(mirror_right(locations))
    mirrored_locs.append(mirror_top(locations))
    mirrored_locs.append(mirror_bottom(locations))

    mirrored_locs = np.vstack(mirrored_locs)
    return mirrored_locs

if __name__ == "__main__":
    locs = np.random.uniform(size=(20, 2))
    mirrored_locs = mirror_unit_sq(locs)

    plt.scatter(locs[:, 0], locs[:, 1], color="black", s=0.3)
    plt.scatter(mirrored_locs[:, 0], mirrored_locs[:, 1], color="gray", s=0.3)

    plt.axvline(0, linestyle="dotted", linewidth=0.3)
    plt.axvline(1, linestyle="dotted", linewidth=0.3)
    plt.axhline(0, linestyle="dotted", linewidth=0.3)
    plt.axhline(1, linestyle="dotted", linewidth=0.3)
    plt.show()
