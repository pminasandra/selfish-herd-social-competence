# Pranav Minasandra and Cecilia Baldoni
# pminasandra.github.io
# Dec 10, 2024

"""
Provides methods to construct Voronoi polygons for groups of simulated animals
and compute their areas.
"""

def mirror_bottom(y_old):
    y_new = -y_old
    return y_new

def mirror_top(y_old):
    y_new = 2-y_old
    return y_new

def mirror_left(x_old):
    x_new = -x_old
    return x_new

def mirror_right(x_old):
    x_new = 2-x_old
    return x_new

print(mirror_bottom(0.2))  
print(mirror_top(0.8))
print(mirror_left(0.3))
print(mirror_right(0.7))