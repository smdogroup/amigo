# Author: Jack Turbush
# Description: Point generation script for the lift-plus-cruise contact mesh.

import numpy as np

center = [1500, 1556, 1612, 1668, 1724, 1780, 1836, 1892, 1948, 2004]

full = []

for val in center:
    full.extend(list(range(val - 5, val + 5)))

print(full)
