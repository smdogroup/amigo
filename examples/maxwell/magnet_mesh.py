import gmsh
import numpy as np
import sys


def check_areas(X, conn):
    # Reshape arrays
    conn = conn.reshape(-1, 3)
    X = X.reshape(-1, 3)

    # Check element area
    for i in range(len(elemTags)):
        # Zero indexing required
        n1 = conn[i, 0] - 1
        n2 = conn[i, 1] - 1
        n3 = conn[i, 2] - 1

        n1_x = X[n1, 0]
        n1_y = X[n1, 1]

        n2_x = X[n2, 0]
        n2_y = X[n2, 1]

        n3_x = X[n3, 0]
        n3_y = X[n3, 1]

        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

        if a_e <= 0.0:
            raise Exception(f"Element area for element {i} = {a_e}")


gmsh.initialize()
gmsh.model.add("magnet")

# Mesh refinement at nodes
lc = 5e-2
lc1 = 3e-2

# Geometry dimentions
hb = 5  # Boundary length and width
hm = 1  # Magnet length and width

# Add points
geom = gmsh.model.geo
geom.addPoint(hb, -hb, 0, lc, 1)
geom.addPoint(hb, hb, 0, lc, 2)
geom.addPoint(-hb, hb, 0, lc, 3)
geom.addPoint(-hb, -hb, 0, lc, 4)
geom.addPoint(hm, -hm, 0, lc1, 5)
geom.addPoint(hm, hm, 0, lc1, 6)
geom.addPoint(-hm, hm, 0, lc1, 7)
geom.addPoint(-hm, -hm, 0, lc1, 8)

# Define lines for the boundary loop
geom.addLine(1, 2, 1)
geom.addLine(2, 3, 2)
geom.addLine(3, 4, 3)
geom.addLine(4, 1, 4)

# Define lines for the magnet loop
geom.addLine(5, 6, 5)
geom.addLine(6, 7, 6)
geom.addLine(7, 8, 7)
geom.addLine(8, 5, 8)

# Define curve loops
geom.addCurveLoop([1, 2, 3, 4], 1, reorient=False)
geom.addCurveLoop([5, 6, 7, 8], 2, reorient=False)

# Define surfaces
geom.addPlaneSurface([1, 2], 1)
geom.addPlaneSurface([2], 2)

# Required to call synchronize in order to be meshed
gmsh.model.geo.synchronize()

# Generate 2d mesh
gmsh.model.mesh.generate(2)

# Check the areas to make sure elements are not flipped
nodeTags, X, _ = gmsh.model.mesh.getNodes(-1, -1)
elementType = gmsh.model.mesh.getElementType("Triangle", 1)
elemTags, conn = gmsh.model.mesh.getElementsByType(elementType)
check_areas(X, conn)

# Save the mesh
gmsh.write("magnet.inp")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
