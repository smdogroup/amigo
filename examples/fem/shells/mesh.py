import numpy as np
import gmsh
import sys

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("plate")

# Mesh refinement
lc = 0.01

# Geometry definition
L = 1.0  # plate size

# Geometry points rectangle
gmsh.model.geo.addPoint(-0.5 * L, -0.5 * L, 0, lc, 1)
gmsh.model.geo.addPoint(0.5 * L, -0.5 * L, 0, lc, 2)
gmsh.model.geo.addPoint(0.5 * L, 0.5 * L, 0, lc, 3)
gmsh.model.geo.addPoint(-0.5 * L, 0.5 * L, 0, lc, 4)

# Define lines for the rectangle
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Define curve loop for the rectangle
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1, reorient=True)

# Define surface
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

gmsh.model.mesh.setRecombine(2, 1)

gmsh.model.mesh.generate(2)

gmsh.write("plate.inp")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
