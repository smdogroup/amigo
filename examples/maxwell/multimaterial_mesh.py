import numpy as np
import gmsh
import sys

gmsh.initialize()

gmsh.model.add("multi_material")

lc = 1e-1
lc1 = 2e-2

gmsh.model.geo.addPoint(0.0, 0.0, 0, lc, 1)
gmsh.model.geo.addPoint(0.0, 0.2, 0, lc1, 2)
gmsh.model.geo.addPoint(0.3, 0.2, 0, lc1, 3)
gmsh.model.geo.addPoint(0.3, 0.3, 0, lc1, 4)
gmsh.model.geo.addPoint(0.0, 0.3, 0, lc1, 5)
gmsh.model.geo.addPoint(0.0, 1.0, 0, lc, 6)
gmsh.model.geo.addPoint(1.0, 1.0, 0, lc, 7)
gmsh.model.geo.addPoint(1.0, 0.2, 0, lc, 8)
gmsh.model.geo.addPoint(1.0, 0.0, 0, lc, 9)

# Define lines that connect points.
# Last value is the label for the line
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 9, 8)
gmsh.model.geo.addLine(9, 1, 9)
gmsh.model.geo.addLine(3, 8, 10)

# define loops
gmsh.model.geo.addCurveLoop([1, 2, 10, 8, 9], 1)
gmsh.model.geo.addCurveLoop([3, 4, 5, 6, 7, -10], 2)

# define surfaces
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)

# required to call synchronize in order to be meshed
gmsh.model.geo.synchronize()

# generate 2d mesh
gmsh.model.mesh.generate(2)

gmsh.write("multimaterial.inp")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
