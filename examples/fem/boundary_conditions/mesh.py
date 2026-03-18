import gmsh
import sys

gmsh.initialize()
gmsh.model.add("plate")
lc = 0.01
L = 1.0  # length of a segment

# Geometry points rectangle
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(0.5 * L, 0, 0, lc, 2)
gmsh.model.geo.addPoint(L, 0, 0, lc, 3)
gmsh.model.geo.addPoint(L, 0.5 * L, 0, lc, 4)
gmsh.model.geo.addPoint(L, L, 0, lc, 5)
gmsh.model.geo.addPoint(0.5 * L, L, 0, lc, 6)
gmsh.model.geo.addPoint(0, L, 0, lc, 7)
gmsh.model.geo.addPoint(0, 0.5 * L, 0, lc, 8)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 1, 8)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("plate.inp")
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
