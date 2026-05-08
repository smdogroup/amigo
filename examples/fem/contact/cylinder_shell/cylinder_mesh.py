# Author: Jack Turbush
# Description: Mesh generation script for the cylindrical shell contact example.

import gmsh

R = 0.5  # radius
L = 2.0  # length
lc = 0.05  # mesh size

gmsh.initialize()
gmsh.model.add("cylinder")

gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()

gmsh.write("cylinder.inp")
gmsh.write("cylinder.msh")
gmsh.finalize()
print("Done")
