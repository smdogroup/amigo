import gmsh
import sys

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("poisson")

# Mesh refinement
lc = 10.0

# Geometry definition
L = 3.0  # plate size
r = 0.5  # hole radius

# Geometry points rectangle
gmsh.model.geo.addPoint(-0.5 * L, -0.5 * L, 0, lc, 1)
gmsh.model.geo.addPoint(0.5 * L, -0.5 * L, 0, lc, 2)
gmsh.model.geo.addPoint(0.5 * L, 0.5 * L, 0, lc, 3)
gmsh.model.geo.addPoint(-0.5 * L, 0.5 * L, 0, lc, 4)

gmsh.model.geo.addPoint(0, 0, 0, lc, 5)
gmsh.model.geo.addPoint(r, 0, 0, lc, 6)
gmsh.model.geo.addPoint(0, r, 0, lc, 7)
gmsh.model.geo.addPoint(-r, 0, 0, lc, 8)
gmsh.model.geo.addPoint(0, -r, 0, lc, 9)

# Define lines for the rectangle
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Define lines for the circle
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)

# Define curve loop for the rectangle
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1, reorient=True)

# Define curve loop for the circle
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2, reorient=True)

# Define surface and holes
gmsh.model.geo.addPlaneSurface([1, 2], 1)
gmsh.model.geo.addPlaneSurface([2], 2)

gmsh.model.geo.synchronize()

# At this level, Gmsh knows everything to display the rectangular surface 1 and
# to mesh it. An optional step is needed if we want to group elementary
# geometrical entities into more meaningful groups, e.g. to define some
# mathematical ("domain", "boundary"), functional ("left wing", "fuselage") or
# material ("steel", "carbon") properties.
#
# Such groups are called "Physical Groups" in Gmsh. By default, if physical
# groups are defined, Gmsh will export in output files only mesh elements that
# belong to at least one physical group. (To force Gmsh to save all elements,
# whether they belong to physical groups or not, set the `Mesh.SaveAll' option
# to 1.) Physical groups are also identified by tags, i.e. stricly positive
# integers, that should be unique per dimension (0D, 1D, 2D or 3D). Physical
# groups can also be given names.
#
# Here we define a physical curve that groups the left, bottom and right curves
# in a single group (with prescribed tag 5); and a physical surface with name
# "My surface" (with an automatic tag) containing the geometrical surface 1:
gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], name="dirichlet")
gmsh.model.addPhysicalGroup(2, [1], name="air")
gmsh.model.addPhysicalGroup(2, [2], name="metal")

# We can then generate a 2D mesh...
gmsh.model.mesh.generate(2)

# # Define variables that stores the dirichlet boundary condition node tags
# dirichletBcNodeTags = []

# # Define the variable that stores the element tags for air
# elemTags_air = []

# # Define the variable that stores the element tags for metal
# elemTags_metal = []

# # Extract mesh information
# phys_groups = gmsh.model.getPhysicalGroups()
# for dim, pg_tag in phys_groups:
#     name = gmsh.model.getPhysicalName(dim=dim, tag=pg_tag)
#     print(f"\nPhysical group '{name}' (dim={dim}, tag={pg_tag})")

#     # Get the entities belonging to this physical group (mesh tags that belong to the group)
#     entities = gmsh.model.getEntitiesForPhysicalGroup(dim=dim, tag=pg_tag)
#     print("\tentities:", entities)

#     for e in entities:
#         # Get elements in this entity
#         elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim=dim, tag=e)
#         for etype, etags, ntags in zip(elem_types, elem_tags, node_tags):
#             print(f"\tElement type {etype}:")
#             print(f"\telement tags: {etags}")
#             print(f"\tnode tags: {ntags}")
#             if name == "dirichlet":
#                 dirichletBcNodeTags.append(ntags)
#             elif name == "air":
#                 elemTags_air.append(etags)
#             elif name == "metal":
#                 elemTags_metal.append(etags)

# Get the global node coordinates for the entire mesh
nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim=-1, tag=-1)

# Reshape the nodeCoords to (-1, 3) -> x, y, z collumns and node rows
nodeCoords = nodeCoords.reshape(-1, 3)

# Get the global element node tags for the entire mesh
# (element type 2 = 3 Node Triangles)
elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elementType=2)

# Reshape the elemNodeTags to (-1, 3) -> n1, n2, n3 collumns, row = element number
elemNodeTags = elemNodeTags.reshape(-1, 3)
elemNodeTags -= 1  # Switch to zero based indexing

# Flip elements that are negative
for row in range(len(elemTags)):
    # Extract the node tags
    n1 = elemNodeTags[row, 0]
    n2 = elemNodeTags[row, 1]
    n3 = elemNodeTags[row, 2]

    # Node 1 coord
    n1_x = nodeCoords[n1, 0]
    n1_y = nodeCoords[n1, 1]

    # Node 2 coord
    n2_x = nodeCoords[n2, 0]
    n2_y = nodeCoords[n2, 1]

    # Node 3 coord
    n3_x = nodeCoords[n3, 0]
    n3_y = nodeCoords[n3, 1]

    # Element area
    a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))

    # Swap nodes if element area is negative
    if a_e < 0.0:
        elemNodeTags[row, 0] = n3
        elemNodeTags[row, 1] = n2
        elemNodeTags[row, 2] = n1

# To visualize the model we can run the graphical user interface with
# `gmsh.fltk.run()'. Here we run it only if "-nopopup" is not provided in the
# command line arguments:
# if "-nopopup" not in sys.argv:
#     gmsh.fltk.run()
gmsh.finalize()
