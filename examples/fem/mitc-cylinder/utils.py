def write_vtu(mesh, conns, u, v, w, filename="cylinder_shell.vtu"):
    import numpy as np

    X = mesh.X
    nnodes = X.shape[0]
    conn = np.vstack(conns) if isinstance(conns, list) else conns
    nelems = conn.shape[0]

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        f.write("<UnstructuredGrid>\n")
        f.write(f'<Piece NumberOfPoints="{nnodes}" NumberOfCells="{nelems}">\n')

        # Points
        f.write(
            '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        )
        for i in range(nnodes):
            f.write(f"{X[i,0]} {X[i,1]} {X[i,2]}\n")
        f.write("</DataArray></Points>\n")

        # Cells (VTK type 9 = quad)
        f.write("<Cells>\n")
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for row in conn:
            f.write(" ".join(str(n) for n in row) + "\n")
        f.write("</DataArray>\n")
        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n')
        for i in range(1, nelems + 1):
            f.write(f"{i*4}\n")
        f.write("</DataArray>\n")
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(("9\n") * nelems)
        f.write("</DataArray>\n")
        f.write("</Cells>\n")

        # Point data
        f.write("<PointData>\n")
        for name, arr in [("u", u), ("v", v), ("w", w)]:
            f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
            f.write("\n".join(str(val) for val in arr) + "\n")
            f.write("</DataArray>\n")
        # displacement vector for Warp by Vector filter
        f.write(
            '<DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">\n'
        )
        for i in range(nnodes):
            f.write(f"{u[i]} {v[i]} {w[i]}\n")
        f.write("</DataArray>\n")
        f.write("</PointData>\n")

        f.write("</Piece></UnstructuredGrid></VTKFile>\n")
