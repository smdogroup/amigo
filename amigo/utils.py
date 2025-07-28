import numpy as np
from scipy.sparse import csr_matrix

try:
    from petsc4py import PETSc
except:
    PETSc = None


def tocsr(mat):
    nrows, ncols, nnz, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()
    return csr_matrix((data, cols, rowp), shape=(nrows, ncols))


def topetsc(mat):
    if PETSc is None:
        return None

    # Extract the data from the matrix
    _, ncols, _, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()

    row_owners = mat.get_row_owners()
    col_owners = mat.get_column_owners()

    comm = row_owners.get_mpi_comm()
    nrows_local = row_owners.get_local_size()
    ncols_local = col_owners.get_local_size()

    A = PETSc.Mat().create(comm=comm)

    sr = (nrows_local, None)
    sc = (ncols_local, ncols)
    A.setSizes((sr, sc), bsize=1)
    A.setType(PETSc.Mat.Type.MPIAIJ)

    nnz_local = rowp[nrows_local]
    A.setValuesCSR(rowp[: nrows_local + 1], cols[:nnz_local], data[:nnz_local])
    A.assemble()

    return A
