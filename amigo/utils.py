import numpy as np
from scipy.sparse import csr_matrix


def tocsr(mat):
    nrows, ncols, nnz, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()
    return csr_matrix((data, cols, rowp), shape=(nrows, ncols))
