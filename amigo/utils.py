import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import BSpline
from .component import Component

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


class BSplineInterpolant(Component):
    def __init__(
        self,
        xi=None,
        npts: int = 0,
        length: float = 1.0,
        k: int = 4,
        n: int = 10,
        deriv: int = 0,
    ):
        """
        Perform a BSpline interpolation from a fixed set of parameter points. If the parameter points
        change as a function of the design variables, you cannot use this method.

        res = output - N(xi) * input

        Args:
            npts (int) : Number of evaluation points
            k (int) : Order of the bspline polynomial (degree + 1)
            n (int) : Number of interpolating points
            deriv (int) : Number of derivatives to compute
        """
        super().__init__()

        # Set the order of the bspline
        self.k = k

        # Set the number of inputs
        self.n = n

        # Set the order of the derivative
        self.deriv = deriv

        # Set the length of the interval
        self.length = length

        # Set the interpolation points
        if xi is None:
            self.npts = npts
            self.xi = np.linspace(0, self.length, self.npts)
        else:
            self.xi = xi
            self.npts = len(xi)

        # Set the size of the input data
        self.add_data("N", shape=self.k)

        # Set the input values to be interpolated
        self.add_input("input", shape=self.k)

        # Set the output values
        self.add_input("output")

        # Set the coupling constraint
        self.add_constraint("res")

        return

    def compute(self):
        """Compute the derivative outputs at the constant set of knots"""
        N = self.data["N"]
        input = self.inputs["input"]
        output = self.inputs["output"]

        value = 0.0
        for i in range(self.k):
            value = value + N[i] * input[i]
        self.constraints["res"] = output - value

        return

    def compute_knots(self):
        """
        Compute the BSpline knots for interpolation
        """

        # Set the knot locations
        t = np.zeros(self.n + self.k)
        t[: self.k] = 0.0
        t[-self.k : :] = self.length
        t[self.k - 1 : -self.k + 1] = np.linspace(0, self.length, self.n - self.k + 2)

        return t

    def compute_basis(self):
        """
        Compute the BSpline basis functions at the given knot locations
        """

        t = self.compute_knots()

        # Special case for right endpoint
        xi_clip = np.clip(self.xi, t[0], t[-1])

        # Find span index i such that t[i] <= x < t[i+1]
        span = np.searchsorted(t, xi_clip, side="right") - 1
        span = np.clip(span, self.k - 1, self.n - 1)

        N = np.zeros((self.npts, self.k), dtype=float)

        # Evaluate using the Cox–de Boor recursion for each point
        for j in range(self.xi.size):
            i = span[j]
            # zeroth-degree basis
            N_j = np.zeros(self.k)
            N_j[0] = 1.0

            for d in range(1, self.k):
                saved = 0.0
                for r in range(d):
                    left = t[i - d + 1 + r]
                    right = t[i + 1 + r]
                    denom = right - left
                    temp = 0.0 if denom == 0.0 else N_j[r] / denom
                    N_j[r] = saved + (right - xi_clip[j]) * temp
                    saved = (xi_clip[j] - left) * temp
                N_j[d] = saved

            N[j, :] = N_j

        # Handle x == t[-1]
        last_mask = self.xi == t[-1]
        if np.any(last_mask):
            N[last_mask, :] = 0.0
            N[last_mask, -1] = 1.0

        return N

    def compute_basis_derivative(self, deriv=1):
        t = self.compute_knots()

        # Special case for right endpoint
        xi_clamped = np.clip(self.xi, t[0], t[-1])

        # Find span index i such that t[i] <= x < t[i+1]
        span = np.searchsorted(t, xi_clamped, side="right") - 1
        span = np.clip(span, self.k - 1, self.n - 1)

        Nd = np.zeros((self.npts, self.k), dtype=float)

        # Evaluate using the Cox–de Boor recursion for each point
        for j in range(self.npts):
            i = span[j]
            x = xi_clamped[j]

            # Build the ndu table (basis values in the last column, denominators below diagonal)
            ndu = np.zeros((self.k, self.k), dtype=float)
            left = np.zeros(self.k, dtype=float)
            right = np.zeros(self.k, dtype=float)
            ndu[0, 0] = 1.0

            for d in range(1, self.k):
                left[d] = x - t[i + 1 - d]
                right[d] = t[i + d] - x
                saved = 0.0
                for r in range(d):
                    denom = right[r + 1] + left[d - r]
                    temp = 0.0 if denom == 0.0 else ndu[r, d - 1] / denom
                    ndu[d, r] = denom  # store denominators (used later)
                    ndu[r, d] = saved + right[r + 1] * temp  # basis value
                    saved = left[d - r] * temp
                ndu[d, d] = saved

            # ders[0, :] are the basis values; ders[1, :] first deriv; ders[2, :] second, etc.
            ders = np.zeros((deriv + 1, self.k), dtype=float)
            for r in range(self.k):
                ders[0, r] = ndu[r, self.k - 1]

            # Workspace for derivative recursion
            a = np.zeros((2, self.k), dtype=float)

            # Compute derivatives up to nd
            for r in range(self.k):
                a[0, 0] = 1.0
                s1, s2 = 0, 1
                for kk in range(1, deriv + 1):
                    dval = 0.0
                    rk = r - kk
                    pk = self.k - 1 - kk

                    if r >= kk:
                        denom = ndu[pk + 1, rk]
                        aval = 0.0 if denom == 0.0 else a[s1, 0] / denom
                        a[s2, 0] = aval
                        dval += aval * ndu[rk, pk]

                    j1 = 1 if rk >= -1 else -rk
                    j2 = kk - 1 if (r - 1) <= pk else p - r
                    for jj in range(j1, j2 + 1):
                        denom = ndu[pk + 1, rk + jj]
                        aval = (
                            0.0 if denom == 0.0 else (a[s1, jj] - a[s1, jj - 1]) / denom
                        )
                        a[s2, jj] = aval
                        dval += aval * ndu[rk + jj, pk]

                    if r <= pk:
                        denom = ndu[pk + 1, r]
                        aval = 0.0 if denom == 0.0 else -a[s1, kk - 1] / denom
                        a[s2, kk] = aval
                        dval += aval * ndu[r, pk]

                    ders[kk, r] = dval
                    s1, s2 = s2, s1

            # Scale derivatives by falling factorial p*(p-1)*...*(p-kk+1)
            if deriv > 0:
                fall = np.cumprod([self.k - 1 - ii for ii in range(deriv)], dtype=float)
                for kk in range(1, deriv + 1):
                    ders[kk, :] *= fall[kk - 1]

            Nd[j, :] = ders[deriv, :]

        return Nd

    def set_data(self, name, data):
        """
        Set the interpolation data for the component group
        """

        # Set the locations for where to evaluate the bspline basis
        if self.deriv == 0:
            data[f"{name}.N"] = self.compute_basis()
        else:
            data[f"{name}.N"] = self.compute_basis_derivative(deriv=self.deriv)

        return

    def add_links(self, name, model, src_name):
        """
        Add the links to the model for the interpolation
        """

        # Set the knot locations
        t = self.compute_knots()

        index = self.k - 1
        for i in range(self.npts):
            if self.xi[i] < t[self.k - 1]:
                index = self.k - 1
            elif self.xi[i] > t[self.n - 1]:
                index = self.n - 1
            else:
                while index < (self.n - 1) and self.xi[i] >= t[index + 1]:
                    index += 1

            for j in range(self.k):
                src = f"{src_name}[{index - self.k + 1 + j}]"
                target = f"{name}.input[{i}, {j}]"
                model.link(src, target)

        return
