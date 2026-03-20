import numpy as np
import amigo as am


class BSplineSource(am.Component):
    def __init__(self, src_name):
        super().__init__()

        self.add_input(src_name)


class BSplineKernel(am.Component):
    def __init__(self, k, input_name, output_name):
        super().__init__()

        self.k = k
        self.input_name = input_name
        self.output_name = output_name

        # Set the size of the input data
        self.add_data("N", shape=self.k)

        # Set the input values to be interpolated
        self.add_input(input_name, shape=self.k)

        # Set the output values
        self.add_input(output_name)

        # Set the coupling constraint
        self.add_constraint("constraint")

    def compute(self):
        """Compute the derivative outputs at the constant set of knots"""
        N = self.data["N"]
        input = self.inputs[self.input_name]
        output = self.inputs[self.output_name]

        value = 0.0
        for i in range(self.k):
            value = value + N[i] * input[i]
        self.constraints["constraint"] = output - value

        return


class BSpline:
    def __init__(
        self,
        input_name: str,
        output_name: str,
        interp_points=None,
        num_interp_points: int = 0,
        length: float = 1.0,
        k: int = 4,
        num_ctrl_points: int = 10,
        deriv: int = 0,
    ):
        """
        Perform a BSpline interpolation from a fixed set of parameter points. If the parameter points
        change as a function of the variables, you cannot use this method.

        constraint = output - N(interp_points) * input

        Args:
            num_interp_points (int) : Number of evaluation points
            k (int) : Order of the bspline polynomial (degree + 1)
            num_ctrl_points (int) : Number of control points
            deriv (int) : Number of derivatives to compute
        """

        self.input_name = input_name
        self.output_name = output_name

        # Set the order of the bspline
        self.k = k

        # Set the number of input points
        self.num_ctrl_points = num_ctrl_points

        # Set the order of the derivative
        self.deriv = deriv

        # Set the length of the interval
        self.length = length

        # Set the interpolation points
        if interp_points is None:
            self.num_interp_points = num_interp_points
            self.interp_points = np.linspace(0, self.length, self.num_interp_points)
        else:
            self.interp_points = interp_points
            self.num_interp_points = len(interp_points)

        return

    def _compute_knots(self):
        """
        Compute the BSpline knots for interpolation
        """

        # Set the knot locations
        t = np.zeros(self.num_ctrl_points + self.k)
        t[: self.k] = 0.0
        t[-self.k : :] = self.length
        t[self.k - 1 : -self.k + 1] = np.linspace(
            0, self.length, self.num_ctrl_points - self.k + 2
        )

        return t

    def _compute_basis(self):
        """
        Compute the BSpline basis functions at the given knot locations
        """

        t = self._compute_knots()

        # Special case for right endpoint
        interp_points_clip = np.clip(self.interp_points, t[0], t[-1])

        # Find span index i such that t[i] <= x < t[i+1]
        span = np.searchsorted(t, interp_points_clip, side="right") - 1
        span = np.clip(span, self.k - 1, self.num_ctrl_points - 1)

        N = np.zeros((self.num_interp_points, self.k), dtype=float)

        # Evaluate using the Cox–de Boor recursion for each point
        for j in range(self.interp_points.size):
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
                    N_j[r] = saved + (right - interp_points_clip[j]) * temp
                    saved = (interp_points_clip[j] - left) * temp
                N_j[d] = saved

            N[j, :] = N_j

        # Handle x == t[-1]
        last_mask = self.interp_points == t[-1]
        if np.any(last_mask):
            N[last_mask, :] = 0.0
            N[last_mask, -1] = 1.0

        return N

    def _compute_basis_derivative(self, deriv=1):
        t = self.compute_knots()

        # Special case for right endpoint
        interp_points_clamped = np.clip(self.interp_points, t[0], t[-1])

        # Find span index i such that t[i] <= x < t[i+1]
        span = np.searchsorted(t, interp_points_clamped, side="right") - 1
        span = np.clip(span, self.k - 1, self.num_ctrl_points - 1)

        Nd = np.zeros((self.num_interp_points, self.k), dtype=float)

        # Evaluate using the Cox–de Boor recursion for each point
        for j in range(self.num_interp_points):
            i = span[j]
            x = interp_points_clamped[j]

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

    def create_model(self, module_name: str | None = None):
        model = am.Model(module_name)

        model.add_component(
            "control_points", self.num_ctrl_points, BSplineSource(self.input_name)
        )
        model.add_component(
            "interp_values", self.num_interp_points, BSplineSource(self.output_name)
        )

        kernel = BSplineKernel(self.k, self.input_name, self.output_name)
        model.add_component("kernel", self.num_interp_points, kernel)

        # Set the knot locations
        t = self._compute_knots()

        index = self.k - 1
        for i in range(self.num_interp_points):
            if self.interp_points[i] < t[self.k - 1]:
                index = self.k - 1
            elif self.interp_points[i] > t[self.num_ctrl_points - 1]:
                index = self.num_ctrl_points - 1
            else:
                while (
                    index < (self.num_ctrl_points - 1)
                    and self.interp_points[i] >= t[index + 1]
                ):
                    index += 1

            for j in range(self.k):
                src = f"control_points.{self.input_name}[{index - self.k + 1 + j}]"
                target = f"kernel.{self.input_name}[{i}, {j}]"
                model.link(src, target)

        model.link(f"kernel.{self.output_name}", f"interp_values.{self.output_name}")

        # Set the locations for where to evaluate the bspline basis
        if self.deriv == 0:
            model.set_data(f"kernel.N", self._compute_basis())
        else:
            model.set_data(
                f"kernel.N", self._compute_basis_derivative(deriv=self.deriv)
            )

        return model
