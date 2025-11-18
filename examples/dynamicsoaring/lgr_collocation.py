"""
LGR (Legendre-Gauss-Radau) Collocation utilities for Amigo

This implements the high-order pseudospectral collocation method used in GPOPS/YAPSS.
"""

import numpy as np
from scipy.special import roots_jacobi


def lgr_points(N):
    """
    Compute the LGR (Legendre-Gauss-Radau) collocation points.

    LGR points are the roots of P_N(τ) + P_{N-1}(τ) on [-1, 1]
    where P_N is the Legendre polynomial of degree N.

    Args:
        N: Number of collocation points (including endpoint at τ=1)

    Returns:
        tau: LGR points on [-1, 1], where tau[-1] = 1.0
    """
    if N == 1:
        return np.array([1.0])

    # Compute roots of (1+τ)P_{N-1}(τ) which gives LGR points
    # These are roots of Jacobi polynomial P^{(0,1)}_{N-1}
    roots, _ = roots_jacobi(N - 1, 0, 1)

    # LGR points include the right endpoint
    tau = np.append(roots, 1.0)

    return tau


def lgr_differentiation_matrix(tau):
    """
    Compute the LGR differentiation matrix.

    The differentiation matrix D maps function values at LGR points to
    derivative values: df/dτ ≈ D @ f

    Uses the formula: D[i,j] = L'_j(τ_i) where L_j is the Lagrange basis polynomial.

    Args:
        tau: LGR collocation points

    Returns:
        D: Differentiation matrix (N x N)
    """
    N = len(tau)
    D = np.zeros((N, N))

    # Barycentric weights for stable Lagrange interpolation
    w = np.ones(N)
    for j in range(N):
        for k in range(N):
            if k != j:
                w[j] /= tau[j] - tau[k]

    # Compute differentiation matrix
    for i in range(N):
        for j in range(N):
            if i == j:
                # Diagonal entry
                for k in range(N):
                    if k != i:
                        D[i, i] += 1.0 / (tau[i] - tau[k])
            else:
                # Off-diagonal entry
                D[i, j] = (w[j] / w[i]) / (tau[i] - tau[j])

    return D


def lgr_integration_weights(tau):
    """
    Compute LGR quadrature weights for integration.

    Args:
        tau: LGR collocation points

    Returns:
        w: Integration weights
    """
    N = len(tau)
    w = np.zeros(N)

    # Barycentric weights for Lagrange interpolation
    for j in range(N):
        prod = 1.0
        for k in range(N):
            if k != j:
                prod *= tau[j] - tau[k]

        # Integrate Lagrange basis polynomial
        # For LGR, there's a closed form
        if j == N - 1:
            # Last point (τ = 1)
            w[j] = 2.0 / (N * N)
        else:
            # Use Legendre polynomial evaluation
            from scipy.special import eval_legendre

            P_N = eval_legendre(N, tau[j])
            w[j] = (1 - tau[j]) / (N * N * P_N * P_N)

    return w


import amigo as am


class LGRCollocation(am.Component):
    """
    LGR pseudospectral collocation for a single segment.

    This enforces the differential equations at LGR collocation points
    using the differentiation matrix.
    """

    def __init__(self, scaling, num_points=6, num_states=6):
        """
        Args:
            scaling: Dictionary of scaling factors
            num_points: Number of LGR collocation points per segment (default: 6)
            num_states: Number of state variables
        """
        super().__init__()

        self.scaling = scaling
        self.num_points = num_points
        self.num_states = num_states

        # Get LGR points and differentiation matrix
        tau = lgr_points(num_points)
        self.D = lgr_differentiation_matrix(tau)

        # Store as constants (for use in compute)
        for i in range(num_points):
            for j in range(num_points):
                self.add_constant(f"D_{i}_{j}", value=self.D[i, j])

        # Inputs: segment time span and states at collocation points
        self.add_input("t0")  # Segment start time
        self.add_input("tf")  # Segment end time (could be segment duration)

        # State values at all collocation points in the segment
        self.add_input("q", shape=(num_points, num_states))

        # State derivatives at all collocation points (from dynamics)
        self.add_input("f", shape=(num_points, num_states))

        # Collocation constraints
        self.add_constraint("defect", shape=(num_points, num_states))

    def compute(self):
        """
        Enforce defect constraints: dq/dt = f(q, u, t)

        At each collocation point i:
            (2/(tf-t0)) * sum_j D[i,j] * q[j] - f[i] = 0
        """
        t0 = self.inputs["t0"]
        tf = self.scaling["time"] * self.inputs["tf"]

        q = self.inputs["q"]  # shape: (num_points, num_states)
        f = self.inputs["f"]  # shape: (num_points, num_states)

        # Time scale factor: τ ∈ [-1,1] maps to t ∈ [t0, tf]
        # dt/dτ = (tf - t0) / 2
        time_factor = 2.0 / (tf - t0) if abs(tf - t0) > 1e-12 else 0.0

        defect = np.zeros((self.num_points, self.num_states))

        for i in range(self.num_points):
            for k in range(self.num_states):
                # Compute dq/dτ using differentiation matrix
                dq_dtau = 0.0
                for j in range(self.num_points):
                    D_ij = self.constants[f"D_{i}_{j}"]
                    dq_dtau += D_ij * q[j, k]

                # Convert to dq/dt and compare with dynamics
                dq_dt = time_factor * dq_dtau
                defect[i, k] = dq_dt - f[i, k]

        self.constraints["defect"] = defect


def test_lgr():
    """Test the LGR utilities"""
    print("Testing LGR Collocation Points")
    print("=" * 60)

    for N in [3, 4, 5, 6]:
        tau = lgr_points(N)
        print(f"\nN = {N} LGR points:")
        print(f"  τ = {tau}")
        print(f"  Last point = {tau[-1]:.10f} (should be 1.0)")

        # Verify they're in [-1, 1]
        assert np.all(tau >= -1.0) and np.all(tau <= 1.0)
        assert abs(tau[-1] - 1.0) < 1e-10

    # Test differentiation matrix
    print("\n" + "=" * 60)
    print("Testing Differentiation Matrix (N=4)")
    tau = lgr_points(4)
    D = lgr_differentiation_matrix(tau)
    print(f"D shape: {D.shape}")
    print(f"D matrix:")
    print(D)

    # Test on a simple function: f(τ) = τ^2, f'(τ) = 2τ
    f_vals = tau**2
    df_exact = 2 * tau
    df_numeric = D @ f_vals
    print(f"\nTest: f(τ) = τ²")
    print(f"  Exact derivative: {df_exact}")
    print(f"  Numeric derivative: {df_numeric}")
    print(f"  Error: {np.max(np.abs(df_numeric - df_exact)):.2e}")

    # Test integration weights
    print("\n" + "=" * 60)
    print("Testing Integration Weights (N=4)")
    w = lgr_integration_weights(tau)
    print(f"Weights: {w}")
    print(f"Sum of weights: {np.sum(w):.10f} (should be 2.0 for [-1,1])")

    # Integrate τ^2 from -1 to 1 (exact = 2/3)
    integral = np.sum(w * tau**2)
    exact = 2.0 / 3.0
    print(f"\nIntegral of τ²: {integral:.10f} (exact: {exact:.10f})")
    print(f"Error: {abs(integral - exact):.2e}")


if __name__ == "__main__":
    test_lgr()
