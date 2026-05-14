"""Functional integration test for the time_to_climb trajectory optimization.

Requires the full compiled amigo.amigo C extension and MUMPS solver.
Run with:
    pytest tests/trajectory -v
"""

import numpy as np
import pytest, os, sys
import amigo as am
from amigo.interp import BSpline


# ── Class definitions (adapted from examples/time_to_climb/time_to_climb_traj.py) ──


class AircraftDynamics(am.TrajectoryComponent):
    def __init__(self, num_time_steps, scaling):
        # num_time_steps is now a constructor parameter instead of a module-level constant
        super().__init__(num_time_steps, state_size=5, aux_inputs=[{"name": "alpha"}])

        self.scaling = scaling

        # Physical constants matching the original code
        self.add_constant("S", value=49.2386)  # m^2
        self.add_constant("CL_alpha", value=3.44)
        self.add_constant("CD0", value=0.013)
        self.add_constant("kappa", value=0.54)
        self.add_constant("Isp", value=1600.0)  # s
        self.add_constant("TtoW", value=0.9)
        self.add_constant("m0", value=19030.0)  # kg
        self.add_constant("gamma_air", value=1.4)
        self.add_constant("R", value=287.058)
        self.add_constant("g", value=9.81)
        self.add_constant("conv", value=np.pi / 180.0)

    def dynamics(self, q, alpha):
        # Get constants
        S = self.constants["S"]
        CL_alpha = self.constants["CL_alpha"]
        CD0 = self.constants["CD0"]
        kappa = self.constants["kappa"]
        Isp = self.constants["Isp"]
        TtoW = self.constants["TtoW"]
        m0 = self.constants["m0"]
        g = self.constants["g"]
        conv = self.constants["conv"]

        # State variables
        v = 100.0 * q[0]  # Convert velocity to [m/s]
        gamma = q[1]  # flight path angle [degrees]
        m = 1000.0 * q[4]  # Convert mass to [kg]

        # Atmospheric properties (simplified)
        T_atm = 288.15  # K
        h_m = q[2] * self.scaling["altitude"]
        rho = 1.225 * am.exp(-h_m / 8500.0)  # kg/m^3

        # Aerodynamics matching original code
        CL = CL_alpha * conv * alpha  # Convert alpha from degrees to radians

        # Drag coefficient (simplified - no compressibility for now)
        CD = CD0 + kappa * CL**2

        # Dynamic pressure and forces
        qinfty = 0.5 * rho * v**2
        D = qinfty * S * CD
        L = qinfty * S * CL

        # Thrust
        T = TtoW * m0 * g

        # Convert angles to radians for dynamics
        alpha_rad = conv * alpha
        gamma_rad = conv * gamma

        # Set intermediate variables
        sin_alpha = am.sin(alpha_rad)
        cos_alpha = am.cos(alpha_rad)
        sin_gamma = am.sin(gamma_rad)
        cos_gamma = am.cos(gamma_rad)

        # Aircraft dynamics equations
        qdot = [
            ((T / m) * cos_alpha - (D / m) - g * sin_gamma) / self.scaling["velocity"],
            ((T / (m * v) * sin_alpha + (L / (m * v)) - (g / v) * cos_gamma)) / conv,
            v * sin_gamma / self.scaling["altitude"],
            v * cos_gamma / self.scaling["range"],
            -(T / (g * Isp)) / self.scaling["mass"],
        ]

        return qdot


class TtcObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("tf", label="final time")
        self.add_objective("obj")

    def compute(self):
        tf = self.inputs["tf"]
        self.objective["obj"] = tf  # Minimize final time


class TtcInitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=5)
        self.add_constraint("res", shape=5)

    def compute(self):
        q = self.inputs["q"]

        # Initial conditions matching original getInitConditions
        self.constraints["res"] = [
            q[0] - 136.0 / self.scaling["velocity"],  # 136 [m/s]
            q[1] - 0.0,  # flight path angle [degrees]
            q[2] - 100.0 / self.scaling["altitude"],  # 100.0 [m]
            q[3] - 0.0 / self.scaling["range"],  # [m]
            q[4] - 19030.0 / self.scaling["mass"],  # 19030 [kg]
        ]


class TtcFinalConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=5)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - 340.0 / self.scaling["velocity"],  # 340 [m/s]
            q[1] - 0.0,  # Final flight path angle [degrees]
            q[2] - 20000.0 / self.scaling["altitude"],  # 20 [km]
        ]


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Reduced parameters for test speed
_NUM_TIME_STEPS = 50
_NCTRL = 10
_SCALING = {"velocity": 100.0, "altitude": 1000.0, "range": 1000.0, "mass": 1000.0}


@pytest.fixture(scope="module")
def opt_result():
    """Build and solve the time_to_climb model once; return (xvec, data).

    Uses reduced parameters (num_time_steps=50) for CI speed while still
    exercising the full trajectory optimization pipeline.
    """
    num_time_steps = _NUM_TIME_STEPS
    nctrl = _NCTRL
    scaling = _SCALING

    # Create component instances
    ac = AircraftDynamics(num_time_steps=num_time_steps, scaling=scaling)
    obj = TtcObjective()
    ic = TtcInitialConditions(scaling)
    fc = TtcFinalConditions(scaling)

    # TrajectoryModel creates a submodel linking dynamics and trajectory source
    traj = am.TrajectoryModel(ac)

    # BSpline interpolates control points to angle-of-attack trajectory
    bspline = BSpline(
        input_name="x",
        output_name="alpha",
        num_interp_points=(num_time_steps + 1),
        num_ctrl_points=nctrl,
    )

    # Build the model
    model = am.Model("time_to_climb_test")
    model.add_model("bspline", bspline.create_model())
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)
    model.add_model("traj", traj.create_model())

    # Link alpha values from bspline output to trajectory source
    model.link("traj.source.alpha", "bspline.interp_values.alpha")

    # Link final time from objective to all trapezoidal rule components
    model.link("obj.tf[0]", "traj.kernel.tf[:]")

    # Link boundary conditions
    traj.link_boundary_conditions(model, "traj", "ic", "fc")

    # Build and initialize — always run build_module from the test file's
    # own directory so generate_cpp() writes the .cpp there and CMake finds
    # it. Then ensure that directory is on sys.path so importlib can find
    # the compiled .so regardless of where pytest is invoked from.
    _test_dir = os.path.dirname(os.path.abspath(__file__))
    _orig_dir = os.getcwd()
    os.chdir(_test_dir)
    try:
        model.build_module()
    finally:
        os.chdir(_orig_dir)
    if _test_dir not in sys.path:
        sys.path.insert(0, _test_dir)
    model.initialize()

    # Create design vector and bounds
    x = model.create_vector()
    lower = model.create_vector()
    upper = model.create_vector()

    # Default to zero
    x[:] = 0.0

    # Initial guess for final time
    tf_guess = 200.0
    t_guess = np.linspace(0, tf_guess, num_time_steps + 1)

    x["obj.tf"] = tf_guess
    x["traj.source.q[:, 0]"] = 1.36 + (3.40 - 1.36) * t_guess / tf_guess  # velocity
    x["traj.source.q[:, 1]"] = 5.0 * np.sin(
        np.pi * t_guess / tf_guess
    )  # flight path angle
    x["traj.source.q[:, 2]"] = 1.0 + (20.0 - 1.0) * t_guess / tf_guess  # altitude
    x["traj.source.q[:, 3]"] = 1.36 * t_guess  # range
    x["traj.source.q[:, 4]"] = 19.03 - 0.2 * t_guess / tf_guess  # mass decrease
    x["traj.source.alpha"] = 1.0

    # Bounds
    lower["obj.tf"] = 1.0
    upper["obj.tf"] = float("inf")

    lower["traj.source.alpha"] = -float("inf")
    upper["traj.source.alpha"] = float("inf")

    lower["traj.source.q"] = -float("inf")
    upper["traj.source.q"] = float("inf")

    lower["traj.source.q[:, 1]"] = -90.0
    upper["traj.source.q[:, 1]"] = 90.0

    lower["traj.source.q[:, 2]"] = 0.0
    upper["traj.source.q[:, 2]"] = 25.0

    lower["bspline.control_points.x"] = -25.0
    upper["bspline.control_points.x"] = 25.0

    # Run optimizer
    opt = am.Optimizer(model, x, lower=lower, upper=upper, solver="mumps")
    data = opt.optimize(
        {
            "initial_barrier_param": 1.0,
            "monotone_barrier_fraction": 0.25,
            "barrier_strategy": "monotone",
            "convergence_tolerance": 1e-8,
            "max_line_search_iterations": 5,
            "max_iterations": 100,
            "init_affine_step_multipliers": False,
        }
    )

    return x, data


# ── Test functions ────────────────────────────────────────────────────────────


def test_converged(opt_result):
    """Verify the optimizer converged successfully.

    Validates: Requirements 7.2
    """
    xvec, data = opt_result
    assert data["converged"], "Optimizer did not converge"


def test_optimal_tf(opt_result):
    """Verify the optimal final time is within tolerance of the reference solution.

    Validates: Requirements 7.3
    """
    xvec, data = opt_result
    tf_opt = xvec["obj.tf"][0]
    assert abs(tf_opt - 112.0) < 15.0, f"tf* = {tf_opt:.1f}s, expected ~112s"


def test_final_altitude(opt_result):
    """Verify the final altitude reaches the target within tolerance.

    Validates: Requirements 7.4
    """
    xvec, data = opt_result
    q = xvec["traj.source.q"]
    h_f = q[-1, 2]
    assert abs(h_f - 20.0) < 0.1, f"h_f = {h_f:.3f}, expected 20.0"


def test_final_velocity(opt_result):
    """Verify the final velocity reaches the target within tolerance.

    Validates: Requirements 7.4
    """
    xvec, data = opt_result
    q = xvec["traj.source.q"]
    v_f = q[-1, 0]
    assert abs(v_f - 3.40) < 0.05, f"v_f = {v_f:.3f}, expected 3.40"
