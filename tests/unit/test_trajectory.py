"""
Unit tests for amigo/trajectory/trajectory.py.

Runs without any C++ build — the conftest.py stubs mock out amigo.amigo,
mpi4py, scipy, and other heavy dependencies.
"""

import pytest
from amigo.trajectory.trajectory import (
    TrajectorySource,
    TrajectoryComponent,
    TrajectoryModel,
)
from amigo.expressions import Expr, VarNode


# ---------------------------------------------------------------------------
# Concrete subclass used by multiple test classes
# ---------------------------------------------------------------------------


class LinearDynamics(TrajectoryComponent):
    """Minimal concrete subclass with linear dynamics: qdot[i] = c * q[i]."""

    def __init__(self, num_time_steps, state_size, c=1.0, aux_inputs=None):
        if aux_inputs is None:
            aux_inputs = []
        super().__init__(num_time_steps, state_size, aux_inputs)
        self.c = c
        self.add_constant("c", value=c)

    def dynamics(self, q, *args):
        c = self.constants["c"]
        return [c * q[i] for i in range(self._state_size)]


# ---------------------------------------------------------------------------
# TestTrajectoryComponentInit
# ---------------------------------------------------------------------------


class TestTrajectoryComponentInit:
    """
    Example-based tests for TrajectoryComponent.__init__.
    Validates: Requirements 1.1, 1.2, 1.3, 1.4
    """

    def test_basic_inputs_registered(self):
        """tf, q1, q2 are registered as inputs when state_size=3 and no aux_inputs."""
        comp = LinearDynamics(num_time_steps=5, state_size=3)
        assert "tf" in comp.inputs
        assert "q1" in comp.inputs
        assert "q2" in comp.inputs

    def test_q1_q2_shapes(self):
        """q1 and q2 each have shape (3,) when state_size=3."""
        comp = LinearDynamics(num_time_steps=5, state_size=3)
        assert comp.inputs.get_shape("q1") == (3,)
        assert comp.inputs.get_shape("q2") == (3,)

    def test_res_constraint_registered(self):
        """Constraint 'res' is registered with shape (3,) when state_size=3."""
        comp = LinearDynamics(num_time_steps=5, state_size=3)
        assert "res" in comp.constraints
        assert comp.constraints.get_shape("res") == (3,)

    def test_single_aux_input_registers_both_endpoints(self):
        """alpha1 and alpha2 are registered when aux_inputs=[{"name": "alpha"}]."""
        comp = LinearDynamics(
            num_time_steps=5,
            state_size=3,
            aux_inputs=[{"name": "alpha"}],
        )
        assert "alpha1" in comp.inputs
        assert "alpha2" in comp.inputs

    def test_two_aux_inputs_register_all_four_endpoints(self):
        """All four auxiliary inputs are registered when two aux_inputs are given."""
        comp = LinearDynamics(
            num_time_steps=5,
            state_size=3,
            aux_inputs=[{"name": "alpha"}, {"name": "throttle"}],
        )
        assert "alpha1" in comp.inputs
        assert "alpha2" in comp.inputs
        assert "throttle1" in comp.inputs
        assert "throttle2" in comp.inputs


# ---------------------------------------------------------------------------
# TestTrajectorySourceInit
# ---------------------------------------------------------------------------


class TestTrajectorySourceInit:
    """
    Example-based tests for TrajectorySource.__init__.
    Validates: Requirements 2.1, 2.2, 2.3
    """

    def test_q_registered_with_correct_shape(self):
        """'q' is registered with shape (4,) when state_size=4 and no inputs."""
        source = TrajectorySource(state_size=4)
        assert "q" in source.inputs
        assert source.inputs.get_shape("q") == (4,)

    def test_named_input_registered(self):
        """'alpha' is registered when inputs=[{"name": "alpha"}]."""
        source = TrajectorySource(state_size=4, inputs=[{"name": "alpha"}])
        assert "alpha" in source.inputs

    def test_named_input_registered_with_shape(self):
        """'u' is registered with shape (2,) when inputs=[{"name": "u", "shape": (2,)}]."""
        source = TrajectorySource(state_size=4, inputs=[{"name": "u", "shape": (2,)}])
        assert "u" in source.inputs
        assert source.inputs.get_shape("u") == (2,)


# ---------------------------------------------------------------------------
# TestTrapezoidRule
# ---------------------------------------------------------------------------


class TestTrapezoidRule:
    """
    Example-based tests for the trapezoid rule residual computed by
    TrajectoryComponent.compute() via _initialize_expressions().
    Validates: Requirements 3.1, 3.2, 3.3
    """

    def test_res_is_list_of_expr(self):
        """constraints['res'] is a list of Expr objects after _initialize_expressions()."""
        comp = LinearDynamics(num_time_steps=5, state_size=3, c=2.0)
        comp._initialize_expressions()
        res = comp.constraints["res"]
        assert isinstance(res, list), "constraints['res'] should be a list"
        assert len(res) == 3, f"Expected 3 residual elements, got {len(res)}"
        for elem in res:
            assert isinstance(elem, Expr), f"Expected Expr, got {type(elem)}"

    def test_res_elements_are_not_zero(self):
        """None of the residual Expr elements is_zero() for non-trivial linear dynamics."""
        comp = LinearDynamics(num_time_steps=5, state_size=3, c=2.0)
        comp._initialize_expressions()
        res = comp.constraints["res"]
        for i, elem in enumerate(res):
            assert not elem.is_zero(), f"residual[{i}] should not be zero"


# ---------------------------------------------------------------------------
# AircraftDynamics concrete subclass (adapted from examples/time_to_climb)
# num_time_steps is a constructor parameter (not a module-level constant)
# ---------------------------------------------------------------------------

import amigo as am
import numpy as np


class AircraftDynamics(TrajectoryComponent):
    """
    Concrete TrajectoryComponent subclass implementing 5-state aircraft dynamics.
    Adapted from examples/time_to_climb/time_to_climb_traj.py with num_time_steps
    as a constructor parameter rather than a module-level constant.
    """

    def __init__(self, num_time_steps, scaling):
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

        return

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
        h_m = q[2] * self.scaling["altitude"]
        rho = 1.225 * am.exp(-h_m / 8500.0)  # kg/m^3

        # Aerodynamics
        CL = CL_alpha * conv * alpha  # Convert alpha from degrees to radians

        # Drag coefficient
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

        # Trig functions (pure Python via amigo/unary_operations.py)
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


# ---------------------------------------------------------------------------
# TestAircraftDynamicsInit
# ---------------------------------------------------------------------------

_DEFAULT_SCALING = {
    "velocity": 100.0,
    "altitude": 1000.0,
    "range": 1000.0,
    "mass": 1000.0,
}

_REQUIRED_CONSTANTS = [
    "S",
    "CL_alpha",
    "CD0",
    "kappa",
    "Isp",
    "TtoW",
    "m0",
    "gamma_air",
    "R",
    "g",
    "conv",
]


class TestAircraftDynamicsInit:
    """
    Example-based tests for AircraftDynamics.__init__.
    Validates: Requirements 4.1, 4.2, 4.3
    """

    def test_is_subclass_of_trajectory_component(self):
        """AircraftDynamics is a subclass of TrajectoryComponent."""
        assert issubclass(AircraftDynamics, TrajectoryComponent)

    def test_state_size_is_five(self):
        """AircraftDynamics is constructed with state_size=5."""
        ac = AircraftDynamics(num_time_steps=50, scaling=_DEFAULT_SCALING)
        assert ac._state_size == 5

    def test_aux_inputs_alpha(self):
        """AircraftDynamics is constructed with aux_inputs=[{"name": "alpha"}]."""
        ac = AircraftDynamics(num_time_steps=50, scaling=_DEFAULT_SCALING)
        assert "alpha1" in ac.inputs
        assert "alpha2" in ac.inputs

    def test_all_required_constants_present(self):
        """All required physical constants are present in self.constants."""
        ac = AircraftDynamics(num_time_steps=50, scaling=_DEFAULT_SCALING)
        for name in _REQUIRED_CONSTANTS:
            assert name in ac.constants, f"Constant '{name}' not found in ac.constants"


# ---------------------------------------------------------------------------
# TestAircraftDynamicsOutput
# ---------------------------------------------------------------------------


class TestAircraftDynamicsOutput:
    """
    Example-based tests for AircraftDynamics dynamics output via _initialize_expressions().
    Validates: Requirements 5.1, 5.2, 5.3
    """

    def test_res_has_five_elements(self):
        """constraints['res'] has exactly 5 elements after _initialize_expressions()."""
        comp = AircraftDynamics(
            num_time_steps=50,
            scaling={
                "velocity": 100.0,
                "altitude": 1000.0,
                "range": 1000.0,
                "mass": 1000.0,
            },
        )
        comp._initialize_expressions()
        res = comp.constraints["res"]
        assert isinstance(res, list), "constraints['res'] should be a list"
        assert len(res) == 5, f"Expected 5 residual elements, got {len(res)}"

    def test_res_elements_are_expr_instances(self):
        """Each element of constraints['res'] is an Expr instance."""
        comp = AircraftDynamics(
            num_time_steps=50,
            scaling={
                "velocity": 100.0,
                "altitude": 1000.0,
                "range": 1000.0,
                "mass": 1000.0,
            },
        )
        comp._initialize_expressions()
        res = comp.constraints["res"]
        for i, elem in enumerate(res):
            assert isinstance(
                elem, Expr
            ), f"Expected Expr at index {i}, got {type(elem)}"

    def test_res_elements_are_not_zero(self):
        """None of the residual Expr elements is_zero() for non-trivial aircraft dynamics."""
        comp = AircraftDynamics(
            num_time_steps=50,
            scaling={
                "velocity": 100.0,
                "altitude": 1000.0,
                "range": 1000.0,
                "mass": 1000.0,
            },
        )
        comp._initialize_expressions()
        res = comp.constraints["res"]
        for i, elem in enumerate(res):
            assert not elem.is_zero(), f"residual[{i}] should not be zero"


# ---------------------------------------------------------------------------
# InitialConditions and FinalConditions (adapted from examples/time_to_climb)
# These are am.Component subclasses (not TrajectoryComponent)
# ---------------------------------------------------------------------------


class InitialConditions(am.Component):
    """
    Boundary condition component enforcing initial state constraints.
    Adapted from examples/time_to_climb/time_to_climb_traj.py.
    """

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


class FinalConditions(am.Component):
    """
    Boundary condition component enforcing final state constraints.
    Adapted from examples/time_to_climb/time_to_climb_traj.py.
    """

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


# ---------------------------------------------------------------------------
# TestBoundaryConditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """
    Example-based tests for InitialConditions and FinalConditions components.
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """

    def test_initial_conditions_res_has_five_elements(self):
        """constraints['res'] has exactly 5 elements for InitialConditions."""
        ic = InitialConditions(scaling=_DEFAULT_SCALING)
        ic._initialize_expressions()
        res = ic.constraints["res"]
        assert isinstance(res, list), "constraints['res'] should be a list"
        assert len(res) == 5, f"Expected 5 residual elements, got {len(res)}"

    def test_initial_conditions_res_elements_are_expr(self):
        """Each element of InitialConditions.constraints['res'] is an Expr instance."""
        ic = InitialConditions(scaling=_DEFAULT_SCALING)
        ic._initialize_expressions()
        res = ic.constraints["res"]
        for i, elem in enumerate(res):
            assert isinstance(
                elem, Expr
            ), f"Expected Expr at index {i}, got {type(elem)}"

    def test_final_conditions_res_has_three_elements(self):
        """constraints['res'] has exactly 3 elements for FinalConditions."""
        fc = FinalConditions(scaling=_DEFAULT_SCALING)
        fc._initialize_expressions()
        res = fc.constraints["res"]
        assert isinstance(res, list), "constraints['res'] should be a list"
        assert len(res) == 3, f"Expected 3 residual elements, got {len(res)}"

    def test_final_conditions_res_elements_are_expr(self):
        """Each element of FinalConditions.constraints['res'] is an Expr instance."""
        fc = FinalConditions(scaling=_DEFAULT_SCALING)
        fc._initialize_expressions()
        res = fc.constraints["res"]
        for i, elem in enumerate(res):
            assert isinstance(
                elem, Expr
            ), f"Expected Expr at index {i}, got {type(elem)}"
