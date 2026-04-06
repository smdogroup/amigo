"""
Single-Track Racecar Minimum Lap Time Optimization (Starter Code)
=================================================================

This script solves a minimum lap time problem using a single-track
(bicycle) vehicle model in curvilinear coordinates with Amigo.

The problem structure is:
    min   t_final                      (minimize lap time)
    s.t.  vehicle dynamics (ODE)       (how the car moves)
          tire friction circles         (grip limits per wheel)
          engine power limit            (powertrain ceiling)
          track boundaries              (stay on track)
          periodic boundary conditions  (closed circuit, start = end)

Curvilinear coordinates

Instead of using time as the independent variable, we parameterize
everything by arc-length s along the track centerline. This is standard
for lap time problems because the track geometry is naturally described
in s. The relation between time and arc-length is:

    ds/dt = V * cos(alpha - lambda) / (1 - n * kappa(s))

where:
    V     = vehicle speed
    alpha = heading angle relative to centerline tangent
    n     = lateral offset from centerline (positive = left)
    kappa = track curvature (from tracks.py)

The ODE right-hand sides are written as dq/ds = f(q, u) / (ds/dt),
i.e. we divide each time-derivative by sdot to change the independent
variable from t to s.

State vector q (8 components, all scaled):
    q[0] = t       time elapsed [s]
    q[1] = n       lateral offset from centerline [m]
    q[2] = V       vehicle speed [m/s]
    q[3] = alpha   heading angle [rad]
    q[4] = lam     body sideslip angle [rad]
    q[5] = omega   yaw rate [rad/s]
    q[6] = ax      longitudinal acceleration (filtered) [m/s^2]
    q[7] = ay      lateral acceleration (filtered) [m/s^2]

Control vector u (2 components):
    u[0] = delta   front steering angle [rad]  (scaled)
    u[1] = thrust  throttle/brake command [-]   (unscaled, >0 throttle, <0 brake)

Scaling

Each state/control is stored as q_scaled = q_physical / scale_factor.
This keeps all optimization variables O(1), which helps the IPM solver
converge. The scaling dictionary maps each name to its scale factor.

Transcription

The ODE is discretized using trapezoidal collocation: the track is
divided into num_intervals equal segments of length ds. On each segment
[s_k, s_{k+1}], the collocation constraint is:

    q_{k+1} - q_k - 0.5 * ds * (f(q_k, u_k) + f(q_{k+1}, u_{k+1})) = 0

This converts the continuous ODE into a large system of algebraic
equations that Amigo solves simultaneously (direct collocation).

How Amigo components work

Each am.Component defines:
  - inputs:      optimization variables (what the solver can change)
  - constants:   fixed values set once at construction
  - data:        values set before each solve (e.g. kappa at each node)
  - constraints: residuals that must equal zero (or satisfy bounds)
  - objectives:  terms added to the cost function

The am.Model collects components and uses model.link() to enforce that
two inputs share the same underlying optimization variable (e.g. q2 of
interval k is the same variable as q1 of interval k+1).
"""

import amigo as am
import numpy as np
import argparse
import json
from pathlib import Path

from tracks import berlin_2018


# ===========================================================================
# Vehicle parameters (single-track model)
# ===========================================================================
# These are representative of a Formula E style car.
M = 800.0  # vehicle mass [kg]
g = 9.8  # gravitational acceleration [m/s^2]
a_wb = 1.8  # distance from CG to front axle [m]
b_wb = 1.6  # distance from CG to rear axle [m]
L_wb = a_wb + b_wb  # wheelbase [m]
tw = 0.73  # half-track width [m] (used for load transfer)
Iz = 450.0  # yaw moment of inertia [kg*m^2]
rho = 1.2  # air density [kg/m^3]
CdA = 2.0  # drag coefficient * frontal area [m^2]
ClA = 4.0  # lift coefficient * planform area [m^2] (downforce)
CoP = 1.6  # center of pressure distance from front axle [m]
h_cg = 0.3  # CG height [m]
chi = 0.5  # lateral load transfer distribution (0=rear, 1=front)
beta_brake = 0.62  # front brake bias [-]
k_lambda = 44.0  # tire cornering stiffness coefficient [1/rad]
mu0 = 1.68  # peak tire friction coefficient [-]
tau_x = 0.2  # longitudinal acceleration filter time constant [s]
tau_y = 0.2  # lateral acceleration filter time constant [s]
P_max = 960000.0  # maximum engine power [W]
EPS_THRUST = 0.001  # smoothing parameter for |thrust| (avoids kink at 0)

# Scaling factors: q_physical = scaling[name] * q_scaled
# Chosen so that typical values of each variable are O(1) in the NLP.
scaling = {
    "t": 60.0,  # lap times are ~60 s
    "n": 5.0,  # lateral offsets are a few meters
    "V": 40.0,  # speeds are ~40 m/s
    "alpha": 0.15,  # heading angles are ~0.15 rad
    "lam": 0.05,  # sideslip angles are small
    "omega": 0.5,  # yaw rates are ~0.5 rad/s
    "ax": 8.0,  # longitudinal accelerations ~8 m/s^2
    "ay": 8.0,  # lateral accelerations ~8 m/s^2
    "delta": 0.1,  # steering angles ~0.1 rad
}

# Discretization
num_intervals = 300  # number of collocation intervals
num_nodes = num_intervals + 1  # number of mesh nodes (0, 1, ..., N)
EPS_DELTA = 1e-3  # steering smoothness regularization weight

# Load the track (provides kappa, widths, centerline at each node)
track = berlin_2018(num_nodes)
s_final = track.s_total
s_nodes = track.s
kappa_nodes = track.kappa
ds = s_final / num_intervals

print(f"Track: {track.name}")
print(f"Track length: {s_final:.2f} m, intervals: {num_intervals}, ds: {ds:.2f} m")
print(f"Curvature: [{kappa_nodes.min():.4f}, {kappa_nodes.max():.4f}] 1/m")


# ===========================================================================
# Component 1: Collocation constraints (dynamics + trapezoidal rule)
# ===========================================================================
class RacecarCollocation(am.Component):
    """
    Trapezoidal collocation for one interval [s_k, s_{k+1}].

    Residual: q2 - q1 - 0.5 * ds * (f(q1,u1,kappa1) + f(q2,u2,kappa2)) = 0

    There are num_intervals instances of this component (one per segment).
    Each instance has its own kappa1 and kappa2 (curvature at left/right
    endpoints), set via the data vector before solving.

    Amigo API used here:
      add_constant  fixed scalar known at construction (ds)
      add_data      scalar set from the data vector before solve (kappa)
      add_input     optimization variable (q, u)
      add_constraint  residual that must be driven to zero
    """

    def __init__(self, scaling, ds):
        super().__init__()
        self.scaling = scaling
        self.add_constant("ds", value=ds)
        self.add_data("kappa1", value=0.0)
        self.add_data("kappa2", value=0.0)
        self.add_input("q1", shape=8)  # state at left node
        self.add_input("q2", shape=8)  # state at right node
        self.add_input("u1", shape=2)  # control at left node
        self.add_input("u2", shape=2)  # control at right node
        self.add_constraint("res", shape=8)

    def _dynamics(self, q, u, kappa):
        """
        Compute dq/ds for the single-track vehicle model.

        The physical equations of motion give dq/dt. To convert to dq/ds
        we divide by sdot = ds/dt = V*cos(alpha-lam) / (1 - n*kappa).

        Returns a tuple of 8 scaled derivatives (one per state).
        """
        # Unscale states and controls to physical units
        V = self.scaling["V"] * q[2]
        n = self.scaling["n"] * q[1]
        alpha = self.scaling["alpha"] * q[3]
        lam = self.scaling["lam"] * q[4]
        omega = self.scaling["omega"] * q[5]
        ax_phys = self.scaling["ax"] * q[6]
        ay_phys = self.scaling["ay"] * q[7]
        delta = self.scaling["delta"] * u[0]
        thrust = u[1]

        # Aerodynamic downforce (split front/rear by CoP)
        downforce = 0.5 * rho * ClA * V * V
        df_front = downforce * (1.0 - CoP / L_wb)
        df_rear = downforce * (CoP / L_wb)

        # Normal loads on each wheel
        # Static weight + longitudinal/lateral load transfer + aero
        N_fl = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb + ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_fr = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb - ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_rl = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb + ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )
        N_rr = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb - ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )

        # Throttle/brake split
        # smooth |thrust| = sqrt(thrust^2 + eps^2) avoids non-differentiable |.|
        smooth_abs = (thrust * thrust + EPS_THRUST * EPS_THRUST) ** 0.5
        throttle = 0.5 * (thrust + smooth_abs)  # max(thrust, 0) smoothed
        brake = 0.5 * (-thrust + smooth_abs)  # max(-thrust, 0) smoothed

        # Longitudinal tire forces (drive/brake)
        S_fl = -(M * g / 2) * brake * beta_brake
        S_fr = -(M * g / 2) * brake * beta_brake
        S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
        S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

        # Lateral tire forces (linear tire model: F = N * k_lambda * slip_angle)
        F_rr = N_rr * k_lambda * (lam + omega * (b_wb + lam * tw) / V)
        F_rl = N_rl * k_lambda * (lam + omega * (b_wb - lam * tw) / V)
        F_fr = N_fr * k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
        F_fl = N_fl * k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

        # Aggregate forces
        F_all = F_fl + F_fr + F_rl + F_rr  # total lateral
        S_all = S_fl + S_fr + S_rl + S_rr  # total longitudinal
        F_front = F_fl + F_fr  # front axle lateral
        S_front = S_fl + S_fr  # front axle longitudinal
        drag = 0.5 * rho * CdA * V * V  # aerodynamic drag

        # Curvilinear kinematics
        # am.cos / am.sin are Amigo's auto-differentiable trig functions
        cos_al = am.cos(alpha - lam)
        sin_al = am.sin(alpha - lam)

        # ds/dt: how fast the car advances along the centerline
        sdot = V * cos_al / (1.0 - n * kappa)

        # Time derivatives of the states (dq/dt)
        ndot = V * sin_al
        alphadot = omega - kappa * V * cos_al / (1.0 - n * kappa)
        Vdot = S_all / M - delta * F_front / M - drag / M - omega * V * lam
        lambdadot = omega - Vdot * lam / V - delta * S_front / (M * V) - F_all / (M * V)
        omegadot = (
            (a_wb * F_front) / Iz
            - (b_wb * (F_rr + F_rl)) / Iz
            + (tw * (-S_rr + S_rl - S_fr + S_fl)) / Iz
        )
        # Filtered accelerations (first-order lag)
        axdot = (Vdot + omega * V * lam - ax_phys) / tau_x
        aydot = (omega * V - (V * lambdadot + Vdot * lam) - ay_phys) / tau_y

        # Convert dq/dt -> dq/ds by dividing by sdot, then divide by
        # the scaling factor so the residual is in scaled units.
        return (
            1.0 / (sdot * self.scaling["t"]),  # dt/ds
            ndot / (sdot * self.scaling["n"]),  # dn/ds
            Vdot / (sdot * self.scaling["V"]),  # dV/ds
            alphadot / (sdot * self.scaling["alpha"]),  # dalpha/ds
            lambdadot / (sdot * self.scaling["lam"]),  # dlambda/ds
            omegadot / (sdot * self.scaling["omega"]),  # domega/ds
            axdot / (sdot * self.scaling["ax"]),  # dax/ds
            aydot / (sdot * self.scaling["ay"]),  # day/ds
        )

    def compute(self):
        q1, q2 = self.inputs["q1"], self.inputs["q2"]
        u1, u2 = self.inputs["u1"], self.inputs["u2"]
        ds_val = self.constants["ds"]
        f1 = self._dynamics(q1, u1, self.data["kappa1"])
        f2 = self._dynamics(q2, u2, self.data["kappa2"])
        # Trapezoidal rule residual: q2 - q1 - 0.5*ds*(f1 + f2) = 0
        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * ds_val * (f1[i] + f2[i]) for i in range(8)
        ]


# ===========================================================================
# Component 2: Inequality constraints at each node
# ===========================================================================
class NodeConstraints(am.Component):
    """
    Tire friction circle (4 wheels) + power limit at each node.

    Friction circle: (Sx / (N*mu))^2 + (Fy / (N*mu))^2 <= 1
        Each wheel's combined longitudinal and lateral force must stay
        inside its friction ellipse. This is the grip limit.

    Power limit: V * S_total / P_max <= 1
        The engine cannot deliver more than P_max watts.

    These are inequality constraints with upper bound 1.0.
    There are num_nodes instances (one per mesh node).
    """

    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=8)
        self.add_input("u", shape=2)
        # upper=1.0 means these constraints must be <= 1
        self.add_constraint("c_fl", lower=float("-inf"), upper=1.0)
        self.add_constraint("c_fr", lower=float("-inf"), upper=1.0)
        self.add_constraint("c_rl", lower=float("-inf"), upper=1.0)
        self.add_constraint("c_rr", lower=float("-inf"), upper=1.0)
        self.add_constraint("c_pow", lower=float("-inf"), upper=1.0)

    def _node_constraints(self, q, u):
        V = self.scaling["V"] * q[2]
        lam = self.scaling["lam"] * q[4]
        omega = self.scaling["omega"] * q[5]
        ax_phys = self.scaling["ax"] * q[6]
        ay_phys = self.scaling["ay"] * q[7]
        delta = self.scaling["delta"] * u[0]
        thrust = u[1]

        downforce = 0.5 * rho * ClA * V * V
        df_front = downforce * (1.0 - CoP / L_wb)
        df_rear = downforce * (CoP / L_wb)

        N_fl = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb + ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_fr = (
            (M * g / 2) * (b_wb / L_wb)
            + (M / 4) * (-(ax_phys * h_cg) / L_wb - ay_phys * chi * h_cg / tw)
            + df_front / 2
        )
        N_rl = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb + ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )
        N_rr = (
            (M * g / 2) * (a_wb / L_wb)
            + (M / 4) * ((ax_phys * h_cg) / L_wb - ay_phys * (1 - chi) * h_cg / tw)
            + df_rear / 2
        )

        smooth_abs = (thrust * thrust + EPS_THRUST * EPS_THRUST) ** 0.5
        throttle = 0.5 * (thrust + smooth_abs)
        brake = 0.5 * (-thrust + smooth_abs)

        S_fl = -(M * g / 2) * brake * beta_brake
        S_fr = -(M * g / 2) * brake * beta_brake
        S_rl = (M * g / 2) * (throttle - brake * (1 - beta_brake))
        S_rr = (M * g / 2) * (throttle - brake * (1 - beta_brake))

        F_rr = N_rr * k_lambda * (lam + omega * (b_wb + lam * tw) / V)
        F_rl = N_rl * k_lambda * (lam + omega * (b_wb - lam * tw) / V)
        F_fr = N_fr * k_lambda * (lam + delta - omega * (a_wb - lam * tw) / V)
        F_fl = N_fl * k_lambda * (lam + delta - omega * (a_wb + lam * tw) / V)

        S_all = S_fl + S_fr + S_rl + S_rr

        # Friction circle: (S/N*mu)^2 + (F/N*mu)^2 <= 1
        c_fl = (S_fl / (N_fl * mu0)) ** 2 + (F_fl / (N_fl * mu0)) ** 2
        c_fr = (S_fr / (N_fr * mu0)) ** 2 + (F_fr / (N_fr * mu0)) ** 2
        c_rl = (S_rl / (N_rl * mu0)) ** 2 + (F_rl / (N_rl * mu0)) ** 2
        c_rr = (S_rr / (N_rr * mu0)) ** 2 + (F_rr / (N_rr * mu0)) ** 2

        # Power limit: V * total_drive_force / P_max <= 1
        return c_fl, c_fr, c_rl, c_rr, V * S_all / P_max

    def compute(self):
        c_fl, c_fr, c_rl, c_rr, power_norm = self._node_constraints(
            self.inputs["q"], self.inputs["u"]
        )
        self.constraints["c_fl"] = c_fl
        self.constraints["c_fr"] = c_fr
        self.constraints["c_rl"] = c_rl
        self.constraints["c_rr"] = c_rr
        self.constraints["c_pow"] = power_norm


# ===========================================================================
# Component 3: Pin the initial time to zero
# ===========================================================================
class InitialTime(am.Component):
    """Constraint: t(s=0) = 0. Removes the time-shift degree of freedom."""

    def __init__(self):
        super().__init__()
        self.add_input("t")
        self.add_constraint("res", shape=1)

    def compute(self):
        self.constraints["res"] = [self.inputs["t"]]


# ===========================================================================
# Component 4: Objective (minimize final time)
# ===========================================================================
class LapTimeObjective(am.Component):
    """Objective: minimize t(s=s_final), i.e. the total lap time."""

    def __init__(self):
        super().__init__()
        self.add_input("t_final")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["t_final"]


# ===========================================================================
# Component 5: Steering smoothness regularization
# ===========================================================================
class DeltaSmoothing(am.Component):
    """
    Adds a small penalty on steering rate to the objective:
        reg = EPS_DELTA * (delta_{k+1} - delta_k)^2

    Without this, the optimizer can produce a chattering steering input
    that is technically feasible but physically unrealistic.
    """

    def __init__(self):
        super().__init__()
        self.add_input("delta_left")
        self.add_input("delta_right")
        self.add_objective("reg")

    def compute(self):
        diff = self.inputs["delta_right"] - self.inputs["delta_left"]
        self.objective["reg"] = EPS_DELTA * diff * diff


# ===========================================================================
# Model assembly: connect all components with model.link()
# ===========================================================================
def create_racecar_model(module_name="racecar_mod", num_intervals=300, ds=1.0):
    """
    Build the Amigo model for the single-track lap time problem.

    The model has 5 component types:
      colloc  (num_intervals copies)  dynamics + trapezoidal collocation
      nc      (num_nodes copies)      friction circle + power constraints
      ic      (1 copy)                pin t(0) = 0
      obj     (1 copy)                minimize t_final
      smooth  (num_intervals copies)  steering smoothness penalty

    model.link(a, b) tells Amigo that input "a" and input "b" are the
    SAME optimization variable. This enforces continuity between intervals
    and connects the collocation states to the node constraints.
    """
    n_states, n_controls = 8, 2

    model = am.Model(module_name)
    model.add_component("colloc", num_intervals, RacecarCollocation(scaling, ds))
    model.add_component("nc", num_nodes, NodeConstraints(scaling))
    model.add_component("ic", 1, InitialTime())
    model.add_component("obj", 1, LapTimeObjective())
    model.add_component("smooth", num_intervals, DeltaSmoothing())

    # Continuity between intervals
    # The right endpoint of interval k (q2[k]) must equal the left
    # endpoint of interval k+1 (q1[k+1]). Same for controls.
    for i in range(n_states):
        model.link(f"colloc.q2[:{num_intervals-1}, {i}]", f"colloc.q1[1:, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u2[:{num_intervals-1}, {i}]", f"colloc.u1[1:, {i}]")

    # Periodic (cyclic) boundary conditions
    # For a closed circuit the state at the end of the lap must match
    # the state at the start. We link all states EXCEPT time (index 0),
    # because time increases monotonically around the lap.
    for i in [1, 2, 3, 4, 5, 6, 7]:
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"colloc.q1[0, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"colloc.u1[0, {i}]")

    # Objective and initial time
    model.link("colloc.q1[0, 0]", "ic.t[0]")
    model.link(f"colloc.q2[{num_intervals-1}, 0]", "obj.t_final[0]")

    # Steering smoothness
    model.link("colloc.u1[:, 0]", "smooth.delta_left")
    model.link("colloc.u2[:, 0]", "smooth.delta_right")

    # Node constraints at all N+1 nodes
    # The first N nodes come from q1 of each interval.
    # The last node (index N) comes from q2 of the last interval.
    for i in range(n_states):
        model.link(f"colloc.q1[:, {i}]", f"nc.q[:{num_intervals}, {i}]")
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"nc.q[{num_intervals}, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u1[:, {i}]", f"nc.u[:{num_intervals}, {i}]")
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"nc.u[{num_intervals}, {i}]")

    return model


# ===========================================================================
# Main: build, initialize, set data, set initial guess, set bounds, solve
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true")
args = parser.parse_args()

# Build and initialize the model
model = create_racecar_model(num_intervals=num_intervals, ds=ds)
if args.build:
    model.build_module(source_dir=Path(__file__).resolve().parent)

model.initialize(order_type=am.OrderingType.AMD)
print(f"Variables: {model.num_variables}, Constraints: {model.num_constraints}")

# Set curvature data
# kappa is not an optimization variable; it is fixed track geometry.
# We pass it through the data vector so each collocation interval
# knows the curvature at its left (kappa1) and right (kappa2) nodes.
data = model.get_data_vector()
for i in range(num_intervals):
    data[f"colloc.kappa1[{i}]"] = kappa_nodes[i]
    data[f"colloc.kappa2[{i}]"] = kappa_nodes[i + 1]

# Initial guess
# A good initial guess is critical for convergence.
# We start with constant speed V_init driving straight along the centerline.
x = model.create_vector()
x[:] = 0.0
V_init = 20.0
t_est = s_final / V_init  # estimated lap time at constant speed

# Time increases linearly from 0 to t_est (in scaled units)
x["colloc.q1[:, 0]"] = (
    np.linspace(0, t_est * (num_intervals - 1) / num_intervals, num_intervals)
    / scaling["t"]
)
x["colloc.q2[:, 0]"] = (
    np.linspace(t_est / num_intervals, t_est, num_intervals) / scaling["t"]
)
# Constant speed (in scaled units)
x["colloc.q1[:, 2]"] = V_init / scaling["V"]
x["colloc.q2[:, 2]"] = V_init / scaling["V"]
# Small positive thrust to avoid starting at zero
x["colloc.u1[:, 1]"] = 0.1
x["colloc.u2[:, 1]"] = 0.1

# Variable bounds
# Amigo uses box bounds: lower <= x <= upper for each variable.
# Because model.link() merges variables via union-find, we must set
# bounds on EVERY name that refers to a given variable. If any alias
# has a tighter bound, that bound wins.
lower = model.create_vector()
upper = model.create_vector()

# Default: all variables unbounded
for end in ["1", "2"]:
    lower[f"colloc.q{end}"] = -float("inf")
    upper[f"colloc.q{end}"] = float("inf")
    lower[f"colloc.u{end}"] = -float("inf")
    upper[f"colloc.u{end}"] = float("inf")

# Linked aliases must also be set to +/-inf so they don't accidentally
# impose a zero bound (the default vector value).
lower["obj.t_final"] = -float("inf")
upper["obj.t_final"] = float("inf")
lower["ic.t"] = -float("inf")
upper["ic.t"] = float("inf")
lower["nc.q"] = -float("inf")
upper["nc.q"] = float("inf")
lower["nc.u"] = -float("inf")
upper["nc.u"] = float("inf")
lower["smooth.delta_left"] = -float("inf")
upper["smooth.delta_left"] = float("inf")
lower["smooth.delta_right"] = -float("inf")
upper["smooth.delta_right"] = float("inf")

# Speed must be positive (V > 1 m/s to avoid division by zero in dynamics)
lower["colloc.q1[:, 2]"] = 1.0 / scaling["V"]
lower["colloc.q2[:, 2]"] = 1.0 / scaling["V"]

# Track boundary + curvature singularity bounds on lateral offset n
# The car must stay between the left and right track edges: -w_right <= n <= w_left.
# Additionally, the term (1 - n*kappa) in the denominator of sdot must stay
# positive, so we tighten the bound to avoid the singularity n = 1/kappa.
kappa_margin = 0.9  # keep n at most 90% of the way to the singularity
for end, w_r, w_l, kn in [
    (
        "1",
        track.w_right[:num_intervals],
        track.w_left[:num_intervals],
        kappa_nodes[:num_intervals],
    ),
    ("2", track.w_right[1:], track.w_left[1:], kappa_nodes[1:]),
]:
    n_lo = -w_r / scaling["n"]
    n_hi = w_l / scaling["n"]
    for i in range(num_intervals):
        k = kn[i]
        if abs(k) > 1e-6:
            n_sing = kappa_margin / (k * scaling["n"])
            if k > 0:
                n_hi[i] = min(n_hi[i], n_sing)
            else:
                n_lo[i] = max(n_lo[i], n_sing)
    lower[f"colloc.q{end}[:, 1]"] = n_lo
    upper[f"colloc.q{end}[:, 1]"] = n_hi

# Heading, sideslip, and steering angle bounds
alpha_max = (np.pi / 3) / scaling["alpha"]
lam_max = 0.25 / scaling["lam"]
delta_max = 0.5 / scaling["delta"]
for end in ["1", "2"]:
    lower[f"colloc.q{end}[:, 3]"] = -alpha_max
    upper[f"colloc.q{end}[:, 3]"] = alpha_max
    lower[f"colloc.q{end}[:, 4]"] = -lam_max
    upper[f"colloc.q{end}[:, 4]"] = lam_max
    lower[f"colloc.u{end}[:, 0]"] = -delta_max
    upper[f"colloc.u{end}[:, 0]"] = delta_max

# Same bounds on nc aliases (the last node is linked to colloc.q2[N-1])
lower["nc.q[:, 2]"] = 1.0 / scaling["V"]
nc_n_lo = -track.w_right / scaling["n"]
nc_n_hi = track.w_left / scaling["n"]
for i in range(num_nodes):
    k = kappa_nodes[i]
    if abs(k) > 1e-6:
        n_sing = kappa_margin / (k * scaling["n"])
        if k > 0:
            nc_n_hi[i] = min(nc_n_hi[i], n_sing)
        else:
            nc_n_lo[i] = max(nc_n_lo[i], n_sing)
lower["nc.q[:, 1]"] = nc_n_lo
upper["nc.q[:, 1]"] = nc_n_hi
lower["nc.q[:, 3]"] = -alpha_max
upper["nc.q[:, 3]"] = alpha_max
lower["nc.q[:, 4]"] = -lam_max
upper["nc.q[:, 4]"] = lam_max
lower["nc.u[:, 0]"] = -delta_max
upper["nc.u[:, 0]"] = delta_max

# ===========================================================================
# Solve
# ===========================================================================
opt = am.Optimizer(model, x, lower=lower, upper=upper)
print(f"\nOptimizing (V_init={V_init:.0f} m/s, t_est={t_est:.1f} s)...")
opt_data = opt.optimize(
    {
        "initial_barrier_param": 1e-1,
        "max_iterations": 300,
        "max_line_search_iterations": 30,
        "convergence_tolerance": 1e-8,
        "init_least_squares_multipliers": True,
        "barrier_strategy": "monotone",
        "filter_line_search": True,
    }
)

# ===========================================================================
# Extract and print results
# ===========================================================================
print(f"\nConverged: {opt_data['converged']}")
print(f"Iterations: {len(opt_data['iterations'])}")

# Reconstruct full trajectory arrays (N+1 nodes)
# Node 0 comes from q1[0], nodes 1..N come from q2[0..N-1]
q_traj = np.vstack(
    [np.array(x["colloc.q1[0]"]).reshape(1, 8), np.array(x["colloc.q2"])]
)
u_traj = np.vstack(
    [np.array(x["colloc.u1[0]"]).reshape(1, 2), np.array(x["colloc.u2"])]
)

# Unscale to physical units
t_sol = q_traj[:, 0] * scaling["t"]
n_sol = q_traj[:, 1] * scaling["n"]
V_sol = q_traj[:, 2] * scaling["V"]
alpha_sol = q_traj[:, 3] * scaling["alpha"]
lam_sol = q_traj[:, 4] * scaling["lam"]
omega_sol = q_traj[:, 5] * scaling["omega"]
delta_sol = u_traj[:, 0] * scaling["delta"]
thrust_sol = u_traj[:, 1]

print(f"\nLap time: {t_sol[-1]:.4f} s")
print(f"Speed: [{V_sol.min():.2f}, {V_sol.max():.2f}] m/s")
print(f"Lateral offset: [{n_sol.min():.3f}, {n_sol.max():.3f}] m")
print(
    f"Steering: [{np.degrees(delta_sol.min()):.2f}, {np.degrees(delta_sol.max()):.2f}] deg"
)
print(f"Thrust: [{thrust_sol.min():.3f}, {thrust_sol.max():.3f}]")

# Check periodicity: start and end values should match for cyclic states
print(f"\nPeriodicity (start vs end):")
for name, vals in [
    ("V", V_sol),
    ("n", n_sol),
    ("alpha", alpha_sol),
    ("omega", omega_sol),
]:
    print(f"  {name}: {vals[0]:.6f} vs {vals[-1]:.6f}")

# Check inequality constraints are satisfied (all should be <= 1)
c_fl_sol = np.array(x["nc.c_fl"])
c_fr_sol = np.array(x["nc.c_fr"])
c_rl_sol = np.array(x["nc.c_rl"])
c_rr_sol = np.array(x["nc.c_rr"])
c_pow_sol = np.array(x["nc.c_pow"])

print(f"\nConstraint max (should be <= 1):")
for name, vals in [
    ("FL", c_fl_sol),
    ("FR", c_fr_sol),
    ("RL", c_rl_sol),
    ("RR", c_rr_sol),
    ("Pow", c_pow_sol),
]:
    print(f"  {name}: {vals.max():.4f}")

with open("racecar_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)


# ===========================================================================
# Plotting: track map (color-coded) + telemetry strips
# ===========================================================================
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def build_track_edges(x_c, y_c, s_c, w_right, w_left):
    """Offset the centerline left/right by the track half-widths."""
    dx, dy = np.gradient(x_c, s_c), np.gradient(y_c, s_c)
    mag = np.sqrt(dx**2 + dy**2)
    nx, ny = -dy / mag, dx / mag
    return x_c + w_left * nx, y_c + w_left * ny, x_c - w_right * nx, y_c - w_right * ny


def racing_line_xy(s_sol, n_sol, s_c, x_c, y_c):
    """Convert curvilinear (s, n) to Cartesian (x, y) for the racing line."""
    xc = np.interp(s_sol, s_c, x_c)
    yc = np.interp(s_sol, s_c, y_c)
    dx, dy = np.gradient(xc, s_sol), np.gradient(yc, s_sol)
    mag = np.sqrt(dx**2 + dy**2)
    return xc + n_sol * (-dy / mag), yc + n_sol * (dx / mag)


try:
    x_left, y_left, x_right, y_right = build_track_edges(
        track.raw_x, track.raw_y, track.raw_s, track.raw_w_right, track.raw_w_left
    )

    # Interpolate solution onto a finer grid for smooth color maps
    s_fine = np.linspace(0, s_final, 500)
    n_fine = np.interp(s_fine, s_nodes, n_sol)
    V_fine = np.interp(s_fine, s_nodes, V_sol)
    thrust_fine = np.interp(s_fine, s_nodes, thrust_sol)
    delta_fine = np.interp(s_fine, s_nodes, delta_sol)
    x_race, y_race = racing_line_xy(
        s_fine, n_fine, track.raw_s, track.raw_x, track.raw_y
    )

    # Track map plots (racing line colored by velocity, thrust, steering)
    for color_val, cmap, label, fname in [
        (V_fine, "viridis", "V (m/s)", "racecar_track_velocity.png"),
        (thrust_fine, "RdYlGn", "thrust", "racecar_track_thrust.png"),
        (delta_fine, "coolwarm", "delta (rad)", "racecar_track_steering.png"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.fill(
            np.append(x_left, x_right[::-1]),
            np.append(y_left, y_right[::-1]),
            color="0.92",
            zorder=0,
        )
        ax.plot(x_left, y_left, "k-", lw=1, alpha=0.7)
        ax.plot(x_right, y_right, "k-", lw=1, alpha=0.7)
        pts = np.column_stack([x_race, y_race]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(
            segs,
            cmap=cmap,
            norm=Normalize(color_val.min(), color_val.max()),
            lw=3,
            zorder=5,
        )
        lc.set_array(color_val)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label=label, shrink=0.8)
        ax.set(
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{track.name} ({t_sol[-1]:.2f} s)",
            aspect="equal",
        )
        ax.autoscale_view()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")

    # Telemetry plot (5 strips vs arc-length)
    fig4, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(s_nodes, V_sol, "b-", lw=1.5)
    axes[0].set_ylabel("V (m/s)")
    axes[0].set_title(f"{track.name} - Telemetry")
    axes[1].plot(s_nodes, n_sol, "r-", lw=1.5)
    axes[1].plot(s_nodes, track.w_left, "k--", alpha=0.3)
    axes[1].plot(s_nodes, -track.w_right, "k--", alpha=0.3)
    axes[1].set_ylabel("n (m)")
    axes[2].plot(s_nodes, thrust_sol, "g-", lw=1.5)
    axes[2].axhline(0, color="k", alpha=0.2)
    axes[2].set_ylabel("thrust")
    axes[3].plot(s_nodes, delta_sol, "m-", lw=1.5)
    axes[3].set_ylabel("delta (rad)")
    axes[4].plot(s_nodes, c_fl_sol, label="FL", alpha=0.7)
    axes[4].plot(s_nodes, c_fr_sol, label="FR", alpha=0.7)
    axes[4].plot(s_nodes, c_rl_sol, label="RL", alpha=0.7)
    axes[4].plot(s_nodes, c_rr_sol, label="RR", alpha=0.7)
    axes[4].plot(s_nodes, c_pow_sol, "k-", label="power", lw=1.5, alpha=0.8)
    axes[4].axhline(1, color="r", ls="--", lw=1.5, label="limit")
    axes[4].set_ylabel("constraint")
    axes[4].set_xlabel("s (m)")
    axes[4].legend(loc="upper right", fontsize=8, ncol=3)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("racecar_telemetry.png", dpi=150)
    print("Saved racecar_telemetry.png")

    plt.show()

except Exception as e:
    import traceback

    print(f"Could not create plots: {e}")
    traceback.print_exc()
