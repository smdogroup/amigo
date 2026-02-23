"""
Racecar Minimum Lap Time Optimization

Based on the Dymos racecar example from OpenMDAO.

Minimize lap time on a closed circuit subject to:
- Vehicle dynamics (curvilinear coordinates)
- Tire friction circle constraints (4 wheels)
- Engine power limit
- Track boundary constraints (variable width from CSV data)
- Cyclic (periodic) boundary conditions

Independent variable: arc length s along track centerline
State q (scaled, shape=8): [t, n, V, alpha, lam, omega, ax, ay]
Control u (shape=2):        [delta, thrust]

RacecarCollocation: q2 - q1 - 0.5*ds*(f(q1,u1) + f(q2,u2)) = 0  (per interval)
  _dynamics(q, u, kappa) isolates the physics: returns 8 scaled d/ds rates.
  delta is a direct control u[0] (no integration, no trapezoidal null-space).
"""

import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

from tracks import berlin_2018

# Vehicle parameters
M = 800.0  # Vehicle mass [kg]
g = 9.8  # Gravity [m/s^2]
a_wb = 1.8  # CG to front axle [m]
b_wb = 1.6  # CG to rear axle [m]
L_wb = a_wb + b_wb
tw = 0.73  # Half track width [m]
Iz = 450.0  # Yaw moment of inertia [kg*m^2]
rho = 1.2  # Air density [kg/m^3]
CdA = 2.0  # Drag coeff * frontal area [m^2]
ClA = 4.0  # Lift (downforce) coeff * area [m^2]
CoP = 1.6  # Center of pressure to front axle [m]
h_cg = 0.3  # CG height [m]
chi = 0.5  # Roll stiffness distribution (front fraction)
beta_brake = 0.62  # Brake bias (front fraction)
k_lambda = 44.0  # Tire lateral stiffness
mu0 = 1.68  # Tire friction coefficient
tau_x = 0.2  # Longitudinal load transfer time constant [s]
tau_y = 0.2  # Lateral load transfer time constant [s]
P_max = 960000.0  # Maximum engine power [W]
EPS_THRUST = 0.00001  # Smooth abs epsilon for thrust/brake split

# Scaling factors: q[i] = q_physical / scaling[name]
# q layout: [t, n, V, alpha, lam, omega, ax, ay]
# u layout: [delta, thrust]  (thrust is unscaled)
scaling = {
    "t": 60.0,  # time [s]
    "n": 5.0,  # lateral offset [m]
    "V": 40.0,  # speed [m/s]
    "alpha": 0.15,  # heading angle [rad]
    "lam": 0.05,  # slip angle [rad]
    "omega": 0.5,  # yaw rate [rad/s]
    "ax": 8.0,  # longitudinal acceleration [m/s^2]
    "ay": 8.0,  # lateral acceleration [m/s^2]
    "delta": 0.1,  # steering angle [rad]
}

# Discretization
num_intervals = 300
num_nodes = num_intervals + 1

EPS_DELTA = 1e-3  # Tikhonov weight on steering rate (scaled units)


track = berlin_2018(num_nodes)
s_final = track.s_total
s_nodes = track.s
kappa_nodes = track.kappa
ds = s_final / num_intervals

print(f"Track: {track.name}")
print(f"Track length: {s_final:.2f} m")
print(f"Intervals: {num_intervals}, ds: {ds:.2f} m")
print(f"Curvature range: [{kappa_nodes.min():.4f}, {kappa_nodes.max():.4f}] 1/m")
print(
    f"Track width: right [{track.w_right.min():.2f}, {track.w_right.max():.2f}] m, "
    f"left [{track.w_left.min():.2f}, {track.w_left.max():.2f}] m"
)


class RacecarCollocation(am.Component):
    """Trapezoidal direct collocation for one racecar interval.

    Residual (shape=9): q2 - q1 - 0.5*ds*(f(q1,u1) + f(q2,u2)) = 0
    _dynamics(q, u, kappa) isolates the physics: returns 8 scaled d/ds rates
    for [t, n, V, alpha, lam, omega, ax, ay]. delta uses u_delta directly.
    """

    def __init__(self, scaling, ds):
        super().__init__()

        self.scaling = scaling
        self.add_constant("ds", value=ds)
        self.add_data("kappa1", value=0.0)
        self.add_data("kappa2", value=0.0)

        self.add_input("q1", shape=8)  # scaled state at left node
        self.add_input("q2", shape=8)  # scaled state at right node
        self.add_input("u1", shape=2)  # control at left node
        self.add_input("u2", shape=2)  # control at right node

        self.add_constraint("res", shape=8)

    def _dynamics(self, q, u, kappa):
        """Scaled d/ds rates for 8 ODE states: [t, n, V, alpha, lam, omega, ax, ay].

        q (scaled): [t, n, V, alpha, lam, omega, ax, ay]
        u:          [delta, thrust]  (delta is a direct control)
        returns:    8 scaled rates
        """
        V = self.scaling["V"] * q[2]
        n = self.scaling["n"] * q[1]
        alpha = self.scaling["alpha"] * q[3]
        lam = self.scaling["lam"] * q[4]
        omega = self.scaling["omega"] * q[5]
        ax_phys = self.scaling["ax"] * q[6]
        ay_phys = self.scaling["ay"] * q[7]
        delta = self.scaling["delta"] * u[0]
        thrust = u[1]

        # Normal forces
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

        # Tire forces (smooth thrust/brake split)
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

        # Car dynamics (time derivatives)
        F_all = F_fl + F_fr + F_rl + F_rr
        S_all = S_fl + S_fr + S_rl + S_rr
        F_front = F_fl + F_fr
        S_front = S_fl + S_fr
        drag = 0.5 * rho * CdA * V * V

        cos_al = am.cos(alpha - lam)
        sin_al = am.sin(alpha - lam)

        sdot = V * cos_al / (1.0 - n * kappa)
        ndot = V * sin_al
        alphadot = omega - kappa * V * cos_al / (1.0 - n * kappa)
        Vdot = S_all / M - delta * F_front / M - drag / M - omega * V * lam
        lambdadot = omega - Vdot * lam / V - delta * S_front / (M * V) - F_all / (M * V)
        omegadot = (
            (a_wb * F_front) / Iz
            - (b_wb * (F_rr + F_rl)) / Iz
            + (tw * (-S_rr + S_rl - S_fr + S_fl)) / Iz
        )

        # Acceleration dynamics (first-order lag)
        axdot = (Vdot + omega * V * lam - ax_phys) / tau_x
        aydot = (omega * V - (V * lambdadot + Vdot * lam) - ay_phys) / tau_y

        # Convert to d/ds = (d/dt) / sdot, then scale
        return (
            1.0 / (sdot * self.scaling["t"]),
            ndot / (sdot * self.scaling["n"]),
            Vdot / (sdot * self.scaling["V"]),
            alphadot / (sdot * self.scaling["alpha"]),
            lambdadot / (sdot * self.scaling["lam"]),
            omegadot / (sdot * self.scaling["omega"]),
            axdot / (sdot * self.scaling["ax"]),
            aydot / (sdot * self.scaling["ay"]),
        )

    def compute(self):
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        u1 = self.inputs["u1"]
        u2 = self.inputs["u2"]
        ds_val = self.constants["ds"]

        f1 = self._dynamics(q1, u1, self.data["kappa1"])
        f2 = self._dynamics(q2, u2, self.data["kappa2"])

        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * ds_val * (f1[i] + f2[i]) for i in range(8)
        ]


class NodeConstraints(am.Component):
    """Tire friction circle (4 wheels) + power constraint at each node.

    5 inequality constraints via slack variables: c_XX + slack_XX - 1 = 0, slack >= 0.
    _node_constraints(q, u) isolates the physics.
    """

    def __init__(self, scaling):
        super().__init__()

        self.scaling = scaling
        self.add_input("q", shape=8)
        self.add_input("u", shape=2)

        for name in ["s_fl", "s_fr", "s_rl", "s_rr", "s_pow"]:
            self.add_input(name)

        self.add_constraint("res", shape=5)

    def _node_constraints(self, q, u):
        """Friction circle utilizations (4 wheels) and normalized power (<= 1)."""
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

        c_fl = (S_fl / (N_fl * mu0)) ** 2 + (F_fl / (N_fl * mu0)) ** 2
        c_fr = (S_fr / (N_fr * mu0)) ** 2 + (F_fr / (N_fr * mu0)) ** 2
        c_rl = (S_rl / (N_rl * mu0)) ** 2 + (F_rl / (N_rl * mu0)) ** 2
        c_rr = (S_rr / (N_rr * mu0)) ** 2 + (F_rr / (N_rr * mu0)) ** 2

        return c_fl, c_fr, c_rl, c_rr, V * S_all / P_max

    def compute(self):
        c_fl, c_fr, c_rl, c_rr, power_norm = self._node_constraints(
            self.inputs["q"], self.inputs["u"]
        )
        self.constraints["res"] = [
            c_fl + self.inputs["s_fl"] - 1.0,
            c_fr + self.inputs["s_fr"] - 1.0,
            c_rl + self.inputs["s_rl"] - 1.0,
            c_rr + self.inputs["s_rr"] - 1.0,
            power_norm + self.inputs["s_pow"] - 1.0,
        ]


class InitialTime(am.Component):
    """Constraint: t(s=0) = 0."""

    def __init__(self):
        super().__init__()
        self.add_input("t")
        self.add_constraint("res", shape=1)

    def compute(self):
        self.constraints["res"] = [self.inputs["t"]]


class LapTimeObjective(am.Component):
    """Minimize total lap time: t at s = s_final."""

    def __init__(self):
        super().__init__()
        self.add_input("t_final")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["t_final"]


class DeltaSmoothing(am.Component):
    """Per-interval steering smoothness: EPS_DELTA * (delta_right - delta_left)^2.

    Instantiated num_intervals times. Each instance handles one consecutive pair.
    u2[k,0] = u1[k+1,0] via continuity links; u2[N-1,0] = u1[0,0] via cyclic BC,
    so the wrap-around term is included automatically.
    Amigo sums the objective contributions from all instances.
    """

    def __init__(self):
        super().__init__()
        self.add_input("delta_left")
        self.add_input("delta_right")
        self.add_objective("reg")

    def compute(self):
        diff = self.inputs["delta_right"] - self.inputs["delta_left"]
        self.objective["reg"] = EPS_DELTA * diff * diff


def create_racecar_model(module_name="racecar_mod", num_intervals=300, ds=1.0):
    """Build and return the racecar minimum lap time model.

    Components:
      colloc (num_intervals): RacecarCollocation — trapezoidal collocation, holds all variables
      nc     (num_nodes):     NodeConstraints    — friction + power at each node
      ic     (1):             InitialTime
      obj    (1):             LapTimeObjective

    Variable layout per collocation instance:
      q1/q2 (shape=8): [t, n, V, alpha, lam, omega, ax, ay]
      u1/u2 (shape=2): [delta, thrust]
    """
    n_states = 8
    n_controls = 2

    colloc = RacecarCollocation(scaling, ds)
    nc = NodeConstraints(scaling)
    ic_time = InitialTime()
    obj = LapTimeObjective()
    smooth = DeltaSmoothing()

    model = am.Model(module_name)
    model.add_component("colloc", num_intervals, colloc)
    model.add_component("nc", num_nodes, nc)
    model.add_component("ic", 1, ic_time)
    model.add_component("obj", 1, obj)
    model.add_component("smooth", num_intervals, smooth)

    # Interior node continuity: q2[k] = q1[k+1] for k = 0..N-2
    for i in range(n_states):
        model.link(f"colloc.q2[:{num_intervals-1}, {i}]", f"colloc.q1[1:, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u2[:{num_intervals-1}, {i}]", f"colloc.u1[1:, {i}]")

    # Cyclic BCs: final right node = first left node (all states except t, and controls)
    for i in [1, 2, 3, 4, 5, 6, 7]:  # n, V, alpha, lam, omega, ax, ay
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"colloc.q1[0, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"colloc.u1[0, {i}]")

    # Initial time and lap time objective
    model.link("colloc.q1[0, 0]", "ic.t[0]")
    model.link(f"colloc.q2[{num_intervals-1}, 0]", "obj.t_final[0]")

    # Delta smoothing: each instance gets the left and right delta of its interval.
    # u2[k,0] = u1[k+1,0] via continuity, and u2[N-1,0] = u1[0,0] via cyclic BC,
    # so the wrap-around term is included automatically at no extra cost.
    model.link("colloc.u1[:, 0]", "smooth.delta_left")
    model.link("colloc.u2[:, 0]", "smooth.delta_right")

    # Node constraints at all num_nodes points
    for i in range(n_states):
        model.link(f"colloc.q1[:, {i}]", f"nc.q[:{num_intervals}, {i}]")
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"nc.q[{num_intervals}, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u1[:, {i}]", f"nc.u[:{num_intervals}, {i}]")
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"nc.u[{num_intervals}, {i}]")

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build C++ module")
args = parser.parse_args()

model = create_racecar_model(
    module_name="racecar_mod",
    num_intervals=num_intervals,
    ds=ds,
)

if args.build:
    source_dir = Path(__file__).resolve().parent
    model.build_module(source_dir=source_dir)

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")

# Per-instance curvature data
data = model.get_data_vector()
for i in range(num_intervals):
    data[f"colloc.kappa1[{i}]"] = kappa_nodes[i]
    data[f"colloc.kappa2[{i}]"] = kappa_nodes[i + 1]

# Initial guess
x = model.create_vector()
x[:] = 0.0

V_init = 20.0
t_est = s_final / V_init

t_left = np.linspace(0, t_est * (num_intervals - 1) / num_intervals, num_intervals)
t_right = np.linspace(t_est / num_intervals, t_est, num_intervals)

x["colloc.q1[:, 0]"] = t_left / scaling["t"]
x["colloc.q2[:, 0]"] = t_right / scaling["t"]
x["colloc.q1[:, 2]"] = V_init / scaling["V"]
x["colloc.q2[:, 2]"] = V_init / scaling["V"]
x["colloc.u1[:, 1]"] = 0.1  # thrust
x["colloc.u2[:, 1]"] = 0.1

x["nc.s_fl"] = 0.9
x["nc.s_fr"] = 0.9
x["nc.s_rl"] = 0.9
x["nc.s_rr"] = 0.9
x["nc.s_pow"] = 0.9

print(f"\nInitial guess: V = {V_init:.1f} m/s, estimated lap time = {t_est:.1f} s")

# Bounds
lower = model.create_vector()
upper = model.create_vector()

w_right_1 = track.w_right[:num_intervals]
w_left_1 = track.w_left[:num_intervals]
w_right_2 = track.w_right[1:]
w_left_2 = track.w_left[1:]

for end in ["1", "2"]:
    lower[f"colloc.q{end}"] = -float("inf")
    upper[f"colloc.q{end}"] = float("inf")
    lower[f"colloc.u{end}"] = -float("inf")
    upper[f"colloc.u{end}"] = float("inf")

# Speed: must stay positive
lower["colloc.q1[:, 2]"] = 1.0 / scaling["V"]
lower["colloc.q2[:, 2]"] = 1.0 / scaling["V"]

# Track width bounds on lateral offset n (per-node, variable)
lower["colloc.q1[:, 1]"] = -w_right_1 / scaling["n"]
upper["colloc.q1[:, 1]"] = w_left_1 / scaling["n"]
lower["colloc.q2[:, 1]"] = -w_right_2 / scaling["n"]
upper["colloc.q2[:, 1]"] = w_left_2 / scaling["n"]

lower["obj.t_final"] = -float("inf")
upper["obj.t_final"] = float("inf")
lower["ic.t"] = -float("inf")
upper["ic.t"] = float("inf")

lower["nc.q"] = -float("inf")
upper["nc.q"] = float("inf")
lower["nc.u"] = -float("inf")
upper["nc.u"] = float("inf")
lower["nc.q[:, 2]"] = 1.0 / scaling["V"]  # V > 0 at all nodes
lower["nc.q[:, 1]"] = -track.w_right / scaling["n"]
upper["nc.q[:, 1]"] = track.w_left / scaling["n"]

for s_name in ["s_fl", "s_fr", "s_rl", "s_rr"]:
    lower[f"nc.{s_name}"] = 0.0
    upper[f"nc.{s_name}"] = 2.0

lower["nc.s_pow"] = 0.0
upper["nc.s_pow"] = 3.0

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper)

print(f"\nOptimizing...")
opt_data = opt.optimize(
    {
        "barrier_strategy": "quality_function",
        "initial_barrier_param": 0.1,
        "monotone_barrier_fraction": 0.2,
        "max_iterations": 1000,
        "fraction_to_boundary": 0.995,
        "use_armijo_line_search": False,
        "max_line_search_iterations": 30,
        "init_least_squares_multipliers": True,
        "acceptable_tol": 1e-7,
        "acceptable_iter": 15,
    }
)

# Results
print(f"\nConverged: {opt_data['converged']}")
print(f"Iterations: {len(opt_data['iterations'])}")
print(f"Final residual: {opt_data['iterations'][-1]['residual']:.6e}")

# Reconstruct N+1-point trajectories: first left node + all right nodes
q_traj = np.vstack(
    [
        np.array(x["colloc.q1[0]"]).reshape(1, 8),
        np.array(x["colloc.q2"]),
    ]
)  # (num_nodes, 8)

u_traj = np.vstack(
    [
        np.array(x["colloc.u1[0]"]).reshape(1, 2),
        np.array(x["colloc.u2"]),
    ]
)  # (num_nodes, 2)

t_sol = q_traj[:, 0] * scaling["t"]
n_sol = q_traj[:, 1] * scaling["n"]
V_sol = q_traj[:, 2] * scaling["V"]
alpha_sol = q_traj[:, 3] * scaling["alpha"]
lam_sol = q_traj[:, 4] * scaling["lam"]
omega_sol = q_traj[:, 5] * scaling["omega"]
ax_sol = q_traj[:, 6] * scaling["ax"]
ay_sol = q_traj[:, 7] * scaling["ay"]
delta_sol = u_traj[:, 0] * scaling["delta"]
thrust_sol = u_traj[:, 1]

print(f"\nLap time: {t_sol[-1]:.4f} s")
print(f"Speed range: [{V_sol.min():.2f}, {V_sol.max():.2f}] m/s")
print(f"Lateral offset range: [{n_sol.min():.3f}, {n_sol.max():.3f}] m")
print(
    f"Steering range: [{np.degrees(delta_sol.min()):.2f}, {np.degrees(delta_sol.max()):.2f}] deg"
)
print(f"Thrust range: [{thrust_sol.min():.3f}, {thrust_sol.max():.3f}]")

print(f"\nPeriodicity check (start vs end):")
print(f"  V:     {V_sol[0]:.4f} vs {V_sol[-1]:.4f} m/s")
print(f"  n:     {n_sol[0]:.4f} vs {n_sol[-1]:.4f} m")
print(f"  alpha: {alpha_sol[0]:.6f} vs {alpha_sol[-1]:.6f} rad")
print(f"  omega: {omega_sol[0]:.6f} vs {omega_sol[-1]:.6f} rad/s")

s_fl_sol = np.array(x["nc.s_fl"])
s_fr_sol = np.array(x["nc.s_fr"])
s_rl_sol = np.array(x["nc.s_rl"])
s_rr_sol = np.array(x["nc.s_rr"])
s_pow_sol = np.array(x["nc.s_pow"])

print(f"\nConstraint slack minimums (should be >= 0):")
print(f"  Tire FL: {s_fl_sol.min():.4f}")
print(f"  Tire FR: {s_fr_sol.min():.4f}")
print(f"  Tire RL: {s_rl_sol.min():.4f}")
print(f"  Tire RR: {s_rr_sol.min():.4f}")
print(f"  Power:   {s_pow_sol.min():.4f}")

with open("racecar_opt_data.json", "w") as fp:
    json.dump(opt_data, fp, indent=2)


# Plotting
def build_track_edges(x_c, y_c, s_c, w_right, w_left):
    """Compute track edges from centerline and per-point widths."""
    dx = np.gradient(x_c, s_c)
    dy = np.gradient(y_c, s_c)
    mag = np.sqrt(dx**2 + dy**2)
    nx = -dy / mag
    ny = dx / mag
    return x_c + w_left * nx, y_c + w_left * ny, x_c - w_right * nx, y_c - w_right * ny


def racing_line_xy(s_sol, n_sol, s_c, x_c, y_c):
    """Compute x-y positions of the racing line."""
    x_center = np.interp(s_sol, s_c, x_c)
    y_center = np.interp(s_sol, s_c, y_c)
    dx = np.gradient(x_center, s_sol)
    dy = np.gradient(y_center, s_sol)
    mag = np.sqrt(dx**2 + dy**2)
    return x_center + n_sol * (-dy / mag), y_center + n_sol * (dx / mag)


def plot_track_colored(
    ax,
    x_left,
    y_left,
    x_right,
    y_right,
    x_race,
    y_race,
    color_values,
    cmap,
    label,
    title,
):
    """Plot track surface, edges, and racing line colored by a scalar field."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    ax.fill(
        np.append(x_left, x_right[::-1]),
        np.append(y_left, y_right[::-1]),
        color="0.92",
        zorder=0,
    )
    ax.plot(x_left, y_left, "k-", linewidth=1, alpha=0.7)
    ax.plot(x_right, y_right, "k-", linewidth=1, alpha=0.7)

    points = np.column_stack([x_race, y_race]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, zorder=5)
    lc.set_array(color_values)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label=label, shrink=0.8)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
    ax.autoscale_view()
    ax.grid(True, alpha=0.2)


try:
    x_left, y_left, x_right, y_right = build_track_edges(
        track.raw_x, track.raw_y, track.raw_s, track.raw_w_right, track.raw_w_left
    )

    s_fine = np.linspace(0, s_final, 500)
    n_fine = np.interp(s_fine, s_nodes, n_sol)
    V_fine = np.interp(s_fine, s_nodes, V_sol)
    thrust_fine = np.interp(s_fine, s_nodes, thrust_sol)
    delta_fine = np.interp(s_fine, s_nodes, delta_sol)
    x_race, y_race = racing_line_xy(
        s_fine, n_fine, track.raw_s, track.raw_x, track.raw_y
    )

    for color_val, cmap, label, fname in [
        (V_fine, "viridis", "V (m/s)", "racecar_track_velocity.png"),
        (thrust_fine, "RdYlGn", "thrust", "racecar_track_thrust.png"),
        (delta_fine, "coolwarm", "delta (rad)", "racecar_track_steering.png"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot_track_colored(
            ax,
            x_left,
            y_left,
            x_right,
            y_right,
            x_race,
            y_race,
            color_val,
            cmap,
            label,
            f"{track.name} ({t_sol[-1]:.2f} s)",
        )
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        print(f"Saved {fname}")

    c_fl = 1.0 - s_fl_sol
    c_fr = 1.0 - s_fr_sol
    c_rl = 1.0 - s_rl_sol
    c_rr = 1.0 - s_rr_sol
    power_util = 1.0 - s_pow_sol

    fig4, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(s_nodes, V_sol, "b-", lw=1.5)
    axes[0].set_ylabel("V (m/s)")
    axes[0].set_title(f"{track.name} - Telemetry")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(s_nodes, n_sol, "r-", lw=1.5)
    axes[1].plot(s_nodes, track.w_left, "k--", alpha=0.3, label="track edge")
    axes[1].plot(s_nodes, -track.w_right, "k--", alpha=0.3)
    axes[1].set_ylabel("n (m)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(s_nodes, thrust_sol, "g-", lw=1.5)
    axes[2].axhline(y=0, color="k", alpha=0.2)
    axes[2].set_ylabel("thrust")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(s_nodes, delta_sol, "m-", lw=1.5)
    axes[3].set_ylabel("delta (rad)")
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(s_nodes, c_fl, label="c_fl", alpha=0.7)
    axes[4].plot(s_nodes, c_fr, label="c_fr", alpha=0.7)
    axes[4].plot(s_nodes, c_rl, label="c_rl", alpha=0.7)
    axes[4].plot(s_nodes, c_rr, label="c_rr", alpha=0.7)
    axes[4].plot(s_nodes, power_util, "k-", label="power/Pmax", lw=1.5, alpha=0.8)
    axes[4].axhline(y=1, color="r", ls="--", lw=1.5, label="Limit")
    axes[4].set_ylabel("Constraint")
    axes[4].set_xlabel("s (m)")
    axes[4].legend(loc="upper right", fontsize=8, ncol=3)
    axes[4].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("racecar_telemetry.png", dpi=150)
    print("Saved racecar_telemetry.png")

    plt.show()

except Exception as e:
    import traceback

    print(f"Could not create plots: {e}")
    traceback.print_exc()
