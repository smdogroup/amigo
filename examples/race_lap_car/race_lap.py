"""
Racecar Minimum Lap Time Optimization

Based on the Dymos racecar example from OpenMDAO.

Minimize lap time on a closed circuit subject to:
- Vehicle dynamics (curvilinear coordinates)
- Tire friction circle constraints (4 wheels)
- Engine power limit
- Track boundary constraints (variable width)
- Cyclic (periodic) boundary conditions

Independent variable: arc length s along track centerline
State q (scaled, shape=8): [t, n, V, alpha, lam, omega, ax, ay]
Control u (shape=2):        [delta, thrust]
"""

import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

from tracks import oval_dymos

# Vehicle parameters
M = 800.0
g = 9.8
a_wb = 1.8
b_wb = 1.6
L_wb = a_wb + b_wb
tw = 0.73
Iz = 450.0
rho = 1.2
CdA = 2.0
ClA = 4.0
CoP = 1.6
h_cg = 0.3
chi = 0.5
beta_brake = 0.62
k_lambda = 44.0
mu0 = 1.68
tau_x = 0.2
tau_y = 0.2
P_max = 960000.0
EPS_THRUST = 0.00001

scaling = {
    "t": 60.0,
    "n": 5.0,
    "V": 40.0,
    "alpha": 0.15,
    "lam": 0.05,
    "omega": 0.5,
    "ax": 8.0,
    "ay": 8.0,
    "delta": 0.1,
}

num_intervals = 300
num_nodes = num_intervals + 1
EPS_DELTA = 1e-3

track = oval_dymos(num_nodes)
s_final = track.s_total
s_nodes = track.s
kappa_nodes = track.kappa
ds = s_final / num_intervals

print(f"Track: {track.name}")
print(f"Track length: {s_final:.2f} m, intervals: {num_intervals}, ds: {ds:.2f} m")
print(f"Curvature: [{kappa_nodes.min():.4f}, {kappa_nodes.max():.4f}] 1/m")


class RacecarCollocation(am.Component):
    """Trapezoidal collocation: q2 - q1 - 0.5*ds*(f1 + f2) = 0."""

    def __init__(self, scaling, ds):
        super().__init__()
        self.scaling = scaling
        self.add_constant("ds", value=ds)
        self.add_data("kappa1", value=0.0)
        self.add_data("kappa2", value=0.0)
        self.add_input("q1", shape=8)
        self.add_input("q2", shape=8)
        self.add_input("u1", shape=2)
        self.add_input("u2", shape=2)
        self.add_constraint("res", shape=8)

    def _dynamics(self, q, u, kappa):
        V = self.scaling["V"] * q[2]
        n = self.scaling["n"] * q[1]
        alpha = self.scaling["alpha"] * q[3]
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
        axdot = (Vdot + omega * V * lam - ax_phys) / tau_x
        aydot = (omega * V - (V * lambdadot + Vdot * lam) - ay_phys) / tau_y

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
        q1, q2 = self.inputs["q1"], self.inputs["q2"]
        u1, u2 = self.inputs["u1"], self.inputs["u2"]
        ds_val = self.constants["ds"]
        f1 = self._dynamics(q1, u1, self.data["kappa1"])
        f2 = self._dynamics(q2, u2, self.data["kappa2"])
        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * ds_val * (f1[i] + f2[i]) for i in range(8)
        ]


class NodeConstraints(am.Component):
    """Tire friction circle (4 wheels) + power limit at each node."""

    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=8)
        self.add_input("u", shape=2)
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

        c_fl = (S_fl / (N_fl * mu0)) ** 2 + (F_fl / (N_fl * mu0)) ** 2
        c_fr = (S_fr / (N_fr * mu0)) ** 2 + (F_fr / (N_fr * mu0)) ** 2
        c_rl = (S_rl / (N_rl * mu0)) ** 2 + (F_rl / (N_rl * mu0)) ** 2
        c_rr = (S_rr / (N_rr * mu0)) ** 2 + (F_rr / (N_rr * mu0)) ** 2

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


class InitialTime(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("t")
        self.add_constraint("res", shape=1)

    def compute(self):
        self.constraints["res"] = [self.inputs["t"]]


class LapTimeObjective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("t_final")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = self.inputs["t_final"]


class DeltaSmoothing(am.Component):
    """Steering smoothness: EPS_DELTA * (delta_right - delta_left)^2."""

    def __init__(self):
        super().__init__()
        self.add_input("delta_left")
        self.add_input("delta_right")
        self.add_objective("reg")

    def compute(self):
        diff = self.inputs["delta_right"] - self.inputs["delta_left"]
        self.objective["reg"] = EPS_DELTA * diff * diff


def create_racecar_model(module_name="racecar_mod", num_intervals=300, ds=1.0):
    n_states, n_controls = 8, 2

    model = am.Model(module_name)
    model.add_component("colloc", num_intervals, RacecarCollocation(scaling, ds))
    model.add_component("nc", num_nodes, NodeConstraints(scaling))
    model.add_component("ic", 1, InitialTime())
    model.add_component("obj", 1, LapTimeObjective())
    model.add_component("smooth", num_intervals, DeltaSmoothing())

    # Interior continuity: q2[k] = q1[k+1], u2[k] = u1[k+1]
    for i in range(n_states):
        model.link(f"colloc.q2[:{num_intervals-1}, {i}]", f"colloc.q1[1:, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u2[:{num_intervals-1}, {i}]", f"colloc.u1[1:, {i}]")

    # Cyclic BCs (all states except t, and all controls)
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

    # Node constraints at all N+1 points
    for i in range(n_states):
        model.link(f"colloc.q1[:, {i}]", f"nc.q[:{num_intervals}, {i}]")
        model.link(f"colloc.q2[{num_intervals-1}, {i}]", f"nc.q[{num_intervals}, {i}]")
    for i in range(n_controls):
        model.link(f"colloc.u1[:, {i}]", f"nc.u[:{num_intervals}, {i}]")
        model.link(f"colloc.u2[{num_intervals-1}, {i}]", f"nc.u[{num_intervals}, {i}]")

    return model


# --- Main ---
parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true")
args = parser.parse_args()

model = create_racecar_model(num_intervals=num_intervals, ds=ds)
if args.build:
    model.build_module(source_dir=Path(__file__).resolve().parent)

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
print(f"Variables: {model.num_variables}, Constraints: {model.num_constraints}")

# Curvature data
data = model.get_data_vector()
for i in range(num_intervals):
    data[f"colloc.kappa1[{i}]"] = kappa_nodes[i]
    data[f"colloc.kappa2[{i}]"] = kappa_nodes[i + 1]

# Initial guess
x = model.create_vector()
x[:] = 0.0
V_init = 20.0
t_est = s_final / V_init

x["colloc.q1[:, 0]"] = (
    np.linspace(0, t_est * (num_intervals - 1) / num_intervals, num_intervals)
    / scaling["t"]
)
x["colloc.q2[:, 0]"] = (
    np.linspace(t_est / num_intervals, t_est, num_intervals) / scaling["t"]
)
x["colloc.q1[:, 2]"] = V_init / scaling["V"]
x["colloc.q2[:, 2]"] = V_init / scaling["V"]
x["colloc.u1[:, 1]"] = 0.1
x["colloc.u2[:, 1]"] = 0.1

# Bounds
lower = model.create_vector()
upper = model.create_vector()

for end in ["1", "2"]:
    lower[f"colloc.q{end}"] = -float("inf")
    upper[f"colloc.q{end}"] = float("inf")
    lower[f"colloc.u{end}"] = -float("inf")
    upper[f"colloc.u{end}"] = float("inf")

# Linked variables: link() merges indices via union-find, so these
# write to the same entries as the colloc bounds. Set ±inf to ensure
# all linked positions are unbounded by default.
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

# Speed > 0
lower["colloc.q1[:, 2]"] = 1.0 / scaling["V"]
lower["colloc.q2[:, 2]"] = 1.0 / scaling["V"]

# Track width + singularity bounds on n
kappa_margin = 0.9
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

# Heading, slip, steering bounds
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

# nc bounds: since link() merges indices, these write to the same
# entries as colloc. Set them for the last node (nc.q[300]) which
# is linked to colloc.q2[299] and needs matching bounds.
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

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper, solver="amigo")
print(f"\nOptimizing (V_init={V_init:.0f} m/s, t_est={t_est:.1f} s)...")
opt_data = opt.optimize(
    {
        "initial_barrier_param": 1.0,
        "max_iterations": 100,
        "max_line_search_iterations": 30,
        "convergence_tolerance": 1e-8,
        "init_least_squares_multipliers": True,
        "barrier_strategy": "quality_function",
        "quality_function_predictor_corrector": False,
        "quality_function_balancing_term": "cubic",
        "adaptive_mu_safeguard_factor": 1e-1,
        "filter_line_search": True,
        "verbose_barrier": True,
    }
)

# Results
print(f"\nConverged: {opt_data['converged']}")
print(f"Iterations: {len(opt_data['iterations'])}")

q_traj = np.vstack(
    [np.array(x["colloc.q1[0]"]).reshape(1, 8), np.array(x["colloc.q2"])]
)
u_traj = np.vstack(
    [np.array(x["colloc.u1[0]"]).reshape(1, 2), np.array(x["colloc.u2"])]
)

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

print(f"\nPeriodicity (start vs end):")
for name, vals in [
    ("V", V_sol),
    ("n", n_sol),
    ("alpha", alpha_sol),
    ("omega", omega_sol),
]:
    print(f"  {name}: {vals[0]:.6f} vs {vals[-1]:.6f}")

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


# --- Plotting ---
def build_track_edges(x_c, y_c, s_c, w_right, w_left):
    dx, dy = np.gradient(x_c, s_c), np.gradient(y_c, s_c)
    mag = np.sqrt(dx**2 + dy**2)
    nx, ny = -dy / mag, dx / mag
    return x_c + w_left * nx, y_c + w_left * ny, x_c - w_right * nx, y_c - w_right * ny


def racing_line_xy(s_sol, n_sol, s_c, x_c, y_c):
    xc = np.interp(s_sol, s_c, x_c)
    yc = np.interp(s_sol, s_c, y_c)
    dx, dy = np.gradient(xc, s_sol), np.gradient(yc, s_sol)
    mag = np.sqrt(dx**2 + dy**2)
    return xc + n_sol * (-dy / mag), yc + n_sol * (dx / mag)


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

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

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

    fig2, ax2 = plt.subplots(1, 1)
    ax2.semilogy(
        [it["residual"] for it in opt_data["iterations"]],
        marker="o",
        clip_on=False,
        lw=2.0,
    )
    ax2.set(xlabel="Iteration", ylabel="KKT residual norm")
    ax2.grid(True)
    fig2.savefig("racecar_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

except Exception as e:
    import traceback

    print(f"Could not create plots: {e}")
    traceback.print_exc()
