import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lgr_collocation import lgr_points, lgr_differentiation_matrix

"""
Dynamic Soaring with LGR Pseudospectral Collocation
====================================================

This uses LGR (Legendre-Gauss-Radau) collocation which is much more accurate
than trapezoidal rule. Matches the method used in GPOPS/YAPSS.

YAPSS uses: 50 segments × 6 LGR points = 300 total collocation points
"""

num_segments = 20  # Number of segments
num_lgr_points = 4  # LGR points per segment (including endpoint)


class LGRSegment(am.Component):
    """
    Single LGR collocation segment for dynamic soaring.

    This component enforces dynamics at all LGR collocation points in one segment.
    """

    def __init__(self, scaling, lgr_D):
        super().__init__()

        self.scaling = scaling
        self.num_points = lgr_D.shape[0]

        # Store differentiation matrix as constants
        for i in range(self.num_points):
            for j in range(self.num_points):
                self.add_constant(f"D_{i}_{j}", value=lgr_D[i, j])

        # Physical constants
        self.add_constant("g", value=32.2)  # ft/s^2
        self.add_constant("m", value=5.6)  # slug
        self.add_constant("S", value=45.09703)  # ft^2
        self.add_constant("rho", value=0.002378)  # slug/ft^3
        self.add_constant("CD0", value=0.00873)
        self.add_constant("K", value=0.045)
        self.add_constant("beta", value=0.08)  # Fixed for now

        # Segment timing
        self.add_input("dt")  # Segment duration (scaled)

        # States at all collocation points in this segment
        self.add_input("q", shape=(self.num_points, 6))

        # Controls at all collocation points
        self.add_input("CL", shape=(self.num_points,))
        self.add_input("phi", shape=(self.num_points,))

        # Collocation defect constraints (flattened: num_points * 6 states)
        self.add_constraint("defect", shape=(self.num_points * 6,))

        # Load factor constraints at each point
        self.add_constraint("load_factor", shape=(self.num_points,))

    def compute(self):
        # Get constants
        g = self.constants["g"]
        m = self.constants["m"]
        S = self.constants["S"]
        rho = self.constants["rho"]
        CD0 = self.constants["CD0"]
        K = self.constants["K"]
        beta = self.constants["beta"]

        # Get inputs
        dt = self.scaling["time"] * self.inputs["dt"]
        q = self.inputs["q"]  # shape: (num_points, 6)
        CL = self.inputs["CL"]  # shape: (num_points,)
        phi = self.inputs["phi"]  # shape: (num_points,)

        # Initialize output arrays (flattened)
        defect = [None] * (self.num_points * 6)
        load_factors = [None] * self.num_points

        # Scaling factors for each state
        state_scaling = [
            self.scaling["distance"],  # x
            self.scaling["distance"],  # y
            self.scaling["altitude"],  # h
            self.scaling["velocity"],  # v
            self.scaling["angle"],  # gamma
            self.scaling["angle"],  # psi
        ]

        # Compute dynamics at each collocation point (using lists)
        f = [[None for _ in range(6)] for _ in range(self.num_points)]

        for k in range(self.num_points):
            # Unscale states
            x_pos = state_scaling[0] * q[k, 0]
            y_pos = state_scaling[1] * q[k, 1]
            h = state_scaling[2] * q[k, 2]
            v = state_scaling[3] * q[k, 3]
            gamma = state_scaling[4] * q[k, 4]
            psi = state_scaling[5] * q[k, 5]

            # Trig functions
            sin_gamma = am.sin(gamma)
            cos_gamma = am.cos(gamma)
            sin_psi = am.sin(psi)
            cos_psi = am.cos(psi)
            sin_phi = am.sin(phi[k])
            cos_phi = am.cos(phi[k])

            # Wind
            Wx = beta * h
            dWx = beta
            hdot = v * sin_gamma

            # Aerodynamics
            q_dyn = 0.5 * rho * v * v
            CD = CD0 + K * CL[k] * CL[k]
            L = q_dyn * S * CL[k]
            D_aero = q_dyn * S * CD

            # Dynamics (unscaled, will be scaled in defect)
            f[k][0] = v * cos_gamma * sin_psi + Wx
            f[k][1] = v * cos_gamma * cos_psi
            f[k][2] = v * sin_gamma
            f[k][3] = -D_aero / m - g * sin_gamma - dWx * hdot * cos_gamma * sin_psi

            # Avoid division by very small v or cos_gamma
            # Add small epsilon to avoid numerical issues
            v_safe = v + 1e-6
            cos_gamma_safe = cos_gamma + 1e-8

            f[k][4] = (
                L * cos_phi - m * g * cos_gamma + m * dWx * hdot * sin_gamma * sin_psi
            ) / (m * v_safe)
            f[k][5] = (L * sin_phi - m * dWx * hdot * cos_psi) / (
                m * v_safe * cos_gamma_safe
            )

            # Load factor
            load_factors[k] = L / (m * g)

        # Compute defects using differentiation matrix
        # Time transformation: τ ∈ [-1, 1] to t ∈ [0, dt]
        # dt/dτ = dt/2, so dq/dt = (2/dt) * dq/dτ

        for i in range(self.num_points):
            for state_idx in range(6):
                # Compute dq/dτ using differentiation matrix
                dq_dtau = 0.0 * q[0, 0]  # Initialize with proper type
                for j in range(self.num_points):
                    D_ij = self.constants[f"D_{i}_{j}"]
                    dq_dtau = dq_dtau + D_ij * q[j, state_idx]

                # Convert to dq/dt: multiply by dτ/dt = 2/dt
                # Add small epsilon to avoid division by zero
                dt_safe = dt + 1e-12
                dq_dt_computed = (2.0 / dt_safe) * dq_dtau

                # Expected derivative from physics (scaled)
                dq_dt_physics = f[i][state_idx] / state_scaling[state_idx]

                # Store in flattened array
                flat_idx = i * 6 + state_idx
                defect[flat_idx] = dq_dt_computed - dq_dt_physics

        self.constraints["defect"] = defect
        self.constraints["load_factor"] = load_factors


class Objective(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("dt")  # Segment duration
        self.add_objective("obj")

    def compute(self):
        dt = self.inputs["dt"]
        # Total time = num_segments * dt
        self.objective["obj"] = num_segments * dt


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint(
            "res", shape=1
        )  # Only constrain h=0 to avoid over-constraining

    def compute(self):
        q = self.inputs["q"]
        # Only fix altitude at start - let x,y be determined by periodic constraint
        self.constraints["res"] = [q[2]]  # h=0


class PeriodicConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q0", shape=6)
        self.add_input("qf", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q0 = self.inputs["q0"]
        qf = self.inputs["qf"]
        two_pi = 2.0 * np.pi / self.scaling["angle"]

        self.constraints["res"] = [
            qf[0] - q0[0],
            qf[1] - q0[1],
            qf[2] - q0[2],
            qf[3] - q0[3],
            qf[4] - q0[4],
            qf[5] - q0[5] - two_pi,
        ]


def create_lgr_model(scaling):
    """Create model with LGR collocation segments"""

    # Get LGR points and differentiation matrix
    tau = lgr_points(num_lgr_points)
    D = lgr_differentiation_matrix(tau)

    model = am.Model("dynamic_soaring_lgr")

    # Add LGR segment components
    lgr_seg = LGRSegment(scaling, D)
    model.add_component("seg", num_segments, lgr_seg)

    # Boundary condition components
    ic = InitialConditions(scaling)
    pc = PeriodicConditions(scaling)
    obj = Objective(scaling)

    model.add_component("ic", 1, ic)
    model.add_component("pc", 1, pc)
    model.add_component("obj", 1, obj)

    # Link continuity between segments
    # seg[i].q[-1, :] == seg[i+1].q[0, :] for all internal segments
    for i in range(num_segments - 1):
        for state_idx in range(6):
            model.link(
                f"seg.q[{i}, {num_lgr_points-1}, {state_idx}]",
                f"seg.q[{i+1}, 0, {state_idx}]",
            )

    # Link all segments to have same duration
    model.link("seg.dt[1:]", "seg.dt[0]")

    # Link segment duration to objective
    model.link("seg.dt[0]", "obj.dt[0]")

    # Initial conditions (first point of first segment)
    for state_idx in range(6):
        model.link(f"seg.q[0, 0, {state_idx}]", f"ic.q[0, {state_idx}]")

    # Periodic conditions (first point of first segment vs last point of last segment)
    for state_idx in range(6):
        model.link(f"seg.q[0, 0, {state_idx}]", f"pc.q0[0, {state_idx}]")
        model.link(
            f"seg.q[{num_segments-1}, {num_lgr_points-1}, {state_idx}]",
            f"pc.qf[0, {state_idx}]",
        )

    return model, tau


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build module")
parser.add_argument("--with-openmp", action="store_true", help="Use OpenMP")
args = parser.parse_args()

# Scaling
scaling = {
    "distance": 1000.0,
    "altitude": 500.0,
    "velocity": 100.0,
    "angle": 1.0,
    "time": 100.0,
    "beta": 10.0,
}

# Create model
model, tau_lgr = create_lgr_model(scaling)

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.with_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")
print(f"Total collocation points: {num_segments * num_lgr_points}")

# Create design vector
x = model.create_vector()
x[:] = 0.0

# Initial guess
# Each segment spans τ ∈ [-1, 1], total time guess = 100s
dt_seg = 100.0 / (num_segments * scaling["time"])  # Each segment duration (scaled)

# For each segment, set states at LGR points
for seg in range(num_segments):
    seg_frac_start = seg / num_segments
    seg_frac_end = (seg + 1) / num_segments

    for pt in range(num_lgr_points):
        # Map LGR τ ∈ [-1, 1] to global time fraction
        tau_local = tau_lgr[pt]
        tau_01 = (tau_local + 1.0) / 2.0  # Map to [0, 1] within segment
        global_frac = seg_frac_start + tau_01 * (seg_frac_end - seg_frac_start)

        # Circular trajectory with smaller radius for easier problem
        theta = 2.0 * np.pi * global_frac
        radius = 300.0  # Smaller radius

        x_north = radius * (np.cos(theta) - 1.0)
        y_east = radius * np.sin(theta)
        altitude = 150.0 + 100.0 * np.sin(theta)  # Simpler: oscillate 50-250 ft

        x[f"seg.q[{seg}, {pt}, 0]"] = x_north / scaling["distance"]
        x[f"seg.q[{seg}, {pt}, 1]"] = y_east / scaling["distance"]
        x[f"seg.q[{seg}, {pt}, 2]"] = altitude / scaling["altitude"]
        x[f"seg.q[{seg}, {pt}, 3]"] = (
            120.0 / scaling["velocity"]
        )  # Lower velocity for stability
        x[f"seg.q[{seg}, {pt}, 4]"] = (
            np.radians(10.0) * np.sin(theta) / scaling["angle"]
        )  # Larger gamma variation
        x[f"seg.q[{seg}, {pt}, 5]"] = theta / scaling["angle"]

        # Controls - smoother variation
        x[f"seg.CL[{seg}, {pt}]"] = 0.8 + 0.4 * np.sin(theta)  # Vary CL
        x[f"seg.phi[{seg}, {pt}]"] = np.radians(45.0)  # Constant moderate bank

# Segment durations
x["seg.dt"] = dt_seg

# Verify the guess is valid
print(f"\nInitial guess summary:")
print(f"  Segment duration (scaled): {dt_seg}")
print(f"  Segment duration (physical): {dt_seg * scaling['time']:.2f} s")
print(f"  Total time: {dt_seg * scaling['time'] * num_segments:.2f} s")
print(f"  Sample altitude: {x['seg.q[0, 0, 2]'] * scaling['altitude']:.1f} ft")
print(f"  Sample velocity: {x['seg.q[0, 0, 3]'] * scaling['velocity']:.1f} ft/s")
print(f"  Sample CL: {x['seg.CL[0, 0]']:.2f}")
print(f"  Sample phi: {np.rad2deg(x['seg.phi[0, 0]']):.1f} deg")

# Check for NaN or inf
x_array = x.get_vector().get_array()
if np.any(np.isnan(x_array)):
    print("  ⚠ WARNING: Initial guess contains NaN!")
    nan_count = np.sum(np.isnan(x_array))
    print(f"    Found {nan_count} NaN values")
if np.any(np.isinf(x_array)):
    print("  ⚠ WARNING: Initial guess contains Inf!")
    inf_count = np.sum(np.isinf(x_array))
    print(f"    Found {inf_count} Inf values")

if not np.any(np.isnan(x_array)) and not np.any(np.isinf(x_array)):
    print("  ✓ Initial guess is valid (no NaN or Inf)")

# Set bounds
lower = model.create_vector()
upper = model.create_vector()

# State bounds (scaled)
lower["seg.q"] = -float("inf")
upper["seg.q"] = float("inf")

lower["seg.q[:, :, 2]"] = 1.0 / scaling["altitude"]  # Altitude > 0
upper["seg.q[:, :, 2]"] = 1500.0 / scaling["altitude"]

lower["seg.q[:, :, 3]"] = 5.0 / scaling["velocity"]  # Min velocity
upper["seg.q[:, :, 3]"] = 400.0 / scaling["velocity"]

# Control bounds
lower["seg.CL"] = 0.0
upper["seg.CL"] = 1.5

lower["seg.phi"] = np.radians(-90.0)
upper["seg.phi"] = np.radians(90.0)

# Time bounds (ensure dt > 0)
lower["seg.dt"] = 1.0 / scaling["time"]  # Min 1 second per segment
upper["seg.dt"] = 20.0 / scaling["time"]  # Max 20 seconds per segment

# Load factor bounds - relax slightly
lower["seg.load_factor"] = -3.0  # Allow slightly more negative
upper["seg.load_factor"] = 6.0  # Allow slightly more positive

# Optimize with MORE CONSERVATIVE settings for this difficult problem
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 100,  # Fewer iterations to see if it converges
        "initial_barrier_param": 100.0,  # MUCH higher barrier for better conditioning
        "monotone_barrier_fraction": 0.9,  # Slower barrier reduction
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-4,  # Relaxed tolerance
        "max_line_search_iterations": 3,  # Fewer line search iterations
        "init_affine_step_multipliers": False,
    }
)

with open("dynamic_soaring_lgr_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

print(f"\nOptimization complete!")
dt_opt = x["seg.dt[0]"] * scaling["time"]
print(f"Optimal time: {dt_opt * num_segments:.2f} s")

# Extract and plot solution
total_points = num_segments * num_lgr_points
t_all = np.zeros(total_points)
q_all = np.zeros((total_points, 6))
CL_all = np.zeros(total_points)
phi_all = np.zeros(total_points)

idx = 0
for seg in range(num_segments):
    dt_phys = x[f"seg.dt[{seg}]"] * scaling["time"]
    for pt in range(num_lgr_points):
        tau_local = tau_lgr[pt]
        t_all[idx] = seg * dt_phys + (tau_local + 1.0) / 2.0 * dt_phys

        for state_idx in range(6):
            q_all[idx, state_idx] = (
                x[f"seg.q[{seg}, {pt}, {state_idx}]"]
                * scaling[
                    ["distance", "distance", "altitude", "velocity", "angle", "angle"][
                        state_idx
                    ]
                ]
            )

        CL_all[idx] = x[f"seg.CL[{seg}, {pt}]"]
        phi_all[idx] = x[f"seg.phi[{seg}, {pt}]"]
        idx += 1

# Plot
fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot(q_all[:, 0], q_all[:, 1], q_all[:, 2], "b-", linewidth=2)
ax1.set_xlabel("x (ft)")
ax1.set_ylabel("y (ft)")
ax1.set_zlabel("h (ft)")
ax1.set_title("3D Flight Path (LGR)")
ax1.grid(True)

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t_all, q_all[:, 3], "b-", linewidth=2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (ft/s)")
ax2.set_title("Velocity")
ax2.grid(True)

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t_all, np.rad2deg(q_all[:, 4]), "b-", linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Flight Path Angle (deg)")
ax3.set_title("Gamma")
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t_all, np.rad2deg(q_all[:, 5]), "b-", linewidth=2)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Heading (deg)")
ax4.set_title("Psi")
ax4.grid(True)

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(t_all, CL_all, "b-", linewidth=2)
ax5.axhline(1.5, color="r", linestyle="--")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("CL")
ax5.set_title("Lift Coefficient")
ax5.grid(True)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t_all, np.rad2deg(phi_all), "b-", linewidth=2)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Bank Angle (deg)")
ax6.set_title("Phi")
ax6.grid(True)

plt.suptitle(f"Dynamic Soaring (LGR, β=0.08 1/ft)", fontsize=14)
plt.tight_layout()
plt.savefig("dynamic_soaring_lgr_results.png", dpi=300)
plt.show()

print("\nPlot saved as dynamic_soaring_lgr_results.png")
