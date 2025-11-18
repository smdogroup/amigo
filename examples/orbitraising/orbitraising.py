import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

"""
Orbit Raising Problem (Bryson & Ho)
====================================

Find the thrust direction history to transfer a spacecraft from an initial
circular orbit to the largest possible final circular orbit.

States:
    r: radial distance from attracting center
    theta: polar angle
    vr: radial velocity component
    vt: tangential velocity component

Controls:
    ur: radial component of thrust direction
    ut: tangential component of thrust direction
    
Constraint: ur² + ut² = 1 (unit thrust direction vector)

References:
    Bryson and Ho, "Applied Optimal Control", 1975
    Moyer and Pinkham, "Several Trajectory Optimization Techniques", 1964
"""

num_time_steps = 100


class TrapezoidRule(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("tf", value=3.32)  # Fixed final time (nondimensional)
        self.add_input("q1")
        self.add_input("q2")
        self.add_input("q1dot")
        self.add_input("q2dot")
        self.add_constraint("res")

    def compute(self):
        tf = self.constants["tf"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]
        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        dt = tf / num_time_steps
        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class OrbitDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Nondimensional constants
        self.add_constant("mu", value=1.0)  # Gravitational parameter
        self.add_constant("m0", value=1.0)  # Initial mass
        self.add_constant("T", value=0.1405)  # Thrust
        self.add_constant("mdot", value=0.0749)  # Fuel consumption rate

        # Time as data (varies for each component instance)
        self.add_data("t", value=0.0)

        # Inputs
        self.add_input("ur", label="control")  # Radial thrust component
        self.add_input("ut", label="control")  # Tangential thrust component
        self.add_input("q", shape=4, label="state")
        self.add_input("qdot", shape=4, label="rate")

        # Constraints
        self.add_constraint("res", shape=4, label="dynamics residual")
        self.add_constraint("thrust_unit", label="unit thrust constraint")

    def compute(self):
        mu = self.constants["mu"]
        m0 = self.constants["m0"]
        T = self.constants["T"]
        mdot = self.constants["mdot"]
        t = self.data["t"]  # Time from data

        ur = self.inputs["ur"]
        ut = self.inputs["ut"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Extract states
        r = q[0]  # radius
        theta = q[1]  # polar angle
        vr = q[2]  # radial velocity
        vt = q[3]  # tangential velocity

        # Current mass (decreases linearly with time)
        m = m0 - mdot * t

        # Dynamics
        res = 4 * [None]
        res[0] = qdot[0] - vr
        res[1] = qdot[1] - vt / r
        res[2] = qdot[2] - (vt * vt / r - mu / (r * r) + (T / m) * ur)
        res[3] = qdot[3] - (-vr * vt / r + (T / m) * ut)

        self.constraints["res"] = res

        # Path constraint: ur² + ut² = 1
        self.constraints["thrust_unit"] = ur * ur + ut * ut - 1.0


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("rf", label="final radius")
        self.add_objective("obj")

    def compute(self):
        rf = self.inputs["rf"]
        # Maximize final radius = minimize negative radius
        # Add factor to scale objective
        self.objective["obj"] = -rf


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=4)

    def compute(self):
        q = self.inputs["q"]
        # Initial circular orbit
        r0 = 1.0
        theta0 = 0.0
        vr0 = 0.0
        vt0 = np.sqrt(1.0 / r0)  # sqrt(mu/r0), mu=1

        self.constraints["res"] = [
            q[0] - r0,
            q[1] - theta0,
            q[2] - vr0,
            q[3] - vt0,
        ]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=2)  # vr=0 and circular orbit

    def compute(self):
        q = self.inputs["q"]
        r = q[0]
        vr = q[2]
        vt = q[3]

        # Final circular orbit conditions
        # 1. vr(tf) = 0 (no radial velocity)
        # 2. vt²·r = μ (circular orbit condition, avoids sqrt)
        mu = 1.0

        self.constraints["res"] = [
            vr,  # vr(tf) = 0
            vt * vt * r - mu,  # Circular orbit: vt² = μ/r
        ]


def create_orbit_model(module_name="orbit_raising"):
    dynamics = OrbitDynamics()
    trap = TrapezoidRule()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 4 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Link trapezoidal rule
    for i in range(4):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link boundary conditions
    model.link("dynamics.q[0, :]", "ic.q[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "fc.q[0, :]")

    # Link final radius to objective
    model.link(f"dynamics.q[{num_time_steps}, 0]", "obj.rf[0]")

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build module")
parser.add_argument("--with-openmp", action="store_true", help="Use OpenMP")
args = parser.parse_args()

model = create_orbit_model()

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

# Create design vector and initial guess
x = model.create_vector()
x[:] = 0.0

N = num_time_steps + 1
t0 = 0.0
tf = 3.32
t = np.linspace(t0, tf, N)

# Set time data values
data = model.get_data_vector()
data["dynamics.t"] = t
t_frac = np.linspace(0, 1, N)

# Initial guess from YAPSS (linear interpolation)
r0 = 1.0
rf_guess = 1.5  # Guess final radius
theta0 = 0.0
theta_f_guess = np.pi
vr0 = 0.0
vrf = 0.0
vt0 = np.sqrt(1.0 / r0)
vtf_guess = 0.5 * vt0

x["dynamics.q[:, 0]"] = r0 + (rf_guess - r0) * t_frac  # r
x["dynamics.q[:, 1]"] = theta0 + (theta_f_guess - theta0) * t_frac  # theta
x["dynamics.q[:, 2]"] = vr0 + (vrf - vr0) * t_frac  # vr
x["dynamics.q[:, 3]"] = vt0 + (vtf_guess - vt0) * t_frac  # vt

# Control guess (switch from tangential to radial)
# Ensure ur² + ut² = 1 by normalizing
ur_unnorm = t_frac  # Start at 0, end at 1
ut_unnorm = 1.0 - t_frac  # Start at 1, end at 0
norm = np.sqrt(ur_unnorm**2 + ut_unnorm**2)
x["dynamics.ur"] = ur_unnorm / norm
x["dynamics.ut"] = ut_unnorm / norm

# Simple derivative guesses
x["dynamics.qdot[:, 0]"] = (rf_guess - r0) / tf
x["dynamics.qdot[:, 1]"] = (theta_f_guess - theta0) / tf
x["dynamics.qdot[:, 2]"] = (vrf - vr0) / tf
x["dynamics.qdot[:, 3]"] = (vtf_guess - vt0) / tf

# Set bounds
lower = model.create_vector()
upper = model.create_vector()

# State bounds - wide
lower["dynamics.q"] = -float("inf")
upper["dynamics.q"] = float("inf")

# Radius must be positive
lower["dynamics.q[:, 0]"] = 0.1

# Derivatives
lower["dynamics.qdot"] = -float("inf")
upper["dynamics.qdot"] = float("inf")

# Control bounds (no explicit bounds on ur, ut - enforced by unit constraint)
lower["dynamics.ur"] = -float("inf")
upper["dynamics.ur"] = float("inf")

lower["dynamics.ut"] = -float("inf")
upper["dynamics.ut"] = float("inf")

# Thrust unit constraint: ur² + ut² = 1
lower["dynamics.thrust_unit"] = 0.0
upper["dynamics.thrust_unit"] = 0.0

# Verify initial guess satisfies unit thrust constraint
thrust_mag = np.sqrt(x["dynamics.ur"] ** 2 + x["dynamics.ut"] ** 2)
print(f"\nInitial guess diagnostics:")
print(
    f"  Thrust magnitude: min={thrust_mag.min():.6f}, max={thrust_mag.max():.6f} (should be 1.0)"
)
print(f"  Initial orbit velocity: vt(0) = {x['dynamics.q[0, 3]']:.6f}")
print(f"  Final guess radius: {x[f'dynamics.q[{num_time_steps}, 0]']:.4f}")

# Optimize
print("\n" + "=" * 60)
print("Orbit Raising Problem")
print("=" * 60)
print("Finding optimal thrust direction to maximize final orbital radius")

opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 300,
        "initial_barrier_param": 10.0,  # Higher barrier for stability
        "monotone_barrier_fraction": 0.5,  # Slower reduction
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-6,  # Relaxed
        "max_line_search_iterations": 10,  # More line search
        "init_affine_step_multipliers": False,
    }
)

with open("orbit_raising_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract solution
r_sol = x["dynamics.q[:, 0]"]
theta_sol = x["dynamics.q[:, 1]"]
vr_sol = x["dynamics.q[:, 2]"]
vt_sol = x["dynamics.q[:, 3]"]
ur_sol = x["dynamics.ur"]
ut_sol = x["dynamics.ut"]

print(f"\n{'='*60}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Initial radius: r(0) = {r_sol[0]:.4f}")
print(f"Final radius:   r(tf) = {r_sol[-1]:.4f}")
print(f"Radius increase: {(r_sol[-1] / r_sol[0] - 1) * 100:.1f}%")
print(f"Final radial velocity: vr(tf) = {vr_sol[-1]:.6f} (should be ~0)")
print(f"Converged: {data.get('converged', False)}")

# Verify circular orbit condition
vt_circular_final = np.sqrt(1.0 / r_sol[-1])
print(f"\nCircular orbit check:")
print(f"  vt(tf) = {vt_sol[-1]:.6f}")
print(f"  sqrt(μ/r(tf)) = {vt_circular_final:.6f}")
print(f"  Error: {abs(vt_sol[-1] - vt_circular_final):.3e}")

# Plot results
fig = plt.figure(figsize=(14, 10))

# Trajectory in x-y plane
ax1 = fig.add_subplot(2, 3, 1)
x_cart = r_sol * np.cos(theta_sol)
y_cart = r_sol * np.sin(theta_sol)
ax1.plot(x_cart, y_cart, "b-", linewidth=2)

# Plot initial and final circular orbits
alpha = np.linspace(0, 2 * np.pi, 200)
ax1.plot(
    r_sol[0] * np.cos(alpha), r_sol[0] * np.sin(alpha), "k--", label="Initial orbit"
)
ax1.plot(
    r_sol[-1] * np.cos(alpha), r_sol[-1] * np.sin(alpha), "k--", label="Final orbit"
)

# Plot attracting center
ax1.plot(0.05 * np.cos(alpha), 0.05 * np.sin(alpha), "k", linewidth=2)
ax1.fill(0.05 * np.cos(alpha), 0.05 * np.sin(alpha), "yellow")

# Plot thrust direction vectors at several points
for i in range(0, N, N // 10):
    if i < len(x_cart):
        # Convert thrust to inertial frame
        u1_inertial = (
            np.cos(theta_sol[i]) * ur_sol[i] - np.sin(theta_sol[i]) * ut_sol[i]
        )
        u2_inertial = (
            np.sin(theta_sol[i]) * ur_sol[i] + np.cos(theta_sol[i]) * ut_sol[i]
        )

        ax1.arrow(
            x_cart[i],
            y_cart[i],
            0.15 * u1_inertial,
            0.15 * u2_inertial,
            head_width=0.04,
            head_length=0.05,
            fc="red",
            ec="red",
        )

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Orbital Trajectory")
ax1.axis("equal")
ax1.legend()
ax1.grid(True)

# State histories
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t, r_sol, label="r (radius)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Radius")
ax2.set_title("Radius History")
ax2.grid(True)

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t, theta_sol, label="θ (angle)")
ax3.set_xlabel("Time")
ax3.set_ylabel("Polar Angle (rad)")
ax3.set_title("Angle History")
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t, vr_sol, label="vr")
ax4.plot(t, vt_sol, label="vt")
ax4.set_xlabel("Time")
ax4.set_ylabel("Velocity")
ax4.set_title("Velocity Components")
ax4.legend()
ax4.grid(True)

# Control histories
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(t, ur_sol, label="ur (radial)")
ax5.plot(t, ut_sol, label="ut (tangential)")
ax5.set_xlabel("Time")
ax5.set_ylabel("Thrust Components")
ax5.set_title("Control History")
ax5.legend()
ax5.grid(True)

# Thrust direction angle
ax6 = fig.add_subplot(2, 3, 6)
phi = np.arctan2(ur_sol, ut_sol)
ax6.plot(t, np.rad2deg(np.unwrap(phi)), "b-", linewidth=2)
ax6.set_xlabel("Time")
ax6.set_ylabel("Thrust Direction (deg)")
ax6.set_title("Thrust Angle φ")
ax6.grid(True)

plt.suptitle(f"Orbit Raising: r(0)={r_sol[0]:.2f} → r(tf)={r_sol[-1]:.2f}", fontsize=14)
plt.tight_layout()
plt.savefig("orbit_raising_results.png", dpi=300)
plt.show()

print("\n✓ Plot saved as 'orbit_raising_results.png'")
