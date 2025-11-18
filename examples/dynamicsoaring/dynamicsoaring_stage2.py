import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

"""
Dynamic Soaring - Stage 2: Full Periodic Problem
=================================================

This uses the Stage 1 solution as initial guess to solve the
full periodic problem with relaxed endpoint constraints.

Load the Stage 1 solution and use it to warm-start this problem.
"""

# Check if Stage 1 solution exists
if not os.path.exists("stage1_solution.json"):
    print("ERROR: Stage 1 solution not found!")
    print("Please run dynamicsoaring_simple.py first to generate initial guess.")
    exit(1)

# Load Stage 1 solution
with open("stage1_solution.json", "r") as fp:
    stage1 = json.load(fp)

print("✓ Loaded Stage 1 solution as initial guess")
print(f"  Stage 1 time: {stage1['tf']:.2f} s")

num_time_steps = len(stage1["q"]) - 1
print(f"  Using {num_time_steps} time steps")

# Import the dynamics from the simple version
from dynamicsoaring_simple import TrapezoidRule, SoaringDynamics, Objective


class InitialConditions(am.Component):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=3)  # Only x, y, h

    def compute(self):
        q = self.inputs["q"]
        # Only constrain position to origin
        self.constraints["res"] = [q[0], q[1], q[2]]


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
            qf[0] - q0[0],  # x periodic
            qf[1] - q0[1],  # y periodic
            qf[2] - q0[2],  # h periodic
            qf[3] - q0[3],  # v periodic
            qf[4] - q0[4],  # gamma periodic
            qf[5] - q0[5] - two_pi,  # psi += 2π
        ]


def create_periodic_model(scaling, module_name="dynamic_soaring_periodic"):
    dynamics = SoaringDynamics(scaling)
    trap = TrapezoidRule(scaling)
    obj = Objective(scaling)
    ic = InitialConditions(scaling)
    pc = PeriodicConditions(scaling)

    model = am.Model(module_name)

    model.add_component("dynamics", num_time_steps + 1, dynamics)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("pc", 1, pc)

    # Link trapezoidal rule
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps
        model.link(f"dynamics.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"dynamics.q[1:, {i}]", f"trap.q2[{start}:{end}]")
        model.link(f"dynamics.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"dynamics.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Initial conditions
    model.link("dynamics.q[0, :]", "ic.q[0, :]")

    # Periodic conditions
    model.link("dynamics.q[0, :]", "pc.q0[0, :]")
    model.link(f"dynamics.q[{num_time_steps}, :]", "pc.qf[0, :]")

    # Link final time
    model.link("obj.tf[0]", "trap.tf")

    return model


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build module")
parser.add_argument("--with-openmp", action="store_true", help="Use OpenMP")
args = parser.parse_args()

# Use same scaling as Stage 1
scaling = stage1["scaling"]

model = create_periodic_model(scaling)

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

# Create design vector and load Stage 1 solution
x = model.create_vector()
x[:] = 0.0

# Load Stage 1 solution as initial guess
q_stage1 = np.array(stage1["q"])
qdot_stage1 = np.array(stage1["qdot"])
CL_stage1 = np.array(stage1["CL"])
phi_stage1 = np.array(stage1["phi"])

x["dynamics.q"] = q_stage1
x["dynamics.qdot"] = qdot_stage1
x["dynamics.CL"] = CL_stage1
x["dynamics.phi"] = phi_stage1
x["obj.tf"] = stage1["tf"] / scaling["time"]

print("\n✓ Initialized with Stage 1 solution")

# Same bounds as Stage 1
lower = model.create_vector()
upper = model.create_vector()

lower["dynamics.q[:, 0]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 0]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 1]"] = -2000.0 / scaling["distance"]
upper["dynamics.q[:, 1]"] = 2000.0 / scaling["distance"]

lower["dynamics.q[:, 2]"] = 10.0 / scaling["altitude"]
upper["dynamics.q[:, 2]"] = 500.0 / scaling["altitude"]

lower["dynamics.q[:, 3]"] = 50.0 / scaling["velocity"]
upper["dynamics.q[:, 3]"] = 300.0 / scaling["velocity"]

lower["dynamics.q[:, 4]"] = np.radians(-45.0) / scaling["angle"]
upper["dynamics.q[:, 4]"] = np.radians(45.0) / scaling["angle"]

lower["dynamics.q[:, 5]"] = np.radians(-10.0) / scaling["angle"]
upper["dynamics.q[:, 5]"] = np.radians(370.0) / scaling["angle"]

lower["dynamics.qdot"] = -float("inf")
upper["dynamics.qdot"] = float("inf")

lower["dynamics.CL"] = 0.1
upper["dynamics.CL"] = 1.5

lower["dynamics.phi"] = np.radians(-89.0)
upper["dynamics.phi"] = np.radians(89.0)

lower["obj.tf"] = 20.0 / scaling["time"]
upper["obj.tf"] = 300.0 / scaling["time"]

# Optimize Stage 2
print("\n" + "=" * 60)
print("STAGE 2: Solving full periodic problem")
print("=" * 60)

opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "max_iterations": 500,
        "initial_barrier_param": 1.0,
        "monotone_barrier_fraction": 0.25,
        "barrier_strategy": "monotone",
        "convergence_tolerance": 1e-6,
        "max_line_search_iterations": 5,
        "init_affine_step_multipliers": False,
    }
)

with open("dynamic_soaring_stage2_opt_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

# Extract final solution
tf_final = x["obj.tf"][0] * scaling["time"]
print(f"\n{'='*60}")
print(f"Stage 2 COMPLETE!")
print(f"{'='*60}")
print(f"Optimal time: {tf_final:.2f} s")
print(f"Converged: {data.get('converged', False)}")
print(f"Final residual: {data['iterations'][-1]['residual']:.3e}")

# Verify periodic conditions
q0 = x["dynamics.q[0, :]"]
qf = x[f"dynamics.q[{num_time_steps}, :]"]

print(f"\nPeriodic condition check:")
print(f"  Δx: {abs(qf[0] - q0[0]) * scaling['distance']:.3e} ft")
print(f"  Δy: {abs(qf[1] - q0[1]) * scaling['distance']:.3e} ft")
print(f"  Δh: {abs(qf[2] - q0[2]) * scaling['altitude']:.3e} ft")
print(f"  Δv: {abs(qf[3] - q0[3]) * scaling['velocity']:.3e} ft/s")
print(f"  Δγ: {abs(qf[4] - q0[4]) * scaling['angle']:.3e} rad")
psi_change = (qf[5] - q0[5]) * scaling["angle"]
print(f"  Δψ: {psi_change:.6f} rad (target: {2*np.pi:.6f})")

# Plot final solution
N = num_time_steps + 1
t = np.linspace(0, tf_final, N)
q_final = np.zeros((N, 6))
q_final[:, 0] = x["dynamics.q[:, 0]"] * scaling["distance"]
q_final[:, 1] = x["dynamics.q[:, 1]"] * scaling["distance"]
q_final[:, 2] = x["dynamics.q[:, 2]"] * scaling["altitude"]
q_final[:, 3] = x["dynamics.q[:, 3]"] * scaling["velocity"]
q_final[:, 4] = x["dynamics.q[:, 4]"] * scaling["angle"]
q_final[:, 5] = x["dynamics.q[:, 5]"] * scaling["angle"]

CL_final = x["dynamics.CL"]
phi_final = x["dynamics.phi"]

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot(q_final[:, 0], q_final[:, 1], q_final[:, 2], "b-", linewidth=2)
ax1.scatter([0], [0], [q_final[0, 2]], c="g", s=100, marker="o")
ax1.set_xlabel("x (ft)")
ax1.set_ylabel("y (ft)")
ax1.set_zlabel("h (ft)")
ax1.set_title("3D Periodic Trajectory")

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(t, q_final[:, 3], "b-", linewidth=2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (ft/s)")
ax2.set_title("Velocity")
ax2.grid(True)

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(t, np.rad2deg(q_final[:, 4]), "b-", linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("γ (deg)")
ax3.set_title("Flight Path Angle")
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t, np.rad2deg(q_final[:, 5]), "b-", linewidth=2)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("ψ (deg)")
ax4.set_title("Heading Angle")
ax4.grid(True)

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(t, CL_final, "b-", linewidth=2)
ax5.axhline(1.5, color="r", linestyle="--", label="CL max")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("CL")
ax5.set_title("Lift Coefficient")
ax5.legend()
ax5.grid(True)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t, np.rad2deg(phi_final), "b-", linewidth=2)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("φ (deg)")
ax6.set_title("Bank Angle")
ax6.grid(True)

plt.suptitle(
    f"Stage 2: Periodic Soaring (tf={tf_final:.1f}s, β=0.08 1/ft)", fontsize=14
)
plt.tight_layout()
plt.savefig("dynamic_soaring_stage2.png", dpi=300)
plt.show()

print("\n✓ Stage 2 solution saved to 'dynamic_soaring_stage2.png'")
print("\nDynamic soaring problem SOLVED using continuation method!")

