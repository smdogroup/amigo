import amigo as am
import numpy as np
import argparse
import json
import matplotlib.pylab as plt
import niceplots

"""
Multiple Shooting Method for Cannon Trajectory Optimization

Multiple shooting divides the trajectory into segments with free boundary states.
Within each segment, states are linked sequentially.
At segment boundaries, states are free design variables.

Key Advantage: RK4 steps can be evaluated in PARALLEL when built with OpenMP!

Structure:
  - Trajectory divided into num_segments segments
  - Within segments: q2[i] linked to q1[i+1] 
  - At segment boundaries: both q2[i] and q1[i+1] are free, but linked for continuity
  - More design variables than single shooting = better numerical conditioning

To enable parallel execution:
  1. Build with OpenMP: python cannon_multipleshooting.py --build --with-openmp
  2. RK4 evaluations will run in parallel during optimization

Design Variables: 
  - State at each time step (with segment boundaries as free variables)
  - rk4.q1[segment_starts] are the primary design variables
"""
num_segments = 5
num_time_steps = 50
final_time = 3.0
dt = final_time / num_time_steps
steps_per_segment = num_time_steps // num_segments


class RK4Integrator(am.Component):
    """
    Single RK4 integration step.
    When organized by segments, steps within each segment can run in parallel.
    """

    def __init__(self):
        super().__init__()

        self.add_constant("dt", value=final_time / num_time_steps)
        self.add_constant("g", value=1.0)
        self.add_constant("c", value=0.4)
        self.add_input("q1", shape=(4))  # State at time i
        self.add_input("q2", shape=(4))  # State at time i+1
        self.add_constraint("res", shape=(4))  # Continuity constraint

    def cannon_dynamics(self, q):
        """Compute state derivatives for cannon trajectory."""
        c = self.constants["c"]
        g = self.constants["g"]
        x, y, vx, vy = q[0], q[1], q[2], q[3]
        v = am.sqrt(vx**2 + vy**2)

        qdot = [None] * 4
        qdot[0] = vx
        qdot[1] = vy
        qdot[2] = -c * (vx * v)
        qdot[3] = -c * (vy * v) - g
        return qdot

    def compute(self):
        dt = self.constants["dt"]
        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]

        # RK4 integration from q1
        k1 = self.cannon_dynamics(q1)

        q_mid = [q1[i] + 0.5 * dt * k1[i] for i in range(4)]
        k2 = self.cannon_dynamics(q_mid)

        q_mid = [q1[i] + 0.5 * dt * k2[i] for i in range(4)]
        k3 = self.cannon_dynamics(q_mid)

        q_mid = [q1[i] + dt * k3[i] for i in range(4)]
        k4 = self.cannon_dynamics(q_mid)

        # Residual: q2 - (q1 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)) = 0
        self.constraints["res"] = [
            q2[i] - (q1[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
            for i in range(4)
        ]


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=2)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0], q[1]]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=4)
        self.add_constraint("res", shape=2)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [q[0] - 6.0, q[1]]


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=(4))
        self.add_objective("obj")

    def compute(self):
        q = self.inputs["q"]
        self.objective["obj"] = q[2] * q[2] + q[3] * q[3]


def create_cannon_model(module_name="cannon_multiple_shooting"):
    """
    Create parallel multiple shooting model.
    Segments are organized so RK4 steps within each segment can run in parallel.
    Segment boundary states are free design variables.
    """
    rk4 = RK4Integrator()
    obj = Objective()
    ic = InitialConditions()
    fc = FinalConditions()

    model = am.Model(module_name)

    # Add RK4 integrators
    model.add_component("rk4", num_time_steps, rk4)
    model.add_component("obj", 1, obj)
    model.add_component("ic", 1, ic)
    model.add_component("fc", 1, fc)

    # Segment boundaries - these will be free design variables
    segment_starts = [i * steps_per_segment for i in range(num_segments)]

    # Link states: within segments they're linked, at boundaries they're free
    for i in range(num_time_steps - 1):
        # If next index is NOT a segment start, link states
        if (i + 1) not in segment_starts:
            model.link(f"rk4.q2[{i}, :]", f"rk4.q1[{i+1}, :]")
        # If next index IS a segment start, both q2[i] and q1[i+1] are free
        # but we add a continuity constraint to enforce they match

    # Add continuity constraints at segment boundaries
    # For each segment transition, enforce q2[end_of_seg] = q1[start_of_next_seg]
    for seg in range(num_segments - 1):
        end_idx = segment_starts[seg + 1] - 1  # Last step of current segment
        start_idx = segment_starts[seg + 1]  # First step of next segment
        # Both are independent, but link them to enforce continuity
        model.link(f"rk4.q2[{end_idx}, :]", f"rk4.q1[{start_idx}, :]")

    # Boundary conditions
    model.link("rk4.q1[0, :]", "ic.q[0, :]")  # Initial state
    model.link(f"rk4.q2[{num_time_steps - 1}, :]", "fc.q[0, :]")  # Final state

    # Objective on initial velocities
    model.link("rk4.q1[0, :]", "obj.q[0, :]")

    return model


def plot_trajectory(x_opt, title="Multiple Shooting: Cannon Trajectory"):
    """Plot trajectory from RK4 states."""
    # Reconstruct trajectory from q1 and q2
    q1_all = x_opt["rk4.q1"].reshape(num_time_steps, 4)
    q2_final = x_opt["rk4.q2"].reshape(num_time_steps, 4)[-1, :]

    # Combine: [q1[0], q1[1], ..., q1[n-1], q2[n-1]]
    q = np.vstack([q1_all, q2_final])
    t = np.linspace(0, final_time, num_time_steps + 1)
    x_pos, y_pos, vx, vy = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Segment boundaries and colors
    segment_starts = [i * steps_per_segment for i in range(num_segments)]
    segment_ends = segment_starts[1:] + [num_time_steps]
    colors = plt.cm.tab10(np.linspace(0, 1, num_segments))

    with plt.style.context(niceplots.get_style()):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Trajectory plot - colored by segment
        for seg in range(num_segments):
            start_idx = segment_starts[seg]
            end_idx = segment_ends[seg] + 1  # +1 to include the endpoint
            ax1.plot(
                x_pos[start_idx:end_idx],
                y_pos[start_idx:end_idx],
                color=colors[seg],
                linewidth=2,
                label=f"Segment {seg+1}",
            )
            # Mark segment start
            ax1.plot(
                x_pos[start_idx],
                y_pos[start_idx],
                "o",
                color=colors[seg],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

        ax1.plot(x_pos[0], y_pos[0], "go", markersize=10, label="Start", zorder=10)
        ax1.plot(x_pos[-1], y_pos[-1], "ro", markersize=10, label="End", zorder=10)
        ax1.plot(6.0, 0.0, "r*", markersize=14, label="Target", zorder=10)
        ax1.set_xlabel("X Position", fontsize=10)
        ax1.set_ylabel("Y Position", fontsize=10)
        ax1.set_title(f"{title}", fontsize=10)
        ax1.legend(fontsize=7, loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.axis("equal")

        # X position vs time - colored by segment
        for seg in range(num_segments):
            start_idx = segment_starts[seg]
            end_idx = segment_ends[seg] + 1
            ax2.plot(
                t[start_idx:end_idx],
                x_pos[start_idx:end_idx],
                color=colors[seg],
                linewidth=2,
            )
        for seg_idx in segment_starts:
            ax2.axvline(
                t[seg_idx], color="gray", linestyle="--", alpha=0.4, linewidth=1
            )
        ax2.set_xlabel("Time (s)", fontsize=10)
        ax2.set_ylabel("X Position", fontsize=10)
        ax2.set_title("X Position vs Time", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Y position vs time - colored by segment
        for seg in range(num_segments):
            start_idx = segment_starts[seg]
            end_idx = segment_ends[seg] + 1
            ax3.plot(
                t[start_idx:end_idx],
                y_pos[start_idx:end_idx],
                color=colors[seg],
                linewidth=2,
            )
        for seg_idx in segment_starts:
            ax3.axvline(
                t[seg_idx], color="gray", linestyle="--", alpha=0.4, linewidth=1
            )
        ax3.set_xlabel("Time (s)", fontsize=10)
        ax3.set_ylabel("Y Position", fontsize=10)
        ax3.set_title("Y Position vs Time", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Velocities vs time - colored by segment
        for seg in range(num_segments):
            start_idx = segment_starts[seg]
            end_idx = segment_ends[seg] + 1
            # Plot vx and vy with different line styles but same segment color
            ax4.plot(
                t[start_idx:end_idx],
                vx[start_idx:end_idx],
                color=colors[seg],
                linewidth=2,
                linestyle="-",
                alpha=0.8,
            )
            ax4.plot(
                t[start_idx:end_idx],
                vy[start_idx:end_idx],
                color=colors[seg],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

        # Add legend for velocity components
        ax4.plot([], [], "k-", linewidth=2, label="V_x")
        ax4.plot([], [], "k--", linewidth=2, label="V_y")

        for seg_idx in segment_starts:
            ax4.axvline(
                t[seg_idx], color="gray", linestyle="--", alpha=0.4, linewidth=1
            )
        ax4.set_xlabel("Time (s)", fontsize=10)
        ax4.set_ylabel("Velocity", fontsize=10)
        ax4.set_title("Velocities vs Time", fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig("cannon_multiple_shooting.png", dpi=300, bbox_inches="tight")
        fig.savefig("cannon_multiple_shooting.svg", bbox_inches="tight")
        plt.show()

    return fig


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
parser.add_argument(
    "--with-openmp", dest="use_openmp", action="store_true", default=False
)
args = parser.parse_args()

model = create_cannon_model()

if args.build:
    compile_args = []
    link_args = []
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]
    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print("=" * 60)
print(f"Multiple Shooting Configuration (Parallel-Ready)")
print("=" * 60)
print(f"Num segments:      {num_segments}")
print(f"Steps per segment: {steps_per_segment}")
print(f"Total time steps:  {num_time_steps}")
print(f"Num variables:     {model.num_variables}")
print(f"Num constraints:   {model.num_constraints}")
if args.use_openmp:
    print(f"OpenMP enabled:    YES (segments will run in parallel)")
else:
    print(f"OpenMP enabled:    NO (use --with-openmp for parallel)")
print("=" * 60)
print()

x = model.create_vector()
x[:] = 0.0

# Initial guesses for the design variables
target_x = 6.0
g = 1.0
vx_init = target_x / final_time
vy_init = 0.5 * g * final_time

t = np.linspace(0, final_time, num_time_steps)
vx_guess = np.full(num_time_steps, vx_init)
vy_guess = vy_init - g * t

x["rk4.q1[:, 2]"] = vx_guess
x["rk4.q1[:, 3]"] = vy_guess

opt = am.Optimizer(model, x)
data = opt.optimize({"max_iterations": 200, "max_line_search_iterations": 10})

with open("cannon_multiple_shooting_data.json", "w") as fp:
    json.dump(data, fp, indent=2)

plot_trajectory(x)
