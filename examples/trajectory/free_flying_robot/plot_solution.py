"""
Generate publication-quality plots for the free flying robot problem
in the same style as the cartpole and hang glider tutorials.
"""

import amigo as am
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Colors matching tutorial style
blue_color = "#0072BD"  # State variables
purple_color = "#8242A2"  # Control variables

# Problem parameters
final_time = 12.0
num_time_steps = 100

# Import the model creation function
from free_flying_robot import create_freeflyingrobot_model

# Check if --build flag is passed
build_module = "--build" in sys.argv

model = create_freeflyingrobot_model()

if build_module:
    print("Building C++ module...")
    model.build_module(source_dir=Path(__file__).parent)

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

print(f"Number of variables:     {model.num_variables}")
print(f"Number of constraints:   {model.num_constraints}")

# Create design variable vector
x = model.create_vector()
x[:] = 0.0

# Initial guess for states - smooth polynomial trajectory
t_normalized = np.linspace(0, 1, num_time_steps + 1)
s = t_normalized
pos_profile = 3 * s**2 - 2 * s**3

# Position states
x["robot.q[:, 0]"] = -10.0 + 10.0 * pos_profile
x["robot.q[:, 1]"] = -10.0 + 10.0 * pos_profile
x["robot.q[:, 2]"] = np.pi / 2.0 * (1.0 - pos_profile)

# Velocity states
vel_profile = (6 * s - 6 * s**2) / final_time
x["robot.q[:, 3]"] = 10.0 * vel_profile
x["robot.q[:, 4]"] = 10.0 * vel_profile
x["robot.q[:, 5]"] = -np.pi / 2.0 * vel_profile

# Initialize qdot
dt = final_time / num_time_steps
for i in range(6):
    x[f"robot.qdot[:, {i}]"] = np.gradient(x[f"robot.q[:, {i}]"], dt)

# Control initial guess
x["robot.u"] = 0.2

# Set bounds
lower = model.create_vector()
upper = model.create_vector()
lower["robot.q"] = -float("inf")
upper["robot.q"] = float("inf")
lower["robot.qdot"] = -float("inf")
upper["robot.qdot"] = float("inf")
lower["robot.u"] = 0.0
upper["robot.u"] = 1.0

# Optimize with same settings as the main script
print("\nOptimizing...")
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "barrier_strategy": "monotone",
        "initial_barrier_param": 10.0,
        "monotone_barrier_fraction": 0.1,
        "max_line_search_iterations": 50,
        "max_iterations": 1000,
        "convergence_tolerance": 1e-6,
        "init_least_squares_multipliers": True,
        "init_affine_step_multipliers": False,
        "use_armijo_line_search": True,
        "regularization_eps_x": 1e-6,
        "regularization_eps_z": 1e-6,
        "adaptive_regularization": True,
        "fraction_to_boundary": 0.99,
    }
)

print(f"\nConverged: {data['converged']}")
if data["converged"]:
    print(f"Final residual: {data['iterations'][-1]['residual']:.2e}")
    print(f"Iterations: {len(data['iterations'])}")

# Extract solution
q = x["robot.q"]
u = x["robot.u"]
t = np.linspace(0, final_time, num_time_steps + 1)

# Compute net thrusts
T1 = u[:, 0] - u[:, 1]
T2 = u[:, 2] - u[:, 3]

# Compute fuel consumption
total_fuel = 0.0
for i in range(num_time_steps):
    sum_u1 = np.sum(u[i])
    sum_u2 = np.sum(u[i + 1])
    total_fuel += 0.5 * dt * (sum_u1 + sum_u2)
print(f"Total fuel consumption: {total_fuel:.4f}")

# Create figure with 3 columns, 3 rows
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
fontname = "Arial"

# Row 1: Position states
axes[0, 0].plot(t, q[:, 0], color=blue_color, linewidth=2.0)
axes[0, 0].set_ylabel("x (m)", fontsize=12)
axes[0, 0].set_xlabel("Time (s)", fontsize=11)
axes[0, 0].grid(True, alpha=0.25, linewidth=0.5)
axes[0, 0].tick_params(labelsize=10)
axes[0, 0].set_title("Horizontal Position", fontsize=14, fontweight="bold", pad=10)

axes[0, 1].plot(t, q[:, 1], color=blue_color, linewidth=2.0)
axes[0, 1].set_ylabel("y (m)", fontsize=12)
axes[0, 1].set_xlabel("Time (s)", fontsize=11)
axes[0, 1].grid(True, alpha=0.25, linewidth=0.5)
axes[0, 1].tick_params(labelsize=10)
axes[0, 1].set_title("Vertical Position", fontsize=14, fontweight="bold", pad=10)

axes[0, 2].plot(t, q[:, 2], color=blue_color, linewidth=2.0)
axes[0, 2].set_ylabel("θ (rad)", fontsize=12)
axes[0, 2].set_xlabel("Time (s)", fontsize=11)
axes[0, 2].grid(True, alpha=0.25, linewidth=0.5)
axes[0, 2].tick_params(labelsize=10)
axes[0, 2].set_title("Orientation", fontsize=14, fontweight="bold", pad=10)

# Row 2: Velocity states
axes[1, 0].plot(t, q[:, 3], color=blue_color, linewidth=2.0)
axes[1, 0].set_ylabel("vx (m/s)", fontsize=12)
axes[1, 0].set_xlabel("Time (s)", fontsize=11)
axes[1, 0].grid(True, alpha=0.25, linewidth=0.5)
axes[1, 0].tick_params(labelsize=10)
axes[1, 0].set_title("Horizontal Velocity", fontsize=14, fontweight="bold", pad=10)

axes[1, 1].plot(t, q[:, 4], color=blue_color, linewidth=2.0)
axes[1, 1].set_ylabel("vy (m/s)", fontsize=12)
axes[1, 1].set_xlabel("Time (s)", fontsize=11)
axes[1, 1].grid(True, alpha=0.25, linewidth=0.5)
axes[1, 1].tick_params(labelsize=10)
axes[1, 1].set_title("Vertical Velocity", fontsize=14, fontweight="bold", pad=10)

axes[1, 2].plot(t, q[:, 5], color=blue_color, linewidth=2.0)
axes[1, 2].set_ylabel("ω (rad/s)", fontsize=12)
axes[1, 2].set_xlabel("Time (s)", fontsize=11)
axes[1, 2].grid(True, alpha=0.25, linewidth=0.5)
axes[1, 2].tick_params(labelsize=10)
axes[1, 2].set_title("Angular Velocity", fontsize=14, fontweight="bold", pad=10)

# Row 3: Trajectory and Controls
axes[2, 0].plot(q[:, 0], q[:, 1], color=blue_color, linewidth=2.0)
axes[2, 0].set_xlabel("x (m)", fontsize=11)
axes[2, 0].set_ylabel("y (m)", fontsize=12)
axes[2, 0].grid(True, alpha=0.25, linewidth=0.5)
axes[2, 0].tick_params(labelsize=10)
axes[2, 0].set_title("Trajectory", fontsize=14, fontweight="bold", pad=10)
axes[2, 0].axis("equal")

axes[2, 1].plot(t, T1, color=purple_color, linewidth=2.0)
axes[2, 1].set_xlabel("Time (s)", fontsize=11)
axes[2, 1].set_ylabel("T1 (N)", fontsize=12)
axes[2, 1].grid(True, alpha=0.25, linewidth=0.5)
axes[2, 1].tick_params(labelsize=10)
axes[2, 1].set_title("Thruster 1", fontsize=14, fontweight="bold", pad=10)

axes[2, 2].plot(t, T2, color=purple_color, linewidth=2.0)
axes[2, 2].set_xlabel("Time (s)", fontsize=11)
axes[2, 2].set_ylabel("T2 (N)", fontsize=12)
axes[2, 2].grid(True, alpha=0.25, linewidth=0.5)
axes[2, 2].tick_params(labelsize=10)
axes[2, 2].set_title("Thruster 2", fontsize=14, fontweight="bold", pad=10)

# Set font for all axes
for ax_row in axes:
    for ax in ax_row:
        ax.xaxis.label.set_fontname(fontname)
        ax.yaxis.label.set_fontname(fontname)
        for tick in ax.get_xticklabels():
            tick.set_fontname(fontname)
        for tick in ax.get_yticklabels():
            tick.set_fontname(fontname)

plt.tight_layout()

# Save with high resolution
output_path = Path(__file__).parent / "freeflyingrobot_solution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nPlot saved to: {output_path}")

plt.show()
