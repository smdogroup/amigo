"""
Generate publication-quality plots for the hang glider problem
in the same style as the cartpole tutorial.
"""

import amigo as am
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Colors matching cartpole style
blue_color = "#0072BD"  # State variables
purple_color = "#8242A2"  # Control variable

# Set up scaling
scaling = {"velocity": 10.0, "distance": 100.0, "time": 10.0}
num_time_steps = 100

# Create the model (same as in hang_glider_collocation.py)
from hang_glider_collocation import create_hang_glide_model

model = create_hang_glide_model(scaling, num_time_steps=num_time_steps)
model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

# Get the design variables
x = model.create_vector()
x[:] = 0.0

# Set initial guess
tf_guess = 100.0
dt = tf_guess / num_time_steps
t_guess = np.linspace(0, tf_guess, num_time_steps + 1)
t_frac = t_guess / tf_guess

x_pos = 0.0 + t_frac * 1250.0
x[f"gd.q[:, 0]"] = x_pos / scaling["distance"]

y_pos = 1000.0 + t_frac * (900.0 - 1000.0)
x[f"gd.q[:, 1]"] = y_pos / scaling["distance"]

vx = 13.227567500
vy = -1.2875005200
x[f"gd.q[:, 2]"] = vx / scaling["velocity"]
x[f"gd.q[:, 3]"] = vy / scaling["velocity"]

x[f"gd.qdot[:, 0]"] = (1250.0 / tf_guess) / scaling["distance"]
x[f"gd.qdot[:, 1]"] = (-100.0 / tf_guess) / scaling["distance"]
x[f"gd.qdot[:, 2]"] = 0.0
x[f"gd.qdot[:, 3]"] = 0.0

x["gd.CL"] = 1.0
x["trap.tf"] = tf_guess / scaling["time"]

# Set bounds
lower = model.create_vector()
upper = model.create_vector()
lower["gd.CL"] = 0.0
upper["gd.CL"] = 1.4
lower["trap.tf"] = 50.0 / scaling["time"]
upper["trap.tf"] = 200.0 / scaling["time"]
lower["gd.q"] = -float("inf")
upper["gd.q"] = float("inf")
lower["gd.qdot"] = -float("inf")
upper["gd.qdot"] = float("inf")

# Optimize
opt = am.Optimizer(model, x, lower=lower, upper=upper)
data = opt.optimize(
    {
        "barrier_strategy": "monotone",
        "initial_barrier_param": 0.1,
        "max_line_search_iterations": 30,
        "max_iterations": 500,
        "convergence_tolerance": 1e-8,
        "init_least_squares_multipliers": True,
        "init_affine_step_multipliers": False,
        "use_armijo_line_search": False,
        "regularization_eps_x": 1e-14,
        "regularization_eps_z": 1e-14,
        "adaptive_regularization": True,
        "fraction_to_boundary": 0.995,
    }
)

print(f"\nConverged: {data['converged']}")

# Extract optimal solution
tf_opt = x["trap.tf[0]"] * scaling["time"]
xf_opt = x[f"gd.q[{num_time_steps}, 0]"] * scaling["distance"]

print(f"Optimal final time: {tf_opt:.2f} seconds")
print(f"Maximum range: {xf_opt:.2f} meters")

# Extract solution vectors
t = np.linspace(0, tf_opt, num_time_steps + 1)
x_distance = x["gd.q[:, 0]"] * scaling["distance"]
y_altitude = x["gd.q[:, 1]"] * scaling["distance"]
vx_velocity = x["gd.q[:, 2]"] * scaling["velocity"]
vy_velocity = x["gd.q[:, 3]"] * scaling["velocity"]
CL_control = x["gd.CL[:]"]

# Create figure with 2 columns, 3 rows (6 subplots total)
fig, axes = plt.subplots(3, 2, figsize=(12, 9))

# Flatten axes for easier indexing
ax = axes.flatten()

# Font settings
fontname = "Arial"  # Use Arial instead of Helvetica for Windows compatibility

# --- Plot 1: Horizontal Distance vs Time ---
ax[0].plot(t, x_distance, color=blue_color, linewidth=2.0)
ax[0].set_xlabel("Time (s)", fontsize=11)
ax[0].set_ylabel("$x$ (m)", fontsize=12)
ax[0].grid(True, alpha=0.25, linewidth=0.5)
ax[0].tick_params(labelsize=10)
ax[0].set_title("Horizontal Position", fontsize=14, fontweight="bold", pad=10)

# --- Plot 2: Altitude vs Time ---
ax[1].plot(t, y_altitude, color=blue_color, linewidth=2.0)
ax[1].set_xlabel("Time (s)", fontsize=11)
ax[1].set_ylabel("$y$ (m)", fontsize=12)
ax[1].grid(True, alpha=0.25, linewidth=0.5)
ax[1].tick_params(labelsize=10)
ax[1].set_title("Altitude", fontsize=14, fontweight="bold", pad=10)

# --- Plot 3: Horizontal Velocity vs Time ---
ax[2].plot(t, vx_velocity, color=blue_color, linewidth=2.0)
ax[2].set_xlabel("Time (s)", fontsize=11)
ax[2].set_ylabel("$v_x$ (m/s)", fontsize=12)
ax[2].grid(True, alpha=0.25, linewidth=0.5)
ax[2].tick_params(labelsize=10)
ax[2].set_title("Horizontal Velocity", fontsize=14, fontweight="bold", pad=10)

# --- Plot 4: Vertical Velocity vs Time ---
ax[3].plot(t, vy_velocity, color=blue_color, linewidth=2.0)
ax[3].set_xlabel("Time (s)", fontsize=11)
ax[3].set_ylabel("$v_y$ (m/s)", fontsize=12)
ax[3].grid(True, alpha=0.25, linewidth=0.5)
ax[3].tick_params(labelsize=10)
ax[3].set_title("Vertical Velocity", fontsize=14, fontweight="bold", pad=10)

# --- Plot 5: Trajectory (x-y plane) ---
ax[4].plot(x_distance, y_altitude, color=blue_color, linewidth=2.0)
ax[4].set_xlabel("$x$ (m)", fontsize=11)
ax[4].set_ylabel("$y$ (m)", fontsize=12)
ax[4].grid(True, alpha=0.25, linewidth=0.5)
ax[4].tick_params(labelsize=10)
ax[4].set_title("Trajectory", fontsize=14, fontweight="bold", pad=10)

# --- Plot 6: Lift Coefficient (Control) ---
ax[5].plot(t, CL_control, color=purple_color, linewidth=2.0)
ax[5].set_xlabel("Time (s)", fontsize=11)
ax[5].set_ylabel("$C_L$", fontsize=12)
ax[5].grid(True, alpha=0.25, linewidth=0.5)
ax[5].tick_params(labelsize=10)
ax[5].set_title("Control", fontsize=14, fontweight="bold", pad=10)

# Set font for all axes
for a in ax:
    a.xaxis.label.set_fontname(fontname)
    a.yaxis.label.set_fontname(fontname)
    for tick in a.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in a.get_yticklabels():
        tick.set_fontname(fontname)

plt.tight_layout()

# Save with high resolution
output_path = Path(__file__).parent / "hang_glider_solution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nPlot saved to: {output_path}")

plt.show()
