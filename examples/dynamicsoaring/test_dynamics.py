"""
Test if the initial guess satisfies the dynamic soaring dynamics
"""

import numpy as np

# Constants (US customary units)
g = 32.2  # ft/s^2
m = 5.6  # slug
S = 45.09703  # ft^2
rho = 0.002378  # slug/ft^3
CD0 = 0.00873
K = 0.045
beta = 0.08  # wind gradient

# Initial guess values
h = 200.0  # altitude (ft)
v = 150.0  # velocity (ft/s)
gamma = 0.0  # flight path angle (rad)
psi = 0.5  # heading angle (rad)
CL = 0.5  # lift coefficient
phi = np.radians(10.0)  # bank angle

print("Testing dynamic soaring dynamics with simple initial guess:")
print(f"h = {h} ft, v = {v} ft/s, gamma = {gamma} rad, psi = {psi} rad")
print(f"CL = {CL}, phi = {np.rad2deg(phi)} deg")
print(f"beta = {beta} 1/ft")

# Compute trig functions
sin_gamma = np.sin(gamma)
cos_gamma = np.cos(gamma)
sin_psi = np.sin(psi)
cos_psi = np.cos(psi)
sin_phi = np.sin(phi)
cos_phi = np.cos(phi)

# Wind model
Wx = beta * h
dWx = beta

# Dynamic pressure
q_dyn = 0.5 * rho * v * v

# Drag coefficient
CD = CD0 + K * CL * CL

# Forces
L = q_dyn * S * CL
D = q_dyn * S * CD

print(f"\nAerodynamic forces:")
print(f"Dynamic pressure q = {q_dyn:.4f} lb/ft^2")
print(f"Lift L = {L:.2f} lb")
print(f"Drag D = {D:.2f} lb")
print(f"Weight W = {m * g:.2f} lb")
print(f"Wind velocity Wx = {Wx:.2f} ft/s")

# For level flight (gamma=0), compute required bank angle
# L*cos(phi) = mg
required_cos_phi = (m * g) / L
if abs(required_cos_phi) <= 1.0:
    required_phi = np.arccos(required_cos_phi)
    print(f"\nFor level flight at this velocity:")
    print(f"Required bank angle = {np.rad2deg(required_phi):.2f} deg")
    print(f"Actual bank angle = {np.rad2deg(phi):.2f} deg")
else:
    print(f"\nCannot maintain level flight - insufficient lift!")
    print(f"L/(mg) = {L/(m*g):.2f}")

# Compute the dynamics (what the time derivatives should be)
hdot = v * sin_gamma
print(f"\nComputed time derivatives:")
print(f"xdot = {v * cos_gamma * sin_psi + Wx:.2f} ft/s")
print(f"ydot = {v * cos_gamma * cos_psi:.2f} ft/s")
print(f"hdot = {hdot:.2f} ft/s")

vdot = -D / m - g * sin_gamma - dWx * hdot * cos_gamma * sin_psi
print(f"vdot = {vdot:.4f} ft/s^2")

# Check if gamma_dot can be zero for level flight
gamma_numerator = L * cos_phi - m * g * cos_gamma + m * dWx * hdot * sin_gamma * sin_psi
gamma_dot_required = gamma_numerator / (m * v)
print(f"gamma_dot = {gamma_dot_required:.6f} rad/s")

psi_numerator = L * sin_phi - m * dWx * hdot * cos_psi
psi_dot_required = psi_numerator / (m * v * cos_gamma)
print(f"psi_dot = {psi_dot_required:.6f} rad/s")

# Check load factor
load_factor = L / (m * g)
print(f"\nLoad factor = {load_factor:.3f} (within bounds: -2 to 5)")

print(f"\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

# Check if the problem is feasible
if abs(vdot) > 10.0:
    print("⚠ WARNING: Large velocity derivative suggests unbalanced forces!")
    print(f"  Drag/Weight ratio D/W = {D/(m*g):.3f}")

if abs(gamma_dot_required) > 0.01:
    print("⚠ WARNING: Non-zero gamma_dot but we assume level flight!")
    print(f"  To maintain level flight, need different CL or phi")

if abs(psi_dot_required) < 0.01:
    print("⚠ WARNING: Very small psi_dot - will take forever to complete rotation!")
    print(
        f"  Time to rotate 2π: {2*np.pi / psi_dot_required if psi_dot_required > 0 else float('inf'):.1f} s"
    )
else:
    print(f"✓ Estimated time for 2π rotation: {2*np.pi / abs(psi_dot_required):.1f} s")

print(f"\nAt constant altitude and velocity with rotating heading:")
print(
    f"  Path forms a circle with radius ≈ {v / abs(psi_dot_required) if psi_dot_required != 0 else 0:.0f} ft"
)
