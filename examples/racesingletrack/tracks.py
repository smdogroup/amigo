"""
Track loading and processing for racecar lap time optimization.

This module provides two tracks:
  - berlin_2018:  Real Formula E circuit loaded from CSV (TUMFTM format)
  - oval_dymos:   Analytically defined oval (from the Dymos/OpenMDAO example)

Both return a Track dataclass containing everything the optimizer needs:
  - s:      uniformly spaced arc-length nodes (independent variable)
  - kappa:  curvature at each node (enters the vehicle dynamics)
  - x, y:   centerline coordinates (for plotting the racing line)
  - w_right, w_left: half-widths (for track boundary constraints)

Why curvature matters
---------------------
In the curvilinear formulation the equation of motion for the heading is:

    d(alpha)/ds = omega/sdot - kappa(s)

so kappa(s) is direct input data to the ODE right-hand side. Noise or
discontinuities in kappa propagate into the dynamics and make the NLP
harder to solve. That is why we smooth the raw curvature before handing
it to the optimizer.

+--------------------------------------------------------------------------+
|  PROJECT TASK: The current smoothing uses a Gaussian convolution kernel.  |
|  Your job is to replace it with a spline-based approach (e.g. using      |
|  scipy.interpolate or scikit-learn) which produces a smoother and more   |
|  differentiable curvature profile. The smoothing parameter controls the  |
|  trade-off between fidelity to the raw data and smoothness.              |
|  Look for the section marked "SMOOTHING (replace this)" below.          |
+--------------------------------------------------------------------------+
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Directory where track CSV files are stored (same folder as this script)
TRACKS_DIR = Path(__file__).resolve().parent / "tracks"


# ---------------------------------------------------------------------------
# Track dataclass: holds all geometric data for one circuit
# ---------------------------------------------------------------------------
@dataclass
class Track:
    name: str

    # Resampled onto a uniform mesh (used by the optimizer)
    s_total: float  # total track length [m]
    s: np.ndarray  # arc-length at each node, shape (num_nodes,)
    kappa: np.ndarray  # curvature at each node [1/m], shape (num_nodes,)
    x: np.ndarray  # centerline x-coordinates [m]
    y: np.ndarray  # centerline y-coordinates [m]
    w_right: np.ndarray  # half-width to the right of centerline [m]
    w_left: np.ndarray  # half-width to the left of centerline [m]

    # Raw data (original resolution, useful for plotting)
    raw_x: np.ndarray
    raw_y: np.ndarray
    raw_s: np.ndarray
    raw_w_right: np.ndarray
    raw_w_left: np.ndarray


# ---------------------------------------------------------------------------
# CSV track loader (berlin_2018 uses this path)
# ---------------------------------------------------------------------------
def _load_csv(csv_path, num_nodes, smooth_sigma=3.0):
    """
    Load a track from a TUMFTM-format CSV and return a Track object.

    The CSV columns are: x, y, w_right, w_left  (one row per centerline point).

    Processing pipeline:
      1. Read raw (x, y) centerline and half-widths from CSV
      2. Close the loop (wrap a few points) so derivatives at the seam are clean
      3. Compute arc-length s along the centerline
      4. Compute curvature kappa from the (x, y) derivatives
      5. Smooth kappa  <-- THIS IS WHAT YOU WILL CHANGE
      6. Resample everything onto a uniform mesh of num_nodes points

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    num_nodes : int
        Number of equally spaced nodes for the optimizer mesh.
    smooth_sigma : float
        Width of the Gaussian smoothing kernel in meters.
        Larger = smoother kappa but less faithful to the real track.
    """

    # ------------------------------------------------------------------
    # Step 1: Read the CSV
    # Columns: [x, y, w_right, w_left]
    # ------------------------------------------------------------------
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    w_right_raw = data[:, 2]
    w_left_raw = data[:, 3]

    # ------------------------------------------------------------------
    # Step 2: Close the loop
    # The track is a closed circuit, but the CSV does not repeat the
    # first point. We append a few points from the beginning so that
    # finite-difference derivatives near the seam (index 0 / N-1)
    # see neighboring data instead of a discontinuity.
    # ------------------------------------------------------------------
    n_wrap = min(5, len(x_raw))
    x_closed = np.concatenate([x_raw, x_raw[:n_wrap]])
    y_closed = np.concatenate([y_raw, y_raw[:n_wrap]])

    # ------------------------------------------------------------------
    # Step 3: Compute arc-length s
    # ds[i] = distance from point i to point i+1
    # s[i]  = cumulative distance from the start to point i
    # ------------------------------------------------------------------
    dx = np.diff(x_closed)
    dy = np.diff(y_closed)
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_closed = np.concatenate([[0], np.cumsum(ds_raw)])

    s_total = s_closed[len(x_raw)]  # total lap length
    s_raw = s_closed[: len(x_raw)]  # arc-length for original points only

    # ------------------------------------------------------------------
    # Step 4: Compute curvature from centerline coordinates
    #
    #   kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
    #
    # where primes denote derivatives with respect to arc-length.
    # Positive kappa = left turn, negative = right turn.
    # ------------------------------------------------------------------
    dxds = np.gradient(x_closed, s_closed)
    dyds = np.gradient(y_closed, s_closed)
    d2xds2 = np.gradient(dxds, s_closed)
    d2yds2 = np.gradient(dyds, s_closed)
    denom = (dxds**2 + dyds**2) ** 1.5
    kappa_closed = (dxds * d2yds2 - dyds * d2xds2) / denom
    kappa_raw = kappa_closed[: len(x_raw)]

    # ------------------------------------------------------------------
    # Step 5: Smooth the curvature
    #
    # SMOOTHING (replace this)
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # Currently: Gaussian convolution with periodic padding.
    #   - smooth_sigma controls the kernel width in meters
    #   - We pad kappa periodically so the convolution wraps around
    #     the start/finish seam without edge artifacts
    #
    # Your task: replace this block with a spline-based smoother.
    # For example, you could fit a smoothing spline to (s_raw, kappa_raw)
    # using scipy.interpolate.UnivariateSpline with a smoothing factor,
    # or use scikit-learn. The smoothing parameter should control the
    # trade-off between fidelity and smoothness.
    # ------------------------------------------------------------------
    if smooth_sigma > 0:
        ds_avg = s_total / len(s_raw)
        sigma_pts = max(1, int(smooth_sigma / ds_avg))
        kernel_size = 6 * sigma_pts + 1
        kernel_x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (kernel_x / sigma_pts) ** 2)
        kernel /= kernel.sum()
        pad = kernel_size
        kappa_padded = np.concatenate([kappa_raw[-pad:], kappa_raw, kappa_raw[:pad]])
        kappa_smooth = np.convolve(kappa_padded, kernel, mode="same")[pad:-pad]
    else:
        kappa_smooth = kappa_raw

    # ------------------------------------------------------------------
    # Step 6: Resample onto a uniform mesh
    # The optimizer needs equally spaced nodes (constant ds between nodes).
    # We use linear interpolation to map all track quantities from the
    # irregularly spaced raw points to the uniform mesh.
    # ------------------------------------------------------------------
    s_nodes = np.linspace(0, s_total, num_nodes)
    kappa_nodes = np.interp(s_nodes, s_raw, kappa_smooth)
    x_nodes = np.interp(s_nodes, s_raw, x_raw)
    y_nodes = np.interp(s_nodes, s_raw, y_raw)
    w_right_nodes = np.interp(s_nodes, s_raw, w_right_raw)
    w_left_nodes = np.interp(s_nodes, s_raw, w_left_raw)

    name = csv_path.stem
    return Track(
        name=name,
        s_total=s_total,
        s=s_nodes,
        kappa=kappa_nodes,
        x=x_nodes,
        y=y_nodes,
        w_right=w_right_nodes,
        w_left=w_left_nodes,
        raw_x=x_raw,
        raw_y=y_raw,
        raw_s=s_raw,
        raw_w_right=w_right_raw,
        raw_w_left=w_left_raw,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def load_track(name, num_nodes=301, smooth_sigma=3.0):
    """Load a CSV track by name (must exist in the tracks/ directory)."""
    csv_path = TRACKS_DIR / f"{name}.csv"
    if not csv_path.exists():
        available = list_tracks()
        raise FileNotFoundError(
            f"Track '{name}' not found. Available tracks: {available}"
        )
    return _load_csv(csv_path, num_nodes, smooth_sigma)


def list_tracks():
    """Return a sorted list of available CSV track names."""
    return sorted(p.stem for p in TRACKS_DIR.glob("*.csv"))


def berlin_2018(num_nodes=301, smooth_sigma=3.0):
    """Load the Berlin 2018 Formula E circuit."""
    return load_track("berlin_2018", num_nodes, smooth_sigma)


# ---------------------------------------------------------------------------
# Analytically defined oval track (Dymos example, not needed for the project)
# ---------------------------------------------------------------------------
def oval_dymos(num_nodes=301):
    """
    Analytically defined oval matching the Dymos racecar example.

    The track is built from 9 segments (counterclockwise, all left turns):
        150 m straight -> R=50 quarter-circle -> 100 m straight ->
        R=90 quarter-circle -> 300 m straight -> R=50 quarter-circle ->
        100 m straight -> R=90 quarter-circle -> 155 m straight

    Total arc length: 805 + 140*pi ~ 1244.8 m
    Track width: uniform +/- 4 m (8 m total)

    Unlike berlin_2018, this track does not need CSV loading or Gaussian
    smoothing because the geometry is defined analytically and curvature
    is computed from a fitted spline (so it is already smooth).
    """
    from scipy import interpolate as sci_interp

    # Each segment: [type, length_or_sweep, radius, direction]
    #   type: 0 = straight, 1 = corner
    #   direction: 0 = left, 1 = right, -1 = N/A (straight)
    segs_dymos = [
        [0, 150.0, 0.0, -1],
        [1, np.pi / 2, 50.0, 0],
        [0, 100.0, 0.0, -1],
        [1, np.pi / 2, 90.0, 0],
        [0, 300.0, 0.0, -1],
        [1, np.pi / 2, 50.0, 0],
        [0, 100.0, 0.0, -1],
        [1, np.pi / 2, 90.0, 0],
        [0, 155.0, 0.0, -1],
    ]

    # Build a set of (x, y) waypoints by walking along each segment
    pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    points = [[0.0, 0.0]]

    for seg in segs_dymos:
        seg_type, length, radius, side = seg
        if seg_type == 0:
            # Straight: sample every 5 m
            for j in range(1, int(length) - 1):
                if j % 5 == 0:
                    points.append(list(pos + direction * j))
            pos = pos + direction * length
        else:
            # Arc: find center, sweep the angle, sample every 10th point
            if side == 0:
                normal = np.array([-direction[1], direction[0]])
            else:
                normal = np.array([direction[1], -direction[0]])
            xc = pos[0] + radius * normal[0]
            yc = pos[1] + radius * normal[1]
            theta_line = np.arctan2(direction[1], direction[0])
            theta_0 = np.arctan2(pos[1] - yc, pos[0] - xc)
            if side == 0:
                theta_end = theta_0 + length
                direction = np.array(
                    [np.cos(theta_line + length), np.sin(theta_line + length)]
                )
            else:
                theta_end = theta_0 - length
                direction = np.array(
                    [np.cos(theta_line - length), np.sin(theta_line - length)]
                )
            theta_vec = np.linspace(theta_0, theta_end, 100)
            xarc = xc + radius * np.cos(theta_vec)
            yarc = yc + radius * np.sin(theta_vec)
            for j in range(len(xarc)):
                if j % 10 == 0:
                    points.append([xarc[j], yarc[j]])
            pos = np.array([xarc[-1], yarc[-1]])

    pts = np.array(points)

    # Fit a degree-5 spline through the waypoints (exact interpolation, s=0)
    tck, u_knots = sci_interp.splprep(pts.T, s=0.0, k=5)
    interval = 0.0001
    u_fine = np.arange(0, 1.0, interval)
    n_fine = len(u_fine)

    # Evaluate spline and its derivatives
    xy_fine = sci_interp.splev(u_fine, tck, der=0)
    dxy_fine = sci_interp.splev(u_fine, tck, der=1)
    ddxy_fine = sci_interp.splev(u_fine, tck, der=2)

    x_fine = np.asarray(xy_fine[0])
    y_fine = np.asarray(xy_fine[1])
    dx = np.asarray(dxy_fine[0])
    dy = np.asarray(dxy_fine[1])
    ddx = np.asarray(ddxy_fine[0])
    ddy = np.asarray(ddxy_fine[1])

    # Curvature from parametric derivatives: kappa = (x'y'' - y'x'') / speed^3
    speed = np.sqrt(dx**2 + dy**2)
    curv = (dx * ddy - dy * ddx) / speed**3

    # Convert from spline parameter u to physical arc-length
    du = u_fine[1] - u_fine[0]
    s_from_u = np.zeros(n_fine)
    s_from_u[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * du)
    s_total_spline = s_from_u[-1]

    # Nominal total length from segment definitions
    s_total_nominal = sum(
        seg[1] if seg[0] == 0 else seg[1] * seg[2] for seg in segs_dymos
    )

    # Sample num_nodes equally spaced points along the track
    s_nodes = np.linspace(0, s_total_nominal, num_nodes)
    indices = np.floor(s_nodes / s_total_nominal * n_fine).astype(int)
    indices = np.minimum(indices, n_fine - 1)
    kappa_nodes = curv[indices]
    x_nodes = x_fine[indices]
    y_nodes = y_fine[indices]

    s_fine_al = s_from_u * (s_total_nominal / s_total_spline)
    w_fine = 4.0 * np.ones(n_fine)

    return Track(
        name="oval_dymos",
        s_total=s_total_nominal,
        s=s_nodes,
        kappa=kappa_nodes,
        x=x_nodes,
        y=y_nodes,
        w_right=4.0 * np.ones(num_nodes),
        w_left=4.0 * np.ones(num_nodes),
        raw_x=x_fine,
        raw_y=y_fine,
        raw_s=s_fine_al,
        raw_w_right=w_fine,
        raw_w_left=w_fine,
    )
