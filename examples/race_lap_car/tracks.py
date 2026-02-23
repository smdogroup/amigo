"""
Track loading and processing for racecar lap time optimization.
Tracks are stored as CSV files (TUMFTM format) in the tracks/ directory.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"


@dataclass
class Track:
    name: str
    s_total: float  # total track length [m]
    s: np.ndarray  # arc length at each node [m]
    kappa: np.ndarray  # curvature at each node [1/m]
    x: np.ndarray  # centerline x [m]
    y: np.ndarray  # centerline y [m]
    w_right: np.ndarray  # half-width to the right [m]
    w_left: np.ndarray  # half-width to the left [m]

    raw_x: np.ndarray
    raw_y: np.ndarray
    raw_s: np.ndarray
    raw_w_right: np.ndarray
    raw_w_left: np.ndarray


def _load_csv(csv_path, num_nodes, smooth_sigma=3.0):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    w_right_raw = data[:, 2]
    w_left_raw = data[:, 3]

    # Close the loop
    n_wrap = min(5, len(x_raw))
    x_closed = np.concatenate([x_raw, x_raw[:n_wrap]])
    y_closed = np.concatenate([y_raw, y_raw[:n_wrap]])

    dx = np.diff(x_closed)
    dy = np.diff(y_closed)
    ds_raw = np.sqrt(dx**2 + dy**2)
    s_closed = np.concatenate([[0], np.cumsum(ds_raw)])

    s_total = s_closed[len(x_raw)]
    s_raw = s_closed[: len(x_raw)]

    # kappa = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
    dxds = np.gradient(x_closed, s_closed)
    dyds = np.gradient(y_closed, s_closed)
    d2xds2 = np.gradient(dxds, s_closed)
    d2yds2 = np.gradient(dyds, s_closed)
    denom = (dxds**2 + dyds**2) ** 1.5
    kappa_closed = (dxds * d2yds2 - dyds * d2xds2) / denom
    kappa_raw = kappa_closed[: len(x_raw)]

    # Gaussian smoothing (periodic wrap)
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

    # Resample to evenly-spaced nodes
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


def load_track(name, num_nodes=301, smooth_sigma=3.0):
    csv_path = TRACKS_DIR / f"{name}.csv"
    if not csv_path.exists():
        available = list_tracks()
        raise FileNotFoundError(
            f"Track '{name}' not found. Available tracks: {available}"
        )
    return _load_csv(csv_path, num_nodes, smooth_sigma)


def list_tracks():
    return sorted(p.stem for p in TRACKS_DIR.glob("*.csv"))


def berlin_2018(num_nodes=301, smooth_sigma=3.0):
    return load_track("berlin_2018", num_nodes, smooth_sigma)


def modena_2019(num_nodes=301, smooth_sigma=3.0):
    return load_track("modena_2019", num_nodes, smooth_sigma)


def handling_track(num_nodes=301, smooth_sigma=3.0):
    return load_track("handling_track", num_nodes, smooth_sigma)


def rounded_rectangle(num_nodes=301, smooth_sigma=3.0):
    return load_track("rounded_rectangle", num_nodes, smooth_sigma)


def oval_dymos(num_nodes=301):
    """Analytically defined oval matching the Dymos racecar example track.

    Nine segments (counterclockwise, all turns are left):
        150 m straight  ->  R=50  quarter-circle  ->  100 m straight  ->
        R=90  quarter-circle  ->  300 m straight  ->  R=50  quarter-circle  ->
        100 m straight  ->  R=90  quarter-circle  ->  155 m straight

    Total arc length: 805 + 140*pi ~ 1244.8 m
    Track width: uniform +/- 4 m (8 m total).

    Curvature is computed by replicating the Dymos get_track_points +
    get_spline pipeline (dymos/examples/racecar/spline.py).
    """
    from scipy import interpolate as sci_interp

    # [type, length_or_sweep, radius, direction]
    # type: 0=straight, 1=corner; direction: 0=left, 1=right, -1=N/A
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

    pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    points = [[0.0, 0.0]]

    for seg in segs_dymos:
        seg_type, length, radius, side = seg
        if seg_type == 0:
            for j in range(1, int(length) - 1):
                if j % 5 == 0:
                    points.append(list(pos + direction * j))
            pos = pos + direction * length
        else:
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

    tck, u_knots = sci_interp.splprep(pts.T, s=0.0, k=5)
    interval = 0.0001
    u_fine = np.arange(0, 1.0, interval)
    n_fine = len(u_fine)

    xy_fine = sci_interp.splev(u_fine, tck, der=0)
    dxy_fine = sci_interp.splev(u_fine, tck, der=1)
    ddxy_fine = sci_interp.splev(u_fine, tck, der=2)

    x_fine = np.asarray(xy_fine[0])
    y_fine = np.asarray(xy_fine[1])
    dx = np.asarray(dxy_fine[0])
    dy = np.asarray(dxy_fine[1])
    ddx = np.asarray(ddxy_fine[0])
    ddy = np.asarray(ddxy_fine[1])

    speed = np.sqrt(dx**2 + dy**2)
    curv = (dx * ddy - dy * ddx) / speed**3

    # Integrate arc length (u is not arc length)
    du = u_fine[1] - u_fine[0]
    s_from_u = np.zeros(n_fine)
    s_from_u[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * du)
    s_total_spline = s_from_u[-1]

    s_total_nominal = sum(
        seg[1] if seg[0] == 0 else seg[1] * seg[2] for seg in segs_dymos
    )

    # Map optimization nodes to curvature index (replicates Dymos curvature.py)
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
