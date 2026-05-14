"""Analyze the KKT matrix structure for block PSD design.

Self-contained: builds model from scratch, no import from spaceshuttle.py.
"""

import sys

sys.path.insert(0, r"c:\Users\merye\Documents\git\amigo")

import numpy as np
import math
import amigo as am

# ---- Minimal model rebuild ----
num_intervals = 100
tf_fixed = 2000.0
N = num_intervals + 1
Re = 20902900.0
mu_grav = 0.14076539e17
rho0 = 0.002378
h_r = 23800.0
S_ref = 2690.0
mass = 203000.0 / 32.174
a0, a1 = -0.20704, 0.029244
b0, b1, b2 = 0.07854, -0.61592e-2, 0.621408e-3
c0, c1, c2, c3 = 1.0672181, -0.19213774e-1, 0.21286289e-3, -0.10117249e-5
qU = 70.0
scaling = {"h": 1e5, "phi": 1.0, "theta": 1.0, "v": 1e4, "gamma": 1.0, "psi": 1.0}

h0_bc, phi0, theta0 = 260000.0, 0.0, 0.0
v0_bc, gamma0, psi0 = 25600.0, math.radians(-1.0), math.radians(90.0)
hf, vf, gamma_f = 80000.0, 2500.0, math.radians(-5.0)


class ShuttleState(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=6)
        self.add_input("u", shape=2)


class ShuttleCollocation(am.Component):
    def __init__(self, scaling, dt):
        super().__init__()
        self.scaling = scaling
        self.add_constant("dt", value=dt)
        self.add_constant("Re", value=Re)
        self.add_constant("mu", value=mu_grav)
        self.add_constant("rho0", value=rho0)
        self.add_constant("h_r", value=h_r)
        self.add_constant("S", value=S_ref)
        self.add_constant("mass", value=mass)
        self.add_constant("a0", value=a0)
        self.add_constant("a1", value=a1)
        self.add_constant("b0", value=b0)
        self.add_constant("b1", value=b1)
        self.add_constant("b2", value=b2)
        self.add_input("q1", shape=6)
        self.add_input("q2", shape=6)
        self.add_input("u1", shape=2)
        self.add_input("u2", shape=2)
        self.add_constraint("res", shape=6, label="residual")

    def _dynamics(self, q, u):
        h = self.scaling["h"] * q[0]
        theta = q[2]
        v = self.scaling["v"] * q[3]
        gamma = q[4]
        psi = q[5]
        alpha = u[0]
        beta = u[1]
        rho = self.constants["rho0"] * am.exp(-h / self.constants["h_r"])
        alpha_deg = alpha * (180.0 / math.pi)
        CL = self.constants["a0"] + self.constants["a1"] * alpha_deg
        CD = (
            self.constants["b0"]
            + self.constants["b1"] * alpha_deg
            + self.constants["b2"] * alpha_deg * alpha_deg
        )
        q_dyn = 0.5 * rho * v * v
        L = q_dyn * self.constants["S"] * CL
        D = q_dyn * self.constants["S"] * CD
        r = self.constants["Re"] + h
        g = self.constants["mu"] / (r * r)
        fh = v * am.sin(gamma)
        fphi = (v / r) * am.cos(gamma) * am.sin(psi) / am.cos(theta)
        ftheta = (v / r) * am.cos(gamma) * am.cos(psi)
        fv = -(D / self.constants["mass"]) - g * am.sin(gamma)
        fgamma = (L / (self.constants["mass"] * v)) * am.cos(beta) + am.cos(gamma) * (
            v / r - g / v
        )
        fpsi = (L * am.sin(beta)) / (self.constants["mass"] * v * am.cos(gamma)) + (
            v / (r * am.cos(theta))
        ) * am.cos(gamma) * am.sin(psi) * am.sin(theta)
        return [
            fh / self.scaling["h"],
            fphi,
            ftheta,
            fv / self.scaling["v"],
            fgamma,
            fpsi,
        ]

    def compute(self):
        q1, q2 = self.inputs["q1"], self.inputs["q2"]
        dt = self.constants["dt"]
        f1 = self._dynamics(q1, self.inputs["u1"])
        f2 = self._dynamics(q2, self.inputs["u2"])
        self.constraints["res"] = [
            q2[i] - q1[i] - 0.5 * dt * (f1[i] + f2[i]) for i in range(6)
        ]


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=6)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - h0_bc / scaling["h"],
            q[1] - phi0,
            q[2] - theta0,
            q[3] - v0_bc / scaling["v"],
            q[4] - gamma0,
            q[5] - psi0,
        ]


class FinalConditions(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("q", shape=6)
        self.add_constraint("res", shape=3)

    def compute(self):
        q = self.inputs["q"]
        self.constraints["res"] = [
            q[0] - hf / scaling["h"],
            q[3] - vf / scaling["v"],
            q[4] - gamma_f,
        ]


class HeatingConstraint(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("rho0", value=rho0)
        self.add_constant("h_r", value=h_r)
        self.add_constant("c0", value=c0)
        self.add_constant("c1", value=c1)
        self.add_constant("c2", value=c2)
        self.add_constant("c3", value=c3)
        self.add_constant("qU", value=qU)
        self.add_input("h")
        self.add_input("v")
        self.add_input("alpha")
        self.add_input("slack")
        self.add_constraint("res", shape=1)

    def compute(self):
        h = scaling["h"] * self.inputs["h"]
        v = scaling["v"] * self.inputs["v"]
        alpha_deg = self.inputs["alpha"] * (180.0 / math.pi)
        q_a = (
            self.constants["c0"]
            + self.constants["c1"] * alpha_deg
            + self.constants["c2"] * alpha_deg * alpha_deg
            + self.constants["c3"] * alpha_deg**3
        )
        q_r = (
            17700.0
            * am.sqrt(self.constants["rho0"])
            * am.exp(-h / (2.0 * self.constants["h_r"]))
            * (0.0001 * v) ** 3.07
        )
        self.constraints["res"] = [
            q_a * q_r / self.constants["qU"] - self.inputs["slack"]
        ]


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("theta_final")
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = -self.inputs["theta_final"]


dt = tf_fixed / num_intervals
model = am.Model("spaceshuttle_mod")
model.add_component("shuttle", N, ShuttleState())
model.add_component("colloc", num_intervals, ShuttleCollocation(scaling, dt))
model.add_component("ic", 1, InitialConditions())
model.add_component("fc", 1, FinalConditions())
model.add_component("heat", N, HeatingConstraint())
model.add_component("obj", 1, Objective())

for i in range(6):
    model.link(f"shuttle.q[:{num_intervals}, {i}]", f"colloc.q1[:, {i}]")
    model.link(f"shuttle.q[1:, {i}]", f"colloc.q2[:, {i}]")
for i in range(2):
    model.link(f"shuttle.u[:{num_intervals}, {i}]", f"colloc.u1[:, {i}]")
    model.link(f"shuttle.u[1:, {i}]", f"colloc.u2[:, {i}]")
model.link("shuttle.q[0, :]", "ic.q[0, :]")
model.link(f"shuttle.q[{num_intervals}, :]", "fc.q[0, :]")
model.link(f"shuttle.q[{num_intervals}, 2]", "obj.theta_final[0]")
model.link("shuttle.q[:, 0]", "heat.h[:]")
model.link("shuttle.q[:, 3]", "heat.v[:]")
model.link("shuttle.u[:, 0]", "heat.alpha[:]")

model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)
print(f"Variables: {model.num_variables}, Constraints: {model.num_constraints}")

# ---- Create optimizer just for KKT structure (no optimization) ----
x = model.create_vector()
h_init = np.linspace(h0_bc, hf, N)
v_init = np.linspace(v0_bc, vf, N)
x["shuttle.q[:, 0]"] = h_init / scaling["h"]
x["shuttle.q[:, 3]"] = v_init / scaling["v"]
x["shuttle.q[:, 4]"] = np.linspace(gamma0, gamma_f, N)
x["shuttle.q[:, 5]"] = np.full(N, psi0)
x["shuttle.u[:, 0]"] = np.radians(np.linspace(30.0, 10.0, N))
x["shuttle.u[:, 1]"] = math.radians(-75.0)
alpha_deg_init = np.degrees(np.radians(np.linspace(30.0, 10.0, N)))
q_a_init = c0 + c1 * alpha_deg_init + c2 * alpha_deg_init**2 + c3 * alpha_deg_init**3
q_r_init = (
    17700.0 * np.sqrt(rho0) * np.exp(-h_init / (2.0 * h_r)) * (0.0001 * v_init) ** 3.07
)
x["heat.slack"] = np.clip(q_a_init * q_r_init / qU, 0.01, 0.99)

lower = model.create_vector()
upper = model.create_vector()
lower["shuttle.q"] = -float("inf")
upper["shuttle.q"] = float("inf")
lower["shuttle.u[:, 0]"] = math.radians(-90.0)
upper["shuttle.u[:, 0]"] = math.radians(90.0)
lower["shuttle.u[:, 1]"] = math.radians(-89.0)
upper["shuttle.u[:, 1]"] = math.radians(1.0)
lower["heat.slack"] = 0.0
upper["heat.slack"] = 1.0

# Build optimizer — access _problem for CSR structure
opt = am.Optimizer(model, x, lower=lower, upper=upper)

from amigo.optimizer import PardisoSolver

problem = opt.problem
solver = PardisoSolver(problem)

rowp = solver.rowp
cols = solver.cols
nrows = solver.nrows
nnz = solver.nnz

mult_ind = np.array(problem.get_multiplier_indicator(), dtype=bool)
primal_indices = np.where(~mult_ind)[0]
n_primal = len(primal_indices)
n_dual = int(np.sum(mult_ind))

print(f"\nKKT: {nrows}x{nrows}, nnz={nnz}, primal={n_primal}, dual={n_dual}")

# ---- Primal-primal degree ----
degree = np.zeros(nrows, dtype=np.int64)
for i in primal_indices:
    for idx in range(rowp[i], rowp[i + 1]):
        j = cols[idx]
        if j != i and not mult_ind[j]:
            degree[i] += 1

pd = degree[primal_indices]
print(f"\n=== Primal-primal degree ===")
print(
    f"min={pd.min()}, max={pd.max()}, mean={pd.mean():.1f}, median={np.median(pd):.0f}"
)
for t in [5, 10, 15, 20, 30, 50]:
    c = np.sum(pd > t)
    if c > 0:
        print(f"  degree > {t}: {c}")

top = np.argsort(pd)[-10:][::-1]
print(f"Top 10 degree:")
for ti in top:
    print(f"  var {primal_indices[ti]}: degree {pd[ti]}")

# ---- Bandwidth ----
bws = []
for i in primal_indices:
    for idx in range(rowp[i], rowp[i + 1]):
        j = cols[idx]
        if j != i and not mult_ind[j]:
            bws.append(abs(int(i) - int(j)))
bws = np.array(bws)
print(f"\n=== Index-space bandwidth ===")
print(f"max={bws.max()}, mean={bws.mean():.1f}, median={np.median(bws):.0f}")
print(
    f"p90={np.percentile(bws, 90):.0f}, p95={np.percentile(bws, 95):.0f}, p99={np.percentile(bws, 99):.0f}"
)

# ---- Connected components ----
primal_set = set(primal_indices.tolist())
adj = {int(p): [] for p in primal_indices}
for i in primal_indices:
    for idx in range(rowp[i], rowp[i + 1]):
        j = cols[idx]
        if j != i and int(j) in primal_set:
            adj[int(i)].append(int(j))

visited = set()
components = []
for start in primal_indices:
    s = int(start)
    if s in visited:
        continue
    comp = []
    queue = [s]
    visited.add(s)
    while queue:
        v = queue.pop(0)
        comp.append(v)
        for nb in adj[v]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    components.append(sorted(comp))

print(f"\n=== Connected components: {len(components)} ===")
for i, comp in enumerate(components[:5]):
    print(f"  #{i}: size={len(comp)}, range=[{comp[0]}, {comp[-1]}]")

# ---- BFS on largest component ----
largest = max(components, key=len)
comp_set = set(largest)
print(f"\n=== Largest component: {len(largest)} vars ===")

min_deg_var = min(largest, key=lambda v: len([nb for nb in adj[v] if nb in comp_set]))
print(
    f"Min-deg vertex: {min_deg_var}, deg={len([nb for nb in adj[min_deg_var] if nb in comp_set])}"
)

bfs_order = []
bfs_visited = {min_deg_var}
queue = [min_deg_var]
level_sizes = []
while queue:
    level_sizes.append(len(queue))
    nq = []
    for v in queue:
        bfs_order.append(v)
        for nb in sorted(adj[v]):
            if nb in comp_set and nb not in bfs_visited:
                bfs_visited.add(nb)
                nq.append(nb)
    queue = nq

print(f"BFS depth: {len(level_sizes)}")
print(f"Level sizes: {level_sizes[:40]}")

bfs_pos = {v: p for p, v in enumerate(bfs_order)}
max_bw = 0
bbws = []
for v in bfs_order:
    for nb in adj[v]:
        if nb in comp_set:
            bw = abs(bfs_pos[v] - bfs_pos[nb])
            bbws.append(bw)
            max_bw = max(max_bw, bw)
bbws = np.array(bbws)
print(
    f"BFS bandwidth: max={max_bw}, mean={bbws.mean():.1f}, median={np.median(bbws):.0f}"
)

# ---- Partitioning quality ----
print(f"\n=== Partitioning quality (BFS order) ===")
for bs in [4, 8, 10, 16, 20, 32]:
    cross = within = 0
    max_cross_row = 0
    for v in bfs_order:
        vb = bfs_pos[v] // bs
        cr = 0
        for nb in adj[v]:
            if nb in comp_set:
                if bfs_pos[nb] // bs == vb:
                    within += 1
                else:
                    cross += 1
                    cr += 1
        max_cross_row = max(max_cross_row, cr)
    tot = within + cross
    print(
        f"  bs={bs:3d}: {len(bfs_order)//bs+1:3d} blocks, "
        f"within={within/tot:.1%}, cross={cross/tot:.1%}, "
        f"max_cross_nbrs={max_cross_row}"
    )

# ---- Detailed adjacency for a few vars ----
print(f"\n=== First 5 primal vars ===")
for pi in range(min(5, n_primal)):
    i = primal_indices[pi]
    pn = sorted(
        j
        for idx in range(rowp[i], rowp[i + 1])
        for j in [cols[idx]]
        if j != i and not mult_ind[j]
    )
    dn = sum(
        1
        for idx in range(rowp[i], rowp[i + 1])
        for j in [cols[idx]]
        if j != i and mult_ind[j]
    )
    print(
        f"  var {i}: {len(pn)} primal nbs {pn[:15]}{'...' if len(pn)>15 else ''}, {dn} dual nbs"
    )

print("\nDone.")
