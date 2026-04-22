import numpy as np
import amigo as am
from amigo.interp import BSpline
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
from scipy.interpolate import BSpline as scipyBSpline

num_ctrl_points = 10
xi_ctrl = np.linspace(0, 1, num_ctrl_points)

num_interp_points = 100
xi = np.linspace(0, 1, num_interp_points)

input_name = "Mach"
output_name = "CD"
bspline = BSpline(
    input_name, output_name, interp_points=xi, num_ctrl_points=num_ctrl_points, deriv=1
)

model = bspline.create_model("bspline")

model.build_module()
model.initialize()

# Create the problem
p = model.get_problem()
mat = p.create_matrix()
g = p.create_vector()

xm = model.create_vector()
x = xm.get_vector()

# Set the inputs to the model
xm["control_points.Mach"] = np.sin(2.0 * np.pi * xi_ctrl)

p.hessian(1.0, x, mat)
p.gradient(1.0, x, g)

dof = model.get_indices(["interp_values.CD"])
con = model.get_indices(["kernel.constraint"])

K = am.tocsr(mat)
K0 = K[con, :][:, dof]
x.get_array()[dof] -= spsolve(K0, g.get_array()[con])

ans = xm["interp_values.CD"]

plt.plot(xi, ans)

# Cubic spline degree
degree = 3

# Knot vector
t = bspline._compute_knots()

# Control points / coefficients
c = xm["control_points.Mach"]

# Construct spline
spline = scipyBSpline(t, c, degree)

# Evaluation points
x = np.linspace(t[0], t[-1], 200)
y = spline(x, nu=1)

plt.plot(x, y, "-.")
plt.show()
