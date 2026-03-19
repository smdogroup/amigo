import numpy as np
import amigo as am
from amigo.rbf import RBF
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt


num_points = 100 * 100
output_name = "CD"
input_names = ["alpha", "Mach"]
theta = np.array([2.0, 2.0])

xt = np.random.uniform(size=(50, 2))
yt = np.cos(2 * np.pi * xt[:, 0]) * np.sin(2 * np.pi * xt[:, 1])

rbf = RBF(num_points, output_name, input_names, xt, yt, theta)
model = rbf.create_model("rbf")

model.build_module()
model.initialize()

# Set the RBF data in the model
rbf.set_data(model)

# Create the problem
p = model.get_problem()
mat = p.create_matrix()
g = p.create_vector()

xm = model.create_vector()
x = xm.get_vector()

xcoord = np.linspace(0, 1, 100)
X, Y = np.meshgrid(xcoord, xcoord)

# Set the inputs to the model
xm["src.alpha"] = X.flatten()
xm["src.Mach"] = Y.flatten()

p.hessian(1.0, x, mat)
p.gradient(1.0, x, g)

dof = model.get_indices(["src.CD"])
con = model.get_indices(["constraint.constraint"])

K = am.tocsr(mat)
K0 = K[con, :][:, dof]
x.get_array()[dof] -= spsolve(K0, g.get_array()[con])

ans = xm["src.CD"].reshape((100, 100))

plt.figure()
plt.contourf(X, Y, ans)
plt.show()
