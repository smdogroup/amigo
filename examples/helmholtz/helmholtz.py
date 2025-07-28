import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import time
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix  # For visualization

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


def eval_shape_funcs(xi, eta):
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    return N, Nxi, Neta


def dot(N, u):
    return N[0] * u[0] + N[1] * u[1] + N[2] * u[2] + N[3] * u[3]


def compute_detJ(xi, eta, X, Y, vars):
    N, N_xi, N_ea = eval_shape_funcs(xi, eta)

    x_xi = dot(N_xi, X)
    x_ea = dot(N_ea, X)

    y_xi = dot(N_xi, Y)
    y_ea = dot(N_ea, Y)

    vars["detJ"] = x_xi * y_ea - x_ea * y_xi

    return x_xi, x_ea, y_xi, y_ea


def compute_shape_derivs(xi, eta, X, Y, vars):
    N, N_xi, N_ea = eval_shape_funcs(xi, eta)

    x_xi, x_ea, y_xi, y_ea = compute_detJ(xi, eta, X, Y, vars)
    detJ = vars["detJ"]

    invJ = vars["invJ"] = [[y_ea / detJ, -x_ea / detJ], [-y_xi / detJ, x_xi / detJ]]

    vars["Nx"] = [
        invJ[0][0] * N_xi[0] + invJ[1][0] * N_ea[0],
        invJ[0][0] * N_xi[1] + invJ[1][0] * N_ea[1],
        invJ[0][0] * N_xi[2] + invJ[1][0] * N_ea[2],
        invJ[0][0] * N_xi[3] + invJ[1][0] * N_ea[3],
    ]

    vars["Ny"] = [
        invJ[0][1] * N_xi[0] + invJ[1][1] * N_ea[0],
        invJ[0][1] * N_xi[1] + invJ[1][1] * N_ea[1],
        invJ[0][1] * N_xi[2] + invJ[1][1] * N_ea[2],
        invJ[0][1] * N_xi[3] + invJ[1][1] * N_ea[3],
    ]

    return N, N_xi, N_ea


class Helmholtz(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(4):
            args.append({"n": n})
        self.set_args(args)

        # The filter radius
        self.add_constant("r_filter", 0.1)

        # The x/y coordinates
        self.add_data("x_coord", shape=(4,))
        self.add_data("y_coord", shape=(4,))

        # The implicit topology input/output
        self.add_input("rho", shape=(4,))

        # Add the residual
        self.add_objective("obj")

        return

    def compute(self, n=None):
        qpts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        xi = qpts[n % 2]
        eta = qpts[n // 2]

        r = self.constants["r_filter"]
        rho = self.inputs["rho"]

        X = self.data["x_coord"]
        Y = self.data["y_coord"]

        N, N_xi, N_ea = compute_shape_derivs(xi, eta, X, Y, self.vars)

        detJ = self.vars["detJ"]
        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        x = self.vars["x"] = dot(N, X)
        y = self.vars["y"] = dot(N, Y)

        rho0 = self.vars["rho0"] = dot(N, rho)
        rho_x = self.vars["rho_x"] = dot(Nx, rho)
        rho_y = self.vars["rho_y"] = dot(Ny, rho)

        rhs = 1.0 - (x - 1) ** 2 + (y - 1) ** 2
        laplace = r * r * (rho_x * rho_x + rho_y * rho_y)

        self.objective["obj"] = 0.5 * detJ * (rho0 * rho0 - rhs * rho0 + laplace)

        return


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Filter input values
        self.add_input("rho")
        self.add_data("x_coord")
        self.add_data("y_coord")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--with-openmp",
    dest="use_openmp",
    action="store_true",
    default=False,
    help="Enable OpenMP",
)
parser.add_argument(
    "--order-type",
    choices=["amd", "nd", "natural"],
    default="amd",
    help="Ordering strategy to use (default: amd)",
)
parser.add_argument(
    "--order-for-block",
    dest="order_for_block",
    action="store_true",
    default=False,
    help="Order for 2x2 block KKT matrix",
)
args = parser.parse_args()

nx = 2 * 256
ny = 2 * 256
nnodes = (nx + 1) * (ny + 1)
nelems = nx * ny

nodes = np.arange(nnodes, dtype=int).reshape((nx + 1, ny + 1))

x_coord = np.zeros(nnodes)
y_coord = np.zeros(nnodes)
conn = np.zeros((nelems, 4), dtype=int)

xpts = np.linspace(0, 2, nx + 1)
ypts = np.linspace(0, 2, ny + 1)
for j in range(ny + 1):
    for i in range(nx + 1):
        x_coord[nodes[i, j]] = xpts[i]
        y_coord[nodes[i, j]] = ypts[j]

for i in range(nx):
    for j in range(ny):
        conn[ny * i + j, 0] = nodes[i, j]
        conn[ny * i + j, 1] = nodes[i + 1, j]
        conn[ny * i + j, 2] = nodes[i + 1, j + 1]
        conn[ny * i + j, 3] = nodes[i, j + 1]

comm = COMM_WORLD
comm_rank = 0
comm_size = 1
if comm is not None:
    comm_rank = comm.rank
    comm_size = comm.size

module_name = "helmholtz"
model = am.Model(module_name)

node_src = NodeSource()
model.add_component("src", nnodes, node_src)

helmholtz = Helmholtz()
model.add_component("helmholtz", nelems, helmholtz)

model.link("helmholtz.y_coord", "src.y_coord", tgt_indices=conn)
model.link("helmholtz.x_coord", "src.x_coord", tgt_indices=conn)
model.link("helmholtz.rho", "src.rho", tgt_indices=conn)

if args.build:
    compile_args = []
    link_args = ["-lblas", "-llapack"]
    define_macros = []
    if args.use_openmp:
        compile_args = ["-fopenmp"]
        link_args += ["-fopenmp"]
        define_macros = [("AMIGO_USE_OPENMP", "1")]

    model.build_module(
        compile_args=compile_args, link_args=link_args, define_macros=define_macros
    )

start = time.perf_counter()

if args.order_type == "amd":
    order_type = am.OrderingType.AMD
elif args.order_type == "nd":
    order_type = am.OrderingType.NESTED_DISSECTION
elif args.order_type == "natural":
    order_type = am.OrderingType.NESTED_DISSECTION

order_for_block = args.order_for_block
model.initialize(order_type=order_type, order_for_block=order_for_block, comm=comm)
serial_problem = model.get_opt_problem()

end = time.perf_counter()
if comm_rank == 0:
    print(f"Initialization time:        {end - start:.6f} seconds")
    print(f"Num variables:              {model.num_variables}")
    print(f"Num constraints:            {model.num_constraints}")

# Set the problem data
data = model.get_data_vector()

if comm_rank == 0:
    data["src.x_coord"] = x_coord
    data["src.y_coord"] = y_coord

if comm_size == 1:
    problem = serial_problem
else:
    mpi_problem = serial_problem.partition_from_root()

    mpi_data = mpi_problem.get_data_vector()
    serial_problem.scatter_data_vector(
        data.get_opt_problem_vec(), mpi_problem, mpi_data
    )

    problem = mpi_problem

start = time.perf_counter()
mat = problem.create_matrix()
diag = problem.create_vector()
end = time.perf_counter()
print(f"Matrix initialization time: {end - start:.6f} seconds")

# Vectors for solving the problem
x = problem.create_vector()
ans = problem.create_vector()
g = problem.create_vector()
rhs = problem.create_vector()

problem.gradient(x, ans)
problem.gradient(x, g)

start = time.perf_counter()
problem.hessian(x, mat)
end = time.perf_counter()
print(f"Matrix computation time:    {end - start:.6f} seconds")

if comm_size == 1:
    # Perform a cholesky factorization
    start = time.perf_counter()
    chol = am.SparseCholesky(mat)
    flag = chol.factor()
    print("flag = ", flag)
    chol.solve(ans)

    end = time.perf_counter()
    print(f"Factor and solve time:      {end - start:.6f} seconds")

    mat.mult(ans, rhs)
    norm = np.linalg.norm(rhs.get_array() - g.get_array())
    print(f"Residual norm:              {norm}")
else:
    petsc_mat = am.topetsc(mat)

    # Create KSP solver
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(petsc_mat)

    # GMRES with Additive Schwarz
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("asm")

    # Create the solution and right-hand-side
    petsc_ans = petsc_mat.createVecRight()
    petsc_rhs = petsc_mat.createVecRight()

    rhs = petsc_rhs.getArray()
    rhs[:] = g.get_array()[: rhs.shape[0]]
    ksp.solve(petsc_rhs, petsc_ans)
    ans.get_array()[: rhs.shape[0]] = petsc_ans.getArray()[:]

if comm_size == 1:
    ans_local = ans
else:
    ans_local = serial_problem.create_vector()
    serial_problem.gather_vector(mpi_problem, ans, ans_local)

if comm_rank == 0:
    X, Y = np.meshgrid(xpts, ypts)
    vals = ans_local.get_array()[model.get_indices("src.rho")]
    vals = vals.reshape((nx + 1, ny + 1))

    # Plot using contourf
    plt.contourf(X, Y, vals, levels=20, cmap="viridis")
    plt.colorbar(label="Z value")
    plt.xlabel("x")
    plt.ylabel("y")

if comm_size == 1:
    nrows, ncols, nnz, rowp, cols = mat.get_nonzero_structure()
    data = mat.get_data()
    data[:] = 1.0
    jac = csr_matrix((data, cols, rowp), shape=(nrows, ncols))

    plt.figure(figsize=(6, 6))
    plt.spy(jac, markersize=0.2)
    plt.title("Sparsity pattern of matrix A")

plt.show()
