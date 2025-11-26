import amigo as am
import numpy as np  # used for plotting/analysis
import argparse
import time
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix  # For visualization
from parser import InpParser
from tabulate import tabulate
from scipy.sparse.linalg import spsolve
import examples.maxwell.utils as utils
from linear_tri_elements import (
    eval_shape_funcs,
    dot,
    compute_detJ,
    compute_shape_derivs,
)

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    COMM_WORLD = MPI.COMM_WORLD
except:
    COMM_WORLD = None


class Maxwell(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Material for each element
        self.add_data("alpha")

        # Define inputs to the problem
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract inputs
        u = self.inputs["u"]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea = compute_shape_derivs(xi, eta, X, Y, self.vars)

        # Extract material
        alpha = self.data["alpha"]

        # Set the values of the shape funcs derivs
        Nx = self.vars["Nx"]
        Ny = self.vars["Ny"]

        # Compute residual (R=Ku)
        detJ = self.vars["detJ"]

        # Compute the local element residual
        K00 = qwts[n] * detJ * alpha * (Nx[0] * Nx[0] + Ny[0] * Ny[0])
        K01 = qwts[n] * detJ * alpha * (Nx[0] * Nx[1] + Ny[0] * Ny[1])
        K02 = qwts[n] * detJ * alpha * (Nx[0] * Nx[2] + Ny[0] * Ny[2])

        K10 = qwts[n] * detJ * alpha * (Nx[1] * Nx[0] + Ny[1] * Ny[0])
        K11 = qwts[n] * detJ * alpha * (Nx[1] * Nx[1] + Ny[1] * Ny[1])
        K12 = qwts[n] * detJ * alpha * (Nx[1] * Nx[2] + Ny[1] * Ny[2])

        K20 = qwts[n] * detJ * alpha * (Nx[2] * Nx[0] + Ny[2] * Ny[0])
        K21 = qwts[n] * detJ * alpha * (Nx[2] * Nx[1] + Ny[2] * Ny[1])
        K22 = qwts[n] * detJ * alpha * (Nx[2] * Nx[2] + Ny[2] * Ny[2])

        res = self.vars["res"] = [
            K00 * u[0] + K01 * u[1] + K02 * u[2],
            K10 * u[0] + K11 * u[1] + K12 * u[2],
            K20 * u[0] + K21 * u[1] + K22 * u[2],
        ]

        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]


class Coil(am.Component):
    def __init__(self):
        super().__init__()

        # Add keyword arguments for the compute function
        args = []
        for n in range(3):
            # 4 gauss quadrature points
            args.append({"n": n})
        self.set_args(args)

        # x/y coords for each node
        self.add_data("x_coord", shape=(3,))
        self.add_data("y_coord", shape=(3,))

        # Constants
        self.add_constant("Jz", value=5.0)

        # Add input
        self.add_input("u", shape=(3,), value=0.0)  # Element solution

        # Objective
        self.add_objective("obj")
        return

    def compute(self, n=None):
        # Extract inputs
        u = self.inputs["u"]

        # Define gauss quad weights and points
        qwts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        qxi_qeta = [[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]]
        xi, eta = qxi_qeta[n]

        # Extract mesh data
        X = self.data["x_coord"]
        Y = self.data["y_coord"]
        N, N_xi, N_ea = compute_shape_derivs(xi, eta, X, Y, self.vars)
        detJ = self.vars["detJ"]
        Jz = self.constants["Jz"]

        res = self.vars["res"] = [
            -1 * qwts[n] * detJ * Jz * N[0],
            -1 * qwts[n] * detJ * Jz * N[1],
            -1 * qwts[n] * detJ * Jz * N[2],
        ]
        self.objective["obj"] = u[0] * res[0] + u[1] * res[1] + u[2] * res[2]
        return


class DirichletBc(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("u", value=1.0)
        self.add_input("lam", value=1.0)
        self.add_objective("obj")

        return

    def compute(self):
        self.objective["obj"] = self.inputs["u"] * self.inputs["lam"]
        return


class NodeSource(am.Component):
    def __init__(self):
        super().__init__()

        # Mesh coordinates
        self.add_data("x_coord")
        self.add_data("y_coord")

        # States
        self.add_input("u")
        return


class MaterialSource(am.Component):
    def __init__(self):
        super().__init__()
        self.add_data("alpha")
        return


if __name__ == "__main__":
    # Retrieve mesh information for the analysis
    inp_filename = "multimaterial.inp"
    parser = InpParser()
    parser.parse_inp(inp_filename)

    # Get the node locations
    X = parser.get_nodes()

    # Get element connectivity
    conn_surface1 = parser.get_conn("SURFACE1", "CPS3")
    conn_surface2 = parser.get_conn("SURFACE2", "CPS3")
    conn = np.concatenate((conn_surface1, conn_surface2))

    # Get the boundary condition nodes
    edge1 = parser.get_conn("LINE1", "T3D2")
    edge2 = parser.get_conn("LINE2", "T3D2")
    edge3 = parser.get_conn("LINE3", "T3D2")
    edge4 = parser.get_conn("LINE4", "T3D2")
    edge5 = parser.get_conn("LINE5", "T3D2")
    edge6 = parser.get_conn("LINE6", "T3D2")
    edge7 = parser.get_conn("LINE7", "T3D2")
    edge8 = parser.get_conn("LINE8", "T3D2")
    edge9 = parser.get_conn("LINE9", "T3D2")
    edge10 = parser.get_conn("LINE10", "T3D2")

    # Concatenate the unique node tgas for the dirichlet bc
    dirichlet_bc_tags = np.concatenate(
        (
            edge9.flatten(),
            edge8.flatten(),
            edge7.flatten(),
            edge6.flatten(),
        ),
        axis=None,
    )
    dirichlet_bc_tags = np.unique(dirichlet_bc_tags, sorted=True)

    # Total number of elements and nodes in the mesh
    nelems = conn.shape[0]
    nnodes = X.shape[0]

    # Define parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=False,
        help="Enable building",
    )
    parser.add_argument(
        "--order-type",
        choices=["amd", "nd", "natural"],
        default="nd",
        help="Ordering strategy to use (default: amd)",
    )
    parser.add_argument(
        "--order-for-block",
        dest="order_for_block",
        action="store_true",
        default=False,
        help="Order for 2x2 block KKT matrix",
    )
    parser.add_argument(
        "--show-sparsity",
        dest="show_sparsity",
        action="store_true",
        default=False,
        help="Show the sparsity pattern",
    )
    args = parser.parse_args()

    # Define amigo module
    model = am.Model("maxwell")

    # Maxwell physics
    maxwell = Maxwell()
    model.add_component(name="maxwell", size=nelems, comp_obj=maxwell)

    coils = Coil()
    model.add_component(name="coils", size=nelems, comp_obj=coils)

    dirichlet_bc = DirichletBc()
    model.add_component(
        "dirichlet_bc", size=len(dirichlet_bc_tags), comp_obj=dirichlet_bc
    )

    node_src = NodeSource()
    model.add_component("node_src", nnodes, node_src)

    materials = MaterialSource()
    model.add_component("material_src", 2, materials)  # 2 materials

    # Link the node coordinates to the the components
    # Ex: maxwell.y_coord = src.y_coord[conn]
    model.link("maxwell.x_coord", "node_src.x_coord", tgt_indices=conn)
    model.link("maxwell.y_coord", "node_src.y_coord", tgt_indices=conn)
    model.link("coils.x_coord", "node_src.x_coord", tgt_indices=conn)
    model.link("coils.y_coord", "node_src.y_coord", tgt_indices=conn)

    # Link the solution vectors
    model.link("coils.u", "node_src.u", tgt_indices=conn)
    model.link("maxwell.u", "node_src.u", tgt_indices=conn)

    # Link the material properties for each surface
    # Element tags for surface 1
    material1_indices = np.linspace(
        0, len(conn_surface1) - 1, len(conn_surface1), dtype=int
    )

    # Element tags for surface 2
    material2_indices = np.linspace(
        0, len(conn_surface2) - 1, len(conn_surface2), dtype=int
    ) + int(len(conn_surface1))

    # material_src.alpha1 = maxwell.alpha[material1_indices]
    # Surface 1 = alpha[0]
    # Surface 2 = alpha[1]
    model.link("material_src.alpha[0]", "maxwell.alpha", tgt_indices=material1_indices)
    model.link("material_src.alpha[1]", "maxwell.alpha", tgt_indices=material2_indices)

    # Link dirichlet bc nodes
    model.link("node_src.u", "dirichlet_bc.u", src_indices=dirichlet_bc_tags)

    # Build module
    if args.build:
        model.build_module()

    # Initialize the model
    model.initialize()

    # Set the problem data
    data = model.get_data_vector()
    data["node_src.x_coord"] = X[:, 0]
    data["node_src.y_coord"] = X[:, 1]
    data["material_src.alpha[0]"] = 10.0
    data["material_src.alpha[1]"] = 100.0
    problem = model.get_opt_problem()

    mat = problem.create_matrix()

    # Vectors for solving the problem
    x = problem.create_vector()
    ans = problem.create_vector()
    g = problem.create_vector()
    rhs = problem.create_vector()
    problem.hessian(x, mat)
    problem.gradient(x, g)
    csr_mat = am.tocsr(mat)

    # Plot matrix
    # utils.plot_matrix(csr_mat.todense())
    # plt.show()

    # Plot solution field
    ans.get_array()[:] = spsolve(csr_mat, g.get_array())
    ans_local = ans
    vals = ans_local.get_array()[model.get_indices("node_src.u")]
    utils.plot_solution(X, conn, vals)
    plt.show()
