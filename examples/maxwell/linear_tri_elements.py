import numpy as np


def eval_shape_funcs(xi, eta):
    N = np.array(
        [
            1.0 - xi - eta,
            xi,
            eta,
        ]
    )
    Nxi = np.array([-1.0, 1.0, 0.0])
    Neta = np.array([-1.0, 0.0, 1.0])

    return N, Nxi, Neta


def dot(N, u):
    return N[0] * u[0] + N[1] * u[1] + N[2] * u[2]


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
    ]

    vars["Ny"] = [
        invJ[0][1] * N_xi[0] + invJ[1][1] * N_ea[0],
        invJ[0][1] * N_xi[1] + invJ[1][1] * N_ea[1],
        invJ[0][1] * N_xi[2] + invJ[1][1] * N_ea[2],
    ]

    return N, N_xi, N_ea
