import numpy as np
import matplotlib.pylab as plt


class Nozzle:
    def __init__(self, gamma=1.4, dx=1.0):
        self.gam1 = gamma - 1.0
        self.dx = dx

        return

    def compute(self, Q, FL, FR, dAdx):
        # Compute the pressure
        rho = Q[0]
        u = Q[1] / Q[0]
        p = self.gam1 * (Q[2] - 0.5 * rho * u * u)

        R = np.array(
            [
                (FR[0] - FL[0]) / self.dx,
                (FR[1] - FL[1]) / self.dx - dAdx * p,
                (FR[2] - FL[2]) / self.dx,
            ]
        )

        return R


class RoeFlux:
    def __init__(self, gamma=1.4):
        self.gamma = gamma
        self.gam1 = gamma - 1.0
        self.ggam1 = gamma / (gamma - 1.0)

    def flux(self, A, QL, QR):
        # Compute the weights
        rhoL = QL[0]
        rhoR = QR[0]
        r = np.sqrt(rhoR / rhoL)
        wL = r / (r + 1.0)
        wR = 1.0 - wL

        # Compute the left and right states
        uL = QL[1] / QL[0]
        pL = self.gam1 * (QL[2] - 0.5 * rhoL * uL * uL)
        HL = self.ggam1 * (pL / rhoL) + 0.5 * uL * uL

        uR = QR[1] / QR[0]
        pR = self.gam1 * (QR[2] - 0.5 * rhoR * uR * uR)
        HR = self.ggam1 * (pR / rhoR) + 0.5 * uR * uR

        # Compute the left and right fluxes
        FL = [rhoL * uL, rhoL * uL * uL + pL, rhoL * HL * uL]
        FR = [rhoR * uR, rhoR * uR * uR + pR, rhoR * HR * uR]

        # Compute the Roe averages
        rho = r * rhoL
        u = wL * uL + wR * uR
        H = wL * HL + wR * HR

        a = np.sqrt(self.gam1 * (H - 0.5 * u * u))
        ainv = 1.0 / a

        fp = ainv * ainv * (pR - pL)
        fu = (uR - uL) * rho * ainv

        # Compute the weights
        w0 = ((rhoR - rhoL) - fp) * np.fabs(u)
        w1 = (0.5 * (fp + fu)) * np.fabs(u + a)
        w2 = (0.5 * (fp - fu)) * np.fabs(u - a)

        Fr = 0.5 * w0 * u * u + w1 * (H + u * a) + w2 * (H - u * a)

        F = np.array(
            [
                0.5 * A * ((FL[0] + FR[0]) - (w0 + w1 + w2)),
                0.5 * A * ((FL[1] + FR[1]) - (w0 * u + w1 * (u + a) + w2 * (u - a))),
                0.5 * A * ((FL[2] + FR[2]) - Fr),
            ]
        )

        return F

    def euler_flux(self, A, Q):
        rho = Q[0]
        u = Q[1] / Q[0]
        p = self.gam1 * (Q[2] - 0.5 * rho * u**2)
        H = self.ggam1 * (p / rho) + 0.5 * u**2

        F = np.array([A * rho * u, A * (rho * u**2 + p), A * rho * u * H])

        return F

    def jacobian(self, A, Q, dh=1e-30):
        J = np.zeros((3, 3))
        for i in range(3):
            Qp = Q.astype(complex)
            Qp[i] += 1j * dh
            J[:, i] = self.euler_flux(A, Qp).imag / dh

        return J

    def roe_flux(self, A, QL, QR, dh=1e-30):
        # Compute the weights
        rhoL = QL[0]
        rhoR = QR[0]
        r = np.sqrt(rhoR / rhoL)
        wL = r / (r + 1.0)
        wR = 1.0 - wL

        # Compute the left and right states
        uL = QL[1] / QL[0]
        pL = self.gam1 * (QL[2] - 0.5 * rhoL * uL * uL)
        HL = self.ggam1 * (pL / rhoL) + 0.5 * uL * uL

        uR = QR[1] / QR[0]
        pR = self.gam1 * (QR[2] - 0.5 * rhoR * uR * uR)
        HR = self.ggam1 * (pR / rhoR) + 0.5 * uR * uR

        # Compute the Roe averages
        rho = r * rhoL
        u = wL * uL + wR * uR
        H = wL * HL + wR * HR
        a2 = self.gam1 * (H - 0.5 * u**2)

        FL = self.euler_flux(A, QL)
        FR = self.euler_flux(A, QR)

        Q0 = np.array([rho, rho * u, rho * H - rho * a2 / self.gamma])
        J = self.jacobian(A, Q0)

        e, R = np.linalg.eig(J)
        Rinv = np.linalg.inv(R)

        D = np.zeros((3, 3))
        for i in range(3):
            D += np.fabs(e[i].real) * np.outer(R[:, i], Rinv[i, :])

        F = 0.5 * (FL + FR - D @ (QR - QL))

        return F

    def test(self, A, QL, QR):
        F1 = self.flux(A, QL, QR)
        F2 = self.roe_flux(A, QL, QR)

        print(np.fabs((F1 - F2) / F2))


class InletFlux:
    def __init__(self, gamma=1.4, T_res=1.0, p_res=1.0):
        self.gamma = gamma
        self.gam1 = gamma - 1.0
        self.ggam1 = gamma / (gamma - 1.0)

        self.T_res = T_res
        self.p_res = p_res
        self.rho_res = self.gamma * self.p_res / self.T_res
        self.S_res = self.rho_res**self.gamma / self.p_res

    def flux(self, A, Q):
        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = Q[1] / Q[0]
        p_int = self.gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = np.sqrt(self.gamma * p_int / rho_int)

        # Compute the two invariants
        a_res = np.sqrt(self.T_res)
        invar_neg = u_int - 2.0 * a_int / self.gam1
        invar_pos = 2 * a_res / self.gam1

        # Based on the invariants, compute the velocity and speed of sound
        u_b = 0.5 * (invar_neg + invar_pos)
        a_b = 0.25 * self.gam1 * (invar_pos - invar_neg)
        rho_b = (a_b * a_b * self.S_res / self.gamma) ** (1.0 / self.gam1)

        # Compute the remaining states
        p_b = rho_b * a_b**2 / self.gamma
        H_b = self.ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        # Compute the flux
        F = np.array(
            [rho_b * u_b * A, (rho_b * u_b**2 + p_b) * A, rho_b * u_b * H_b * A]
        )

        return F


class OutletFlux:
    def __init__(self, gamma=1.4, p_back=1.0):
        self.gamma = gamma
        self.gam1 = gamma - 1.0
        self.ggam1 = gamma / (gamma - 1.0)

        self.p_back = p_back

    def flux(self, A, Q):
        # Compute the velocity and speed of sound at the input
        rho_int = Q[0]
        u_int = Q[1] / Q[0]
        p_int = self.gam1 * (Q[2] - 0.5 * rho_int * u_int * u_int)
        a_int = np.sqrt(self.gamma * p_int / rho_int)

        # Compute the two invariants from the interior
        invar_int = u_int + 2.0 * a_int / self.gam1
        S_int = rho_int**self.gamma / p_int

        # Set the back pressure
        p_b = self.p_back

        # Keep the entropy from the interior S_int = rho_b**gamma / p_b
        rho_b = (p_b * S_int) ** (1.0 / self.gamma)
        a_b = np.sqrt(self.gamma * p_b / rho_b)
        u_b = invar_int - 2 * a_b / self.gam1

        # Set the enthalpy
        H_b = self.ggam1 * p_b / rho_b + 0.5 * u_b * u_b

        # Compute the flux
        F = np.array(
            [rho_b * u_b * A, (rho_b * u_b**2 + p_b) * A, rho_b * u_b * H_b * A]
        )

        return F


class NozzleFlow:
    def __init__(
        self,
        num_cells=200,
        gamma=1.4,
        p_res=1.0,
        T_res=1.0,
        p_outlet=0.9,
        length=10.0,
        A_inlet=1.0,
        A_outlet=1.0,
        A_min=0.4,
    ):
        self.num_cells = num_cells
        self.gamma = gamma
        self.p_res = p_res
        self.T_res = T_res
        self.p_outlet = p_outlet
        self.length = length
        self.dx = length / self.num_cells

        self.roe = RoeFlux(self.gamma)
        self.nozzle = Nozzle(self.gamma, self.dx)
        self.inlet = InletFlux(self.gamma, self.T_res, self.p_res)
        self.outlet = OutletFlux(self.gamma, self.p_outlet)

        a = (2 * A_inlet - 4 * A_min + 2 * A_outlet) / self.length**2
        b = (4 * A_min - A_outlet - 3 * A_inlet) / self.length
        c = A_inlet

        self.x = np.linspace(0, self.length, self.num_cells + 1)
        self.A = c + b * self.x + a * self.x**2

        self.x_cell = np.linspace(
            0.5 * self.dx, self.length - 0.5 * self.dx, self.num_cells
        )
        self.A_cell = c + b * self.x_cell + a * self.x_cell**2
        self.dAdx_cell = b + 2.0 * a * self.x_cell

        return

    def compute_residual(self, Q):

        # Compute the fluxes
        F = np.zeros((self.num_cells + 1, 3))

        # Compute the boundary fluxes
        F[0, :] = self.inlet.flux(self.A[0], Q[0, :])
        F[-1, :] = self.outlet.flux(self.A[-1], Q[-1, :])

        for i in range(1, self.num_cells):
            # F[i, :] = self.roe.flux(self.A[i], Q[i - 1, :], Q[i, :])
            F[i, :] = self.roe.roe_flux(self.A[i], Q[i - 1, :], Q[i, :])

        # Assemble the residual
        R = np.zeros((self.num_cells, 3))

        for i in range(self.num_cells):
            R[i, :] = self.nozzle.compute(
                Q[i, :], F[i, :], F[i + 1, :], self.dAdx_cell[i]
            )

        return R

    def explicit(self, Q, dt=1e-3, niters=1000):
        for i in range(niters):
            R = self.compute_residual(Q)
            Q -= dt * R / self.A_cell[:, None]

            if i % 20 == 0:
                print(f"R[{i}] {np.sqrt(np.sum(R**2))}")

        return Q


gamma = 1.4


# Set reference and reservoir temperature, pressure in physical units
gamma = 1.4
R = 287.0  # J/(kg K) Gas constant
T_reservoir = 300.0  # degK
p_reservoir = 100.0e3  # Pa
rho_reservoir = p_reservoir / (R * T_reservoir)  # kg / m^3

# Set the reference temperature
T_ref = T_reservoir
rho_ref = rho_reservoir

# Compute the remaining reference values
a_ref = np.sqrt(gamma * R * T_ref)
p_ref = rho_ref * a_ref**2

# Compute the non-dimensional output values
T_res = T_reservoir / T_ref
p_res = p_reservoir / p_ref

# Set the back-pressure
p_back = 0.9 * p_res

# Set values for the length
length = 10.0

# Number of cells
num_cells = 100

nozzle = NozzleFlow(
    num_cells=num_cells,
    gamma=gamma,
    p_res=p_res,
    T_res=T_res,
    p_outlet=p_back,
    A_inlet=1.0,
    A_outlet=1.0,
    A_min=0.4,
)

# Set the initial solution guess
rho = 1.0
u = 0.01
p = 1.0 / gamma
E = (p / rho) / (gamma - 1.0) + 0.5 * u**2
H = E + p / rho

Q = np.zeros((num_cells, 3))
Q[:, 0] = rho
Q[:, 1] = rho * u
Q[:, 2] = rho * E

# Q = nozzle.explicit(Q, dt=0.01, niters=3500)

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(nozzle.x_cell, nozzle.A_cell)
# ax[1].plot(nozzle.x_cell, nozzle.dAdx_cell)
# rho = Q[:, 0]
# u = Q[:, 1] / rho
# p = (gamma - 1.0) * (Q[:, 2] - rho * u**2)

# fig, ax = plt.subplots(4, 1)
# ax[0].plot(nozzle.x_cell, rho)
# ax[1].plot(nozzle.x_cell, u)
# ax[2].plot(nozzle.x_cell, p)
# ax[3].plot(nozzle.x_cell, rho**gamma / p)
# plt.show()

A = 0.73
QL = np.array([0.5, 0.25, 1.3])
QR = np.array([0.489, 0.245, 1.29])

roe = RoeFlux()
roe.test(A, QL, QR)
