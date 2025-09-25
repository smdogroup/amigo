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
        p_inlet=1.0,
        T_inlet=1.0,
        p_outlet=0.9,
        length=10.0,
        A_inlet=1.0,
        A_outlet=1.0,
        A_min=0.4,
    ):
        self.num_cells = num_cells
        self.gamma = gamma
        self.p_inlet = p_inlet
        self.T_inlet = T_inlet
        self.p_outlet = p_outlet
        self.length = length
        self.dx = length / self.num_cells

        self.roe = RoeFlux(self.gamma)
        self.nozzle = Nozzle(self.gamma, self.dx)
        self.inlet = InletFlux(self.gamma, self.T_inlet, self.p_inlet)
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
            F[i, :] = self.roe.flux(self.A[i], Q[i - 1, :], Q[i, :])

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

num_cells = 100
nozzle = NozzleFlow(num_cells=num_cells, gamma=gamma)

Q = np.zeros((num_cells, 3))
Q[:, 0] = 1.0
Q[:, 1] = 0.25
Q[:, 2] = 0.95

Q = nozzle.explicit(Q, dt=0.04, niters=10000)

fig, ax = plt.subplots(2, 1)
ax[0].plot(nozzle.x_cell, nozzle.A_cell)
ax[1].plot(nozzle.x_cell, nozzle.dAdx_cell)

rho = Q[:, 0]
u = Q[:, 1]
p = (gamma - 1.0) * (Q[:, 2] - rho * u**2)

fig, ax = plt.subplots(3, 1)
ax[0].plot(nozzle.x_cell, rho)
ax[1].plot(nozzle.x_cell, u)
ax[2].plot(nozzle.x_cell, p)
plt.show()
