"""
Defines an Assembly object. An Assembly is meant to be a collection of one or several different components, collected
together for easy simulation and coupling. The output of one component can easily by piped into the input of another
component, helping to faciliate coupled motion (for example, coupling two GCA's with an inchworm motor shuttle).
"""

import numpy as np
from gca import GCA
from inchworm_motor import InchwormMotor
from process import SOI


class Assembly:
    def __init__(self, process, **kwargs):
        self.parts = []
        self.process = process

    def x0(self):
        return np.hstack([part.x0 for part in self.parts])

    def dx_dt(self, t, X, U, verbose=False, **kwargs):
        x_parts = self.unzip_state(X)
        u_parts = self.unzip_input(X, U)
        dx_dt = np.hstack([part.dx_dt(t, x, u(t, x), **kwargs) for (part, x, u) in zip(self.parts, x_parts, u_parts)])
        if verbose:
            print("t: {}, x: {}, u: {}, dx/dt: {}".format(t, X, [u(t, x) for (x, u) in zip(x_parts, u_parts)], dx_dt))
        return dx_dt

    def unzip_state(self, x):
        """
        Unzips the assembly state x to give the part states [x_part0, x_part1, ...]
        :param x: assembly state (1D np.array)
        :param u: assembly input (1D np.array)
        :return: list of assembly states for each part (list of 1D np.arrays)
        """
        return NotImplementedError()

    def unzip_input(self, x, u):
        """
        Unzips the assembly input u to give the part inputs [u_part0, u_part1, ...]
        :param x: assembly state (1D np.array)
        :param u: assembly input (1D np.array)
        :return: list of assembly input for each part (list of 1D np.arrays)
        """
        return NotImplementedError()

    def terminate_simulation(self, t, x):
        x_parts = self.unzip_state(x)
        terminate = np.any([part.terminate_simulation(t, x) for part, x in zip(self.parts, x_parts)])
        return not terminate  # I don't know why I need to invert it, but it works


class AssemblyGCA(Assembly):
    def __init__(self, drawn_dimensions_filename="../layouts/fawn.csv", process=SOI(), **kwargs):
        Assembly.__init__(self, process=process, **kwargs)
        self.gca = GCA(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.parts = [self.gca]

    def unzip_state(self, x):
        return [x]

    def unzip_input(self, x, u):
        return [u]


class AssemblyInchwormMotor(Assembly):
    def __init__(self, period, drawn_dimensions_filename="../layouts/fawn.csv", process=SOI(), **kwargs):
        Assembly.__init__(self, process=process, **kwargs)
        self.gca_pullin = GCA(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.gca_release = GCA(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.inchworm = InchwormMotor(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.tanalpha = np.tan(self.inchworm.alpha)
        self.parts = [self.gca_pullin, self.gca_release, self.inchworm]
        self.period = period  # release pawl waits period/4 before retracting

    def dx_dt(self, t, X, U, verbose=False, **kwargs):
        x_parts = self.unzip_state(X)
        u_parts = self.unzip_input(X, U)

        dx_dt0 = self.gca_pullin.dx_dt(t, x_parts[0], u_parts[0](t, x_parts[0]), **kwargs)
        dx_dt1 = self.gca_release.dx_dt(t, x_parts[1], u_parts[1](t, x_parts[1]), **kwargs)

        if self.gca_pullin.x_impact <= X[0] < 0.99 * self.gca_pullin.x_GCA:
            Estar = self.gca_pullin.process.E / (1 - self.gca_pullin.process.v**2)
            I_pawl = self.gca_pullin.pawlW**3 * self.gca_pullin.process.t_SOI / 12
            k = 3 * Estar * I_pawl / self.gca_pullin.pawlL**3
            N = self.inchworm.Ngca * 2 if X[1] > 0 else 0
            N += self.inchworm.Ngca * 2 if X[3] > 0 else 0
            F_GCA = N * dx_dt0[1] * self.gca_pullin.m_total
            Fext_shuttle = -N * k * (X[0] - self.gca_pullin.x_impact) / np.cos(self.gca_pullin.alpha) - F_GCA / np.tan(
                self.gca_pullin.alpha) / 10
            # Fext_shuttle = -F_GCA / np.tan(self.gca_pullin.alpha) / 3
        else:
            Fext_shuttle = 0.
        u_parts[2] = lambda t, x: np.array([Fext_shuttle, ])

        dx_dt2 = self.inchworm.dx_dt(t, x_parts[2], u_parts[2](t, x_parts[2]), **kwargs)
        dx_dt = np.hstack([dx_dt0, dx_dt1, dx_dt2])
        # dx_dt = np.hstack([part.dx_dt(t, x, u(t, x), **kwargs) for (part, x, u) in zip(self.parts, x_parts, u_parts)])
        xp, dxp, xr, dxr, y, dy = X
        ddxp, ddxr, ddy = dx_dt[1], dx_dt[3], dx_dt[5]
        pawly = self.gca_pullin.pawlL * np.sin(self.gca_pullin.alpha)

        if t < self.period / 4:  # the release stage (V_release = 0) only occurs after T/4 time delay
            dx_dt[2], dx_dt[3] = 0., 0.
            dxr, ddxr = 0., 0.
        if xp >= self.gca_pullin.x_GCA:
            dx_dt[0], dx_dt[1] = 0., 0.
            dxp, ddxp = 0., 0.
        if (self.gca_pullin.impacted_shuttle(0, xp) or self.gca_release.impacted_shuttle(0, xr)) and dx_dt[4] < 0:
            dx_dt[4], dx_dt[5] = 0., 0.
            dy, ddy = 0., 0.

        # if self.gca_pullin.x_impact <= xp < self.gca_pullin.x_GCA:  # and dxp > 0
        # #     # ddy =
        # #     # dx_dt[5] = abs(self.tanalpha * ((self.gca_pullin.x_GCA - xp)*ddxp - dxp**2 - dy**2))
        # #     dx_dt[5] = abs((ddxp * (self.gca_pullin.x_GCA - xp) - dxp**2 - dy**2) / (
        # #                 (self.gca_pullin.x_GCA - xp) / self.tanalpha))
        # #     # dx_dt[5] = (dxp**2 + dy**2 + (self.gca_pullin.x_GCA - xp)*ddxp) / self.tanalpha
        #     # print(ddy, dx_dt[5])
        #     N = 1  # 2 * self.inchworm.Ngca
        #     F_GCA = ddxp * (self.gca_pullin.mcon * self.gca_pullin.m_total)
        #     F_shut = ddy * (self.inchworm.mcon * self.inchworm.m_total)
        #
        #     ddy_new = (F_shut * self.tanalpha / N + F_GCA - self.gca_pullin.m_total * (dxp**2 + dy**2) / pawly) / (
        #             self.inchworm.m_total * self.tanalpha / N + self.gca_pullin.m_total / self.tanalpha)
        #     ddxp_new = (xp * ddy_new + dxp**2 + dy**2) / pawly
        #     dx_dt[1] = ddxp_new
        #     dx_dt[5] = ddy_new
        # elif not self.gca_pullin.impacted_shuttle(xp) and self.gca_release.impacted_shuttle(xr):
        #     # stop pawl from slipping backwards if release pawl is still connected, but allow it to keep moving forwards
        #     if dx_dt[4] < 0:
        #         dx_dt[4] = 0.
        #         dx_dt[5] = 0.

        if verbose:
            print("t: {}, x: {}, u: {}, dx/dt: {}".format(t, X, [u(t, x) for (x, u) in zip(x_parts, u_parts)], dx_dt))
        # print(list(dx_dt), list(X))
        return dx_dt

    def unzip_state(self, x):
        return [x[:2], x[2:4], x[4:]]

    def unzip_input(self, x, u):
        return [u[0], u[1], u[2]]  # (V_pullin, F_ext,pullin), (V_release, F_ext,release), (F_ext,shuttle)
        # if self.gca_pullin.x_impact <= x[0] < self.gca_pullin.x_GCA:
        #     Estar = self.gca_pullin.process.E / (1 - self.gca_pullin.process.v**2)
        #     I_pawl = self.gca_pullin.pawlW**3 * self.gca_pullin.process.t_SOI / 12
        #     k = 3 * Estar * I_pawl / self.gca_pullin.pawlL**3
        #     N = self.inchworm.Ngca * 2
        #     Fext_shuttle = -N * k * (x[0] - self.gca_pullin.x_impact) / np.sin(self.gca_pullin.alpha)  # 65))
        # else:
        #     Fext_shuttle = 0.
        # return [u[0], u[1], lambda t, x: np.array([Fext_shuttle, ])]
