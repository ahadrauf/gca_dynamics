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
    def __init__(self, drawn_dimensions_filename="../layouts/fawn.csv", process=SOI(), **kwargs):
        Assembly.__init__(self, process=process, **kwargs)
        self.gca_pullin = GCA(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.gca_release = GCA(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.inchworm = InchwormMotor(drawn_dimensions_filename=drawn_dimensions_filename, process=process)
        self.tanalpha = np.tan(self.inchworm.alpha)
        self.parts = [self.gca_pullin, self.gca_release, self.inchworm]

    def dx_dt(self, t, X, U, verbose=False, **kwargs):
        x_parts = self.unzip_state(X)
        u_parts = self.unzip_input(X, U)
        dx_dt = np.hstack([part.dx_dt(t, x, u(t, x), **kwargs) for (part, x, u) in zip(self.parts, x_parts, u_parts)])
        xp, dxp, xr, dxr, y, dy = X
        ddxp, ddxr, ddy = dx_dt[1], dx_dt[3], dx_dt[5]
        pawly = self.gca_pullin.pawlL * np.sin(self.gca_pullin.alpha)
        if self.gca_pullin.impacted_shuttle(xp):
            N = 2 * self.inchworm.Ngca
            F_GCA = ddxp * (self.gca_pullin.mcon * self.gca_pullin.m_total)
            F_shut = ddy * (self.inchworm.mcon * self.inchworm.m_total)

            ddxp_new = (F_shut * self.tanalpha / N + F_GCA - self.gca.m_total * (dxp**2 + dy**2) / pawly) / (
                    self.inchworm.m_total * self.tanalpha / N + self.gca_pullin.m_total / self.tanalpha)
            ddy_new = (xp * ddxp_new + dxp**2 + dy**2) / pawly
            dx_dt[1] = ddxp_new
            dx_dt[5] = ddy_new
        elif not self.gca_pullin.impacted_shuttle(xp) and self.gca_release.impacted_shuttle(xr):
            # stop pawl from slipping backwards if release pawl is still connected, but allow it to keep moving forwards
            if dx_dt[5] < 0:
                dx_dt[5] = 0.
                dx_dt[6] = 0.

        if verbose:
            print("t: {}, x: {}, u: {}, dx/dt: {}".format(t, X, [u(t, x) for (x, u) in zip(x_parts, u_parts)], dx_dt))
        return dx_dt

    def unzip_state(self, x):
        return [x[:2], x[2:4], x[4:]]

    def unzip_input(self, x, u):
        return [u[:2], u[2:4], u[4:]]  # (V_pullin, F_ext,pullin), (V_release, F_ext,release), (F_ext,shuttle)
