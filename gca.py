from process import SOI
import numpy as np
import csv


class GCA:
    def __init__(self, drawn_dimensions_filename, x0=None):
        self.process = SOI()
        self.extract_real_dimensions_from_drawn_dimensions(drawn_dimensions_filename)

        self.x0 = self.init_state() if (x0 is None) else x0
        self.terminate_simulation = lambda t, x: False  # Can be set manually if needed

    def dx_dt(self, t, x, u):
        t_SOI = self.process.t_SOI
        density = self.process.density
        m = self.spineA * t_SOI * density
        m_spring = 2 * self.supportW * self.supportL * t_SOI * density
        m_eff = m + m_spring / 3
        m_total = m_eff + self.Nfing * self.process.density_fluid * (self.fingerL ** 2) * (self.process.t_SOI ** 2) / (
                    2 * (self.fingerL + self.process.t_SOI))

        return np.array([x[1],
                         (self.Fes(x, u) - self.Fb(x, u) - self.Fk(x, u)) / m_total])

    def Fes(self, x, u):
        x, xdot = self.unzip_state(x)
        V = self.unzip_input(u)
        return self.Nfing * 0.5 * (V ** 2) * self.process.eps0 * self.process.t_SOI * self.fingerL * (
                1 / (self.gf - x) ** 2 - 1 / (self.gb + x) ** 2)

    def Fk(self, x, u):
        x, xdot = self.unzip_state(x)
        return self.k_support * x

    def Fb(self, x, u):
        x, xdot = self.unzip_state(x)
        fingerW = self.fingerW
        fingerL = self.fingerL
        t_SOI = self.process.t_SOI
        gf = self.gf
        gb = self.gb

        S1 = max(fingerL, t_SOI)
        S2 = min(fingerL, t_SOI)
        beta = lambda eta: 1 - (1 - 0.42) * eta

        t_SOI_primef = t_SOI + 0.81 * (1 + 0.94 * self.process.mfp / (gf - x)) * (gf - x)
        t_SOI_primeb = t_SOI + 0.81 * (1 + 0.94 * self.process.mfp / (gb + x)) * (gb + x)
        bsff = self.process.mu * self.Nfing * S1 * (S2 ** 3) * beta(S2 / S1) * (1 / (gf - x) ** 3)
        bsfb = self.process.mu * self.Nfing * S1 * (S2 ** 3) * beta(S2 / S1) * (1 / (gb + x) ** 3)
        bsf_adjf = (4 * (gf - x) ** 3 * fingerW + 2 * self.process.t_ox ** 3 * t_SOI_primef) / (
                (gf - x) ** 3 * fingerW + 2 * self.process.t_ox ** 3 * t_SOI_primef)
        bsf_adjb = (4 * (gb + x) ** 3 * fingerW + 2 * self.process.t_ox ** 3 * t_SOI_primeb) / (
                (gb + x) ** 3 * fingerW + 2 * self.process.t_ox ** 3 * t_SOI_primeb)
        bsf = bsff * bsf_adjf + bsfb * bsf_adjb

        # Couette flow damping
        bcf = self.process.mu * self.spineA / self.process.t_ox

        # Total damping
        b = bsf + bcf
        return b * xdot

    def pulled_in(self, t, x):
        x, xdot = self.unzip_state(x)
        return x >= self.x_GCA

    def pulled_out(self, t, x):
        x, xdot = self.unzip_state(x)
        return x <= 0

    ### Helper functions ###
    def extract_real_dimensions_from_drawn_dimensions(self, drawn_dimensions_filename):
        overetch = self.process.overetch

        drawn_dimensions = {}
        with open(drawn_dimensions_filename, 'r') as data:
            next(data)  # skip header row
            for name, value in csv.reader(data):
                drawn_dimensions[name] = float(value)

        self.gf = drawn_dimensions["gf"] + 2 * overetch
        self.gb = drawn_dimensions["gb"] + 2 * overetch
        self.x_GCA = drawn_dimensions["x_GCA"] + 2 * overetch
        self.supportW = drawn_dimensions["supportW"] - 2 * overetch
        self.supportL = drawn_dimensions["supportL"] - overetch
        self.Nfing = drawn_dimensions["Nfing"]
        self.fingerW = drawn_dimensions["fingerW"] - 2 * overetch
        self.fingerL = drawn_dimensions["fingerL"] - overetch
        self.fingerL_buffer = drawn_dimensions["fingerL_buffer"]
        self.spineW = drawn_dimensions["spineW"] - 2 * overetch
        self.spineL = drawn_dimensions["spineL"] - 2 * overetch
        self.etch_hole_size = drawn_dimensions["etch_hole_size"] + 2 * overetch
        self.etch_hole_spacing = drawn_dimensions["etch_hole_spacing"] - 2 * overetch
        self.gapstopW = drawn_dimensions["gapstopW"] - 2 * overetch
        self.gapstopL_half = drawn_dimensions["gapstopL_half"] - overetch
        self.anchored_electrodeW = drawn_dimensions["anchored_electrodeW"] - 2 * overetch
        self.anchored_electrodeL = drawn_dimensions["anchored_electrodeL"] - overetch

        # Simulating GCAs attached to inchworm motors
        if "armW" in drawn_dimensions:
            self.alpha = drawn_dimensions["alpha"]
            self.armW = drawn_dimensions["armW"] - 2 * overetch
            self.armL = drawn_dimensions["armL"] - overetch
            self.x_impact = drawn_dimensions["x_impact"]
            self.k_arm = self.process.E * (self.armW ** 3) * self.process.t_SOI / (self.armL ** 3)

        if not hasattr(self, "k_support"):  # Might be overridden if taking data from papers
            self.k_support = 2 * self.process.E * (self.supportW ** 3) * self.process.t_SOI / (self.supportL ** 3)
        if not hasattr(self, "gs"):
            self.gs = self.gf - self.x_GCA
        if not hasattr(self, "fingerL_total"):
            self.fingerL_total = self.fingerL + self.fingerL_buffer
        if not hasattr(self, "num_etch_holes"):
            self.num_etch_holes = round((self.spineL - self.etch_hole_spacing - overetch) /
                                        (self.etch_hole_spacing + self.etch_hole_size))
        if not hasattr(self, "mainspineA"):
            self.mainspineA = self.spineW * self.spineL - self.num_etch_holes * (self.etch_hole_size ** 2)
        if not hasattr(self, "spineA"):
            self.spineA = self.mainspineA + self.Nfing * self.fingerL_total * self.fingerW + \
                          2 * self.gapstopW * self.gapstopL_half
            if "armW" in drawn_dimensions:  # GCA includes arm (attached to inchworm motor)
                self.spineA += self.armW * self.armL

    def init_state(self):
        return np.array([0, 0])

    @staticmethod
    def unzip_state(x):
        x, xdot = x
        return x, xdot

    @staticmethod
    def unzip_input(u):
        V = u[0]
        return V


if __name__ == "__main__":
    gca = GCA("fawn.csv")
    print(gca.process.overetch)
