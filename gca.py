from process import SOI
import numpy as np

np.set_printoptions(precision=3)
from scipy.integrate import quad
import csv


class GCA:
    def __init__(self, drawn_dimensions_filename, process=SOI(), x0=None, Fescon=1., Fbcon=1., Fkcon=1., mcon=1.):
        self.process = process
        self.extract_real_dimensions_from_drawn_dimensions(drawn_dimensions_filename)

        self.x0 = np.array([0., 0.]) if (x0 is None) else x0
        self.terminate_simulation = lambda t, x: False  # Can be set manually if needed
        self.Fescon = Fescon
        self.Fbcon = Fbcon
        self.Fkcon = Fkcon
        self.mcon = mcon

        # To access important simulation variables after the simulation
        self.sim_log = {}

    def dx_dt(self, t, x, u):
        t_SOI = self.process.t_SOI
        density = self.process.density
        m = self.spineA * t_SOI * density
        m_spring = 2 * self.supportW * self.supportL * t_SOI * density
        m_eff = m + m_spring / 3
        m_total = m_eff + self.Nfing * self.process.density_fluid * (self.fingerL**2) * (self.process.t_SOI**2) / (
                2 * (self.fingerL + self.process.t_SOI))

        V, Fext = self.unzip_input(u)
        Fes = self.Fes(x, u)
        Fb = self.Fb(x, u)
        Fk = self.Fk(x, u)
        self.add_to_sim_log(['t', 'Fes', 'Fb', 'Fk'], [t, Fes, Fb, Fk])

        return np.array([x[1],
                         (Fes - Fb - Fk - Fext) / (self.mcon * m_total)])

    def Fes(self, x, u):
        x, xdot = self.unzip_state(x)
        V, Fext = self.unzip_input(u)

        print("Dimensions")
        print(self.gf, self.gb, V, self.process.E)
        print(self.fingerL, self.fingerL_buffer, self.fingerL_total, self.fingerW)
        print(x, xdot, V, Fext)

        def Fes_calc1():
            return self.Fescon * self.Nfing * 0.5 * (
                    V**2) * self.process.eps0 * self.process.t_SOI * self.fingerL * \
                   (1 / (self.gf - x)**2 - 1 / (self.gb + x)**2)

        def Fes_calc2():
            Estar = self.process.E / (1 - self.process.v**2)
            a = self.fingerL_buffer / self.fingerL_total
            gf, gb = self.gf - x, self.gb + x
            beta = self.gb / self.gf
            Vtilde = V * np.sqrt(6 * self.process.eps0 * self.fingerL_total**4 / (Estar * self.fingerW**3 * self.gf**3))
            l = (Vtilde**2 * (2 / beta**3 + 2))**(0.25)
            print(a, beta, Vtilde, l)

            A = np.array([[-a**2, -a**3, np.exp(-l * a), np.exp(l * a), np.sin(l * a), np.cos(l * a)],
                          [-2 * a, -3 * a**2, -l * np.exp(-l * a), l * np.exp(l * a), l * np.cos(l * a),
                           -l * np.sin(l * a)],
                          [-2, -6 * a, l**2 * np.exp(-l * a), l**2 * np.exp(l * a), -l**2 * np.sin(l * a),
                           -l**2 * np.cos(l * a)],
                          [0, -6, -l**3 * np.exp(-l * a), l**3 * np.exp(l * a), -l**3 * np.cos(l * a),
                           l**3 * np.sin(l * a)],
                          [0, 0, l**2 * np.exp(-l), l**2 * np.exp(l), -l**2 * np.sin(l), -l**2 * np.cos(l)],
                          [0, 0, -l**3 * np.exp(-l), l**3 * np.exp(l), -l**3 * np.cos(l), l**3 * np.sin(l)]])
            b = np.array([(beta**3 - beta) / (2 * beta**3 + 2), 0, 0, 0, 0, 0])
            b2, b3, c0, c1, c2, c3 = np.linalg.pinv(A).dot(b)  # np.linalg.solve(A, b)
            print(A, b, b2, b3, c0, c1, c2, c3)

            y = lambda xi: (xi < a) * (self.gf * (b2 * xi**2 + b3 * xi**3)) + \
                           (xi >= a) * (self.gf * (c0 * np.exp(-l * xi) + c1 * np.exp(l * xi) + c2 *
                                                   np.sin(l * xi) + c3 * np.cos(l * xi) - b[0]))

            # def y(xi):  # xi_tilde, not xi (just abbreviated for brevity)
            #     if xi < a:
            #         return self.gf * (b2 * xi**2 + b3 * xi**3)
            #     else:
            #         return self.gf * (
            #                 c0 * np.exp(-l * xi) + c1 * np.exp(l * xi) + c2 * np.sin(l * xi) + c3 * np.cos(l * xi) - b[
            #             0])

            dF_dx = lambda xi: self.Nfing * 0.5 * V**2 * self.process.eps0 * self.process.t_SOI * \
                               (1 / (self.gf - y(xi / self.fingerL_total))**2 -
                                1 / (self.gb + y(xi / self.fingerL_total))**2)

            print("y values", y(0), y(a/2), y(a), y(2*a), y(1))

            Fes = quad(dF_dx, a * self.fingerL_total, self.fingerL_total)[0]
            return self.Fescon * Fes

        print("Fes1: %0.3e, Fes2: %0.3e" % (Fes_calc1(), Fes_calc2()))
        if V == 0:  # for speed
            Fes = 0
        else:
            Fes = Fes_calc2()

        return Fes

    def Fk(self, x, u):
        x, xdot = self.unzip_state(x)
        return self.Fkcon * self.k_support * x

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

        def bsf_calc1():
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gf - x)**3)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gb + x)**3)
            bsf = bsff + bsfb
            return bsf

        def bsf_calc2():
            t_SOI_primef = t_SOI
            t_SOI_primeb = t_SOI
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gf - x)**3)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gb + x)**3)
            bsf_adjf = (4 * (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef) / (
                    (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef)
            bsf_adjb = (4 * (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb) / (
                    (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        def bsf_calc3():
            t_SOI_primef = t_SOI + 0.81 * (gf - x + 0.94 * self.process.mfp)
            t_SOI_primeb = t_SOI + 0.81 * (gb + x + 0.94 * self.process.mfp)
            # t_SOI_primef = t_SOI + 0.81*(1 + 0.94*self.process.mfp/self.process.t_ox)*self.process.t_ox
            # t_SOI_primeb = t_SOI + 0.81*(1 + 0.94*self.process.mfp/self.process.t_ox)*self.process.t_ox
            S1, S2 = max(fingerL, t_SOI_primef), min(fingerL, t_SOI_primef)
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gf - x)**3)
            S1, S2 = max(fingerL, t_SOI_primeb), min(fingerL, t_SOI_primeb)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gb + x)**3)
            bsf_adjf = (4 * (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef) / (
                    (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef)
            bsf_adjb = (4 * (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb) / (
                    (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        def bsf_calc4():
            muf = self.process.mu / (1 + 9.638 * np.power(self.process.mfp / (self.gf - x), 1.159))
            mub = self.process.mu / (1 + 9.638 * np.power(self.process.mfp / (self.gb + x), 1.159))
            t_SOI_primef = t_SOI + 0.81 * (gf - x + 0.94 * self.process.mfp)
            t_SOI_primeb = t_SOI + 0.81 * (gb + x + 0.94 * self.process.mfp)
            S1, S2 = max(fingerL, t_SOI_primef), min(fingerL, t_SOI_primef)
            bsff = muf * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gf - x)**3)
            S1, S2 = max(fingerL, t_SOI_primeb), min(fingerL, t_SOI_primeb)
            bsfb = mub * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / (gb + x)**3)
            bsf_adjf = (4 * (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef) / (
                    (gf - x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef)
            bsf_adjb = (4 * (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb) / (
                    (gb + x)**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        bsf = bsf_calc2()

        # Couette flow damping
        bcf = self.process.mu * self.spineA / self.process.t_ox

        # Total damping
        b = bsf + bcf
        return self.Fbcon * b * xdot

    def pulled_in(self, t, x):
        x, xdot = self.unzip_state(x)
        # print("Termination check: {}, {}".format(x, self.x_GCA))
        return x >= self.x_GCA

    def released(self, t, x):
        x, xdot = self.unzip_state(x)
        return x <= 0

    def x0_pullin(self):
        return np.array([0, 0])

    def x0_release(self, u):
        V, Fext = self.unzip_input(u)
        x = np.array([self.x_GCA, 0])
        Fes = self.Fes(x, u) / self.Nfing  # Fes for one finger!

        # Finger release dynamics
        I_fing = (self.fingerW**3) * self.process.t_SOI / 12
        k_fing = 8 * self.process.E * I_fing / (self.fingerL_total**3)
        m_fing = self.fingerW * self.fingerL * self.process.t_SOI * self.process.density
        x_fing = Fes / k_fing
        w1 = (1.875**2) * np.sqrt(self.process.E * I_fing / (self.process.density * self.process.t_SOI *
                                                             (self.fingerL_total**4) * self.fingerW))
        print("masses: ", m_fing, k_fing / w1**2)
        v_fing = w1 * x_fing

        # Spine axial spring compression
        k_spine = self.process.E * (self.process.t_SOI * self.spineW) / self.spineL
        # m_spine = self.mainspineA*self.process.t_SOI*self.process.density
        m_spine = self.spineW * self.spineL * self.process.t_SOI * self.process.density
        x_spine = self.Nfing * Fes / k_spine
        v_spine = x_spine * np.sqrt(k_spine / m_spine)

        print("k_spine", k_spine, "k_fing", k_fing)

        # Conservation of linear momentum
        v0 = (self.Nfing * m_fing * v_fing + m_spine * v_spine) / (self.Nfing * m_fing + m_spine)
        print('Release values (Fes, v_fing, v_spine, v0):', Fes, v_fing, v_spine, v0)
        return np.array([self.x_GCA, -v0])

    # Helper functions
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
            self.k_arm = self.process.E * (self.armW**3) * self.process.t_SOI / (self.armL**3)

        self.update_dependent_variables()

    def update_dependent_variables(self):
        # if not hasattr(self, "k_support"):  # Might be overridden if taking data from papers
        self.k_support = 2 * self.process.E * (self.supportW**3) * self.process.t_SOI / (self.supportL**3)
        self.gs = self.gf - self.x_GCA
        self.fingerL_total = self.fingerL + self.fingerL_buffer
        self.num_etch_holes = round((self.spineL - self.etch_hole_spacing - self.process.overetch) /
                                    (self.etch_hole_spacing + self.etch_hole_size))
        self.mainspineA = self.spineW * self.spineL - self.num_etch_holes * (self.etch_hole_size**2)
        self.spineA = self.mainspineA + self.Nfing * self.fingerL_total * self.fingerW + \
                      2 * self.gapstopW * self.gapstopL_half
        if hasattr(self, "armW"):  # GCA includes arm (attached to inchworm motor)
            self.spineA += self.armW * self.armL

    def add_to_sim_log(self, names, values):
        for name, value in zip(names, values):
            if name not in self.sim_log:
                self.sim_log[name] = np.array([])
            self.sim_log[name] = np.append(self.sim_log[name], value)

    @staticmethod
    def unzip_state(x):
        x, xdot = x
        return x, xdot

    @staticmethod
    def unzip_input(u):
        V = u[0]
        Fext = u[1]
        return V, Fext


if __name__ == "__main__":
    gca = GCA("fawn.csv")
    print(gca.process.overetch)
