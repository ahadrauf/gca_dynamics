from process import SOI
import numpy as np
import csv


class GCA:
    def __init__(self, drawn_dimensions_filename, x0=None):
        self.process = SOI()
        self.extract_real_dimensions_from_drawn_dimensions(drawn_dimensions_filename)

        self.x0 = np.array([0., 0.]) if (x0 is None) else x0
        self.terminate_simulation = lambda t, x: False  # Can be set manually if needed
        self.extra_spring_area = 0.
        self.extra_spring_k = 0.

        # To access important simulation variables after the simulation
        self.sim_log = {}

    def dx_dt(self, t, x, u):
        t_SOI = self.process.t_SOI
        density = self.process.density
        m = self.spineA*t_SOI*density
        m_spring = (2*self.supportW*self.supportL + self.extra_spring_area)*t_SOI*density
        m_eff = m + m_spring/3
        # print(m, m_spring, 2*self.supportW*self.supportL, self.extra_spring_area)
        m_total = m_eff + self.Nfing*self.process.density_fluid*(self.fingerL_overlap**2)*(self.process.t_SOI**2)/(
                2*(self.fingerL_overlap + self.process.t_SOI))
        # print(m, m_spring/3, self.Nfing*self.process.density_fluid*(self.fingerL**2)*(self.process.t_SOI**2)/(
        #         2*(self.fingerL + self.process.t_SOI)))

        V, Fext = self.unzip_input(u)
        Fes = self.Fes(x, u)
        Fb = self.Fb(x, u)
        Fk = self.Fk(x, u)
        self.add_to_sim_log(['t', 'Fes', 'Fb', 'Fk'], [t, Fes, Fb, Fk])
        # print('t: {:.4E}, V: {:.1f}, x: {:.4E}, v: {:.4E} | Fes: {:.4E}, Fb: {:.4E}, Fk: {:.4E}, Fext: {:.4E}'.format(
        #     t, u[0], x[0], x[1], Fes, Fb, Fk, u[1]
        # ))
        # print('t:', t, '(x, v)', x, '(Fes, Fb, Fk, u)', Fes, Fb, Fk, u)

        return np.array([x[1],
                         (Fes - Fb - Fk - Fext)/m_total])

    def Fes(self, x, u):
        x, xdot = self.unzip_state(x)
        V, Fext = self.unzip_input(u)
        eps = 1e-12
        Ff = self.Nfing*0.5*(V**2)*self.process.eps0*self.process.t_SOI*self.fingerL_overlap/(self.gf - x)**2
        multf = self.gf*self.k_fing/(self.gf*self.k_fing - 2*Ff + eps)
        Fb = self.Nfing*0.5*(V**2)*self.process.eps0*self.process.t_SOI*self.fingerL_overlap/(self.gb + x)**2
        multb = self.gb*self.k_fing/(self.gb*self.k_fing - 2*Fb + eps)
        # return multf*Ff + multb*Fb
        return Ff + Fb

    def Fk(self, x, u):
        x, xdot = self.unzip_state(x)
        # print(self.k_support*x, self.extra_spring_k*(x - self.x0[0]))
        return self.k_support*x  # + self.extra_spring_k*(x - (385.33e-6 + 2*self.process.overetch))

    def Fb(self, x, u):
        x, xdot = self.unzip_state(x)
        fingerW = self.fingerW
        fingerL = self.fingerL
        fingerL_overlap = self.fingerL_overlap
        t_SOI = self.process.t_SOI
        gf = self.gf
        gb = self.gb

        S1 = max(fingerL_overlap, t_SOI)
        S2 = min(fingerL_overlap, t_SOI)
        beta = lambda eta: 1 - (1 - 0.42)*eta

        def bsf_calc1():
            bsff = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gf - x)**3)
            bsfb = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gb + x)**3)
            bsf = bsff + bsfb
            return bsf

        def bsf_calc2():
            t_SOI_primef = t_SOI
            t_SOI_primeb = t_SOI
            bsff = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gf - x)**3)
            bsfb = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gb + x)**3)
            bsf_adjf = (4*(gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)/(
                    (gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)
            bsf_adjb = (4*(gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)/(
                    (gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)
            bsf = bsff*bsf_adjf + bsfb*bsf_adjb
            return bsf

        def bsf_calc3():
            t_SOI_primef = t_SOI + 0.81*(1 + 0.94*self.process.mfp/(gf - x))*(gf - x)
            t_SOI_primeb = t_SOI + 0.81*(1 + 0.94*self.process.mfp/(gb + x))*(gb + x)
            bsff = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gf - x)**3)
            bsfb = self.process.mu*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gb + x)**3)
            bsf_adjf = (4*(gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)/(
                    (gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)
            bsf_adjb = (4*(gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)/(
                    (gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)
            bsf = bsff*bsf_adjf + bsfb*bsf_adjb
            return bsf

        def bsf_calc4():
            """
            Based on T. Veijola, H. Kuisma, J. Lahdenperä, and T. Ryhänen, "Equivalent-circuit
            model of the squeezed gas film in a silicon accelerometer," Sensors and Actuators
            A: Physical, vol. 48, pp. 239-248, 1995.
            """
            t_SOI_primef = t_SOI + 0.81*(1 + 0.94*self.process.mfp/(gf - x))*(gf - x)
            t_SOI_primeb = t_SOI + 0.81*(1 + 0.94*self.process.mfp/(gb + x))*(gb + x)
            muf = self.process.mu / (1 + 9.638*np.power(self.process.mfp/(gf - x), 1.159))
            mub = self.process.mu/(1 + 9.638*np.power(self.process.mfp/(gb + x), 1.159))
            bsff = muf*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gf - x)**3)
            bsfb = mub*self.Nfing*S1*(S2**3)*beta(S2/S1)*(1/(gb + x)**3)
            bsf_adjf = (4*(gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)/(
                    (gf - x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primef)
            bsf_adjb = (4*(gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)/(
                    (gb + x)**3*fingerW + 2*self.process.t_ox**3*t_SOI_primeb)
            bsf = bsff*bsf_adjf + bsfb*bsf_adjb
            return bsf

        bsf = bsf_calc3()

        # Couette flow damping
        bcf = self.process.mu*self.spineA/self.process.t_ox

        # Total damping
        b = bsf + bcf
        return b*xdot

    def pulled_in(self, t, x):
        x, xdot = self.unzip_state(x)
        # print("Termination check: {}, {}".format(x, self.x_GCA))
        return x >= self.x_GCA

    def released(self, t, x):
        x, xdot = self.unzip_state(x)
        return x <= 0

    def x0_pullin(self):
        return np.array([0., 0.])

    def x0_release(self, u):
        V, Fext = self.unzip_input(u)
        x = np.array([self.x_GCA, 0])
        Fes = self.Fes(x, u)/self.Nfing  # Fes for one finger!

        # Finger release dynamics
        m_fing = self.fingerW*self.fingerL*self.process.t_SOI*self.process.density
        x_fing = Fes/self.k_fing
        w1 = (1.875**2)*np.sqrt(self.process.E*self.I_fing/(self.process.density*self.process.t_SOI*
                                                       (self.fingerL_total**4)*self.fingerW))
        v_fing = w1*x_fing/2

        # Spine axial spring compression
        k_spine = self.process.E*(self.process.t_SOI*self.spineW)/self.spineL
        m_spine = self.mainspineA*self.process.t_SOI*self.process.density
        x_spine = self.Nfing*Fes/k_spine
        v_spine = x_spine*np.sqrt(k_spine/m_spine)

        # Conservation of linear momentum
        v0 = (self.Nfing*m_fing*v_fing + m_spine*v_spine)/(self.Nfing*m_fing + m_spine)
        print('Release values (Fes, v_fing, v_spine, v0):', Fes, v_fing, v_spine, v0)
        return np.array([self.x_GCA, -v0])

    # Mostly for Craig's force test
    def add_support_spring(self, springW, springL, nBeams, endcapW, endcapL, etchholeSize, nEtchHoles, nEndCaps, k):
        # Include both half beams as one beam
        springW -= 2*self.process.overetch
        endcapW -= 2*self.process.overetch
        endcapL -= self.process.overetch
        etchholeSize += 2*self.process.overetch
        area = springW*springL*nBeams + (endcapW*endcapL - etchholeSize*etchholeSize*nEtchHoles)*nEndCaps
        self.extra_spring_area = area
        self.extra_spring_k = k


    # Helper functions
    def extract_real_dimensions_from_drawn_dimensions(self, drawn_dimensions_filename):
        overetch = self.process.overetch
        small_overetch = self.process.small_overetch
        sothresh = self.process.small_overetch_threshold

        drawn_dimensions = {}
        with open(drawn_dimensions_filename, 'r') as data:
            next(data)  # skip header row
            for info in csv.reader(data):
                name, value = info[:2]
                drawn_dimensions[name] = float(value)

        self.gf = drawn_dimensions["gf"] + 2*small_overetch if drawn_dimensions["gf"] < sothresh else drawn_dimensions["gf"] + 2*overetch
        self.gb = drawn_dimensions["gb"] + 2*overetch
        self.x_GCA = drawn_dimensions["x_GCA"] + 2*small_overetch if drawn_dimensions["x_GCA"] < sothresh else drawn_dimensions["x_GCA"] + 2*overetch
        self.supportW = drawn_dimensions["supportW"] - 2*overetch
        self.supportL = drawn_dimensions["supportL"] - overetch
        self.Nfing = drawn_dimensions["Nfing"]
        self.fingerW = drawn_dimensions["fingerW"] - small_overetch - overetch if drawn_dimensions["gf"] < sothresh else drawn_dimensions["fingerW"] - 2*overetch
        self.fingerL = drawn_dimensions["fingerL"] - overetch
        self.fingerL_buffer = drawn_dimensions["fingerL_buffer"]
        self.spineW = drawn_dimensions["spineW"] - 2*overetch
        self.spineL = drawn_dimensions["spineL"] - 2*overetch
        self.etch_hole_size = drawn_dimensions["etch_hole_size"] + 2*overetch
        self.etch_hole_spacing = drawn_dimensions["etch_hole_spacing"] - 2*overetch
        self.gapstopW = drawn_dimensions["gapstopW"] - 2*small_overetch if drawn_dimensions["x_GCA"] < sothresh else drawn_dimensions["gapstopW"] - 2*overetch
        self.gapstopL_half = drawn_dimensions["gapstopL_half"] - overetch
        self.anchored_electrodeW = drawn_dimensions["anchored_electrodeW"] - 2*overetch
        self.anchored_electrodeL = drawn_dimensions["anchored_electrodeL"] - overetch

        # Simulating GCAs attached to inchworm motors
        if "armW" in drawn_dimensions:
            self.alpha = drawn_dimensions["alpha"]
            self.armW = drawn_dimensions["armW"] - 2*overetch
            self.armL = drawn_dimensions["armL"] - overetch
            self.x_impact = drawn_dimensions["x_impact"]
            self.k_arm = self.process.E*(self.armW**3)*self.process.t_SOI/(self.armL**3)

        self.update_dependent_variables()

    def update_dependent_variables(self):
        if not hasattr(self, "k_support"):  # Might be overridden if taking data from papers
            self.k_support = 2*self.process.E*(self.supportW**3)*self.process.t_SOI/(self.supportL**3)
        self.gs = self.gf - self.x_GCA
        self.fingerL_overlap = self.fingerL - self.process.overetch
        self.fingerL_total = self.fingerL + self.fingerL_buffer
        self.I_fing = (self.fingerW**3)*self.process.t_SOI/12
        self.k_fing = 8*self.process.E*self.I_fing/(self.fingerL_total**3)
        print("k_fing", self.k_fing)
        self.num_etch_holes = round((self.spineL - self.etch_hole_spacing - self.process.overetch)/
                                    (self.etch_hole_spacing + self.etch_hole_size))
        self.mainspineA = self.spineW*self.spineL - self.num_etch_holes*(self.etch_hole_size**2)
        self.spineA = self.mainspineA + self.Nfing*self.fingerL_total*self.fingerW + \
                      4*self.gapstopW*self.gapstopL_half
        if hasattr(self, "armW"):  # GCA includes arm (attached to inchworm motor)
            self.spineA += self.armW*self.armL

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
