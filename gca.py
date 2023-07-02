"""
Defines a laterally oriented electrostatic gap closing actuator (GCA)
"""

from process import SOI
import numpy as np
from scipy.integrate import quad, solve_bvp
import csv
from timeit import default_timer as timer


class GCA:
    def __init__(self, drawn_dimensions_filename, process=SOI(), x0=None, Fescon=1., Fbcon=1., Fkcon=1., mcon=1.):
        """
        Sets up a GCA object
        :param drawn_dimensions_filename: A CSV file with the drawn dimensions of the physical layout. Often found in the layouts/ folder.
        :param process: Process parameters, from the process.py file.
        :param x0: The initial position of the GCA. Defaults to [0, 0] if not set explicitly, but it can be reset later by setting the variable gca.x0.
        :param Fescon: A multiplier applied to the electrostatic force during dynamics. Used for parameter sweeps.
        :param Fbcon: A multiplier applied to the damping force during dynamics. Used for parameter sweeps.
        :param Fkcon: A multiplier applied to the spring force during dynamics. Used for parameter sweeps.
        :param mcon: A multiplier applied to the mass during dynamics. Used for parameter sweeps.
        """
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

    def dx_dt(self, t, x, u, Fes_calc_method=2, Fb_calc_method=2):
        """
        Calculates dx/dt for dynamics simulations
        :param t: The time of the simulation. Generally not used for dynamics.
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]
        :param Fes_calc_method: Which version of the electrostatic force calculation to perform. Version 2 is used in the paper.
        :param Fb_calc_method: Which version of the damping force calculation to perform. Version 2 is used in the paper.
        :return: dx/dt (np.array)
        """
        V, Fext = self.unzip_input(u)
        Fes = self.Fes(x, u, calc_method=Fes_calc_method)
        Fb, Fbsf, Fbcf = self.Fb(x, u, calc_method=Fb_calc_method)
        Fk = self.Fk(x, u)
        self.add_to_sim_log(['t', 'Fes', 'Fb', 'Fk', 'Fbsf', 'Fbcf'], [t, Fes, Fb, Fk, Fbsf, Fbcf])

        return np.array([x[1],
                         (Fes - Fb - Fk - Fext) / (self.mcon * self.m_total)])

    def Fes(self, x, u, calc_method=2):
        """
        Electrostatic force felt by the GCA
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]
        :param calc_method: Which version of the electrostatic force calculation to perform. Version 2 is used in the
        paper. Only versions 1 and 2 are supported here.
        :return: Electrostatic force
        """
        x, xdot = self.unzip_state(x)
        V, Fext = self.unzip_input(u)

        if V == 0:  # for speed when modelling release dynamics
            Fes = 0
        else:
            if calc_method == 1:
                Fes = self.Fes_calc1(x, V)
            elif calc_method == 2:
                Fes = self.Fes_calc2(x, V)[0]
            else:
                Fes = 0

        return Fes  # Fescon handled within Fes functions

    def Fk(self, x, u):
        """
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]. Not used.
        :return: Spring force
        """
        x, xdot = self.unzip_state(x)
        k = self.k_support
        Fk = self.Fkcon * k * x

        # A stop-gap measure for velocity testing simulations, uncomment when testing modelling those
        # Note that 0.2um undercut for pawl-to-shuttle gaps, as mentioned in Craig's dissertation pg. 32
        # https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-73.pdf
        if hasattr(self, "x_impact") and x > (self.x_impact + 2 * 0.2e-6):
            Estar = self.process.E / (1 - self.process.v**2)
            I_pawl = self.pawlW**3 * self.process.t_SOI / 12
            k += 3 * Estar * I_pawl / self.pawlL**3
            Fk += self.Fkcon * k * (x - (self.x_impact + 2 * 0.2e-6)) / np.cos(self.alpha)  # 65))
        # if x > (3e-6 + 2*0.2e-6):
        #     Estar = self.process.E/(1 - self.process.v**2)
        #     w_pawl = 4e-6 - 2*self.process.undercut
        #     L_pawl = 122e-6
        #     I_pawl = w_pawl**3*self.process.t_SOI/12
        #     k += 3*Estar*I_pawl/L_pawl**3
        #     Fk += self.Fkcon*k*(x - (3e-6 + 2*0.2e-6))/np.cos(np.deg2rad(67.4))  # 65))

        return Fk

    def Fb(self, x, u, calc_method=2):
        """
        Damping force felt by the GCA
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]
        :param calc_method: Which version of the damping force calculation to perform. Version 2 is used in the
        paper. Versions 1-4 are supported here.
        :return: Damping force
        """
        x, xdot = self.unzip_state(x)
        fingerW = self.fingerW
        fingerL = self.fingerL
        t_SOI = self.process.t_SOI
        gf = self.gf - x
        gb = self.gb + x

        S1 = max(fingerL, t_SOI)
        S2 = min(fingerL, t_SOI)
        beta = lambda eta: 1 - (1 - 0.42) * eta

        def bsf_calc1():
            # Simple squeeze film damping formula
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gf**3)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gb**3)
            bsf = bsff + bsfb
            return bsf

        def bsf_calc2():
            # Squeeze film damping with a multiplier to account for fluid flow in the 2um gap between the fingers and substrate
            t_SOI_primef = t_SOI
            t_SOI_primeb = t_SOI
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) / (gf**3)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) / (gb**3)
            bsf_adjf = (4 * (gf**3) * fingerW + 2 * (self.process.t_ox**3) * t_SOI_primef) / (
                    (gf**3) * fingerW + 2 * (self.process.t_ox**3) * t_SOI_primef)
            bsf_adjb = (4 * (gb**3) * fingerW + 2 * (self.process.t_ox**3) * t_SOI_primeb) / (
                    (gb**3) * fingerW + 2 * (self.process.t_ox**3) * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        def bsf_calc3():
            # Method 2, but using a heuristic modification of the thickness term adapted from the source below
            # Didn't use it for the paper because (a) it didn't converge as well, and (b) because the mean free path
            # term used doesn't have a great definition in vacuum/water
            # Source: M. Li, V. Rouf, D. Horsley, “Substrate Effect in Squeeze Film Damping of Lateral Oscillating
            # Microstructures”, Digest Tech. MEMS ’13 Conference, Taipei, January 20-24, 2013, pp. 393-396.
            t_SOI_primef = t_SOI + 0.81 * (gf - x + 0.94 * self.process.mfp)
            t_SOI_primeb = t_SOI + 0.81 * (gb + x + 0.94 * self.process.mfp)
            S1, S2 = max(fingerL, t_SOI_primef), min(fingerL, t_SOI_primef)
            bsff = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gf**3)
            S1, S2 = max(fingerL, t_SOI_primeb), min(fingerL, t_SOI_primeb)
            bsfb = self.process.mu * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gb**3)
            bsf_adjf = (4 * gf**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef) / (
                    gf**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef)
            bsf_adjb = (4 * gb**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb) / (
                    gb**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        def bsf_calc4():
            # Method 3, but using a heuristic calculation for the dynamic viscosity of a fluid from the below source
            # Didn't use it in the paper for similar reasons as Method 3
            # Source: T. Veijola, H. Kuisma, J. Lahdenperä, and T. Ryhänen, “Equivalent-circuit model of the squeezed
            # gas film in a silicon accelerometer,” Sensors and Actuators A: Physical, vol. 48, no. 3, pp. 239–248,
            # May 1995, doi: 10.1016/0924-4247(95)00995-7.
            muf = self.process.mu / (1 + 9.638 * np.power(self.process.mfp / (self.gf - x), 1.159))
            mub = self.process.mu / (1 + 9.638 * np.power(self.process.mfp / (self.gb + x), 1.159))
            t_SOI_primef = t_SOI + 0.81 * (gf - x + 0.94 * self.process.mfp)
            t_SOI_primeb = t_SOI + 0.81 * (gb + x + 0.94 * self.process.mfp)
            S1, S2 = max(fingerL, t_SOI_primef), min(fingerL, t_SOI_primef)
            bsff = muf * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gf**3)
            S1, S2 = max(fingerL, t_SOI_primeb), min(fingerL, t_SOI_primeb)
            bsfb = mub * self.Nfing * S1 * (S2**3) * beta(S2 / S1) * (1 / gb**3)
            bsf_adjf = (4 * gf**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef) / (
                    gf**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primef)
            bsf_adjb = (4 * gb**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb) / (
                    gb**3 * fingerW + 2 * self.process.t_ox**3 * t_SOI_primeb)
            bsf = bsff * bsf_adjf + bsfb * bsf_adjb
            return bsf

        calc_methods = [bsf_calc1, bsf_calc2, bsf_calc3, bsf_calc4]
        bsf = calc_methods[calc_method - 1]()

        # Couette flow damping
        bcf = self.process.mu * self.spineA / self.process.t_ox

        # Damping of support spring
        # CD = 2.  # https://en.wikipedia.org/wiki/Drag_coefficient
        # Ltot = self.supportL
        # r1 = lambda xi: Ltot*xi**2/4 - xi**3/6  # y(x)/y(L) for support spring
        # r2 = lambda xi: Ltot**3/12 - (Ltot*(Ltot-xi)**2/4 - (Ltot-xi)**3/6)
        # rL = r2(Ltot)
        # FD = 0.5*self.process.density_fluid*CD*self.process.t_SOI*(
        #     quad(lambda xi: (xdot*r1(xi)/rL), 0, rL/2)[0] + quad(lambda xi: (xdot*r2(xi)/rL), rL/2, rL)[0])

        # print("Damping:", gf, gb, bsf, bcf, bsf_calc1(), bsf_calc2(), bsf_calc3(), bsf_calc4())
        # print("Damping bsf bcf FD:", self.Fbcon*bsf*xdot, self.Fbcon*bcf*xdot, self.Fbcon*FD)

        # Total damping
        b = bsf + bcf
        return self.Fbcon * b * xdot, self.Fbcon * bsf * xdot, self.Fbcon * bcf * xdot

    def Fes_calc1(self, x, V):
        I_fing = (self.fingerW**3) * self.process.t_SOI / 12
        Estar = self.process.E / (1 - self.process.v**2)
        Fes_parallelplate = self.Nfing * 0.5 * V**2 * self.process.eps0 * self.process.t_SOI * self.fingerL * (
                1 / (self.gf - x)**2 - 1 / (self.gb + x)**2)
        wpp = Fes_parallelplate / self.fingerL_total
        y_pp = lambda xi: wpp / (24 * self.process.E * I_fing) * (
                xi**4 - 4 * self.fingerL_total * xi**3 + 6 * self.fingerL_total**2 * xi**2)
        dy_pp = lambda xi: wpp / (24 * self.process.E * I_fing) * (
                4 * xi**3 - 12 * self.fingerL_total * xi**2 + 12 * self.fingerL_total**2 * xi)
        ddy_pp = lambda xi: wpp / (24 * self.process.E * I_fing) * (
                12 * xi**2 - 24 * self.fingerL_total * xi + 12 * self.fingerL_total**2)
        E2_pp = 0.5 * Estar * I_fing * (
            quad(lambda xi: (ddy_pp(xi) / (1 + dy_pp(xi)**2)**1.5)**2, 0, self.fingerL_total)[0])
        return self.Fescon * Fes_parallelplate, [y_pp(xi) for xi in np.linspace(0, 1, 11)], E2_pp

    def Fes_calc2(self, x, V):
        """
        Modified electrostatic force calculation factoring in finger bending and fringe fields
        Derivation modified from:
        [1] D. Elata and V. Leus, “How slender can comb-drive fingers be?,” J. Micromech.
        Microeng., vol. 15, no. 5, pp. 1055–1059, May 2005, doi: 10.1088/0960-1317/15/5/023.
        [2] V. Leus, D. Elata, V. Leus, and D. Elata, “Fringing field effect in electrostatic actuators,” 2004.

        :param x: GCA state
        :param V: Input voltage
        :return: (Output force, an array (size 11,) of the finger deflection at evenly spaced intervals)
        """
        # start_time = timer()
        Estar = self.process.E / (1 - self.process.v**2)
        a = self.fingerL_buffer / self.fingerL_total
        gf, gb = self.gf - x, self.gb + x
        if gf < 0:
            print("gf < 0", gf, self.gf, x)
            # gf = 1e-12
            # gb = self.gb + self.gf
        beta = gb / gf
        Vtilde = V * np.sqrt(
            6 * self.process.eps0 * self.fingerL_total**4 / (Estar * self.fingerW**3 * gf**3)) * np.sqrt(self.Fescon)
        l = np.power(Vtilde**2 * (2 / beta**3 + 2), 0.25)
        b2, b3, c0, c1, c2, c3 = self.Fes_calc2_helper(x, V)

        # y = lambda xi: (xi < a) * (gf * (b2 * xi**2 + b3 * xi**3)) + \
        #                (xi >= a) * (gf * (c0 * np.exp(-l * xi) + c1 * np.exp(l * xi) + c2 *
        #                                        np.sin(l * xi) + c3 * np.cos(l * xi) - b[0]))
        y = lambda xi_tilde: (gf * (c0 * np.exp(-l * xi_tilde) + c1 * np.exp(l * xi_tilde) + c2 * np.sin(l * xi_tilde) +
                                    c3 * np.cos(l * xi_tilde) - (beta**3 - beta) / (
                                            2 * beta**3 + 2)))  # We only integrate over xi >= a

        dF_dx = lambda xi: self.Fescon * self.Nfing * 0.5 * V**2 * self.process.eps0 * self.process.t_SOI * \
                           (1 / (gf - y(xi / self.fingerL_total))**2 -
                            1 / (gb + y(xi / self.fingerL_total))**2)

        Fes = quad(dF_dx, a * self.fingerL_total, self.fingerL_total)[0]
        # end_time = timer()
        # print("Runtime for Fes_calc2, V =", V, "=", (end_time - start_time)*1e6, 'us --> ', Fes, y(1))

        I_fing = (self.fingerW**3) * self.process.t_SOI / 12
        # m_fing = self.fingerW*self.fingerL_total*self.process.t_SOI*self.process.density
        # x_fing = y(1.0)
        # w1 = (1.875**2)*np.sqrt(self.process.E*I_fing/(m_fing*(self.fingerL_total**3)))
        # v_fing = w1*x_fing/2
        # E_orig = 0.5*m_fing*v_fing**2

        b2m, b3m = b2 / self.fingerL_total**2, b3 / self.fingerL_total**3
        lm = l / self.fingerL_total
        dy1 = lambda xi: gf * (2 * b2m * xi + 3 * b3m * xi**2)
        dy2 = lambda xi: gf * (c0 * -lm * np.exp(-lm * xi) + c1 * lm * np.exp(lm * xi) + c2 * lm * np.cos(lm * xi) +
                               c3 * lm * -np.sin(lm * xi))
        ddy1 = lambda xi: gf * (2 * b2m + 6 * b3m * xi)
        ddy2 = lambda xi: gf * (
                c0 * lm**2 * np.exp(-lm * xi) + c1 * lm**2 * np.exp(lm * xi) + c2 * lm**2 * -np.sin(lm * xi) +
                c3 * lm**2 * -np.cos(lm * xi))

        # E1 = simplification of curvature of formula, E2 = actual curvature formula. Not much difference for small
        # deflections, but might as well use the full formula!
        # E1 = 0.5*Estar*I_fing*(quad(lambda xi: ddy1(xi)**2, 0, self.fingerL_buffer)[0] +
        #                        quad(lambda xi: ddy2(xi)**2, self.fingerL_buffer, self.fingerL_total)[0])
        E2 = 0.5 * Estar * I_fing * (quad(lambda xi: (ddy1(xi) / (1 + dy1(xi)**2)**1.5)**2, 0, self.fingerL_buffer)[0] +
                                     quad(lambda xi: (ddy2(xi) / (1 + dy2(xi)**2)**1.5)**2, self.fingerL_buffer,
                                          self.fingerL_total)[0])
        # print("Comparing energy stored in comb fingers: E_orig={}, E_calc1={}, E_calc2={}".format(E_orig, E1, E2))

        # calculate fringing field
        # Source: V. Leus, D. Elata, “Fringing field effect in electrostatic actuators,” 2004.
        h = gf
        t = self.fingerL
        w = self.process.t_SOI
        Fescon = 1 + h / np.pi / w * (1 + t / np.sqrt(t * h + t**2))  # F = 1/2*V^2*dC/dx (C taken from Eq. 10)
        # Fescon = 1.
        # print("Fescon", Fescon)
        return Fescon * Fes, [y(xi) for xi in np.linspace(0, 1, 11)], E2

    def Fes_calc2_helper(self, x, V):
        """
        Helper function for Fes_calc2(). Easier to debug this compared to the full output of Fes_calc2().

        :param x: GCA state
        :param V: Input voltage
        :return: (Output force, an array (size 11,) of the finger deflection at evenly spaced intervals)
        """
        Estar = self.process.E / (1 - self.process.v**2)
        a = self.fingerL_buffer / self.fingerL_total
        gf, gb = self.gf - x, self.gb + x
        if gf < 0:
            print("gf < 0", gf, self.gf, x)
            # gf = 1e-12
            # gb = self.gb + self.gf
        beta = gb / gf
        Vtilde = V * np.sqrt(
            6 * self.process.eps0 * self.fingerL_total**4 / (Estar * self.fingerW**3 * gf**3)) * np.sqrt(self.Fescon)
        l = np.power(Vtilde**2 * (2 / beta**3 + 2), 0.25)

        try:
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
            b2, b3, c0, c1, c2, c3 = np.linalg.solve(A, b)  # np.linalg.pinv(A).dot(b)
            return b2, b3, c0, c1, c2, c3
        except Exception as e:
            print("Exception occured:", a, gf, gb, beta, Vtilde, l)

    def Fes_calc3(self, x, V):
        """
        Written in response to reviewer comments asking whether finger bending between both the rotor and stator fingers
        made a significant difference. This is a heuristic approximation, since the full form of solving the differential
        equation based on both y(xi) and y(1-xi) is kinda complicated (it involves creating two functions f(xi)=y(xi) + y(1-xi)
        and g(xi) = y(xi) - y(1-xi), rewriting all the boundary conditions/kinematics in terms of f(xi) and g(xi), and solving
        the new boundary value problem). Instead, my heuristic was to calculate finger bending normally using
        Fes_calc2_helper(), then evaluating the integral for force by adding/subtracting y(1-xi) to the gap terms.

        Conclusion from my heuristic calculation below: dual finger bending doesn't matter
        nearly as much as single finger bending, which makes sense intuitively.

        :param x: GCA state
        :param V: Input voltage
        :return: (Output force, an array (size 11,) of the finger deflection at evenly spaced intervals)
        """
        Estar = self.process.E / (1 - self.process.v**2)
        a = self.fingerL_buffer / self.fingerL_total
        gf, gb = self.gf - x, self.gb + x
        if gf < 0:
            print("gf < 0", gf, self.gf, x)
            # gf = 1e-12
            # gb = self.gb + self.gf
        beta = gb / gf
        Vtilde = V * np.sqrt(
            6 * self.process.eps0 * self.fingerL_total**4 / (Estar * self.fingerW**3 * gf**3)) * np.sqrt(self.Fescon)
        l = np.power(Vtilde**2 * (2 / beta**3 + 2), 0.25)

        try:
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
            b2, b3, c0, c1, c2, c3 = np.linalg.solve(A, b)  # np.linalg.pinv(A).dot(b)
        except Exception as e:
            print("Exception occured:", a, gf, gb, beta, Vtilde, l)

        y = lambda xi_tilde: (gf * (c0 * np.exp(-l * xi_tilde) + c1 * np.exp(l * xi_tilde) + c2 *
                                    np.sin(l * xi_tilde) + c3 * np.cos(l * xi_tilde) - b[
                                        0]))  # We only integrate over xi >= a

        dF_dx = lambda xi: self.Fescon * self.Nfing * 0.5 * V**2 * self.process.eps0 * self.process.t_SOI * \
                           (1 / (gf - y(xi / self.fingerL_total) - y((1 + a) - xi / self.fingerL_total))**2 -
                            1 / (gb + y(xi / self.fingerL_total) + y((1 + a) - xi / self.fingerL_total))**2)

        Fes = quad(dF_dx, a * self.fingerL_total, self.fingerL_total)[0]

        I_fing = (self.fingerW**3) * self.process.t_SOI / 12

        b2m, b3m = b2 / self.fingerL_total**2, b3 / self.fingerL_total**3
        lm = l / self.fingerL_total
        dy1 = lambda xi: gf * (2 * b2m * xi + 3 * b3m * xi**2)
        dy2 = lambda xi: gf * (c0 * -lm * np.exp(-lm * xi) + c1 * lm * np.exp(lm * xi) + c2 * lm * np.cos(lm * xi) +
                               c3 * lm * -np.sin(lm * xi))
        ddy1 = lambda xi: gf * (2 * b2m + 6 * b3m * xi)
        ddy2 = lambda xi: gf * (
                c0 * lm**2 * np.exp(-lm * xi) + c1 * lm**2 * np.exp(lm * xi) + c2 * lm**2 * -np.sin(lm * xi) +
                c3 * lm**2 * -np.cos(lm * xi))

        E2 = 0.5 * Estar * I_fing * (quad(lambda xi: (ddy1(xi) / (1 + dy1(xi)**2)**1.5)**2, 0, self.fingerL_buffer)[0] +
                                     quad(lambda xi: (ddy2(xi) / (1 + dy2(xi)**2)**1.5)**2, self.fingerL_buffer,
                                          self.fingerL_total)[0])

        # calculate fringing field
        # Source: [1]V. Leus, D. Elata, V. Leus, and D. Elata, “Fringing field effect in electrostatic actuators,” 2004.
        h = gf
        t = self.fingerL
        w = self.process.t_SOI
        Fescon = 1 + h / np.pi / w * (1 + t / np.sqrt(t * h + t**2))  # F = 1/2*V^2*dC/dx (C taken from Eq. 10)
        # Fescon = 1.
        # print("Fescon", Fescon)
        return Fescon * Fes, [y(xi) for xi in np.arange(0, 1.1, 0.1)], E2

    def Fes_calc4(self, x, V):
        """
        Written in response to reviewer comments about how the linearized model compared to the numerical solution.
        Solves the bounded value problem in Eq. 20 with boundary conditions in Eq. 22/23.

        :param x: GCA state
        :param V: Input voltage
        :return: (Output force, an array (size 11,) of the finger deflection at evenly spaced intervals)
        """
        start_time = timer()
        Estar = self.process.E / (1 - self.process.v**2)
        I_fing = (self.fingerW**3) * self.process.t_SOI / 12
        a = self.fingerL_buffer / self.fingerL_total
        gf, gb = self.gf - x, self.gb + x
        if gf < 0:
            print("gf < 0", gf, self.gf, x)
        beta = gb / gf
        Vtilde = V * np.sqrt(
            6 * self.process.eps0 * self.fingerL_total**4 / (Estar * self.fingerW**3 * gf**3)) * np.sqrt(self.Fescon)
        l = np.power(Vtilde**2 * (2 / beta**3 + 2), 0.25)

        def dy_dxi(xi, state):
            y, dy, ddy, dddy = state
            ddddy = Vtilde**2 * (xi >= a) * (np.divide(1., np.square(1 - y)) - np.divide(1., np.square(beta + y)))
            ret = np.vstack((dy, ddy, dddy, ddddy))
            return ret

        def bc(ya, yb):
            return np.array([ya[0], ya[1], yb[2], yb[3]])  # yb[0] - y0[0, len(xi_range) - 1]])  #

        # xi = np.linspace(0., self.fingerL_total, 11)
        xi_range = np.linspace(0., 1., 1001)
        b2, b3, c0, c1, c2, c3 = self.Fes_calc2_helper(x, V)

        y = lambda xi: (xi < a) * (b2 * xi**2 + b3 * xi**3) + (xi >= a) * (c0 * np.exp(-l * xi) + c1 * np.exp(l * xi) +
                                                                           c2 * np.sin(l * xi) + c3 * np.cos(l * xi) -
                                                                           (beta**3 - beta) / (2 * beta**3 + 2))
        dy = lambda xi: (xi < a) * (2 * b2 * xi + 3 * b3 * xi**2) + (xi >= a) * (
                c0 * -l * np.exp(-l * xi) + c1 * l * np.exp(l * xi) +
                c2 * l * np.cos(l * xi) + c3 * -l * np.sin(l * xi))
        ddy = lambda xi: (xi < a) * (2 * b2 + 6 * b3 * xi) + (xi >= a) * (
                c0 * l**2 * np.exp(-l * xi) + c1 * l**2 * np.exp(l * xi) +
                c2 * -l**2 * np.sin(l * xi) + c3 * -l**2 * np.cos(l * xi))
        dddy = lambda xi: (xi < a) * (6 * b3) + (xi >= a) * (c0 * -l**3 * np.exp(-l * xi) + c1 * l**3 * np.exp(l * xi) +
                                                             c2 * -l**3 * np.cos(l * xi) + c3 * l**3 * np.sin(l * xi))
        y0 = np.zeros((4, np.size(xi_range)))
        for i in range(len(xi_range)):
            xi = xi_range[i]
            y0[0, i] = y(xi)
            y0[1, i] = dy(xi)
            y0[2, i] = ddy(xi)
            y0[3, i] = dddy(xi)

        sol = solve_bvp(dy_dxi, bc, xi_range, y0, verbose=0, tol=0.0001)

        y = lambda xi: gf * sol.sol(xi / self.fingerL_total)[0]

        dF_dx = lambda xi: self.Fescon * self.Nfing * 0.5 * V**2 * self.process.eps0 * self.process.t_SOI * \
                           (1 / (gf - y(xi))**2 - 1 / (gb + y(xi))**2)
        Fes = quad(dF_dx, a * self.fingerL_total, self.fingerL_total)[0]
        end_time = timer()
        # print("Runtime for Fes_calc4, V =", V, "=", (end_time - start_time)*1e6, 'us --> ', Fes, y(self.fingerL_total))

        # calculate fringing field
        # Source: [1]V. Leus, D. Elata, V. Leus, and D. Elata, “Fringing field effect in electrostatic actuators,” 2004.
        h = gf
        t = self.fingerL
        w = self.process.t_SOI
        Fescon = 1 + h / np.pi / w * (1 + t / np.sqrt(t * h + t**2))  # F = 1/2*V^2*dC/dx (C taken from Eq. 10)
        # Fescon = 1.
        # print("Fescon", Fescon)
        # return Fescon*Fes, sol, None
        return Fescon * Fes, [y(xi) for xi in np.linspace(0, self.fingerL_total, 11)], None

    def pulled_in(self, t, x):
        """
        Checks whether the GCA has pulled in, i.e. when the GCA spine hits the gap stop
        :param t: Time (unused)
        :param x: GCA state
        :return: True/False of whether the GCA has pulled in
        """
        x, xdot = self.unzip_state(x)
        # print("Termination check: {}, {}".format(x, self.x_GCA))
        return x >= self.x_GCA

    def released(self, t, x):
        """
        Checks whether the GCA has returned back to its initial position x[0] = 0
        :param t: Time (unused)
        :param x: GCA state
        :return: True/False of whether the GCA has returned to its initial position again
        """
        x, xdot = self.unzip_state(x)
        return x <= 0

    def x0_pullin(self):
        """
        Initial state for pullin simulations, i.e. the GCA begins at position x[0] = 0 and velocity x[1] = 0
        :return: Initial state for pullin simulations
        """
        return np.array([0., 0.])

    def x0_release(self, u, x_curr=None, v_curr=0.):
        """
        Initial state for release simulations

        :param u: Input = [V, Fext]
        :return: Initial state for release simulations
        """
        V, Fext = self.unzip_input(u)
        if x_curr is None:
            x = np.array([self.x_GCA, v_curr])
        else:
            x = np.array([x_curr, v_curr])
        # Fes = self.Fes(x, u)/self.Nfing  # Fes for one finger!
        Fes, y, Ues = self.Fes_calc2(x[0], V)
        # Fes, y, Ues = self.Fes_calc1(x[0], V)
        Fes = Fes / self.Nfing

        # Finger release dynamics
        # I_fing = (self.fingerW**3)*self.process.t_SOI/12
        # Estar = self.process.E/(1 - self.process.v**2)
        # k_fing = 8*self.process.E*I_fing/(self.fingerL_total**3)
        # m_fing = self.fingerW*self.fingerL_total*self.process.t_SOI*self.process.density
        # x_fing_orig = Fes/k_fing
        # Fes_parallelplate = self.Fes_calc1(x[0], V)[0]/self.Nfing
        # x_fing_orig_parallelplate = Fes_parallelplate/k_fing
        # x_fing = y[-1]
        # w1 = (1.875**2)*np.sqrt(self.process.E*I_fing/(m_fing*(self.fingerL_total**3)))
        # # m_fing_2 = k_fing/w1**2
        # m_fing_2 = Fes/x_fing/w1**2
        # v_fing = w1*x_fing/2
        # U_fing_parallelplate = 0.5*k_fing*x_fing_orig_parallelplate**2

        # wpp = Fes_parallelplate/self.fingerL_total
        # y_pp = lambda xi: wpp/(24*self.process.E*I_fing)*(
        #         xi**4 - 4*self.fingerL_total*xi**3 + 6*self.fingerL_total**2*xi**2)
        # dy_pp = lambda xi: wpp/(24*self.process.E*I_fing)*(
        #         4*xi**3 - 12*self.fingerL_total*xi**2 + 12*self.fingerL_total**2*xi)
        # ddy_pp = lambda xi: wpp/(24*self.process.E*I_fing)*(
        #         12*xi**2 - 24*self.fingerL_total*xi + 12*self.fingerL_total**2)
        # E2_pp = 0.5*Estar*I_fing*(quad(lambda xi: (ddy_pp(xi)/(1 + dy_pp(xi)**2)**1.5)**2, 0, self.fingerL_total)[0])
        #
        # print("Fes versions:", Fes, Fes_parallelplate)
        # print("x_fing versions:", x_fing, x_fing_orig_parallelplate, y_pp(self.fingerL_total))
        # print("k_eff versions:", Fes/x_fing, k_fing)
        # print("U_fing", Ues, U_fing_parallelplate, 0.5*Fes_parallelplate**2/k_fing, E2_pp)

        # Spine axial spring compression
        if not hasattr(self, "x_impact") or x[0] >= self.x_impact:
            k_spine = self.process.E * (self.process.t_SOI * self.spineW) / self.spineL
            m_spine = self.mainspineA * self.process.t_SOI * self.process.density
            m_spine_v2 = self.spineW * self.spineL * self.process.t_SOI * self.process.density
            x_spine = self.Nfing * Fes / k_spine
            v_spine = x_spine * np.sqrt(k_spine / m_spine)
            v_spine_v2 = x_spine * np.sqrt(k_spine / m_spine_v2)
            U_spine = 0.5 * k_spine * x_spine**2
        else:
            U_spine = 0.

        # Support spring bending
        m_support = 2 * self.supportW * self.supportL * self.process.t_SOI * self.process.density
        v_support = x[0] * np.sqrt(self.Fkcon * self.k_support / m_support)
        U_support = 0.5 * self.k_support * x[0]**2

        # print("k_spine", k_spine, "k_fing", k_fing)
        m_tot = self.spineA * self.process.t_SOI * self.process.density

        # Calculate Q factor
        # b = self.Fb((self.x_GCA, 1.), [0, 0])[0]/self.Nfing
        # k_fing_natural = 3*self.process.E*I_fing/(self.fingerL_total**3)
        # print("Q factor = sqrt(mk)/b, m={}, k={}, b={}".format(m_fing, k_fing_natural, b))
        # print("Q = ", np.sqrt(m_fing*k_fing_natural)/b)

        # Many many many heuristic calculations for initial velocity
        # Some conserve momentum, some energy, some include the support spring, some are just plain weird.
        # v11 turned out to be the best fit.
        # v0 = (self.Nfing*m_fing*v_fing + m_spine*v_spine + m_support*v_support)/(
        #         self.Nfing*m_fing + m_spine + m_support)
        # v0_orig = (self.Nfing*m_fing*v_fing + m_spine*v_spine)/(self.Nfing*m_fing + m_spine)
        # v0_2 = (self.Nfing*m_fing_2*v_fing + m_spine*v_spine)/(self.Nfing*m_fing_2 + m_spine)
        # v0_3 = (self.Nfing*m_fing*v_fing + m_spine_v2*v_spine_v2)/(self.Nfing*m_fing + m_spine_v2)
        # v0_4 = (self.Nfing*m_fing_2*v_fing + m_spine_v2*v_spine_v2)/(self.Nfing*m_fing_2 + m_spine_v2)
        # v0_5 = np.sqrt((self.Nfing*m_fing*v_fing**2 + m_spine*v_spine**2)/(self.Nfing*m_fing + m_spine))
        # v0_6 = np.sqrt((self.Nfing*m_fing_2*v_fing**2 + m_spine*v_spine**2)/(self.Nfing*m_fing_2 + m_spine))
        # v0_7 = np.sqrt((self.Nfing*m_fing*v_fing**2 + m_spine_v2*v_spine_v2**2)/(self.Nfing*m_fing + m_spine_v2))
        # v0_8 = np.sqrt((self.Nfing*m_fing_2*v_fing**2 + m_spine_v2*v_spine_v2**2)/(self.Nfing*m_fing_2 + m_spine_v2))
        # v0_9 = np.sqrt((self.Nfing*m_fing*v_fing**2 + m_spine*v_spine**2 + m_support*v_support**2)/(
        #         self.Nfing*m_fing + m_spine + m_support))
        # v0_10 = np.sqrt((self.Nfing*m_fing_2*v_fing**2 + m_spine*v_spine**2 + m_support*v_support**2)/(
        #         self.Nfing*m_fing_2 + m_spine + m_support))
        v0_11 = np.sqrt(2 * (self.Nfing * Ues + U_spine) / self.m_total)
        # v0_12 = np.sqrt(2*(self.Nfing*Ues + U_spine + U_support)/self.m_total)

        # print("xfing", x_fing_orig, x_fing, Fes, y)
        # print("masses: ", m_fing, k_fing/w1**2, m_spine, m_support)
        # print("velocities: ", m_spine, m_spine_v2, v_spine, v_spine_v2, v_support)
        # # print("Mass Dimension vs. Mass Spring", m_fing, m_fing_2)
        # print("Velocities with Momentum Conservation", v0, v0_orig, v0_2, v0_3, v0_4)
        # print("Velocities with Energy Conservation", v0_5, v0_6, v0_7, v0_8)
        # print('Release values (Fes, v_fing, v_spine, v0):', Fes, v_fing, v_spine, v0)
        # print("mf0 mfeff mshut mshutv2 Amainspine Ashut", m_fing, m_fing_2, m_spine, m_spine_v2, self.mainspineA,
        #       self.spineW*self.spineL)
        # print("Masses m_fing, m_spine, m_support, m_fing+m_spine+m_support, m_tot, m_total", self.Nfing*m_fing, m_spine,
        #       m_support, self.Nfing*m_fing + m_spine + m_support, m_tot, self.m_total)
        # print("Electrostatic Force Ifing Fes/0.5E_starI_fing", self.Nfing*Fes, I_fing, self.Nfing*Fes/0.5/Estar/I_fing)
        # print("Energies U_es U_spine U_support", self.Nfing*Ues, U_spine, U_support, self.Nfing*E2_pp)
        # print("Original Finger Energy", self.Nfing*U_fing_parallelplate)
        # print("Deformation of Finger", max(y))
        # print("Deformation of Spine", x_spine)
        # print("Predicted velocity", np.sqrt(Ues/(0.5*m_fing)), v0_11)
        # print("If omega kf F x_orig x varr vshut vsupport", I_fing, w1, k_fing, Fes, x_fing_orig, x_fing, v_fing,
        #       v_spine, v_support)
        # print("Release values (L, k, V, x0, v0_orig)", self.fingerL, self.k_support, V, self.x_GCA, v0_orig,
        #       "------ v0s", v0, v0_2, v0_3, v0_4, v0_5, v0_6, v0_7, v0_8, v0_9, v0_10, v0_11, v0_12)
        return np.array([x[0], x[1] - v0_11])

    # Helper functions
    def extract_real_dimensions_from_drawn_dimensions(self, drawn_dimensions_filename):
        """
        Extracts the real dimensions (including the effect of undercut) from the drawn dimensions file
        :param drawn_dimensions_filename: Path to the drawn dimensions file
        :return: None
        """
        undercut = self.process.undercut

        drawn_dimensions = {}
        with open(drawn_dimensions_filename, 'r') as data:
            next(data)  # skip header row
            for line in csv.reader(data):
                name, value = line[:2]
                drawn_dimensions[name] = float(value)

        self.gf = drawn_dimensions["gf"] + 2 * undercut
        self.gb = drawn_dimensions["gb"] + 2 * undercut
        self.x_GCA = drawn_dimensions["x_GCA"] + 2 * undercut
        self.supportW = drawn_dimensions["supportW"] - 2 * undercut
        self.supportL = drawn_dimensions["supportL"]  # - undercut
        self.Nfing = drawn_dimensions["Nfing"]
        self.fingerW = drawn_dimensions["fingerW"] - 2 * undercut
        self.fingerL = drawn_dimensions["fingerL"] - undercut
        self.fingerL_buffer = drawn_dimensions["fingerL_buffer"]
        self.spineW = drawn_dimensions["spineW"] - 2 * undercut
        self.spineL = drawn_dimensions["spineL"] - 2 * undercut
        self.etch_hole_spacing = drawn_dimensions["etch_hole_spacing"] - 2 * undercut
        self.gapstopW = drawn_dimensions["gapstopW"] - 2 * undercut
        self.gapstopL_half = drawn_dimensions["gapstopL_half"] - undercut
        # self.anchored_electrodeW = drawn_dimensions["anchored_electrodeW"] - 2*undercut
        # self.anchored_electrodeL = drawn_dimensions["anchored_electrodeL"] - undercut

        if "etch_hole_size" in drawn_dimensions:
            self.etch_hole_width = drawn_dimensions["etch_hole_size"] + 2 * undercut
            self.etch_hole_height = drawn_dimensions["etch_hole_size"] + 2 * undercut
        else:
            self.etch_hole_width = drawn_dimensions["etch_hole_width"] + 2 * undercut
            self.etch_hole_height = drawn_dimensions["etch_hole_height"] + 2 * undercut

        # Simulating GCAs attached to inchworm motors
        if "pawlW" in drawn_dimensions:
            print("Drawing contains pawl")
            self.alpha = np.deg2rad(drawn_dimensions["alpha"])
            self.pawlW = drawn_dimensions["pawlW"] - 2 * undercut
            self.pawlL = drawn_dimensions["pawlL"] - undercut
            self.x_impact = drawn_dimensions["x_impact"]
            self.k_arm = self.process.E * (self.pawlW**3) * self.process.t_SOI / (self.pawlL**3)

        self.update_dependent_variables()

    def update_dependent_variables(self):
        """
        Updates dependent variables. Call this after changing any of the independent device dimensions.
        :return: None
        """
        # if not hasattr(self, "k_support"):  # Might be overridden if taking data from papers
        Estar = self.process.E / (1 - self.process.v**2)
        self.k_support = 2 * Estar * (self.supportW**3) * self.process.t_SOI / (self.supportL**3)
        self.gs = self.gf - self.x_GCA
        self.fingerL_total = self.fingerL + self.fingerL_buffer
        self.num_etch_holes = round((self.spineL - self.etch_hole_spacing - self.process.undercut) /
                                    (self.etch_hole_spacing + self.etch_hole_width))
        self.mainspineA = self.spineW * self.spineL - self.num_etch_holes * (
                self.etch_hole_width * self.etch_hole_height)
        self.spineA = self.mainspineA + self.Nfing * self.fingerL_total * self.fingerW + 2 * self.gapstopW * self.gapstopL_half
        if hasattr(self, "pawlW"):  # GCA includes arm (attached to inchworm motor)
            self.spineA += self.pawlW * self.pawlL

        m = self.spineA * self.process.t_SOI * self.process.density
        m_spring = 2 * self.supportW * self.supportL * self.process.t_SOI * self.process.density
        m_eff = m + m_spring / 3
        self.m_total = m_eff + self.Nfing * self.process.density_fluid * (self.fingerL**2) * (self.process.t_SOI**2) / (
                2 * (self.fingerL + self.process.t_SOI))

        # print("spineW, spineL, num_etch_holes, mainspineA, spineA", self.spineW, self.spineL, self.num_etch_holes,
        #       self.mainspineA, self.spineA)

    def add_to_sim_log(self, names, values):
        for name, value in zip(names, values):
            if name not in self.sim_log:
                self.sim_log[name] = np.array([])
            self.sim_log[name] = np.append(self.sim_log[name], value)

    def impacted_shuttle(self, x):
        return x > (self.x_impact + 2 * 0.2e-6) if hasattr(self, "x_impact") else False

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
    gca = GCA("../layouts/fawn.csv")
    print(gca.process.undercut)
