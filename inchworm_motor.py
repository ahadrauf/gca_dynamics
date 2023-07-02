"""
Defines a laterally oriented electrostatic gap closing actuator (GCA)
"""

from process import SOI
import numpy as np
from scipy.integrate import quad, solve_bvp
import csv
from timeit import default_timer as timer


class InchwormMotor:
    def __init__(self, drawn_dimensions_filename, process=SOI(), x0=None, Fbcon=1., Fkcon=1., mcon=1.):
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
        self.Fbcon = Fbcon
        self.Fkcon = Fkcon
        self.mcon = mcon

        # To access important simulation variables after the simulation
        self.sim_log = {}

    def dx_dt(self, t, x, u, Fb_calc_method=2):
        """
        Calculates dx/dt for dynamics simulations
        :param t: The time of the simulation. Generally not used for dynamics.
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]
        :param Fb_calc_method: Which version of the damping force calculation to perform. Version 2 is used in the paper.
        :return: dx/dt (np.array)
        """
        Fext = self.unzip_input(u)
        Fb = self.Fb(x, u, calc_method=Fb_calc_method)
        Fk = self.Fk(x, u)
        self.add_to_sim_log(['t', 'Fb', 'Fk'], [t, Fb, Fk])

        return np.array([x[1], (Fext - Fb - Fk) / (self.mcon * self.m_total)])

    def Fk(self, x, u):
        """
        :param x: The state of the GCA [position of spine, velocity of spine]
        :param u: The inputs into the system [applied voltage, external applied load]. Not used.
        :return: Spring force
        """
        x, xdot = self.unzip_state(x)
        return self.Fkcon * self.shuttle_spring_k * x

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

        # Couette flow damping
        bcf = self.process.mu * (self.shuttle_area + self.shuttle_spring_area / 3) / self.process.t_ox
        return self.Fbcon * bcf * xdot

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

        self.Ngca = drawn_dimensions["N_gca"]
        self.alpha = drawn_dimensions["alpha"]
        self.shuttleW = drawn_dimensions["shuttleW"] - 2 * undercut
        self.shuttleL = drawn_dimensions["shuttleL"] - 2 * undercut
        self.shuttle_area = drawn_dimensions["shuttle_area"]
        self.shuttle_mass = drawn_dimensions["shuttle_mass"]
        self.shuttle_spring_k = drawn_dimensions["shuttle_spring_k"]
        self.shuttle_spring_area = drawn_dimensions["shuttle_spring_area"]
        self.update_dependent_variables()

    def update_dependent_variables(self):
        """
        Updates dependent variables. Call this after changing any of the independent device dimensions.
        :return: None
        """
        self.m_total = self.shuttle_mass + self.shuttle_spring_area * self.process.t_SOI * self.process.density / 3

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
        Fext = u[0]
        return Fext


if __name__ == "__main__":
    gca = InchwormMotor("../layouts/bliss.csv")
    print(gca.process.undercut)
