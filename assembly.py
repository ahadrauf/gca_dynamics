"""
Defines an Assembly object. An Assembly is meant to be a collection of one or several different components, collected
together for easy simulation and coupling. The output of one component can easily by piped into the input of another
component, helping to faciliate coupled motion (for example, coupling two GCA's with an inchworm motor shuttle).
"""

import numpy as np
from gca import GCA
from process import SOI


class Assembly:
    def __init__(self, process, **kwargs):
        self.parts = []
        self.process = process

    def x0(self):
        return np.hstack([part.x0 for part in self.parts])

    def dx_dt(self, t, x, u, verbose=False, **kwargs):
        x_parts = self.unzip_state(x)
        u_parts = self.unzip_input(u)
        dx_dt = np.hstack([part.dx_dt(t, x, u(t, x), **kwargs) for (part, x, u) in zip(self.parts, x_parts, u_parts)])
        if verbose:
            print("t: {}, x: {}, u: {}, dx/dt: {}".format(t, x, u(t, x), dx_dt))
        return dx_dt

    def unzip_state(self, x):
        """
        Unzips the assembly state x to give the part states [x_part0, x_part1, ...]
        :param x: assembly state (1D np.array)
        :return: list of assembly states for each part (list of 1D np.arrays)
        """
        return NotImplementedError()

    def unzip_input(self, u):
        """
        Unzips the assembly input u to give the part inputs [u_part0, u_part1, ...]
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

    def unzip_input(self, u):
        return [u]
