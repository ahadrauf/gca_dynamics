import numpy as np
from gca import GCA


class Assembly:
    def __init__(self):
        self.parts = []

    def x0(self):
        return np.hstack([part.x0 for part in self.parts])

    def dx_dt(self, t, x, u):
        x_parts = self.unzip_state(x)
        u_parts = self.unzip_input(u)
        return np.hstack([part.dx_dt(t, x, u(t, x)) for (part, x, u) in zip(self.parts, x_parts, u_parts)])

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
        return np.any([part.terminate_simulation(t, x) for part, x in zip(self.parts, x_parts)])


class AssemblyGCA(Assembly):
    def __init__(self):
        Assembly.__init__(self)
        self.gca = GCA("fawn.csv")
        self.parts = [self.gca]

    def unzip_state(self, x):
        return [x]

    def unzip_input(self, u):
        return [u]
