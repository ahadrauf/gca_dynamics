"""
Defines the manufacturing processes by which we created our layouts. These parameters are specific to our testing setup
and manufacturing facilities. All of our chips were fabricated in the Marvell Nanofabrication Laboratory at UC Berkeley.
"""

class SOI:
    def __init__(self):
        self.undercut = 0.4e-6  # process undercut (m)
        self.small_undercut = 0.2e-6  # process undercut (m) for small gaps (<small_undercut_threshold)
        self.small_undercut_threshold = 3.5e-6  # threshold for applying small_undercut
        self.t_SOI = 40e-6  # thickness of silicon (m)
        self.t_ox = 2e-6  # thickness of oxide (m)
        self.t_gold = 500e-9  # depends on exact process (m)
        self.eps0 = 8.85e-12  # permittivity of free space
        self.E = 169e9  # Young's modulus of silicon (N/m2)
        self.v = 0.069  # Poisson's ratio (fabricated on a (100) wafer in the [110] direction)
        self.density = 2300  # density of silicon (kg/m3)
        self.density_fluid = 1.1839	 # density of air (kg/m3)
        self.Rs = 0.1/self.t_SOI  # sheet resistance of silicon (assuming resistivity = 10 Ohm-cm)
        self.Rs_gold = 2.4e-8/self.t_gold  # sheet resistance of gold
        self.mu = 1.85e-5  # dynamic viscosity of air
        self.mfp = 68e-9  # mean free path of air (m)

class SOIwater:
    def __init__(self):
        self.undercut = 0.5e-6  # process undercut (m)
        self.small_undercut = 0.19e-6  # process undercut (m) for small gaps (<4 um) (probably ignored for this work)
        self.small_undercut_threshold = 3.5e-6  # threshold for applying small_undercut
        self.t_SOI = 40e-6  # thickness of silicon (m)
        self.t_ox = 2e-6  # thickness of oxide (m)
        self.t_gold = 500e-9  # depends on exact process (m)
        self.eps0 = 8.85e-12*80  # permittivity of free space
        self.E = 169e9  # Young's modulus of silicon (N/m2)
        self.v = 0.069  # Poisson's ratio (fabricated on a (100) wafer in the [110] direction)
        self.density = 2300  # density of silicon (kg/m3)
        self.density_fluid = 1000.	 # density of water (kg/m3)
        self.Rs = 0.1/self.t_SOI  # sheet resistance of silicon (assuming resistivity = 10 Ohm-cm)
        self.Rs_gold = 2.4e-8/self.t_gold  # sheet resistance of gold
        self.mu = 8.88e-4  # dynamic viscosity of water
        self.mfp = 68e-9  # mean free path of air (m) <---- value for water?


class SOIvacuum:
    def __init__(self):
        self.undercut = 0.48e-6  # process undercut (m)
        self.small_undercut = 0.19e-6  # process undercut (m) for small gaps (<4 um)
        self.small_undercut_threshold = 3.5e-6  # threshold for applying small_undercut
        self.t_SOI = 40e-6  # thickness of silicon (m)
        self.t_ox = 2e-6  # thickness of oxide (m)
        self.t_gold = 500e-9  # depends on exact process (m)
        self.eps0 = 8.85e-12  # permittivity of free space
        self.E = 169e9  # Young's modulus of silicon (N/m2)
        self.v = 0.069  # Poisson's ratio (fabricated on a (100) wafer in the [110] direction)
        self.density = 2300  # density of silicon (kg/m3)
        self.density_fluid = 0.	 # density of water (kg/m3)
        self.Rs = 0.1/self.t_SOI  # sheet resistance of silicon (assuming resistivity = 10 Ohm-cm)
        self.Rs_gold = 2.4e-8/self.t_gold  # sheet resistance of gold
        self.mu = 0.  # dynamic viscosity of water
        self.mfp = 68e-9  # mean free path of air (m) <---- value for vacuum?