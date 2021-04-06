class SOI:
    def __init__(self):
        self.overetch = 0.48e-6  # process overetch (m)
        self.small_overetch = 0.19e-6  # process overetch (m) for small gaps (<4 um)
        self.small_overetch_threshold = 3.5e-6  # threshold for applying small_overetch
        self.t_SOI = 40e-6  # thickness of silicon (m)
        self.t_ox = 2e-6  # thickness of oxide (m)
        self.t_gold = 500e-9  # depends on exact process (m)
        self.eps0 = 8.85e-12  # permittivity of free space
        self.E = 170e9  # Young's modulus of silicon (N/m2)
        self.density = 2300  # density of silicon (kg/m3)
        self.density_fluid = 1.1839	 # density of air (kg/m3)
        self.mu = 1.85e-5  # dynamic viscosity of air
        self.mfp = 68e-9  # mean free path of air (m)

class SOIunderwater:
    def __init__(self):
        self.overetch = 0.48e-6  # process overetch (m)
        self.small_overetch = 0.19e-6  # process overetch (m) for small gaps (<4 um)
        self.small_overetch_threshold = 3.5e-6  # threshold for applying small_overetch
        self.t_SOI = 40e-6  # thickness of silicon (m)
        self.t_ox = 2e-6  # thickness of oxide (m)
        self.t_gold = 500e-9  # depends on exact process (m)
        self.eps0 = 8.85e-12  # permittivity of free space
        self.E = 170e9  # Young's modulus of silicon (N/m2)
        self.density = 2300  # density of silicon (kg/m3)
        self.density_fluid = 1000	 # density of water (kg/m3)
        self.mu = 8.9e-4  # dynamic viscosity of water
        self.mfp = 68e-9  # mean free path of air (m) <---- value for water?
