function [process] = SOI()
process.undercut = 0.4e-6;  % process undercut (m)
process.small_undercut = 0.2e-6;  % process undercut (m) for small gaps (<small_undercut_threshold)
process.small_undercut_threshold = 3.5e-6;  % threshold for applying small_undercut
process.t_SOI = 40e-6;  % thickness of silicon (m)
process.t_ox = 2e-6;  % thickness of oxide (m)
process.t_gold = 500e-9;  % depends on exact process (m)
process.eps0 = 8.85e-12;  % permittivity of free space
process.E = 169e9;  % Young's modulus of silicon (N/m2)
process.v = 0.069;  % Poisson's ratio (fabricated on a (100) wafer in the [110] direction)
process.density = 2300;  % density of silicon (kg/m3)
process.density_fluid = 1.1839;  %  density of air (kg/m3)
process.Rs = 0.1/process.t_SOI;  % sheet resistance of silicon (assuming resistivity = 10 Ohm-cm)
process.Rs_gold = 2.4e-8/process.t_gold;  % sheet resistance of gold
process.mu = 1.85e-5;  % dynamic viscosity of air
process.mfp = 68e-9;  % mean free path of air (m)
end