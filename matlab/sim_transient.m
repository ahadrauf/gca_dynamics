% process = SOI();
% GCA = gca_import_layout(fawn(), process);
% Define process
process.undercut = 0.4e-6;  % process undercut (m)
process.small_undercut = 0.2e-6;  % process undercut (m) for small gaps (<small_undercut_threshold)
process.small_undercut_threshold = 3.5e-6;  % threshold for applying small_undercut
process.t_SOI = 40e-6;  % thickness of silicon (m)
process.t_ox = 2e-6;  % thickness of oxide (m)
process.t_gold = 500e-9;  % depends on exact process (m)
process.eps0 = 8.85e-12;  % permittivity of free space
process.E = 169e9;  % Young's modulus of silicon (N/m2)
process.v = 0.069;  % Poisson's ratio (fabricated on a (100) wafer in the [110] direction)
process.density_SOI = 2300;  % density of silicon (kg/m3)
process.density_fluid = 1.1839;  %  density of air (kg/m3)
process.Rs = 0.1/process.t_SOI;  % sheet resistance of silicon (assuming resistivity = 10 Ohm-cm)
process.Rs_gold = 2.4e-8/process.t_gold;  % sheet resistance of gold
process.mu = 1.85e-5;  % dynamic viscosity of air
process.mfp = 68e-9;  % mean free path of air (m)

% fawn.gds GCA dimensions
GCA.gf = 4.83e-6 + 2*process.undercut;  % front gap (called x_0 in the paper)
GCA.gb = 7.75e-6 + 2*process.undercut;  % back gap (called x_b in the paper)
GCA.x_GCA = 3.83e-6 + 2*process.undercut;  % distance the spine has to travel before hitting the gap stop (x0 - xf in the paper)
GCA.supportW = 3e-6 - 2*process.undercut;  % width of the support beams (called w_spr in the paper)
GCA.supportL = 240.851e-6;  % length of the support beams (called L_spr in the paper) 
GCA.Nfing = 70;  % number of GCA fingers (called N in the paper)
GCA.fingerW = 5.005e-6 - 2*process.undercut;  % width of GCA fingers (called wf in the paper)
GCA.fingerL = 76.472e-6 - process.undercut;  % overlap length of the GCA fingers (called Lol in the paper)
GCA.fingerL_buffer = 10e-6;  % extra length at the base of the GCA fingers but which doesn't overlap adjacent fingers (called L - Lol in the paper)
GCA.spineW = 20e-6 - 2*process.undercut;  % width of the spine (called w_spine in the paper) 
GCA.spineL = 860e-6 - 2*process.undercut;  % length of the spine (called L_spine in the paper)
GCA.etch_hole_width = 8e-6 + 2*process.undercut;  % size of etch hole squares in the spine (not in paper, see figure below)
GCA.etch_hole_height = 8e-6 + 2*process.undercut;  % size of etch hole squares in the spine (not in paper, see figure below)
GCA.etch_hole_spacing = 6e-6 - 2*process.undercut;  % spacing between etch hole squares in the spine (not in paper, see figure below)
GCA.gapstopW = 10e-6 - 2*process.undercut;  % width of the gap stop jutting out from either side of the spine (not in paper, see figure below)
GCA.gapstopL_half = 45e-6 - process.undercut;  % length of the gap stop jutting out from either side of the spine (not in paper, see figure below). To avoid double-counting area, this is only the length from the side of the spine to the end of the gapstop.
[GCA, process] = update_dependent_variables(GCA, process);

function [GCA, process] = update_dependent_variables(GCA, process)
process.Estar = process.E / (1 - process.v^2);
GCA.k_support = 2 * process.Estar * (GCA.supportW**3) * process.t_SOI / (GCA.supportL**3);
GCA.gs = GCA.gf - GCA.x_GCA
GCA.fingerL_total = GCA.fingerL + GCA.fingerL_buffer
GCA.num_etch_holes = round((GCA.spineL - GCA.etch_hole_spacing - process.undercut) /
                            (GCA.etch_hole_spacing + GCA.etch_hole_width))
GCA.mainspineA = GCA.spineW * GCA.spineL - GCA.num_etch_holes * (
GCA.etch_hole_width * GCA.etch_hole_height)
if hasattr(self, "fingerW"):
    GCA.spineA = GCA.mainspineA + GCA.Nfing * GCA.fingerL_total * GCA.fingerW + \
                  2 * GCA.gapstopW * GCA.gapstopL_half
elif hasattr(self, "fingerWtip"):
    GCA.spineA = GCA.mainspineA + GCA.Nfing * GCA.fingerL_total * \
                  (GCA.fingerWtip + GCA.fingerWbase) / 2 + 2 * GCA.gapstopW * GCA.gapstopL_half
if hasattr(self, "pawlW"):  # GCA includes arm (attached to inchworm motor)
    GCA.spineA += GCA.pawlW * GCA.pawlL

m = GCA.spineA * process.t_SOI * process.density_SOI
m_spring = 2 * GCA.supportW * GCA.supportL * process.t_SOI * process.density_SOI
m_eff = m + m_spring / 3
GCA.m_total = m_eff + GCA.Nfing * process.density_fluid * (GCA.fingerL**2) * (process.t_SOI**2) / (
        2 * (GCA.fingerL + process.t_SOI))
end