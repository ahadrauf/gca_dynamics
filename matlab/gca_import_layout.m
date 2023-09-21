function [GCA] = gca_import_layout(GCA, process)
undercut = process.undercut;
GCA.gf = GCA.gf + 2*undercut;
GCA.gb = GCA.gb + 2*undercut;
GCA.x_GCA = GCA.x_GCA + 2*undercut;
GCA.supportW = GCA.supportW - 2 * undercut;
GCA.supportL = GCA.supportL;
GCA.Nfing = GCA.Nfing;
GCA.fingerL = GCA.fingerL - undercut;
GCA.fingerL_buffer = GCA.fingerL_buffer;
GCA.spineW = GCA.spineW - 2 * undercut;
GCA.spineL = GCA.spineL - 2 * undercut;
GCA.etch_hole_spacing = GCA.etch_hole_spacing - 2 * undercut;
GCA.gapstopW = GCA.gapstopW - 2 * undercut;
GCA.gapstopL_half = GCA.gapstopL_half - undercut;
end