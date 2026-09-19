# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of the TRT-VLBM experiment reproduction code.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (see LICENSE). If not, see
# <https://www.gnu.org/licenses/>.

"""Shared dimensions, typography and colors for the experiment plots."""
import matplotlib

TEXTWIDTH_IN = 5.125
TEXTWIDTH_BP = 72 * TEXTWIDTH_IN
MIN_LINEWIDTH_PT = 1.0
DATA_LINEWIDTH_PT = 1.4
FONT_SIZE_PT = 10.0
RASTER_DPI = 600

# Ten-point nominal text also keeps ordinary math subscripts at seven points.
STYLE = {
    'font.family': 'serif', 'mathtext.fontset': 'stix', 'text.usetex': False,
    'font.size': FONT_SIZE_PT, 'axes.labelsize': FONT_SIZE_PT,
    'axes.titlesize': FONT_SIZE_PT, 'xtick.labelsize': FONT_SIZE_PT,
    'ytick.labelsize': FONT_SIZE_PT, 'legend.fontsize': FONT_SIZE_PT,
    'figure.labelsize': FONT_SIZE_PT, 'figure.titlesize': FONT_SIZE_PT,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'lines.linewidth': DATA_LINEWIDTH_PT, 'lines.markersize': 4,
    'lines.markeredgewidth': MIN_LINEWIDTH_PT,
    'axes.linewidth': MIN_LINEWIDTH_PT, 'patch.linewidth': MIN_LINEWIDTH_PT,
    'grid.linewidth': MIN_LINEWIDTH_PT, 'hatch.linewidth': MIN_LINEWIDTH_PT,
    'xtick.major.width': MIN_LINEWIDTH_PT, 'xtick.minor.width': MIN_LINEWIDTH_PT,
    'ytick.major.width': MIN_LINEWIDTH_PT, 'ytick.minor.width': MIN_LINEWIDTH_PT,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'savefig.dpi': RASTER_DPI, 'savefig.bbox': None, 'savefig.pad_inches': 0,
}


def apply_style():
    matplotlib.rcParams.update(STYLE)


