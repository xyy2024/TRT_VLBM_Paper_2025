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

"""Plot all eight final velocity fields of the supplementary refinement from CSV."""
import _bootstrap
import matplotlib.pyplot as plt
import numpy as np

from _artifact_io import DATA_CSV_DIR, FIGURE_OUTPUT_DIR, read_records
from _figure_export import save_figure
from _figure_style import STYLE, TEXTWIDTH_IN, MIN_LINEWIDTH_PT
from _figure_layout import align_colorbar_height, colorbar_header
from _plottools import ARTICLE_SEQUENTIAL_CMAP


def main():
    rows = read_records(DATA_CSV_DIR/'data_09_kolmogorov_startup_fields.csv')
    grids = sorted({r['n'] for r in rows})
    fields = {}
    for method in ('TRT', 'SRT'):
        for n in grids:
            records = [r for r in rows if r['n'] == n and r['method'] == method]
            if len(records) != n*n or len({(r['i'],r['j']) for r in records}) != n*n:
                raise ValueError(f'Incomplete field: {method}, N={n}.')
            values = {key:np.empty((n,n)) for key in ('x','y','u1','u2')}
            for r in records:
                for key in values:
                    values[key][r['i'], r['j']] = r[key]
            fields[method,n] = values
    vmax = max(np.hypot(f['u1'], f['u2']).max() for f in fields.values())
    with plt.rc_context(STYLE):
        figure_height = 2.75
        panel_size = 0.85
        fig = plt.figure(figsize=(TEXTWIDTH_IN, figure_height))
        axes = np.empty((2, 4), dtype=object)
        for row, method in enumerate(('TRT', 'SRT')):
            bottom = (1.62, 0.38)[row]
            fig.text(0.08 / TEXTWIDTH_IN, (bottom + panel_size / 2) / figure_height,
                     method, rotation=90, ha='center', va='center')
            for col, n in enumerate(grids):
                ax = fig.add_axes([(0.60 + 1.01 * col) / TEXTWIDTH_IN,
                                   bottom / figure_height,
                                   panel_size / TEXTWIDTH_IN,
                                   panel_size / figure_height])
                axes[row, col] = ax
                f = fields[method,n]
                edges = np.linspace(0, 1, n+1)
                plot = ax.pcolormesh(edges, edges, np.hypot(f['u1'],f['u2']).T,
                                     vmin=0, vmax=vmax, cmap=ARTICLE_SEQUENTIAL_CMAP,
                                     shading='flat', rasterized=True)
                stride = max(1, n//6)
                selection = np.s_[::stride,::stride]
                ax.quiver(f['x'][selection], f['y'][selection],
                          f['u1'][selection], f['u2'][selection],
                          color='white', scale=22, scale_units='width', pivot='mid',
                          units='inches', width=MIN_LINEWIDTH_PT/72)
                if row == 0:
                    ax.set_title(rf'$N={n}$', pad=3)
                ax.set(xlim=(0,1), ylim=(0,1), aspect='equal')
                ax.set_xticks((0,0.5,1))
                ax.set_yticks((0,0.5,1))
                ax.set_xticklabels(['0', '0.5', '1'])
                ax.set_yticklabels(['0', '0.5', '1'])
                if col > 0:
                    ax.tick_params(labelleft=False)
                ax.tick_params(pad=2)
                if row == 0:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(r'$x$', labelpad=1)
                if col == 0:
                    ax.set_ylabel(r'$y$', labelpad=1)
        bar = fig.colorbar(plot, cax=fig.add_axes(
            (4.68 / TEXTWIDTH_IN, 0.38 / figure_height,
             0.055 / TEXTWIDTH_IN, 2.09 / figure_height)))
        align_colorbar_height(fig, bar, axes.flat)
        colorbar_header(fig, bar, r'$|u_h|$')
        save_figure(fig, FIGURE_OUTPUT_DIR/'figure_09_kolmogorov_startup_fields.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
