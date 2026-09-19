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

"""Render periodic velocity convergence from CSV only."""
import _bootstrap
import numpy as np
import matplotlib.pyplot as plt
from _artifact_io import FIGURE_OUTPUT_DIR
from _periodic_presentation import common_convergence, groups
from _figure_style import STYLE, TEXTWIDTH_IN
from _figure_export import save_figure
from _figure_layout import legend_below_xlabels


def main():
    data = common_convergence()
    mms, tg = data['02_nonlinear_manufactured'], data['03_taylor_green']
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.88))
        colors = ('#D55E00', '#0072B2', '#CC79A7', '#009E73')
        markers = ('^', 'o', 'v', 's')
        for ax, rows, title in zip(axes, (mms, tg),
                ('(a) Manufactured velocity', '(b) Taylor--Green velocity')):
            finite_values = []
            for (method, alpha, selected), color, marker in zip(groups(rows), colors, markers):
                errors = [r['e_u2'] if r['status'] == 'completed' else np.nan for r in selected]
                finite_values.extend(value for value in errors if np.isfinite(value))
                ax.loglog([r['h'] for r in selected], errors, color=color, marker=marker,
                          linestyle='-' if method == 'OTRT' else '--', markersize=4,
                          markerfacecolor=color if method == 'OTRT' else 'white',
                          label=method + r', $\alpha=' + f'{alpha:g}' + '$')
            grids = sorted({r['n'] for r in rows})
            h = np.array([1/grids[-1], 1/grids[0]])
            ax.loglog(h, min(finite_values)*0.45*(h/h.min())**2, ':', color='0.45',
                      linewidth=1, label=r'$O(h^2)$')
            ax.set(title=title, xlabel='$h$', ylabel='$E_{u,2}$')
            tick_grids = (grids[-1], grids[len(grids)//2], grids[0])
            ax.set_xticks([1/n for n in tick_grids], [f'1/{n}' for n in tick_grids])
            ax.tick_params(axis='x', which='minor', labelbottom=False)
            ax.grid(alpha=0.2)
        legend = fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0.20, 1, 1), w_pad=1.3)
        legend_below_xlabels(fig, axes, legend, center_on_axes=True)
        output = FIGURE_OUTPUT_DIR / 'figure_03_taylor_green_periodic_convergence.pdf'
        output.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, output)
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
