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

"""Publication-size Taylor--Green plots from grouped experimental CSV records.

The artifact reader supplies the records; this module performs no numerical
integration and does not read printed tables or previous figure files.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import NullFormatter
from _figure_export import save_figure
from _figure_layout import legends_below_xlabels

COLORS = ('#0072b2', '#d55e00', '#009e73', '#cc79a7')
from _figure_style import STYLE, TEXTWIDTH_IN
WIDTH = TEXTWIDTH_IN


def regular(grid):
    return grid.completed and not grid.outlier


def groups(results):
    return [(a, sorted([r for r in results if r.a == a and
                       r.rate_choice == 'interior'], key=lambda r: r.s_plus))
            for a in sorted({r.a for r in results})]


def save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the designed physical width; no tight-bbox rescaling.
    save_figure(fig, output)
    plt.close(fig)
    return output.resolve()


def parameter_domain(results, output):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 3.2))
        ax = fig.add_axes((.145, .34, .34, .55))
        status_ax = fig.add_axes((.73, .34, .235, .55))
        marks = ('^', '>', '<', 'v')
        for index, (a, rows) in enumerate(groups(results)):
            x = np.array([float(r.s_plus) for r in rows])
            y = [r.fine.error if regular(r.fine) else np.nan for r in rows]
            ax.semilogy(x, y, color=COLORS[index], marker=marks[index],
                        markevery=[0, 1, *range(4, len(rows), 4), len(rows)-1],
                        markersize=3.5, markerfacecolor='white', linewidth=1.2,
                        label=rf'$a={float(a):.2f}$')
            for method, marker in [('OTRT', 'P'), ('SRT', 'X')]:
                r = next(r for r in results if r.a == a and r.rate_choice == method)
                ax.scatter(float(r.s_plus), r.fine.error, marker=marker, s=44,
                           facecolors='white', edgecolors=COLORS[index],
                           linewidths=1.25, zorder=5)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color='black', marker=m, linestyle='',
                           markerfacecolor='white', markersize=6, label=s)
                    for s, m in [('OTRT', 'P'), ('SRT', 'X')]]
        error_legend = fig.legend(handles=handles, loc='upper center',
                  ncol=2, frameon=False, borderaxespad=0,
                  columnspacing=.6, handlelength=1.25, fontsize=10)
        ax.set(xlim=(0.025, 2.025), ylim=(9e-6, 4.8e-4), xlabel=r'$s_+$',
               ylabel=r'$E_{u,2}$', title=r'(a) Error on $N=128$')
        ax.set_xticks([.05, 1, 2])
        ax.tick_params(pad=2)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.25, linewidth=1)
        status, rowlabels = [], []
        for a, rows in groups(results):
            for attribute, n in [('coarse', 96), ('fine', 128)]:
                status.append([0 if not getattr(r, attribute).completed else
                               1 if getattr(r, attribute).outlier else 2
                               for r in rows if float(r.s_plus) >= 1.9])
                rowlabels.append(rf'$a={float(a):.2f},\ N={n}$')
        status = np.array(status)
        assert status.shape == (8, 3)
        cmap = ListedColormap(('#8b1a1a', '#e69f00', '#dceff4'))
        status_ax.imshow(status, cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5],3),
                         aspect='auto', interpolation='nearest')
        for i in range(8):
            for j in range(3):
                if status[i, j] < 2:
                    status_ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                        facecolor='none', edgecolor='black', linewidth=1,
                        hatch='xxx' if status[i,j] == 0 else '///'))
        status_ax.set_yticks(range(8), rowlabels)
        status_ax.set_xticks(range(3), ['1.90','1.95','2.00'])
        for y in [.5, 2.5, 4.5, 6.5]:
            status_ax.axhline(y, color='.65', linewidth=1)
        for y in [1.5, 3.5, 5.5]:
            status_ax.axhline(y, color='white', linewidth=2)
        status_ax.set(xlabel=r'$s_+$', title='(b) Outcomes')
        status_ax.tick_params(pad=2)
        handles = [Patch(facecolor=c, edgecolor='black', hatch=h, label=l)
                   for c, h, l in [('#dceff4','','regular'),
                                   ('#e69f00','///','finite outlier'),
                                   ('#8b1a1a','xxx','no finite result')]]
        status_legend = fig.legend(handles=handles, loc='upper center',
                         ncol=1, frameon=False, borderaxespad=0, handlelength=1.25,
                         fontsize=10)
        legends_below_xlabels(fig, [([ax], error_legend), ([status_ax], status_legend)])
        return save(fig, output)


