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

"""Shared manuscript layouts for figures generated from experimental CSV data."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from _figure_export import save_figure
from _figure_layout import legend_right_of_axes

from _figure_style import STYLE, TEXTWIDTH_IN


def slope(h, error):
    return float(np.polyfit(np.log(h), np.log(error), 1)[0])


def ethier_convergence(rows, output):
    """Main Fig. 4: compact axes with a separate right-side legend."""
    with plt.rc_context({**matplotlib.rcParamsDefault, **STYLE}):
        figure_height, plot_height = 2.9, 1.82
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, figure_height))
        ax.set_position([.20, .28, 2.35/TEXTWIDTH_IN, plot_height/figure_height])
        slopes = {}
        for choice, color, marker, linestyle in [('OTRT', '#0072B2', 'o', '-'),
                                                 ('SRT', '#D55E00', '^', '--')]:
            selected = sorted([r for r in rows if r['choice'] == choice], key=lambda r: r['n'])
            h = np.array([r['h'] for r in selected])
            error = np.array([r['velocity_error'] for r in selected])
            slopes[choice] = slope(h, error)
            ax.loglog(h, error, linestyle=linestyle, color=color, linewidth=1.8,
                      marker=marker, markersize=6,
                      markerfacecolor=color if choice == 'OTRT' else 'white',
                      markeredgewidth=1.4, label=f'{choice} (LS {slopes[choice]:.3f})')
        h = np.array([1/128, 1/64, 1/32, 1/16])
        reference = .5*min(r['velocity_error'] for r in rows)*(h/(1/128))**2
        ax.loglog(h, reference, ':', color='0.4', linewidth=1.5, label=r'$O(h^2)$')
        ax.set_xticks(h, [r'$1/128$', r'$1/64$', r'$1/32$', r'$1/16$'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(xlabel=r'$h$', ylabel=r'$E_{u,2}$', title='Velocity convergence')
        ax.grid(True, which='major', alpha=.25, linewidth=1)
        legend = fig.legend(*ax.get_legend_handles_labels(), loc='center left',
                            ncol=1, frameon=False, handlelength=1.8, handletextpad=.6)
        legend_right_of_axes(fig, ax, legend)
        save_figure(fig, output)
        plt.close(fig)
    return slopes


