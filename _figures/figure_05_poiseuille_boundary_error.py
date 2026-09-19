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

"""Render this figure using CSV only; no solver, checkpoint, TeX table, or old PDF input."""
import _bootstrap
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
from _artifact_io import (DATA_CSV_DIR, FIGURE_OUTPUT_DIR, np, read_records)
import matplotlib.pyplot as plt
from _figure_export import save_figure
from _plottools import ARTICLE_DATA_LINEWIDTH, ARTICLE_MIN_SOURCE_LINEWIDTH, ARTICLE_SERIES_COLORS, article_font_sizes, set_adaptive_line_limits
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FIGURE_FILE = FIGURE_OUTPUT_DIR/'figure_05_poiseuille_boundary_error.pdf'
from _figure_style import TEXTWIDTH_IN
from _figure_layout import legend_right_of_axes
FIGURE_WIDTH_INCHES = TEXTWIDTH_IN
LARGE_FONT_SIZE, SMALL_FONT_SIZE = article_font_sizes(FIGURE_WIDTH_INCHES)
X_BLANK_FRACTIONS = Y_BLANK_FRACTIONS = (0.0454545454545, 0.0454545454545)
ELL_VALUES = (0.0,)
def make_figure(results):
    figure_height, plot_height = 2.45, 1.30
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    ell = ELL_VALUES[0]
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_title(r'Poiseuille flow, $\ell=0$', fontsize=LARGE_FONT_SIZE)
    ax.set_xlabel(r'$y$', fontsize=LARGE_FONT_SIZE)
    ax.set_ylabel(r'$(u_{h,1}-u_1)/h^2$', fontsize=LARGE_FONT_SIZE)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(-0.1, 0.1, 5))
    ax.set_yticklabels(['-0.10', '-0.05', '0.00', '0.05', '0.10'])
    ax.tick_params(axis='both', labelsize=SMALL_FONT_SIZE, rotation=0)
    ax.grid(True, color='0.88', linewidth=ARTICLE_MIN_SOURCE_LINEWIDTH)
    styles = (
        ('SRT', ARTICLE_SERIES_COLORS[1], 's', '--', 'white'),
        ('OTRT', ARTICLE_SERIES_COLORS[0], 'o', '-', ARTICLE_SERIES_COLORS[0]),
        ('tuned', ARTICLE_SERIES_COLORS[2], '^', ':', 'white'),
    )
    for choice, color, marker, linestyle, markerfacecolor in styles:
        series = [r for r in results if r['choice'] == choice and r['ell'] == ell]
        series.sort(key=lambda r: r['y'])
        ax.plot([r['y'] for r in series],
                [(r['u_numerical']-r['u_exact'])/r['h']**2 for r in series],
            color=color, marker=marker, markerfacecolor=markerfacecolor,
            markeredgecolor=color, markersize=3.5, linestyle=linestyle,
            linewidth=ARTICLE_DATA_LINEWIDTH,
            label=(choice if choice != 'tuned' else
                   rf'tuned $s_+={next(r['s_plus'] for r in results if r['choice'] == 'tuned'):.3f}$'),
        )
    set_adaptive_line_limits(
        ax,
        x_blank_fractions=X_BLANK_FRACTIONS,
        y_blank_fractions=Y_BLANK_FRACTIONS,
    )
    # Reserve room for a separate right legend and balance the outer margins.
    ax.set_position([.20, .30, 2.35/FIGURE_WIDTH_INCHES, plot_height/figure_height])
    legend = fig.legend(*ax.get_legend_handles_labels(), loc='center left',
        frameon=False, ncol=1, borderaxespad=0, handlelength=1.8, handletextpad=.6)
    legend_right_of_axes(fig, ax, legend)
    save_figure(fig, FIGURE_FILE)
    plt.close(fig)
    output = FIGURE_FILE.resolve()
    print(f'Saved Poiseuille figure: {output}')
    return output
if __name__ == '__main__':
    make_figure(read_records(DATA_CSV_DIR/'data_05_poiseuille_profiles.csv'))
