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
import matplotlib.pyplot as plt
from _artifact_io import (DATA_CSV_DIR, FIGURE_OUTPUT_DIR, np, read_records)
from _figure_export import save_figure
from _figure_style import STYLE, TEXTWIDTH_IN
from _figure_layout import legend_below_xlabels
from _plottools import ARTICLE_SERIES_COLORS
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PREFIX = '01_periodic_acoustic'
N_LIST = (128,)
RATES = ('otrt', 'srt')
GAIN_LIMIT = 1000.0
TARGET_STEPS = 800
def plot_histories(histories, spectra):
    """Plot the finest-grid histories; tables retain all eight grids.

    Every saved time step is shown. Open circles are sparse samples of the
    linear prediction, not a resampling or smoothing of the numerical data.
    """
    with plt.rc_context(STYLE):
        _plot_finest_histories(histories, spectra)


def _plot_finest_histories(histories, spectra):
    n_plot = max(N_LIST)
    large = small = 10
    figure_height, plot_height, plot_bottom = 2.6, 1.35, .83
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, figure_height), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=plot_bottom/figure_height,
                        top=(plot_bottom+plot_height)/figure_height, wspace=0.42)
    styles = {'otrt': (ARTICLE_SERIES_COLORS[0], '-'),
              'srt': (ARTICLE_SERIES_COLORS[1], '--')}
    terminal = {}
    for choice in RATES:
        rows = [row for row in histories
                if row['choice'] == choice and int(row['n']) == n_plot]
        rows.sort(key=lambda row: int(row['step']))
        steps = np.array([int(row['step']) for row in rows])
        gains = np.array([float(row['modal_gain']) for row in rows])
        divergence = np.array([float(row['divergence_l2']) for row in rows])
        assert np.array_equal(steps, np.arange(steps[-1]+1))
        assert np.all(gains > 0) and np.all(divergence[1:] > 0)
        color, linestyle = styles[choice]
        axes[0].semilogy(steps, gains, color=color, linestyle=linestyle,
                         linewidth=1.6, label=choice.upper())
        # The zero initial divergence is omitted only on the logarithmic axis.
        axes[1].semilogy(steps[1:], divergence[1:], color=color,
                         linestyle=linestyle, linewidth=1)
        reference_steps = np.unique(np.r_[np.arange(0, steps[-1]+1, 80), steps[-1]])
        axes[0].plot(reference_steps, abs(spectra[choice]['value'])**reference_steps,
                     color='black', marker='o', linestyle='none', markersize=3.6,
                     markerfacecolor='white', markeredgewidth=1, zorder=5)
        terminal[choice] = (steps[-1], gains[-1])
    stop_step, stop_gain = terminal['srt']
    axes[0].axhline(GAIN_LIMIT, color='0.4', linewidth=1, linestyle=':')
    axes[0].text(785, GAIN_LIMIT*1.28, r'$10^3$ threshold',
                 ha='right', va='bottom', fontsize=small, color='0.2')
    axes[0].annotate(f'$n={stop_step}$', xy=(stop_step, stop_gain),
                     xytext=(500, 35), fontsize=small,
                     arrowprops=dict(arrowstyle='->', color='0.25', linewidth=1))
    axes[0].text(80, 140, 'SRT', fontsize=small)
    axes[0].text(270, 0.006, 'OTRT', fontsize=small)
    # Physical offsets remain exact because the PDF is included at scale=1.
    axes[1].annotate('SRT', xy=(250, 0.2), xytext=(-2, -9),
                     textcoords='offset points', fontsize=small)
    axes[1].annotate('OTRT', xy=(610, 2e-5), xytext=(-2, -10),
                     textcoords='offset points', fontsize=small)
    axes[0].set_ylabel(r'$\mathcal{A}_n$', fontsize=large)
    axes[1].set_ylabel(r'$D_2$', fontsize=large)
    axes[0].set_title('(a) Eigencomponent gain', fontsize=large)
    axes[1].set_title('(b) Velocity divergence', fontsize=large)
    for ax in axes:
        ax.tick_params(labelsize=small, width=1, which='both')
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        ax.set_xlim(0, TARGET_STEPS)
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_xlabel('Time step $n$', fontsize=large)
        ax.minorticks_off()
        ax.grid(True, which='major', linewidth=1, alpha=0.2)
    axes[0].set_ylim(1e-3, 1e4)
    axes[0].set_yticks([1e-3, 1e-1, 1e1, 1e3])
    axes[1].set_ylim(3e-9, 1)
    axes[1].set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    # Keep the last tick label within the fixed figure canvas.
    axes[1].get_xticklabels()[-1].set_horizontalalignment('right')
    handles = [plt.Line2D([], [], color=styles[choice][0], linestyle=styles[choice][1],
                          linewidth=1.6, label=choice.upper()) for choice in RATES]
    handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='none',
                              markerfacecolor='white', markeredgewidth=1, markersize=3.6,
                              label=r'Linear prediction $|z_*|^n$'))
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.005),
               ncol=3, fontsize=small, frameon=False, columnspacing=1.4, handlelength=2)
    legend_below_xlabels(fig, axes, legend, center_on_axes=True)
    path = FIGURE_OUTPUT_DIR/f'figure_{PREFIX}_histories.pdf'
    from _figure_export import save_figure
    save_figure(fig, path)
    plt.close(fig)
if __name__ == '__main__':
    histories = read_records(DATA_CSV_DIR/'data_01_periodic_acoustic_histories.csv')
    spectra = {r['choice']: {'value': r['modulus']}
               for r in read_records(DATA_CSV_DIR/'data_01_periodic_acoustic_modes.csv')}
    plot_histories(histories, spectra)
