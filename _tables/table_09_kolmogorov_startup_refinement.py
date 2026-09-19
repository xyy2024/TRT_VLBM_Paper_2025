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

"""Generate the supplementary forced Kolmogorov refinement table from CSV."""
import _bootstrap
from _artifact_io import DATA_CSV_DIR, read_records, save_table, sci_times


def main():
    rows = read_records(DATA_CSV_DIR/'data_09_kolmogorov_startup_results.csv')
    if {(r['method'], r['n']) for r in rows} != {
            (method,n) for method in ('TRT','SRT') for n in (6,12,24,48)}:
        raise ValueError('The forced Kolmogorov comparison requires both methods on all four grids.')
    body = []
    for r in sorted(rows, key=lambda r:(r['n'],r['method'] != 'TRT')):
        values = [sci_times(r[key], 2) for key in (
            'relative_deviation','transverse_ratio','stationarity_tail_max',
            'divergence_l2','density_fluctuation','lattice_speed')]
        body.append(f"{r['n']} & {r['method']} & " + ' & '.join(values) + r'\\')
    print(save_table(
        'table_09_kolmogorov_startup_refinement.tex',
        r'Forced Kolmogorov start-up problem at $T=150$. '
        r'$B_2$ measures the relative distance to the analytical steady shear, '
        r'$Q_2$ measures transverse velocity, and $R_{[145,150]}$ is the maximum '
        r'normalized temporal residual over the final five time units. '
        r'All eight runs reach the common endpoint.',
        'tab:sm-kolmogorov-original-refinement', 'rlrrrrrr',
        r'$N$ & Method & $B_2$ & $Q_2$ & $R_{[145,150]}$ & $D_2$ & $\delta\rho_\infty$ & $M_h$\\',
        body))


if __name__ == '__main__':
    main()
