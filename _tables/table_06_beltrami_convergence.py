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

"""Generate this LaTeX table from experimental CSV, without running a solver."""
import _bootstrap
from _artifact_io import (TABLE_OUTPUT_DIR, complete_table, completed_rows_for_choice, read_3d, sci, slope)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


results = read_3d('06')
RATE_CHOICES = list(dict.fromkeys((r.choice, r.s_plus) for r in results))
MAIN_TABLE_FILE = TABLE_OUTPUT_DIR/'table_06_beltrami_convergence.tex'
def write_main_table(results):
    lines = [
        r'\begin{table}[htbp]', r'', r'\centering',
        r'\scriptsize', r'\setlength{\tabcolsep}{4.2pt}',
        r'\caption{D3N7 ABC/Beltrami convergence.  Slopes use all eight grids; errors',
        r'and divergence are at $N=128$.}',
        r'\label{tab:numerical-beltrami-convergence}',
        r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'choice & $s_{+}$ & $p_u$ & $E_{u,2}$ & $D_2$\\',
        r'\midrule',
    ]
    for choice, s_plus in RATE_CHOICES:
        rows = completed_rows_for_choice(results, choice)
        finest = rows[-1]
        h = [row.h for row in rows]
        velocity_slope = slope([row.velocity_error for row in rows], h)
        table_choice = 'SRT ' if choice == 'SRT' else choice
        lines.append(
            f'{table_choice} & {s_plus:.6f} & '
            f'{velocity_slope:.4f} & '
            f'{sci(finest.velocity_error)} & '
            f'{sci(finest.divergence_l2)}'
            + r'\\'
        )
    lines.extend([
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ])
    MAIN_TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MAIN_TABLE_FILE.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    return MAIN_TABLE_FILE.resolve()
if __name__ == '__main__':
    print(complete_table(write_main_table(results)))

