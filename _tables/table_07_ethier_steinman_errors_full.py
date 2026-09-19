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
from _artifact_io import (TABLE_OUTPUT_DIR, complete_table, math, read_3d, rows_for_choice, sci)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


results = read_3d('07')
N_LIST = sorted({r.n for r in results})
RATE_CHOICES = list(dict.fromkeys((r.choice, r.s_plus) for r in results))
TARGET_TIME = 0.0625
FULL_ERROR_TABLE_FILE = TABLE_OUTPUT_DIR/'table_07_ethier_steinman_errors_full.tex'
def write_full_tables(results):
    # Omit repeated time/status columns only when every run has completed at T.
    error_lines = [
        r'\begin{table}[htbp]', r'', r'\centering',
        r'\caption{Complete D3N7 Ethier--Steinman error data.  Brackets give the adjacent-grid order.}',
        r'\label{tab:numerical-ethier-errors-full}', r'\begingroup',
        r'\scriptsize', r'\setlength{\tabcolsep}{2.8pt}',
        r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
        r'choice & $N$ & $E_{u_1,2}[p_1]$ & $E_{u_2,2}[p_2]$ & $E_{u_3,2}[p_3]$ & $E_{u,2}$\\',
        r'\midrule',
    ]
    for choice, _ in RATE_CHOICES:
        rows = rows_for_choice(results, choice)
        previous = None
        for row in rows:
            entries = []
            fields = row.component_error
            for field_index, value in enumerate(fields):
                if (
                    previous is None
                    or previous.status != 'completed'
                    or row.status != 'completed'
                    or not math.isfinite(value)
                ):
                    order = '--'
                else:
                    previous_value = previous.component_error[field_index]
                    order_value = math.log(previous_value/value)/math.log(
                        previous.h/row.h
                    )
                    order = f'{order_value:.3f}'
                value_text = sci(value)
                if row.status != 'completed' and value_text.endswith('$'):
                    value_text = value_text[:-1] + r'^{\dagger}$'
                entries.append(f'{value_text} [{order}]')
            error_lines.append(
                f'{choice} & {row.n} & ' + ' & '.join(entries)
                + f' & {sci(row.velocity_error)}' + r'\\'
            )
            previous = row
        error_lines.append(r'\addlinespace[2pt]')
    error_lines.extend([
        r'\bottomrule', r'\end{tabular}', r'\endgroup', r'\end{table}',
    ])
    FULL_ERROR_TABLE_FILE.write_text(
        '\n'.join(error_lines)+'\n', encoding='utf-8', newline='\n'
    )
    return FULL_ERROR_TABLE_FILE.resolve()
if __name__ == '__main__':
    print(complete_table(write_full_tables(results)))

