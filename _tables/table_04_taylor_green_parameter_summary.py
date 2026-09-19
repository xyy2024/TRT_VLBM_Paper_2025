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
from _artifact_io import (Fraction, TABLE_OUTPUT_DIR, complete_table, np, read_scan)
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = TABLE_OUTPUT_DIR


results = read_scan()
A_VALUES = [(a, str(a)) for a in sorted({row.a for row in results})]
ALPHA = Fraction(1, 5)
NU_FRACTION = Fraction(1, 10)
FIXED_RATE_CASES = [(None, min(r.s_plus for r in results if r.rate_choice == 'interior'))]
def s_minus_for(a):
    return Fraction(2)/(1+NU_FRACTION*ALPHA/a)

SUMMARY_TABLE_FILE_NAME = 'table_04_taylor_green_parameter_summary.tex'
def is_regular(grid_result):
    '''Return whether one target-time result is finite and non-outlying.'''
    return (
        grid_result.completed
        and not grid_result.outlier
        and grid_result.error is not None
        and grid_result.d2 is not None
    )


def joint_status(row):
    '''Classify the two-grid finite-time outcome for the domain map.'''
    if not row.coarse.completed or not row.fine.completed:
        return 0
    if row.coarse.outlier or row.fine.outlier:
        return 1
    return 2


def observed_order(row):
    if not is_regular(row.coarse) or not is_regular(row.fine):
        return None
    return float(
        np.log(row.coarse.error/row.fine.error)
        / np.log(row.coarse.h/row.fine.h)
    )


def fixed_scan_rows(results, a):
    return tuple(sorted(
        (
            row for row in results
            if row.a == a and row.rate_choice == 'interior'
        ),
        key=lambda row: row.s_plus,
    ))


def regular_rate_intervals(rows):
    rates = [row.s_plus for row in rows if joint_status(row) == 2]
    if not rates:
        return r'\textemdash'
    groups = [[rates[0]]]
    for rate in rates[1:]:
        if rate-groups[-1][-1] == Fraction(1, 20):
            groups[-1].append(rate)
        else:
            groups.append([rate])
    labels = []
    for group in groups:
        first = f'{float(group[0]):.2f}'
        last = f'{float(group[-1]):.2f}'
        labels.append(first if first == last else f'{first}--{last}')
    return ', '.join(labels)


def scientific_latex(value, *, bold=False):
    mantissa, exponent = f'{value:.3e}'.split('e')
    body = rf'{mantissa}\times10^{{{int(exponent)}}}'
    return rf'$\mathbf{{{body}}}$' if bold else rf'${body}$'


def ratio_text(value):
    '''Format ratios with three useful significant digits.'''
    return f'{value:.1f}' if value >= 10 else f'{value:.2f}'
def make_summary_table(results):
    uniform_regular = [
        row for row in results
        if row.rate_choice == 'interior' and joint_status(row) == 2
    ]
    global_best = min(uniform_regular, key=lambda row: row.fine.error)
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3.0pt}',
        r'\caption{Summary of the uniform two-parameter scan.  The interval column lists sampled rates that are regular on both grids; the best sampled $s_{+}$ has the smallest regular $N=128$ velocity error in that row.  A dagger marks a minimum at the lower scan boundary, and boldface identifies the global sampled minimum.  The ratios $R_{\rm OTRT}=E_{\rm OTRT}^{128}/E_{\min}^{128}$ and $R_{\rm SRT}=E_{\rm SRT}^{128}/E_{\min}^{128}$ compare the exact reference runs with that sampled minimum.}',
        r'\label{tab:numerical-tg-parameter-summary}',
        r'\begin{tabular}{@{}rrlrrrrr@{}}',
        r'\toprule',
        r'$a$ & $s_-$ & jointly regular $s_{+}$ & best sampled $s_{+}$ & $E_{u,2}^{128,\min}$ & $p_{96\to128}$ & $R_{\rm OTRT}$ & $R_{\rm SRT}$\\',
        r'\midrule',
    ]
    for a, _ in A_VALUES:
        scan_rows = fixed_scan_rows(results, a)
        regular_rows = [row for row in scan_rows if joint_status(row) == 2]
        if not regular_rows:
            raise RuntimeError(f'No jointly regular scan result for a={a}.')
        best = min(regular_rows, key=lambda row: row.fine.error)
        otrt = next(
            row for row in results if row.a == a and row.rate_choice == 'OTRT'
        )
        srt = next(
            row for row in results if row.a == a and row.rate_choice == 'SRT'
        )
        order = observed_order(best)
        is_global_best = best is global_best
        a_text = f'{float(a):.2f}'
        best_rate_text = f'{float(best.s_plus):.2f}'
        if best.s_plus == FIXED_RATE_CASES[0][1]:
            best_rate_text += r'$^{\dagger}$'
        if is_global_best:
            a_text = rf'\textbf{{{a_text}}}'
            best_rate_text = rf'\textbf{{{best_rate_text}}}'
        lines.append(
            f'{a_text} & {float(s_minus_for(a)):.3f} & '
            f'{regular_rate_intervals(scan_rows)} & '
            f'{best_rate_text} & '
            f'{scientific_latex(best.fine.error, bold=is_global_best)} & '
            f'{order:.3f} & '
            f'{ratio_text(otrt.fine.error/best.fine.error)} & '
            f'{ratio_text(srt.fine.error/best.fine.error)}' + r'\\'
        )
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR/SUMMARY_TABLE_FILE_NAME
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path.resolve()
if __name__ == '__main__':
    print(complete_table(make_summary_table(results)))

