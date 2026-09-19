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

"""CSV-only statistics and tables for periodic velocity convergence."""
from __future__ import annotations
import math
import numpy as np
from _artifact_io import DATA_CSV_DIR, TABLE_OUTPUT_DIR, read_records
from _periodic_config import N_LIST as GRIDS

PREFIXES = ('02_nonlinear_manufactured', '03_taylor_green')
PROBLEM_NAMES = dict(zip(PREFIXES, ('Manufactured', 'Taylor--Green')))
CONFIGURATIONS = (('SRT', 0.2), ('OTRT', 0.2), ('SRT', 0.5), ('OTRT', 0.5))


def time_value(rows):
    if not rows or any('target_time' not in r for r in rows):
        raise ValueError('Regenerate this dataset with experiment 02 or 03.')
    values = {r['target_time'] for r in rows}
    if len(values) != 1:
        raise ValueError('A periodic dataset must have one common target time.')
    return values.pop()


def read_convergence(prefix):
    rows = read_records(DATA_CSV_DIR / f'data_{prefix}_results.csv')
    expected = {(m,a,n) for m,a in CONFIGURATIONS for n in GRIDS}
    keys = [(r['method'],r['alpha'],r['n']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'{prefix}: incomplete or duplicate convergence results.')
    target = time_value(rows)
    for row in rows:
        if row['status']=='completed' and not math.isclose(row['actual_time'],target,rel_tol=0,abs_tol=1e-14):
            raise ValueError('A completed convergence trajectory has the wrong endpoint.')
        if row['status']!='completed' and row['actual_time'] > target+1e-14:
            raise ValueError('An interrupted trajectory is later than the target time.')
    return rows


def common_convergence():
    datasets = {prefix:read_convergence(prefix) for prefix in PREFIXES}
    if len({time_value(rows) for rows in datasets.values()}) != 1:
        raise ValueError('The two periodic convergence datasets have different target times.')
    return datasets


def groups(rows):
    return [(method, alpha, sorted((r for r in rows
             if r['method'] == method and r['alpha'] == alpha), key=lambda r: r['n']))
            for method, alpha in CONFIGURATIONS]


def fit(rows, field='e_u2'):
    if any(r['status'] != 'completed' or r.get(field) is None for r in rows):
        return None
    return float(np.polyfit(np.log([r['h'] for r in rows]),
                            np.log([r[field] for r in rows]), 1)[0])


def adjacent(previous, row, field):
    if previous is None or any(r['status'] != 'completed' or r.get(field) is None
                               for r in (previous, row)):
        return None
    return math.log(previous[field] / row[field]) / math.log(previous['h'] / row['h'])


def number(value, digits=3):
    return '--' if value is None or not math.isfinite(value) else f'{value:.{digits}f}'


def sci(value):
    if value is None or not math.isfinite(value):
        return '--'
    mantissa, exponent = f'{value:.3e}'.split('e')
    return '$' + mantissa + r'\mathrm{e}{' + str(int(exponent)) + '}$'


def error_order(previous, row, field):
    if row[field] is None:
        return '--'
    return sci(row[field]) + ' [' + number(adjacent(previous, row, field)) + ']'


def write_table(stem, caption, label, columns, header, body, *, column_sep=2.1):
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = (r'% Generated from unrounded experimental CSV.' + '\n'
            r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n'
            + r'\caption{' + caption + '}\n' + r'\label{' + label + '}\n'
            + r'\begingroup\scriptsize\setlength{\tabcolsep}{' + str(column_sep) + 'pt}' + '\n'
            + r'\begin{tabular}{@{}' + columns + r'@{}}\toprule' + '\n'
            + header + r'\\' + '\n' + r'\midrule' + '\n' + '\n'.join(body)
            + '\n' + r'\bottomrule\end{tabular}\endgroup\end{table}' + '\n')
    path = TABLE_OUTPUT_DIR / (stem + '.tex')
    path.write_text(text, encoding='utf-8')
    return path


def convergence_table(prefix):
    rows = read_convergence(prefix)
    manufactured = prefix.startswith('02')
    fields = ('e_u1', 'e_u2', 'e_uinf') if manufactured else ('e_u2',)
    body = []
    for method, alpha, selected in groups(rows):
        if not any(row['status'] == 'completed' for row in selected):
            continue
        if body:
            body.append(r'\midrule')
        previous = None
        for row in selected:
            if row['status'] != 'completed':
                previous = row
                continue
            entries = [method, f'{alpha:g}', str(row['n'])]
            entries += [error_order(previous, row, field) for field in fields]
            entries += [number(None if row['e_u2'] is None else row['e_u2']/row['h']**2),
                        sci(row['d2']), sci(row['density_fluctuation']), sci(row['m_h'])]
            body.append(' & '.join(entries) + r'\\')
            previous = row
    name = 'nonlinear_manufactured' if manufactured else 'taylor_green'
    label = 'tab:sm-mms-convergence-full' if manufactured else 'tab:sm-tg-convergence-full'
    error_headers = (r'$E_{u,1}[p_1]$ & $E_{u,2}[p_2]$ & $E_{u,\infty}[p_\infty]$'
                     if manufactured else r'$E_{u,2}[p_u]$')
    caption = (PROBLEM_NAMES[prefix] + r' data for completed runs at $\nu=0.1$, $a=0.2$ and $T='
        + f'{time_value(rows):g}' + r'$. '
        r'Brackets contain adjacent orders between completed neighboring grids. '
        r'Early stopping times and diagnostics are listed in '
        r'Table~\ref{tab:sm-periodic-stopping}.')
    return write_table(f'table_{prefix[:2]}_{name}_convergence_full', caption, label,
        'lrr' + 'r' * (len(fields) + 4),
        r'method & $\alpha$ & $N$ & ' + error_headers
        + r' & $E_{u,2}/h^2$ & $D_2$ & $\delta\rho_\infty$ & $M_h$', body,
        column_sep=1.05 if manufactured else 2.1)


def accuracy_summary():
    """Report refinement errors and observed rates, with no completion column."""
    body = []
    datasets = common_convergence()
    error_grids = (64, 128, 256)
    if not set(error_grids).issubset(GRIDS):
        raise ValueError('The main accuracy table requires N=64,128,256.')

    def error_entry(row):
        if row['status'] != 'completed' or row['e_u2'] is None:
            return '--'
        mantissa, exponent = f"{row['e_u2']:.3e}".split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'

    for prefix in PREFIXES:
        rows = datasets[prefix]
        for method, alpha, selected in groups(rows):
            by_grid = {r['n']: r for r in selected}
            complete = all(r['status'] == 'completed' for r in selected)
            local_rates = [adjacent(p, q, 'e_u2')
                           for p, q in zip(selected[:-1], selected[1:])]
            rate_range = (rf'$[{min(local_rates):.3f},\,{max(local_rates):.3f}]$'
                          if complete else '--')
            scheme = method + (r'$^\dagger$' if not complete else '')
            entries = [PROBLEM_NAMES[prefix], scheme, f'{alpha:g}']
            entries += [error_entry(by_grid[n]) for n in error_grids]
            entries += [number(fit(selected)),
                        number(adjacent(selected[-2], selected[-1], 'e_u2')),
                        rate_range]
            body.append(' & '.join(entries) + r'\\')
        body.append(r'\midrule')
    body.pop()
    return write_table('table_03_periodic_accuracy_summary',
        r'Velocity errors and observed convergence rates at $T='
        + f'{time_value(rows):g}' + r'$, $\nu=0.1$ and $a=0.2$. '
        r'$p_u^{\rm LS}$ is the least-squares rate over all nine meshes; '
        r'$[p_{\min},p_{\max}]$ is the range of the eight successive-mesh rates. '
        r'A dagger marks a refinement sequence interrupted by the stopping '
        r'criterion before $T$; dashes denote unavailable errors or rates.',
        'tab:numerical-periodic-summary', 'llrrrrrrr',
        r' & & & \multicolumn{3}{c}{$L^2$ velocity error $E_{u,2}$}'
        r' & \multicolumn{3}{c}{Observed convergence rates}\\' + '\n'
        r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}' + '\n'
        r'problem & scheme & $\alpha$ & $N=64$ & $N=128$ & $N=256$'
        r' & $p_u^{\rm LS}$ & $p_{192\to256}$ & $[p_{\min},p_{\max}]$',
        body)


def stopping_table():
    body = []
    datasets = common_convergence()
    for prefix in PREFIXES:
        rows = datasets[prefix]
        for row in sorted(rows, key=lambda r: (r['n'], r['alpha'], r['method'])):
            if row['status'] == 'completed':
                continue
            body.append(' & '.join([PROBLEM_NAMES[prefix], row['method'], f'{row["alpha"]:g}',
                str(row['n']), str(row['steps']), f'{row["actual_time"]:.8f}',
                sci(row['d2']), sci(row['density_fluctuation']), sci(row['m_h'])]) + r'\\')
    return write_table('table_03_periodic_stopping',
        r'Periodic configurations that stop before $T=' + f'{time_value(rows):g}'
        + r'$. All remaining configurations '
        r'complete. Listed convergence diagnostics are at the first crossing. A diagnostic crossing is not '
        r'a claim of blow-up of the continuum solution.',
        'tab:sm-periodic-stopping', 'llrrrrrrr',
        r'problem & method & $\alpha$ & $N$ & step & $t_{\rm stop}$ & $D_2$ & $\delta\rho_\infty$ & $M_h$',
        body)


