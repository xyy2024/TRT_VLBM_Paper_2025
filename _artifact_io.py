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

"""Shared, solver-free CSV readers and LaTeX output formatting."""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from _csv_io import read_records as _read_records, read_3d as _read_3d
from _paths import CODE_DIR, DATA_CSV_DIR, TABLE_OUTPUT_DIR, FIGURE_OUTPUT_DIR
from _periodic_config import N_LIST as PERIODIC_GRIDS


def read_records(path):
    """Refuse an in-progress or incomplete dataset for a final article table."""
    rows = _read_records(path)
    expected_counts = {
        'data_02_nonlinear_manufactured_results.csv': 4*len(PERIODIC_GRIDS),
        'data_03_taylor_green_results.csv': 4*len(PERIODIC_GRIDS),
        'data_04_taylor_green_parameter_scan.csv': 336,
        'data_05_poiseuille_boundary_tuning.csv': 35,
        'data_09_kolmogorov_startup_results.csv': 8,
        'data_09_kolmogorov_startup_fields.csv': 6120,
        'data_08_double_shear_results.csv': 24,
        'data_06_beltrami_results.csv': 16,
        'data_07_ethier_steinman_results.csv': 8,
        'data_01_periodic_acoustic_results.csv': 16,
        'data_01_periodic_acoustic_modes.csv': 2,
    }
    expected = expected_counts.get(Path(path).name)
    if expected is not None and len(rows) != expected:
        raise ValueError(f'{path}: expected {expected} article records, found {len(rows)}. Finish the full experiment first.')
    return rows


def read_3d(number):
    stem = 'beltrami' if number == '06' else 'ethier_steinman'
    read_records(DATA_CSV_DIR/f'data_{number}_{stem}_results.csv')
    return _read_3d(number)


def read_scan():
    rows = read_records(DATA_CSV_DIR/'data_04_taylor_green_parameter_scan.csv')
    groups = {}
    for row in rows:
        key = (Fraction(str(row['a_exact'])), row['rate_choice'], Fraction(str(row['s_plus_exact'])))
        groups.setdefault(key, []).append(row)
    results = []
    for (a, choice, sp), values in groups.items():
        values.sort(key=lambda row: row['n'])
        if len(values) != 2:
            raise ValueError(f'Incomplete two-grid parameter pair: {a}, {sp}')
        grids = [SimpleNamespace(n=r['n'], h=r['h'], error=r['e_u2'], d2=r['d2'],
                                 completed=r['status'] != 'failure', outlier=r['status'] == 'outlier') for r in values]
        results.append(SimpleNamespace(a=a, rate_choice=choice, s_plus=sp,
                                       coarse=grids[0], fine=grids[1]))
    return results


def rows_for_choice(results, choice):
    return sorted((row for row in results if row.choice == choice), key=lambda row: row.n)


def completed_rows_for_choice(results, choice):
    rows = rows_for_choice(results, choice)
    if any(row.status != 'completed' for row in rows):
        raise ValueError(f'Incomplete convergence series for {choice}; do not silently omit failed grids.')
    return rows


def slope(values, h):
    values = np.asarray(values, dtype=float)
    h = np.asarray(h, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('A convergence fit needs at least two positive finite errors.')
    return float(np.polyfit(np.log(h), np.log(values), 1)[0])


def sci(value, precision=3):
    if not math.isfinite(value):
        return r'\textit{unavailable}'
    mantissa, exponent = f'{value:.{precision}e}'.split('e')
    return rf'${mantissa}\mathrm{{e}}{{{int(exponent)}}}$'


def sci_times(value, precision=3):
    """Retain the multiplication/power notation of the acoustic/shear tables."""
    if not math.isfinite(value):
        return r'\textit{unavailable}'
    mantissa, exponent = f'{value:.{precision}e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def complete_table(path):
    """Keep the reviewed caption/layout while replacing only numerical rows."""
    path = Path(path)
    template = CODE_DIR/'_tables/templates'/path.name
    if not template.exists():
        return path
    latex = path.read_text(encoding='utf-8')
    # Longtable data begin after the repeated header/footer definitions.
    start = latex.index(r'\endlastfoot') + len(r'\endlastfoot') if r'\endlastfoot' in latex else latex.index(r'\midrule')+len(r'\midrule')
    end = latex.rfind(r'\bottomrule') if r'\endlastfoot' not in latex else latex.rfind(r'\end{longtable}')
    body = latex[start:end].strip()
    text = template.read_text(encoding='utf-8').replace('% CSV_ROWS', body)
    path.write_text('% Generated from experimental CSV; edit the generator or template.\n'+text, encoding='utf-8')
    return path


def save_table(name, caption, label, columns, headings, body):
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = ('\\begin{table}[htbp]\n\\centering\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
            '\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{3.2pt}\n'
            f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\n\\toprule\n'
            + headings + '\n\\midrule\n' + '\n'.join(body)
            + '\n\\bottomrule\n\\end{tabular}\n\\endgroup\n\\end{table}\n')
    path = TABLE_OUTPUT_DIR/name
    path.write_text(text, encoding='utf-8')
    return complete_table(path)
