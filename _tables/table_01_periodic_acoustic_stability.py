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

"""Summarize all acoustic grids from CSV, without running a solver."""
import _bootstrap
from _artifact_io import CODE_DIR, DATA_CSV_DIR, TABLE_OUTPUT_DIR, complete_table, read_records
from _artifact_io import sci_times as sci


def write_table():
    prefix = '01_periodic_acoustic'
    results = read_records(DATA_CSV_DIR / f'data_{prefix}_results.csv')
    modes = read_records(DATA_CSV_DIR / f'data_{prefix}_modes.csv')
    theoretical = {row['choice']: row['modulus'] for row in modes}
    body = []
    for choice in ('otrt', 'srt'):
        rows = [row for row in results if row['choice'] == choice]
        if sorted(row['n'] for row in rows) != [12, 16, 24, 32, 48, 64, 96, 128]:
            raise ValueError(f'Incomplete acoustic grid sequence for {choice}')
        steps = {row['step'] for row in rows}
        if len(steps) != 1:
            raise ValueError(f'Acoustic stopping steps differ across grids for {choice}')
        radius = theoretical[choice]
        fit_error = max(abs(row['rate_fit'] - radius) for row in rows)
        divergence = max(row['divergence_l2'] for row in rows)
        body.append(f'{choice.upper()} & {radius:.10f} & {sci(fit_error)} & '
                    f'{steps.pop()} & {sci(divergence)}' + r'\\')
    name = f'table_{prefix}_stability.tex'
    template = (CODE_DIR / '_tables/templates' / name).read_text(encoding='utf-8')
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_OUTPUT_DIR / name
    path.write_text(template.replace('% CSV_ROWS', '\n'.join(body)), encoding='utf-8')
    return complete_table(path)


if __name__ == '__main__':
    print(write_table())
