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
from _artifact_io import (DATA_CSV_DIR, TABLE_OUTPUT_DIR, read_records, save_table)
from _artifact_io import sci_times as sci
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


rows = read_records(DATA_CSV_DIR/'data_08_double_shear_results.csv')
by_case = {(r['n'], r['key']): r for r in rows}
N_LIST = sorted({r['n'] for r in rows})

def main():
    body = []
    for n in N_LIST:
        o, s, f = [by_case[n, key] for key in ('otrt_a05', 'srt_a02', 'srt_a05')]
        assert o['status'] == s['status'] == 'completed' and f['status'] == 'threshold'
        assert o['step']*5 == s['step']*2
        body.append(f"{n} & {o['step']} & {f['step']} & {f['actual_time']:.6f} & {s['step']} & {sci(o['divergence_l2'], 2)}"+r'\\')
    print(save_table('table_08_double_shear_alpha.tex', '', 'tab:numerical-double-shear-alpha',
                     'rrrrrr', '', body))
if __name__ == '__main__':
    main()

