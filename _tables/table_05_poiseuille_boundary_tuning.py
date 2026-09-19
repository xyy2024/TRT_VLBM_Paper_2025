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
from _artifact_io import (DATA_CSV_DIR, TABLE_OUTPUT_DIR, complete_table, math, read_records)
from _tabletools import poiseuille_tuning_table
TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = read_records(DATA_CSV_DIR/'data_05_poiseuille_boundary_tuning.csv')
    table = poiseuille_tuning_table(dir=TABLE_OUTPUT_DIR, file_name='table_05_poiseuille_boundary_tuning.tex')
    for ell, choice in dict.fromkeys((r['ell'], r['choice']) for r in rows):
        series = sorted((r for r in rows if (r['ell'], r['choice']) == (ell, choice)), key=lambda r: r['n'])
        coarse, fine = series[-2:]
        order = None if choice == 'tuned' else math.log(coarse['velocity_error']/fine['velocity_error'])/math.log(coarse['h']/fine['h'])
        label = {'0.8 tuned': r'$0.8\,s_{+}^{\rm tune}$', '1.2 tuned': r'$1.2\,s_{+}^{\rm tune}$'}.get(choice, choice)
        table.add_row(ell, label, fine['s_plus'], fine['velocity_error'], order,
                      fine['fitted_slip'], fine['predicted_slip'])
    print(complete_table(table.save()))

if __name__ == '__main__':
    main()

