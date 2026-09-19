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

"""Unrounded CSV values needed by the article and supplement generators.

Case settings, full histories and restart arrays belong to solver instances.
The schemas below define the complete CSV interface; display rounding belongs
exclusively to the table generators.
"""
from __future__ import annotations
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
from _paths import DATA_CSV_DIR

CSV_COLUMNS = {
    'data_02_nonlinear_manufactured_results.csv': 'method alpha n h target_time status stop_reason steps actual_time e_u1 e_u2 e_uinf d2 density_fluctuation m_h',
    'data_03_taylor_green_results.csv': 'method alpha n h target_time status stop_reason steps actual_time e_u2 d2 density_fluctuation m_h',
    'data_04_taylor_green_parameter_scan.csv': 'rate_choice a_exact s_plus_exact n h status e_u2 d2',
    'data_05_poiseuille_boundary_tuning.csv': 'ell choice s_plus n h velocity_error fitted_slip predicted_slip fixed_point_full_residual',
    'data_05_poiseuille_profiles.csv': 'choice ell s_plus h y u_numerical u_exact',
    'data_09_kolmogorov_startup_results.csv': 'method n relative_deviation transverse_ratio stationarity_tail_max divergence_l2 density_fluctuation lattice_speed',
    'data_09_kolmogorov_startup_fields.csv': 'method n i j x y u1 u2',
    'data_08_double_shear_results.csv': 'n key status step actual_time energy_ratio divergence_l2 density_fluctuation',
    'data_06_beltrami_results.csv': 'choice s_plus n h status e_u1_l2 e_u2_l2 e_u3_l2 e_u_l2 d2 mass_drift density_min density_max density_fluctuation energy_ratio m_h',
    'data_07_ethier_steinman_results.csv': 'choice s_plus n h actual_time status stop_reason e_u1_l2 e_u2_l2 e_u3_l2 e_u_l2 d2 mass_drift density_min density_max density_fluctuation energy_ratio m_h',
    'data_01_periodic_acoustic_results.csv': 'choice n step time status modal_gain divergence_l2 density_fluctuation lattice_speed max_mass_drift rate_fit',
    'data_01_periodic_acoustic_histories.csv': 'choice n step modal_gain divergence_l2',
    'data_01_periodic_acoustic_modes.csv': 'choice modulus',
}


def write_records(path, rows):
    """Project onto the artifact schema and write round-trip decimals atomically."""
    path = Path(path)
    if path.suffix != '.csv' or path.name not in CSV_COLUMNS:
        raise ValueError(f'No article CSV schema registered for {path.name}.')
    rows = list(rows)
    if path.name == 'data_05_poiseuille_profiles.csv':
        rows = [row for row in rows if row['choice'] in ('OTRT', 'SRT', 'tuned')]
    elif path.name == 'data_01_periodic_acoustic_histories.csv':
        rows = [row for row in rows if row['n'] == 128]
    if not rows:
        raise ValueError(f'No records to export: {path}')
    columns = CSV_COLUMNS[path.name].split()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                dir=path.parent, prefix='.pending_', suffix='.csv', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in columns})
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def read_records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing experimental CSV: {path}. Run the corresponding numbered experiment first.')
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    def convert(value):
        if value == '':
            return None
        if value in ('True', 'False'):
            return value == 'True'
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return [{key: convert(value) for key, value in row.items()} for row in rows]


def read_3d(number):
    suffix = 'beltrami' if number == '06' else 'ethier_steinman'
    result = []
    for row in read_records(DATA_CSV_DIR/f'data_{number}_{suffix}_results.csv'):
        aliases = dict(component_error=tuple(row[f'e_u{i}_l2'] for i in (1, 2, 3)),
                       velocity_error=row['e_u_l2'], divergence_l2=row['d2'],
                       lattice_speed=row['m_h'])
        result.append(SimpleNamespace(**row, **aliases))
    return result
