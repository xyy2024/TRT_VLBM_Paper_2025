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

"""Run the three-group unforced double-shear experiment and export article CSV.

Every case saves its solver, complete diagnostic history
under solver_instances. Matching instances are reused on subsequent runs.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from d2n5_doubleshear import d2n5_doubleshear
from _double_shear_diagnostics import diagnostics, threshold_reason
from _csv_io import write_records
from _paths import (CODE_DIR, DATA_CSV_DIR, INSTANCE_OUTPUT_DIR,
                    configure_experiment, reuse_instances)

N_LIST = (12, 16, 24, 32, 48, 64, 96, 128)
CASES = {'otrt_a05': (0.5, 'OTRT'), 'srt_a05': (0.5, 'SRT'),
         'srt_a02': (0.2, 'SRT')}
PROTOCOL = 'double-shear-alpha-every-step-monitoring-v4'
CHECKPOINT_INTERVAL = 512
DATA_FILE = DATA_CSV_DIR / 'data_08_double_shear_results.csv'
INSTANCE_DIR = INSTANCE_OUTPUT_DIR / 'DoubleShearAlphaArticle'


def rates(key):
    alpha, method = CASES[key]
    sm = 0.4/(0.2+alpha*0.01)
    return sm, 2-sm if method == 'OTRT' else sm


def endpoint(n):
    return (n*n+4)//5


def metadata(key, n):
    digest = hashlib.sha256()
    for name in ('08_double_shear_dynamics.py', '_double_shear_diagnostics.py',
                 'd2n5_doubleshear.py', 'd2n5_taylorgreen.py', '_nproll.py'):
        digest.update((CODE_DIR/name).read_bytes())
    alpha, method = CASES[key]
    return dict(protocol=PROTOCOL, key=key, n=n, method=method, alpha=alpha,
                a=0.2, nu=0.01, u0=0.25, delta=0.1, epsilon_v=0.0125,
                initial_pressure=True, s_minus=rates(key)[0], s_plus=rates(key)[1],
                target_steps=endpoint(n)*(2 if alpha == 0.5 else 5),
                target_time=endpoint(n)/n**2, source_hash=digest.hexdigest())


def solver_for(key, n):
    solver = d2n5_doubleshear(h=1/n, nu=0.01, U0=0.25, delta=0.1,
                            epsilon_v=0.0125, a=0.2, alpha=CASES[key][0],
                            s_plus=rates(key)[1], initial_pressure=True)
    initial = diagnostics(solver, 1., 1., 1.)
    solver.experiment_metadata = metadata(key, n)
    solver.experiment_initial_mass = float(solver.h**2*np.sum(solver.w[..., 0]))
    solver.experiment_initial_energy = initial['energy']
    solver.experiment_initial_enstrophy = initial['enstrophy']
    solver.experiment_history = [dict(step=0, time=0., **diagnostics(solver,
        solver.experiment_initial_mass, initial['energy'], initial['enstrophy']))]
    solver.experiment_status = 'running'
    solver.experiment_stop_reason = ''
    return solver


def instance_matches(solver, expected):
    if type(solver) is not d2n5_doubleshear or getattr(solver, 'experiment_metadata', None) != expected:
        return False
    for name, value in dict(h=1/expected['n'], nu=expected['nu'], a=expected['a'],
                            alpha=expected['alpha'], s_plus=expected['s_plus'],
                            U0=expected['u0'], delta=expected['delta'], epsilon_v=expected['epsilon_v']).items():
        if not math.isclose(float(getattr(solver, name)), value, rel_tol=0, abs_tol=1e-14):
            return False
    history = solver.experiment_history
    if not solver.initial_pressure or not history or solver.iter_count > expected['target_steps']:
        return False
    if solver.experiment_status not in ('running', 'completed', 'threshold', 'overflow'):
        return False
    if solver.experiment_status == 'completed' and solver.iter_count != expected['target_steps']:
        return False
    if solver.experiment_status != 'overflow' and history[-1]['step'] != solver.iter_count:
        return False
    return all(row['step'] == step for step, row in enumerate(history))


def save_solver(solver, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    solver.save_instance(path)


def run_case(key, n):
    path = INSTANCE_DIR / f'{key}_N{n}.pkl'
    expected = metadata(key, n)
    solver = None
    if reuse_instances() and path.is_file():
        try:
            saved = d2n5_doubleshear.load_instance(path)
            if instance_matches(saved, expected):
                solver = saved
        except (OSError, ValueError, TypeError, AttributeError, EOFError) as error:
            print(f'Ignoring unusable instance {path}: {error}', flush=True)
    if solver is None:
        solver = solver_for(key, n)
    if solver.experiment_status == 'running':
        with np.errstate(divide='raise', over='raise', invalid='raise', under='ignore'):
            while solver.iter_count < expected['target_steps']:
                try:
                    solver.iter()
                    if not np.isfinite(solver.f).all():
                        raise FloatingPointError('nonfinite population')
                    values = diagnostics(solver, solver.experiment_initial_mass,
                        solver.experiment_initial_energy, solver.experiment_initial_enstrophy)
                    solver.experiment_history.append(dict(step=solver.iter_count, time=float(solver.t), **values))
                    reason = threshold_reason(values)
                    if reason:
                        solver.experiment_status = 'threshold'
                        solver.experiment_stop_reason = reason
                        break
                    if solver.iter_count % CHECKPOINT_INTERVAL == 0:
                        save_solver(solver, path)
                except FloatingPointError as error:
                    solver.overflowed = True
                    solver.experiment_status = 'overflow'
                    solver.experiment_stop_reason = str(error)
                    break
        if solver.experiment_status == 'running':
            solver.experiment_status = 'completed'
        save_solver(solver, path)
    values = solver.experiment_history[-1]
    row = dict(n=n, key=key, status=solver.experiment_status,
               step=solver.iter_count, actual_time=float(solver.t),
               energy_ratio=values['energy_ratio'], divergence_l2=values['divergence_l2'],
               density_fluctuation=values['density_fluctuation'])
    print(f"N={n} {key}: {row['status']}, step={row['step']}/{expected['target_steps']}, "
          f"D2={row['divergence_l2']:.6g}; instance: {path}", flush=True)
    return row, solver


def main():
    rows = []
    for n in N_LIST:
        for key in CASES:
            row, solver = run_case(key, n)
            rows.append(row)
            write_records(DATA_FILE, rows)
            del solver


if __name__ == '__main__':
    configure_experiment()
    main()
