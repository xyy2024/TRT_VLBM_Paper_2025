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

"""Supplementary refinement of the original forced Kolmogorov start-up problem.

The full initial velocity is a 1e-3 Taylor--Green vortex, without the steady
Kolmogorov shear. Only N varies. Numerical data are exported as unrounded CSV;
full states, parameters and diagnostics are retained in solver_instances/.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os

import numpy as np

from _csv_io import write_records
from _paths import (DATA_CSV_DIR, INSTANCE_OUTPUT_DIR, configure_experiment,
                    reuse_instances)
from d2n5_kolmogorov import d2n5_kolmogorov_startup

NU = 0.01
A = ALPHA = 0.2
N_LIST = (6, 12, 24, 48)
S_MINUS = 200/101
RATE_CHOICES = (('TRT', 0.02), ('SRT', S_MINUS))
FINAL_TIME = 150.0
SAMPLE_INTERVAL = 0.1
STATIONARITY_WINDOW = 5.0
STATIONARITY_TOLERANCE = 1e-8
PROTOCOL = 'kolmogorov-startup-refinement-v1'
INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'KolmogorovStartupRefinement'


def configuration(method, s_plus, n):
    return dict(protocol=PROTOCOL, method=method, n=n, nu=NU, a=A, alpha=ALPHA,
                s_plus=s_plus, s_minus=S_MINUS, initial_amplitude=1e-3,
                final_time=FINAL_TIME, sample_interval=SAMPLE_INTERVAL,
                stationarity_window=STATIONARITY_WINDOW,
                stationarity_tolerance=STATIONARITY_TOLERANCE,
                force_amplitude=1.0, kf=math.tau, xshift=0.5,
                initial_pressure=0.0, initial_density=1.0,
                force_protocol='rest-population-alpha-h-cubed-force')


def diagnostics(solver, previous_velocity=None):
    velocity = np.asarray(solver.get_numerical_speed())
    reference = np.asarray(solver.get_precise_speed())
    reference_norm = float(np.sqrt(solver.h**2*np.sum(reference**2)))
    u, v = velocity
    divergence = ((np.roll(u, -1, axis=0)-np.roll(u, 1, axis=0))
                  + (np.roll(v, -1, axis=1)-np.roll(v, 1, axis=1)))/(2*solver.h)
    record = dict(
        step=solver.iter_count, time=float(solver.t),
        relative_deviation=float(np.linalg.norm(velocity-reference)*solver.h/reference_norm),
        transverse_ratio=float(np.linalg.norm(v)*solver.h/reference_norm),
        stationarity=(None if previous_velocity is None else
                      float(np.linalg.norm(velocity-previous_velocity)*solver.h
                            /(SAMPLE_INTERVAL*reference_norm))),
        divergence_l2=float(np.linalg.norm(divergence)*solver.h),
        density_fluctuation=float(np.max(np.abs(solver.w[..., 0]-1))),
        lattice_speed=float(solver.h*np.max(np.hypot(u, v))),
        mass_drift=float(abs(solver.h**2*np.sum(solver.w[..., 0])-1)),
        density_min=float(solver.w[..., 0].min()),
        density_max=float(solver.w[..., 0].max()),
    )
    return record, velocity


def save(solver, path):
    temporary = path.with_suffix('.pending.pkl')
    solver.save_instance(temporary)
    temporary.replace(path)


def validate_initial_state(solver):
    expected = solver.init_exact()
    for actual, target in zip(solver.get_numerical_speed(), expected):
        np.testing.assert_allclose(actual, target, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(solver.w[..., 0], 1, rtol=0, atol=0)
    np.testing.assert_allclose(solver.f.sum(axis=2), solver.w, rtol=0, atol=3e-16)
    np.testing.assert_allclose(solver.relax1-solver.relax2, S_MINUS, rtol=0, atol=1e-14)
    # At equilibrium the collision increment is exactly the prescribed source.
    increment = solver.get_fstar(solver.get_m()).sum(axis=2)-solver.w
    target = np.zeros_like(solver.w)
    target[..., 1:] = solver.alpha*solver.h**3*solver.get_outerforce()
    np.testing.assert_allclose(increment, target, rtol=0, atol=5e-16)


def compatible(solver, expected, config):
    saved_config = getattr(solver, 'refinement_config', {}).copy()
    saved_endpoint = saved_config.pop('final_time', math.inf)
    expected_config = config.copy()
    expected_config.pop('final_time')
    if (type(solver) is not type(expected) or saved_config != expected_config
            or saved_endpoint > FINAL_TIME):
        return False
    for name in ('h', 'nu', 'a', 'alpha', 's_plus', 'U0', 'xshift',
                 'kf', 'initial_amplitude', 'dt'):
        if not math.isclose(getattr(solver, name), getattr(expected, name), rel_tol=0, abs_tol=1e-14):
            return False
    return (solver.initialization_protocol == expected.initialization_protocol
            and bool(getattr(solver, 'refinement_history', []))
            and solver.refinement_history[-1]['step'] == solver.iter_count
            and solver.t <= FINAL_TIME+1e-12)


def run_case(method, s_plus, n):
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_DIR/f'{method.lower()}_N{n}.pkl'
    config = configuration(method, s_plus, n)
    expected = d2n5_kolmogorov_startup(h=1/n, nu=NU, a=A, alpha=ALPHA,
                                      s_plus=s_plus, xshift=0.5)
    validate_initial_state(expected)
    solver = expected
    if reuse_instances() and path.is_file():
        try:
            candidate = type(expected).load_instance(path)
            if compatible(candidate, expected, config):
                solver = candidate
                solver.refinement_config = config
                if solver.t < FINAL_TIME and solver.refinement_status == 'completed':
                    solver.refinement_status = 'running'
                print(f'Reusing {method}, N={n}, t={solver.t:g}.', flush=True)
        except Exception as error:
            print(f'Ignoring incompatible instance {path}: {error}', flush=True)
    if solver is expected:
        solver.refinement_config = config
        record, _ = diagnostics(solver)
        solver.refinement_history = [record]
        solver.refinement_status = 'running'
        save(solver, path)
    if solver.refinement_status != 'running':
        return str(path)

    sample_steps = round(SAMPLE_INTERVAL/solver.dt)
    final_steps = round(FINAL_TIME/solver.dt)
    assert math.isclose(sample_steps*solver.dt, SAMPLE_INTERVAL, abs_tol=1e-14)
    assert math.isclose(final_steps*solver.dt, FINAL_TIME, abs_tol=1e-12)
    print(f'Running {method}, N={n}, {final_steps-solver.iter_count} remaining steps.', flush=True)
    try:
        while solver.iter_count < final_steps:
            previous = np.asarray(solver.get_numerical_speed())
            completed = solver.until_step(sample_steps, show_progress=False)
            if (not completed or not np.isfinite(solver.w).all()
                    or solver.w[..., 0].min() <= 0):
                solver.refinement_status = 'invalid_state'
                solver.refinement_stop_reason = 'nonfinite state or nonpositive density'
                save(solver, path)
                raise RuntimeError(f'{method}, N={n}: invalid state at t={solver.t:g}.')
            record, _ = diagnostics(solver, previous)
            solver.refinement_history.append(record)
            if solver.iter_count % (sample_steps*10) == 0:
                save(solver, path)
            if solver.iter_count % (sample_steps*50) == 0:
                print(f'{method} N={n} t={solver.t:g}: B2={record["relative_deviation"]:.6e}, '
                      f'Q2={record["transverse_ratio"]:.6e}, R={record["stationarity"]:.3e}', flush=True)
        solver.refinement_status = 'completed'
        save(solver, path)
    except KeyboardInterrupt:
        # A saved integer-sample checkpoint is already available for resumption.
        raise
    return str(path)


def export(paths):
    summaries, fields = [], []
    for path in paths:
        solver = d2n5_kolmogorov_startup.load_instance(path)
        config = solver.refinement_config
        if (solver.refinement_status != 'completed'
                or not math.isclose(solver.t, FINAL_TIME, rel_tol=0, abs_tol=1e-10)):
            raise RuntimeError(f'Incomplete supplementary comparison: {path}')
        key = dict(method=config['method'], n=config['n'])
        tail = [r for r in solver.refinement_history if r['time'] > FINAL_TIME-STATIONARITY_WINDOW+1e-12]
        summary = dict(key, **solver.refinement_history[-1],
                       stationarity_tail_max=max(r['stationarity'] for r in tail))
        summaries.append(summary)
        u, v = solver.get_numerical_speed()
        for i, j in np.ndindex(solver.shape):
            fields.append(dict(key, i=i, j=j, x=float(solver.X[i,j]),
                               y=float(solver.Y[i,j]), u1=float(u[i,j]), u2=float(v[i,j])))
    summaries.sort(key=lambda r: (r['n'], r['method'] != 'TRT'))
    fields.sort(key=lambda r: (r['n'], r['method'] != 'TRT', r['i'], r['j']))
    for suffix, rows in (('results', summaries), ('fields', fields)):
        print(write_records(DATA_CSV_DIR/f'data_09_kolmogorov_startup_{suffix}.csv', rows), flush=True)


def main():
    cases = [(method, sp, n) for n in N_LIST for method, sp in RATE_CHOICES]
    workers = max(1, int(os.environ.get('TRTVLBM_KOLMOGOROV_WORKERS', '2')))
    if workers == 1:
        paths = [run_case(*case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(cases))) as pool:
            futures = [pool.submit(run_case, *case) for case in cases]
            paths = [future.result() for future in as_completed(futures)]
    export(paths)


if __name__ == '__main__':
    configure_experiment()
    main()
