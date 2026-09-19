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

"""Convergence studies with compatible saved-state reuse and continuation."""
from __future__ import annotations
from _periodic_runtime import environment, serial_run, trajectory
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from _csv_io import write_records
from _paths import CODE_DIR, DATA_CSV_DIR, INSTANCE_OUTPUT_DIR, reuse_instances
from _periodic_config import (A, NU, FINAL_TIME, N_LIST,
                              CONFIGURATIONS, LIMITS, MODEL_SOURCES, time_key)

PROTOCOL = 'periodic-saved-convergence-v1'


def model_hashes():
    return {name: hashlib.sha256((CODE_DIR/name).read_bytes()).hexdigest()
            for name in MODEL_SOURCES}


def state_hash(solver):
    return hashlib.sha256(solver.f.tobytes()).hexdigest()


def saved_metadata(solver):
    return getattr(solver, 'convergence', None)


def compatible(solver, initial, initial_sha, sources):
    """Validate the physical problem, initialization, kernel source and history."""
    if type(solver) is not type(initial):
        return False
    metadata = saved_metadata(solver)
    if not metadata or metadata.get('initial_state_sha256') != initial_sha:
        return False
    prior_sources = metadata.get('environment', {}).get('source_sha256', {})
    if any(prior_sources.get(name) != value for name, value in sources.items()):
        return False
    if metadata.get('environment', {}).get('limits') != LIMITS:
        return False
    for key in ('h', 'nu', 'a', 'alpha', 's_plus', 'U0', 'k1', 'k2', 'b1', 'b2',
                'epsilon', 'xshift', 'outerforce_type', 'initialization_protocol'):
        if getattr(solver, key, None) != getattr(initial, key, None):
            return False
    history = metadata.get('history', [])
    if (len(history) != solver.iter_count+1 or history[0]['step'] != 0
            or history[-1]['step'] != solver.iter_count
            or not math.isclose(history[-1]['actual_time'], solver.t, abs_tol=1e-14)):
        return False
    if metadata.get('final_state_sha256') != state_hash(solver):
        return False
    return True


def candidate_files(base, method, alpha, n, resume_from=None):
    roots = ([Path(resume_from).resolve()] if resume_from else
             [INSTANCE_OUTPUT_DIR/(base+'Convergence')])
    stem = f'{method.lower()}_alpha{alpha:g}_N{n}'
    found = []
    for root in roots:
        candidates = [root] if root.is_file() else root.glob(stem+'*.pkl')
        if root.is_dir():
            candidates = root.rglob(stem+'*.pkl')
        for path in candidates:
            if path.stem == stem or path.stem.startswith(stem+'_'):
                found.append(path)
    return sorted(set(found), key=lambda p:p.stat().st_mtime_ns, reverse=True)


def find_saved(initial, base, method, alpha, n, final_time, *, resume_from=None):
    initial_sha, sources = state_hash(initial), model_hashes()
    best = None
    for path in candidate_files(base, method, alpha, n, resume_from):
        try:
            prior = type(initial).load_instance(path)
        except TypeError:
            continue
        if not compatible(prior, initial, initial_sha, sources):
            continue
        metadata = saved_metadata(prior)
        if prior.t > final_time+1e-14:
            continue
        if best is None or prior.iter_count > best[0].iter_count:
            best = prior, metadata, path
        if math.isclose(prior.t, final_time, abs_tol=1e-14):
            break
    return best


def target_steps(solver, final_time):
    if not math.isfinite(final_time) or final_time <= 0:
        raise ValueError('The final time must be positive and finite.')
    steps = round(final_time/solver.dt)
    if final_time <= 0 or not math.isclose(steps*solver.dt, final_time, rel_tol=0, abs_tol=1e-14):
        raise ValueError('The positive final time must be an integer number of time steps.')
    return steps


def result_row(solver, method, alpha, n, final_time, measurement):
    completed = measurement['status'] == 'completed'
    errors = solver.get_error() if completed else (None,)*3
    return dict(method=method, alpha=alpha, n=n, h=solver.h, target_time=final_time,
                e_u1=errors[0], e_u2=errors[1], e_uinf=errors[2],
                **measurement)


def run_convergence(factory, prefix, base, *, final_time=FINAL_TIME,
                    rerun=False, resume_from=None):
    """Reuse, continue or freshly compute every case; always save final solvers."""
    if resume_from is not None and not Path(resume_from).exists():
        raise FileNotFoundError(f'Restart directory does not exist: {resume_from}')
    for n in N_LIST:
        for method,alpha in CONFIGURATIONS:
            target_steps(factory(method,alpha,n),final_time)
    with serial_run():
        metadata = environment(final_time)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory = INSTANCE_OUTPUT_DIR/(base+'Convergence')/time_key(final_time)/run_id
        directory.mkdir(parents=True)
        results = []
        print(f'Convergence {prefix}: T={final_time}; N={N_LIST}', flush=True)
        for n in N_LIST:
            for method, alpha in CONFIGURATIONS:
                initial = factory(method, alpha, n)
                steps = target_steps(initial, final_time)
                initial_sha = state_hash(initial)
                prior = (find_saved(initial, base, method, alpha, n, final_time, resume_from=resume_from)
                         if not rerun and reuse_instances() else None)
                if prior is None:
                    solver, history, source = initial, None, None
                    start_step = 0
                else:
                    solver, prior_metadata, source_path = prior
                    history = prior_metadata['history']
                    source = str(source_path.relative_to(CODE_DIR)
                                 if source_path.is_relative_to(CODE_DIR) else source_path)
                    start_step = solver.iter_count
                measurement, history = trajectory(solver, steps, resume_history=history)
                row = result_row(solver, method, alpha, n, final_time, measurement)
                solver.convergence = dict(protocol=PROTOCOL, run_id=run_id,
                    environment=metadata, method=method, alpha=alpha, a=A, nu=NU, n=n,
                    target_time=final_time, source_instance=source, resumed_at_step=start_step,
                    initial_state_sha256=initial_sha, final_state_sha256=state_hash(solver),
                    measurement=measurement,
                    errors=row, history=history)
                solver.save_instance(directory/f'{method.lower()}_alpha{alpha:g}_N{n}.pkl')
                results.append(row)
                write_records(DATA_CSV_DIR/f'data_{prefix}_results.csv', results)
                operation = 'fresh' if source is None else ('reused' if start_step==solver.iter_count else 'continued')
                print(f'{prefix} N={n:3d} {method:4s} alpha={alpha:g}: '
                      f'{operation}, {start_step}->{solver.iter_count}/{steps}, {row["status"]}', flush=True)
        print(f'Saved convergence instances: {directory}', flush=True)
    return results


def run_experiment(factory, prefix, base, args):
    return run_convergence(factory, prefix, base, rerun=args.rerun,
                           resume_from=args.resume_from)
