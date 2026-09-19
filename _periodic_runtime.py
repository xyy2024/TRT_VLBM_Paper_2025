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

"""Shared evolution, stopping criteria and saved-run metadata for experiments 02/03."""
from contextlib import contextmanager
import gc
import hashlib
import math
import os
import platform
import sys
import numpy as np
from _paths import CODE_DIR, INSTANCE_OUTPUT_DIR
from _periodic_config import FINAL_TIME, LIMITS, MODEL_SOURCES


def environment(final_time=FINAL_TIME):
    sources = (*MODEL_SOURCES, '_periodic_runtime.py', '_periodic_config.py',
               '_periodic_convergence.py', '_csv_io.py',
               '02_nonlinear_manufactured_convergence.py', '03_taylor_green_convergence.py')
    return dict(os=platform.platform(), python=sys.version, numpy=np.__version__,
                source_sha256={name: hashlib.sha256((CODE_DIR/name).read_bytes()).hexdigest()
                               for name in sources}, target_time=final_time, limits=LIMITS.copy())


@contextmanager
def serial_run():
    INSTANCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = INSTANCE_OUTPUT_DIR/'.periodic_run.lock'
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f'Another periodic run is active. If its process has stopped, remove {lock}.') from error
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def diagnostics(solver):
    rho = solver.w[..., 0]
    if not np.isfinite(solver.w).all() or not np.isfinite(solver.f).all():
        return {key: math.nan for key in LIMITS}
    u, v = solver.get_numerical_speed()
    divergence = ((np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0))
                  + (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1))) / (2 * solver.h)
    return dict(d2=float(solver.h * np.linalg.norm(divergence.ravel())),
        density_min=float(rho.min()), density_max=float(rho.max()),
        density_fluctuation=float(np.max(np.abs(rho - rho.mean()))),
        m_h=float(solver.h * np.sqrt(np.max(u * u + v * v))))


def threshold(values):
    if not all(math.isfinite(value) for value in values.values()):
        return 'nonfinite'
    for key in ('density_min', 'density_max', 'density_fluctuation', 'd2', 'm_h'):
        if ((values[key] < LIMITS[key]) if key == 'density_min'
                else (values[key] > LIMITS[key])):
            return key
    return ''


def trajectory(solver, target_steps, *, resume_history=None):
    """Advance to an absolute step; only convergence supplies a saved history."""
    if resume_history is None and (solver.iter_count != 0 or solver.t != 0):
        raise ValueError('A fresh trajectory must start from t=0.')
    if solver.iter_count > target_steps:
        raise ValueError('A saved state later than the target cannot be rewound.')
    values = diagnostics(solver)
    if solver.iter_count == 0 and threshold(values):
        raise ValueError('Initial data violate the prescribed diagnostic limits.')
    history = (list(resume_history) if resume_history is not None else
               [dict(step=0, actual_time=0.0, **values)])
    if len(history) != solver.iter_count+1 or history[-1]['step'] != solver.iter_count:
        raise ValueError('The saved per-step history is incomplete.')
    reason = threshold(values)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            for _ in range(solver.iter_count, target_steps):
                if reason:
                    break
                try:
                    solver.iter()
                    values = diagnostics(solver)
                    reason = threshold(values)
                except FloatingPointError:
                    reason = 'nonfinite'
                    solver.overflowed = True
                    values = {key: math.nan for key in values}
                history.append(dict(step=solver.iter_count, actual_time=float(solver.t), **values))
                if reason:
                    break
    finally:
        if was_enabled:
            gc.enable()
    status = 'nonfinite' if reason == 'nonfinite' else ('threshold' if reason else 'completed')
    return dict(status=status, stop_reason=reason, steps=solver.iter_count,
                actual_time=float(solver.t), **values), history
