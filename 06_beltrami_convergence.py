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

"""Numerical experiment only: export English CSV, without rendering figures or tables."""


from __future__ import annotations
from _paths import configure_experiment, reuse_instances
import gc
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from _d3n7_fast import iter_zero_force
from _paths import DATA_CSV_DIR, INSTANCE_OUTPUT_DIR
from d3n7_beltrami import d3n7_beltrami


AMPLITUDE = 0.05


WAVENUMBER = 2*math.pi


NU = 0.05


A = 1/7


ALPHA = 1/7


S_MINUS = 40/21


TARGET_TIME = 3/14


N_LIST = (12, 16, 24, 32, 48, 64, 96, 128)


RATE_CHOICES = (
    ('OTRT', 2/21),
    ('SRT', 40/21),
)


INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'BeltramiArticle'


DATA_FILE = DATA_CSV_DIR/'data_06_beltrami_results.csv'


EXPERIMENT_PROTOCOL = 'beltrami-target-time-v2'


CACHE_PARAMETER_NAMES = (
    'h', 'dt', 'nu', 'a', 'alpha', 's_plus', 'amplitude', 'wavenumber',
)


CACHE_RESUMABLE_STATUSES = {'completed', 'running', 'interrupted'}


CACHE_TERMINAL_STATUSES = {'overflow'}


@dataclass(frozen=True)
class CaseResult:
    choice: str
    s_plus: float
    n: int
    h: float
    target_steps: int
    actual_time: float
    status: str
    stop_reason: str
    component_error: tuple[float, float, float]
    velocity_error: float
    divergence_l2: float
    mass_drift: float
    density_min: float
    density_max: float
    density_fluctuation: float
    energy_ratio: float
    lattice_speed: float
    instance_file: Path


def target_step_count(solver):
    raw_steps = TARGET_TIME/solver.dt
    target_steps = round(raw_steps)
    if not math.isclose(raw_steps, target_steps, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError(
            f'TARGET_TIME/dt must be an integer; received {raw_steps:.16g} '
            f'for h={solver.h:.16g} and dt={solver.dt:.16g}.'
        )
    return int(target_steps)


def centered_periodic_divergence(solver, velocity):
    u, v, w = velocity
    return (
        (np.roll(u, -1, axis=0)-np.roll(u, 1, axis=0))/(2*solver.h)
        + (np.roll(v, -1, axis=1)-np.roll(v, 1, axis=1))/(2*solver.h)
        + (np.roll(w, -1, axis=2)-np.roll(w, 1, axis=2))/(2*solver.h)
    )


def monitoring_diagnostics(solver, initial_mass, initial_energy):
    rho = solver.w[..., 0]
    velocity = solver.get_numerical_speed()
    speed_squared = sum(component**2 for component in velocity)
    cell_volume = solver.h**3
    mass = float(cell_volume*np.sum(rho))
    energy = float(0.5*cell_volume*np.sum(speed_squared))
    divergence = centered_periodic_divergence(solver, velocity)
    density_mean = float(np.mean(rho))
    return {
        'density_min': float(np.min(rho)),
        'density_max': float(np.max(rho)),
        'density_fluctuation': float(np.max(np.abs(rho-density_mean))),
        'energy_ratio': energy/initial_energy,
        'divergence_l2': float(np.sqrt(cell_volume*np.sum(divergence**2))),
        'mass_drift': abs(mass-initial_mass),
        'lattice_speed': float(solver.h*np.sqrt(np.max(speed_squared))),
    }


def cache_file(choice, n):
    tag = choice.lower().replace(' ', '_').replace('.', 'p')
    return INSTANCE_DIR/f'{tag}_N{n}.pkl'


def inspect_saved_instance(solver, expected, target_steps):
    '''Return ``(action, details)`` after validating a cached solver.

    ``action`` is one of ``reuse``, ``resume``, ``terminal`` or ``reject``.
    A completed result at an earlier compatible target is resumable; its
    original mass and energy references are deliberately retained.
    '''
    problems = []
    if type(solver) is not type(expected):
        problems.append(
            f'type={type(solver).__module__}.{type(solver).__qualname__}'
        )
    for name in CACHE_PARAMETER_NAMES:
        saved_value = getattr(solver, name, np.nan)
        expected_value = getattr(expected, name, np.nan)
        if not np.isclose(
            saved_value, expected_value, rtol=0, atol=1e-14,
        ):
            problems.append(
                f'{name}={saved_value!r}, expected {expected_value!r}'
            )
    protocol = getattr(solver, 'experiment_protocol', None)
    if protocol != EXPERIMENT_PROTOCOL:
        problems.append(
            f'protocol={protocol!r}, expected {EXPERIMENT_PROTOCOL!r}'
        )

    iteration = getattr(solver, 'iter_count', None)
    if not isinstance(iteration, (int, np.integer)) or iteration < 0:
        problems.append(f'invalid iter_count={iteration!r}')
    elif iteration > target_steps:
        problems.append(
            f'iter_count={iteration} is beyond new target {target_steps}'
        )
    elif not math.isclose(
        float(getattr(solver, 't', math.nan)), iteration*expected.dt,
        rel_tol=0, abs_tol=1e-13,
    ):
        problems.append(
            f't={getattr(solver, "t", None)!r} is inconsistent with '
            f'iter_count={iteration}'
        )

    status = getattr(solver, 'experiment_status', None)
    known_statuses = CACHE_RESUMABLE_STATUSES | CACHE_TERMINAL_STATUSES
    if status not in known_statuses:
        problems.append(f'unknown status={status!r}')

    old_target_steps = getattr(solver, 'requested_target_steps', None)
    old_target_time = getattr(solver, 'requested_target_time', None)
    if not isinstance(old_target_steps, (int, np.integer)):
        problems.append(f'invalid requested_target_steps={old_target_steps!r}')
    elif old_target_steps <= 0:
        problems.append(f'invalid requested_target_steps={old_target_steps!r}')
    if not isinstance(old_target_time, (int, float, np.integer, np.floating)):
        problems.append(f'invalid requested_target_time={old_target_time!r}')
    elif not math.isfinite(float(old_target_time)) or old_target_time <= 0:
        problems.append(f'invalid requested_target_time={old_target_time!r}')
    if (
        status == 'completed'
        and isinstance(iteration, (int, np.integer))
        and isinstance(old_target_steps, (int, np.integer))
        and iteration != old_target_steps
    ):
        problems.append(
            f'completed iter_count={iteration} does not match saved target '
            f'{old_target_steps}'
        )
    elif (
        isinstance(iteration, (int, np.integer))
        and isinstance(old_target_steps, (int, np.integer))
        and iteration > old_target_steps
    ):
        problems.append(
            f'iter_count={iteration} exceeds saved target {old_target_steps}'
        )

    for name in ('experiment_initial_mass', 'experiment_initial_energy'):
        value = getattr(solver, name, math.nan)
        if not isinstance(value, (int, float, np.integer, np.floating)):
            problems.append(f'invalid {name}={value!r}')
        elif not math.isfinite(float(value)) or value <= 0:
            problems.append(f'invalid {name}={value!r}')

    if problems:
        return 'reject', '; '.join(problems)
    details = (
        f'status={status}, saved target={float(old_target_time):.12g} '
        f'({int(old_target_steps)} steps), current t={solver.t:.12g} '
        f'({int(iteration)} steps), new target={TARGET_TIME:.12g} '
        f'({target_steps} steps)'
    )
    if status in CACHE_TERMINAL_STATUSES:
        return 'terminal', details
    if iteration == target_steps and status == 'completed':
        return 'reuse', details
    if iteration < target_steps:
        return 'resume', details
    return 'reject', details + '; target state is not marked completed'


def prepare_experiment_metadata(solver, target_steps, *, preserve_baseline=False):
    if not preserve_baseline:
        rho = solver.w[..., 0]
        velocity = solver.get_numerical_speed()
        speed_squared = sum(component**2 for component in velocity)
        solver.experiment_initial_mass = float(solver.h**3*np.sum(rho))
        solver.experiment_initial_energy = float(
            0.5*solver.h**3*np.sum(speed_squared)
        )
    solver.experiment_protocol = EXPERIMENT_PROTOCOL
    solver.requested_target_time = TARGET_TIME
    solver.requested_target_steps = target_steps
    solver.experiment_status = 'running'
    solver.experiment_stop_reason = ''
    solver.overflowed = False


def save_compact_instance(solver, instance_file):
    '''Save the restartable state without disposable work-array caches.'''
    for name in (
        'm', 'fstar', 'nextf', '_A1', '_A2', '_A3',
        '_outerforce', '_force_term',
    ):
        solver.__dict__[name] = None
    solver._general_border_property_value = None
    solver._general_border_property_time = None
    solver.save_instance(instance_file)


def advance_to_target(solver, target_steps, instance_file):
    remaining_steps = target_steps-solver.iter_count
    update_interval = max(1, remaining_steps//20)
    starting_iteration = solver.iter_count
    try:
        with np.errstate(
            divide='raise', over='raise', invalid='raise', under='ignore'
        ):
            while solver.iter_count < target_steps:
                iter_zero_force(solver)
                if (
                    (solver.iter_count-starting_iteration) % update_interval == 0
                    or solver.iter_count == target_steps
                ):
                    print(
                        f'  step {solver.iter_count}/{target_steps}, '
                        f't={solver.t:.9f}'
                    )
            else:
                solver.experiment_status = 'completed'
                solver.experiment_stop_reason = ''
    except FloatingPointError as error:
        solver.experiment_status = 'overflow'
        solver.experiment_stop_reason = str(error)
        solver.overflowed = True
    except KeyboardInterrupt:
        solver.experiment_status = 'interrupted'
        solver.experiment_stop_reason = 'KeyboardInterrupt'
        save_compact_instance(solver, instance_file)
        print(f'Saved interrupted instance: {instance_file.resolve()}')
        raise
    save_compact_instance(solver, instance_file)


def run_case(choice, s_plus, n):
    solver = d3n7_beltrami(
        h=1/n,
        nu=NU,
        a=A,
        alpha=ALPHA,
        s_plus=s_plus,
        amplitude=AMPLITUDE,
        wavenumber=WAVENUMBER,
    )
    calculated_s_minus = solver.relax1-solver.relax2
    if np.any(solver.get_outerforce() != 0):
        raise RuntimeError('The zero-force D3N7 accelerator is not applicable.')
    if not np.isclose(calculated_s_minus, S_MINUS, rtol=0, atol=1e-14):
        raise RuntimeError(
            f'Unexpected s_minus={calculated_s_minus}; expected {S_MINUS}.'
        )
    if not (ALPHA/2 < A < 1/6):
        raise RuntimeError('The three-dimensional equilibrium condition fails.')

    target_steps = target_step_count(solver)
    instance_file = cache_file(choice, n)
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    if reuse_instances() and instance_file.is_file():
        try:
            saved = d3n7_beltrami.load_instance(instance_file)
            action, details = inspect_saved_instance(
                saved, solver, target_steps
            )
            print(f'Inspected Beltrami cache {instance_file.resolve()}: {details}.')
            if action in {'reuse', 'terminal'}:
                del solver
                gc.collect()
                if instance_file.stat().st_size > 512*1024**2:
                    save_compact_instance(saved, instance_file)
                    print(f'Compacted saved instance: {instance_file.resolve()}')
                print(f'Reusing Beltrami result: {instance_file.resolve()}')
                return saved, instance_file
            if action == 'resume':
                old_iteration = saved.iter_count
                old_time = saved.t
                del solver
                gc.collect()
                prepare_experiment_metadata(
                    saved, target_steps, preserve_baseline=True
                )
                print(
                    f'Resuming Beltrami {choice}, N={n}, from step '
                    f'{old_iteration} (t={old_time:.12g}) to step '
                    f'{target_steps} (t={TARGET_TIME:.12g}).'
                )
                advance_to_target(saved, target_steps, instance_file)
                return saved, instance_file
            print(f'Ignoring incompatible saved instance: {details}.')
        except Exception as error:
            print(f'Ignoring unusable saved instance {instance_file}: {error}')

    prepare_experiment_metadata(solver, target_steps)
    print(
        f'Running Beltrami {choice}, N={n}, steps={target_steps}, '
        f't_end={target_steps*solver.dt:.12g}, s_plus={s_plus:.12g}.'
    )
    advance_to_target(solver, target_steps, instance_file)
    return solver, instance_file


def result_from_solver(choice, s_plus, n, solver, instance_file):
    diagnostics = monitoring_diagnostics(
        solver,
        solver.experiment_initial_mass,
        solver.experiment_initial_energy,
    )
    if solver.experiment_status == 'overflow':
        component_error = (math.nan, math.nan, math.nan)
        velocity_error = math.nan
    else:
        component_error_array = solver.get_component_error_l2()
        component_error = tuple(float(value) for value in component_error_array)
        velocity_error = float(np.linalg.norm(component_error_array))
    return CaseResult(
        choice=choice,
        s_plus=float(s_plus),
        n=n,
        h=solver.h,
        target_steps=target_step_count(solver),
        actual_time=solver.t,
        status=solver.experiment_status,
        stop_reason=solver.experiment_stop_reason,
        component_error=component_error,
        velocity_error=velocity_error,
        divergence_l2=diagnostics['divergence_l2'],
        mass_drift=diagnostics['mass_drift'],
        density_min=diagnostics['density_min'],
        density_max=diagnostics['density_max'],
        density_fluctuation=diagnostics['density_fluctuation'],
        energy_ratio=diagnostics['energy_ratio'],
        lattice_speed=diagnostics['lattice_speed'],
        instance_file=instance_file.resolve(),
    )


def write_csv(results):
    rows = (dict(choice=row.choice, s_plus=row.s_plus, n=row.n, h=row.h,
                 actual_time=row.actual_time, status=row.status, stop_reason=row.stop_reason,
                 e_u1_l2=row.component_error[0], e_u2_l2=row.component_error[1],
                 e_u3_l2=row.component_error[2], e_u_l2=row.velocity_error,
                 d2=row.divergence_l2,
                 mass_drift=row.mass_drift, density_min=row.density_min,
                 density_max=row.density_max, density_fluctuation=row.density_fluctuation,
                 energy_ratio=row.energy_ratio, m_h=row.lattice_speed) for row in results)
    return write_records(DATA_FILE, rows)


from _csv_io import write_records
from _paths import DATA_CSV_DIR


def main():
    results = []
    for choice, s_plus in RATE_CHOICES:
        for n in N_LIST:
            solver, instance_file = run_case(choice, s_plus, n)
            results.append(result_from_solver(choice, s_plus, n, solver, instance_file))
            del solver
            write_csv(results)


if __name__ == '__main__':
    configure_experiment()
    main()

