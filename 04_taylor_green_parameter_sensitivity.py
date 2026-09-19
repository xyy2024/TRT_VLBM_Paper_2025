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
from _paths import configure_experiment, reuse_instances


from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import math
import os
from pathlib import Path
from _paths import DATA_CSV_DIR, INSTANCE_OUTPUT_DIR
import numpy as np
from d2n5_taylorgreen import d2n5_taylorgreen


U0 = 1.0


K1 = 2*math.pi


K2 = 2*math.pi


B1 = -0.5*math.pi


B2 = -0.5*math.pi


N_COARSE = 96


N_FINE = 128


FINAL_TIME = Fraction(1, 20)


NU_FRACTION = Fraction(1, 10)


NU = float(NU_FRACTION)


ALPHA = Fraction(1, 5)


ANOMALOUS_ERROR_LIMIT = 2e-3


ANOMALOUS_DIVERGENCE_LIMIT = 3e-4


A_VALUES = (
    (Fraction(3, 25), '0.12'),
    (Fraction(4, 25), '0.16'),
    (Fraction(1, 5), '0.20'),
    (Fraction(6, 25), '0.24'),
)


FIXED_RATE_CASES = (
    ('interior', Fraction(1, 20), '1_20'),
    ('interior', Fraction(1, 10), '1_10'),
    ('interior', Fraction(3, 20), '3_20'),
    ('interior', Fraction(1, 5), '1_5'),
    ('interior', Fraction(1, 4), '1_4'),
    ('interior', Fraction(3, 10), '3_10'),
    ('interior', Fraction(7, 20), '7_20'),
    ('interior', Fraction(2, 5), '2_5'),
    ('interior', Fraction(9, 20), '9_20'),
    ('interior', Fraction(1, 2), '1_2'),
    ('interior', Fraction(11, 20), '11_20'),
    ('interior', Fraction(3, 5), '3_5'),
    ('interior', Fraction(13, 20), '13_20'),
    ('interior', Fraction(7, 10), '7_10'),
    ('interior', Fraction(3, 4), '3_4'),
    ('interior', Fraction(4, 5), '4_5'),
    ('interior', Fraction(17, 20), '17_20'),
    ('interior', Fraction(9, 10), '9_10'),
    ('interior', Fraction(19, 20), '19_20'),
    ('interior', Fraction(1, 1), '1'),
    ('interior', Fraction(21, 20), '21_20'),
    ('interior', Fraction(11, 10), '11_10'),
    ('interior', Fraction(23, 20), '23_20'),
    ('interior', Fraction(6, 5), '6_5'),
    ('interior', Fraction(5, 4), '5_4'),
    ('interior', Fraction(13, 10), '13_10'),
    ('interior', Fraction(27, 20), '27_20'),
    ('interior', Fraction(7, 5), '7_5'),
    ('interior', Fraction(29, 20), '29_20'),
    ('interior', Fraction(3, 2), '3_2'),
    ('interior', Fraction(31, 20), '31_20'),
    ('interior', Fraction(8, 5), '8_5'),
    ('interior', Fraction(33, 20), '33_20'),
    ('interior', Fraction(17, 10), '17_10'),
    ('interior', Fraction(7, 4), '7_4'),
    ('interior', Fraction(9, 5), '9_5'),
    ('interior', Fraction(37, 20), '37_20'),
    ('interior', Fraction(19, 10), '19_10'),
    ('interior', Fraction(39, 20), '39_20'),
    ('interior', Fraction(2, 1), '2'),
)


INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'TaylorGreenASPlusSweep_T005'


DATA_FILE_NAME = 'data_04_taylor_green_parameter_scan.csv'


@dataclass(frozen=True)
class GridResult:
    n: int
    h: float
    error: float | None
    d2: float | None
    completed: bool
    outlier: bool
    instance_file: Path


def s_minus_for(a):
    '''Return the link-odd rate imposed by nu, fixed alpha and a.'''
    return Fraction(2, 1)/(1+NU_FRACTION*ALPHA/a)


def rate_cases(a):
    '''Return unique requested s_plus choices for one a value.'''
    s_minus = s_minus_for(a)
    otrt = ('OTRT', 2-s_minus, 'otrt')
    srt = ('SRT', s_minus, 'srt')

    special_rates = {otrt[1], srt[1]}
    cases = (
        otrt,
        *(case for case in FIXED_RATE_CASES if case[1] not in special_rates),
        srt,
    )
    rates = [case[1] for case in cases]
    keys = [case[2] for case in cases]
    if len(rates) != len(set(rates)) or len(keys) != len(set(keys)):
        raise RuntimeError(f'Duplicate rate case for a={a}: {cases!r}')
    return cases


def validate_fixed_rate_cases():
    '''Reject duplicate or mistyped fixed-rate definitions before any run.'''
    rates = []
    keys = []
    for _, rate, rate_key in FIXED_RATE_CASES:
        expected_key = (
            str(rate.numerator)
            if rate.denominator == 1
            else f'{rate.numerator}_{rate.denominator}'
        )
        if rate_key != expected_key:
            raise RuntimeError(
                f'Rate key {rate_key!r} does not match s_plus={rate}; '
                f'expected {expected_key!r}.'
            )
        rates.append(rate)
        keys.append(rate_key)
    if len(rates) != len(set(rates)):
        raise RuntimeError('FIXED_RATE_CASES contains duplicate s_plus values.')
    if len(keys) != len(set(keys)):
        raise RuntimeError('FIXED_RATE_CASES contains duplicate cache keys.')


validate_fixed_rate_cases()


def target_step_count(n):
    steps = FINAL_TIME*n**2/ALPHA
    if steps.denominator != 1:
        raise RuntimeError(f'T={FINAL_TIME} is not an integer step for N={n}.')
    return steps.numerator


def build_expected_solver(n, s_plus, a):
    solver = d2n5_taylorgreen(
        h=1/n,
        nu=NU,
        U0=U0,
        k1=K1,
        k2=K2,
        b1=B1,
        b2=B2,
        a=float(a),
        alpha=float(ALPHA),
        s_plus=float(s_plus),
    )
    return solver


def solver_configuration_matches(solver, expected_solver):
    '''Check static parameters and boundaries without requiring the same time.'''
    if type(solver) is not type(expected_solver):
        return False
    scalar_parameters = (
        'h', 'dx', 'dt', 'nu', 'a', 'alpha', 's_plus', 'tau',
        'relax1', 'relax2', 'xshift', 'U0', 'k1', 'k2', 'b1', 'b2',
    )
    for name in scalar_parameters:
        saved_value = getattr(solver, name, None)
        expected_value = getattr(expected_solver, name, None)
        if (
            saved_value is None
            or expected_value is None
            or not np.isclose(saved_value, expected_value, rtol=0, atol=1e-14)
        ):
            return False
    return (
        solver.ND == expected_solver.ND
        and solver.NV == expected_solver.NV
        and solver.NE == expected_solver.NE
        and solver.Ex == expected_solver.Ex
        and solver.opp == expected_solver.opp
        and solver.shape == expected_solver.shape
        and solver.outerforce_type == expected_solver.outerforce_type
        and solver.border_type == expected_solver.border_type
        and solver.border_geometry_cache_key()
            == expected_solver.border_geometry_cache_key()
    )


def load_reusable_instance(instance_file, expected_solver, target_step):
    '''Load a matching cached instance no later than the requested step.'''
    if not reuse_instances() or not instance_file.is_file():
        return None
    try:
        solver = d2n5_taylorgreen.load_instance(instance_file)
    except Exception as error:
        print(f'Ignoring unreadable saved instance {instance_file}: {error}')
        return None
    if not solver_configuration_matches(solver, expected_solver):
        print(f'Saved instance has different parameters: {instance_file}')
        return None
    if solver.iter_count < 0 or solver.iter_count > target_step:
        print(f'Saved instance is later than the requested time: {instance_file}')
        return None
    print(
        f'Using saved instance for N={solver.Nx}, t={solver.t:.12g}: '
        f'{instance_file.resolve()}'
    )
    return solver


def periodic_diagnostics(solver):
    '''Return the centered-periodic divergence norm and lattice speed.'''
    u, v = solver.get_numerical_speed()
    divergence = (
        (np.roll(u, -1, axis=0)-np.roll(u, 1, axis=0))/(2*solver.h)
        + (np.roll(v, -1, axis=1)-np.roll(v, 1, axis=1))/(2*solver.h)
    )
    d2 = np.sqrt(solver.h**2*np.sum(divergence**2))
    m_h = solver.h*np.max(np.sqrt(u**2+v**2))
    return d2, m_h


def validate_solver_before_iteration(solver, expected_solver, n, s_plus, a,
                                     target_step):
    '''Validate every model parameter and relaxation rate before advancing.'''
    s_minus = float(s_minus_for(a))
    expected_tau = 1/s_minus
    expected_phi = 0.5*(float(s_plus) + s_minus)
    expected_psi = 0.5*(float(s_plus) - s_minus)
    expected_scalars = {
        'h': 1/n,
        'dx': 1/n,
        'dt': float(ALPHA)/n**2,
        'nu': NU,
        'a': float(a),
        'alpha': float(ALPHA),
        's_plus': float(s_plus),
        'tau': expected_tau,
        'relax1': expected_phi,
        'relax2': expected_psi,
        'xshift': expected_solver.xshift,
        'U0': U0,
        'k1': K1,
        'k2': K2,
        'b1': B1,
        'b2': B2,
    }
    mismatches = []
    for name, expected_value in expected_scalars.items():
        actual_value = getattr(solver, name, None)
        if (
            actual_value is None
            or not np.isclose(
                actual_value, expected_value, rtol=0, atol=1e-14
            )
        ):
            mismatches.append(
                f'{name}={actual_value!r} (expected {expected_value!r})'
            )

    actual_s_minus = solver.relax1-solver.relax2
    reconstructed_nu = (
        2*solver.a/solver.alpha*(1/actual_s_minus-0.5)
    )
    if not np.isclose(actual_s_minus, s_minus, rtol=0, atol=1e-14):
        mismatches.append(
            f's_minus={actual_s_minus!r} (expected {s_minus!r})'
        )
    if not np.isclose(reconstructed_nu, NU, rtol=0, atol=1e-14):
        mismatches.append(
            f'nu_from_s_minus={reconstructed_nu!r} (expected {NU!r})'
        )
    if not (0 < solver.s_plus <= 2 and 0 < actual_s_minus < 2):
        mismatches.append(
            f'relaxation rates outside admissible interval: '
            f's_plus={solver.s_plus!r}, s_minus={actual_s_minus!r}'
        )
    if solver.Nx != n or solver.Ny != n:
        mismatches.append(
            f'grid shape={(solver.Nx, solver.Ny)!r} (expected {(n, n)!r})'
        )
    if solver.iter_count < 0 or solver.iter_count > target_step:
        mismatches.append(
            f'iter_count={solver.iter_count!r} '
            f'(expected 0 <= iter_count <= {target_step})'
        )
    if not np.isclose(
        target_step*solver.dt, float(FINAL_TIME), rtol=0, atol=1e-14
    ):
        mismatches.append(
            f'target_time={target_step*solver.dt!r} '
            f'(expected {float(FINAL_TIME)!r})'
        )
    if solver.__dict__.get('initialization_protocol') != (
        expected_solver.initialization_protocol
    ):
        mismatches.append('outdated equilibrium-initialization protocol')
    if not np.all(solver.in_border):
        mismatches.append('boundary is not fully periodic')
    if np.any(solver._get_outerforce_array() != 0):
        mismatches.append('Taylor--Green body force is not zero')

    if solver.iter_count == 0:
        population_sum_error = np.max(
            np.abs(solver.f.sum(axis=2)-solver.w)
        )
        numerical_velocity = solver.get_numerical_speed()
        exact_velocity = solver.exact()
        initial_velocity_error = max(
            np.max(np.abs(numerical-exact))
            for numerical, exact in zip(numerical_velocity, exact_velocity)
        )
        expected_density = 1 + solver.h**2*solver.exact_pressure()
        if population_sum_error > 2e-15:
            mismatches.append(
                f'initial population-sum error={population_sum_error!r}'
            )
        if initial_velocity_error > 2e-15:
            mismatches.append(
                f'initial velocity error={initial_velocity_error!r}'
            )
        if not np.array_equal(solver.w[..., 0], expected_density):
            mismatches.append('initial density does not match the prescribed equilibrium')

    if mismatches:
        detail = '; '.join(mismatches)
        raise RuntimeError(
            f'Invalid pre-iteration configuration for a={float(a):.12g}, '
            f's_plus={float(s_plus):.12g}, N={n}: {detail}'
        )

    print(
        f'Parameter check N={n}: a={solver.a:.12g}, '
        f'alpha={solver.alpha:.12g}, nu={solver.nu:.12g}, '
        f'dt={solver.dt:.12g}, s_plus={solver.s_plus:.12g}, '
        f's_minus={actual_s_minus:.12g}, tau={solver.tau:.12g}, '
        f'phi={solver.relax1:.12g}, psi={solver.relax2:.12g}; passed.'
    )


def instance_file_for(a, rate_key, n):
    a_key = f'{a.numerator}_{a.denominator}'
    file_stem = rate_key if rate_key in {'otrt', 'srt'} else f'sp{rate_key}'
    return INSTANCE_DIR/f'a{a_key}'/f'{file_stem}_N{n}.pkl'


def save_snapshot(solver, instance_file, reason):
    instance_file.parent.mkdir(parents=True, exist_ok=True)
    solver.save_instance(instance_file)
    print(
        f'Saved {reason} solver state at step {solver.iter_count}, '
        f't={solver.t:.12g}: {instance_file.resolve()}'
    )


def run_case(rate_key, s_plus, n, a):
    '''Run one grid case, saving and skipping rather than aborting on overflow.'''
    expected_solver = build_expected_solver(n, s_plus, a)
    target_step = target_step_count(n)
    instance_file = instance_file_for(a, rate_key, n)
    solver = load_reusable_instance(
        instance_file, expected_solver, target_step
    )
    if (
        solver is not None
        and solver.__dict__.get('initialization_protocol')
            != expected_solver.initialization_protocol
    ):
        print(
            'Ignoring saved instance without the current equilibrium '
            f'initialization protocol: {instance_file.resolve()}'
        )
        solver = None
    if solver is None:
        solver = expected_solver
    validate_solver_before_iteration(
        solver, expected_solver, n, s_plus, a, target_step
    )
    if getattr(solver, 'overflowed', False):
        print(
            f'Skipping previously overflowed case at step {solver.iter_count}: '
            f'{instance_file.resolve()}'
        )
        return GridResult(
            n, solver.h, None, None, False, False, instance_file
        )

    remaining_steps = target_step-solver.iter_count
    if remaining_steps:
        print(
            f'Running a={float(a):.12g}, s_plus={float(s_plus):.12g}, '
            f'N={n}: {remaining_steps} additional steps.'
        )
        numerical_error = None
        try:
            completed = solver.until_step(remaining_steps, show_progress=False)
        except (FloatingPointError, OverflowError) as error:
            solver.overflowed = True
            completed = False
            numerical_error = error
        if not completed or solver.iter_count != target_step:
            save_snapshot(solver, instance_file, 'partial/overflowed')
            detail = '' if numerical_error is None else f' ({numerical_error})'
            print(
                f'WARNING: case stopped before its target{detail}; '
                'continuing with the next grid case.'
            )
            return GridResult(
                n, solver.h, None, None, False, False, instance_file
            )
        save_snapshot(solver, instance_file, 'completed')

    velocity_error = float(solver.get_error()[1])
    d2 = float(periodic_diagnostics(solver)[0])
    if not np.isfinite(velocity_error) or not np.isfinite(d2):
        solver.overflowed = True
        save_snapshot(solver, instance_file, 'non-finite')
        return GridResult(
            n, solver.h, None, None, False, False, instance_file
        )
    if (
        velocity_error > ANOMALOUS_ERROR_LIMIT
        or d2 > ANOMALOUS_DIVERGENCE_LIMIT
    ):
        print(
            f'WARNING: anomalous E_u2={velocity_error:.12g}, D2={d2:.12g}; '
            'marking this grid result as an outlier in Table 4.'
        )
        return GridResult(
            n, solver.h, velocity_error, d2, True, True, instance_file
        )
    return GridResult(
        n, solver.h, velocity_error, d2, True, False, instance_file
    )


def grid_status_label(grid_result):
    '''Return the manuscript-facing status label for one grid result.'''
    if not grid_result.completed:
        return 'failure'
    if grid_result.outlier:
        return 'outlier'
    return 'regular'


from _csv_io import write_records
from _paths import DATA_CSV_DIR


def run_pair(job):
    """Independent rate pair; only the parent writes the aggregate CSV."""
    a, rate_choice, s_plus, rate_key = job
    rows = []
    for n in (N_COARSE, N_FINE):
        result = run_case(rate_key, s_plus, n, a)
        rows.append(dict(rate_choice=rate_choice, rate_key=rate_key,
            a=float(a), a_exact=str(a), alpha=float(ALPHA), nu=NU,
            s_minus=float(s_minus_for(a)), s_plus=float(s_plus), s_plus_exact=str(s_plus),
            n=n, h=result.h, target_time=float(FINAL_TIME),
            status=grid_status_label(result), completed=result.completed,
            outlier=result.outlier, e_u2=result.error, d2=result.d2))
    return rows


def main():
    rows = []
    path = DATA_CSV_DIR/DATA_FILE_NAME
    jobs = [(a, *case) for a, _ in A_VALUES for case in rate_cases(a)]
    workers = int(os.environ.get('TRTVLBM_SCAN_WORKERS', min(4, os.cpu_count() or 1)))
    if workers < 1:
        raise ValueError('TRTVLBM_SCAN_WORKERS must be positive.')
    if workers == 1:
        for pair in map(run_pair, jobs):
            rows.extend(pair)
            write_records(path, rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for pair in executor.map(run_pair, jobs):
                rows.extend(pair)
                write_records(path, rows)


if __name__ == '__main__':
    configure_experiment()
    main()

