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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from _d3n7_fast import iter_zero_force
from _paths import DATA_CSV_DIR, INSTANCE_OUTPUT_DIR
from d3n7_ethiersteinmann import d3n7_ethiersteinmann as _EthierSteinman

_STABLE_MODULE_NAME = '07_ethier_steinman_convergence'


if __name__ == '__main__':
    sys.modules.setdefault(_STABLE_MODULE_NAME, sys.modules[__name__])


class d3n7_ethiersteinmann_l0(_EthierSteinman):
    '''Ethier--Steinman solver with ell=0 on every Dirichlet link.'''

    initialization_protocol = _EthierSteinman.initialization_protocol+'-ell0'

    @property
    def l(self):
        return 0.0


d3n7_ethiersteinmann_l0.__module__ = _STABLE_MODULE_NAME


d3n7_ethiersteinmann = d3n7_ethiersteinmann_l0


FLOW_A = math.pi/4


FLOW_D = math.pi/2


NU = 0.05


A = 1/7


ALPHA = 1/7


S_MINUS = 40/21


TARGET_TIME = 1/16


N_LIST = (16, 32, 64, 128)


RATE_CHOICES = (
    ('OTRT', 2/21),
    ('SRT', 40/21),
)


DENSITY_INTERVAL = (0.5, 1.5)


ENERGY_RATIO_LIMIT = 1.25


DENSITY_FLUCTUATION_LIMIT = 0.05


DIVERGENCE_LIMIT = 0.5


LATTICE_SPEED_LIMIT = 0.25


INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'EthierSteinmannArticle_l0'


DATA_FILE = DATA_CSV_DIR/'data_07_ethier_steinman_results.csv'


EXPERIMENT_PROTOCOL = 'ethier-steinmann-dirichlet-ell0-appendix-c-time-n-v2'


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
    return int(math.ceil(TARGET_TIME/solver.dt-1e-14))


def dirichlet_divergence(solver, velocity):
    '''Second-order divergence without periodic wrap-around at cube faces.'''
    u, v, w = velocity
    return (
        np.gradient(u, solver.h, axis=0, edge_order=2)
        + np.gradient(v, solver.h, axis=1, edge_order=2)
        + np.gradient(w, solver.h, axis=2, edge_order=2)
    )


def monitoring_diagnostics(solver, initial_mass, initial_energy):
    rho = solver.w[..., 0]
    velocity = solver.get_numerical_speed()
    speed_squared = sum(component**2 for component in velocity)
    cell_volume = solver.h**3
    mass = float(cell_volume*np.sum(rho))
    energy = float(0.5*cell_volume*np.sum(speed_squared))
    divergence = dirichlet_divergence(solver, velocity)
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


def threshold_reason(diagnostics):
    if diagnostics['density_min'] < DENSITY_INTERVAL[0]:
        return 'density below 0.5'
    if diagnostics['density_max'] > DENSITY_INTERVAL[1]:
        return 'density above 1.5'
    if diagnostics['energy_ratio'] > ENERGY_RATIO_LIMIT:
        return 'energy ratio above 1.25'
    if diagnostics['density_fluctuation'] > DENSITY_FLUCTUATION_LIMIT:
        return 'density fluctuation above 0.05'
    if diagnostics['divergence_l2'] > DIVERGENCE_LIMIT:
        return 'D2 above 0.5'
    if diagnostics['lattice_speed'] > LATTICE_SPEED_LIMIT:
        return 'M_h above 0.25'
    return ''


def cache_file(choice, n):
    tag = choice.lower().replace(' ', '_').replace('.', 'p')
    return INSTANCE_DIR/f'{tag}_N{n}.pkl'


def instance_matches(solver, expected, target_steps):
    if type(solver) is not type(expected):
        return False
    for name in (
        'h', 'nu', 'a', 'alpha', 's_plus', 'flow_a', 'flow_d',
    ):
        if not np.isclose(
            getattr(solver, name, np.nan), getattr(expected, name),
            rtol=0, atol=1e-14,
        ):
            return False
    if getattr(solver, 'experiment_protocol', None) != EXPERIMENT_PROTOCOL:
        return False
    if (
        getattr(solver, 'initialization_protocol', None)
        != getattr(expected, 'initialization_protocol', None)
    ):
        return False
    if getattr(solver, 'requested_target_steps', None) != target_steps:
        return False
    status = getattr(solver, 'experiment_status', None)
    return (
        (status == 'completed' and solver.iter_count == target_steps)
        or (status in {'threshold', 'overflow'} and solver.iter_count <= target_steps)
    )


def initialize_experiment_metadata(solver, target_steps):
    rho = solver.w[..., 0]
    velocity = solver.get_numerical_speed()
    speed_squared = sum(component**2 for component in velocity)
    solver.experiment_protocol = EXPERIMENT_PROTOCOL
    solver.requested_target_time = TARGET_TIME
    solver.requested_target_steps = target_steps
    solver.experiment_initial_mass = float(solver.h**3*np.sum(rho))
    solver.experiment_initial_energy = float(
        0.5*solver.h**3*np.sum(speed_squared)
    )
    solver.experiment_status = 'running'
    solver.experiment_stop_reason = ''


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


def advance_with_thresholds(solver, target_steps, instance_file):
    update_interval = max(1, target_steps//20)
    monitor_interval = max(1, target_steps//400)
    solver.experiment_monitor_interval = monitor_interval
    try:
        with np.errstate(
            divide='raise', over='raise', invalid='raise', under='ignore'
        ):
            while solver.iter_count < target_steps:
                iter_zero_force(solver)
                should_monitor = (
                    solver.iter_count % monitor_interval == 0
                    or solver.iter_count == target_steps
                )
                if should_monitor:
                    diagnostics = monitoring_diagnostics(
                        solver,
                        solver.experiment_initial_mass,
                        solver.experiment_initial_energy,
                    )
                    if not all(np.isfinite(tuple(diagnostics.values()))):
                        solver.experiment_status = 'overflow'
                        solver.experiment_stop_reason = 'non-finite diagnostic'
                        solver.overflowed = True
                        break
                    reason = threshold_reason(diagnostics)
                    if reason:
                        solver.experiment_status = 'threshold'
                        solver.experiment_stop_reason = reason
                        break
                if should_monitor and (
                    solver.iter_count % update_interval < monitor_interval
                    or solver.iter_count == target_steps
                ):
                    print(
                        f'  step {solver.iter_count}/{target_steps}, '
                        f't={solver.t:.9f}, D2={diagnostics["divergence_l2"]:.3e}'
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
    solver = d3n7_ethiersteinmann(
        h=1/n,
        nu=NU,
        a=A,
        alpha=ALPHA,
        s_plus=s_plus,
        flow_a=FLOW_A,
        flow_d=FLOW_D,
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

    initial = monitoring_diagnostics(
        solver,
        float(solver.h**3*np.sum(solver.w[..., 0])),
        float(0.5*solver.h**3*np.sum(sum(
            component**2 for component in solver.get_numerical_speed()
        ))),
    )
    initial_reason = threshold_reason(initial)
    if initial_reason:
        raise RuntimeError(
            f'Invalid initial state for {choice}, N={n}: {initial_reason}.'
        )

    target_steps = target_step_count(solver)
    instance_file = cache_file(choice, n)
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    if reuse_instances() and instance_file.is_file():
        try:
            saved = d3n7_ethiersteinmann.load_instance(instance_file)
            if instance_matches(saved, solver, target_steps):
                if instance_file.stat().st_size > 512*1024**2:
                    save_compact_instance(saved, instance_file)
                    print(f'Compacted saved instance: {instance_file.resolve()}')
                print(f'Reusing Ethier--Steinman result: {instance_file.resolve()}')
                return saved, instance_file
            # A fine-grid saved solver can occupy hundreds of MiB.  Release
            # an incompatible protocol instance before allocating work arrays
            # for the replacement run.
            del saved
        except Exception as error:
            print(f'Ignoring unusable saved instance {instance_file}: {error}')

    initialize_experiment_metadata(solver, target_steps)
    print(
        f'Running Ethier--Steinman {choice}, N={n}, steps={target_steps}, '
        f't_end={target_steps*solver.dt:.12g}, s_plus={s_plus:.12g}.'
    )
    advance_with_thresholds(solver, target_steps, instance_file)
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

