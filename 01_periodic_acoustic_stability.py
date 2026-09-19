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
import cmath
import hashlib
import math
import numpy as np
from _paths import CODE_DIR, DATA_CSV_DIR, INSTANCE_OUTPUT_DIR
from d2n5_periodic_acoustic import d2n5_periodic_acoustic
from _double_shear_diagnostics import collision_matrix


A = 0.2


ALPHA = 0.5


NU = 0.01


PRESSURE_AMPLITUDE = 1e-4


N_LIST = (12, 16, 24, 32, 48, 64, 96, 128)


TARGET_STEPS = 800


GAIN_LIMIT = 1000.0


FIT_WINDOW = (50, 250)


CHECKPOINT_INTERVAL = 200


S_MINUS = 2*A/(A+ALPHA*NU)


RATES = {'otrt': 2-S_MINUS, 'srt': S_MINUS}


PROTOCOL = 'periodic-acoustic-full-grid-gain-v1'


INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'PeriodicAcousticArticle'


PREFIX = '01_periodic_acoustic'


def source_hash():
    digest = hashlib.sha256()
    for filename in ('01_periodic_acoustic_stability.py', 'd2n5_periodic_acoustic.py',
                     'd2n5_taylorgreen.py', '_nproll.py', '_double_shear_diagnostics.py'):
        digest.update((CODE_DIR/filename).read_bytes())
    return digest.hexdigest()


def metadata(choice, n):
    return dict(protocol=PROTOCOL, choice=choice, n=n, a=A, alpha=ALPHA,
                nu=NU, h=1/n, s_plus=RATES[choice], s_minus=S_MINUS,
                pressure_amplitude=PRESSURE_AMPLITUDE, target_steps=TARGET_STEPS,
                gain_limit=GAIN_LIMIT, fit_window=list(FIT_WINDOW), source_hash=source_hash())


def rest_populations():
    rest = np.zeros((5, 3))
    rest[:, 0] = [A, A, A, A, 1-4*A]
    return rest


def equilibrium_jacobians():
    velocities = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])
    blocks = []
    for ex, ey in velocities:
        weight = A if ex or ey else 1-4*A
        block = weight*np.eye(3)
        block[0, 1:] += ALPHA/2*np.array([ex, ey])
        block[1:, 0] += ALPHA/2*np.array([ex, ey])
        blocks.append(block)
    return velocities, np.vstack(blocks)


def mode_analysis(choice):
    velocities, collision = collision_matrix(2, A, ALPHA, RATES[choice], S_MINUS)
    theta = np.array([math.pi/2, 0.0])
    phases = np.repeat(np.exp(-1j*(velocities@theta)), 3)
    amplification = phases[:, None]*collision
    if choice == 'srt':
        polynomial = [1, -(9-40j)/41, (273+1560j)/1681, -1521/1681]
        target = max(np.roots(polynomial), key=abs)
    else:
        b = 1-S_MINUS
        mu = (1+b)*(1-2*A)-1j*ALPHA*(1-b)
        roots = [(mu+sign*cmath.sqrt(mu*mu-4*b))/2 for sign in (1, -1)]
        target = next(z for z in roots if z.real > 0)
    values, right_vectors = np.linalg.eig(amplification)
    index = int(np.argmin(abs(values-target)))
    value = values[index]
    right = right_vectors[:, index]
    left_values, left_vectors = np.linalg.eig(amplification.T)
    left = left_vectors[:, int(np.argmin(abs(left_values-value)))]
    left = left/(left@right)
    _, equilibrium = equilibrium_jacobians()
    seed = equilibrium[:, 0]
    projection_ratio = float(abs(left@seed)*np.linalg.norm(right)/np.linalg.norm(seed))
    right_residual = float(np.linalg.norm(amplification@right-value*right))
    left_residual = float(np.linalg.norm(left@amplification-value*left)/np.linalg.norm(left))
    assert abs(value-target) < 1e-11
    assert right_residual < 1e-11 and left_residual < 1e-11
    assert projection_ratio > 1e-3
    return dict(amplification=amplification, value=value, right=right, left=left,
                projection_ratio=projection_ratio, right_residual=right_residual,
                left_residual=left_residual)


def mode_coefficient(solver):
    # A repeated phase is also used for measuring the same cell-centered mode.
    phase = np.tile(np.exp(-1j*math.pi/4)*np.array([1, -1j, -1, 1j]), solver.Nx//4)
    profile = solver.f.mean(axis=1)-rest_populations()
    return (np.tensordot(phase, profile, axes=(0, 0))/solver.Nx).ravel()


def diagnostics(solver, spectrum):
    rho = solver.w[..., 0]
    u = solver.w[..., 1]/(solver.h*rho)
    v = solver.w[..., 2]/(solver.h*rho)
    divergence = ((np.roll(u, -1, axis=0)-np.roll(u, 1, axis=0))
                  +(np.roll(v, -1, axis=1)-np.roll(v, 1, axis=1)))/(2*solver.h)
    vorticity = ((np.roll(v, -1, axis=0)-np.roll(v, 1, axis=0))
                 -(np.roll(u, -1, axis=1)-np.roll(u, 1, axis=1)))/(2*solver.h)
    speed2 = u*u+v*v
    coefficient = spectrum['left']@mode_coefficient(solver)
    return dict(step=solver.iter_count, time=float(solver.t),
                modal_real=float(coefficient.real), modal_imag=float(coefficient.imag),
                modal_gain=float(abs(coefficient)/solver.experiment_initial_modal_amplitude),
                divergence_l2=float(np.sqrt(np.mean(divergence**2))),
                density_fluctuation=float(np.max(abs(rho-rho.mean()))),
                density_min=float(rho.min()), density_max=float(rho.max()),
                lattice_speed=float(solver.h*np.sqrt(speed2.max())),
                energy=float(0.5*np.mean(speed2)), enstrophy=float(0.5*np.mean(vorticity**2)),
                mass_drift=float(abs(np.mean(rho)-solver.experiment_initial_mass)))


def threshold_reason(values):
    if not all(math.isfinite(value) for value in values.values()):
        return 'nonfinite'
    if values['density_min'] < 0.5 or values['density_max'] > 1.5:
        return 'density interval'
    if values['density_fluctuation'] > 0.05:
        return 'density fluctuation'
    if values['divergence_l2'] > 0.5:
        return 'D2 above 0.5'
    if values['lattice_speed'] > 0.25:
        return 'M_h above 0.25'
    if values['modal_gain'] >= GAIN_LIMIT:
        return 'modal gain above 1000'
    return ''


def save_cache(solver, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    solver.save_instance(path)


def run_case(choice, n, spectrum):
    path = INSTANCE_DIR/f'{choice}_N{n}.pkl'
    expected = metadata(choice, n)
    solver = None
    if reuse_instances() and path.exists():
        saved = d2n5_periodic_acoustic.load_instance(path)
        if saved.experiment_metadata == expected:
            solver = saved
        else:
            print(f'Cache parameters or source changed; recomputing {path.name}.', flush=True)
    if solver is None:
        solver = d2n5_periodic_acoustic(h=1/n, nu=NU, a=A, alpha=ALPHA,
                                      s_plus=RATES[choice], pressure_amplitude=PRESSURE_AMPLITUDE)
        solver.experiment_metadata = expected
        solver.experiment_status = 'running'
        solver.experiment_stop_reason = ''
        solver.experiment_initial_mass = float(solver.w[..., 0].mean())
        solver.experiment_initial_modal_amplitude = float(abs(spectrum['left']@mode_coefficient(solver)))
        assert solver.experiment_initial_modal_amplitude > 0
        solver.experiment_initial_coefficient = mode_coefficient(solver)
        solver.experiment_history = [diagnostics(solver, spectrum)]
        solver.experiment_first_step_residual = None
        solver.experiment_spectrum = spectrum

    if solver.experiment_status == 'running':
        with np.errstate(divide='raise', over='raise', invalid='raise', under='ignore'):
            while solver.iter_count < TARGET_STEPS:
                try:
                    solver.iter()
                    values = diagnostics(solver, spectrum)
                    solver.experiment_history.append(values)
                    if solver.iter_count == 1:
                        reference = spectrum['amplification']@solver.experiment_initial_coefficient
                        solver.experiment_first_step_residual = float(
                            np.linalg.norm(mode_coefficient(solver)-reference)/np.linalg.norm(reference))
                        # Initially u=0, so the first collision is exactly at equilibrium.
                        assert solver.experiment_first_step_residual < 1e-6
                    reason = threshold_reason(values)
                    if reason:
                        solver.experiment_status = 'amplified' if reason == 'modal gain above 1000' else 'threshold'
                        solver.experiment_stop_reason = reason
                        break
                    if solver.iter_count % CHECKPOINT_INTERVAL == 0:
                        save_cache(solver, path)
                except FloatingPointError as error:
                    solver.experiment_status = 'overflow'
                    solver.experiment_stop_reason = str(error)
                    break
        if solver.iter_count == TARGET_STEPS and solver.experiment_status == 'running':
            solver.experiment_status = 'completed'
        save_cache(solver, path)

    history = solver.experiment_history
    fit = [row for row in history if FIT_WINDOW[0] <= row['step'] <= FIT_WINDOW[1]]
    if len(fit) < 10:
        raise ValueError(f'Too few samples for the declared fit window: {choice}, N={n}')
    slope, intercept = np.polyfit([row['step'] for row in fit],
                                np.log([row['modal_gain'] for row in fit]), 1)
    result = dict(choice=choice, n=n, mode_x=n//4, **history[-1],
                  status=solver.experiment_status, stop_reason=solver.experiment_stop_reason,
                  rate_theory=float(abs(spectrum['value'])), rate_fit=float(np.exp(slope)),
                  fit_intercept=float(intercept), projection_ratio=spectrum['projection_ratio'],
                  initial_modal_amplitude=solver.experiment_initial_modal_amplitude,
                  first_step_relative_residual=solver.experiment_first_step_residual,
                  max_mass_drift=max(row['mass_drift'] for row in history),
                  max_density_fluctuation=max(row['density_fluctuation'] for row in history),
                  max_divergence_l2=max(row['divergence_l2'] for row in history),
                  max_lattice_speed=max(row['lattice_speed'] for row in history),
                  instance_file=str(path.relative_to(CODE_DIR)))
    print(f"{choice.upper():4s} N={n:3d}: {result['status']:9s}, step={result['step']:3d}, "
          f"gain={result['modal_gain']:.6e}, r_fit={result['rate_fit']:.10f}, "
          f"D2={result['divergence_l2']:.3e}", flush=True)
    return result, [dict(choice=choice, n=n, **row) for row in history]


from _csv_io import write_records
from _paths import DATA_CSV_DIR


def main():
    results, histories = [], []
    spectra = {choice: mode_analysis(choice) for choice in RATES}
    result_path = DATA_CSV_DIR/f'data_{PREFIX}_results.csv'
    history_path = DATA_CSV_DIR/f'data_{PREFIX}_histories.csv'
    for choice in RATES:
        for n in N_LIST:
            row, history = run_case(choice, n, spectra[choice])
            results.append(row)
            histories.extend(history)
            write_records(result_path, results)
    write_records(history_path, histories)
    mode_path = DATA_CSV_DIR/f'data_{PREFIX}_modes.csv'
    write_records(mode_path, (dict(choice=choice, eigenvalue_real=float(s['value'].real),
        eigenvalue_imag=float(s['value'].imag), modulus=float(abs(s['value'])),
        projection_ratio=s['projection_ratio'], right_residual=s['right_residual'],
        left_residual=s['left_residual']) for choice, s in spectra.items()))
    assert all(row['status'] == ('completed' if row['choice'] == 'otrt' else 'amplified') for row in results)
    assert max(abs(row['rate_fit']-row['rate_theory']) for row in results) < 1e-5


if __name__ == '__main__':
    configure_experiment()
    main()

