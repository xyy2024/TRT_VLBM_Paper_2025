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


import numpy as np
from scipy.optimize import root
from _paths import DATA_CSV_DIR, INSTANCE_OUTPUT_DIR
from d2n5_poiseuille import d2n5_poiseuille_article


NU = 0.1


A = 0.2


ALPHA = 0.2


G = 0.08


GAMMA = 0.5


S_MINUS = 20/11


N_LIST = (16, 24, 32, 48, 64, 96, 128)


ELL_VALUES = (0.0,)


INSTANCE_DIR = INSTANCE_OUTPUT_DIR/'PoiseuilleArticle_l0'


def tuned_rate(ell):
    return 1/(
        GAMMA-ell + S_MINUS*GAMMA**2/(2-S_MINUS)
    )


def rate_choices(ell):
    tuned = tuned_rate(ell)
    return (
        ('OTRT', 2-S_MINUS),
        ('0.8 tuned', 0.8*tuned),
        ('tuned', tuned),
        ('1.2 tuned', 1.2*tuned),
        ('SRT', S_MINUS),
    )


def reconstruct_invariant_state(solver, tangential_populations):
    tangential = np.asarray(tangential_populations).reshape(
        solver.Nx, solver.Ny, solver.NE
    )
    w = np.zeros(solver.shapew)
    w[..., 0] = 1
    w[..., 1] = np.sum(tangential, axis=2)
    f = solver.get_m(w).copy()
    f[..., 1] = tangential
    return f, w


def fixed_point_residual(tangential_populations, solver):
    solver.f, solver.w = reconstruct_invariant_state(
        solver, tangential_populations
    )
    previous = solver.f[..., 1].copy()
    solver.iter()
    return (solver.f[..., 1]-previous).ravel()


def component_fixed_point_residual(
    populations, solver, component, target_w, tangential
):
    solver.f = solver.get_m(target_w).copy()
    solver.f[..., 1] = tangential
    solver.f[..., component] = np.asarray(populations).reshape(
        solver.Nx, solver.Ny, solver.NE
    )
    solver.w = target_w.copy()
    previous = solver.f[..., component].copy()
    solver.iter()
    return (solver.f[..., component]-previous).ravel()


def instance_matches(solver, expected):
    if type(solver) is not type(expected):
        return False
    for name in ('h', 'nu', 'a', 'alpha', 's_plus', 'ell', 'G'):
        if not np.isclose(
            getattr(solver, name, np.nan), getattr(expected, name),
            rtol=0, atol=1e-14,
        ):
            return False
    residual = getattr(solver, 'fixed_point_full_residual', np.inf)
    return (
        getattr(solver, 'fixed_point_reconstruction_protocol', None)
        == 'all-components-v1'
        and np.isfinite(residual)
        and residual <= 1e-14
    )


def cache_file(n, ell, s_plus):
    ell_tag = str(ell).replace('.', 'p')
    rate_tag = f'{s_plus:.12g}'.replace('.', 'p')
    return INSTANCE_DIR/f'ell{ell_tag}_splus{rate_tag}_N{n}.pkl'


def solve_case(n, ell, s_plus):
    solver = d2n5_poiseuille_article(
        h=1/n, nu=NU, a=A, alpha=ALPHA, s_plus=s_plus, G=G, ell=ell,
    )
    calculated_s_minus = solver.relax1-solver.relax2
    if not np.isclose(calculated_s_minus, S_MINUS, rtol=0, atol=1e-14):
        raise RuntimeError(
            f'Unexpected s_minus={calculated_s_minus}; expected {S_MINUS}.'
        )
    if not np.allclose(solver.gamma[solver.border_index], GAMMA, atol=1e-13):
        raise RuntimeError('The first fluid nodes are not gamma=1/2 from the walls.')

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    instance_file = cache_file(n, ell, s_plus)
    if reuse_instances() and instance_file.is_file():
        try:
            saved = d2n5_poiseuille_article.load_instance(instance_file)
            if instance_matches(saved, solver):
                print(f'Reusing Poiseuille fixed point: {instance_file.resolve()}')
                return saved
        except Exception as error:
            print(f'Ignoring unusable saved instance {instance_file}: {error}')

    u_exact, _ = solver.exact()
    initial_w = np.zeros(solver.shapew)
    initial_w[..., 0] = 1
    initial_w[..., 1] = solver.h*u_exact
    initial = solver.get_m(initial_w)[..., 1].ravel()
    solution = root(
        fixed_point_residual,
        initial,
        args=(solver,),
        method='hybr',
        options={'xtol': 1e-11},
    )
    residual = float(np.max(np.abs(solution.fun)))
    if not solution.success or residual > 1e-12:
        raise RuntimeError(
            f'Poiseuille fixed point failed for N={n}, ell={ell}, '
            f's_plus={s_plus}: {solution.message}; residual={residual:.3e}.'
        )

    tangential = solution.x.reshape(solver.Nx, solver.Ny, solver.NE)
    _, target_w = reconstruct_invariant_state(solver, solution.x)
    equilibrium = solver.get_m(target_w)
    component_solutions = {}
    component_evaluations = 0
    for component in (0, 2):
        component_solution = root(
            component_fixed_point_residual,
            equilibrium[..., component].ravel(),
            args=(solver, component, target_w, tangential),
            method='hybr',
            options={'xtol': 1e-11},
        )
        component_residual = float(np.max(np.abs(component_solution.fun)))
        if not component_solution.success or component_residual > 1e-12:
            raise RuntimeError(
                f'Poiseuille component {component} reconstruction failed for '
                f'N={n}, ell={ell}, s_plus={s_plus}: '
                f'{component_solution.message}; residual={component_residual:.3e}.'
            )
        component_solutions[component] = component_solution.x.reshape(
            solver.Nx, solver.Ny, solver.NE
        )
        component_evaluations += int(component_solution.nfev)

    solver.f = equilibrium.copy()
    solver.f[..., 0] = component_solutions[0]
    solver.f[..., 1] = tangential
    solver.f[..., 2] = component_solutions[2]
    np.sum(solver.f, axis=solver.ND, out=solver.w)
    if not np.allclose(solver.w, target_w, rtol=0, atol=5e-14):
        raise RuntimeError('Reconstructed populations left the invariant subspace.')
    previous = solver.f.copy()
    solver.iter()
    solver.fixed_point_full_residual = float(np.max(np.abs(solver.f-previous)))
    solver.fixed_point_function_evaluations = (
        int(solution.nfev)+component_evaluations
    )
    solver.fixed_point_reconstruction_protocol = 'all-components-v1'
    solver.iter_count = 0
    if solver.fixed_point_full_residual > 1e-12:
        raise RuntimeError('The reconstructed full state is not a fixed point.')
    solver.save_instance(instance_file)
    print(
        f'Solved Poiseuille N={n}, ell={ell:g}, s_plus={s_plus:.12g}; '
        f'full residual={solver.fixed_point_full_residual:.3e}.'
    )
    return solver


def fitted_slip(solver):
    u_numerical, _ = solver.get_numerical_speed()
    rhs = u_numerical[0] + G*solver.y**2/(2*NU)
    design = np.column_stack((np.ones(solver.Ny), solver.y))
    intercept, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    return float(intercept)


def predicted_slip(solver):
    return (
        ALPHA*solver.h**2*G/(2*A)
        * (
            solver.ell-GAMMA + 1/solver.s_plus
            - S_MINUS*GAMMA**2/(2-S_MINUS)
        )
    )


from _csv_io import write_records
from _paths import DATA_CSV_DIR


def main():
    rows, profiles = [], []
    path = DATA_CSV_DIR/'data_05_poiseuille_boundary_tuning.csv'
    profile_path = DATA_CSV_DIR/'data_05_poiseuille_profiles.csv'
    plot_n = 32 if 32 in N_LIST else max(N_LIST)
    for ell in ELL_VALUES:
        for choice, s_plus in rate_choices(ell):
            for n in N_LIST:
                solver = solve_case(n, ell, s_plus)
                rows.append(dict(ell=ell, gamma=GAMMA, choice=choice, s_plus=s_plus,
                    s_minus=S_MINUS, nu=NU, a=A, alpha=ALPHA, force=G, n=n, h=solver.h,
                    velocity_error=solver.get_error_channel(), fitted_slip=fitted_slip(solver),
                    predicted_slip=predicted_slip(solver),
                    fixed_point_full_residual=solver.fixed_point_full_residual))
                if n == plot_n:
                    numerical, exact = solver.get_numerical_speed()[0][0], solver.exact()[0][0]
                    profiles.extend(dict(choice=choice, ell=ell, s_plus=s_plus, n=n,
                                         h=solver.h, y=y, u_numerical=u, u_exact=v)
                                    for y, u, v in zip(solver.y, numerical, exact))
    write_records(path, rows)
    write_records(profile_path, profiles)


if __name__ == '__main__':
    configure_experiment()
    main()
