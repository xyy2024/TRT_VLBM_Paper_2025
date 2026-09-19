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

"""Recompute the published experiments or render their supplied CSV data."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
from _paths import CODE_DIR, FIGURE_OUTPUT_DIR, TABLE_OUTPUT_DIR, artifact_output_directory

EXPERIMENTS = {
    '01': '01_periodic_acoustic_stability.py',
    '02': '02_nonlinear_manufactured_convergence.py',
    '03': '03_taylor_green_convergence.py',
    '04': '04_taylor_green_parameter_sensitivity.py',
    '05': '05_poiseuille_boundary_tuning.py',
    '06': '06_beltrami_convergence.py',
    '07': '07_ethier_steinman_convergence.py',
    '08': '08_double_shear_dynamics.py',
    '09': '09_kolmogorov_startup_refinement.py',
}


def run_script(path, env, options=()):
    print(f'Running {path.relative_to(CODE_DIR)}', flush=True)
    subprocess.run([sys.executable, '-B', str(path), *options], env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    experiments = commands.add_parser('experiments', help='Recompute experiments and save CSV and solver states.')
    experiments.add_argument('--only', nargs='+', choices=EXPERIMENTS,
                             help='Experiment IDs; default: all nine in manuscript order.')
    experiments.add_argument('--rerun', action='store_true', help='Start afresh instead of reusing compatible saved states.')
    experiments.add_argument('--resume-from', type=Path,
                             help='Search this saved-state directory for experiment 02 or 03; select one with --only.')
    for action in ('render', 'figures', 'tables'):
        command = commands.add_parser(action, help='Generate figures and/or tables from CSV, without simulation.')
        command.add_argument('--only', nargs='+', help='Generator stems or filenames.')
        if action != 'tables':
            command.add_argument('--figure-output-dir', type=artifact_output_directory, default=FIGURE_OUTPUT_DIR)
        if action != 'figures':
            command.add_argument('--table-output-dir', type=artifact_output_directory, default=TABLE_OUTPUT_DIR)
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('MPLBACKEND', 'Agg')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(name, '1')
    if args.action == 'experiments':
        selected = args.only or list(EXPERIMENTS)
        if args.resume_from and (args.rerun or len(selected) != 1 or selected[0] not in ('02', '03')):
            parser.error('--resume-from requires --only 02 or --only 03, without --rerun.')
        for key in EXPERIMENTS:
            if key not in selected:
                continue
            options = ['--rerun'] if args.rerun else []
            if args.resume_from:
                options += ['--resume-from', str(args.resume_from.resolve())]
            run_script(CODE_DIR / EXPERIMENTS[key], env, options)
        return

    categories = ('figures', 'tables') if args.action == 'render' else (args.action,)
    scripts = [path for category in categories for path in sorted((CODE_DIR/f'_{category}').glob('*.py'))
               if not path.name.startswith('_')]
    if args.only:
        requested = {Path(name).stem for name in args.only}
        unknown = requested - {path.stem for path in scripts}
        if unknown:
            parser.error('Unknown generator(s): ' + ', '.join(sorted(unknown)))
        scripts = [path for path in scripts if path.stem in requested]
    figure_dir = getattr(args, 'figure_output_dir', FIGURE_OUTPUT_DIR)
    table_dir = getattr(args, 'table_output_dir', TABLE_OUTPUT_DIR)
    env['TRTVLBM_FIGURE_OUTPUT_DIR'] = str(figure_dir)
    env['TRTVLBM_TABLE_OUTPUT_DIR'] = str(table_dir)
    for path in scripts:
        run_script(path, env)
        is_table = path.parent.name == '_tables'
        output = (table_dir if is_table else figure_dir)/(path.stem + ('.tex' if is_table else '.pdf'))
        if not output.is_file():
            raise FileNotFoundError(f'Generator did not produce its output: {output}')
        print(output, flush=True)


if __name__ == '__main__':
    main()
