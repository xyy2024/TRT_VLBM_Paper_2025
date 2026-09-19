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

"""Working-directory-independent data paths and command-line output selection."""
from pathlib import Path
import argparse
import os

CODE_DIR = Path(__file__).resolve().parent
DATA_CSV_DIR = CODE_DIR / 'data_csv'
INSTANCE_OUTPUT_DIR = CODE_DIR / 'solver_instances'


def artifact_output_directory(value):
    directory = Path(value).expanduser().resolve()
    for reserved in (DATA_CSV_DIR, INSTANCE_OUTPUT_DIR):
        if directory == reserved or reserved in directory.parents:
            raise ValueError(f'Figure/table outputs must be outside {reserved}.')
    return directory


FIGURE_OUTPUT_DIR = artifact_output_directory(
    os.environ.get('TRTVLBM_FIGURE_OUTPUT_DIR', CODE_DIR / 'figure_output'))
TABLE_OUTPUT_DIR = artifact_output_directory(
    os.environ.get('TRTVLBM_TABLE_OUTPUT_DIR', CODE_DIR / 'table_output'))


def configure_renderer(kind):
    """Called before a standalone generator imports its output directory."""
    global FIGURE_OUTPUT_DIR, TABLE_OUTPUT_DIR
    parser = argparse.ArgumentParser(description=f'Generate one {kind} from article CSV data.')
    parser.add_argument('--output-dir', type=artifact_output_directory,
                        help=f'Output directory (default: {kind}_output/ in the repository).')
    args = parser.parse_args()
    if args.output_dir is not None:
        if kind == 'figure':
            FIGURE_OUTPUT_DIR = args.output_dir
        else:
            TABLE_OUTPUT_DIR = args.output_dir


def configure_experiment(*, periodic=False):
    parser = argparse.ArgumentParser(description='Run this article experiment and export its CSV data. Solver instances are always saved.')
    parser.add_argument('--rerun', action='store_true',
                        help='Integrate fresh cases instead of reusing matching saved instances.')
    if periodic:
        parser.add_argument('--resume-from', type=Path,
                            help='Read compatible convergence starts from this saved-instance directory.')
    args = parser.parse_args()
    if periodic and args.resume_from and args.rerun:
        parser.error('Use either --resume-from or --rerun.')
    if args.rerun:
        os.environ['TRTVLBM_RERUN'] = '1'
    return args


def reuse_instances():
    return os.environ.get('TRTVLBM_RERUN') != '1'
