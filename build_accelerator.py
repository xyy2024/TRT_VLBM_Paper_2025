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

"""Build the optional D3N7 C accelerator with GCC or a compatible C compiler."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC', 'gcc' if os.name == 'nt' else 'cc'),
                        help='C compiler executable or full path (default: CC, then gcc/cc).')
    parser.add_argument('--openmp', action='store_true',
                        help='Enable OpenMP; requires a compiler and runtime supporting -fopenmp.')
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if compiler is None:
        parser.error(f'C compiler not found: {args.cc}. Use --cc with its executable path.')
    root = Path(__file__).resolve().parent
    suffix = '.dll' if os.name == 'nt' else ('.dylib' if sys.platform == 'darwin' else '.so')
    output = root / ('_d3n7_fast' + suffix)
    flags = ['-O3', '-std=c11']
    if os.name == 'nt':
        flags += ['-shared']
    elif sys.platform == 'darwin':
        flags += ['-dynamiclib', '-fPIC']
    else:
        flags += ['-shared', '-fPIC']
    if args.openmp:
        flags += ['-fopenmp']
    with tempfile.TemporaryDirectory(prefix='.build-', dir=root) as temporary:
        built = Path(temporary) / output.name
        command = [compiler, *flags, str(root / '_d3n7_fast.c'), '-o', str(built)]
        print('Building:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
        built.replace(output)
    print(f'Built {output}')
    print('Start a new Python process to use the accelerator.')


if __name__ == '__main__':
    main()
