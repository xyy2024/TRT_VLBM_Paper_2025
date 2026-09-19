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

'''Optional compiled accelerator for the zero-force D3N7 experiment steps.'''

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from numpy.ctypeslib import ndpointer


_SUFFIX = '.dll' if os.name == 'nt' else ('.dylib' if sys.platform == 'darwin' else '.so')
_LIBRARY = Path(__file__).with_suffix(_SUFFIX)
_DOUBLE_ARRAY = ndpointer(dtype=np.float64, ndim=None, flags='C_CONTIGUOUS')
_DLL_DIRECTORY_HANDLES = []

if os.name == 'nt':
    gcc = shutil.which('gcc')
    if gcc is not None:
        _DLL_DIRECTORY_HANDLES.append(
            os.add_dll_directory(str(Path(gcc).resolve().parent))
        )


def _load_library():
    try:
        library = ctypes.CDLL(str(_LIBRARY))
    except OSError:
        return None
    library.d3n7_collide_stream.argtypes = [
        _DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
    ]
    library.d3n7_collide_stream.restype = None
    library.d3n7_sum_populations.argtypes = [
        _DOUBLE_ARRAY, _DOUBLE_ARRAY, ctypes.c_size_t,
    ]
    library.d3n7_sum_populations.restype = None
    return library


_LIB = _load_library()


def iter_zero_force(solver):
    '''Advance one mathematically identical zero-force D3N7 step.'''
    if _LIB is None:
        solver.iter()
        return
    solver._ensure_work_arrays()
    for name in ('f', 'w', 'fstar', 'nextf'):
        array = getattr(solver, name)
        if array.dtype != np.float64 or not array.flags.c_contiguous:
            raise TypeError(f'{name} must be a C-contiguous float64 array.')

    nx, ny, nz = solver.shape
    _LIB.d3n7_collide_stream(
        solver.f,
        solver.w,
        solver.fstar,
        solver.nextf,
        nx,
        ny,
        nz,
        solver.h,
        solver.a,
        solver.alpha,
        solver.relax1,
        solver.relax2,
    )
    nextf = solver.border_condition(solver.nextf)
    solver.f, solver.nextf = nextf, solver.f
    solver.iter_count += 1
    _LIB.d3n7_sum_populations(solver.f, solver.w, nx*ny*nz)
