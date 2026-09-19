#!/usr/bin/python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Xu Yuyang
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

'''Circshift'''

# I think there is a problem with numpy.roll function, so I made this

import numpy as np
from itertools import product
from operator import index

def circshift(
    array:np.ndarray,
    shift:tuple[int, ...]|int,
    out:np.ndarray|None = None,
    ) -> np.ndarray:
    '''Circularly shift *array*, optionally writing directly into *out*.

    ``shift[i]`` applies to axis ``i``.  If ``out`` does not overlap the
    input, data is copied directly between corresponding blocks without a
    full-size temporary array.  Overlapping input and output are supported
    through a protective input copy.
    '''
    if isinstance(shift, (int, np.integer)):
        shift = (index(shift),)
    else:
        try:
            shift = tuple(index(value) for value in shift)
        except TypeError as error:
            raise TypeError('shift values must be integers.') from error

    if len(shift) > array.ndim:
        raise ValueError(
            f'shift has {len(shift)} axes, but array has only '
            f'{array.ndim} dimensions.'
        )

    if out is None:
        result = np.empty_like(array)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('out must be a numpy.ndarray.')
        if out.shape != array.shape:
            raise ValueError(
                f'out has shape {out.shape}, expected {array.shape}.'
            )
        result = out

    source = array.copy() if np.shares_memory(array, result) else array

    # For every axis, describe the one or two source/destination blocks
    # produced by the circular shift.  Their Cartesian product covers the
    # full output exactly once, including shifts along multiple axes.
    axis_blocks = []
    for axis, axis_size in enumerate(array.shape):
        axis_shift = shift[axis] if axis < len(shift) else 0
        if axis_size == 0:
            normalized_shift = 0
        else:
            normalized_shift = axis_shift % axis_size

        if normalized_shift == 0:
            axis_blocks.append(((slice(None), slice(None)),))
        else:
            axis_blocks.append((
                (
                    slice(normalized_shift, None),
                    slice(None, -normalized_shift),
                ),
                (
                    slice(None, normalized_shift),
                    slice(-normalized_shift, None),
                ),
            ))

    for block in product(*axis_blocks):
        destination = tuple(part[0] for part in block)
        origin = tuple(part[1] for part in block)
        result[destination] = source[origin]

    return result
