#!/usr/bin/python
# -*- coding: utf-8 -*-

#    _nproll.py
#    Copyright (C) 2025 Xu Yuyang

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

'''Circshift'''

# I think there is a problem with numpy.roll function, so I made this

import numpy as np

def circshift(array:np.ndarray, shift:tuple[int, ...]|int):
    if isinstance(shift, int):
        shift = (shift, )
    result = array.copy()
    NoneSliceTuple = (slice(None),)
    for i in range(len(shift)):
        sub_shift = shift[i]
        if sub_shift == 0:
            continue
        else:
            (result[NoneSliceTuple*i+(slice(sub_shift, None),)],
             result[NoneSliceTuple*i+(slice(None, sub_shift),)]
             ) = (
             result[NoneSliceTuple*i+(slice(None, -sub_shift),)].copy(),
             result[NoneSliceTuple*i+(slice(-sub_shift, None),)].copy())
    return result