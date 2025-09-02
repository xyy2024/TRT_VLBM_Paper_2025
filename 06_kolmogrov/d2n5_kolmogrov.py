#!/usr/bin/python
# -*- coding: utf-8 -*-

#    d2n5_kolmogrov.py
#    Copyright (C) 2025 Xu Yuyang (https://github.com/xyy2024)

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

import numpy as np
import math
from _plottools import prt_2d, gridfig, prt_mask_2d
from _systools import cached_property
from d2n5_taylorgreen import d2n5_taylorgreen

class d2n5_kolmogrov(d2n5_taylorgreen):
    kf = math.tau
    def get_outerforce(self):
        this = np.zeros((self.Nx, self.Ny, 2))
        this[:,:,0] = np.sin(self.kf*self.Y)
        return this

    def init_exact(self):
        # Set Taylor-Green vortex as initial disturbance
        scale = 0.001
        return (scale*uv for uv in d2n5_taylorgreen.exact(self))
    
    @cached_property
    def U0(self):
        return 1/(self.nu*self.kf**2)
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y

        return self.U0*np.sin(self.kf*y), np.zeros(x.shape)

class d2n5_kolmogrov_without_disturbance(d2n5_kolmogrov):
    Nx = 1

    def init_exact(self):
        return np.zeros(self.X.shape), np.zeros(self.Y.shape)

if __name__ == "__main__":
    test = d2n5_kolmogrov(h = 0.1, nu = 1/6)
    test.animation(time = 50, fps=10)