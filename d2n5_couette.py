#!/usr/bin/python
# -*- coding: utf-8 -*-

#    d2n5_couette.py
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
from d2n5_taylorgreen import d2n5_taylorgreen
from _systools import cached_property

class d2n5_couette(d2n5_taylorgreen):
    # Due to symmetry, there is no need for excessive sampling of x.
    @cached_property
    def Nx(self): return 1

    # The outerforce
    @cached_property
    def G(self):
        return 0.8*self.nu
    
    def get_outerforce(self):
        return np.array([self.G, 0])[None, None, :]

    def init_exact(self):
        return np.zeros(self.X.shape), np.zeros(self.Y.shape)
    
    def exact(self, x = None, y = None):
        if x is None: x = self.X
        if y is None: y = self.Y

        H = 1 # = ymax - ymin
        U = self.G*H**2/8/self.nu
        Ub = 0.5*U
        u = 4*U*(1-y/H)*y/H + y/H*Ub
        v = np.zeros(x.shape)
        return u, v

    def border_func(self, x, y):
        return np.abs(y-0.5) - 0.5

if __name__ == "__main__":
    test = d2n5_couette(h = 0.025, nu = 0.002)

    test.until_stable(1e-11)
    print(test.get_error())

    test.until_stable(1e-12)
    print(test.get_error())

    test.until_stable(1e-13)
    print(test.get_error())

    test.until_stable(1e-14)
    print(test.get_error())