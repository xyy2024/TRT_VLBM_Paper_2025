#!/usr/bin/python
# -*- coding: utf-8 -*-

#    d3n7_hagenpoiseuille.py
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

import numpy as np
from _systools import cached_property
from d3n7_taylorgreen import d3n7_taylorgreen

class d3n7_hagenpoiseuille(d3n7_taylorgreen):
    Nx = 1

    # 设置模型外力
    @cached_property
    def G(self):
        # 外力大小
        return 0.8*self.nu
    
    def get_outerforce(self):
        return np.array([self.G, 0, 0])[None, None, None, :]
    
    # 设置初态
    def init_exact(self):
        return (np.zeros(self.X.shape) for _ in range(3))
    
    # 设置精确解
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None, z:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        if z is None: z = self.Z

        H = 1 # = 直径
        U = self.G*H**2/16/self.nu
        u1 = 4*U*(H**2*0.25 - (y-0.5)**2 - (z-0.5)**2)
        u2 = np.zeros(x.shape)
        u3 = np.zeros(x.shape)
        return u1, u2, u3

    # 设置边界情况
    def border_func(self, x, y, z):
        return (y-0.5)**2 + (z-0.5)**2 - 0.25
    
    @cached_property
    def l(self): return self.gamma**1.5


if __name__ == "__main__":
    nu_list = [0.001, 0.01, 0.1]
    for nu in nu_list:
        test = d3n7_hagenpoiseuille(h=0.2, nu=nu)
        test.until_stable(1e-10)
        print(test.get_error())