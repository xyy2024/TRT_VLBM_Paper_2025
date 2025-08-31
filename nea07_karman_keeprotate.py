#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Appendix 07
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

from d2n5_karman import d2n5_karman, np, d2n5_poiseuille

class model(d2n5_karman):
    center_x = 1.5
    center_y = 0.5
    radius = 0.1

    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        u, v = d2n5_poiseuille.exact(self, x, y)
        u[x > self.dx] = 0
        index = (x > self.dx) & (abs(y - 0.5) < 0.5 - self.dx)
        u[index] = -(y[index] - self.center_y)
        v[index] = (x[index] - self.center_x)
        return u, v
    
test = model(h = 1/32, nu = 0.002)
test.save.prefix = f"karman"
test.save.dir = "graphic\\Karman\\07_keeprotate"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")