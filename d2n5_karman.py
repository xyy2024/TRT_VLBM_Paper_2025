#!/usr/bin/python
# -*- coding: utf-8 -*-

#    d2n5_karman.py
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
from _plottools import prt_2d, gridfig, prt_mask_2d, cmap
from d2n5_poiseuille import d2n5_taylorgreen, d2n5_poiseuille
from _systools import cached_property
from matplotlib.colors import Normalize

class d2n5_karman(d2n5_taylorgreen):
    xmin, xmax = 0, 10

    @cached_property
    def G(self):
        # For using Poiseuille class method
        return 8*self.nu

    def get_outerforce(self):
        return np.zeros((self.Nx, self.Ny, 2))

    def init_exact(self):
        # u = np.zeros(self.X.shape)
        # v = np.zeros(self.Y.shape)
        # return u, v
        return d2n5_poiseuille.exact(self)

    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        u, v = d2n5_poiseuille.exact(self, x, y)
        u[x > self.dx] = 0 # The cylinder is motionless
        return u, v
    
    vnorm = Normalize(vmin=-0.8, vmax=0.8)

    def fig_default_setting(self):
        u_num, v_num = self.get_numerical_speed()
        fig, axs = gridfig(2, 1)
        fig.set_figheight(8)
        fig.set_figwidth(20)

        prt_2d(self.x, self.y, u_num, fig=fig, ax=axs[0], xlabel="u", cmap = cmap)
        prt_2d(self.x, self.y, v_num, fig=fig, ax=axs[1], xlabel="v", cmap = cmap, norm = self.vnorm)

        for ax in axs:
            prt_mask_2d(ax, (self.xmin, self.xmax), (self.ymin, self.ymax), lambda x, y: self.border_func(x, y))
        
        fig.tight_layout()
        return fig, axs

    # border condition
    center_x = 1.5
    center_y = 0.5
    radius = 0.1

    def border_func(self, x, y):
        area = np.abs(y-0.5) - 0.5       # up and down border

        area[area < -x] = -x[area < -x]  # left border
        
        # the cylinder at middle
        center1 = self.radius**2-(y-self.center_y)**2-(x-self.center_x)**2
        area[area<center1]=center1[area<center1]

        return area

    # right border
    def border_condition(self, nextf) -> np.ndarray:
        nextf = super().border_condition(nextf)
        nextf[-1,:,:,:] = nextf[-2,:,:,:]
        return nextf
    
class d2n5_karman_with_disturbance(d2n5_karman):
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        u, v = d2n5_poiseuille.exact(self, x, y)
        u[x > self.dx] = 0
        # Add disturbance
        # If no disturbance is added, it will take a long time before the vortex street appear.
        if 2.5 <= self.t < 3 + self.dt:
            index = (x > self.dx) & (abs(y - 0.5) < 0.5 - self.dx)
            u[index] = -(y[index] - self.center_y)
            v[index] = x[index] - self.center_x
        return u, v
