#!/usr/bin/python
# -*- coding: utf-8 -*-
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

import numpy as np
from d2n5_taylorgreen import d2n5_taylorgreen
from _systools import cached_property

class d2n5_poiseuille(d2n5_taylorgreen):
    # Due to symmetry, there is no need for excessive sampling of x.
    @cached_property
    def Nx(self): return 1

    # The outerforce
    @cached_property
    def G(self):
        # 外力大小
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
        u = 4*U*(1-y/H)*y/H
        v = np.zeros(x.shape)
        return u, v

    def exact_pressure(self, x = None, y = None):
        if x is None: x = self.X
        return np.zeros_like(x, dtype=float)

    def border_func(self, x, y):
        return np.abs(y-0.5) - 0.5


class d2n5_poiseuille_article(d2n5_poiseuille):
    '''Section 4 Poiseuille test with a prescribed force and link parameter.'''

    initialization_protocol = 'poiseuille-fixed-point-v1'

    def __init__(self, *args, G=0.08, ell=0.0, **kwargs):
        self._G = float(G)
        self.ell = float(ell)
        super().__init__(*args, **kwargs)

    @property
    def G(self):
        return self._G

    @property
    def l(self):
        # The reported stationary halfway-wall experiment uses ell=0.
        return self.ell

    def get_error_channel(self):
        '''Return the velocity L2 error for a unit streamwise channel.'''
        u_exact, v_exact = self.exact()
        u_numerical, v_numerical = self.get_numerical_speed()
        error_squared = (
            (u_numerical-u_exact)**2 + (v_numerical-v_exact)**2
        )
        return float(np.sqrt(self.h*np.sum(error_squared)))

