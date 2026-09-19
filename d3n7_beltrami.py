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

'''Periodic D3N7 ABC/Beltrami exact-solution model.'''

import math

import numpy as np

from d3n7_taylorgreen import d3n7_taylorgreen


class d3n7_beltrami(d3n7_taylorgreen):
    r'''D3N7 model for the decaying periodic ABC/Beltrami flow.

    The velocity is

    ``A exp(-nu*k**2*t) * (sin(kz)+cos(ky), sin(kx)+cos(kz),
    sin(ky)+cos(kx))``.

    Since ``curl(u)=k*u``, its convection is a gradient.  The mean-free
    pressure ``P=-(|u|**2-<|u|**2>)/2`` therefore makes this an exact
    unforced Navier--Stokes solution.
    '''

    default_a = 1/7
    default_alpha = 1/7
    initialization_protocol = 'beltrami-exact-pressure-equilibrium-v1'

    def __init__(
        self,
        *args,
        amplitude=0.05,
        wavenumber=2*math.pi,
        **kwargs,
        ):
        self.amplitude = float(amplitude)
        self.wavenumber = float(wavenumber)
        super().__init__(*args, **kwargs)

    def velocity_scale(self, time=None):
        if time is None:
            time = self.t
        return self.amplitude*math.exp(
            -self.nu*self.wavenumber**2*float(time)
        )

    def exact(self, x=None, y=None, z=None):
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        if z is None:
            z = self.Z
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        k = self.wavenumber
        scale = self.velocity_scale()
        return (
            scale*(np.sin(k*z)+np.cos(k*y)),
            scale*(np.sin(k*x)+np.cos(k*z)),
            scale*(np.sin(k*y)+np.cos(k*x)),
        )

    def exact_pressure(self, x=None, y=None, z=None):
        u, v, w = self.exact(x, y, z)
        scale = self.velocity_scale()
        # The exact spatial average of |u|^2 is 3*scale^2.
        return -0.5*(u**2+v**2+w**2-3*scale**2)

    def get_component_error_l2(self):
        '''Return the three componentwise discrete L2 velocity errors.'''
        numerical = self.get_numerical_speed()
        precise = self.exact()
        cell_volume = self.h**3
        return np.asarray([
            math.sqrt(cell_volume*np.sum((u_h-u_exact)**2))
            for u_h, u_exact in zip(numerical, precise)
        ])

    def get_combined_error_l2(self):
        component_error = self.get_component_error_l2()
        return float(np.sqrt(np.sum(component_error**2)))


