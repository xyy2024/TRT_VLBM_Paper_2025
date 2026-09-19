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

'''D3N7 model for the fully three-dimensional Ethier--Steinman flow.'''

import math

import numpy as np

from d3n7_taylorgreen import d3n7_taylorgreen


class d3n7_ethiersteinmann(d3n7_taylorgreen):
    r'''Ethier--Steinman exact solution on the unit cube.

    ``flow_a`` and ``flow_d`` are the two parameters in the analytical
    solution.  The distinct names are intentional: ``self.a`` is reserved
    for the D3N7 equilibrium weight and must not be confused with the
    analytical parameter called ``a`` in Ethier and Steinman's paper.

    The pressure returned by :meth:`exact_pressure` uses the gauge printed
    with the exact solution.  It is not mean-centred, because D3N7 density
    initialization requires ``rho = 1 + h**2*P``.
    '''

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    zmin, zmax = 0, 1

    default_a = 1/7
    default_alpha = 1/7
    default_flow_a = math.pi/4
    default_flow_d = math.pi/2
    initialization_protocol = (
        'ethier-steinmann-exact-pressure-dirichlet-appendix-c-time-n-v4'
    )

    def __init__(
        self,
        *args,
        flow_a=None,
        flow_d=None,
        **kwargs,
        ):
        self.flow_a = float(
            type(self).default_flow_a if flow_a is None else flow_a
        )
        self.flow_d = float(
            type(self).default_flow_d if flow_d is None else flow_d
        )
        super().__init__(*args, **kwargs)

    def decay_scale(self, time=None):
        if time is None:
            time = self.t
        return math.exp(-self.nu*self.flow_d**2*float(time))

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
        flow_a = self.flow_a
        flow_d = self.flow_d
        scale = self.decay_scale()

        phase_xy = flow_a*x+flow_d*y
        phase_yz = flow_a*y+flow_d*z
        phase_zx = flow_a*z+flow_d*x
        exp_x = np.exp(flow_a*x)
        exp_y = np.exp(flow_a*y)
        exp_z = np.exp(flow_a*z)

        return (
            -flow_a*scale*(
                exp_x*np.sin(phase_yz)+exp_z*np.cos(phase_xy)
            ),
            -flow_a*scale*(
                exp_y*np.sin(phase_zx)+exp_x*np.cos(phase_yz)
            ),
            -flow_a*scale*(
                exp_z*np.sin(phase_xy)+exp_y*np.cos(phase_zx)
            ),
        )

    def exact_pressure(self, x=None, y=None, z=None):
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        if z is None:
            z = self.Z
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        flow_a = self.flow_a
        flow_d = self.flow_d

        phase_xy = flow_a*x+flow_d*y
        phase_yz = flow_a*y+flow_d*z
        phase_zx = flow_a*z+flow_d*x
        pressure_shape = (
            np.exp(2*flow_a*x)
            + np.exp(2*flow_a*y)
            + np.exp(2*flow_a*z)
            + 2*np.exp(flow_a*(y+z))
            * np.sin(phase_xy)*np.cos(phase_zx)
            + 2*np.exp(flow_a*(z+x))
            * np.sin(phase_yz)*np.cos(phase_xy)
            + 2*np.exp(flow_a*(x+y))
            * np.sin(phase_zx)*np.cos(phase_yz)
        )
        return -0.5*flow_a**2*self.decay_scale()**2*pressure_shape

    def get_numerical_pressure(self, w=None):
        '''Return ``(rho-1)/h**2`` in the same pressure scaling as above.'''
        return super().get_numerical_pressure(w)

    def border_func(self, x, y, z):
        '''Signed distance surrogate for the Dirichlet unit-cube boundary.'''
        centre_x = 0.5*(self.xmin+self.xmax)
        centre_y = 0.5*(self.ymin+self.ymax)
        centre_z = 0.5*(self.zmin+self.zmax)
        half_x = 0.5*(self.xmax-self.xmin)
        half_y = 0.5*(self.ymax-self.ymin)
        half_z = 0.5*(self.zmax-self.zmin)
        return np.maximum.reduce((
            np.abs(x-centre_x)-half_x,
            np.abs(y-centre_y)-half_y,
            np.abs(z-centre_z)-half_z,
        ))

    def border_condition(self, nextf):
        '''Apply Appendix C with analytical wall velocity at the current time.

        The incoming populations at ``t + dt`` use post-collision populations
        and boundary equilibria at ``t``. Both the NumPy and compiled stepping
        paths call this method before incrementing ``iter_count``, so the
        inherited rule evaluates ``exact`` at the required time level.
        '''
        return super().border_condition(nextf)

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


