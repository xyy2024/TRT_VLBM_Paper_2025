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

'''Periodic D2N5 model for the unforced nonlinear double-shear test.'''

from __future__ import annotations

import math

import numpy as np

from d2n5_taylorgreen import d2n5_taylorgreen


class d2n5_doubleshear(d2n5_taylorgreen):
    r'''Unforced periodic double-shear initial-value problem.

    The velocity is initialized by

    .. math::
       u_1=U_0\tanh\!\left(\frac{\sin(2\pi(y-1/4))}
       {2\pi\delta}\right),\qquad
       u_2=\epsilon_v\sin(2\pi x).

    When ``initial_pressure`` is true, the zero-mean initial pressure solves
    ``Delta P_0 = -2 U'(y)V'(x)`` by a periodic Fourier inversion.  The model
    has no exact solution after the initial time and is intended for
    finite-time diagnostics rather than error measurement.
    '''

    U0 = 0.25
    delta = 0.1
    epsilon_v = 0.0125
    default_a = 0.2
    default_alpha = 0.2
    outerforce_type = 'static'

    def __init__(
        self,
        *args,
        delta: float | None = None,
        epsilon_v: float | None = None,
        initial_pressure: bool = True,
        **kwargs,
        ):
        self.delta = type(self).delta if delta is None else float(delta)
        self.epsilon_v = (
            type(self).epsilon_v if epsilon_v is None else float(epsilon_v)
        )
        self.initial_pressure = bool(initial_pressure)
        super().__init__(*args, **kwargs)
        self.initialization_protocol = (
            'double-shear-spectral-pressure-v1'
            if self.initial_pressure
            else 'double-shear-zero-pressure-v1'
        )

    def exact(self, x=None, y=None):
        '''Return the prescribed initial velocity field.

        This method defines initial data only; it is not an exact solution at
        positive time.
        '''
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        phase_y = 2*math.pi*(y-0.25)
        u = self.U0*np.tanh(
            np.sin(phase_y)/(2*math.pi*self.delta)
        )
        v = self.epsilon_v*np.sin(2*math.pi*x)
        return u, v

    def exact_pressure(self, x=None, y=None):
        '''Return the zero-mean pressure used only at initialization.'''
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError('x and y must be supplied together.')
            if np.shape(x) != self.shape or np.shape(y) != self.shape:
                raise ValueError(
                    'Double-shear pressure is available on the solver grid.'
                )
        if not self.initial_pressure:
            return np.zeros(self.shape, dtype=float)

        phase_y = 2*math.pi*(self.Y-0.25)
        argument = np.sin(phase_y)/(2*math.pi*self.delta)
        derivative_u = (
            self.U0*np.cos(phase_y)
            *(1/np.cosh(argument))**2/self.delta
        )
        derivative_v = 2*math.pi*self.epsilon_v*np.cos(2*math.pi*self.X)
        poisson_rhs = -2*derivative_u*derivative_v

        kx = 2*math.pi*np.fft.fftfreq(self.Nx, d=self.h)
        ky = 2*math.pi*np.fft.fftfreq(self.Ny, d=self.h)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        wave_number_squared = kx_grid**2+ky_grid**2
        pressure_hat = np.zeros(self.shape, dtype=complex)
        rhs_hat = np.fft.fftn(poisson_rhs)
        nonzero = wave_number_squared > 0
        pressure_hat[nonzero] = -rhs_hat[nonzero]/wave_number_squared[nonzero]
        pressure = np.fft.ifftn(pressure_hat).real
        pressure -= pressure.mean()
        return pressure


