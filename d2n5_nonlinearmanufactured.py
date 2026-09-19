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

'''D2N5 nonlinear manufactured solution from Section 4.2 of the paper.'''

import math

import numpy as np

from d2n5_taylorgreen import d2n5_taylorgreen


class d2n5_nonlinearmanufactured(d2n5_taylorgreen):
    r'''Nonlinear periodic manufactured solution with zero exact pressure.

    The streamfunction is

    .. math::
        \chi=\epsilon e^{-t}\left[
        \sin(2\pi x)\sin(2\pi y)
        +\tfrac14\sin(4\pi x)\sin(2\pi y)\right],

    with u = (partial_y chi, -partial_x chi) and
    F = partial_t u + (u dot grad)u - nu Laplacian(u).
    '''

    epsilon = 0.05
    default_a = 0.2
    default_alpha = 0.2
    outerforce_type = 'dynamic'

    def exact(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        ):
        if x is None: x = self.X
        if y is None: y = self.Y

        k = 2*math.pi
        amplitude = self.epsilon*math.exp(-self.t)
        x1 = k*x
        y1 = k*y

        u = amplitude*k*np.cos(y1)*(
            np.sin(x1) + 0.25*np.sin(2*x1)
        )
        v = -amplitude*k*np.sin(y1)*(
            np.cos(x1) + 0.5*np.cos(2*x1)
        )
        return u, v

    def exact_pressure(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        ):
        if x is None: x = self.X
        return np.zeros_like(x, dtype=float)

    def get_outerforce(self):
        k = 2*math.pi
        amplitude = self.epsilon*math.exp(-self.t)
        x1 = k*self.X
        y1 = k*self.Y

        sin_x = np.sin(x1)
        sin_2x = np.sin(2*x1)
        cos_x = np.cos(x1)
        cos_2x = np.cos(2*x1)
        sin_y = np.sin(y1)
        cos_y = np.cos(y1)

        stream_x = sin_x + 0.25*sin_2x
        stream_y = cos_x + 0.5*cos_2x
        derivative_x = sin_x + sin_2x

        u = amplitude*k*cos_y*stream_x
        v = -amplitude*k*sin_y*stream_y

        convection_u = amplitude**2*k**3*stream_x*stream_y
        convection_v = (
            amplitude**2*k**3*sin_y*cos_y
            * (stream_x*derivative_x + stream_y**2)
        )

        laplacian_u = -amplitude*k**3*cos_y*(
            2*sin_x + 1.25*sin_2x
        )
        laplacian_v = amplitude*k**3*sin_y*(
            2*cos_x + 2.5*cos_2x
        )

        force = np.empty(self.shape + (2,), dtype=float)
        force[:,:,0] = -u + convection_u - self.nu*laplacian_u
        force[:,:,1] = -v + convection_v - self.nu*laplacian_v
        return force


