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

"""Equilibrium-initialized periodic acoustic perturbation of the D2N5 rest state."""

from __future__ import annotations

import numpy as np

from d2n5_taylorgreen import d2n5_taylorgreen


class d2n5_periodic_acoustic(d2n5_taylorgreen):
    """Unforced full nonlinear solver with pressure data at lattice frequency pi/2.

    On the cell-centered unit torus, P(x,0)=epsilon*cos(2*pi*(N/4)*x)
    and u(x,0)=0. The pressure pattern is formed by exact repetition of
    four floating-point values, avoiding grid-dependent trigonometric
    roundoff seeds in unrelated wave numbers. This is a grid-scale
    acoustic stability test, not refinement of one fixed NS solution.
    The collision, streaming and velocity reconstruction are inherited.
    """

    default_a = 0.2
    default_alpha = 0.5
    outerforce_type = 'static'
    initialization_protocol = 'four-cell-acoustic-equilibrium-v1'

    def __init__(self, *args, pressure_amplitude=1e-4, **kwargs):
        self.pressure_amplitude = float(pressure_amplitude)
        super().__init__(*args, **kwargs)
        if self.Nx != self.Ny or self.Nx % 4:
            raise ValueError('The acoustic test requires a square grid with N divisible by four.')
        if self.xshift != 0.5:
            raise ValueError('The repeated pressure pattern uses cell-centered nodes.')

    def exact(self, x=None, y=None):
        """Initial velocity only; no exact positive-time NS solution is claimed."""
        return np.zeros(self.shape), np.zeros(self.shape)

    def exact_pressure(self, x=None, y=None):
        """Initial pressure only, following the existing solver initialization API."""
        if self.Nx % 4:
            raise ValueError('N must be divisible by four.')
        pattern = np.tile(np.array([1., -1., -1., 1.])/np.sqrt(2), self.Nx//4)
        return self.pressure_amplitude*np.broadcast_to(pattern[:, None], self.shape)

    def get_outerforce(self):
        return np.zeros(self.shape+(2,))
