#!/usr/bin/python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Xu Yuyang (https://github.com/xyy2024)
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
import math
from d2n5_taylorgreen import d2n5_taylorgreen


class d2n5_kolmogorov_startup(d2n5_taylorgreen):
    """Forced start-up problem, with a small vortex as the full initial velocity."""

    kf = math.tau
    initial_amplitude = 1e-3
    initialization_protocol = 'kolmogorov-startup-small-vortex-equilibrium-v1'

    def __init__(self, h=1/6, nu=0.01, **kwargs):
        super().__init__(h=h, nu=nu, U0=1/(nu*self.kf**2), **kwargs)

    def get_outerforce(self):
        force = np.zeros((self.Nx, self.Ny, 2))
        force[..., 0] = np.sin(self.kf*self.Y)
        return force

    def init_exact(self):
        # No steady shear is added; the vortex amplitude is independent of U0.
        phase_x = self.kf*self.X - math.pi/2
        phase_y = self.kf*self.Y - math.pi/2
        return (
            -self.initial_amplitude*np.cos(phase_x)*np.sin(phase_y),
            self.initial_amplitude*np.sin(phase_x)*np.cos(phase_y),
        )

    def exact(self, x=None, y=None):
        if x is None:
            x = self.X
        if y is None:
            y = self.Y
        return self.U0*np.sin(self.kf*y), np.zeros_like(x, dtype=float)


    def exact_pressure(self, x=None, y=None):
        if x is None:
            x = self.X
        return np.zeros_like(x, dtype=float)

