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

"""Shared rest-state Fourier matrices and double-shear monitoring diagnostics.

Used by acoustic mode analysis and the three-group double-shear experiment.
"""
from __future__ import annotations
import math
import numpy as np

DENSITY_INTERVAL = (0.5, 1.5)
ENERGY_RATIO_LIMIT = 1.25
DENSITY_FLUCTUATION_LIMIT = 0.05
DIVERGENCE_LIMIT = 0.5
LATTICE_SPEED_LIMIT = 0.25


def collision_matrix(d, a, alpha, s_plus, s_minus):
    '''Build the linearized rest-state collision matrix.'''
    moment_count = d+1
    velocity_count = 2*d+1
    velocities = np.vstack((
        np.eye(d, dtype=int), -np.eye(d, dtype=int),
        np.zeros((1, d), dtype=int),
    ))
    canonical = np.eye(moment_count)
    identity_moment = np.eye(moment_count)
    coupling = [
        np.outer(canonical[0], canonical[index+1])
        + np.outer(canonical[index+1], canonical[0])
        for index in range(d)
    ]
    equilibrium_blocks = [
        a*identity_moment+0.5*alpha*coupling[index]
        for index in range(d)
    ]
    equilibrium_blocks.extend(
        a*identity_moment-0.5*alpha*coupling[index]
        for index in range(d)
    )
    equilibrium_blocks.append((1.0-2.0*d*a)*identity_moment)
    equilibrium_projection = np.vstack(equilibrium_blocks) @ np.hstack(
        [identity_moment]*velocity_count
    )
    opposite = np.r_[np.arange(d, 2*d), np.arange(d), 2*d]
    reversal = np.kron(np.eye(velocity_count)[opposite], identity_moment)
    identity = np.eye(velocity_count*moment_count)
    phi = 0.5*(s_plus+s_minus)
    psi = 0.5*(s_plus-s_minus)
    collision = identity+(phi*identity+psi*reversal) @ (
        equilibrium_projection-identity
    )
    return velocities, collision


def periodic_derivative(field, h, axis):
    return (np.roll(field, -1, axis=axis)-np.roll(field, 1, axis=axis))/(2*h)


def diagnostics(solver, initial_mass, initial_energy, initial_enstrophy):
    rho = solver.w[..., 0]
    u, v = solver.get_numerical_speed()
    speed_squared = u*u+v*v
    divergence = (
        periodic_derivative(u, solver.h, 0)
        + periodic_derivative(v, solver.h, 1)
    )
    vorticity = (
        periodic_derivative(v, solver.h, 0)
        - periodic_derivative(u, solver.h, 1)
    )
    measure = solver.h**2
    mass = measure*np.sum(rho)
    energy = 0.5*measure*np.sum(speed_squared)
    enstrophy = 0.5*measure*np.sum(vorticity*vorticity)
    density_mean = np.mean(rho)
    return {
        'energy': float(energy),
        'energy_ratio': float(energy/initial_energy),
        'enstrophy': float(enstrophy),
        'enstrophy_ratio': float(enstrophy/initial_enstrophy),
        'vorticity_l2': float(math.sqrt(2*enstrophy)),
        'divergence_l2': float(math.sqrt(measure*np.sum(divergence**2))),
        'density_fluctuation': float(np.max(np.abs(rho-density_mean))),
        'lattice_speed': float(solver.h*math.sqrt(float(np.max(speed_squared)))),
        'mass_drift': float(abs(mass-initial_mass)),
        'density_min': float(np.min(rho)),
        'density_max': float(np.max(rho)),
    }


def threshold_reason(values):
    if values['density_min'] < DENSITY_INTERVAL[0]:
        return 'density below 0.5'
    if values['density_max'] > DENSITY_INTERVAL[1]:
        return 'density above 1.5'
    if values['energy_ratio'] > ENERGY_RATIO_LIMIT:
        return 'energy ratio above 1.25'
    if values['density_fluctuation'] > DENSITY_FLUCTUATION_LIMIT:
        return 'density fluctuation above 0.05'
    if values['divergence_l2'] > DIVERGENCE_LIMIT:
        return 'D2 above 0.5'
    if values['lattice_speed'] > LATTICE_SPEED_LIMIT:
        return 'M_h above 0.25'
    return ''
