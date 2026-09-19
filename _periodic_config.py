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

"""Published parameters for the manufactured and Taylor--Green convergence studies."""
A, NU = 0.2, 0.1
FINAL_TIME = 0.25
N_LIST = (16, 24, 32, 48, 64, 96, 128, 192, 256)
CONFIGURATIONS = (('OTRT', 0.2), ('SRT', 0.5), ('SRT', 0.2), ('OTRT', 0.5))
LIMITS = dict(density_min=0.5, density_max=1.5, density_fluctuation=0.05,
              d2=0.5, m_h=0.25)
MODEL_SOURCES = ('d2n5_taylorgreen.py', 'd2n5_nonlinearmanufactured.py',
                 '_nproll.py', '_systools.py')


def rates(method, alpha):
    s_minus = 2*A/(A+alpha*NU)
    return (s_minus if method == 'SRT' else 2-s_minus), s_minus


def time_key(value):
    return 'T'+format(value, '.12g').replace('.', 'p')
