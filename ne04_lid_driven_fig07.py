#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 04 (Lid Driven Cavity) Fig 07
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

from _plottools import prt_flow_2d, gridfig, save, show
from d2n5_lid_driven import d2n5_lid_driven

nu_list = [0.005, 0.002, 0.001]

test1 = d2n5_lid_driven(h = 1/64, nu = nu_list[0])
test2 = d2n5_lid_driven(h = 1/64, nu = nu_list[1])
test3 = d2n5_lid_driven(h = 1/64, nu = nu_list[2])

test1.until_time(20)
test2.until_time(20)
test3.until_time(20)

fig, axs = gridfig(1, 3)
kwargs = dict(broken_streamlines = False, linewidth = 0.3, density = (1, 0.5))
u, v = test1.get_numerical_speed()
prt_flow_2d(test1.x, test1.y, u, v, fig=fig, ax=axs[0], xlabel=f"$\\nu = {test1.nu}$", cmap="copper", **kwargs)
u, v = test2.get_numerical_speed()
prt_flow_2d(test2.x, test2.y, u, v, fig=fig, ax=axs[1], xlabel=f"$\\nu = {test2.nu}$", cmap="copper", **kwargs)
u, v = test3.get_numerical_speed()
prt_flow_2d(test3.x, test3.y, u, v, fig=fig, ax=axs[2], xlabel=f"$\\nu = {test3.nu}$", cmap="copper", **kwargs)
fig.tight_layout()
save.prefix = "LidDriven"
save(fig)
show(fig, False)