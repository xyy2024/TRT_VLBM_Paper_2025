#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Appendix 01
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

from _plottools import prt_flow_2d, gridfig, prt_mask_2d, save, show
from d2n5_lid_driven import d2n5_lid_driven, np

class model(d2n5_lid_driven):
    def border_func(self, x, y):
        x_area = np.abs(x-0.5) - 0.5               # 0 <= x <= 1 在区域内
        y_area = np.abs(y-0.5) - 0.5               # 0 <= y <= 1 在区域内

        x_area[x_area<y_area] = y_area[x_area<y_area]  # 两者取最大值

        corner = -np.abs(x-1)
        corner2= -np.abs(y)
        corner[corner > corner2] = corner2[corner > corner2]
        
        d = 0.4
        corner += d

        x_area[x_area<corner] = corner[x_area<corner]

        return x_area

nu_list = [0.005, 0.002, 0.001]

test1 = model(h = 1/64, nu = nu_list[0])
test2 = model(h = 1/64, nu = nu_list[1])
test3 = model(h = 1/64, nu = nu_list[2])

test1.until_time(20)
test2.until_time(20)
test3.until_time(20)

fig, axs = gridfig(1, 3)
kwargs = dict(broken_streamlines = False, linewidth = 0.3, density = (1, 0.5))
u, v = test1.get_numerical_speed()
prt_flow_2d(test1.x, test1.y, u, v, fig=fig, ax=axs[0], xlabel=f"$\\nu = {test1.nu}$", cmap="copper", **kwargs)
prt_mask_2d(axs[0], (test1.xmin, test1.xmax), (test1.ymin, test1.ymax), lambda x, y: test1.border_func(x, y))
u, v = test2.get_numerical_speed()
prt_flow_2d(test2.x, test2.y, u, v, fig=fig, ax=axs[1], xlabel=f"$\\nu = {test2.nu}$", cmap="copper", **kwargs)
prt_mask_2d(axs[1], (test2.xmin, test2.xmax), (test2.ymin, test2.ymax), lambda x, y: test2.border_func(x, y))
u, v = test3.get_numerical_speed()
prt_flow_2d(test3.x, test3.y, u, v, fig=fig, ax=axs[2], xlabel=f"$\\nu = {test3.nu}$", cmap="copper", **kwargs)
prt_mask_2d(axs[2], (test3.xmin, test3.xmax), (test3.ymin, test3.ymax), lambda x, y: test3.border_func(x, y))
fig.tight_layout()
save.prefix = "LidDriven_swsqr"
save.dir = "graphic\\LidDriven_Appendix"
save(fig)
show(fig, False)