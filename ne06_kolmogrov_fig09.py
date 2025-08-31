#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 06 (Kolmogrov Flow) Fig 09
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

from d2n5_kolmogrov import d2n5_kolmogrov as model
from _plottools import gridfig, Save, show, prt_flow_2d, prt_mask_2d

save = Save(prefix = "Kolmogrov")
test1 = model(h = 1/6, nu = 0.01)
test1.average_relax = 0.5/test1.tau # SRT
test2 = model(h = 1/6, nu = 0.01)
test2.average_relax = 0.01          # TRT
test1.until_stable()
test2.until_stable()
fig, axs = gridfig(1, 3)

from matplotlib.colors import Normalize
norm = Normalize(vmin=1.3, vmax=2.8)

kwargs = dict(norm = norm) #dict(cmap="copper", broken_streamlines = False, linewidth = 0.3, density = (1, 0.5))
u, v = test1.get_numerical_speed()
prt_mask_2d(axs[0], (test1.xmin, test1.xmax), (test1.ymin, test1.ymax), lambda x, y: test1.border_func(x, y))
prt_flow_2d(test1.x, test1.y, u, v, fig=fig, ax=axs[0], xlabel=f"SRT") #, **kwargs)
u, v = test1.get_precise_speed()
prt_mask_2d(axs[1], (test1.xmin, test1.xmax), (test1.ymin, test1.ymax), lambda x, y: test1.border_func(x, y))
prt_flow_2d(test1.x, test1.y, u, v, fig=fig, ax=axs[1], xlabel=f"Exact", **kwargs)
u, v = test2.get_numerical_speed()
prt_mask_2d(axs[2], (test2.xmin, test2.xmax), (test2.ymin, test2.ymax), lambda x, y: test2.border_func(x, y))
prt_flow_2d(test2.x, test2.y, u, v, fig=fig, ax=axs[2], xlabel=f"TRT", **kwargs)

fig.tight_layout()
save(fig)
show(fig, False)