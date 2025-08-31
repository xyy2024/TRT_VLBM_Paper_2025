#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 01 (Taylor Green Vortex) Fig 01
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

from d2n5_taylorgreen import d2n5_taylorgreen as model
from _plottools import gridfig, prt_2d, prt_mask_2d

def flow_fig(self:model, u_num, v_num, u_pre, v_pre, u_err, v_err, L2_Relative_error):
    fig, axs = gridfig(1, 3)

    prt_2d(self.x, self.y, u_num, v_num, fig=fig, ax=axs[0], xlabel=f"Numerical")
    prt_2d(self.x, self.y, u_pre, v_pre, fig=fig, ax=axs[1], xlabel=f"Exact")
    prt_2d(self.x, self.y, (u_err**2 + v_err**2)**(1/2), fig=fig, ax=axs[2], xlabel=f"Error")

    for ax in axs:
        prt_mask_2d(ax, (self.xmin, self.xmax), (self.ymin, self.ymax), lambda x, y: self.border_func(x, y))
    
    fig.tight_layout()
    return fig, axs

initial = model(h=1/128)
initial.save.prefix = "TaylorGreen"
initial.until_time(0.2)
initial.fig_with_error(save_fig=True, show_fig=False, setting=flow_fig)
initial.until_time(0.4)
initial.fig_with_error(save_fig=True, show_fig=False, setting=flow_fig)