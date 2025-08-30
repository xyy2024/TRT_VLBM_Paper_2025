#!/usr/bin/python
# -*- coding: utf-8 -*-

from d2n5_taylorgreen import d2n5_taylorgreen as model
from _plottools import rankfig, gridfig, prt_2d, prt_mask_2d
import numpy as np

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