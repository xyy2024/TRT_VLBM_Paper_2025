#!/usr/bin/python
# -*- coding: utf-8 -*-

from d2n5_kolmogrov import d2n5_kolmogrov as model
import numpy as np
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