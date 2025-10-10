#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 05 (Flow Pass a Cylinder) Fig 08
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

from d2n5_karman import d2n5_karman, gridfig, prt_2d, cmap, prt_mask_2d

test = d2n5_karman(h = 1/32, nu = 0.002)
test.save.prefix = "karman"
test.save.dir = "graphic\\Karman\\00_default"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")

'''
Because I've changed $u, v$ in the paper to $u_1, u_2$, the xlabel of the result graphic should also be changed. 

In order to change the xlabel, I should run:

`
def setting(self):
    u_num, v_num = self.get_numerical_speed()
    fig, axs = gridfig(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    prt_2d(self.x, self.y, u_num, fig=fig, ax=axs[0], xlabel="$u_1$", cmap = cmap)
    prt_2d(self.x, self.y, v_num, fig=fig, ax=axs[1], xlabel="$u_2$", cmap = cmap, norm = self.vnorm)

    for ax in axs:
        prt_mask_2d(ax, (self.xmin, self.xmax), (self.ymin, self.ymax), lambda x, y: self.border_func(x, y))
    
    fig.tight_layout()
    return fig, axs

test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4", setting=setting)
`

But run this will cost a lot of time. So I won't run it again only for changing the xlabel.
'''