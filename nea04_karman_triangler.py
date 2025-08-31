#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Appendix 04
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

from d2n5_karman import d2n5_karman, np, math

class model(d2n5_karman):
    center_x = 1.5
    center_y = 0.5
    radius = 0.1

    def border_func(self, x, y):
        area = np.abs(y-0.5) - 0.5       # 上下边界

        area[area < -x] = -x[area < -x]  # 左边界
        
        # 内接正三角形
        center1 = -(x-self.center_x)-np.abs(math.sqrt(3)*(y-self.center_y))+self.radius
        center2 = (0.5*self.radius+(x-self.center_x))
        center1[center2<center1]=center2[center2<center1]

        area[area<center1]=center1[area<center1]

        return area
    
test = model(h = 1/32, nu = 0.002)
test.save.prefix = f"karman"
test.save.dir = "graphic\\Karman\\04_triangler"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")