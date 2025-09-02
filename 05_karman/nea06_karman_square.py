#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Appendix 06
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

from d2n5_karman import d2n5_karman, np

class model(d2n5_karman):
    center_x = 1.5
    center_y = 0.5
    radius = 0.1

    def border_func(self, x, y):
        area = np.abs(y-0.5) - 0.5       # 上下边界

        area[area < -x] = -x[area < -x]  # 左边界
        
        # 外接正四边形
        center1 = self.radius-np.abs(y-self.center_y)
        center2 = self.radius-np.abs(x-self.center_x)
        center1[center2<center1]=center2[center2<center1]

        area[area<center1]=center1[area<center1]

        return area
    
test = model(h = 1/32, nu = 0.002)
test.save.prefix = f"karman"
test.save.dir = "graphic\\Karman\\06_square"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")