#!/usr/bin/python
# -*- coding: utf-8 -*-

from d2n5_karman import d2n5_karman, np, Normalize

class model(d2n5_karman):
    center_x = 1.5
    center_y = 0.3
    radius = 0.1

    def border_func(self, x, y):
        area = np.abs(y-0.5) - 0.5       # 上下边界

        area[area < -x] = -x[area < -x]  # 左边界
        
        center1 = self.radius**2-(y-self.center_y)**2-(x-self.center_x)**2
        area[area<center1]=center1[area<center1]

        center1 = self.radius**2-(y-1+self.center_y)**2-(x-self.center_x)**2
        area[area<center1]=center1[area<center1]

        return area

    vnorm = Normalize(vmin=-1.0, vmax=1.0)
    
test = model(h = 1/32, nu = 0.002)
test.save.prefix = f"karman"
test.save.dir = "graphic\\Karman\\02_colon"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")