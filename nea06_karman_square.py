#!/usr/bin/python
# -*- coding: utf-8 -*-

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
test.save.dir = "graphic\\Karman_Appendix\\06_square"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")