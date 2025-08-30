#!/usr/bin/python
# -*- coding: utf-8 -*-

from d2n5_karman import d2n5_karman

test = d2n5_karman(h = 1/32, nu = 0.002)
test.save.prefix = "karman"
test.save.dir = "graphic\\Karman\\00_default"
test.animation(time = 100, time_step = None, fps = 10, file_name = f"animation.mp4")