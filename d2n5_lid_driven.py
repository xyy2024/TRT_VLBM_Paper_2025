#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from d2n5_taylorgreen import d2n5_taylorgreen

class d2n5_lid_driven(d2n5_taylorgreen):
    def init_exact(self):
        return np.zeros(self.X.shape), np.zeros(self.Y.shape)
    
    u_border = 1

    def exact(self, x = None, y = None):
        if x is None: x = self.X
        if y is None: y = self.Y

        u = np.zeros(x.shape) 
        v = np.zeros(y.shape)
        u[y == 1] = self.u_border # We don't know the exact solution, this is only for boundary speed

        return u, v

    def border_func(self, x, y):
        x_area = np.abs(x-0.5) - 0.5
        y_area = np.abs(y-0.5) - 0.5

        x_area[x_area<y_area] = y_area[x_area<y_area]

        return x_area

