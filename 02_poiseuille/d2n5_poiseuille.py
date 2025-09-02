#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from d2n5_taylorgreen import d2n5_taylorgreen
from _systools import cached_property

class d2n5_poiseuille(d2n5_taylorgreen):
    # Due to symmetry, there is no need for excessive sampling of x.
    @cached_property
    def Nx(self): return 1

    # The outerforce
    @cached_property
    def G(self):
        # 外力大小
        return 0.8*self.nu
    
    def get_outerforce(self):
        return np.array([self.G, 0])[None, None, :]

    def init_exact(self):
        return np.zeros(self.X.shape), np.zeros(self.Y.shape)

    def exact(self, x = None, y = None):
        if x is None: x = self.X
        if y is None: y = self.Y

        H = 1 # = ymax - ymin
        U = self.G*H**2/8/self.nu
        u = 4*U*(1-y/H)*y/H
        v = np.zeros(x.shape)
        return u, v

    def border_func(self, x, y):
        return np.abs(y-0.5) - 0.5

if __name__ == "__main__":
    test = d2n5_poiseuille(h = 0.05, nu = 0.002)

    test.until_stable(1e-11)
    print(test.get_error())

    test.until_stable(1e-12)
    print(test.get_error())

    test.until_stable(1e-13)
    print(test.get_error())

    test.until_stable(1e-14)
    print(test.get_error())