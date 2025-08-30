#!/usr/bin/python
# -*- coding: utf-8 -*-

#    d3n7_ethiersteinman.py
#    Copyright (C) 2025 Xu Yuyang

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

import numpy as np
from _plottools import gridfig
from d3n7_taylorgreen import d3n7_taylorgreen

import math

class d3n7_ethiersteinman(d3n7_taylorgreen):
    # 设置初态
    def init_exact(self):
        return self.exact()
    
    # 设置精确解
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None, z:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        if z is None: z = self.Z

        a = 1
        d = 1

        scale = math.exp(-self.nu*d**2*self.t)

        u1 = -a*(np.exp(a*x)*np.sin(a*y+d*z)+np.exp(a*z)*np.cos(a*x+d*y))*scale
        u2 = -a*(np.exp(a*y)*np.sin(a*z+d*x)+np.exp(a*x)*np.cos(a*y+d*z))*scale
        u3 = -a*(np.exp(a*z)*np.sin(a*x+d*y)+np.exp(a*y)*np.cos(a*z+d*x))*scale
        
        return u1, u2, u3

    # 设置边界情况
    def border_func(self, x, y, z):
        areax = np.abs(x-0.5)
        areay = np.abs(y-0.5)
        areaz = np.abs(z-0.5)
        areax[areax < areay] = areay[areax < areay] # 取最大值
        areax[areax < areaz] = areaz[areax < areaz] # 取最大值
        return areax - 0.5

class d3n7_ethiersteinman2(d3n7_taylorgreen):
    xmax = 2
    ymax = 2
    zmax = 2
    # 设置初态
    def init_exact(self):
        return self.exact()
    
    # 设置精确解
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None, z:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        if z is None: z = self.Z

        a = math.pi
        d = math.pi

        scale = math.exp(-self.nu*d**2*self.t)

        u1 = -a*(np.exp(a*x)*np.sin(a*y+d*z)+np.exp(a*z)*np.cos(a*x+d*y))*scale
        u2 = -a*(np.exp(a*y)*np.sin(a*z+d*x)+np.exp(a*x)*np.cos(a*y+d*z))*scale
        u3 = -a*(np.exp(a*z)*np.sin(a*x+d*y)+np.exp(a*y)*np.cos(a*z+d*x))*scale
        
        return u1, u2, u3

    # 设置边界情况
    def border_func(self, x, y, z):
        areax = np.abs(x-1)
        areay = np.abs(y-1)
        areaz = np.abs(z-1)
        areax[areax < areay] = areay[areax < areay] # 取最大值
        areax[areax < areaz] = areaz[areax < areaz] # 取最大值
        return areax - 1

if __name__ == "__main__":
    nu_list = [0.03,]
    for nu in nu_list:
        test1 = d3n7_ethiersteinman(h=0.1, nu=nu)
        test2 = d3n7_ethiersteinman(h=0.05, nu=nu)
        test1.until_time(0.1)
        test2.until_time(0.1)
        print(test1.get_error())
        print(test2.get_error())
    # if False:
        test1.until_time(0.2)
        test2.until_time(0.2)
        print(test1.get_error())
        print(test2.get_error())

        test1.until_time(0.3)
        test2.until_time(0.3)
        print(test1.get_error())
        print(test2.get_error())

        test1.until_time(0.4)
        test2.until_time(0.4)
        print(test1.get_error())
        print(test2.get_error())