#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Two-Relaxation-Time D2N5 VLBM Base Program
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
from _systools import error_behavior, cached_property, printPercent
from _plottools import prt_2d, gridfig, prt_mask_2d, show, Save
from _nproll import circshift

from typing import Callable, Literal

import math

class d2n5_taylorgreen:
    '''D2N5 Solver Base Class
    D2N5 基本解算器'''

    ################################################################################################
    ################################### Basic Setting of *D2N5* ####################################
    ###################################     *D2N5* 基本设置     ####################################
    ################################################################################################

    ND = 2      # 维数        # Dimension Number
    NV = ND + 1 # 向量分量数  # Vector Components Number
    NE = 5      # 速度分量数  # Descrete Velocity Number
    Ex = ((1, 0, -1 ,0, 0),   # Descrete Velocity Direction
          (0, 1, 0, -1, 0))
    opp = (2, 3, 0, 1, 4)     # Oppocite Velocity Index

    @cached_property
    def E(self): return tuple(zip(*self.Ex))

    ################################################################################################
    ################################### Initializing Solver Nodes ##################################
    ###################################       求解空间设置        ##################################
    ################################################################################################

    # 
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    # Set the value of Δx and Δt
    # 设置Δx和Δt的数值
    def init_delta(self, *args):
        self.dx = self.h
        self.dt = self.alpha*self.h**2

    @property                                   # 给出当前时间
    def t(self): return self.iter_count*self.dt # Solver Time

    # Base Setting for Solver Nodes
    # 空间节点基本设置
    @cached_property
    def Nx(self): return math.ceil((self.xmax-self.xmin)/self.dx)
    @cached_property
    def Ny(self): return math.ceil((self.ymax-self.ymin)/self.dx)
    @cached_property
    def x(self): return np.arange(self.Nx)*self.dx + self.xmin + 0.5*self.dx
    @cached_property
    def y(self): return np.arange(self.Ny)*self.dx + self.ymin + 0.5*self.dx
    @property
    def shape(self): return self.Nx, self.Ny
    @property
    def shapew(self): return self.Nx, self.Ny, self.NV
    @property
    def shapef(self): return self.Nx, self.Ny, self.NE, self.NV

    # Make meshgrid for Solver Nodes
    # As a result: self.X[i, j] == self.x[i], self.Y[i, j] == self.y[j]
    # 通过空间节点设置生成网格
    # 结果：使得 X[i, j] == x[i], Y[i, j] == y[j]
    def init_node(self):
        '''初始化空间节点'''
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    ################################################################################################
    ###################################### Initializing Solver #####################################
    ######################################    求解器初始化     #####################################
    ################################################################################################

    # Convert Initial Velocity to Initial Distribution (Step 1)
    # 将初始速度变为初始平衡分布（的第一步）
    def init_value(self):
        '''处理初始状态'''
        u0, v0 = self.init_exact()

        self.w = np.zeros(self.shapew)
        self.w[:,:,0] = np.ones(self.shape)
        self.w[:,:,1] = u0*self.h*self.w[:,:,0]
        self.w[:,:,2] = v0*self.h*self.w[:,:,0]

    def __init__(self, h = 0.1, nu = 1/6):
        self.iter_count = 0

        self.h = h
        self.nu = nu

        self.init_relax()
        self.init_delta()
        self.init_node()
        self.init_value()

        self.f = self.get_m()
        self.save = Save(self.default_prefix)
    
    # 保存图像时的文件名前缀设置
    # NOTE: 如果要事后修改此项设置，请修改 self.save.prefix
    # File Prefix when Saving Figure
    # NOTE: If you need to change the prefix after solver instance was initialized. You can edit `self.save.prefix`
    @property
    def default_prefix(self):
        return 'test_result_' + self.__class__.__name__ + (f'_h{float(self.h):.5}nu{float(self.nu):.5}').replace('.','')

    ################################################################################################
    #################################### Initializing Relaxation ###################################
    ####################################       设置松弛系数       ##################################
    ################################################################################################

    # Automatically calculate the relaxation coefficient
    # 自动计算松弛系数
    def init_relax(self, *args):
        self.tau = (self.nu*self.alpha/self.a + 1)*0.5
        self.relax1 = self.average_relax + 0.5/self.tau
        self.relax2 = self.average_relax - 0.5/self.tau

    # NOTE: To achieve the following functions, use a modified cached_property decorator:
    #  1. Allow subclasses to override default values.
    #  2. Automatically update the relaxation coefficient after changing the attribute of solver instance.
    # NOTE: 如此写的目的是：
    #  1. 允许子类自行设置这些系数的默认值。
    #  2. 修改实例参数后、松弛系数和各种参数的自动更新.
    @property
    def default_average_relax(self): return 0.25/self.tau + 0.5
    default_alpha = 0.2
    default_a = 0.2
    
    @cached_property
    def alpha(self): return self.default_alpha
    @cached_property
    def a(self): return self.default_a
    @cached_property
    def average_relax(self): return self.default_average_relax

    alpha.run_after_set(init_relax)
    alpha.run_after_set(init_delta)
    a.run_after_set(init_relax)
    average_relax.run_after_set(init_relax)

    ################################################################################################
    ############################### Initial Velocity and Outer Force ###############################
    ###############################          初态与模型外力          ###############################
    ################################################################################################

    # Outer body force
    # 设置模型外力
    def get_outerforce(self):
        return np.zeros((self.Nx, self.Ny, 2))
    
    # Initial velocity
    # 设置初态
    def init_exact(self):
        # NOTE: 
        #   When override this method, you shall not pass any arguments
        #   but directly gives the result using expressions of `self.X`
        #   and `self.Y`. This method shall return `u, v`.
        #   在子类重载此项时，应当不传入任何值。
        #   而是直接从self.X和self.Y计算出结果，并return u, v
        return self.exact()
    
    # Precise solution for error analysis and get border velocity.
    # 设置精确解，用于给出边界速度与计算误差
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        # NOTE: 
        #   When override this method, you shall notice that the x and y passed
        #   in this method don't need to be `self.X` and `self.Y`
        #   在子类重载此项时，应当注意传入的 x 和 y 不一定是 self.X 和 self.Y

        if x is None: x = self.X
        if y is None: y = self.Y

        k = 2*math.pi
        bx = -0.5*math.pi
        U0 = 1

        scale = math.exp(-2*k**2*self.t*self.nu)*U0
        u = -np.cos(k*x+bx)*np.sin(k*y+bx)*scale
        v = np.sin(k*x+bx)*np.cos(k*y+bx)*scale

        return u, v
    
    # Border Position Setting
    # 设置边界
    def border_func(self, x:np.ndarray, y:np.ndarray):
        # NOTE: 
        #   If `self.border_func(x, y) > 0`, then (x, y) will be treated as it's out of border.
        #   If `self.border_func(x, y) == 0`, then (x, y) will be treated as it's on the Dirichlet border.
        #   函数值 <=0 的部分被视为在区域内。
        #   函数值 ==0 的部分被视为 Dirichlet 边界。

        #   If you want to set other kind of borders (rather than Dirichlet border), 
        #   you can override the `border_condition` method.
        #   如果要设置其它边界，请重载 border_condition 方法。
        
        #   在设置时，请使用可以传入 np.ndarray 的方式进行设置
        #   如果不可避免的要使用其它方式，可以用 np.vectorize 装饰器

        #   If `self.border_func(self.xmax, y) < 0`, then a periodic border is setted here.
        #   若在例如 x = self.xmax 处的函数值 < 0，则说明此处采用了循环边界条件。

        return 0*x-1  # 完全采用循环边界 # All Periodic Border as default
    
        # NOTE: 
        #   The following is a square Dirichlet boundary example.
        #   以下是示例的方形 Dirichlet 边界
        areax = np.abs(x-0.5)
        areay = np.abs(y-0.5)
        areax[areax < areay] = areay[areax < areay] # 取最大值 # Take the maximum
        return areax - 0.5
    
    # Setting if the Dirichlet border position will change over time.
    # NOTE: The function of changing boundaries over time is not yet fully developed. Please do not attempt to modify this.
    # 设置边界是否会随时间改变
    # NOTE: 边界随时间改变的功能暂不完善，请不要尝试修改此项
    border_type:Literal['static', 'dynamic'] = 'static'

    ################################################################################################
    ########################################## Collision ###########################################
    ##########################################  碰撞步骤 ###########################################
    ################################################################################################

    # Numerical Pressure
    # 计算数值压强
    def get_p(self, w:np.ndarray|None = None):
        if w is None:
            w = self.w
        return (w[:,:,0] - 1)/self.h**2
    
    # Equilibrium Distribution
    # 计算平衡分布
    def get_m(self, w:np.ndarray|None = None):
        if w is None:
            w = self.w

        P = self.get_p(w)
        A1 = np.zeros(w.shape)
        A2 = np.zeros(w.shape)
        A1[:,:,0] = w[:,:,1]
        A1[:,:,1] = (w[:,:,1]**2)/w[:,:,0] + self.h**2*P
        A1[:,:,2] = (w[:,:,1]*w[:,:,2])/w[:,:,0]
        A2[:,:,0] = w[:,:,2]
        A2[:,:,1] = A1[:,:,2]
        A2[:,:,2] = (w[:,:,2]**2)/w[:,:,0] + self.h**2*P

        m = np.zeros(w.shape[0:2] + (5, 3))

        m[:,:,0,:] = self.a*w + 0.5*self.alpha*A1
        m[:,:,1,:] = self.a*w + 0.5*self.alpha*A2
        m[:,:,2,:] = self.a*w - 0.5*self.alpha*A1
        m[:,:,3,:] = self.a*w - 0.5*self.alpha*A2
        m[:,:,4,:] = (1-4*self.a)*w
        return m
    
    # Result of the Collision Step
    # 计算碰撞步骤的结果
    def get_fstar(self, m):
        fne = m - self.f
        fstar = self.f + self.relax1*fne + self.relax2*fne[:,:,self.opp,:]
        fstar[:,:,4,1:] += self.alpha*self.h**3*self.get_outerforce()
        return fstar

    ################################################################################################
    ################################## Transport: Border Condition #################################
    ##################################     迁移步骤：处理边界      #################################
    ################################################################################################

    _general_border_property_value = None
    _general_border_property_time = None

    @property
    def _general_border_property(self):
        def border_property():
            this = dict() # 最终要 return this  # Will return `this` in the end.

            # If the nodes are out of border
            # 给出是否在区域内
            this["in_border"] = (self.border_func(self.X, self.Y) < 0)
            this["in_border_numtype"] = this["in_border"].astype(float)
            
            # 需要给出边界的 X,Y 坐标和 gamma
            # # 先给出所有坐标，然后裁切。
            Xin = self.X[:,:,None] + np.zeros((len(self.E),))[None,None,:]
            Yin = self.Y[:,:,None] + np.zeros((len(self.E),))[None,None,:]
            Gin = np.zeros((self.Nx, self.Ny, len(self.E)), dtype=float)
            Xout = Xin - np.array(self.Ex[0])[None,None,:]*self.dx
            Yout = Yin - np.array(self.Ex[1])[None,None,:]*self.dx
            Gout = np.ones((self.Nx, self.Ny, len(self.E)), dtype=float)

            # # 裁切：我们不需要考虑那些不与边界相邻的情况
            # # 裁切使用的内容，需要在边界处理时再次使用。所以在这里给出
            out_border = (self.border_func(Xout, Yout) > 0)                 # 传播自边界外格点
            this["near_border"] = out_border & this["in_border"][:,:,None]  # 传播自边界外格点 & 位于边界内
            this["near_border_numtype"] = this["near_border"].astype(float) # => 和边界相邻

            # # 执行裁切
            Xin[np.logical_not(this["near_border"])] *= 0
            Xout[np.logical_not(this["near_border"])] *= 0
            Yin[np.logical_not(this["near_border"])] *= 0
            Yout[np.logical_not(this["near_border"])] *= 0
            Gout[np.logical_not(this["near_border"])] *= 0

            # 用二分法求边界坐标，已经裁切的部分可以直接忽略
            for _ in range(52):
                Xmid = (Xout + Xin)*0.5
                Ymid = (Yout + Yin)*0.5
                Gmid = (Gout + Gin)*0.5
                Zmid = self.border_func(Xmid, Ymid)

                mid_out_border = (Zmid >= 0)
                Xout[mid_out_border] = Xmid[mid_out_border]
                Yout[mid_out_border] = Ymid[mid_out_border]
                Gout[mid_out_border] = Gmid[mid_out_border]

                mid_in_border = (Zmid <= 0)
                Xin[mid_in_border] = Xmid[mid_in_border]
                Yin[mid_in_border] = Ymid[mid_in_border]
                Gin[mid_in_border] = Gmid[mid_in_border]
            this["borderX"] = (Xout + Xin)*0.5
            this["borderY"] = (Yout + Yin)*0.5
            this["gamma"] = (Gout + Gin)*0.5

            # Stability for `l`:
            # l 的稳定性条件：
            #   l >= 0
            #   l >= 2*gamma - 1
            #   2*gamma >= l

            # If there is a point which gamma is 0.1
            # and another point where gamma is 0.9
            # then any constant l cannot be stable

            Lmax = Gout + Gin # gamma * 2
            Lmin = Lmax - 1   # gamma * 2 - 1
            Lmin[Lmin<0] = 0  # max(gamma * 2 - 1, 0)
            this["l"] = (Lmin + Lmax)*0.5

            return this
        if self.border_type == 'static':
            if self._general_border_property_value is None:
                self._general_border_property_value = border_property()
            return self._general_border_property_value
        #elif self.border_type == 'dynamic':
        #    if self._general_border_property_time != self.t:
        #        self._general_border_property_value = border_property()
        #        self._general_border_property_time = self.t
        #    return self._general_border_property_value
        else: 
            raise ValueError('Invalid value for attribute `border_type`.')
            
    '''
    def __getattr__(self, name):
        if name in self._general_border_property.keys():
            return self._general_border_property[name]
        else:
            raise AttributeError(f"'{self.__class__}' object has no attribute '{name}'")
    '''
    # To facilitate auto-completion, here we do not overload `__getattr__` but write them one by one. 
    # Anyway, the count of these properties is not large (even if it is large, we can write an automatic program to generate these things).
    # 为了便于自动补全，这里不重载 __getattr__ 而是一个一个写。反正数量不多（就算数量多，也可以写个自动程序来写这些东西）
    @property
    def in_border(self): return self._general_border_property["in_border"]
    @property
    def in_border_numtype(self): return self._general_border_property["in_border_numtype"]
    @property
    def in_border_numtype_w(self): return np.zeros(self.shapew) + self.in_border_numtype[:,:,None]
    @property
    def near_border(self): return self._general_border_property["near_border"]
    @property
    def near_border_numtype(self): return self._general_border_property["near_border_numtype"]
    @property
    def borderX(self): return self._general_border_property["borderX"]
    @property
    def borderY(self): return self._general_border_property["borderY"]
    @property
    def gamma(self): return self._general_border_property["gamma"]
    @property
    def l(self): return self._general_border_property["l"]

    # Dirichlet Border
    def border_condition(self, nextf) -> np.ndarray:
        '''Apply Dirichlet Border or Periodic Border which setted by method `border_func`
        视情况使用Dirichlet边界或周期边界'''
        ub, vb = self.exact(self.borderX, self.borderY)
        ''' 原始算式：
        nextf[x_index, y_index, i, 0] = (
            l*self.fstar[x_index, y_index, i, 0]
            +(1+l-2*gamma)*self.f[x_index, y_index, self.opp[i], 0]
            +(2*gamma-l)*self.fstar[x_index, y_index, self.opp[i], 0]
            +self.h*self.alpha*密度*法向速度*(-1)
            # 特别的：
            # i = 0 -> 法向速度*(-1) = u
            # i = 1 -> 法向速度*(-1) = v
            # i = 2 -> 法向速度*(-1) = -u
            # i = 3 -> 法向速度*(-1) = -v
            # 只计算近似不可压流，密度 = 1 + O(h^2)
            )/(1+l)
        '''
        delta_rho_border = np.zeros((self.Nx, self.Ny, 5))
        delta_rho_border[:,:,0] += ub[:,:,0]
        delta_rho_border[:,:,1] += vb[:,:,1]
        delta_rho_border[:,:,2] -= ub[:,:,2]
        delta_rho_border[:,:,3] -= vb[:,:,3]
        nextf[:,:,:,0] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,0]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,0]
                + (1 + self.l - 2*self.gamma)*self.f[:,:,self.opp,0]
                + (2*self.gamma - self.l)*self.fstar[:,:,self.opp,0]
                + self.h*self.alpha*delta_rho_border
            )/(1+self.l))
        ''' 原始算式：
        nextf[x_index, y_index, i, 1:] = (
            l*self.fstar[x_index, y_index, i, 1:]
            -(1+l-2*gamma)*self.f[x_index, y_index, self.opp[i], 1:]
            -(2*gamma-l)*self.fstar[x_index, y_index, self.opp[i], 1:]
            + 2*self.h*self.a*np.array([u,v])[None,:]
            # u, v 指的是边界处的速度
            )/(1+l)
        '''
        nextf[:,:,:,1] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,1]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,1]
                - (1 + self.l - 2*self.gamma)*self.f[:,:,self.opp,1]
                - (2*self.gamma - self.l)*self.fstar[:,:,self.opp,1]
                + 2*self.h*self.a*ub
            )/(1+self.l))
        nextf[:,:,:,2] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,2]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,2]
                - (1 + self.l - 2*self.gamma)*self.f[:,:,self.opp,2]
                - (2*self.gamma - self.l)*self.fstar[:,:,self.opp,2]
                + 2*self.h*self.a*vb
            )/(1+self.l))
        return nextf

    ################################################################################################
    ###################################### Iteration Process #######################################
    ######################################      迭代过程     #######################################
    ################################################################################################

    def _speed_index(self, i:int):
        return (slice(None), slice(None), i, slice(None))

    # Iterate Once
    # 单次迭代
    def iter(self):
        # Collision Step
        # 碰撞步骤
        self.m = self.get_m()                # 求平衡分布
        self.fstar = self.get_fstar(self.m)  # 求碰撞结果（这一步设置了模型外力）
        
        # Transport Step
        # 迁移步骤
        nextf = np.zeros(self.shapef)
        for i in range(self.NE):             # 按照周期边界条件作默认迁移
            nextf[self._speed_index(i)] = circshift(self.fstar[self._speed_index(i)], shift = self.E[i])
        nextf = self.border_condition(nextf) # 单独处理 Dirichlet 边界

        self.f = nextf

        # Update iter count and physical value
        # 迭代计数
        self.iter_count = self.iter_count + 1
        self.w = self.f.sum(axis=(self.ND,))
        return None

    # True if error occured when calculating
    # * The error could be not Overflow. 
    # ** This attribute named Overflow because the name doesn't matter so we don't want to change.
    # 记录计算过程中是否出现错误
    # *这个变量名不代表错误一定是 Overflow。
    # **只是在最开始测试时、出错都是 Overflow，所以起了变量名。现在不好改了而已。
    overflowed = False

    # Iteration Method for Different Stop Condition 
    # 不同停止条件下的迭代方法
    def until_step(self, step:int, show_progress:bool = True, catch_ctrl_c: bool = False):
        if self.overflowed:
            print(f'Already overflow.')
            return False
        with error_behavior(divide = 'raise', over = 'raise', invalid = 'raise', under = 'ignore'):
            if show_progress:
                printPercent(0, step, prefix='Iteration: ')
            for step_count in range(step):
                try:
                    self.iter()
                    if show_progress:
                        printPercent(step_count, step, prefix='\rIteration: ')
                except FloatingPointError as e:
                    printPercent(step_count, step, prefix='\rIteration: ', suffix=' stopped with error: ')
                    print(e, f'. Now at {self.t}', sep='')
                    self.overflowed = True
                    return False
                except KeyboardInterrupt as e:
                    if catch_ctrl_c:
                        print(f'\rIteration... stopped because user aborted. Now at {self.t}')
                        return True
                    raise
        if show_progress:
            print(f'\rIteration: 100.00% complete, the solver time is now {self.t}.')
        return True

    def until_time(self, time, show_progress:bool = True, catch_ctrl_c: bool = False):
        if self.overflowed:
            print(f'Already overflow.')
            return False
        with error_behavior(divide = 'raise', over = 'raise', invalid = 'raise', under = 'ignore'):
            if time < self.t:
                print(f'The solver time cannot decrease. Current solver time is {self.t}.')
                return True
            if time == self.t:
                print(f'The solver time is already {self.t}, no need for iteration.')
                return True
            start = self.t
            if show_progress:
                printPercent(self.t-start, time-start, prefix='Iteration: ')
            while self.t < time:
                try:
                    self.iter()
                    if show_progress:
                        printPercent(self.t-start, time-start, prefix='\rIteration: ')
                except FloatingPointError as e:
                    printPercent(self.t-start, time-start, prefix='\rIteration: ', suffix=' stopped with error: ')
                    print(e, f'. Now at {self.t}', sep='')
                    self.overflowed = True
                    return False
                except KeyboardInterrupt as e:
                    if catch_ctrl_c:
                        print(f'\rIteration... stopped because user aborted. Now at {self.t}')
                        return True
                    raise
        if show_progress:
            print(f'\rIteration: 100.00% complete, the solver time is now {self.t}.')
        return True

    def until_stable(self, max_step_delta:float = 1e-14, step:int = 1000, max_time:float = float('inf'), show_progress:bool = True, catch_ctrl_c:bool = False):
        if self.overflowed:
            print(f'Already overflow.')
            return False
        max_step_delta_sqr = max_step_delta**2

        def iter():
            for step_count in range(step):
                try:
                    self.iter()
                except FloatingPointError as e:
                    return e
            return None

        with error_behavior(divide = 'raise', over = 'raise', invalid = 'raise', under = 'ignore'):
            in_border = self.in_border_numtype_w
            previous = self.w
            if show_progress:
                dot_count = 1
                print('Iteration.', end='')
            iter()
            #while ((self.f - previous) > self.f*max_step_delta).any():
            try:
                while (((self.w - previous)*in_border)**2).sum() > ((self.w*in_border)**2).sum()*max_step_delta_sqr:
                    previous = self.w
                    if show_progress:
                        dot_count += 1
                        print('\rIteration'+'.'*dot_count+' '*(3-dot_count), end='')
                        if dot_count == 3: dot_count = 0
                    if (e:=iter()) is not None:
                        raise e
                    if self.t > max_time:
                        print('\rIteration... stopped because reached time limit:', max_time)
                        return True
            except FloatingPointError as e:
                print('\rIteration... stopped with error: ', e, f'. Now at {self.t}', sep='')
                self.overflowed = True
                return False
            except KeyboardInterrupt as e:
                if catch_ctrl_c:
                    print(f'\rIteration... stopped because user aborted. Now at {self.t}')
                    return True
                raise
        if show_progress:
            print(f'\rIteration complete, the result is stable at time {self.t}.')
        return True

    # Output animation after iteration
    # 输出动画
    _animation_last_save = -1
    def animation(self, time = 1, time_step = None, fps = 10, file_name:str|None = None, setting:Callable|None = None, show_progress:bool = True, catch_ctrl_c:bool = False, output_animation:bool = True):
        if self.overflowed:
            print(f'Already overflow.')
            return False
        with error_behavior(divide = 'raise', over = 'raise', invalid = 'raise', under = 'ignore'):
            if time < self.t:
                print(f'The solver time cannot decrease. Current solver time is {self.t}.')
                return True
            if time == self.t:
                print(f'The solver time is already {self.t}, no need for iteration.')
                return True
            if time_step is None: time_step = 1/fps
            if file_name is None: file_name = self.save.prefix + '.mp4'
            start = self.t
            next_step = start + time_step
            step_count = 1
            if self._animation_last_save < self.t:
                self.fig(save_fig=True, show_fig=False, setting=setting)
                self._animation_last_save = self.t
            if show_progress:
                printPercent(self.t-start, time-start, prefix='Iteration: ')
            while self.t < time:
                try:
                    self.iter()
                    if show_progress:
                        printPercent(self.t-start, time-start, prefix='\rIteration: ')
                    if self.t >= next_step:
                        step_count += 1
                        next_step = start + time_step*step_count
                        self.fig(save_fig=True, show_fig=False, setting=setting)
                        self._animation_last_save = self.t
                except FloatingPointError as e:
                    printPercent(self.t-start, time-start, prefix='\rIteration: ', suffix=' stopped with error: ')
                    print(e)
                    if output_animation:
                        self.save.animation(fps=fps, file_name=file_name)
                    self.overflowed = True
                    return False
                except KeyboardInterrupt as e:
                    if catch_ctrl_c:
                        print(f'\rIteration... stopped because user aborted. Now at {self.t}')
                        if output_animation:
                            self.save.animation(fps=fps, file_name=file_name)
                        return True
                    raise
        print(f'\rIteration: 100.00% complete, the solver time is now {self.t}.')
        if output_animation:
            self.save.animation(fps=fps, file_name=file_name)
        return True

    ################################################################################################
    ##################################### Get Calculate Result #####################################
    #####################################     获取计算结果     #####################################
    ################################################################################################

    def get_numerical_speed(self) -> tuple[np.ndarray, np.ndarray]:
        u_num = self.w[:,:,1]/self.w[:,:,0]/self.h*self.in_border_numtype
        v_num = self.w[:,:,2]/self.w[:,:,0]/self.h*self.in_border_numtype
        return u_num, v_num
    
    def get_precise_speed(self) -> tuple[np.ndarray, np.ndarray]:
        speeds = list(self.exact())
        
        for i in range(len(speeds)):
            speeds[i] = speeds[i]*self.in_border_numtype
        
        return tuple(speeds)

    def get_numerical_dencity(self) -> np.ndarray:
        return 1-(1-self.w[:,:,0])*self.in_border_numtype

    def get_error(self) -> float:
        return self.fig_with_error(False, False)
    
    def get_numerical_vorticity(self) -> np.ndarray:
        if self.E != d2n5_taylorgreen.E:
            raise 
        # 2D vorticity for D2N5
        u, v = self.get_numerical_speed()
        # vorticity = pv/px - pu/py
        vorticity = np.zeros(self.shape)

        du = (circshift(u, (0, -1)) - circshift(u, (0, 1)))*0.5  # Use second-order approximation with period border condition by default
        du[
            self.near_border[:,:,3]         # Hard coded with D2N5 indexes: y + dx outbound, using y and y - dx
            ] = (u - circshift(u, (0, 1)))[self.near_border[:,:,3]]
        du[
            self.near_border[:,:,1]         # Hard coded with D2N5 indexes: y - dx outbound, using y and y + dx
            ] = (circshift(u, (0, -1)) - u)[self.near_border[:,:,1]]
        du = du/self.dx*self.in_border_numtype

        dv = (circshift(v, (-1, 0)) - circshift(v, (1, 0)))*0.5
        dv[
            self.near_border[:,:,2]         # Hard coded with D2N5 indexes: x + dx outbound, using x and x - dx
            ] = (v - circshift(v, (1, 0)))[self.near_border[:,:,2]]
        dv[
            self.near_border[:,:,0]         # Hard coded with D2N5 indexes: x - dx outbound, using x and x + dx
            ] = (circshift(v, (-1, 0)) - v)[self.near_border[:,:,0]]
        dv = dv/self.dx*self.in_border_numtype

        vorticity = dv - du
        vorticity[np.logical_not(self.in_border)] = 0
        return vorticity

    ################################################################################################
    ######################################### Draw Figure  #########################################
    ######################################### 绘制计算结果 #########################################
    ################################################################################################

    def fig_with_error(self, save_fig:bool = False, show_fig:bool|None = True, setting:None|Callable = None):
        if self.overflowed:
            L2_Relative_error = float('nan')
        else:
            u_pre, v_pre = self.get_precise_speed()
            u_num, v_num = self.get_numerical_speed()
            
            u_err = u_pre - u_num
            v_err = v_pre - v_num

            L2_Relative_error = math.sqrt((u_err**2 + v_err**2).sum()/(u_pre**2 + v_pre**2).sum())

            if L2_Relative_error > 1:
                L2_Relative_error = float('nan')
        
        if show_fig or save_fig:
            if setting is None:
                fig, axs = self.fig_default_setting_with_error(u_num, v_num, u_pre, v_pre, u_err, v_err, L2_Relative_error)
            else:
                fig, axs = setting(self, u_num, v_num, u_pre, v_pre, u_err, v_err, L2_Relative_error)
            self.save(fig = fig, save_fig = save_fig)
            if show_fig is not None:
                show(fig = fig, show_fig = show_fig)
        return L2_Relative_error

    def fig_default_setting_with_error(self, u_num, v_num, u_pre, v_pre, u_err, v_err, L2_Relative_error):
        fig, axs = gridfig(4, 3, title=f"Result at t = {self.t:.5}")

        prt_2d(self.x, self.y, u_num, v_num, fig=fig, ax=axs[0], xlabel=f"Numerical")
        prt_2d(self.x, self.y, u_num, fig=fig, ax=axs[3])
        prt_2d(self.x, self.y, v_num, fig=fig, ax=axs[6])
        prt_2d(self.x, self.y, self.w[:,:,0], fig=fig, ax=axs[9])

        prt_2d(self.x, self.y, u_pre, v_pre, fig=fig, ax=axs[1], xlabel=f"Exact")
        prt_2d(self.x, self.y, u_pre, fig=fig, ax=axs[4])
        prt_2d(self.x, self.y, v_pre, fig=fig, ax=axs[7])
        prt_2d(self.x, self.y, np.ones((self.Nx, self.Ny)), fig=fig, ax=axs[10])

        prt_2d(self.x, self.y, u_err, v_err, fig=fig, ax=axs[2], xlabel=f"L2 Relative Error = {L2_Relative_error}")
        prt_2d(self.x, self.y, u_err, fig=fig, ax=axs[5])
        prt_2d(self.x, self.y, v_err, fig=fig, ax=axs[8])
        prt_2d(self.x, self.y, self.w[:,:,0] - 1, fig=fig, ax=axs[11])

        for ax in axs:
            prt_mask_2d(ax, (self.xmin, self.xmax), (self.ymin, self.ymax), lambda x, y: self.border_func(x, y))
        return fig, axs

    def fig(self, save_fig:bool = False, show_fig:bool = True, setting:None|Callable = None):
        if show_fig or save_fig:
            if setting is None:
                fig, axs = self.fig_default_setting()
            else:
                fig, axs = setting(self)
            self.save(fig = fig, save_fig = save_fig)
            if show_fig is not None:
                show(fig = fig, show_fig = show_fig)
        return None
    
    def fig_default_setting(self):
        u_num, v_num = self.get_numerical_speed()

        fig, axs = gridfig(2, 2, title=f"Numerical Result at t = {self.t:.5}")

        prt_2d(self.x, self.y, u_num, v_num, fig=fig, ax=axs[0], xlabel=f"streamplot")
        prt_2d(self.x, self.y, u_num, fig=fig, ax=axs[1], xlabel=f"u")
        prt_2d(self.x, self.y, v_num, fig=fig, ax=axs[2], xlabel="v")
        prt_2d(self.x, self.y, self.w[:,:,0], fig=fig, ax=axs[3], xlabel="rho")

        for ax in axs:
            prt_mask_2d(ax, (self.xmin, self.xmax), (self.ymin, self.ymax), lambda x, y: self.border_func(x, y))
        return fig, axs

