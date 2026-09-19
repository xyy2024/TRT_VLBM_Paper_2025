#!/usr/bin/python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Xu Yuyang
#
# This file is part of the TRT-VLBM experiment reproduction code.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program (see LICENSE). If not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
from _systools import error_behavior, cached_property, printPercent
from _nproll import circshift

from typing import Literal

import math
import os
import pickle


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
    xshift = 0.5

    # Taylor--Green parameters from Section 4.3.  They are constructor
    # parameters so that both the initial density and velocity are built from
    # the same analytical solution when a different exact vortex is requested.
    U0 = 1.0
    k1 = 2*math.pi
    k2 = 2*math.pi
    b1 = -0.5*math.pi
    b2 = -0.5*math.pi
    initialization_protocol = 'parameterized-equilibrium-v1'

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
    def x(self): return np.arange(self.Nx)*self.dx + self.xmin + self.xshift*self.dx
    @cached_property
    def y(self): return np.arange(self.Ny)*self.dx + self.ymin + self.xshift*self.dx
    @property
    def shape(self): return self.Nx, self.Ny
    @property
    def shapew(self): return self.Nx, self.Ny, self.NV
    @property
    def shapef(self): return self.Nx, self.Ny, self.NE, self.NV

    # Make meshgrid for Solver Nodes
    # As a result: self.X[i, j] == self.x[i], self.Y[i, j] == self.y[j]
    #
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
        pressure0 = np.asarray(self.exact_pressure(), dtype=float)

        self.w = np.zeros(self.shapew)
        self.w[:,:,0] = 1 + self.h**2*pressure0
        self.w[:,:,1] = u0*self.h*self.w[:,:,0]
        self.w[:,:,2] = v0*self.h*self.w[:,:,0]

    def __init__(
        self,
        h=0.1,
        nu=1/6,
        xshift=None,
        *,
        U0=None,
        k1=None,
        k2=None,
        b1=None,
        b2=None,
        a=None,
        alpha=None,
        s_plus=None,
        ):
        self.iter_count = 0

        self.h = h
        self.nu = nu
        self.xshift = type(self).xshift if xshift is None else xshift
        self.U0 = type(self).U0 if U0 is None else U0
        self.k1 = type(self).k1 if k1 is None else k1
        self.k2 = type(self).k2 if k2 is None else k2
        self.b1 = type(self).b1 if b1 is None else b1
        self.b2 = type(self).b2 if b2 is None else b2
        # The equilibrium populations must be formed with the requested
        # parameters.  Store constructor overrides before init_relax(),
        # init_delta(), init_value(), and get_m() are called.
        if a is not None:
            self.__dict__['a'] = a
        if alpha is not None:
            self.__dict__['alpha'] = alpha
        if s_plus is not None:
            self._s_plus = s_plus
        self.initialization_protocol = type(self).initialization_protocol
        self._general_border_property_value = None
        self._general_border_property_time = None

        self.init_relax()
        self.init_delta()
        self.init_node()
        self.init_value()

        self.f = self.get_m()
        self.init_work_arrays()

    def init_work_arrays(self):
        '''Allocate arrays reused by the collision and transport steps.'''
        self.m = np.empty_like(self.f)
        self.fstar = np.empty_like(self.f)
        self.nextf = np.empty_like(self.f)
        self._A1 = np.empty_like(self.w)
        self._A2 = np.empty_like(self.w)
        if self.ND == 3:
            self._A3 = np.empty_like(self.w)

        self._outerforce = np.asarray(self.get_outerforce(), dtype=self.f.dtype).copy()
        self._force_term = np.empty_like(self._outerforce)

    def _ensure_work_arrays(self):
        '''Create work arrays missing from old saved solver instances.'''
        required = ('m', 'fstar', 'nextf', '_A1', '_A2')
        if self.ND == 3:
            required += ('_A3',)

        expected_shapes = {
            'm': self.f.shape,
            'fstar': self.f.shape,
            'nextf': self.f.shape,
            '_A1': self.w.shape,
            '_A2': self.w.shape,
            '_A3': self.w.shape,
        }
        if any(
            not isinstance(getattr(self, name, None), np.ndarray)
            or getattr(self, name).shape != expected_shapes[name]
            for name in required
        ):
            self.init_work_arrays()
            return

        if not isinstance(getattr(self, '_outerforce', None), np.ndarray):
            self._outerforce = np.asarray(self.get_outerforce(), dtype=self.f.dtype).copy()
        if (
            not isinstance(getattr(self, '_force_term', None), np.ndarray)
            or self._force_term.shape != self._outerforce.shape
        ):
            self._force_term = np.empty_like(self._outerforce)

    def refresh_outerforce(self):
        '''Refresh the cached value used by a static outer force.'''
        self._outerforce = np.asarray(self.get_outerforce(), dtype=self.f.dtype).copy()
        self._force_term = np.empty_like(self._outerforce)

    def _get_outerforce_array(self):
        if self.outerforce_type == 'static':
            return self._outerforce
        if self.outerforce_type == 'dynamic':
            return np.asarray(self.get_outerforce(), dtype=self.f.dtype)
        raise ValueError('Invalid value for attribute `outerforce_type`.')

    def save_instance(self, file: str | os.PathLike) -> None:
        '''Save the complete solver instance to *file*.

        The solver class, numerical fields, iteration state, cached values,
        user-adjusted parameters and experiment diagnostics are preserved.
        Only load files from trusted sources, since this method uses pickle.
        '''
        with open(file, 'wb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_instance(cls, file: str | os.PathLike):
        '''Rebuild and return a solver instance previously saved to *file*.

        Calling this method on a subclass also verifies that the saved solver
        is an instance of that subclass.  Pickle files from untrusted sources
        must not be loaded.
        '''
        with open(file, 'rb') as stream:
            instance = pickle.load(stream)

        if not isinstance(instance, cls):
            raise TypeError(
                f'The file contains {type(instance).__module__}.'
                f'{type(instance).__qualname__}, not an instance of '
                f'{cls.__module__}.{cls.__qualname__}.'
            )
        instance.__dict__.pop('save', None)
        instance.__dict__.pop('_animation_save', None)
        return instance
    
    ################################################################################################
    #################################### Initializing Relaxation ###################################
    ####################################       设置松弛系数       ##################################
    ################################################################################################

    # Automatically calculate the relaxation coefficient
    # 自动计算松弛系数
    def init_relax(self, *args):
        self.tau = (self.nu*self.alpha/self.a + 1)*0.5
        self.relax1 = 0.5*(self.s_plus + 1/self.tau)
        self.relax2 = 0.5*(self.s_plus - 1/self.tau)

    # NOTE: To achieve the following functions, use a modified cached_property decorator:
    #  1. Allow subclasses to override default values.
    #  2. Automatically update the relaxation coefficient after changing the attribute of solver instance.
    # NOTE: 如此写的目的是：
    #  1. 允许子类自行设置这些系数的默认值。
    #  2. 修改实例参数后、松弛系数和各种参数的自动更新.
    @property
    def default_s_plus(self): return 2 - 1/self.tau
    default_alpha = 0.2
    default_a = 0.2
    
    @cached_property
    def alpha(self): return self.default_alpha
    @cached_property
    def a(self): return self.default_a
    @property
    def s_plus(self):
        return getattr(self, '_s_plus', self.default_s_plus)

    @s_plus.setter
    def s_plus(self, value):
        self._s_plus = value
        self.init_relax()

    alpha.run_after_set(init_relax)
    alpha.run_after_set(init_delta)
    a.run_after_set(init_relax)

    ################################################################################################
    ############################### Initial Velocity and Outer Force ###############################
    ###############################          初态与模型外力          ###############################
    ################################################################################################

    # Outer body force
    # 设置模型外力
    outerforce_type:Literal['static', 'dynamic'] = 'static'

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
        r'''Return the Section 4.3 Taylor--Green velocity.'''
        # NOTE: 
        #   When override this method, you shall notice that the x and y passed
        #   in this method don't need to be `self.X` and `self.Y`
        #   在子类重载此项时，应当注意传入的 x 和 y 不一定是 self.X 和 self.Y

        if x is None: x = self.X
        if y is None: y = self.Y

        scale = self.U0*math.exp(
            -self.nu*self.t*(self.k1**2+self.k2**2)
        )
        u = -np.cos(self.k1*x+self.b1)*np.sin(self.k2*y+self.b2)*scale
        v = (
            (self.k1/self.k2)
            * np.sin(self.k1*x+self.b1)
            * np.cos(self.k2*y+self.b2)
            * scale
        )

        return u, v

    def exact_pressure(self, x:np.ndarray|None = None, y:np.ndarray|None = None):
        '''Return the exact pressure in the gauge stated in Section 4.3.'''
        if x is None: x = self.X
        if y is None: y = self.Y

        scale = math.exp(
            -2*self.nu*self.t*(self.k1**2+self.k2**2)
        )
        return -0.25*self.U0**2*(
            np.cos(2*self.k1*x+2*self.b1)
            + (self.k1/self.k2)**2*np.cos(2*self.k2*y+2*self.b2)
        )*scale
    
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

    ################################################################################################
    ########################################## Collision ###########################################
    ##########################################  碰撞步骤 ###########################################
    ################################################################################################

    # Numerical Pressure
    # 计算数值压强
    def get_numerical_pressure(self, w:np.ndarray|None = None):
        if w is None:
            w = self.w
        return (w[...,0] - 1)/self.h**2

    def get_p(self, w:np.ndarray|None = None):
        return self.get_numerical_pressure(w)
    
    # Equilibrium Distribution
    # 计算平衡分布
    def get_m(self, w:np.ndarray|None = None, out:np.ndarray|None = None):
        if w is None:
            w = self.w

        P = self.get_p(w)
        if (
            isinstance(getattr(self, '_A1', None), np.ndarray)
            and self._A1.shape == w.shape
        ):
            A1 = self._A1
            A2 = self._A2
        else:
            A1 = np.empty(w.shape, dtype=float)
            A2 = np.empty(w.shape, dtype=float)
        A1[:,:,0] = w[:,:,1]
        A1[:,:,1] = (w[:,:,1]**2)/w[:,:,0] + self.h**2*P
        A1[:,:,2] = (w[:,:,1]*w[:,:,2])/w[:,:,0]
        A2[:,:,0] = w[:,:,2]
        A2[:,:,1] = A1[:,:,2]
        A2[:,:,2] = (w[:,:,2]**2)/w[:,:,0] + self.h**2*P

        if out is None:
            m = np.empty(w.shape[0:2] + (5, 3), dtype=float)
        else:
            m = out

        m[:,:,0,:] = self.a*w + 0.5*self.alpha*A1
        m[:,:,1,:] = self.a*w + 0.5*self.alpha*A2
        m[:,:,2,:] = self.a*w - 0.5*self.alpha*A1
        m[:,:,3,:] = self.a*w - 0.5*self.alpha*A2
        m[:,:,4,:] = (1-4*self.a)*w
        return m
    
    # Result of the Collision Step
    # 计算碰撞步骤的结果
    def get_fstar(
        self,
        m,
        out:np.ndarray|None = None,
        fne_out:np.ndarray|None = None,
        ):
        if fne_out is None:
            fne = np.empty_like(self.f)
        else:
            fne = fne_out
        np.subtract(m, self.f, out=fne)

        if out is None:
            fstar = np.empty_like(self.f)
        else:
            fstar = out
        np.multiply(fne, self.relax1, out=fstar)
        np.add(fstar, self.f, out=fstar)
        for direction, opposite in enumerate(self.opp):
            fstar[:,:,direction,:] += self.relax2*fne[:,:,opposite,:]

        outerforce = self._get_outerforce_array()
        if self._force_term.shape != outerforce.shape:
            self._force_term = np.empty_like(outerforce)
        np.multiply(
            outerforce,
            self.alpha*self.h**3,
            out=self._force_term,
        )
        fstar[:,:,4,1:] += self._force_term
        return fstar

    ################################################################################################
    ################################## Transport: Border Condition #################################
    ##################################     迁移步骤：处理边界      #################################
    ################################################################################################

    border_type:Literal['static', 'dynamic'] = 'static'
    _general_border_property_value = None
    _general_border_property_time = None
    _shared_border_geometry_cache = {}

    # Names of additional scalar/tuple attributes which affect border_func.
    # Parameterized boundary subclasses can extend this tuple, for example:
    # border_geometry_parameters = ('center_x', 'center_y', 'radius')
    border_geometry_parameters = ()

    @staticmethod
    def _hashable_geometry_value(value):
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, list):
            return tuple(d2n5_taylorgreen._hashable_geometry_value(v) for v in value)
        if isinstance(value, tuple):
            return tuple(d2n5_taylorgreen._hashable_geometry_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted(
                (key, d2n5_taylorgreen._hashable_geometry_value(item))
                for key, item in value.items()
            ))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    def border_geometry_cache_key(self):
        '''Return the key used to share static boundary geometry.

        Subclasses whose boundary depends on instance attributes should list
        those names in ``border_geometry_parameters`` or override this method.
        '''
        axes = (
            (self.xmin, self.xmax, self.Nx),
            (self.ymin, self.ymax, self.Ny),
        )
        if self.ND == 3:
            axes += ((self.zmin, self.zmax, self.Nz),)

        boundary_method = self.border_func
        boundary_definition = getattr(boundary_method, '__func__', boundary_method)
        parameters = tuple(
            (name, self._hashable_geometry_value(getattr(self, name)))
            for name in self.border_geometry_parameters
        )
        return (
            type(self),
            self._hashable_geometry_value(boundary_definition),
            self.ND,
            self.NE,
            self._hashable_geometry_value(self.h),
            self._hashable_geometry_value(self.dx),
            self._hashable_geometry_value(self.xshift),
            self._hashable_geometry_value(axes),
            parameters,
        )

    @classmethod
    def clear_border_geometry_cache(cls):
        '''Discard geometry shared by static-boundary solver instances.'''
        cls._shared_border_geometry_cache.clear()

    def _calculate_general_border_property(self):
        this = {}

        in_border = self.border_func(self.X, self.Y) < 0
        this["in_border"] = in_border
        this["in_border_numtype"] = in_border.astype(float)

        ex = np.asarray(self.Ex[0])
        ey = np.asarray(self.Ex[1])

        # 保留完整布尔掩码，但每次只创建二维临时数组。
        near_border = np.empty(
            self.shape + (self.NE,),
            dtype=bool,
        )

        for direction in range(self.NE):
            near_border[:, :, direction] = (
                in_border
                & (
                    self.border_func(
                        self.X - ex[direction] * self.dx,
                        self.Y - ey[direction] * self.dx,
                    ) > 0
                )
            )

        this["near_border"] = near_border
        this["near_border_numtype"] = near_border.astype(float)

        # 每个数组都是一维的，长度等于实际边界链数量。
        border_index = np.nonzero(near_border)
        x_index, y_index, direction_index = border_index

        this["border_index"] = border_index
        this["border_opposite"] = np.asarray(
            self.opp,
            dtype=np.intp,
        )[direction_index]
        this["border_E"] = np.column_stack((
            ex[direction_index],
            ey[direction_index],
        ))

        # 为保持原接口兼容，以下结果仍保存为完整形状；
        # 但只有 border_index 处被赋值。
        borderX = np.zeros(near_border.shape)
        borderY = np.zeros(near_border.shape)
        gamma = np.zeros(near_border.shape)
        l_value = np.zeros(near_border.shape)

        # 全周期边界时没有边界链，可以立即返回。
        if direction_index.size == 0:
            this["borderX"] = borderX
            this["borderY"] = borderY
            this["gamma"] = gamma
            this["l"] = l_value
            return this

        # 以下数组长度仅为边界链数量。
        Xin = self.X[x_index, y_index].copy()
        Yin = self.Y[x_index, y_index].copy()
        Gin = np.zeros(direction_index.size)

        Xout = Xin - ex[direction_index] * self.dx
        Yout = Yin - ey[direction_index] * self.dx
        Gout = np.ones(direction_index.size)

        # 二分查找只处理实际边界链。
        for _ in range(52):
            Xmid = (Xout + Xin) * 0.5
            Ymid = (Yout + Yin) * 0.5
            Gmid = (Gout + Gin) * 0.5

            Zmid = self.border_func(Xmid, Ymid)

            mid_out_border = Zmid >= 0
            mid_in_border = Zmid <= 0

            Xout[mid_out_border] = Xmid[mid_out_border]
            Yout[mid_out_border] = Ymid[mid_out_border]
            Gout[mid_out_border] = Gmid[mid_out_border]

            Xin[mid_in_border] = Xmid[mid_in_border]
            Yin[mid_in_border] = Ymid[mid_in_border]
            Gin[mid_in_border] = Gmid[mid_in_border]

        borderX_link = (Xout + Xin) * 0.5
        borderY_link = (Yout + Yin) * 0.5
        gamma_link = (Gout + Gin) * 0.5

        Lmax = 2 * gamma_link
        Lmin = np.maximum(Lmax - 1, 0)
        l_link = (Lmin + Lmax) * 0.5

        # 只写入实际边界位置。
        borderX[border_index] = borderX_link
        borderY[border_index] = borderY_link
        gamma[border_index] = gamma_link
        l_value[border_index] = l_link

        this["borderX"] = borderX
        this["borderY"] = borderY
        this["gamma"] = gamma
        this["l"] = l_value

        return this

    @property
    def _general_border_property(self):
        if self.border_type == 'static':
            if (
                self._general_border_property_value is None
                or "border_index" not in self._general_border_property_value
            ):
                cache_key = self.border_geometry_cache_key()
                border_data = self._shared_border_geometry_cache.get(cache_key)
                if border_data is None:
                    border_data = self._calculate_general_border_property()
                    self._shared_border_geometry_cache[cache_key] = border_data
                self._general_border_property_value = border_data
            return self._general_border_property_value
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
    def border_index(self): return self._general_border_property["border_index"]
    @property
    def border_opposite(self): return self._general_border_property["border_opposite"]
    @property
    def border_E(self): return self._general_border_property["border_E"]
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
        '''Apply the boundary condition only at links crossing the boundary.'''
        border_data = self._general_border_property
        border_index = border_data["border_index"]
        direction_index = border_index[-1]

        # No boundary links means that all boundaries are periodic.
        if direction_index.size == 0:
            return nextf

        x_index, y_index, _ = border_index
        opposite_index = (
            x_index,
            y_index,
            border_data["border_opposite"],
        )

        borderX = border_data["borderX"][border_index]
        borderY = border_data["borderY"][border_index]
        gamma = border_data["gamma"][border_index]

        # Access self.l so subclasses can override the boundary parameter.
        l_full = np.broadcast_to(np.asarray(self.l), border_data["gamma"].shape)
        l_value = l_full[border_index]

        ub, vb = self.exact(borderX, borderY)
        ub = np.broadcast_to(np.asarray(ub), gamma.shape)
        vb = np.broadcast_to(np.asarray(vb), gamma.shape)
        wall_velocity = np.column_stack((ub, vb))

        delta_rho = np.sum(wall_velocity * border_data["border_E"], axis=1)
        denominator = 1 + l_value
        coefficient_old = 1 + l_value - 2 * gamma
        coefficient_star = 2 * gamma - l_value

        nextf[border_index + (0,)] = (
            l_value * self.fstar[border_index + (0,)]
            + coefficient_old * self.f[opposite_index + (0,)]
            + coefficient_star * self.fstar[opposite_index + (0,)]
            + self.h * self.alpha * delta_rho
        ) / denominator

        nextf[border_index + (slice(1, None),)] = (
            l_value[:, None] * self.fstar[border_index + (slice(1, None),)]
            - coefficient_old[:, None] * self.f[opposite_index + (slice(1, None),)]
            - coefficient_star[:, None] * self.fstar[opposite_index + (slice(1, None),)]
            + 2 * self.h * self.a * wall_velocity
        ) / denominator[:, None]
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
        self._ensure_work_arrays()
        # Collision Step
        # 碰撞步骤
        self.get_m(out=self.m)                       # 求平衡分布
        self.get_fstar(                              # 求碰撞结果（这一步设置了模型外力）
            self.m,
            out=self.fstar,
            fne_out=self.nextf,
        )
        
        # Transport Step
        # 迁移步骤
        nextf = self.nextf
        for i in range(self.NE):             # 按照周期边界条件作默认迁移
            speed_index = self._speed_index(i)
            circshift(
                self.fstar[speed_index],
                shift=self.E[i],
                out=nextf[speed_index],
            )
        nextf = self.border_condition(nextf) # 单独处理 Dirichlet 边界

        self.f, self.nextf = nextf, self.f

        # Update iter count and physical value
        # 迭代计数
        self.iter_count = self.iter_count + 1
        np.sum(self.f, axis=self.ND, out=self.w)
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


    # Output animation after iteration
    # 输出动画

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

    def get_error(self) -> np.ndarray:
        if self.overflowed:
            return np.full(3, float('nan'))

        precise_speed = self.get_precise_speed()
        numerical_speed = self.get_numerical_speed()
        pointwise_error_sqr = np.zeros(self.shape)
        for precise, numerical in zip(precise_speed, numerical_speed):
            pointwise_error_sqr += (numerical-precise)**2

        pointwise_error = np.sqrt(pointwise_error_sqr)
        cell_measure = self.h**self.ND
        return np.array((
            cell_measure*pointwise_error.sum(),
            math.sqrt(cell_measure*pointwise_error_sqr.sum()),
            pointwise_error.max(),
        ))

    
    def get_numerical_vorticity(self) -> np.ndarray:
        if self.E != d2n5_taylorgreen.E:
            raise NotImplementedError
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


    
