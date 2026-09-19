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
from _systools import cached_property
from d2n5_taylorgreen import d2n5_taylorgreen


import math

class d3n7_taylorgreen(d2n5_taylorgreen):
    '''D3N7 基本解算器'''

    ################################################################################################
    ######################################### 速度方向设置 #########################################
    ################################################################################################

    ND = 3
    NV = ND + 1
    NE = 7
    Ex = ((1, 0, 0, -1 ,0, 0, 0),
          (0, 1, 0, 0, -1, 0, 0),
          (0, 0, 1, 0, 0, -1, 0))
    opp = (3, 4, 5, 0, 1, 2, 6)

    ################################################################################################
    ######################################### 求解空间设置 #########################################
    ################################################################################################

    # 
    zmin, zmax = 0, 1

    # 空间节点基本设置
    @cached_property
    def Nz(self): return math.ceil((self.zmax-self.zmin)/self.dx)
    @cached_property
    def z(self): return np.arange(self.Nz)*self.dx + self.zmin + self.xshift*self.dx
    @property
    def shape(self): return self.Nx, self.Ny, self.Nz
    @property
    def shapew(self): return self.Nx, self.Ny, self.Nz, self.NV
    @property
    def shapef(self): return self.Nx, self.Ny, self.Nz, self.NE, self.NV

    # 将空间节点转化为网格
    # 结果：使得 X[i, j, k] == x[i], Y[i, j, k] == y[j], Z[i, j, k] == z[k]
    def init_node(self):
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

    # 处理初态
    def init_value(self):
        '''处理初始状态'''
        u0, v0, w0 = self.init_exact()
        pressure0 = np.asarray(self.exact_pressure(), dtype=float)

        self.w = np.zeros(self.shapew)
        self.w[:,:,:,0] = 1 + self.h**2*pressure0
        self.w[:,:,:,1] = u0*self.h*self.w[:,:,:,0]
        self.w[:,:,:,2] = v0*self.h*self.w[:,:,:,0]
        self.w[:,:,:,3] = w0*self.h*self.w[:,:,:,0]

    ################################################################################################
    ######################################### 设置松弛系数 #########################################
    ################################################################################################

    default_a = 1/7
    default_alpha = 1/7
    
    ################################################################################################
    ######################################## 初态与模型外力 ########################################
    ################################################################################################

    # 设置模型外力
    def get_outerforce(self):
        return np.zeros((self.Nx, self.Ny, self.Nz, 3))

    # 设置初态
    def init_exact(self):
        return self.exact()
    
    # 设置精确解，用于给出边界速度与计算误差
    def exact(self, x:np.ndarray|None = None, y:np.ndarray|None = None, z:np.ndarray|None = None):
        if x is None: x = self.X
        if y is None: y = self.Y
        if z is None: z = self.Z

        k = 2*math.pi
        bx = -0.5*math.pi
        U0 = 1

        scale = math.exp(-2*k**2*self.t*self.nu)*U0
        
        u = -np.cos(k*x+bx)*np.sin(k*y+bx)*scale
        v = np.sin(k*x+bx)*np.cos(k*y+bx)*scale
        w = np.zeros(z.shape)
        
        return u, v, w

    def exact_pressure(
        self,
        x:np.ndarray|None = None,
        y:np.ndarray|None = None,
        z:np.ndarray|None = None,
        ):
        if x is None: x = self.X
        if y is None: y = self.Y
        return super().exact_pressure(x, y)
    
    # 设置边界
    def border_func(self, x:np.ndarray, y:np.ndarray, z:np.ndarray):
        # NOTE: 
        #   函数值 <=0 的部分被视为在区域内。
        #   函数值 ==0 的部分被视为 Dirichlet 边界。

        #   如果要设置其它边界，请重载 border_condition 方法。
        
        #   在设置时，请使用可以传入 np.ndarray 的方式进行设置
        #   如果不可避免的要使用其它方式，可以用 np.vectorize 装饰器

        #   若在例如 x = self.xmax 处的函数值 < 0，则说明此处采用了循环边界条件。

        return 0*x-1  # 完全采用循环边界
    
        # NOTE: 
        #   以下是示例的方形 Dirichlet 边界
        areax = np.abs(x-0.5)
        areay = np.abs(y-0.5)
        areaz = np.abs(z-0.5)
        areax[areax < areay] = areay[areax < areay] # 取最大值
        areax[areax < areaz] = areaz[areax < areaz]
        return areax - 0.5

    ################################################################################################
    ########################################### 碰撞步骤 ###########################################
    ################################################################################################

    # 计算数值压强
    def get_numerical_pressure(self, w:np.ndarray|None = None):
        return super().get_numerical_pressure(w)

    def get_p(self, w:np.ndarray|None = None):
        return self.get_numerical_pressure(w)

    # 计算平衡分布
    def get_m(self, w:np.ndarray|None = None, out:np.ndarray|None = None):
        if w is None:
            w = self.w
        Nx, Ny, Nz = w.shape[0:3]

        P = self.get_p(w)
        if (
            isinstance(getattr(self, '_A1', None), np.ndarray)
            and self._A1.shape == w.shape
        ):
            A1 = self._A1
            A2 = self._A2
            A3 = self._A3
        else:
            A1 = np.empty(w.shape, dtype=float)
            A2 = np.empty(w.shape, dtype=float)
            A3 = np.empty(w.shape, dtype=float)
        A1[:,:,:,0] = w[:,:,:,1]
        A1[:,:,:,1] = (w[:,:,:,1]**2)/w[:,:,:,0] + self.h**2*P
        A1[:,:,:,2] = (w[:,:,:,1]*w[:,:,:,2])/w[:,:,:,0]
        A1[:,:,:,3] = (w[:,:,:,1]*w[:,:,:,3])/w[:,:,:,0]
        A2[:,:,:,0] = w[:,:,:,2]
        A2[:,:,:,1] = A1[:,:,:,2]
        A2[:,:,:,2] = (w[:,:,:,2]**2)/w[:,:,:,0] + self.h**2*P
        A2[:,:,:,3] = (w[:,:,:,2]*w[:,:,:,3])/w[:,:,:,0]
        A3[:,:,:,0] = w[:,:,:,3]
        A3[:,:,:,1] = A1[:,:,:,3]
        A3[:,:,:,2] = A2[:,:,:,3]
        A3[:,:,:,3] = (w[:,:,:,3]**2)/w[:,:,:,0] + self.h**2*P

        if out is None:
            m = np.empty((Nx, Ny, Nz, 7, 4), dtype=float)
        else:
            m = out

        m[:,:,:,0,:] = self.a*w + 0.5*self.alpha*A1
        m[:,:,:,1,:] = self.a*w + 0.5*self.alpha*A2
        m[:,:,:,2,:] = self.a*w + 0.5*self.alpha*A3
        m[:,:,:,3,:] = self.a*w - 0.5*self.alpha*A1
        m[:,:,:,4,:] = self.a*w - 0.5*self.alpha*A2
        m[:,:,:,5,:] = self.a*w - 0.5*self.alpha*A3
        m[:,:,:,6,:] = (1-6*self.a)*w
        return m
    
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
            fstar[:,:,:,direction,:] += self.relax2*fne[:,:,:,opposite,:]

        outerforce = self._get_outerforce_array()
        if self._force_term.shape != outerforce.shape:
            self._force_term = np.empty_like(outerforce)
        np.multiply(
            outerforce,
            self.alpha*self.h**3,
            out=self._force_term,
        )
        fstar[:,:,:,6,1:] += self._force_term
        return fstar
    
    ################################################################################################
    ###################################### 迁移步骤：处理边界 ######################################
    ################################################################################################

    @property
    def _general_border_property_full(self):
        def border_property():
            this = dict() # 最终要 return this

            # 给出是否在区域内
            this["in_border"] = (self.border_func(self.X, self.Y, self.Z) < 0)
            this["in_border_numtype"] = this["in_border"].astype(float)
            
            # 需要给出边界的 X,Y 坐标和 gamma
            # # 给出所有坐标：我们先给出所有坐标，然后裁切。
            Xin = self.X[:,:,:,None] + np.zeros((len(self.E),))[None,None,None,:]
            Yin = self.Y[:,:,:,None] + np.zeros((len(self.E),))[None,None,None,:]
            Zin = self.Z[:,:,:,None] + np.zeros((len(self.E),))[None,None,None,:]
            Gin = np.zeros((self.Nx, self.Ny, self.Nz, len(self.E)), dtype=float)
            Xout = Xin - np.array(self.Ex[0])[None,None,:]*self.dx
            Yout = Yin - np.array(self.Ex[1])[None,None,:]*self.dx
            Zout = Zin - np.array(self.Ex[2])[None,None,:]*self.dx
            Gout = np.ones((self.Nx, self.Ny, self.Nz, len(self.E)), dtype=float)

            # # 裁切：我们不需要考虑那些不与边界相邻的情况
            # # 裁切使用的内容，需要在边界处理时再次使用。所以在这里给出
            out_border = (self.border_func(Xout, Yout, Zout) > 0)                 # 传播自边界外格点
            this["near_border"] = out_border & this["in_border"][:,:,:,None]  # 传播自边界外格点 & 位于边界内
            this["near_border_numtype"] = this["near_border"].astype(float) # => 和边界相邻

            # # 执行裁切
            Xin[np.logical_not(this["near_border"])] *= 0
            Xout[np.logical_not(this["near_border"])] *= 0
            Yin[np.logical_not(this["near_border"])] *= 0
            Yout[np.logical_not(this["near_border"])] *= 0
            Zin[np.logical_not(this["near_border"])] *= 0
            Zout[np.logical_not(this["near_border"])] *= 0
            Gout[np.logical_not(this["near_border"])] *= 0

            # 用二分法求边界坐标，已经裁切的部分可以直接忽略
            for _ in range(52):
                Xmid = (Xout + Xin)*0.5
                Ymid = (Yout + Yin)*0.5
                Zmid = (Zout + Zin)*0.5
                Gmid = (Gout + Gin)*0.5
                Bmid = self.border_func(Xmid, Ymid, Zmid)
                
                mid_out_border = (Bmid >= 0)
                Xout[mid_out_border] = Xmid[mid_out_border]
                Yout[mid_out_border] = Ymid[mid_out_border]
                Zout[mid_out_border] = Zmid[mid_out_border]
                Gout[mid_out_border] = Gmid[mid_out_border]

                mid_in_border = (Bmid <= 0)
                Xin[mid_in_border] = Xmid[mid_in_border]
                Yin[mid_in_border] = Ymid[mid_in_border]
                Zin[mid_in_border] = Zmid[mid_in_border]
                Gin[mid_in_border] = Gmid[mid_in_border]
            this["borderX"] = (Xout + Xin)*0.5
            this["borderY"] = (Yout + Yin)*0.5
            this["borderZ"] = (Zout + Zin)*0.5
            this["gamma"] = (Gout + Gin)*0.5

            Lmax = Gout + Gin
            Lmin = Lmax - 1
            Lmin[Lmin<0] = 0
            this["l"] = (Lmin + Lmax)*0.5

            return this
        if self.border_type == 'static':
            if self._general_border_property_value is None:
                self._general_border_property_value = border_property()
            return self._general_border_property_value
        else: #elif self.border_type == 'dynamic':
            if self._general_border_property_time != self.t:
                self._general_border_property_value = border_property()
                self._general_border_property_time = self.t
            return self._general_border_property_value

    def _calculate_general_border_property(self):
        this = {}

        in_border = self.border_func(self.X, self.Y, self.Z) < 0
        this["in_border"] = in_border
        this["in_border_numtype"] = in_border.astype(float)

        ex = np.asarray(self.Ex[0])
        ey = np.asarray(self.Ex[1])
        ez = np.asarray(self.Ex[2])

        near_border = np.empty(self.shape + (self.NE,), dtype=bool)
        for direction in range(self.NE):
            near_border[:, :, :, direction] = (
                in_border
                & (
                    self.border_func(
                        self.X - ex[direction] * self.dx,
                        self.Y - ey[direction] * self.dx,
                        self.Z - ez[direction] * self.dx,
                    ) > 0
                )
            )

        this["near_border"] = near_border
        this["near_border_numtype"] = near_border.astype(float)

        border_index = np.nonzero(near_border)
        x_index, y_index, z_index, direction_index = border_index
        this["border_index"] = border_index
        this["border_opposite"] = np.asarray(
            self.opp,
            dtype=np.intp,
        )[direction_index]
        this["border_E"] = np.column_stack((
            ex[direction_index],
            ey[direction_index],
            ez[direction_index],
        ))

        borderX = np.zeros(near_border.shape)
        borderY = np.zeros(near_border.shape)
        borderZ = np.zeros(near_border.shape)
        gamma = np.zeros(near_border.shape)
        l_value = np.zeros(near_border.shape)

        if direction_index.size == 0:
            this["borderX"] = borderX
            this["borderY"] = borderY
            this["borderZ"] = borderZ
            this["gamma"] = gamma
            this["l"] = l_value
            return this

        Xin = self.X[x_index, y_index, z_index].copy()
        Yin = self.Y[x_index, y_index, z_index].copy()
        Zin = self.Z[x_index, y_index, z_index].copy()
        Gin = np.zeros(direction_index.size)

        Xout = Xin - ex[direction_index] * self.dx
        Yout = Yin - ey[direction_index] * self.dx
        Zout = Zin - ez[direction_index] * self.dx
        Gout = np.ones(direction_index.size)

        for _ in range(52):
            Xmid = (Xout + Xin) * 0.5
            Ymid = (Yout + Yin) * 0.5
            Zmid = (Zout + Zin) * 0.5
            Gmid = (Gout + Gin) * 0.5
            Bmid = self.border_func(Xmid, Ymid, Zmid)

            mid_out_border = Bmid >= 0
            mid_in_border = Bmid <= 0

            Xout[mid_out_border] = Xmid[mid_out_border]
            Yout[mid_out_border] = Ymid[mid_out_border]
            Zout[mid_out_border] = Zmid[mid_out_border]
            Gout[mid_out_border] = Gmid[mid_out_border]

            Xin[mid_in_border] = Xmid[mid_in_border]
            Yin[mid_in_border] = Ymid[mid_in_border]
            Zin[mid_in_border] = Zmid[mid_in_border]
            Gin[mid_in_border] = Gmid[mid_in_border]

        borderX_link = (Xout + Xin) * 0.5
        borderY_link = (Yout + Yin) * 0.5
        borderZ_link = (Zout + Zin) * 0.5
        gamma_link = (Gout + Gin) * 0.5

        Lmax = 2 * gamma_link
        Lmin = np.maximum(Lmax - 1, 0)
        l_link = (Lmin + Lmax) * 0.5

        borderX[border_index] = borderX_link
        borderY[border_index] = borderY_link
        borderZ[border_index] = borderZ_link
        gamma[border_index] = gamma_link
        l_value[border_index] = l_link

        this["borderX"] = borderX
        this["borderY"] = borderY
        this["borderZ"] = borderZ
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
        else: # dynamic border
            if (
                self._general_border_property_time != self.t
                or self._general_border_property_value is None
                or "border_index" not in self._general_border_property_value
            ):
                self._general_border_property_value = self._calculate_general_border_property()
                self._general_border_property_time = self.t
            return self._general_border_property_value

    @property
    def in_border_numtype_w(self): return np.zeros(self.shapew) + self.in_border_numtype[:,:,:,None]
    @property
    def borderZ(self): return self._general_border_property["borderZ"]

    def border_condition(self, nextf) -> np.ndarray:
        '''Apply the boundary condition only at links crossing the boundary.'''
        border_data = self._general_border_property
        border_index = border_data["border_index"]
        direction_index = border_index[-1]

        if direction_index.size == 0:
            return nextf

        x_index, y_index, z_index, _ = border_index
        opposite_index = (
            x_index,
            y_index,
            z_index,
            border_data["border_opposite"],
        )

        borderX = border_data["borderX"][border_index]
        borderY = border_data["borderY"][border_index]
        borderZ = border_data["borderZ"][border_index]
        gamma = border_data["gamma"][border_index]

        # Access self.l so subclasses can override the boundary parameter.
        l_full = np.broadcast_to(np.asarray(self.l), border_data["gamma"].shape)
        l_value = l_full[border_index]

        ub, vb, wb = self.exact(borderX, borderY, borderZ)
        ub = np.broadcast_to(np.asarray(ub), gamma.shape)
        vb = np.broadcast_to(np.asarray(vb), gamma.shape)
        wb = np.broadcast_to(np.asarray(wb), gamma.shape)
        wall_velocity = np.column_stack((ub, vb, wb))

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
    ########################################### 迭代过程 ###########################################
    ################################################################################################

    def _speed_index(self, i:int):
        return (slice(None), slice(None), slice(None), i, slice(None))

    ################################################################################################
    ######################################### 获取计算结果 #########################################
    ################################################################################################

    def get_numerical_speed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_num = self.w[:,:,:,1]/self.w[:,:,:,0]/self.h*self.in_border_numtype
        v_num = self.w[:,:,:,2]/self.w[:,:,:,0]/self.h*self.in_border_numtype
        w_num = self.w[:,:,:,3]/self.w[:,:,:,0]/self.h*self.in_border_numtype
        return u_num, v_num, w_num
    
    def get_numerical_dencity(self) -> np.ndarray:
        return 1-(1-self.w[:,:,:,0])*self.in_border_numtype

    def get_precise_speed(self):
        return super().get_precise_speed()

    def get_error(self) -> np.ndarray:
        return super().get_error()


    ################################################################################################
    ######################################### 绘制计算结果 #########################################
    ################################################################################################


    

