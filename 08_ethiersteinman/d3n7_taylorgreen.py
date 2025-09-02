#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Two-Relaxation-Time D3N7 VLBM Base Program
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
from _systools import cached_property
from _plottools import gridfig, show
from d2n5_taylorgreen import d2n5_taylorgreen

from typing import Callable

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
    def z(self): return np.arange(self.Nz)*self.dx + 0.5*self.dx + self.zmin
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

        self.w = np.zeros(self.shapew)
        self.w[:,:,:,0] = np.ones(self.shape)
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
    def get_p(self, w:np.ndarray|None = None):
        if w is None:
            w = self.w
        return (w[:,:,:,0] - 1)/self.h**2

    # 计算平衡分布
    def get_m(self, w:np.ndarray|None = None):
        if w is None:
            w = self.w
        Nx, Ny, Nz = w.shape[0:3]

        P = self.get_p(w)
        A1 = np.zeros(self.shapew)
        A2 = np.zeros(self.shapew)
        A3 = np.zeros(self.shapew)
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

        m = np.zeros((Nx, Ny, Nz, 7, 4))

        m[:,:,:,0,:] = self.a*w + 0.5*self.alpha*A1
        m[:,:,:,1,:] = self.a*w + 0.5*self.alpha*A2
        m[:,:,:,2,:] = self.a*w + 0.5*self.alpha*A3
        m[:,:,:,3,:] = self.a*w - 0.5*self.alpha*A1
        m[:,:,:,4,:] = self.a*w - 0.5*self.alpha*A2
        m[:,:,:,5,:] = self.a*w - 0.5*self.alpha*A3
        m[:,:,:,6,:] = (1-6*self.a)*w
        return m
    
    # 计算碰撞步骤的结果
    def get_fstar(self, m):
        fne = m - self.f
        fstar = self.f + self.relax1*fne + self.relax2*fne[:,:,:,self.opp,:]
        fstar[:,:,:,6,1:] += self.alpha*self.h**3*self.get_outerforce()
        return fstar
    
    ################################################################################################
    ###################################### 迁移步骤：处理边界 ######################################
    ################################################################################################

    @property
    def _general_border_property(self):
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

    @property
    def in_border_numtype_w(self): return np.zeros(self.shapew) + self.in_border_numtype[:,:,:,None]
    @property
    def borderZ(self): return self._general_border_property["borderZ"]

    def border_condition(self, nextf) -> np.ndarray:
        '''视情况使用Dirichlet边界或周期边界'''
        ub, vb, wb = self.exact(self.borderX, self.borderY, self.borderZ)
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
            )/(1+l)
        '''
        delta_rho_border = np.zeros((self.Nx, self.Ny, self.Nz, 7))
        delta_rho_border[:,:,:,0] += ub[:,:,:,0]
        delta_rho_border[:,:,:,1] += vb[:,:,:,1]
        delta_rho_border[:,:,:,2] += wb[:,:,:,2]
        delta_rho_border[:,:,:,3] -= ub[:,:,:,3]
        delta_rho_border[:,:,:,4] -= vb[:,:,:,4]
        delta_rho_border[:,:,:,5] -= wb[:,:,:,5]
        nextf[:,:,:,:,0] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,:,0]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,:,0]
                + (1 + self.l - 2*self.gamma)*self.f[:,:,:,self.opp,0]
                + (2*self.gamma - self.l)*self.fstar[:,:,:,self.opp,0]
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
        nextf[:,:,:,:,1] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,:,1]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,:,1]
                - (1 + self.l - 2*self.gamma)*self.f[:,:,:,self.opp,1]
                - (2*self.gamma - self.l)*self.fstar[:,:,:,self.opp,1]
                + 2*self.h*self.a*ub
            )/(1+self.l))
        nextf[:,:,:,:,2] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,:,2]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,:,2]
                - (1 + self.l - 2*self.gamma)*self.f[:,:,:,self.opp,2]
                - (2*self.gamma - self.l)*self.fstar[:,:,:,self.opp,2]
                + 2*self.h*self.a*vb
            )/(1+self.l))
        nextf[:,:,:,:,3] = (
            (1 - self.near_border_numtype)*nextf[:,:,:,:,3]
            + self.near_border_numtype*(
                self.l*self.fstar[:,:,:,:,3]
                - (1 + self.l - 2*self.gamma)*self.f[:,:,:,self.opp,3]
                - (2*self.gamma - self.l)*self.fstar[:,:,:,self.opp,3]
                + 2*self.h*self.a*wb
            )/(1+self.l))
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

    ################################################################################################
    ######################################### 绘制计算结果 #########################################
    ################################################################################################

    def fig_with_error(self, save_fig:bool = False, show_fig:bool = True, setting:None|Callable = None):
        if self.overflowed:
            L2_Relative_error = float('nan')
        else:
            u_pre, v_pre, w_pre = self.get_precise_speed()
            u_num, v_num, w_num = self.get_numerical_speed()
            
            u_err = u_pre - u_num
            v_err = v_pre - v_num
            w_err = w_pre - w_num

            L2_Relative_error = math.sqrt((u_err**2 + v_err**2 + w_err**2).sum()/(u_pre**2 + v_pre**2 + w_pre**2).sum())

            if L2_Relative_error > 1:
                L2_Relative_error = float('nan')
        
        if show_fig or save_fig:
            if setting is None:
                fig, axs = self.fig_default_setting_with_error(u_num, v_num, w_num, u_pre, v_pre, w_pre, u_err, v_err, w_err, L2_Relative_error)
            else:
                fig, axs = setting(self, u_num, v_num, w_num, u_pre, v_pre, w_pre, u_err, v_err, w_err, L2_Relative_error)
            self.save(fig = fig, save_fig = save_fig)
            show(fig = fig, show_fig = show_fig)
        return L2_Relative_error

    def fig_default_setting_with_error(self, u_num, v_num, w_num, u_pre, v_pre, w_pre, u_err, v_err, w_err, L2_Relative_error):
        fig, axs = gridfig(1, 1)
        axs[0].text(0.5, 0.5, "There is no default fig for D3 cases.")
        return fig, axs
    
    def fig_default_setting(self):
        # u_num, v_num, w_num = self.get_numerical_speed()
        fig, axs = gridfig(1, 1)
        axs[0].text(0.5, 0.5, "There is no default fig for D3 cases.")
        return fig, axs

if __name__ == "__main__":
    nu_list = [0.001, 0.01, 0.1]
    for nu in nu_list:
        test = d3n7_taylorgreen(h=0.1, nu=nu)
        test.until_time(1)
        print(test.get_error())