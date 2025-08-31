#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 01 (Taylor Green Vortex) Fig 02
#    Copyright (C) 2025 Xu Yuyang (https://github.com/xyy2024)

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

from d2n5_taylorgreen import d2n5_taylorgreen as model
from _plottools import rankfig, gridfig, prt_2d, prt_mask_2d
import numpy as np

def test(h_list, nu_list, file_id, average_relax_list):
    h_array = np.array(h_list)

    rank_graph = rankfig(cols=2, prefix="TaylorGreen", supxlabel='', log=f"graphic\\TaylorGreen_0000{file_id}_log.py")
    rank_graph.save.file_id = file_id
    rank_graph[0].set_xlabel(f"$\\nu = {nu_list[0]}$")
    rank_graph[1].set_xlabel(f"$\\nu = {nu_list[1]}$")

    for i in [-1, 0]:
        nu = nu_list[i]
        for average_relax in average_relax_list:
            err = []
            for h in h_list:
                test = model(h=h, nu=nu)
                if not isinstance(average_relax, bool):
                    test.average_relax = average_relax[0](test)
                if test.until_time(time=0.25):
                    err.append(test.get_error())
                else:
                    err.append(float('nan'))
            print(err)
            if not isinstance(average_relax, bool):
                print(rank_graph[i].add_line(
                    h = h_array,
                    err = np.array(err),
                    label=f'$(\\phi + \\psi)/2 = {average_relax[1]}$'
                ))
    rank_graph.fig.tight_layout()
    rank_graph.save()
    rank_graph.show(False)
    del rank_graph

h_list = [1/8, 1/12, 1/16, 1/24]
nu_list = [1/10, 1/40]
average_relax_list = [
    (lambda solver: (solver.nu + 2)/(2*solver.nu + 2), "\\frac{\\nu + 2}{2\\nu + 2}"),
    (lambda solver: 1 - solver.nu, "1 - \\nu"),
    (lambda solver: 1/(2*solver.nu + 1), "\\frac{1}{2\\nu + 1}"),
    (lambda solver: 0.75, "0.75")]

test(h_list, nu_list, 2, average_relax_list)

h_list = [1/32, 1/48, 1/64, 1/96]
nu_list = [1/160, 1/640]
average_relax_list = [
    (lambda solver: (solver.nu + 2)/(2*solver.nu + 2), "\\frac{\\nu + 2}{2\\nu + 2}"),
    (lambda solver: 1 - solver.nu, "1 - \\nu"),
    (lambda solver: 1/(2*solver.nu + 1), "\\frac{1}{2\\nu + 1}"),
    (lambda solver: 0.95, "0.95")]

test(h_list, nu_list, 3, average_relax_list)