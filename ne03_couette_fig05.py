#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 03 (Couette Flow) Fig 05
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

from d2n5_couette import d2n5_couette as model
from _plottools import rankfig
import numpy as np

def test(h_list, nu, file_id, average_relax_list):
    h_array = np.array(h_list)

    rank_graph = rankfig(prefix="Couette", supxlabel='', supylabel=('L2 Relative Error' if file_id == 0 else ''), log=f"graphic\\Couette_0000{file_id}_log.py")
    rank_graph.save.file_id = file_id
    rank_graph[0].set_xlabel(f"$\\nu = {nu}$")

    for average_relax in average_relax_list:
        err = []
        for h in h_list:
            test = model(h=h, nu=nu)
            test.average_relax = average_relax[0](test)
            if test.until_stable(max_time=5e5):
                err.append(test.get_error())
            else:
                err.append(float('nan'))
        print(err)
        print(rank_graph[0].add_line(
            h = h_array,
            err = np.array(err),
            label=f'$(\\phi + \\psi)/2 = {average_relax[1]}$'
        ))
    rank_graph.fig.tight_layout()
    rank_graph.save()
    rank_graph.show(False)
    del rank_graph

h_list = [1/8, 1/12, 1/16, 1/24]
nu = 1/40
average_relax_list = [
    (lambda solver: (solver.nu + 2)/(2*solver.nu + 2), "\\frac{\\nu + 2}{2\\nu + 2}"),
    (lambda solver: 1 - solver.nu, "1 - \\nu"),
    (lambda solver: 1/(2*solver.nu + 1), "\\frac{1}{2\\nu + 1}"),
    (lambda solver: 0.75, "0.75")]
test(h_list, nu, 0, average_relax_list)

h_list = [1/32, 1/48, 1/64, 1/96]
nu = 1/160
average_relax_list = [
    (lambda solver: (solver.nu + 2)/(2*solver.nu + 2), "\\frac{\\nu + 2}{2\\nu + 2}"),
    (lambda solver: 1 - solver.nu, "1 - \\nu"),
    (lambda solver: 1/(2*solver.nu + 1), "\\frac{1}{2\\nu + 1}"),
    (lambda solver: 0.95, "0.95")]
test(h_list, nu, 1, average_relax_list)