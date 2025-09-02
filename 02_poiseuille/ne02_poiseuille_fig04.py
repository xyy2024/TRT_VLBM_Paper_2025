#!/usr/bin/python
# -*- coding: utf-8 -*-

#    Test code for Numerical Experiment Section 02 (Poiseuille Flow) Fig 04
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

from d2n5_poiseuille import d2n5_poiseuille as model
from _systools import cached_property
from _plottools import rankfig
import numpy as np

h_list = [0.1, 0.05, 0.025, 0.0125]
h_array = np.array(h_list)

test = model(h=1, nu=0.01)
print("\\phi=",test.relax1, "\\\\\n\\psi=", test.relax2)
del test

rank_graph = rankfig(prefix="Poiseuille", supxlabel='', log=f"graphic\\Poiseuille_00002_log.py")
rank_graph.save.file_id = 2
for gamma_low in [0.01, 0.1, 0.3, 0.5]:
    class submodel(model):
        @cached_property
        def y(self): return np.arange(self.Ny)*self.dx + gamma_low*self.dx + self.ymin
    
    err = []
    for h in h_list:
        test = submodel(h=h, nu=0.01)
        test.until_stable(1e-14)
        err.append(test.get_error())
        print(err[-1])
    print(np.array([err[j] for j in range(len(h_list))]))
    print(rank_graph[0].add_line(
        h = h_array, 
        err = np.array([err[j] for j in range(len(h_list))]), 
        label=f'$\\gamma$ = {gamma_low}'))
rank_graph.fig.tight_layout()
rank_graph.save()
rank_graph.show(False)
del rank_graph