#!/usr/bin/python
# -*- coding: utf-8 -*-

from d2n5_couette import d2n5_couette as model
# from d2n5_poiseuille import d2n5_poiseuille as model
from _systools import cached_property
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