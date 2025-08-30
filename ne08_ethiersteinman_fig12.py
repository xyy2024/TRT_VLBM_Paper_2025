#!/usr/bin/python
# -*- coding: utf-8 -*-

from d3n7_ethiersteinman import d3n7_ethiersteinman as model
from _plottools import rankfig, gridfig, prt_2d, prt_mask_2d
import numpy as np

def test(h_list, nu_list, file_id, average_relax_list):
    h_array = np.array(h_list)

    rank_graph = rankfig(cols=2, prefix="EthierSteinman", supxlabel='', log=f"graphic\\EthierSteinman_0000{file_id}_log.py")
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
                if test.until_time(time=0.3):
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

h_list = [1/16, 1/32, 1/48, 1/64]
nu_list = [1/100, 1/500]
average_relax_list = [
    (lambda solver: (solver.nu + 2)/(2*solver.nu + 2), "\\frac{\\nu + 2}{2\\nu + 2}"),
    (lambda solver: 1 - solver.nu, "1 - \\nu"),
    (lambda solver: 1/(2*solver.nu + 1), "\\frac{1}{2\\nu + 1}"),
    (lambda solver: 0.95, "0.95")]

test(h_list, nu_list, 0, average_relax_list)
