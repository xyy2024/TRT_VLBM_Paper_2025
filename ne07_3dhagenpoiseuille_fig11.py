#!/usr/bin/python
# -*- coding: utf-8 -*-

from d3n7_hagenpoiseuille import d3n7_hagenpoiseuille as model
from _systools import cached_property
from _plottools import rankfig
import numpy as np

def prt(value):
    print("Error:", value)
    return value

def test(h_list, nu, file_id, average_relax_list):
    rank_graph = rankfig(prefix="HagenPoiseuille", supxlabel='', supylabel=('L2 Relative Error' if file_id == 0 else ''))
    rank_graph.save.file_id = file_id
    rank_graph[0].set_xlabel(f"$\\nu = {nu}$")
    h_array = np.array(h_list)

    for average_relax in average_relax_list:
        if not isinstance(average_relax, bool):
            print(f"{nu =}, {average_relax =}")
        elif average_relax:
            print(f"{nu =}, average_relax = default")
        err = []
        for h in h_list:
            test = model(h=h, nu=nu)
            if not isinstance(average_relax, bool):
                test.average_relax = average_relax
            elif not average_relax:
                test.average_relax = 0.5/test.tau
            if test.until_stable(max_step_delta= 1e-10,max_time=5e5):
                err.append(prt(test.get_error()))
            else:
                err.append(prt(float('nan')))
        print(err)
        if not isinstance(average_relax, bool):
            print(rank_graph[0].add_line(
                h = h_array,
                err = np.array(err),
                label=f'($\\phi + \\psi)/2 = {average_relax}$'
            ))
        elif average_relax:
            print(rank_graph[0].add_line(
                h = h_array,
                err = np.array(err),
                label=f'($\\phi + \\psi)/2 = \\frac{{2 + \\nu}}{{2 + 2\\nu}}$'
            ))
    rank_graph.fig.tight_layout()
    rank_graph.save()
    rank_graph.show(False)
    del rank_graph

h_list = [1/10, 1/25, 1/40, 1/55]
nu = 0.05
average_relax_list = [0.7, 0.9, True]
test(h_list, nu, 0, average_relax_list)

h_list = [1/10, 1/25, 1/40, 1/55]
nu = 0.1
average_relax_list = [0.9, 0.99, True]
test(h_list, nu, 1, average_relax_list)

h_list = [1/10, 1/25, 1/40, 1/55]
nu = 0.01
average_relax_list = [0.9, 0.99, True]
test(h_list, nu, 2, average_relax_list)