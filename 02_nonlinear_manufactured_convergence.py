# SPDX-License-Identifier: GPL-3.0-or-later
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

"""Manufactured convergence with optional saved-state continuation."""
from _periodic_config import A, NU, rates
from _periodic_convergence import run_experiment
from _paths import configure_experiment
from d2n5_nonlinearmanufactured import d2n5_nonlinearmanufactured


def create_solver(method, alpha, n):
    s_plus, _ = rates(method, alpha)
    return d2n5_nonlinearmanufactured(h=1/n, nu=NU, a=A, alpha=alpha, s_plus=s_plus)


def main():
    args = configure_experiment(periodic=True)
    run_experiment(create_solver, "02_nonlinear_manufactured", "NonlinearManufactured", args)


if __name__ == "__main__":
    main()
