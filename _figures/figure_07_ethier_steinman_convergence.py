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

"""Render this figure using CSV only; no solver, checkpoint, TeX table, or old PDF input."""
import _bootstrap
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
from _artifact_io import (FIGURE_OUTPUT_DIR, read_3d)
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from _article_plots import ethier_convergence
    rows = read_3d('07')
    if any(row.status != 'completed' for row in rows):
        raise ValueError('All convergence cases must complete; failed runs cannot be silently omitted.')
    ethier_convergence([dict(choice=r.choice, n=r.n, h=r.h, velocity_error=r.velocity_error) for r in rows],
                      FIGURE_OUTPUT_DIR/'figure_07_ethier_steinman_convergence.pdf')
if __name__ == '__main__':
    main()

