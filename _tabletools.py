#!/usr/bin/python
# -*- coding: utf-8 -*-
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

'''Utilities for writing convergence data as LaTeX tables.'''

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


__all__ = ['poiseuille_tuning_table']


def _as_finite_float(
    name: str,
    value,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    ) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f'{name} must be finite; received {value!r}.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive; received {value!r}.')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be non-negative; received {value!r}.')
    return value


def _scientific(value: float, precision: int) -> str:
    mantissa, exponent = f'{value:.{precision}e}'.split('e')
    return rf'{mantissa}\mathrm{{e}}{{{int(exponent)}}}'


@dataclass(frozen=True)
class _poiseuille_tuning_row:
    ell: float
    choice: str
    s_plus: float
    e_u2: float
    order: float | None
    fitted_slip: float
    predicted_slip: float


class poiseuille_tuning_table:
    '''LaTeX table for the Section 4 Poiseuille boundary-tuning test.'''

    def __init__(
        self, *, dir='.',
        file_name='table_05_poiseuille_boundary_tuning.tex',
    ):
        self.dir = Path(dir)
        self.file_name = Path(file_name)
        self.data: list[_poiseuille_tuning_row] = []

    def add_row(self, ell, choice, s_plus, e_u2, order, fitted_slip,
                predicted_slip):
        self.data.append(_poiseuille_tuning_row(
            _as_finite_float('ell', ell, nonnegative=True),
            str(choice),
            _as_finite_float('s_plus', s_plus, positive=True),
            _as_finite_float('e_u2', e_u2, positive=True),
            None if order is None else _as_finite_float('order', order),
            _as_finite_float('fitted_slip', fitted_slip),
            _as_finite_float('predicted_slip', predicted_slip),
        ))
        return self

    @staticmethod
    def _signed(value):
        if value == 0:
            return '$0$'
        sign = '+' if value > 0 else '-'
        return rf'${sign}{_scientific(abs(value), 3)}$'

    def to_latex(self):
        if not self.data:
            raise ValueError('At least one Poiseuille row is required.')
        lines = [
            r'\begin{table}[htbp]', r'', r'\centering',
            r'\caption{Poiseuille boundary tuning at $N=128$.  The order is from $N=96$ to $128$; a dash marks roundoff-dominated tuned errors.}',
            r'\label{tab:numerical-poiseuille-tuning}', r'\begingroup',
            r'\scriptsize', r'\setlength{\tabcolsep}{3.7pt}',
            r'\begin{tabular}{@{}clrrrrr@{}}', r'\toprule',
            r'$\ell$ & choice & $s_{+}$ & $E_{u,2}$ & $p_{96\to128}$ & fitted slip & predicted slip\\',
            r'\midrule',
        ]
        previous_ell = None
        for row in self.data:
            if previous_ell is not None and row.ell != previous_ell:
                lines.append(r'\midrule')
            order = '--' if row.order is None else f'{row.order:.3f}'
            ell = f'{row.ell:g}'
            lines.append(
                f'{ell} & {row.choice} & {row.s_plus:.6f} & '
                f'${_scientific(row.e_u2, 3)}$ & {order} & '
                f'{self._signed(row.fitted_slip)} & '
                f'{self._signed(row.predicted_slip)}' + r'\\'
            )
            previous_ell = row.ell
        lines.extend([
            r'\bottomrule', r'\end{tabular}', r'\endgroup', r'\end{table}',
        ])
        return '\n'.join(lines) + '\n'

    def save(self):
        output = self.dir/self.file_name.with_suffix('.tex')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_latex(), encoding='utf-8', newline='\n')
        return output.resolve()


