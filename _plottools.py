#!/usr/bin/python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Xu Yuyang (https://github.com/xyy2024)
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

"""Shared styles and layout helpers used by the article figure generators."""

import numpy as np

from _figure_style import TEXTWIDTH_IN, FONT_SIZE_PT, MIN_LINEWIDTH_PT, DATA_LINEWIDTH_PT, apply_style

ARTICLE_MIN_SOURCE_LINEWIDTH = MIN_LINEWIDTH_PT
ARTICLE_DATA_LINEWIDTH = DATA_LINEWIDTH_PT
apply_style()

import math

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from numbers import Real
from typing import Sequence


# Full-colour, colour-vision-friendly defaults.  The colormaps deliberately
# contain several perceptually ordered anchors, so field structure remains
# legible in colour while their luminance progression also survives grayscale
# printing.  Curves additionally use line style, marker shape, and marker fill.
ARTICLE_SERIES_COLORS = (
    '#0072B2',  # blue
    '#D55E00',  # vermilion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#332288',  # indigo
    '#AA4499',  # purple
    '#117733',  # dark green
)

ARTICLE_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'article_multihue_sequential',
    ('#28134b', '#3b4cc0', '#208f8c', '#73c476', '#f4d44d'),
    N=256,
)

def article_font_sizes(figure_width_inches: Real) -> tuple[float, float]:
    """Return actual publication sizes; figures must use the template width."""
    if abs(float(figure_width_inches) - TEXTWIDTH_IN) > 1e-8:
        raise ValueError('Article figures must be generated at TEXTWIDTH_IN.')
    return FONT_SIZE_PT, FONT_SIZE_PT


def axis_limits_with_blank_fractions(
        values,
        lower_blank_fraction: Real,
        upper_blank_fraction: Real,
        *,
        scale: str = 'linear',
        ) -> tuple[float, float]:
    '''Return adaptive limits preserving measured blank-axis fractions.

    ``lower_blank_fraction`` and ``upper_blank_fraction`` are measured in the
    displayed axis coordinate.  For a logarithmic axis this is log space, so
    the lower limit is exactly

    ``min**((1-r)/(1-l-r))*max**(-l/(1-l-r))``.
    '''
    lower_fraction = float(lower_blank_fraction)
    upper_fraction = float(upper_blank_fraction)
    if (
        not math.isfinite(lower_fraction)
        or not math.isfinite(upper_fraction)
        or lower_fraction < 0
        or upper_fraction < 0
        or lower_fraction+upper_fraction >= 1
    ):
        raise ValueError(
            'Blank fractions must be finite, non-negative, and sum to less '
            'than one.'
        )

    data = np.ma.asarray(values, dtype=float).filled(np.nan).ravel()
    data = data[np.isfinite(data)]
    if scale == 'log':
        data = data[data > 0]
        transform = np.log
        inverse = np.exp
    elif scale == 'linear':
        transform = np.asarray
        inverse = np.asarray
    else:
        raise ValueError("scale must be 'linear' or 'log'.")
    if data.size == 0:
        raise ValueError('No finite displayable values were provided.')

    transformed_min = float(np.min(transform(data)))
    transformed_max = float(np.max(transform(data)))
    transformed_span = transformed_max-transformed_min
    if transformed_span <= 0:
        raise ValueError('Adaptive limits require at least two distinct values.')

    occupied_fraction = 1-lower_fraction-upper_fraction
    axis_span = transformed_span/occupied_fraction
    lower_limit = transformed_min-lower_fraction*axis_span
    upper_limit = transformed_max+upper_fraction*axis_span
    return float(inverse(lower_limit)), float(inverse(upper_limit))


def _marked_line_points(axis: Axes) -> tuple[np.ndarray, np.ndarray]:
    '''Return finite points represented by markers on one line-plot axis.'''
    x_columns = []
    y_columns = []
    for line in axis.lines:
        if line.get_marker() in (None, '', ' ', 'None', 'none'):
            continue
        x = np.ma.asarray(line.get_xdata(), dtype=float).filled(np.nan).ravel()
        y = np.ma.asarray(line.get_ydata(), dtype=float).filled(np.nan).ravel()
        finite = np.isfinite(x) & np.isfinite(y)
        x_columns.append(x[finite])
        y_columns.append(y[finite])
    for collection in axis.collections:
        if not isinstance(collection, PathCollection):
            continue
        offsets = np.ma.asarray(
            collection.get_offsets(), dtype=float
        ).filled(np.nan).reshape(-1, 2)
        finite = np.all(np.isfinite(offsets), axis=1)
        x_columns.append(offsets[finite, 0])
        y_columns.append(offsets[finite, 1])
    if not x_columns or not any(column.size for column in x_columns):
        raise ValueError('The axis contains no finite marked data points.')
    return np.concatenate(x_columns), np.concatenate(y_columns)


def set_adaptive_line_limits(
        axes: Axes | Sequence[Axes],
        *,
        x_blank_fractions: tuple[Real, Real] | None = None,
        y_blank_fractions: tuple[Real, Real] | None = None,
        ) -> dict[str, tuple[float, float]]:
    '''Set shared adaptive limits from all marked data points on ``axes``.

    Unmarked reference and threshold lines are deliberately excluded.  When
    several axes are supplied, their marked points define one common range,
    which preserves shared-axis comparisons in multi-panel figures.
    '''
    if isinstance(axes, Axes):
        axes = (axes,)
    else:
        axes = tuple(axes)
    if not axes:
        raise ValueError('At least one axis is required.')
    points = [_marked_line_points(axis) for axis in axes]
    limits = {}

    if x_blank_fractions is not None:
        scales = {axis.get_xscale() for axis in axes}
        if len(scales) != 1:
            raise ValueError('All supplied axes must use the same x scale.')
        x_values = np.concatenate([point[0] for point in points])
        x_limits = axis_limits_with_blank_fractions(
            x_values, *x_blank_fractions, scale=scales.pop()
        )
        for axis in axes:
            axis.set_xlim(*x_limits)
        limits['x'] = x_limits

    if y_blank_fractions is not None:
        scales = {axis.get_yscale() for axis in axes}
        if len(scales) != 1:
            raise ValueError('All supplied axes must use the same y scale.')
        y_values = np.concatenate([point[1] for point in points])
        y_limits = axis_limits_with_blank_fractions(
            y_values, *y_blank_fractions, scale=scales.pop()
        )
        for axis in axes:
            axis.set_ylim(*y_limits)
        limits['y'] = y_limits

    return limits
