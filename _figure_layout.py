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

"""Geometry-based spacing shared by the CSV figure generators."""
import numpy as np
from _figure_style import TEXTWIDTH_IN

# Visible gap measured in the reference Fig. SM6 PDF, at its 5.125-inch width.
LEGEND_GAP_PT = 6.5
REFERENCE_WIDTH_IN = TEXTWIDTH_IN


def _ink_bounds(fig, box, pixels=None):
    """Visible bounds in display pixels, excluding text-box whitespace."""
    if pixels is None:
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    height, width = pixels.shape[:2]
    x0, x1 = max(0, int(box.x0)), min(width, int(np.ceil(box.x1)))
    y0, y1 = max(0, int(height-box.y1)), min(height, int(np.ceil(height-box.y0)))
    rows, cols = np.nonzero(np.any(pixels[y0:y1, x0:x1] < 245, axis=2))
    if not len(rows):
        raise ValueError('Expected visible legend/label content inside the canvas.')
    return (x0+cols.min(), height-(y0+rows.max()+1),
            x0+cols.max()+1, height-(y0+rows.min()))


def _pdf_pixels(fig, dpi):
    """Measure PDF glyphs too: Agg and PDF math-text extents differ slightly."""
    from io import BytesIO
    import matplotlib
    import pymupdf
    stream = BytesIO()
    with matplotlib.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        fig.savefig(stream, format='pdf')
    with pymupdf.open(stream=stream.getvalue(), filetype='pdf') as document:
        pix = document[0].get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72),
                                   colorspace=pymupdf.csRGB, alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    # PDF drawing temporarily uses 72 dpi and updates axis-label positions.
    fig.canvas.draw()
    return pixels


def legends_below_xlabels(fig, groups, gap_pt=LEGEND_GAP_PT, *, center_on_axes=True):
    """Place each (axes, legend) below its labels, measuring all groups together.

    Centers use only plotting rectangles. The gap is between visible text and
    legend ink, as in Fig. SM6. Expand the bottom once if necessary, preserving
    all plotting dimensions and the relative positions of multiple legends.
    """
    groups = [(list(axes), legend) for axes, legend in groups]
    original_dpi = fig.dpi
    fig.set_dpi(288)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_visibility = [(ax, ax.get_visible()) for ax in fig.axes]
    legend_visibility = [(legend, legend.get_visible()) for legend in fig.legends]
    centers = []
    for axes, legend in groups:
        boxes = [ax.get_window_extent(renderer) for ax in axes]
        if center_on_axes:
            center = (min(box.x0 for box in boxes)+max(box.x1 for box in boxes))/2
        else:
            box = legend.get_window_extent(renderer)
            center = (box.x0+box.x1)/2
        centers.append(center/fig.bbox.width)
    for legend, _ in legend_visibility:
        legend.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottoms = [min(_ink_bounds(fig, ax.xaxis.label.get_window_extent(renderer))[1]
                   for ax in axes) for axes, _ in groups]
    for ax, _ in axes_visibility:
        ax.set_visible(False)
    placements = []
    for (_, legend), center, bottom in zip(groups, centers, bottoms):
        legend.set_loc('upper center')
        legend.borderaxespad = legend.borderpad = 0
        legend.set_bbox_to_anchor((center, .8), transform=fig.transFigure)
        legend.set_visible(True)
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        delta = bottom-gap_pt*fig.dpi/72-_ink_bounds(fig, box)[3]
        placements.append((center, box.y1+delta, box.y0+delta))
        legend.set_visible(False)
    for ax, visible in axes_visibility:
        ax.set_visible(visible)
    for legend, visible in legend_visibility:
        legend.set_visible(visible)
    extra_px = max(0, 4*fig.dpi/72-min(bottom for _, _, bottom in placements))
    width, height = fig.get_size_inches()
    extra = extra_px/fig.dpi
    if extra:
        positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        fig.set_size_inches(width, height+extra)
        for ax, pos in positions:
            ax.set_position((pos.x0, (pos.y0*height+extra)/(height+extra),
                             pos.width, pos.height*height/(height+extra)))
    for (_, legend), (center, top, _) in zip(groups, placements):
        legend.set_bbox_to_anchor((center, (top+extra_px)/fig.bbox.height),
                                  transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Correct the small backend-dependent difference using the final PDF glyphs.
    # The PDF exporter later crops only outer whitespace, preserving this gap.
    pixels = _pdf_pixels(fig, fig.dpi)
    pad = 2*fig.dpi/72
    for axes, legend in groups:
        bottom = min(_ink_bounds(fig, ax.xaxis.label.get_window_extent(renderer).padded(pad), pixels)[1]
                     for ax in axes)
        top = _ink_bounds(fig, legend.get_window_extent(renderer).padded(pad), pixels)[3]
        shift = bottom-top-gap_pt*fig.dpi/72
        anchor = legend.get_bbox_to_anchor()
        legend.set_bbox_to_anchor((anchor.x0/fig.bbox.width,
                                   (anchor.y0+shift)/fig.bbox.height),
                                  transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels = _pdf_pixels(fig, fig.dpi)
    actual_gaps = []
    for axes, legend in groups:
        actual = (min(_ink_bounds(fig, ax.xaxis.label.get_window_extent(renderer).padded(pad), pixels)[1]
                      for ax in axes)-_ink_bounds(fig, legend.get_window_extent(renderer).padded(pad), pixels)[3])*72/fig.dpi
        if abs(actual-gap_pt) > .6:
            raise RuntimeError(f'Unexpected xlabel-to-legend gap: {actual:g} pt.')
        actual_gaps.append(actual)
    fig.set_dpi(original_dpi)
    fig.canvas.draw()
    return actual_gaps


def legend_below_xlabels(fig, axes, legend, gap_pt=LEGEND_GAP_PT, *, center_on_axes=False):
    """Place one legend; opt into rectangle centering for the specified figures."""
    return legends_below_xlabels(fig, [(axes, legend)], gap_pt,
                                center_on_axes=center_on_axes)[0]


def legend_right_of_axes(fig, ax, legend, gap_pt=10, min_margin_pt=14):
    """Center the axes, labels and right legend within the unchanged canvas."""
    legend.set_loc('center left')
    legend.borderaxespad = legend.borderpad = 0

    def place():
        fig.canvas.draw()
        box = ax.get_window_extent(fig.canvas.get_renderer())
        legend.set_bbox_to_anchor(((box.x1+gap_pt*fig.dpi/72)/fig.bbox.width,
                                   (box.y0+box.y1)/(2*fig.bbox.height)),
                                  transform=fig.transFigure)
        fig.canvas.draw()

    place()
    renderer = fig.canvas.get_renderer()
    boxes = (ax.get_tightbbox(renderer), legend.get_window_extent(renderer))
    left, right = min(box.x0 for box in boxes), max(box.x1 for box in boxes)
    margin = (fig.bbox.width-(right-left))/2
    if margin < min_margin_pt*fig.dpi/72:
        raise ValueError(f'The axes and right legend leave only {margin*72/fig.dpi:g} pt '
                         f'per side; at least {min_margin_pt:g} pt is required.')
    shift = (fig.bbox.width-left-right)/(2*fig.bbox.width)
    position = ax.get_position()
    ax.set_position((position.x0+shift, position.y0, position.width, position.height))
    place()
    renderer = fig.canvas.get_renderer()
    actual_gap = (legend.get_window_extent(renderer).x0
                  - ax.get_window_extent(renderer).x1)*72/fig.dpi
    if abs(actual_gap-gap_pt) > .1:
        raise RuntimeError(f'Unexpected axes-to-legend gap: {actual_gap:g} pt.')
    return actual_gap


def align_colorbar_height(fig, colorbar, axes):
    """Match the bar to the actual plotting bounds after equal-aspect layout."""
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    bottom, top = min(box.y0 for box in boxes), max(box.y1 for box in boxes)
    box = colorbar.ax.get_position()
    colorbar.ax.set_position((box.x0, bottom, box.width, top-bottom))
    fig.canvas.draw()


def colorbar_header(fig, colorbar, label, *, exponent=0, label_fontsize=10,
                    scale_fontsize=10, label_pad=4, scale_gap_pt=1):
    """Center the quantity above the bar, with a scale above the tick column.

    The scale's visible top is slightly below the quantity's visible bottom.
    Preserve default tick padding and align the scale with the tick text column.
    """
    ax = colorbar.ax
    colorbar.set_label('')
    ax.set_xlabel('')
    ax.yaxis.get_offset_text().set_visible(False)
    title = ax.set_title(label, x=.5, y=1, ha='center', fontsize=label_fontsize,
                        pad=label_pad)
    ax._quantity_title = title
    ax._scale_header = None
    if not exponent:
        return
    scale = ax.text(.5, 1.15, r'$\times10^{' + str(exponent) + '}$',
                    transform=ax.transAxes, fontsize=scale_fontsize,
                    ha='center', va='bottom', clip_on=False)
    original_dpi = fig.dpi
    fig.set_dpi(288)
    fig.canvas.draw()
    scale.set_visible(False)
    pixels = _pdf_pixels(fig, fig.dpi)
    renderer = fig.canvas.get_renderer()
    quantity_ink = _ink_bounds(fig, title.get_window_extent(renderer), pixels)
    # Measure separately so neighboring mathematical glyphs cannot contaminate
    # the visible bounds of either header.
    title.set_visible(False)
    scale.set_visible(True)
    pixels = _pdf_pixels(fig, fig.dpi)
    scale_ink = _ink_bounds(fig, scale.get_window_extent(renderer), pixels)
    ticks = [t for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
    tick_boxes = [t.get_window_extent(renderer) for t in ticks]
    tick_left = min(_ink_bounds(fig, box, pixels)[0] for box in tick_boxes)
    anchor = ax.transAxes.transform(scale.get_position())
    anchor += [tick_left-scale_ink[0],
               quantity_ink[1]-scale_gap_pt*fig.dpi/72-scale_ink[3]]
    scale.set_position(ax.transAxes.inverted().transform(anchor))
    title.set_visible(True)
    ax._scale_header = scale
    fig.set_dpi(original_dpi)
    fig.canvas.draw()
