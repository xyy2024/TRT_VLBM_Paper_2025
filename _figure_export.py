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

"""Export vector figures with zero top/bottom padding and unchanged width.

PDF clipping uses rendered ink (including titles, legends, labels and ticks),
not the axes rectangle. White page backgrounds are excluded. Only the page
boundary and a translation change; scientific paths and images remain vector
objects or their original embedded images. Axis limits are never modified.
"""
from pathlib import Path

import pymupdf as fitz
import numpy as np
from _figure_style import RASTER_DPI, TEXTWIDTH_BP

MEASUREMENT_DPI = 720


def ink_bounds(page, dpi=MEASUREMENT_DPI):
    """Return the occupied pixel rows and their physical PDF coordinates."""
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False,
                         colorspace=fitz.csRGB)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    rows = np.flatnonzero(np.any(rgb < 254, axis=(1, 2)))
    if not len(rows):
        raise ValueError('Cannot crop an empty figure.')
    return dict(top=int(rows[0]), bottom=int(pix.height-rows[-1]-1),
                y0=float(rows[0]*72/dpi),
                y1=float(min(page.rect.height, (rows[-1]+1)*72/dpi)),
                pixel_height=pix.height, dpi=dpi)


def crop_pdf_vertical(source, destination=None):
    """Remove only vertical outer whitespace; retain horizontal bounds exactly."""
    source = Path(source)
    destination = Path(destination or source)
    document = fitz.open(stream=source.read_bytes(), filetype='pdf')
    if len(document) != 1 or document[0].rotation:
        raise ValueError('Expected one unrotated figure page.')
    page = document[0]
    original_width, original_height = page.rect.width, page.rect.height
    before = ink_bounds(page)
    total_top = total_bottom = 0.0
    for _ in range(3):
        bounds = ink_bounds(document[0])
        if bounds['top'] == 0 and bounds['bottom'] == 0:
            break
        old_page = document[0]
        clip = fitz.Rect(0, bounds['y0'], old_page.rect.width, bounds['y1'])
        total_top += clip.y0
        total_bottom += old_page.rect.height-clip.y1
        trimmed = fitz.open()
        target = trimmed.new_page(width=clip.width, height=clip.height)
        target.show_pdf_page(target.rect, document, 0, clip=clip, keep_proportion=False)
        document.close()
        document = trimmed
    after = ink_bounds(document[0])
    if after['top'] > 1 or after['bottom'] > 1:
        raise RuntimeError(f'Outer whitespace remains: {after}')
    if abs(document[0].rect.width-original_width) > 1e-4:
        raise RuntimeError('Horizontal figure width changed.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(document.tobytes(garbage=4, deflate=True))
    output_height = document[0].rect.height
    document.close()
    return dict(file=destination.name, original_size_pt=[original_width, original_height],
                output_size_pt=[original_width, output_height],
                removed_top_pt=total_top, removed_bottom_pt=total_bottom,
                remaining_top_pt=after['top']*72/MEASUREMENT_DPI,
                remaining_bottom_pt=after['bottom']*72/MEASUREMENT_DPI,
                measurement_dpi=MEASUREMENT_DPI)


def save_figure(fig, output, **kwargs):
    """Export at textwidth, with no horizontal cropping or subsequent scaling."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if kwargs.get('bbox_inches') is not None:
        raise ValueError('Tight/custom export boxes change the publication width.')
    kwargs.update(bbox_inches=None, pad_inches=0, dpi=RASTER_DPI)
    fig.savefig(output, **kwargs)
    if output.suffix.lower() == '.pdf':
        report = crop_pdf_vertical(output)
        if abs(report['output_size_pt'][0] - TEXTWIDTH_BP) > 1e-4:
            raise RuntimeError('Exported PDF does not match the template textwidth.')
        return report
    return output
