"""Minimal FID pre-processing used in the paper.

No phase or baseline correction is needed: the model has free complex
amplitudes and fits the raw time-domain signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Scaling", "normalize", "select_window", "trim_to_maximum"]

ComplexArray = NDArray[np.complex128]


def select_window(y: ArrayLike, start: int = 0, length: int | None = None) -> ComplexArray:
    """Return ``y[start:start + length]`` as a contiguous complex array."""
    yc = np.asarray(y, dtype=np.complex128).ravel()
    if start < 0 or start >= yc.size:
        raise ValueError(f"start={start} outside data of length {yc.size}")
    stop = yc.size if length is None else start + length
    if stop > yc.size:
        raise ValueError(f"window [{start}, {stop}) exceeds data length {yc.size}")
    return np.ascontiguousarray(yc[start:stop])


def trim_to_maximum(y: ArrayLike) -> tuple[ComplexArray, int]:
    """Drop leading points so that the first sample has maximal magnitude.

    This removes the digital-filter group delay of Bruker data, as done in the
    paper ("the first 80 points were removed so the signal at the first time
    point would have the maximal magnitude"). Returns the trimmed FID and the
    number of points removed.
    """
    yc = np.asarray(y, dtype=np.complex128).ravel()
    i = int(np.argmax(np.abs(yc)))
    return np.ascontiguousarray(yc[i:]), i


@dataclass(frozen=True, slots=True)
class Scaling:
    """Affine transformation ``y' = (y - offset) / scale`` applied to an FID."""

    offset: complex
    scale: float

    def apply(self, y: ArrayLike) -> ComplexArray:
        """Transform data."""
        out: ComplexArray = (np.asarray(y, dtype=np.complex128) - self.offset) / self.scale
        return out

    def amplitude_to_original(self, a: ArrayLike) -> ComplexArray:
        """Map fitted amplitudes back to the units of the original data."""
        out: ComplexArray = np.asarray(a, dtype=np.complex128) * self.scale
        return out


def normalize(y: ArrayLike, *, center: bool = True) -> tuple[ComplexArray, Scaling]:
    """Centre an FID and scale it to unit standard deviation (paper, Section 4.2)."""
    yc = np.asarray(y, dtype=np.complex128).ravel()
    offset = complex(yc.mean()) if center else 0j
    scale = float(np.std(yc - offset))
    if scale == 0.0:
        scale = 1.0
    s = Scaling(offset, scale)
    return s.apply(yc), s
