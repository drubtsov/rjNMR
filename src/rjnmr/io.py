"""Readers for NMR time-domain data.

* JCAMP-DX 5/6 ``NTUPLES`` FIDs as exported by Bruker XWIN-NMR/TopSpin
  (real and imaginary pages, AFFN or compressed ASDF ordinates), and
  ``XYDATA`` spectra.
* Raw Bruker ``fid`` files with an optional ``acqus`` parameter file.
"""

from __future__ import annotations

import contextlib
import re
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FID",
    "Spectrum",
    "decode_asdf_line",
    "parse_jcamp",
    "read_bruker_fid",
    "read_fid",
    "read_jcamp_fid",
    "read_jcamp_spectrum",
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FID:
    """A complex free induction decay plus acquisition parameters.

    Attributes
    ----------
    data
        Complex time-domain samples.
    spectral_width
        Spectral width in Hz (Bruker ``SW_h``), if known.
    observe_frequency
        Spectrometer frequency in MHz (``SFO1``), if known.
    carrier_offset
        Carrier offset from the reference in Hz (``O1``), if known.
    group_delay
        Digital-filter group delay in points (``GRPDLY``), if known.
    parameters
        All parsed header parameters (normalised labels → raw string values).
    """

    data: ComplexArray
    spectral_width: float | None = None
    observe_frequency: float | None = None
    carrier_offset: float | None = None
    group_delay: float | None = None
    parameters: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.data.size)

    @property
    def dwell_time(self) -> float | None:
        """Sampling interval in seconds."""
        return None if not self.spectral_width else 1.0 / self.spectral_width

    def frequency_to_hz(self, omega: float | FloatArray) -> float | FloatArray:
        """Convert angular frequency (rad/sample) to Hz relative to the carrier."""
        if not self.spectral_width:
            raise ValueError("spectral width unknown")
        return np.asarray(omega) / (2.0 * np.pi) * self.spectral_width

    def frequency_to_ppm(self, omega: float | FloatArray) -> float | FloatArray:
        """Convert angular frequency (rad/sample) to chemical shift in ppm.

        Uses ``(f + O1) / SFO1`` with the carrier offset ``O1`` in Hz.
        """
        if not self.observe_frequency:
            raise ValueError("observe frequency unknown")
        hz = self.frequency_to_hz(omega)
        return (hz + (self.carrier_offset or 0.0)) / self.observe_frequency

    def ppm_to_frequency(self, ppm: float | FloatArray) -> float | FloatArray:
        """Convert chemical shift in ppm to angular frequency (rad/sample)."""
        if not (self.observe_frequency and self.spectral_width):
            raise ValueError("observe frequency and spectral width are needed for ppm")
        hz = np.asarray(ppm) * self.observe_frequency - (self.carrier_offset or 0.0)
        return 2.0 * np.pi * hz / self.spectral_width


@dataclass(slots=True)
class Spectrum:
    """A real frequency-domain spectrum from a JCAMP-DX ``XYDATA`` block."""

    x: FloatArray
    y: FloatArray
    parameters: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------------- JCAMP-DX ---

_LDR = re.compile(r"^##([^=]*)=(.*)$")


def _norm_label(label: str) -> str:
    """Normalise a JCAMP label: upper case, strip spaces, dashes, slashes and underscores.

    Bruker private labels (``$``) and dotted labels keep their prefix character.
    """
    return re.sub(r"[\s\-/_]", "", label).upper()


@dataclass(slots=True)
class _Block:
    label: str
    value: str
    lines: list[str]


def _iter_blocks(text: Iterable[str]) -> Iterator[_Block]:
    current: _Block | None = None
    for raw in text:
        line = raw.rstrip("\r\n")
        if line.startswith("$$"):
            continue  # comment line
        m = _LDR.match(line)
        if m:
            if current is not None:
                yield current
            value = m.group(2).split("$$", 1)[0].strip()
            current = _Block(_norm_label(m.group(1)), value, [])
        elif current is not None:
            current.lines.append(line.split("$$", 1)[0])
    if current is not None:
        yield current


def parse_jcamp(path: str | Path) -> tuple[dict[str, str], list[_Block]]:
    """Parse a JCAMP-DX file into a label → value dict and the ordered list of blocks."""
    with open(path, encoding="latin-1") as fh:
        blocks = list(_iter_blocks(fh))
    params: dict[str, str] = {}
    for b in blocks:
        params.setdefault(b.label, b.value)
    return params, blocks


_SQZ = {c: i for i, c in enumerate("@ABCDEFGHI")} | {
    c: -(i + 1) for i, c in enumerate("abcdefghi")
}
_DIF = {c: i for i, c in enumerate("%JKLMNOPQR")} | {
    c: -(i + 1) for i, c in enumerate("jklmnopqr")
}
_DUP = {c: i + 1 for i, c in enumerate("STUVWXYZ")} | {"s": 9}
_AFFN = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?")
_AFFN_NO_EXP = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
# ASDF pseudo-digits other than E/e, which are ambiguous with AFFN exponents
_ASDF_CHARS = re.compile(r"[@A-DF-Ia-df-i%J-Rj-rS-Zs]")


def _tokens(line: str) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, text)`` tokens; ``kind`` is ``'affn'``, ``'sqz'``, ``'dif'`` or ``'dup'``.

    ``E``/``e`` are SQZ digits (+5/-5) in compressed lines but exponent markers in
    AFFN numbers; a line containing any other ASDF character is treated as
    compressed, so exponents are not recognised in it.
    """
    affn = _AFFN_NO_EXP if _ASDF_CHARS.search(line) else _AFFN
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t,;":
            i += 1
            continue
        if (c in _SQZ or c in _DIF or c in _DUP) and not (affn is _AFFN and c in "Ee"):
            kind = "sqz" if c in _SQZ else "dif" if c in _DIF else "dup"
            j = i + 1
            while j < n and (line[j].isdigit() or (kind != "dup" and line[j] == ".")):
                j += 1
            yield kind, line[i:j]
            i = j
            continue
        m = affn.match(line, i)
        if not m:
            raise ValueError(f"cannot parse JCAMP data near {line[i : i + 20]!r}")
        yield "affn", m.group(0)
        i = m.end()


def decode_asdf_line(line: str) -> tuple[float, list[float], bool]:
    """Decode one ``(X++(Y..Y))`` data line.

    Returns the abscissa, the ordinates and whether the line ended in DIF
    form (in which case the first ordinate of the next line is a Y-check
    duplicate that must be dropped).
    """
    x: float | None = None
    ys: list[float] = []
    last_kind = ""
    last_dif = 0.0
    for kind, text in _tokens(line):
        if x is None:
            if kind != "affn" and kind != "sqz":
                raise ValueError(f"data line does not start with an abscissa: {line!r}")
            x = float(text) if kind == "affn" else float(f"{_SQZ[text[0]]}{text[1:]}")
            continue
        if kind == "affn":
            ys.append(float(text))
            last_kind = "abs"
        elif kind == "sqz":
            d = _SQZ[text[0]]
            sign = "-" if d < 0 else ""
            ys.append(float(f"{sign}{abs(d)}{text[1:]}"))
            last_kind = "abs"
        elif kind == "dif":
            d = _DIF[text[0]]
            sign = "-" if d < 0 else ""
            last_dif = float(f"{sign}{abs(d)}{text[1:]}")
            if not ys:
                raise ValueError("DIF token without a preceding ordinate")
            ys.append(ys[-1] + last_dif)
            last_kind = "dif"
        else:  # dup
            count = int(f"{_DUP[text[0]]}{text[1:]}") - 1
            if not ys:
                raise ValueError("DUP token without a preceding ordinate")
            for _ in range(count):
                ys.append(ys[-1] + last_dif if last_kind == "dif" else ys[-1])
    if x is None:
        raise ValueError("empty data line")
    return x, ys, last_kind == "dif"


def _decode_table(lines: list[str]) -> FloatArray:
    out: list[float] = []
    prev_dif = False
    for line in lines:
        if not line.strip():
            continue
        _, ys, ends_dif = decode_asdf_line(line)
        if prev_dif and ys:
            ys = ys[1:]  # Y-check value duplicated from previous line
        out.extend(ys)
        prev_dif = ends_dif
    return np.asarray(out, dtype=np.float64)


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",")]


def _param_float(params: dict[str, str], *labels: str) -> float | None:
    for label in labels:
        v = params.get(_norm_label(label))
        if v is None:
            continue
        try:
            return float(v.strip().strip("<>"))
        except ValueError:
            continue
    return None


def _fid_from_params(data: ComplexArray, params: dict[str, str]) -> FID:
    return FID(
        data=data,
        spectral_width=_param_float(params, "$SW_h"),
        observe_frequency=_param_float(params, "$SFO1", ".OBSERVE FREQUENCY"),
        carrier_offset=_param_float(params, "$O1"),
        group_delay=_param_float(params, "$GRPDLY"),
        parameters=params,
    )


def read_jcamp_fid(path: str | Path) -> FID:
    """Read a complex FID stored as a JCAMP-DX ``NTUPLES`` block.

    The first data page holds the real part and the second the imaginary
    part (``(X++(R..R))`` and ``(X++(I..I))``); per-variable ``FACTOR``
    values are applied.
    """
    params, blocks = parse_jcamp(path)
    factors = [1.0, 1.0, 1.0]
    if "FACTOR" in params:
        with contextlib.suppress(ValueError):
            factors = [float(v) for v in _split_csv(params["FACTOR"])]
    n_points: int | None = None
    if "VARDIM" in params:
        n_points = int(float(_split_csv(params["VARDIM"])[-1]))
    pages: dict[str, FloatArray] = {}
    for b in blocks:
        if b.label == "DATATABLE":
            sym = re.search(r"\(\s*X\s*\+\+\s*\(\s*(\w)\.\.", b.value)
            key = sym.group(1).upper() if sym else str(len(pages))
            pages[key] = _decode_table(b.lines)
    if not pages:
        # Some exporters store a single XYDATA block of interleaved values.
        raise ValueError(f"{path}: no NTUPLES data tables found")
    real = pages.get("R", next(iter(pages.values())))
    imag = pages.get("I")
    if imag is None:
        warnings.warn(f"{path}: no imaginary page, using real data only", stacklevel=2)
        imag = np.zeros_like(real)
    n = min(real.size, imag.size) if n_points is None else n_points
    if real.size < n or imag.size < n:
        raise ValueError(f"{path}: expected {n} points, found {real.size} real / {imag.size} imag")
    fr = factors[1] if len(factors) > 1 else 1.0
    fi = factors[2] if len(factors) > 2 else fr
    data = real[:n] * fr + 1j * (imag[:n] * fi)
    return _fid_from_params(data.astype(np.complex128), params)


def read_jcamp_spectrum(path: str | Path) -> Spectrum:
    """Read a JCAMP-DX ``XYDATA=(X++(Y..Y))`` spectrum."""
    params, blocks = parse_jcamp(path)
    block = next((b for b in blocks if b.label == "XYDATA"), None)
    if block is None:
        raise ValueError(f"{path}: no XYDATA block")
    y = _decode_table(block.lines) * (_param_float(params, "YFACTOR") or 1.0)
    npts = int(_param_float(params, "NPOINTS") or y.size)
    y = y[:npts]
    first = _param_float(params, "FIRSTX") or 0.0
    last = _param_float(params, "LASTX")
    x: FloatArray
    if last is not None and npts > 1:
        x = np.linspace(first, last, npts, dtype=np.float64)
    else:
        x = first + np.arange(npts, dtype=np.float64) * (_param_float(params, "DELTAX") or 1.0)
    return Spectrum(x=x, y=y, parameters=params)


# ---------------------------------------------------------------------------- Bruker raw ---


def read_bruker_fid(
    path: str | Path,
    *,
    byte_order: Literal["big", "little"] | None = None,
    dtype: Literal["int32", "float64"] | None = None,
) -> FID:
    """Read a raw Bruker ``fid`` file.

    Parameters
    ----------
    path
        Experiment directory or the ``fid`` file itself. Acquisition
        parameters are read from ``acqus`` in the same directory when present.
    byte_order, dtype
        Override ``BYTORDA``/``DTYPA``. If neither ``acqus`` nor overrides are
        available the classic Bruker format (big-endian int32) is assumed and a
        warning is issued.
    """
    p = Path(path)
    fid_path = p / "fid" if p.is_dir() else p
    params: dict[str, str] = {}
    acqus = fid_path.parent / "acqus"
    if acqus.exists():
        try:
            params, _ = parse_jcamp(acqus)
        except (OSError, ValueError):  # pragma: no cover - malformed parameter file
            params = {}
    if byte_order is None:
        bo = _param_float(params, "$BYTORDA")
        byte_order = None if bo is None else ("big" if int(bo) == 1 else "little")
    if dtype is None:
        dt = _param_float(params, "$DTYPA")
        dtype = None if dt is None else ("float64" if int(dt) == 2 else "int32")
    if byte_order is None or dtype is None:
        warnings.warn(
            f"{fid_path}: data format not specified and no usable acqus; "
            "assuming big-endian int32",
            stacklevel=2,
        )
        byte_order = byte_order or "big"
        dtype = dtype or "int32"
    np_dtype = np.dtype("i4" if dtype == "int32" else "f8").newbyteorder(
        ">" if byte_order == "big" else "<"
    )
    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float64)
    td = _param_float(params, "$TD")
    if td is not None:
        raw = raw[: int(td)]
    if raw.size % 2:
        raw = raw[:-1]
    data = raw[0::2] + 1j * raw[1::2]
    return _fid_from_params(data, params)


def read_fid(
    path: str | Path,
    *,
    byte_order: Literal["big", "little"] | None = None,
    dtype: Literal["int32", "float64"] | None = None,
) -> FID:
    """Read an FID, dispatching on the file type.

    ``.jdx``/``.dx`` files are read as JCAMP-DX, ``.npy`` as a NumPy array of
    complex samples, and directories or files named ``fid`` as raw Bruker data
    (``byte_order``/``dtype`` override the ``acqus`` values).
    """
    p = Path(path)
    if p.is_dir() or p.name == "fid":
        return read_bruker_fid(p, byte_order=byte_order, dtype=dtype)
    suffix = p.suffix.lower()
    if suffix in {".jdx", ".dx", ".jcamp"}:
        return read_jcamp_fid(p)
    if suffix == ".npy":
        return FID(data=np.load(p).astype(np.complex128).ravel())
    if suffix in {".txt", ".csv"}:
        arr = np.loadtxt(p, delimiter="," if suffix == ".csv" else None, ndmin=2)
        data = arr[:, 0] + 1j * arr[:, 1] if arr.shape[1] >= 2 else arr[:, 0].astype(complex)
        return FID(data=data.astype(np.complex128))
    raise ValueError(f"unrecognised data format: {p}")
