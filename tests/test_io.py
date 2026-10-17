from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rjnmr.io import (
    decode_asdf_line,
    read_bruker_fid,
    read_fid,
    read_jcamp_fid,
    read_jcamp_spectrum,
)

HEADER = """##TITLE=synthetic
##JCAMPDX= 6.0         $$ Bruker NMR JCAMP-DX V2.0
##DATA TYPE= NMR FID
##$SW_h= 12019.2307692308
##$SFO1= 600.13
##$O1= 2816.65
##$GRPDLY= 67.98
$$ End of Bruker specific parameters
##NTUPLES= NMR FID
##VAR_NAME=  TIME,          FID/REAL,        FID/IMAG
##VAR_DIM=   {n},         {n},           {n}
##FACTOR=    8.3e-05, 2,        2
"""


def _affn_page(values: np.ndarray, per_line: int = 8) -> str:
    lines = []
    for i in range(0, values.size, per_line):
        chunk = values[i : i + per_line]
        lines.append(f"{i} " + " ".join(str(int(v)) for v in chunk))
    return "\n".join(lines)


def _dif_encode(v: int) -> str:
    pos, neg = "%JKLMNOPQR", "%jklmnopqr"
    s = str(abs(v))
    return (pos if v >= 0 else neg)[int(s[0])] + s[1:]


def _sqz_encode(v: int) -> str:
    pos, neg = "@ABCDEFGHI", "@abcdefghi"
    s = str(abs(v))
    return (pos if v >= 0 else neg)[int(s[0])] + s[1:]


def _difdup_page(values: np.ndarray, per_line: int = 10) -> str:
    """Encode in DIFDUP form with Y-check duplication between lines."""
    lines = []
    i = 0
    while i < values.size:
        chunk = values[i : i + per_line + 1]
        toks = [_sqz_encode(int(chunk[0]))]
        diffs = np.diff(chunk).astype(int)
        j = 0
        while j < diffs.size:
            run = 1
            while j + run < diffs.size and diffs[j + run] == diffs[j]:
                run += 1
            toks.append(_dif_encode(int(diffs[j])))
            if run > 1:
                dup = str(run)
                toks.append("STUVWXYZs"[int(dup[0]) - 1] + dup[1:])
            j += run
        lines.append(f"{i}" + "".join(toks))
        i += per_line
    return "\n".join(lines)


@pytest.fixture
def signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    re = np.round(1000 * np.cos(0.3 * np.arange(53))) + rng.integers(-3, 3, 53)
    im = np.round(1000 * np.sin(0.3 * np.arange(53)))
    im[20:30] = 7  # a run that exercises DUP
    return re + 1j * im


@pytest.mark.parametrize("encoding", ["affn", "difdup"])
def test_read_jcamp_fid(tmp_path: Path, signal: np.ndarray, encoding: str) -> None:
    enc = _affn_page if encoding == "affn" else _difdup_page
    text = HEADER.format(n=signal.size)
    text += "##PAGE= N=1\n##DATA TABLE= (X++(R..R)), XYDATA\n" + enc(signal.real) + "\n"
    text += "##PAGE= N=2\n##DATA TABLE= (X++(I..I)), XYDATA\n" + enc(signal.imag) + "\n"
    text += "##END NTUPLES=NMR FID\n##END=\n"
    path = tmp_path / "fid.jdx"
    path.write_text(text)
    fid = read_jcamp_fid(path)
    np.testing.assert_allclose(fid.data, 2 * signal)
    assert fid.spectral_width == pytest.approx(12019.2307692308)
    assert fid.observe_frequency == pytest.approx(600.13)
    assert fid.carrier_offset == pytest.approx(2816.65)
    assert fid.group_delay == pytest.approx(67.98)
    assert fid.dwell_time == pytest.approx(1 / 12019.2307692308)
    assert fid.frequency_to_hz(np.pi) == pytest.approx(12019.2307692308 / 2)
    assert fid.frequency_to_ppm(0.0) == pytest.approx(2816.65 / 600.13)
    assert read_fid(path).data.size == signal.size


def test_decode_asdf_line() -> None:
    x, ys, dif = decode_asdf_line("10 A2j3K%S3")
    assert x == 10
    assert ys == [12, -1, 1, 1] + [1] * 12
    assert dif
    _, ys, dif = decode_asdf_line("0 1 -2 +3 4.5E1")
    assert ys == [1, -2, 3, 45.0]
    assert not dif
    with pytest.raises(ValueError):
        decode_asdf_line("0 ?")


def test_read_jcamp_spectrum(tmp_path: Path) -> None:
    y = np.arange(20) ** 2
    text = (
        "##TITLE=s\n##NPOINTS=20\n##FIRSTX=100\n##LASTX=81\n##YFACTOR=0.5\n##XYDATA=(X++(Y..Y))\n"
    )
    text += _affn_page(y.astype(float), 5) + "\n##END=\n"
    p = tmp_path / "spec.jdx"
    p.write_text(text)
    s = read_jcamp_spectrum(p)
    np.testing.assert_allclose(s.y, 0.5 * y)
    np.testing.assert_allclose(s.x, np.linspace(100, 81, 20))


@pytest.mark.parametrize(
    ("bytorda", "dtypa", "fmt"), [(1, 0, ">i4"), (0, 0, "<i4"), (0, 2, "<f8")]
)
def test_read_bruker_fid(tmp_path: Path, bytorda: int, dtypa: int, fmt: str) -> None:
    data = np.array([1, -2, 3, 4, 5, 6, 0, 0], dtype=fmt)  # TD=6 plus zero padding
    (tmp_path / "fid").write_bytes(data.tobytes())
    (tmp_path / "acqus").write_text(
        f"##TITLE=x\n##$BYTORDA= {bytorda}\n##$DTYPA= {dtypa}\n##$TD= 6\n##$SW_h= 5000\n##END=\n"
    )
    fid = read_bruker_fid(tmp_path)
    np.testing.assert_array_equal(fid.data, [1 - 2j, 3 + 4j, 5 + 6j])
    assert fid.spectral_width == 5000
    assert read_fid(tmp_path / "fid").data.size == 3


def test_read_fid_other_formats(tmp_path: Path) -> None:
    y = np.array([1 + 2j, 3 - 1j])
    np.save(tmp_path / "y.npy", y)
    np.testing.assert_array_equal(read_fid(tmp_path / "y.npy").data, y)
    np.savetxt(tmp_path / "y.txt", np.c_[y.real, y.imag])
    np.testing.assert_array_equal(read_fid(tmp_path / "y.txt").data, y)
    with pytest.raises(ValueError, match="unrecognised"):
        read_fid(tmp_path / "y.xyz")
