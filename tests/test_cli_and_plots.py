from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rjnmr import RJMCMCResult, SamplerConfig
from rjnmr.cli import main

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sim = tmp_path / "sim.npy"
    res = tmp_path / "res.npz"
    png = tmp_path / "res.png"
    assert main(["simulate", "-o", str(sim), "-n", "256", "--seed", "1"]) == 0
    assert (
        main(
            [
                "-q",
                "fit",
                str(sim),
                "-n",
                "150",
                "-b",
                "150",
                "--seed",
                "2",
                "--init",
                "peaks",
                "-o",
                str(res),
                "--length",
                "200",
            ]
        )
        == 0
    )
    loaded = RJMCMCResult.load(res)
    assert loaded.data.size == 200
    assert loaded.metadata["start"] == 0
    capsys.readouterr()
    assert main(["summary", str(res), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["k_map"] == loaded.k_map
    assert main(["summary", str(res)]) == 0
    assert main(["plot", str(res), "-o", str(png)]) == 0
    assert main(["plot", str(res), "-o", str(png), "--xlim", "1.5", "2.5"]) == 0
    assert png.stat().st_size > 10_000
    assert main(["baseline", str(sim), "--trim-to-max", "--normalize"]) == 0


def test_cli_multiple_chains(tmp_path: Path) -> None:
    sim = tmp_path / "sim.npy"
    res = tmp_path / "res.npz"
    main(["simulate", "-o", str(sim), "-n", "128", "--seed", "1"])
    assert (
        main(
            [
                "-q",
                "fit",
                str(sim),
                "-n",
                "40",
                "-b",
                "10",
                "--chains",
                "2",
                "--jobs",
                "1",
                "-o",
                str(res),
                "--update-kernel",
                "polar",
            ]
        )
        == 0
    )
    assert len(RJMCMCResult.load(res).trace) == 80


def test_cli_deconvolve(tmp_path: Path) -> None:
    from rjnmr import Component, simulate_fid

    comps = [Component(-1.0, 0.004, 8.0, 0.4), Component(0.5, 0.003, 6.0, 0.4)]
    y = simulate_fid(comps, 1024, 0.05, seed=0)
    src = tmp_path / "fid.npy"
    np.save(src, y)
    out = tmp_path / "dec.json"
    png = tmp_path / "dec.png"
    args = ["-q", "deconvolve", str(src), "--phase", "-n", "200", "-b", "200", "--seed", "1"]
    assert main([*args, "-o", str(out), "--plot", str(png)]) == 0
    payload = json.loads(out.read_text())
    freqs = sorted(c["frequency"] for c in payload["components"])
    assert any(abs(f + 1.0) < 2e-3 for f in freqs)
    assert any(abs(f - 0.5) < 2e-3 for f in freqs)
    assert "phase_correction" in payload["metadata"]
    assert png.stat().st_size > 10_000
    assert main([*args, "-o", str(out), "--bands", "-1.1", "-0.9", "0.4", "0.6"]) == 0
    assert len(json.loads(out.read_text())["bands"]) == 2
    with pytest.raises(SystemExit):
        main([*args, "-o", str(out), "--bands", "0.1"])


def test_plot_functions(two_peaks) -> None:
    from rjnmr import RJMCMCSampler
    from rjnmr import plotting as P

    _, y = two_peaks
    r = RJMCMCSampler(y, SamplerConfig(n_iterations=200, burn_in=200), seed=1).run()
    P.plot_posterior_spectra(r, n_samples=5)
    P.plot_frequency_trace(r, every=10)
    P.plot_spectrum(r, freq_axis=lambda w: w * 10, show_residual=False)
    fig = P.plot_diagnostics(r, freq_axis=lambda w: -w, xlim=(-1, 1))
    assert len(fig.axes) == 4
    matplotlib.pyplot.close("all")
