"""Tests for analyse.py window metrics and n* selection."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.pocus.analyse import (
    analyse_run,
    analyse_series,
    choose_n_star,
    gpuh_encoder,
    gpuh_rl,
)
from scripts.pocus.wp5_results import render


def _steps(n: int, t: float = 1.0, typ: str = "image"):
    rows = []
    acc = 0.0
    for i in range(n):
        acc += t
        rows.append({
            "step": i,
            "stage": 1,
            "type": typ,
            "t_step": t,
            "t_data_wait": 0.1,
            "t_fwd_student": 0.4,
            "t_fwd_teachers": 0.2,
            "t_bwd": 0.2,
            "t_opt": 0.05,
            "t_allreduce": 0.05,
            "n_images": 32,
            "n_clips": 0,
            "n_frames": 32,
            "mem_peak_GB": 56.9,
            "loss": 1.0,
            "nonfinite": False,
        })
    return rows


def test_analyse_run_window_and_unaccounted():
    rows = _steps(1200)
    out = analyse_run(rows, gbs_img=2048, n_gpus=4)
    block = out["types"]["image"]
    assert block["window"]["passes_3pct"]
    assert block["unaccounted_ok"]
    assert block["throughput"]["R_img"] == 2048.0  # t̄=1s


def test_analyse_series_n_star():
    runs = []
    for n, t in [(4, 1.0), (8, 0.55), (16, 0.35), (32, 0.30)]:
        parsed = analyse_run(_steps(1200, t=t), gbs_img=2048, n_gpus=n)
        parsed["n_gpus"] = n
        runs.append(parsed)
    series = analyse_series(runs, "R_img")
    assert series["n0"] == 4
    assert series["n_star"] is not None
    assert series["efficiency"]


def test_choose_n_star_threshold():
    n, reason = choose_n_star({4: 1.0, 8: 0.9, 16: 0.72, 32: 0.50})
    assert n == 16
    assert "0.72" in reason or "16" in reason


def test_gpuh_encoder_contingency():
    mix = [
        {"image_frac": 0.9, "video_frac": 0.1},
        {"image_frac": 0.4, "video_frac": 0.6},
        {"image_frac": 0.5, "video_frac": 0.3},
        {"image_frac": 0.5, "video_frac": 0.3},
    ]
    out = gpuh_encoder(t_img=1.0, t_vid=2.0, n_star_gpus=16, mix=mix, probe_rounds=0)
    assert out["GPUh_enc_total"] == 1.05 * out["GPUh_enc"]
    assert len(out["per_stage"]) == 4


def test_gpuh_rl():
    out = gpuh_rl(t_bar=10.0, n_star_gpus=8, P=256, p=256, epochs=1, R_gen=100.0, L_ref=10.0)
    assert out["N_steps"] == 1
    assert out["GPUh_RL"] > 0


def test_wp5_results_uses_analysis_numbers(tmp_path: Path):
    analysis = {
        "series": {
            "A-stage1": {
                "n0": 4,
                "n_star": 16,
                "n_star_reason": "n*=4 nodes",
                "amdahl_f": 0.9,
                "efficiency": {"4": 1.0, "16": 0.8},
                "speedup": {"4": 1.0, "16": 3.2},
            }
        },
        "gpuh": {"encoder": {"GPUh_enc_total": 123.0}},
        "storage": {},
    }
    text = render(analysis, None)
    assert "123" in text or "123.0" in text
    assert "0.800" in text or "0.8" in text
    assert "A-stage1" in text


def test_attach_derived_fills_gpuh():
    from scripts.pocus.analyse import attach_derived
    analysis = {
        "series": {
            "A-stage1": {
                "n_star": 16,
                "n_star_reason": "n*=4 nodes",
                "t_mean": {"4": 1.0, "16": 0.35},
                "rates": {"4": 2048.0, "16": 2048 / 0.35},
            },
            "A-stage2-vid": {
                "n_star": 16,
                "t_mean": {"16": 2.0},
                "rates": {"16": 16.0},
            },
        }
    }
    attach_derived(analysis, loader={"images_per_s": 10000.0})
    enc = analysis["gpuh"]["encoder"]
    assert enc["n_star"] == 16
    assert enc["GPUh_enc_total"] > enc["GPUh_enc"]
    assert analysis["loader_headroom"]["images"]["H"] > 1.0


def test_analyse_main_writes_gpuh(tmp_path: Path, monkeypatch):
    from scripts.pocus.analyse import main

    def _write_run(jobid: str, n_gpus: int, t: float):
        d = tmp_path / "encoder" / jobid
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({
            "experiment": "E1", "workload": "A-stage1", "n_gpus": n_gpus, "jobid": jobid,
        }))
        rows = []
        for i in range(1200):
            rows.append({
                "step": i, "stage": 1, "type": "image", "t_step": t,
                "t_data_wait": 0.1 * t, "t_fwd_student": 0.4 * t,
                "t_fwd_teachers": 0.2 * t, "t_bwd": 0.2 * t, "t_opt": 0.05 * t,
                "t_allreduce": 0.05 * t, "n_images": 32, "n_clips": 0,
                "n_frames": 32, "mem_peak_GB": 56.9, "loss": 1.0, "nonfinite": False,
            })
        (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    for jobid, n, t in [("a", 4, 1.0), ("b", 8, 0.55), ("c", 16, 0.35), ("d", 32, 0.30)]:
        _write_run(jobid, n, t)

    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["analyse", "--evidence", str(tmp_path), "--out", str(out)])
    main()
    analysis = json.loads((out / "analysis.json").read_text())
    assert "A-stage1" in analysis["series"]
    assert analysis["gpuh"]["encoder"]["GPUh_enc_total"] > 0
    assert analysis["series"]["A-stage1"]["n_star"] is not None
    csv = (out / "runs.csv").read_text()
    assert "A-stage1" in csv
