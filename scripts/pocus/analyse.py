#!/usr/bin/env python3
"""
analyse.py  ·  Compute every WP5 quantity in spec §6 from JSONL logs
====================================================================

Reads ``steps.jsonl`` files (one per run) plus optional ``runs.csv``, and
writes ``analysis.json``, ``runs.csv``, ``components.csv``.

All intermediate values are stored (window mean, std, CV, half-window
difference, component shares, unaccounted time, speed-up, efficiency).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.bench import (
    GBS_IMG_DEFAULT,
    GBS_VID_DEFAULT,
    amdahl_fit,
    efficiency,
    select_measurement_window,
    speedup,
    throughput,
)

EFFICIENCY_THRESHOLD = 0.70
UNACCOUNTED_TARGET = 0.10
STAGE_STEPS = [25_000, 20_000, 20_000, 35_000]
STAGE_MIX_DEFAULT = [
    {"image_frac": 0.90, "video_frac": 0.10, "paired_frac": 0.00},
    {"image_frac": 0.40, "video_frac": 0.60, "paired_frac": 0.00},
    {"image_frac": 0.50, "video_frac": 0.30, "paired_frac": 0.20},
    {"image_frac": 0.50, "video_frac": 0.30, "paired_frac": 0.20},
]
GPUS_PER_NODE = 4
CONTINGENCY = 1.05
N_STAR_JUSTIFY_BELOW = 0.80
MONTHLY_GPUH_PRO_RATA = 47_500


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def analyse_run(
    rows: Sequence[dict],
    *,
    gbs_img: int = GBS_IMG_DEFAULT,
    gbs_vid: int = GBS_VID_DEFAULT,
    n_gpus: int = 4,
    n_frames: int = 32,
) -> dict:
    if not rows:
        return {"error": "empty log", "n_rows": 0}

    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_type[str(r.get("type") or r.get("sample_type") or "image")].append(r)

    def _window(subset: Sequence[dict]) -> dict:
        t_steps = _finite(r.get("t_step", 0.0) for r in subset)
        ts = []
        acc = 0.0
        for t in t_steps:
            acc += t
            ts.append(acc)
        wr = select_measurement_window(t_steps, ts)
        win = subset[wr.start_idx:wr.end_idx] if wr.end_idx > wr.start_idx else subset
        components = {
            "t_data_wait": _mean_key(win, "t_data_wait"),
            "t_fwd_student": _mean_key(win, "t_fwd_student"),
            "t_fwd_teachers": _mean_key(win, "t_fwd_teachers"),
            "t_bwd": _mean_key(win, "t_bwd"),
            "t_opt": _mean_key(win, "t_opt"),
            "t_allreduce": _mean_key(win, "t_allreduce"),
            # RL keys (ignored if absent)
            "t_generation": _mean_key(win, "t_generation"),
            "t_teacher": _mean_key(win, "t_teacher"),
            "t_training": _mean_key(win, "t_training"),
            "t_logprobs": _mean_key(win, "t_logprobs"),
            "t_refit": _mean_key(win, "t_refit"),
        }
        accounted = sum(v for v in components.values() if math.isfinite(v))
        unaccounted = wr.t_mean - accounted if math.isfinite(wr.t_mean) else float("nan")
        share = {
            k: (v / wr.t_mean if wr.t_mean else float("nan"))
            for k, v in components.items()
            if math.isfinite(v)
        }
        return {
            "window": {
                "t_mean": wr.t_mean,
                "s_t": wr.t_std,
                "cv": wr.cv,
                "half_rel_diff": wr.half_rel_diff,
                "stable": wr.stable,
                "extended": wr.extended,
                "n_steps": wr.n_steps,
                "warmup_discarded": wr.warmup_discarded,
                "wall_s": wr.wall_s,
                "reason": wr.reason,
                "passes_3pct": bool(wr.stable and math.isfinite(wr.half_rel_diff) and wr.half_rel_diff <= 0.03),
            },
            "components": components,
            "component_share": share,
            "unaccounted_s": unaccounted,
            "unaccounted_frac": (unaccounted / wr.t_mean) if wr.t_mean else float("nan"),
            "unaccounted_ok": (
                math.isfinite(unaccounted)
                and wr.t_mean
                and abs(unaccounted) / wr.t_mean < UNACCOUNTED_TARGET
            ),
        }

    out: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_gpus": n_gpus,
        "types": {},
    }
    for typ, subset in by_type.items():
        block = _window(subset)
        tbar = block["window"]["t_mean"]
        if typ == "video":
            r_clip = throughput(gbs_vid, tbar)
            block["throughput"] = {
                "R_clip": r_clip,
                "R_frames": r_clip * n_frames if math.isfinite(r_clip) else float("nan"),
                "unit": "clips/s",
                "GBS": gbs_vid,
            }
        else:
            r_img = throughput(gbs_img, tbar)
            # RL: tokens/s and samples/s if present
            toks = _mean_key(subset, "generated_tokens")
            prompts = _mean_key(subset, "prompts")
            responses = _mean_key(subset, "responses")
            block["throughput"] = {
                "R_img": r_img,
                "unit": "images/s",
                "GBS": gbs_img,
            }
            if math.isfinite(toks) and toks > 0 and tbar:
                block["throughput"]["R_tok"] = toks / tbar
            if math.isfinite(responses) and responses > 0 and tbar:
                block["throughput"]["R_samp"] = responses / tbar
            elif math.isfinite(prompts) and prompts > 0 and tbar:
                block["throughput"]["R_samp"] = prompts / tbar
        out["types"][typ] = block
    return out


def _mean_key(rows: Sequence[dict], key: str) -> float:
    xs = _finite(r.get(key) for r in rows if key in r)
    return sum(xs) / len(xs) if xs else float("nan")


def analyse_series(runs: List[dict], rate_key: str = "R_img") -> dict:
    rates = {}
    tmeans = {}
    components = defaultdict(dict)
    for r in runs:
        n = int(r["n_gpus"])
        # pick first type's throughput
        types = r.get("types") or {}
        block = None
        for typ in ("image", "video", "rl", "prod", "handbook"):
            if typ in types:
                block = types[typ]
                break
        if block is None and types:
            block = next(iter(types.values()))
        if block is None:
            continue
        thr = block.get("throughput") or {}
        rate = thr.get(rate_key)
        if rate is None:
            rate = thr.get("R_clip") or thr.get("R_tok") or thr.get("R_samp")
        if rate is not None and math.isfinite(rate):
            rates[n] = rate
        tmeans[n] = block.get("window", {}).get("t_mean")
        for k, v in (block.get("components") or {}).items():
            if math.isfinite(v):
                components[k][n] = 1.0 / v if v else float("nan")  # rate-like for efficiency
    n0 = min(rates) if rates else None
    s = speedup(rates, n0)
    e = efficiency(rates, n0)
    f = amdahl_fit(list(s.keys()), list(s.values()), n0) if n0 else None
    n_star, n_star_reason = choose_n_star(e)
    return {
        "rates": {str(k): v for k, v in rates.items()},
        "t_mean": {str(k): v for k, v in tmeans.items()},
        "speedup": {str(k): v for k, v in s.items()},
        "efficiency": {str(k): v for k, v in e.items()},
        "amdahl_f": f,
        "n0": n0,
        "n_star": n_star,
        "n_star_reason": n_star_reason,
        "component_efficiency": {
            name: {str(k): v for k, v in efficiency(vals, n0).items()}
            for name, vals in components.items()
            if vals
        },
    }


def choose_n_star(
    eff: Dict[int, float],
    threshold: float = EFFICIENCY_THRESHOLD,
    gpus_per_node: int = GPUS_PER_NODE,
) -> tuple:
    if not eff:
        return None, "no efficiency numbers"
    eligible = [n for n, e in eff.items() if math.isfinite(e) and e >= threshold]
    if not eligible:
        n = max(eff)
        return n, (
            f"no scale reached E(n)≥{threshold:.2f}; reporting largest scale "
            f"{n} GPUs with E={eff[n]:.3f} (must be justified if E<0.80)"
        )
    n = max(eligible)
    e = eff[n]
    nodes = n // gpus_per_node
    extra = ""
    if e < N_STAR_JUSTIFY_BELOW:
        extra = f" E(n*)={e:.3f} is below 0.80 — template asks for a justification."
    return n, (
        f"n*={nodes} nodes ({n} GPUs): largest scale with E(n)={e:.3f} ≥ {threshold:.2f}."
        + extra
    )


def gpuh_encoder(
    t_img: float,
    t_vid: float,
    n_star_gpus: int,
    mix: Sequence[dict],
    stage_steps: Sequence[int] = STAGE_STEPS,
    probe_node_hours: float = 140.0,
    probe_rounds: int = 0,
    restart_from_zero: bool = True,
) -> dict:
    """Spec §6.4."""
    gpuh = 0.0
    per_stage = []
    for k, n_k in enumerate(stage_steps):
        f_img = float(mix[k]["image_frac"]) if k < len(mix) else 1.0
        f_vid = float(mix[k].get("video_frac", 0.0)) if k < len(mix) else 0.0
        t_k = f_img * t_img + f_vid * t_vid
        hours = n_k * t_k * n_star_gpus / 3600.0
        gpuh += hours
        per_stage.append({
            "stage": k + 1,
            "N_k": n_k,
            "f_img": f_img,
            "f_vid": f_vid,
            "t_k": t_k,
            "GPUh": hours,
        })
    probes = probe_node_hours * probe_rounds * (n_star_gpus / GPUS_PER_NODE) / 1.0
    # probe_node_hours is already node-hours; convert to GPUh
    probes_gpuh = probe_node_hours * probe_rounds * GPUS_PER_NODE
    total = CONTINGENCY * (gpuh + probes_gpuh)
    t_enc_days = sum(s["N_k"] * s["t_k"] for s in per_stage) / 86400.0
    return {
        "per_stage": per_stage,
        "GPUh_enc": gpuh,
        "GPUh_probes": probes_gpuh,
        "contingency": CONTINGENCY,
        "GPUh_enc_total": total,
        "T_enc_days": t_enc_days,
        "T_enc_days_with_queue_margin": t_enc_days * 1.20,
        "restart_from_zero": restart_from_zero,
        "note": (
            "request is a full 100k-step run from step 0"
            if restart_from_zero
            else "request continues from step 30,800 [TBC]"
        ),
    }


def gpuh_rl(
    t_bar: float,
    n_star_gpus: int,
    P: int,
    p: int,
    epochs: int = 1,
    L_ref: float = 128.0,
    R_gen: float = float("nan"),
    n_ref_gpus: int = 4,
    n_eval: int = 0,
    gpuh_per_eval: float = 0.0,
) -> dict:
    """Spec §6.5."""
    n_steps = math.ceil(P / p) if p else 0
    gpuh = epochs * n_steps * t_bar * n_star_gpus / 3600.0
    if R_gen and math.isfinite(R_gen) and R_gen > 0:
        gpuh_ref = P * L_ref / R_gen * n_ref_gpus / 3600.0
    else:
        gpuh_ref = float("nan")
    gpuh_eval = n_eval * gpuh_per_eval
    parts = [x for x in (gpuh, gpuh_ref, gpuh_eval) if math.isfinite(x)]
    total = CONTINGENCY * sum(parts) if parts else float("nan")
    t_days = epochs * n_steps * t_bar / 86400.0
    return {
        "P": P,
        "p": p,
        "N_steps": n_steps,
        "epochs": epochs,
        "t_bar": t_bar,
        "GPUh_RL": gpuh,
        "GPUh_ref": gpuh_ref,
        "GPUh_eval": gpuh_eval,
        "GPUh_RL_total": total,
        "T_RL_days": t_days,
        "T_RL_days_with_queue_margin": t_days * 1.20,
        "monthly_gpuh_if_this_month": total,
        "monthly_pro_rata": MONTHLY_GPUH_PRO_RATA,
    }


def storage_block(data_facts: Optional[dict], ckpt: Optional[dict], n_star: Optional[int], rates: dict) -> dict:
    return {
        "data_facts": data_facts or {},
        "checkpoints": ckpt or {},
        "B_read": _required_read_bw(rates),
        "n_star_gpus": n_star,
        "file_count_note": "compare shards + checkpoints + logs with the 2.5 M-file request",
    }


def _required_read_bw(rates: dict) -> dict:
    r_img = rates.get("R_img")
    r_clip = rates.get("R_clip")
    mean_img = rates.get("mean_bytes_image")
    mean_clip = rates.get("mean_bytes_clip")
    b = 0.0
    if r_img and mean_img and math.isfinite(r_img) and math.isfinite(mean_img):
        b += r_img * mean_img
    if r_clip and mean_clip and math.isfinite(r_clip) and math.isfinite(mean_clip):
        b += r_clip * mean_clip
    return {"B_read_bytes_s": b, "R_img": r_img, "R_clip": r_clip}


def write_runs_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["workload", "gpus", "throughput", "unit", "parallel_setup", "jobid", "log"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_components_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


WORKLOAD_FROM_EXP = {
    "E0": "E0",
    "E1": "A-stage1",
    "E2": "A-stage2",
    "E3": "A-stage3",
    "E4": "E4",
    "E5": "E5",
    "E6": "E6",
    "R0": "B-probe",
    "R1": "B-prod",
    "R2": "B-handbook",
    "R3": "B-learn",
    "R4": "B-ref",
    "R5": "B-video",
}


def loader_headroom(r_loader: float, r_needed: float) -> dict:
    """Spec §6.1: H = R_loader(E5) / R_needed(n*). H < 1.5 ⇒ I/O-bound."""
    if not r_loader or not r_needed or not math.isfinite(r_loader) or not math.isfinite(r_needed) or r_needed <= 0:
        return {"H": float("nan"), "R_loader": r_loader, "R_needed": r_needed, "io_bound": None}
    h = r_loader / r_needed
    return {"H": h, "R_loader": r_loader, "R_needed": r_needed, "io_bound": h < 1.5}


def load_run_meta(job_dir: Path) -> dict:
    p = job_dir / "run.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    env: Dict[str, str] = {}
    envp = job_dir / "env.txt"
    if envp.exists():
        for line in envp.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    exp = env.get("POCUS_EXPERIMENT") or job_dir.parent.name
    exp = str(exp).upper() if str(exp).lower() in {e.lower() for e in WORKLOAD_FROM_EXP} else str(exp)
    if exp.lower() in {k.lower() for k in WORKLOAD_FROM_EXP}:
        exp = next(k for k in WORKLOAD_FROM_EXP if k.lower() == exp.lower())
    return {
        "experiment": exp,
        "workload": WORKLOAD_FROM_EXP.get(exp, str(exp)),
        "n_gpus": env.get("SLURM_NTASKS") or env.get("WORLD_SIZE"),
    }


def series_workload_name(base: str, typ: str) -> str:
    if base == "A-stage2":
        if typ == "video":
            return "A-stage2-vid"
        return "A-stage2-img"
    if base == "A-stage1" and typ == "video":
        return "A-stage1-vid"
    return base


def attach_derived(
    analysis: dict,
    *,
    mix: Optional[Sequence[dict]] = None,
    P: int = 218_402,
    p: int = 256,
    epochs: int = 1,
    data_facts: Optional[dict] = None,
    ckpt: Optional[dict] = None,
    loader: Optional[dict] = None,
) -> dict:
    """Fill gpuh / storage / loader-headroom (spec §6.3–6.6) from series."""
    mix = list(mix or STAGE_MIX_DEFAULT)
    series = analysis.get("series") or {}
    img_series = series.get("A-stage1") or series.get("A-stage2-img")
    vid_series = series.get("A-stage2-vid")
    n_star = None
    n_star_reason = ""
    t_img = float("nan")
    t_vid = float("nan")
    r_img = float("nan")
    r_clip = float("nan")
    if img_series:
        n_star = img_series.get("n_star")
        n_star_reason = img_series.get("n_star_reason") or ""
        t_img = (img_series.get("t_mean") or {}).get(str(n_star), float("nan"))
        r_img = (img_series.get("rates") or {}).get(str(n_star), float("nan"))
    if vid_series:
        if n_star is None:
            n_star = vid_series.get("n_star")
            n_star_reason = vid_series.get("n_star_reason") or n_star_reason
        t_vid = (vid_series.get("t_mean") or {}).get(str(n_star), float("nan"))
        r_clip = (vid_series.get("rates") or {}).get(str(n_star), float("nan"))
    gpuh: Dict[str, Any] = {}
    if n_star is not None:
        gpuh["encoder"] = gpuh_encoder(
            t_img=float(t_img) if t_img is not None and math.isfinite(float(t_img)) else 0.0,
            t_vid=float(t_vid) if t_vid is not None and math.isfinite(float(t_vid)) else 0.0,
            n_star_gpus=int(n_star),
            mix=mix,
        )
        gpuh["encoder"]["n_star"] = n_star
        gpuh["encoder"]["n_star_reason"] = n_star_reason
        gpuh["encoder"]["t_img"] = t_img
        gpuh["encoder"]["t_vid"] = t_vid
    rl_series = series.get("B-prod") or series.get("B-handbook")
    if rl_series and rl_series.get("n_star") is not None:
        t_bar = (rl_series.get("t_mean") or {}).get(str(rl_series["n_star"]), float("nan"))
        gpuh["rl"] = gpuh_rl(
            t_bar=float(t_bar) if t_bar and math.isfinite(float(t_bar)) else 0.0,
            n_star_gpus=int(rl_series["n_star"]),
            P=P,
            p=p,
            epochs=epochs,
        )
        gpuh["rl"]["n_star"] = rl_series["n_star"]
        gpuh["rl"]["n_star_reason"] = rl_series.get("n_star_reason")
    analysis["gpuh"] = gpuh
    rates = {"R_img": r_img, "R_clip": r_clip}
    if data_facts:
        img_m = (data_facts.get("manifests") or {}).get("enc_images") or {}
        vid_m = (data_facts.get("manifests") or {}).get("enc_videos") or {}
        rates["mean_bytes_image"] = img_m.get("bytes_mean")
        rates["mean_bytes_clip"] = vid_m.get("bytes_mean")
    analysis["storage"] = storage_block(data_facts, ckpt, int(n_star) if n_star else None, rates)
    headroom = {}
    if loader:
        if loader.get("images_per_s") and r_img and math.isfinite(float(r_img)):
            headroom["images"] = loader_headroom(float(loader["images_per_s"]), float(r_img))
        if loader.get("clips_per_s") and r_clip and math.isfinite(float(r_clip)):
            headroom["clips"] = loader_headroom(float(loader["clips_per_s"]), float(r_clip))
    analysis["loader_headroom"] = headroom
    return analysis


def _infer_n_gpus(job_dir: Path, rows: List[dict], meta: Optional[dict] = None) -> int:
    if meta:
        for key in ("n_gpus", "gpus", "world_size"):
            v = meta.get(key)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        if meta.get("nodes") is not None:
            try:
                return int(meta["nodes"]) * GPUS_PER_NODE
            except (TypeError, ValueError):
                pass
    env = job_dir / "env.txt"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("SLURM_NTASKS=") or line.startswith("WORLD_SIZE="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
            if line.startswith("SLURM_NNODES="):
                try:
                    nodes = int(line.split("=", 1)[1].strip())
                    return nodes * GPUS_PER_NODE
                except ValueError:
                    pass
    if rows and "n_gpus" in rows[0]:
        return int(rows[0]["n_gpus"])
    return 4


def collect_jobs(root: Path) -> List[Path]:
    jobs = []
    for p in root.rglob("steps.jsonl"):
        jobs.append(p.parent)
    return sorted(jobs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True, help="pocus/ evidence root")
    p.add_argument("--data-facts", type=Path, default=None)
    p.add_argument("--gbs-img", type=int, default=GBS_IMG_DEFAULT)
    p.add_argument("--gbs-vid", type=int, default=GBS_VID_DEFAULT)
    p.add_argument("--P", type=int, default=218_402, help="RL prompt set size [TBC]")
    p.add_argument("--p", type=int, default=256, help="RL prompts per step (production)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    evidence: Path = args.evidence
    out_dir = args.out or evidence
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    runs_csv = []
    components_csv = []
    series: Dict[str, List[dict]] = defaultdict(list)
    loader_stats: Dict[str, float] = {}
    ckpt_stats: Dict[str, Any] = {}
    skip_types = {"ckpt_probe", "loader_only", "mixed"}

    for job_dir in collect_jobs(evidence):
        meta = load_run_meta(job_dir)
        rows = load_jsonl(job_dir / "steps.jsonl")
        n_gpus = _infer_n_gpus(job_dir, rows, meta)
        parsed = analyse_run(rows, gbs_img=args.gbs_img, gbs_vid=args.gbs_vid, n_gpus=n_gpus)
        experiment = str(meta.get("experiment") or "")
        base = str(meta.get("workload") or WORKLOAD_FROM_EXP.get(experiment, job_dir.parent.name))
        jobid = str(meta.get("jobid") or job_dir.name)
        parsed["workload"] = base
        parsed["experiment"] = experiment
        parsed["jobid"] = jobid
        parsed["n_gpus"] = n_gpus
        parsed["log"] = str(job_dir / "steps.jsonl")
        parsed["meta"] = meta
        per_run.append(parsed)

        loader_path = job_dir / "loader_only.json"
        if loader_path.exists():
            paths = [loader_path]
        else:
            paths = []
        paths.extend(p for p in job_dir.glob("loader_only*.json") if p not in paths)
        for loader_path in paths:
            lo = json.loads(loader_path.read_text(encoding="utf-8"))
            for k in ("images_per_s", "clips_per_s"):
                v = lo.get(k)
                if v and math.isfinite(float(v)):
                    loader_stats[k] = max(float(v), float(loader_stats.get(k, 0.0)))
        ckpt_path = job_dir / "ckpt_probe.json"
        if ckpt_path.exists():
            ckpt_stats = json.loads(ckpt_path.read_text(encoding="utf-8"))

        scaling = experiment in {"E1", "E2", "E3", "R1", "R2"} or base in {
            "A-stage1", "A-stage2", "A-stage3", "B-prod", "B-handbook",
        }
        for typ, block in (parsed.get("types") or {}).items():
            if typ in skip_types:
                continue
            wl = series_workload_name(base, typ)
            if scaling:
                typed = dict(parsed)
                typed["types"] = {typ: block}
                series[wl].append(typed)
            thr = block.get("throughput") or {}
            rate = thr.get("R_img") or thr.get("R_clip") or thr.get("R_tok") or thr.get("R_samp")
            unit = thr.get("unit") or ""
            runs_csv.append({
                "workload": wl,
                "gpus": n_gpus,
                "throughput": rate,
                "unit": unit,
                "parallel_setup": f"DDP/{n_gpus}",
                "jobid": jobid,
                "log": str(job_dir / "steps.jsonl"),
            })
            row = {"workload": wl, "gpus": n_gpus, "jobid": jobid, **(block.get("components") or {})}
            components_csv.append(row)

    series_out = {}
    for name, runs in series.items():
        rate_key = "R_clip" if "vid" in name else "R_img"
        if name.startswith("B") or name in ("rl",):
            rate_key = "R_tok"
        series_out[name] = analyse_series(runs, rate_key=rate_key)

    data_facts = None
    facts_path = args.data_facts
    if facts_path is None:
        cand = evidence / "data_facts.json"
        facts_path = cand if cand.exists() else None
    if facts_path and Path(facts_path).exists():
        data_facts = json.loads(Path(facts_path).read_text(encoding="utf-8"))

    analysis = {
        "runs": per_run,
        "series": series_out,
        "gpuh": {},
        "storage": {},
        "constants": {
            "GBS_img": args.gbs_img,
            "GBS_vid": args.gbs_vid,
            "efficiency_threshold": EFFICIENCY_THRESHOLD,
            "unaccounted_target": UNACCOUNTED_TARGET,
            "stage_steps": STAGE_STEPS,
            "stage_mix": STAGE_MIX_DEFAULT,
            "contingency": CONTINGENCY,
            "P": args.P,
            "p": args.p,
            "epochs": args.epochs,
        },
    }
    attach_derived(
        analysis,
        mix=STAGE_MIX_DEFAULT,
        P=args.P,
        p=args.p,
        epochs=args.epochs,
        data_facts=data_facts,
        ckpt=ckpt_stats or None,
        loader=loader_stats or None,
    )
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str) + "\n")
    write_runs_csv(out_dir / "runs.csv", runs_csv)
    write_components_csv(out_dir / "components.csv", components_csv)
    print(f"wrote {out_dir / 'analysis.json'}  ({len(per_run)} runs, series={list(series_out)})")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
