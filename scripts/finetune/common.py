#!/usr/bin/env python3
"""
scripts/finetune/common.py  ·  Shared helpers for finetune experiment launchers
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent.parent.parent
ACCOUNT = "a127"
PARTITION = "normal"
EDF_ENV = Path.home() / ".edf" / "ultatron.toml"
LOG_ROOT = REPO_DIR / "logs" / "finetune"

# Prefer Python >=3.10 (login-node python3 may be 3.6).
import shutil
import subprocess
import sys as _sys


def _find_finetune_python() -> str:
    if _sys.version_info >= (3, 10):
        return _sys.executable
    for name in ("python3.12", "python3.11"):
        exe = shutil.which(name)
        if exe:
            r = subprocess.run(
                [exe, "-c", "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)"],
                capture_output=True,
            )
            if r.returncode == 0:
                return exe
    return _sys.executable


FINETUNE_PYTHON = _find_finetune_python()

STORE = Path("/capstor/store/cscs/swissai/a127/ultrasound")
RESULTS_ROOT = REPO_DIR / "results" / "finetune"
DEFAULT_COMPARISON_CONFIG = REPO_DIR / "configs" / "finetune" / "comparison_representative.yaml"

DEFAULT_BACKBONES = [
    "student_img",
    "student_vid",
    "resnet50",
    "vit_b_16",
    "dinov3_l",
    "biomedclip",
    "usfm",
    "echocare",
]

DATASET_ROOTS = {
    "busi": STORE / "raw" / "breast" / "BUSI",
    "busbra": STORE / "raw" / "breast" / "BUSBRA",
    "tn3k": STORE / "raw" / "thyroid" / "TN3K",
    "echonet": STORE / "raw" / "cardiac" / "EchoNet-Dynamic",
    "camus": STORE / "raw" / "cardiac" / "CAMUS",
    "benin": STORE / "raw" / "lung" / "Benin_Videos",
    "rsa": STORE / "raw" / "lung" / "RSA_Videos",
    "fetal_planes_db": STORE / "raw" / "fetal" / "FETAL-PLANES-DB",
}

ALL_EXPERIMENTS = ["busi", "echonet", "camus", "lus", "lus_video"]
MAX_GPUS_PER_NODE = 4


def _die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _info(msg: str) -> None:
    print(f"[INFO]  {msg}")


def resolve_output_dir(comparison_config: str | Path) -> Path:
    """Read output_dir from comparison YAML."""
    cfg_path = Path(comparison_config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_DIR / cfg_path
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg.get("output_dir", "results/finetune/representative"))
    if not out.is_absolute():
        out = REPO_DIR / out
    return out


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--comparison-config",
        default=str(DEFAULT_COMPARISON_CONFIG),
        help="Comparison YAML (default: comparison_representative.yaml)",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--local", action="store_true", help="Run on current node (no sbatch)")
    parser.add_argument("--after-job", default=None, help="Slurm dependency: afterok:JOBID")
    parser.add_argument(
        "--backbones", nargs="+", default=None,
        help=f"Backbone subset (default: {', '.join(DEFAULT_BACKBONES)})",
    )
    parser.add_argument("--all-backbones", action="store_true", help="Use every backbone in config")
    parser.add_argument("--gpus", type=int, default=None, help="GPUs for run_all parallel layout")


def resolve_backbones_arg(args: argparse.Namespace) -> str:
    if args.all_backbones and args.backbones:
        _die("Use either --all-backbones or --backbones, not both")
    if args.all_backbones:
        return ""
    # Custom comparison configs define their own backbone list (e.g. openus
    # instead of echocare); use all entries unless the caller overrides.
    custom_config = Path(args.comparison_config).resolve() != DEFAULT_COMPARISON_CONFIG.resolve()
    if custom_config and args.backbones is None:
        return ""
    keys = args.backbones or DEFAULT_BACKBONES
    return "--backbones " + " ".join(keys)


def finetune_python_cmd(
    *,
    comparison_config: str,
    experiments: list[str] | None = None,
    backbones_arg: str = "",
    eval_only: bool = False,
    parallel_experiments: bool = False,
    num_gpus: int | None = None,
) -> str:
    """Shell command fragment for scripts/finetune.py (runtime Python resolved in inner script)."""
    parts = [
        f'"${{FINETUNE_PYTHON}}" scripts/finetune.py',
        f"--comparison-config {comparison_config}",
        f"--busi-root    {DATASET_ROOTS['busi']}",
        f"--echonet-root {DATASET_ROOTS['echonet']}",
        f"--camus-root   {DATASET_ROOTS['camus']}",
        f"--benin-root   {DATASET_ROOTS['benin']}",
        f"--rsa-root     {DATASET_ROOTS['rsa']}",
    ]
    if experiments:
        parts.append("--experiments " + " ".join(experiments))
    if eval_only:
        parts.append("--eval-only")
    if backbones_arg:
        parts.append(backbones_arg)
    if parallel_experiments:
        parts.append("--parallel-experiments")
        if num_gpus is not None:
            parts.append(f"--num-gpus {num_gpus}")
    else:
        parts.append("--no-parallel-experiments")
    return " \\\n    ".join(parts)


def write_inner_script(
    *,
    comparison_config: str,
    experiments: list[str] | None,
    backbones_arg: str,
    eval_only: bool,
    parallel_experiments: bool,
    num_gpus: int | None,
    log_label: str,
) -> Path:
    output_dir = resolve_output_dir(comparison_config)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    inner = LOG_ROOT / f".inner_finetune_{log_label}.sh"

    ft_cmd = finetune_python_cmd(
        comparison_config=comparison_config,
        experiments=experiments,
        backbones_arg=backbones_arg,
        eval_only=eval_only,
        parallel_experiments=parallel_experiments,
        num_gpus=num_gpus,
    )

    content = f"""#!/bin/bash
set -euo pipefail
ulimit -c 0
cd {REPO_DIR}
export PYTHONPATH="{REPO_DIR}:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=$(echo "${{LD_LIBRARY_PATH:-}}" | tr ':' '\\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

echo "================================================================"
echo " Ultatron — finetune ({log_label})"
echo " Job    : ${{SLURM_JOB_ID:-local}}"
echo " Node   : $(hostname)"
echo " Start  : $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \\
    | awk '{{print " GPU     :", $0}}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""
echo "Config     : {comparison_config}"
echo "Results dir : {output_dir}"
echo "Checkpoints  : /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Finetune/"
echo ""

bash "{REPO_DIR}/scripts/ensure_deps.sh"
bash "{REPO_DIR}/scripts/ensure_ablation_deps.sh"
source "{REPO_DIR}/scripts/setup_hf_cache.sh"
source "{REPO_DIR}/scripts/finetune/runtime_python.sh"
FINETUNE_PYTHON="$(resolve_finetune_python)"
echo "Python: ${{FINETUNE_PYTHON}} ($(${{FINETUNE_PYTHON}} --version 2>&1))"

{ft_cmd}

FT_EXIT=$?

"${{FINETUNE_PYTHON}}" scripts/finetune/report.py --dashboard || true

echo ""
echo "================================================================"
echo " FINETUNE COMPLETE  exit=${{FT_EXIT}}  $(date)"
[[ ${{FT_EXIT}} -eq 0 ]] && echo " Report  : {output_dir}/comparison_report.md"
[[ ${{FT_EXIT}} -eq 0 ]] && echo " Charts  : {output_dir}/charts/"
[[ ${{FT_EXIT}} -eq 0 ]] && echo " Dashboard: {RESULTS_ROOT}/_dashboard/"
echo "================================================================"
exit ${{FT_EXIT}}
"""
    inner.write_text(content)
    inner.chmod(0o755)
    return inner


def run_local(inner_script: Path, log_name: str) -> int:
    from datetime import datetime

    log_path = LOG_ROOT / f"{log_name}_local_{datetime.now():%Y%m%d_%H%M%S}.log"
    _info(f"Running locally on {os.uname().nodename} (log → {log_path})")
    with log_path.open("w") as log_f:
        proc = subprocess.Popen(
            ["bash", str(inner_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        return proc.wait()


def submit_slurm(
    inner_script: Path,
    *,
    job_name: str,
    gpus: int = 1,
    cpus: int | None = None,
    time_limit: str = "12:00:00",
    after_job: str | None = None,
) -> str:
    cpus = cpus or max(16, gpus * 4)
    outer = tempfile.NamedTemporaryFile(
        mode="w", prefix="ultatron_outer_finetune_", suffix=".sh", delete=False,
    )
    outer.write(f"""#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment={EDF_ENV} \\
     bash {inner_script}
""")
    outer.close()
    os.chmod(outer.name, 0o755)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--nodes=1",
        "--ntasks-per-node=1",
        f"--gpus-per-node={gpus}",
        f"--cpus-per-task={cpus}",
        f"--time={time_limit}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--output={LOG_ROOT}/{job_name}_%j.out",
        f"--error={LOG_ROOT}/{job_name}_%j.err",
        "--parsable",
    ]
    if after_job:
        cmd.append(f"--dependency=afterok:{after_job}")

    result = subprocess.run(cmd + [outer.name], capture_output=True, text=True)
    if result.returncode != 0:
        _die(
            "sbatch failed.\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}\n"
            f"  GH200 nodes have at most {MAX_GPUS_PER_NODE} GPUs."
        )
    return result.stdout.strip()


def launch_experiment(
    experiment: str,
    args: argparse.Namespace,
    *,
    job_name: str | None = None,
) -> int | str:
    """Run or submit a single comparison experiment (all backbones on 1 GPU)."""
    comparison_config = args.comparison_config
    if not Path(comparison_config).is_file():
        _die(f"Comparison config not found: {comparison_config}")

    backbones_arg = resolve_backbones_arg(args)
    label = experiment
    inner = write_inner_script(
        comparison_config=comparison_config,
        experiments=[experiment],
        backbones_arg=backbones_arg,
        eval_only=args.eval_only,
        parallel_experiments=False,
        num_gpus=None,
        log_label=label,
    )

    if args.local:
        return run_local(inner, f"finetune_{experiment}")

    jname = job_name or f"ultatron_ft_{experiment}"
    job_id = submit_slurm(inner, job_name=jname, gpus=1, after_job=args.after_job)
    _info(f"Submitted {jname}  job_id={job_id}")
    _info(f"  log: {LOG_ROOT}/{jname}_{job_id}.out")
    return job_id


def launch_all(args: argparse.Namespace) -> int | str:
    """Run or submit the full comparison sweep (parallel per experiment)."""
    import yaml

    comparison_config = args.comparison_config
    cfg_path = Path(comparison_config)
    if not cfg_path.is_file():
        _die(f"Comparison config not found: {comparison_config}")

    with cfg_path.open() as f:
        cmp_cfg = yaml.safe_load(f)
    n_experiments = len(cmp_cfg.get("experiments", ALL_EXPERIMENTS))

    backbones_arg = resolve_backbones_arg(args)
    requested_gpus = args.gpus if args.gpus is not None else n_experiments
    gpus = min(requested_gpus, MAX_GPUS_PER_NODE)
    if requested_gpus > MAX_GPUS_PER_NODE:
        _info(
            f"Capping parallel GPUs: {requested_gpus} experiment(s) → "
            f"{gpus} GPU(s) (node max {MAX_GPUS_PER_NODE})"
        )

    inner = write_inner_script(
        comparison_config=comparison_config,
        experiments=None,
        backbones_arg=backbones_arg,
        eval_only=args.eval_only,
        parallel_experiments=True,
        num_gpus=gpus,
        log_label="all",
    )

    if args.local:
        return run_local(inner, "finetune_all")

    job_id = submit_slurm(
        inner,
        job_name="ultatron_finetune",
        gpus=gpus,
        after_job=args.after_job,
    )
    _info(f"Submitted ultatron_finetune  job_id={job_id}  gpus={gpus}")
    _info(f"  log: {LOG_ROOT}/ultatron_finetune_{job_id}.out")
    return job_id
