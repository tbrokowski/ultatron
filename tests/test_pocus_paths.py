"""WP5 billed account is a0238; evidence is under that store, not infra01."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    drop = {
        "POCUS_ACCOUNT",
        "ULTATRON_ACCOUNT",
        "POCUS_EVIDENCE_ROOT",
        "POCUS_STORE_ACCT",
        "POCUS_RAW_ROOT",
        "POCUS_SHARD_ROOT",
        "POCUS_MANIFEST_ROOT",
        "POCUS_DATA_ROOT",
        "_POCUS_ACCOUNT_SH",
    }
    return {k: v for k, v in os.environ.items() if k not in drop}


def test_python_defaults_use_a0238_not_infra01():
    code = (
        "from scripts.pocus.paths import ACCOUNT_DEFAULT, EVIDENCE_ROOT, RAW_ROOT\n"
        "print(ACCOUNT_DEFAULT)\n"
        "print(EVIDENCE_ROOT)\n"
        "print(RAW_ROOT)\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=_clean_env(),
        text=True,
    )
    acct, evid, raw = out.strip().splitlines()
    assert acct == "a0238"
    assert evid == "/capstor/store/cscs/swissai/a0238/meditron-feasibility-review/pocus"
    assert raw == "/capstor/store/cscs/swissai/a0238/pocus-bench/raw"
    assert "infra01" not in evid


def test_account_sh_defaults():
    script = (
        "source scripts/pocus/account.sh\n"
        "printf '%s\\n' \"$ACCOUNT\" \"$EVIDENCE\" \"$MANIFESTS\"\n"
    )
    out = subprocess.check_output(
        ["bash", "-c", script],
        cwd=str(REPO),
        env=_clean_env(),
        text=True,
    )
    acct, evid, mans = out.strip().splitlines()
    assert acct == "a0238"
    assert evid.endswith("/a0238/meditron-feasibility-review/pocus")
    assert mans.endswith("/a0238/pocus-bench/manifests")
    assert "infra01" not in evid


def test_account_sh_honours_pocus_account():
    env = _clean_env()
    env["POCUS_ACCOUNT"] = "a0238"
    script = "source scripts/pocus/account.sh; printf '%s\\n' \"$ACCOUNT\" \"$EVIDENCE\""
    out = subprocess.check_output(
        ["bash", "-c", script],
        cwd=str(REPO),
        env=env,
        text=True,
    )
    acct, evid = out.strip().splitlines()
    assert acct == "a0238"
    assert "/swissai/a0238/" in evid
    assert "infra01" not in evid


def test_pocus_ensure_dir_falls_back(tmp_path: Path):
    fallback = tmp_path / "fallback"
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    dest = blocker / "nccl"
    script = (
        "source scripts/pocus/account.sh\n"
        f'pocus_ensure_dir "{dest}" "{fallback}"\n'
    )
    out = subprocess.check_output(
        ["bash", "-c", script],
        cwd=str(REPO),
        env=_clean_env(),
        text=True,
        stderr=subprocess.STDOUT,
    )
    last = out.strip().splitlines()[-1]
    assert last == str(fallback)
    assert fallback.is_dir()


def test_launch_all_dry_run_prints_a0238():
    env = _clean_env()
    env["POCUS_ACCOUNT"] = "a0238"
    proc = subprocess.run(
        ["bash", "scripts/pocus/launch_all.sh", "--dry-run", "--encoder-only"],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    blob = proc.stdout + proc.stderr
    assert "account=a0238" in blob
    assert "/swissai/a0238/meditron-feasibility-review/pocus" in blob
    assert "infra01" not in blob
    assert "submit_nccl.sh --slingshot" in blob


def test_ckpt_probe_store_follows_account():
    env = _clean_env()
    env["POCUS_ACCOUNT"] = "a0238"
    code = (
        "import argparse, os\n"
        "from train.student_pretrain import _ckpt_probe_paths\n"
        "args = argparse.Namespace(ckpt_probe_scratch=None, ckpt_probe_store=None)\n"
        "_, store = _ckpt_probe_paths(args)\n"
        "print(store)\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        text=True,
    )
    assert out.strip() == "/capstor/store/cscs/swissai/a0238/pocus-bench/ckpt_probe"
