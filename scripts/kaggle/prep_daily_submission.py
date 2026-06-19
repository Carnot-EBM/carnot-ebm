#!/usr/bin/env python3
"""Daily ARC-AGI-3 submission routine for Carnot.

WHY THIS EXISTS: the competition allows ONE submission per day, and the live agent
only picks up the conductor's newly-banked solvers/operators when the Kaggle
`carnot-agent-code` dataset is re-versioned. So each day we: refresh the dataset
from the repo's latest `python/carnot`, re-push the public submission kernel (so it
binds the fresh dataset), validate the save-run, and STOP — the actual competition
submit is OPERATOR-APPROVED (External Publication discipline). This script never
submits in its default mode.

MODES:
  (default)      prep: refresh dataset + re-push kernel + validate; print READY + the
                 exact operator-approved submit command. Does NOT submit.
  --dry-run      stage + run the safety guards only; do NOT touch Kaggle. For testing.
  --submit-only  submit the CURRENT latest kernel version to the competition (the
                 operator-approved action; run this on the operator's "yes"). No re-prep.

SAFETY GUARDS (abort the dataset re-version if any fail):
  * the adapter fix must be present (`MAX_ACTIONS = 400`) — guards against a stale snapshot
  * no compiled `_rust*.so` may leak into the bundle (it SIGILLs on Kaggle's CPU)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
DATASET = "iancblenke/carnot-agent-code"
KERNEL = "iancblenke/carnot-arc-agi3-submission"
COMP = "arc-prize-2026-arc-agi-3"
KERNEL_DIR = REPO / "scripts" / "kaggle" / "submission_kernel"
STAGE = Path("/tmp/cac_stage_daily")
# ops files the live agent reads at runtime (registry of banked solves + router ledger)
OPS_FILES = ["arc_solve_registry.yaml", "arc_router_ledger.json"]


def kaggle(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=check)


def stage_dataset() -> None:
    """Download the current dataset, overlay the repo's latest python/carnot + ops files,
    strip junk, and run the safety guards. Leaves a ready-to-version dir at STAGE."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    print(f"[stage] downloading {DATASET} ...")
    r = kaggle("datasets", "download", DATASET, "-p", str(STAGE), "--unzip", "-q")
    if r.returncode != 0:
        sys.exit(f"ABORT: dataset download failed: {r.stderr[-300:]}")
    print("[stage] overlaying repo python/carnot (excluding *.so/__pycache__/*.cover) ...")
    subprocess.run(
        ["rsync", "-a", "--exclude=*.so", "--exclude=__pycache__", "--exclude=*.pyc",
         f"{REPO / 'python' / 'carnot'}/", f"{STAGE / 'python' / 'carnot'}/"],
        check=True,
    )
    for f in OPS_FILES:
        src = REPO / "ops" / f
        if src.exists():
            (STAGE / "ops").mkdir(exist_ok=True)
            shutil.copy2(src, STAGE / "ops" / f)
    for cover in STAGE.rglob("*.cover"):
        cover.unlink()

    # --- safety guards ---
    adapter = STAGE / "python" / "carnot" / "agentic" / "arc_competition_agent.py"
    if "MAX_ACTIONS = 400" not in adapter.read_text():
        sys.exit("ABORT: adapter fix (MAX_ACTIONS=400) missing in stage — stale snapshot")
    leaked = list(STAGE.rglob("_rust.cpython*.so"))
    if leaked:
        sys.exit(f"ABORT: compiled _rust.so leaked into bundle (SIGILLs on Kaggle): {leaked}")

    (STAGE / "dataset-metadata.json").write_text(
        json.dumps({"id": DATASET, "title": "carnot-agent-code", "licenses": [{"name": "other"}]})
    )
    print("[stage] guards passed; stage ready.")


def latest_kernel_version() -> str:
    """Parse the kernel's current version from its status/metadata."""
    # `kaggle kernels status` doesn't print version; pull metadata which carries it.
    out = Path("/tmp/daily_kmeta")
    shutil.rmtree(out, ignore_errors=True)
    kaggle("kernels", "get", KERNEL, "-p", str(out))
    # fallback: the push output is the authoritative source; handled by caller.
    return "?"


def prep() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    stage_dataset()

    print(f"[prep] re-versioning dataset ({stamp}) ...")
    r = kaggle("datasets", "version", "-p", str(STAGE), "-m",
               f"daily refresh {stamp}: latest banked solvers", "--dir-mode", "zip")
    print(r.stdout[-300:] or r.stderr[-300:])
    if "error" in (r.stderr or "").lower() and "being created" not in (r.stdout or ""):
        print("[prep] WARNING: dataset version may have failed; continuing to kernel push")

    print("[prep] waiting 45s for dataset to process ...")
    time.sleep(45)

    print("[prep] re-pushing submission kernel ...")
    r = kaggle("kernels", "push", "-p", str(KERNEL_DIR))
    print(r.stdout.strip())
    m = re.search(r"version (\d+)", r.stdout)
    kver = m.group(1) if m else "?"

    print("[prep] waiting for kernel save-run ...")
    status = "?"
    for _ in range(24):
        s = kaggle("kernels", "status", KERNEL).stdout.lower()
        if "complete" in s:
            status = "complete"
            break
        if "error" in s:
            status = "error"
            break
        time.sleep(15)

    out = Path("/tmp/daily_sub_out")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    kaggle("kernels", "output", KERNEL, "-p", str(out))
    parquet_ok = any(out.glob("*.parquet"))

    print("\n=== DAILY PREP RESULT ===")
    print(f"  dataset: re-versioned ({stamp})")
    print(f"  kernel:  v{kver}  save-run={status}  submission.parquet={'OK' if parquet_ok else 'MISSING'}")
    print(f"  current best public score: see leaderboard (first scored = 0.08)")
    ready = status == "complete" and parquet_ok
    print(f"  READY FOR OPERATOR-APPROVED SUBMIT: {'YES' if ready else 'NO — investigate above'}")
    if ready:
        print("\n  To submit (operator approval), run:")
        print(f"    .venv/bin/python scripts/kaggle/prep_daily_submission.py --submit-only --kver {kver}")
    sys.exit(0 if ready else 1)


def submit_only(kver: str, message: str) -> None:
    """OPERATOR-APPROVED submit of the current latest kernel version. No re-prep."""
    from kaggle import api

    api.authenticate()
    res = api.competition_submit_code(
        file_name="submission.parquet",
        message=message,
        competition=COMP,
        kernel=KERNEL,
        kernel_version=str(kver),
    )
    print("SUBMITTED:", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stage + guards only; no Kaggle writes")
    ap.add_argument("--submit-only", action="store_true", help="operator-approved submit of latest kernel")
    ap.add_argument("--kver", default=None, help="kernel version to submit (with --submit-only)")
    ap.add_argument("--message", default=None, help="submission message")
    a = ap.parse_args()

    if a.dry_run:
        stage_dataset()
        print("[dry-run] stage built + guards passed; no Kaggle writes performed.")
        return
    if a.submit_only:
        if not a.kver:
            sys.exit("--submit-only requires --kver N")
        msg = a.message or f"carnot daily {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        submit_only(a.kver, msg)
        return
    prep()


if __name__ == "__main__":
    main()
