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
from datetime import UTC, datetime
from pathlib import Path
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO = repo_root()
DATASET = "iancblenke/carnot-agent-code"
KERNEL = "iancblenke/carnot-arc-agi3-submission"
COMP = "arc-prize-2026-arc-agi-3"
KERNEL_DIR = REPO / "scripts" / "kaggle" / "submission_kernel"
STAGE = Path("/tmp/cac_stage_daily")
# ops files the live agent reads at runtime (registry of banked solves + router ledger)
OPS_FILES = ["arc_solve_registry.yaml", "arc_router_ledger.json"]


# Fields that describe a SUBMISSION rather than a prep. They belong to the kernel version that
# was submitted, so when a new version is prepped they are retired into history rather than left
# beside a fresh, unsubmitted prep where they would read as "this one is already submitted".
_SUBMISSION_FIELDS = (
    "submitted",
    "submitted_at",
    "submission_ref",
    "submission_status_at_check",
    "local_gate_result_at_submit",
)


def _merge_prep_status(prior: dict, fresh: dict) -> dict:
    """Merge a fresh prep record over the durable status file, preserving everything else.

    WHY THIS EXISTS. `ops/arc-daily-prep-status.json` is a never-prune record. Beside the six
    prep fields this script computes, it carries the submission trail -- `submission_ref`,
    `submitted_at`, `local_gate_result_at_submit`, and `prior_submission_scores`, which is the
    actual leaderboard score-by-date history. The original code wrote the file with a bare
    `write_text(json.dumps({...six keys}))`, which DELETED all of it. Because this script runs
    from an unattended systemd timer, the destruction was silent, and the next `git add -A`
    (the conductor's normal path) would have published it. Recovery required knowing to look in
    one specific commit.

    A plain `{**prior, **fresh}` fixes the deletion but introduces a subtler lie: the previous
    version's `submitted: true` would sit next to a brand-new, unsubmitted kernel version and
    read as though THIS prep had been submitted. So on a version change the submission fields
    are moved into an append-only `submission_history` (kept, per never-prune) and cleared from
    the live record. `prior_submission_scores` is deliberately NOT retired -- it is a cumulative
    score history, not a per-version fact.
    """
    merged = {**prior, **fresh}
    changed_version = prior.get("kernel_version") not in (None, fresh.get("kernel_version"))
    if changed_version and prior.get("submitted"):
        history = list(merged.get("submission_history") or [])
        history.append(
            {
                k: prior[k]
                for k in ("kernel_version", "prepped_at", "note", *_SUBMISSION_FIELDS)
                if k in prior
            }
        )
        merged["submission_history"] = history
        for key in _SUBMISSION_FIELDS:
            merged.pop(key, None)
    return merged


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
        [
            "rsync",
            "-a",
            "--exclude=*.so",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            f"{REPO / 'python' / 'carnot'}/",
            f"{STAGE / 'python' / 'carnot'}/",
        ],
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

    # import+build smoke against the STAGED tree: a non-importable / broken agent (e.g. a core file
    # caught mid-edit by the daily rsync) would be 0 on EVERY eval game. The existing guards only
    # string-check MAX_ACTIONS and the .so leak -- neither validates that the agent actually imports
    # and constructs. Catch the catastrophic-zero case at bundle time, before a scored slot is spent.
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r); "
            "import carnot.agentic.arc_competition_agent as m; "
            "cls = m.make_carnot_agent(type('S', (), {'MAX_ACTIONS': 80})); "
            "assert isinstance(cls, type), 'make_carnot_agent did not return a class'; "
            "print('agent import+build smoke OK')" % str(STAGE / "python"),
        ],
        capture_output=True,
        text=True,
    )
    if smoke.returncode != 0:
        sys.exit(
            "ABORT: staged agent failed import+build smoke (would be 0 on every eval game):\n"
            f"{smoke.stdout[-400:]}\n{smoke.stderr[-1000:]}"
        )
    print("[stage] agent import+build smoke passed.")

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
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    stage_dataset()

    print(f"[prep] re-versioning dataset ({stamp}) ...")
    r = kaggle(
        "datasets",
        "version",
        "-p",
        str(STAGE),
        "-m",
        f"daily refresh {stamp}: latest banked solvers",
        "--dir-mode",
        "zip",
    )
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

    ready = status == "complete" and parquet_ok
    # Status file: the durable hand-off between the (session-independent) systemd timer that
    # runs this prep and the in-session watchdog/operator that approves + submits. The timer
    # can prep but cannot push-notify or submit (those need an agent / are operator-only), so
    # it records readiness here; the watchdog surfaces it and the operator approves the submit.
    status_path = REPO / "ops" / "arc-daily-prep-status.json"
    # MERGE, never replace. This file is a never-prune record: alongside the six prep fields
    # written here it carries the SUBMISSION history -- `submission_ref`, `submitted_at`,
    # `local_gate_result_at_submit`, and `prior_submission_scores` (the actual leaderboard
    # score-by-date trail). A bare `write_text(json.dumps({...6 keys}))` destroyed all of it on
    # 2026-07-29: the file is written by an unattended systemd timer, so the loss was silent
    # and would have been published by the next `git add -A`. Recovering it needed a specific
    # commit (bc2623761) that nobody would have known to look for.
    prior: dict = {}
    if status_path.exists():
        try:
            prior = json.loads(status_path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}  # a corrupt/unreadable prior must not block the prep; start fresh

    merged = _merge_prep_status(
        prior,
        {
            "prepped_at": stamp,
            "kernel_version": kver,
            "save_run": status,
            "parquet_ok": parquet_ok,
            "ready_for_operator_submit": ready,
            "submit_command": f".venv/bin/python scripts/kaggle/prep_daily_submission.py --submit-only --kver {kver}",
        },
    )
    status_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

    print("\n=== DAILY PREP RESULT ===")
    print(f"  dataset: re-versioned ({stamp})")
    print(
        f"  kernel:  v{kver}  save-run={status}  submission.parquet={'OK' if parquet_ok else 'MISSING'}"
    )
    print(f"  status file: {status_path}")
    print(f"  READY FOR OPERATOR-APPROVED SUBMIT: {'YES' if ready else 'NO — investigate above'}")
    if ready:
        print("\n  To submit (operator approval), run:")
        print(
            f"    .venv/bin/python scripts/kaggle/prep_daily_submission.py --submit-only --kver {kver}"
        )
    sys.exit(0 if ready else 1)


def submit_only(kver: str, message: str, force: bool = False) -> None:
    """OPERATOR-APPROVED submit of the current latest kernel version. No re-prep.

    GATED (operator directive 2026-06-20): before submitting, run the LOCAL SUBMISSION GATE
    (scripts/kaggle/arc_local_submission_gate.py) -- it measures the current submitted-default config and
    REFUSES the submit if it is a local regression vs the verified baseline (solve-rate or action
    efficiency). This prevents wasting a 1/day slot on a regression (the value_weight=5 / E3-cascade
    lesson). Bypass only with --force (logged) when you knowingly accept the risk."""
    import subprocess

    gate = Path(__file__).resolve().parent / "arc_local_submission_gate.py"
    if force:
        print("[submit] --force: SKIPPING the local submission gate (operator override).")
    else:
        print("[submit] running the local submission gate (refuses on regression)...")
        rc = subprocess.run([sys.executable, str(gate), "--check"], cwd=str(REPO)).returncode
        if rc != 0:
            sys.exit(
                f"[submit] BLOCKED by the local gate (exit {rc}) -- the current config is a local "
                f"regression. Fix it (or re-run with --force to override). NOT submitting."
            )
        print("[submit] gate PASSED -- proceeding to submit.")

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
    print(
        "[submit] reminder: on a CONFIRMED leaderboard improvement, refresh the gate baseline with "
        "`.venv/bin/python scripts/kaggle/arc_local_submission_gate.py --update-baseline`."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stage + guards only; no Kaggle writes")
    ap.add_argument(
        "--submit-only", action="store_true", help="operator-approved submit of latest kernel"
    )
    ap.add_argument("--kver", default=None, help="kernel version to submit (with --submit-only)")
    ap.add_argument("--message", default=None, help="submission message")
    ap.add_argument(
        "--force",
        action="store_true",
        help="bypass the local submission gate (only when knowingly accepting a regression)",
    )
    a = ap.parse_args()

    if a.dry_run:
        stage_dataset()
        print("[dry-run] stage built + guards passed; no Kaggle writes performed.")
        return
    if a.submit_only:
        if not a.kver:
            sys.exit("--submit-only requires --kver N")
        msg = a.message or f"carnot daily {datetime.now(UTC).strftime('%Y-%m-%d')}"
        submit_only(a.kver, msg, force=a.force)
        return
    prep()


if __name__ == "__main__":
    main()
