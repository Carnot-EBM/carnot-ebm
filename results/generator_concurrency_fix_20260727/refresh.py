#!/usr/bin/env python3
"""Rebuild every ANALYSER artifact the n_ctx/witness change made stale, and DEEP-DIFF it.

scripts/artifact_freshness_lint.py flagged 6 artifacts as stale because their recorded
`provenance.code` includes `arc_competition_agent.py` and/or `arc_executable_world_model.py`,
both of which this change edits. The lint's own instruction is: rebuild, then diff, and
report exactly which numbers moved -- "a rebuild that silently changes a published figure is
a correction owed, not a formality."

CRITICAL DISTINCTION this script enforces. Five of the six are ANALYSERS over rows already
persisted on disk: re-running them re-derives the same numbers from the same rows, so the
only legitimate movement is `run_date` and the provenance code hashes. The sixth,
`arc_per_level_reset_attribution_20260726.json`, is a CAPTURE -- its "rebuild" command
re-runs 6 games x 3 seeds of the live agent. That is not a rebuild, it is a NEW MEASUREMENT,
and under the changed n_ctx it would legitimately produce different numbers. Overwriting a
dated 2026-07-26 measurement with 2026-07-27 numbers would destroy a historical record
(never-prune) and silently relabel one measurement as another. So it is EXCLUDED here and
reported as a declared, unresolved staleness instead.

Backups are taken before every rebuild and restored if the rebuild fails.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
BAK = Path(__file__).resolve().parent / "freshness_backups"
BAK.mkdir(exist_ok=True)

# Artifacts whose rebuild is a pure analyser pass over PERSISTED rows.
ANALYSER_REBUILDS = {
    "results/arc_gateway_card_ground_truth_20260727.json": None,  # use provenance cmd
    "results/outer_loop_arc_early_stop_grace_sweep_20260726.json": None,
    "results/outer_loop_arc_reset_charge_attribution_20260726.json": None,
    # provenance records no rebuild_command; the registered analyser takes no required args
    "results/outer_loop_arc_gateway_rescore_20260726.json": [
        PY,
        "scripts/analyze_arc_gateway_rescore.py",
    ],
    "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json": [
        PY,
        "scripts/analyze_arc_llm_on_wallclock_envelope.py",
    ],
}

# EXCLUDED: a live re-capture, not a rebuild. See the module docstring.
CAPTURE_EXCLUDED = "results/arc_per_level_reset_attribution_20260726.json"


def flatten(obj, prefix="") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix or "$"] = obj
    return out


def deep_diff(a: dict, b: dict) -> dict:
    fa, fb = flatten(a), flatten(b)
    keys = set(fa) | set(fb)
    changed = {}
    for k in sorted(keys):
        if fa.get(k, "<<absent>>") != fb.get(k, "<<absent>>"):
            changed[k] = [fa.get(k, "<<absent>>"), fb.get(k, "<<absent>>")]
    return changed


BENIGN_SUBSTRINGS = (
    "run_date",
    "built_at",
    "sha256",
    "generated_at",
    "timestamp",
    "duration_s",
    "wall_s",
    "elapsed",
    "measurement_wall_s",
    "reproducibility_checksum",
    "git_commit",
    "analysis_duration",
)


def classify(changed: dict) -> dict:
    benign, substantive = {}, {}
    for k, v in changed.items():
        (benign if any(s in k for s in BENIGN_SUBSTRINGS) else substantive)[k] = v
    return {
        "n_changed": len(changed),
        "n_benign": len(benign),
        "n_substantive": len(substantive),
        "substantive": dict(list(substantive.items())[:40]),
        "benign_keys": sorted(benign)[:40],
    }


def main() -> int:
    report = {"rebuilt": [], "excluded": {}}
    for rel, override in ANALYSER_REBUILDS.items():
        path = REPO / rel
        if not path.exists():
            report["rebuilt"].append({"artifact": rel, "status": "MISSING"})
            continue
        before = json.loads(path.read_text())
        cmd = override
        if cmd is None:
            raw = (before.get("provenance") or {}).get("rebuild_command") or ""
            raw = raw.replace("<this file>", str(path))
            if raw.startswith("python "):  # normalise to the project venv
                raw = PY + raw[len("python") :]
            cmd = raw.split()
        bak = BAK / path.name
        shutil.copy2(path, bak)
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            shutil.copy2(bak, path)  # restore; a failed rebuild must not corrupt the file
            report["rebuilt"].append(
                {
                    "artifact": rel,
                    "status": "REBUILD_FAILED",
                    "returncode": proc.returncode,
                    "cmd": " ".join(cmd)[:200],
                    "stderr_tail": proc.stderr[-600:],
                    "restored_from_backup": True,
                }
            )
            print(f"{rel}: REBUILD FAILED rc={proc.returncode}")
            continue
        after = json.loads(path.read_text())
        cls = classify(deep_diff(before, after))
        report["rebuilt"].append(
            {"artifact": rel, "status": "REBUILT", "cmd": " ".join(cmd)[:200], **cls}
        )
        print(
            f"{rel}: REBUILT changed={cls['n_changed']} "
            f"substantive={cls['n_substantive']} benign={cls['n_benign']}"
        )
        if cls["n_substantive"]:
            for k, v in list(cls["substantive"].items())[:12]:
                print(f"    SUBSTANTIVE {k}: {v[0]!r} -> {v[1]!r}")

    report["excluded"][CAPTURE_EXCLUDED] = {
        "reason": "its provenance rebuild_command is a LIVE RE-CAPTURE "
        "(arc_per_level_reset_attribution_capture.py --games vc33,tu93,sc25,dc22,"
        "r11l,cd82 --seeds 20260724,20260725,20260726 --budget 400), not an "
        "analyser pass over persisted rows.",
        "why_not_rebuilt": [
            "It would be a NEW MEASUREMENT, not a rebuild: the change under test alters the "
            "generator's context pool, so re-captured numbers would legitimately differ.",
            "Overwriting a dated 2026-07-26 measurement with 2026-07-27 numbers destroys a "
            "historical record (never-prune) and silently relabels one measurement as another.",
            "Corrections belong in a NEW artifact citing the original by sha256.",
        ],
        "status": "DECLARED_STALE_NOT_REWRITTEN",
    }
    Path(__file__).with_name("refresh.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {Path(__file__).with_name('refresh.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
