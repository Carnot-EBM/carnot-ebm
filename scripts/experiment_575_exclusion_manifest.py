#!/usr/bin/env python3
"""Experiment 575: Conductor Exclusion Manifest.

**Researcher summary:**
    RETRO-056 has been recommended in every retrospective since milestone .37
    (7 consecutive milestones) and has NEVER been built.  The same five experiments
    (308, 260, 309, 425, 410) appear in the slowest-5 for the seventh consecutive
    milestone, consuming approximately 385 minutes/milestone.  Cumulative waste
    since RETRO-056 was first raised: 2,485 minutes = 41.4 hours.

    This experiment builds the conductor exclusion manifest infrastructure:
      - python/carnot/pipeline/exclusion_manifest.py (ExclusionManifest, ExclusionEntry)
      - scripts/conductor_exclusion_manifest.json (the manifest itself)
      - scripts/check_exclusion_manifest.py (CLI check for conductor session start)

    The manifest is NOT yet wired into scripts/research_conductor.py — that
    requires a separate conductor modification.  This experiment produces the
    infrastructure and verifies it works correctly.  The conductor wiring is
    documented in the artifact's instructions_for_conductor field.

Spec: REQ-INFRA-070, REQ-INFRA-071,
      SCENARIO-INFRA-075, SCENARIO-INFRA-076, SCENARIO-INFRA-077
"""

from __future__ import annotations

# apply_env_autofix MUST be called first — sets CARNOT_FORCE_LIVE and related env vars
# before any pipeline code reads them at import time.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.exclusion_manifest import (
    DEFAULT_MANIFEST_PATH,
    ExclusionManifest,
    build_default_manifest,
)
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 575
TITLE = "Conductor Exclusion Manifest"
DELIVERABLE = "results/experiment_575_exclusion_manifest.json"
SCHEMA = "carnot.exclusion_manifest.v1"
MANIFEST_PATH = "scripts/conductor_exclusion_manifest.json"

# Per-milestone cost of the 5 slowest experiments (rough estimate from RETRO-056 data).
ESTIMATED_SAVINGS_PER_MILESTONE = 385
# Seven consecutive milestones (.37 through .43) without the manifest.
CUMULATIVE_WASTED_MINUTES = 7 * ESTIMATED_SAVINGS_PER_MILESTONE  # 2485

# ---------------------------------------------------------------------------
# Watchdog (must be registered before any experiment work)
# ---------------------------------------------------------------------------

ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)

# ---------------------------------------------------------------------------
# Template setup
# ---------------------------------------------------------------------------

tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE, requires_gpu=False)
tmpl.setup()

# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------

manifest_abs_path = _REPO_ROOT / MANIFEST_PATH
manifest = ExclusionManifest(str(manifest_abs_path))

# Load or build the default manifest.
# If the manifest file already exists (written earlier in this session), load it.
# Otherwise build the default five entries and save them.
entries = manifest.load()
if not entries:
    entries = build_default_manifest()
    manifest.save(entries)

# Verify the manifest file now exists on disk.
manifest_file_exists = manifest_abs_path.exists()
assert manifest_file_exists, f"Manifest file not found at {manifest_abs_path}"

# Compute summary statistics.
n_excluded = len(entries)
excluded_ids = [e.experiment_id for e in entries]

# ---------------------------------------------------------------------------
# Build and write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "n_excluded": n_excluded,
        "excluded_ids": excluded_ids,
        "manifest_path": MANIFEST_PATH,
        "exclusion_manifest_built": True,
        "conductor_consulted": False,
        "retro_056_partial": True,
        "estimated_savings_minutes_per_milestone": ESTIMATED_SAVINGS_PER_MILESTONE,
        "cumulative_wasted_minutes_since_retro": CUMULATIVE_WASTED_MINUTES,
        "instructions_for_conductor": (
            "Before each session: python scripts/check_exclusion_manifest.py <experiment_id> "
            "— exits nonzero if experiment_id is excluded. "
            "Wire into research_conductor.py pick_next_task() before spawning any agent."
        ),
        "honest_verdict": "manifest_built_conductor_not_wired",
    },
    status="success",
)
# build_result() overwrites "schema" with a sorted key list; restore the named schema string.
artifact["schema"] = SCHEMA

deliverable_path = _REPO_ROOT / DELIVERABLE
deliverable_path.parent.mkdir(parents=True, exist_ok=True)
deliverable_path.write_text(__import__("json").dumps(artifact, indent=2))

print(f"[Exp {EXP_ID}] Deliverable written: {DELIVERABLE}")
print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
print(f"[Exp {EXP_ID}] n_excluded={n_excluded}")
print(f"[Exp {EXP_ID}] excluded_ids={excluded_ids}")

# assert_deliverable_written MUST be the final line — it raises if the file is missing.
tmpl.assert_deliverable_written()
