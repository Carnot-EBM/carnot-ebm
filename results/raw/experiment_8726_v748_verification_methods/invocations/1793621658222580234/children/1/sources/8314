"""Experiment 868 — Pre-flight v16: manifest enforcement side-channel deployment.

This experiment deploys ExclusionManifestEnforcer as the side-channel solution
to RETRO-MANIFEST-FULL-SCOPE (eight consecutive milestones of unapplied patch).

The enforcer reads ops/exclusion_manifest.yaml and writes a gate section to
MILESTONE_PREREQS.md without requiring any modification to
scripts/research_conductor.py (CLAUDE.md constraint).

Spec: REQ-INFRA-072, SCENARIO-INFRA-081
"""

import os
import sys

# Allow running from repo root without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# ExperimentTemplate setup
# ---------------------------------------------------------------------------

RESULT_PATH = "results/experiment_868_preflight_v16.json"

tmpl = ExperimentTemplate(
    868,
    "Pre-flight v16: manifest enforcement side-channel deployment",
    RESULT_PATH,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step 1: Deploy ExclusionManifestEnforcer
# ---------------------------------------------------------------------------

MANIFEST_YAML = "ops/exclusion_manifest.yaml"
PREREQS_PATH = "MILESTONE_PREREQS.md"

manifest_enforcer_deployed = False
retired_ids: list[int] = []

try:
    from carnot.pipeline.manifest_enforcer import ExclusionManifestEnforcer

    enforcer = ExclusionManifestEnforcer()
    enforcer.load_manifest(MANIFEST_YAML)
    enforcer.write_prereqs_section(PREREQS_PATH)
    retired_ids = list(enforcer._retired.keys())
    manifest_enforcer_deployed = True
except Exception as exc:
    print(f"[ERROR] ExclusionManifestEnforcer deployment failed: {exc}")

# ---------------------------------------------------------------------------
# Step 2: Verify MILESTONE_PREREQS.md has .67 section
# ---------------------------------------------------------------------------

prereqs_updated = False
try:
    with open(PREREQS_PATH) as f:
        content = f.read()
    prereqs_updated = "2026.04.67" in content and "## Exclusion Manifest Gate" in content
except Exception:
    pass

# ---------------------------------------------------------------------------
# Step 3: Collect metadata from .66 retro
# ---------------------------------------------------------------------------

OPEN_RETROS = [
    "RETRO-MANIFEST-FULL-SCOPE",
    "RETRO-JEPA-OOD",
    "RETRO-SVAMP-ZERO-AUC",
    "RETRO-XILINX-TOOLS-UNAVAILABLE",
    "RETRO-SOTA-MODEL-DOWNLOAD",
    "RETRO-HALLUSAE-AUC-BELOW-THRESHOLD",
    "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
]

RETROS_CLOSED_IN_66 = [
    "RETRO-CONSTRAINT-ZERO-DELTA",
    "RETRO-ISING-INJECTION-NO-DISCRIMINATION",
    "RETRO-ICE40-PNR-LUT-OVERFLOW",
    "RETRO-ICE40-N16-UNEXPECTED-EXPANSION",
    "RETRO-LIVE-ENV-NOT-PROPAGATED",
]

# ---------------------------------------------------------------------------
# Step 4: Determine honest_verdict
# ---------------------------------------------------------------------------

if manifest_enforcer_deployed and prereqs_updated:
    honest_verdict = "governance_ready"
else:
    honest_verdict = "governance_partial"

# ---------------------------------------------------------------------------
# Step 5: Build and write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "manifest_enforcer_deployed": manifest_enforcer_deployed,
        "manifest_yaml_path": MANIFEST_YAML,
        "retired_experiment_ids": sorted(retired_ids),
        "retired_count": len(retired_ids),
        "open_retros_count": len(OPEN_RETROS),
        "open_retros": OPEN_RETROS,
        "retros_closed_in_66": RETROS_CLOSED_IN_66,
        "prereqs_updated": prereqs_updated,
        "prereqs_path": PREREQS_PATH,
        "experiment_count_at_close": 794,
        "wall_time_at_close_minutes": 4107,
        "avg_time_per_experiment_minutes": 5.17,
        "slowest_5_frozen_milestones": 7,
        "schema": "carnot.preflight.v16",
        "honest_verdict": honest_verdict,
    },
    status="success" if honest_verdict == "governance_ready" else "partial",
)

import json
import os as _os

_os.makedirs(_os.path.dirname(_os.path.abspath(RESULT_PATH)), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"[Exp 868] honest_verdict={honest_verdict}")
print(f"[Exp 868] manifest_enforcer_deployed={manifest_enforcer_deployed}")
print(f"[Exp 868] prereqs_updated={prereqs_updated}")
print(f"[Exp 868] retired_ids={sorted(retired_ids)}")

tmpl.assert_deliverable_written()
