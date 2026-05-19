"""Exp 2528 capstone: paper-v6 synthesis for milestone 2026.05.243.

This is a thin validator module for the .243 capstone deliverable.

Unlike the .242 capstone (which procedurally derives every field from
source artifacts), the .243 deliverable is authored as a written
synthesis because the work is largely a multi-artifact judgement call
under ambiguity — operator-decision framing for the Phase 4
blocked-precondition state, the regression interpretation of
ensemble v7 (Tier 0r dragged Group C mean down rather than lifting
the headline), and the honest-call on arxiv_ready under
exp2527's stale carry-forward gate readout.

The module here exists to make the artifact's schema invariants
machine-checkable: the deliverable JSON exists at the expected path,
the required schema fields are present, the honest_verdict carries
a terminal-prefix, the 4 arXiv gates are individually recorded, and
the capstone's headline determinations (phase4_final_status,
arxiv_ready, operator_recommendation) are internally consistent.

The deliverable file is the source of truth; this module verifies it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260519"
MILESTONE = "2026.05.243"
EXPERIMENT = "2528_capstone_v243"
SCHEMA = "carnot.paper_v6_capstone_2528.v1"
OUTPUT_FILENAME = "experiment_2528_capstone_v243.json"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME

PRIOR_242_BEST_AUROC = 0.9750
HIVE_EXTERNAL_AUROC = 0.9236

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "best_243_auroc",
        "auroc_adversarially_verified",
        "phase4_final_status",
        "phase4_validated_any",
        "arxiv_ready",
        "arxiv_gates",
        "operator_recommendation",
        "external_baselines",
        "kv260_status",
        "n_experiments_completed",
        "top_3_successes",
        "top_3_gaps_for_244",
        "preconditions_checked",
        "synthesis",
        "field_principles",
        "corrigendum_pending",
    }
)

ALLOWED_PHASE4_STATUSES = frozenset(
    {
        "validated_clean",
        "methodology_fallback_retired",
        "blocked_precondition",
        "carry_forward_caveat",
    }
)

ALLOWED_OPERATOR_RECOMMENDATIONS = frozenset(
    {
        "submit_now",
        "revise_paper_then_submit",
        "request_phase4_operator_decision",
    }
)

ARXIV_GATE_KEYS = frozenset(
    {
        "gate_1_phase1_ship",
        "gate_2_audit",
        "gate_3_phase4_validated_any",
        "gate_4_auroc_adversarially_verified",
    }
)

TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def load_artifact(path: str | Path = DEFAULT_OUT_PATH) -> Mapping[str, Any]:
    """Read the .243 capstone JSON from disk.

    Raises FileNotFoundError if the deliverable has not been written
    yet. Lets json.JSONDecodeError propagate so corruption is loud
    rather than silently producing a partial dict.
    """

    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate every schema invariant the .243 capstone must satisfy.

    Raises ValueError on any violation with a specific message so the
    caller crashes loudly rather than emitting a malformed deliverable.
    The intent is to catch authoring drift between this module's
    schema definition and the hand-authored JSON file.
    """

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    if artifact.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    if artifact.get("experiment") != EXPERIMENT:
        raise ValueError(f"experiment must be {EXPERIMENT!r}")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError(f"milestone must be {MILESTONE!r}")
    if artifact.get("status") != "complete":
        raise ValueError("status must be 'complete'")

    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError(
            "honest_verdict must start with a terminal prefix "
            "(complete:/success:/passed:/shipped:)"
        )

    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, (int, float)) or duration_s < 0:
        raise ValueError("duration_s must be a non-negative number")

    if not isinstance(artifact["best_243_auroc"], (int, float)):
        raise ValueError("best_243_auroc must be numeric")
    if not 0.0 <= float(artifact["best_243_auroc"]) <= 1.0:
        raise ValueError("best_243_auroc must lie in [0, 1]")

    phase4_status = artifact["phase4_final_status"]
    if phase4_status not in ALLOWED_PHASE4_STATUSES:
        raise ValueError(
            f"phase4_final_status must be one of {sorted(ALLOWED_PHASE4_STATUSES)}"
        )

    phase4_validated_any = artifact["phase4_validated_any"]
    if not isinstance(phase4_validated_any, bool):
        raise ValueError("phase4_validated_any must be a bool")
    if phase4_validated_any and phase4_status != "validated_clean":
        raise ValueError(
            "phase4_validated_any may only be True when phase4_final_status == "
            "'validated_clean'"
        )

    if not isinstance(artifact["arxiv_ready"], bool):
        raise ValueError("arxiv_ready must be a bool")
    gates = artifact["arxiv_gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("arxiv_gates must be a mapping")
    if set(gates) != ARXIV_GATE_KEYS:
        raise ValueError(f"arxiv_gates must contain exactly {sorted(ARXIV_GATE_KEYS)}")
    if artifact["arxiv_ready"] and not all(bool(v) for v in gates.values()):
        raise ValueError("arxiv_ready=True requires all 4 gates True")
    if artifact["arxiv_ready"] and not phase4_validated_any:
        raise ValueError(
            "arxiv_ready=True requires phase4_validated_any=True per task spec"
        )

    op_rec = artifact["operator_recommendation"]
    if op_rec not in ALLOWED_OPERATOR_RECOMMENDATIONS:
        raise ValueError(
            f"operator_recommendation must be one of "
            f"{sorted(ALLOWED_OPERATOR_RECOMMENDATIONS)}"
        )
    if phase4_status == "validated_clean" and op_rec != "submit_now":
        raise ValueError(
            "phase4_final_status=validated_clean implies operator_recommendation="
            "'submit_now'"
        )
    if phase4_status == "blocked_precondition" and op_rec != (
        "request_phase4_operator_decision"
    ):
        raise ValueError(
            "phase4_final_status=blocked_precondition implies "
            "operator_recommendation='request_phase4_operator_decision'"
        )

    if not isinstance(artifact["top_3_successes"], list):
        raise ValueError("top_3_successes must be a list")
    if len(artifact["top_3_successes"]) != 3:
        raise ValueError("top_3_successes must contain exactly 3 entries")
    if not isinstance(artifact["top_3_gaps_for_244"], list):
        raise ValueError("top_3_gaps_for_244 must be a list")
    if len(artifact["top_3_gaps_for_244"]) != 3:
        raise ValueError("top_3_gaps_for_244 must contain exactly 3 entries")

    baselines = artifact["external_baselines"]
    if not isinstance(baselines, Mapping):
        raise ValueError("external_baselines must be a mapping")
    if baselines.get("hive_external_auroc") != HIVE_EXTERNAL_AUROC:
        raise ValueError(
            f"external_baselines.hive_external_auroc must equal "
            f"{HIVE_EXTERNAL_AUROC} (carried forward)"
        )

    if not isinstance(artifact["corrigendum_pending"], list):
        raise ValueError("corrigendum_pending must be a list")
    for item in artifact["corrigendum_pending"]:
        if not isinstance(item, Mapping):
            raise ValueError("each corrigendum_pending entry must be a mapping")
        if "kind" not in item or "severity" not in item or "detail" not in item:
            raise ValueError(
                "each corrigendum_pending entry must have kind/severity/detail"
            )


def run(path: str | Path = DEFAULT_OUT_PATH) -> Mapping[str, Any]:
    """Load and validate the .243 capstone deliverable at the given path.

    Returns the validated artifact dict. Used by tests and by ad-hoc CLI
    invocations to confirm the on-disk deliverable matches the schema.
    """

    artifact = load_artifact(path)
    validate_artifact(artifact)
    return artifact


def main() -> int:
    """CLI entrypoint — validate the deliverable, return 0 on success."""

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
