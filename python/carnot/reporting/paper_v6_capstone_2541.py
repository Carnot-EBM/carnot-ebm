"""Exp 2541 capstone: paper-v6 synthesis for milestone 2026.05.244.

This is a thin validator module for the .244 capstone deliverable.

The .244 deliverable is authored as a written synthesis because the
work is largely a multi-artifact judgement call under ambiguity. Five
of the planned eleven capstone inputs (exp2530 archive, exp2531
IsingVerifier implementation, exp2532 Phase 4 ARM-EBM v4, exp2533
ensemble v7b, exp2534 adaptive conformal v2) did not produce
artifacts during this milestone, which means the critical path
through IsingVerifier -> Phase 4 -> arXiv gate-3 did not advance.
The capstone records that reality honestly: headline AUROC carries
forward 0.9750 from .241, phase4_final_status stays
'blocked_precondition', and arxiv_ready stays False.

What this module does: makes the artifact's schema invariants
machine-checkable. The deliverable JSON exists at the expected path,
the required fields are present, the honest_verdict carries a
terminal prefix, the four arXiv gates are individually recorded,
and the capstone's headline determinations (phase4_final_status,
arxiv_ready, operator_recommendation) are internally consistent
with each other and with the principles declared in the task spec.

The deliverable file is the source of truth; this module verifies it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260519"
MILESTONE = "2026.05.244"
EXPERIMENT = "2541_capstone_v244"
SCHEMA = "carnot.paper_v6_capstone_2541.v1"
OUTPUT_FILENAME = "experiment_2541_capstone_v244.json"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME

PRIOR_243_BEST_AUROC = 0.9750
HIVE_EXTERNAL_AUROC = 0.9236

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "best_244_auroc",
        "auroc_adversarially_verified",
        "phase4_final_status",
        "phase4_permanently_retired",
        "arxiv_ready",
        "arxiv_gates",
        "operator_recommendation",
        "external_baselines",
        "kv260_status",
        "gatemate_status",
        "n_experiments_completed",
        "top_3_successes",
        "top_3_gaps_for_245",
        "preconditions_checked",
        "synthesis",
        "field_principles",
        "corrigendum_pending",
    }
)

ALLOWED_PHASE4_STATUSES = frozenset(
    {
        "validated_clean",
        "retired_negative",
        "blocked_precondition",
        "carry_forward",
    }
)

ALLOWED_OPERATOR_RECOMMENDATIONS = frozenset(
    {
        "submit_now",
        "latex_fix_remaining",
        "revise_paper_first",
        "request_operator_decision",
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
    """Read the .244 capstone JSON from disk.

    Raises FileNotFoundError if the deliverable has not been written
    yet. Lets json.JSONDecodeError propagate so corruption is loud
    rather than silently producing a partial dict.
    """

    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate every schema invariant the .244 capstone must satisfy.

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

    if not isinstance(artifact["best_244_auroc"], (int, float)):
        raise ValueError("best_244_auroc must be numeric")
    if not 0.0 <= float(artifact["best_244_auroc"]) <= 1.0:
        raise ValueError("best_244_auroc must lie in [0, 1]")

    phase4_status = artifact["phase4_final_status"]
    if phase4_status not in ALLOWED_PHASE4_STATUSES:
        raise ValueError(
            f"phase4_final_status must be one of {sorted(ALLOWED_PHASE4_STATUSES)}"
        )

    phase4_permanently_retired = artifact["phase4_permanently_retired"]
    if not isinstance(phase4_permanently_retired, bool):
        raise ValueError("phase4_permanently_retired must be a bool")
    if phase4_permanently_retired and phase4_status != "retired_negative":
        raise ValueError(
            "phase4_permanently_retired=True requires "
            "phase4_final_status == 'retired_negative'"
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
    if artifact["arxiv_ready"] and phase4_status not in {
        "validated_clean",
        "retired_negative",
    }:
        raise ValueError(
            "arxiv_ready=True requires phase4_final_status in "
            "{'validated_clean', 'retired_negative'}"
        )

    op_rec = artifact["operator_recommendation"]
    if op_rec not in ALLOWED_OPERATOR_RECOMMENDATIONS:
        raise ValueError(
            f"operator_recommendation must be one of "
            f"{sorted(ALLOWED_OPERATOR_RECOMMENDATIONS)}"
        )
    if phase4_status == "blocked_precondition" and op_rec != (
        "request_operator_decision"
    ):
        raise ValueError(
            "phase4_final_status=blocked_precondition implies "
            "operator_recommendation='request_operator_decision'"
        )
    if phase4_status == "validated_clean" and artifact["arxiv_ready"] and (
        op_rec != "submit_now"
    ):
        raise ValueError(
            "phase4_final_status=validated_clean with arxiv_ready=True "
            "implies operator_recommendation='submit_now'"
        )

    if not isinstance(artifact["top_3_successes"], list):
        raise ValueError("top_3_successes must be a list")
    if len(artifact["top_3_successes"]) != 3:
        raise ValueError("top_3_successes must contain exactly 3 entries")
    if not isinstance(artifact["top_3_gaps_for_245"], list):
        raise ValueError("top_3_gaps_for_245 must be a list")
    if len(artifact["top_3_gaps_for_245"]) != 3:
        raise ValueError("top_3_gaps_for_245 must contain exactly 3 entries")

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
    """Load and validate the .244 capstone deliverable at the given path.

    Returns the validated artifact dict. Used by tests and by ad-hoc CLI
    invocations to confirm the on-disk deliverable matches the schema.
    """

    artifact = load_artifact(path)
    validate_artifact(artifact)
    return artifact


def main() -> int:
    """CLI entrypoint: validate the deliverable and return 0 on success."""

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
