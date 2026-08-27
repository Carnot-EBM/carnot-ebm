"""Reusable ARC solve/scoring artifact discipline.

Spec refs: REQ-VERIFY-4437, SCENARIO-VERIFY-4437.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
ARC_LIVE_AGENT_NO_LLM_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE = "offline_arcade_live_agent_runtime_filters_no_new_llm"
ARC_SUPERVISOR_RECEIPT_REPLAY_SUBSTRATE = "live_arc_trajectory_supervisor_receipt_replay_no_llm"
ARC_LIVE_E3_ARCHIVE_PROJECTION_SUBSTRATE = (
    "live_e3_world_model_archived_transition_invariant_projection_no_new_llm"
)
ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE = (
    "canonical_live_e3_environment_outcome_transport_no_new_llm"
)
LIVE_LLM_SUBSTRATE = "live_llm_inference"

SUBSTRATE_DURATION_FLOORS = {
    AGGREGATION_SUBSTRATE: 0.0001,
    VERIFIER_SCORING_SUBSTRATE: 1.0,
    ARC_LIVE_AGENT_NO_LLM_SUBSTRATE: 0.01,
    ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE: 0.01,
    ARC_SUPERVISOR_RECEIPT_REPLAY_SUBSTRATE: 0.01,
    ARC_LIVE_E3_ARCHIVE_PROJECTION_SUBSTRATE: 0.01,
    ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE: 0.01,
    LIVE_LLM_SUBSTRATE: 60.0,
}

# CLAUDE.md's "Verdict Terminal-Prefix Discipline" lists EIGHT accepted prefixes -- both the
# colon and the UNDERSCORE form of each: `complete:` / `complete_` / `success:` / `success_` /
# `passed:` / `passed_` / `shipped:` / `shipped_`. This tuple carried only the four colon forms,
# so it REFUSED verdicts the project's own standard explicitly permits (found 2026-07-27, when it
# blocked exp6011's `complete_four_arm_matrix_measured`).
#
# The underscore form is not merely permitted, it is often the SAFER choice: a literal ": " inside
# a verdict string is what caused the `research-complete.yaml` colon-poison incident (an unquoted
# colon-space broke the YAML and cascaded into a pretest SKIP of the whole milestone). So a lint
# that forces the colon form pushes authors toward the riskier one.
#
# Fixed by widening the tuple rather than by editing the artifact: the artifact was already
# compliant with CLAUDE.md, and rewriting a measurement's verdict to satisfy a stricter-than-spec
# checker would be bending the evidence to fit the tool. The conductor's own
# `_verdict_is_untrustworthy` classifier already accepts the underscore forms, so this also
# removes a disagreement between two gates that are supposed to encode one rule.
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked:",
    "blocked_",
)
SPEC_REFS = ("REQ-VERIFY-4437", "SCENARIO-VERIFY-4437")

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed",
    "inference_substrate": (
        "canonical ARC solve/scoring substrate; offline aggregation uses "
        "aggregation_from_upstream_artifacts, cached scoring uses "
        "verifier_ensemble_against_cached_candidates, and real LLM induction "
        "uses live_llm_inference; no-LLM live ARC environment stepping uses "
        "offline_arcade_live_agent_runtime_self_discovery_no_llm; no-LLM live "
        "ARC filter A/B stepping uses offline_arcade_live_agent_runtime_filters_no_new_llm"
        "; trajectory-supervisor receipt replay uses "
        "live_arc_trajectory_supervisor_receipt_replay_no_llm"
        "; live E3 archived transition projection uses "
        "live_e3_world_model_archived_transition_invariant_projection_no_new_llm"
        "; canonical live E3 outcome transport uses "
        "canonical_live_e3_environment_outcome_transport_no_new_llm"
    ),
    "duration_s": "bare float; must meet the selected substrate floor",
    "template_shipped": "bare bool: the helper + lint + tests landed green",
    "tests_pass": "bare bool: the new unit tests run and assert (Tests-Must-Run-and-Assert)",
}


@dataclass(frozen=True)
class ArtifactDisciplineIssue:
    """One schema/discipline problem found in an ARC solve artifact."""

    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


def duration_floor_s(inference_substrate: Any) -> float | None:
    """Return the required duration floor for a canonical substrate."""

    inference_substrate = _unwrap_principle_value(inference_substrate)
    if not isinstance(inference_substrate, str):
        return None
    return SUBSTRATE_DURATION_FLOORS.get(inference_substrate)


def terminal_prefixed(value: Any) -> bool:
    """Return true when a verdict starts with an accepted terminal prefix."""

    value = _unwrap_principle_value(value)
    return isinstance(value, str) and value.startswith(TERMINAL_VERDICT_PREFIXES)


def validate_arc_solve_artifact(
    artifact: Mapping[str, Any],
    *,
    allow_live: bool = False,
) -> list[ArtifactDisciplineIssue]:
    """Validate the ARC artifact fields required by REQ-VERIFY-4437."""

    issues: list[ArtifactDisciplineIssue] = []
    substrate = _unwrap_principle_value(artifact.get("inference_substrate"))
    if not isinstance(substrate, str) or not substrate:
        issues.append(
            ArtifactDisciplineIssue(
                "MISSING_INFERENCE_SUBSTRATE",
                "ARC solve/scoring artifacts must declare inference_substrate.",
            )
        )
    else:
        floor = duration_floor_s(substrate)
        if floor is None:
            issues.append(
                ArtifactDisciplineIssue(
                    "INVALID_INFERENCE_SUBSTRATE",
                    f"inference_substrate must be one of {sorted(SUBSTRATE_DURATION_FLOORS)}.",
                )
            )
        else:
            duration = _unwrap_principle_value(artifact.get("duration_s"))
            if not _is_finite_number(duration):
                issues.append(
                    ArtifactDisciplineIssue(
                        "DURATION_MISSING",
                        "duration_s must be a finite number for substrate floor checks.",
                    )
                )
            elif float(duration) < floor:
                issues.append(
                    ArtifactDisciplineIssue(
                        "DURATION_BELOW_SUBSTRATE_FLOOR",
                        f"{substrate} requires duration_s >= {floor}.",
                    )
                )
            if substrate == LIVE_LLM_SUBSTRATE and not allow_live:
                issues.append(
                    ArtifactDisciplineIssue(
                        "LIVE_LLM_NOT_ALLOWLISTED",
                        "live_llm_inference artifacts must be explicitly allow-listed.",
                    )
                )

    if not terminal_prefixed(artifact.get("honest_verdict")):
        issues.append(
            ArtifactDisciplineIssue(
                "NON_TERMINAL_HONEST_VERDICT",
                "honest_verdict must start with complete:, success:, passed:, shipped:, or blocked:.",
            )
        )
    return issues


def build_arc_solve_artifact(
    *,
    experiment: str,
    honest_verdict: str,
    inference_substrate: str,
    duration_s: float,
    artifact_kind: str,
    result_path: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    allow_live: bool = False,
) -> dict[str, Any]:
    """Build a validated ARC solve/scoring artifact template."""

    artifact: dict[str, Any] = {
        "experiment": experiment,
        "schema": "carnot.arc_solve_artifact_discipline.v1",
        "artifact_kind": artifact_kind,
        "honest_verdict": honest_verdict,
        "inference_substrate": inference_substrate,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
    }
    if result_path is not None:
        artifact["result_path"] = result_path
    if extra_fields:
        for key, value in extra_fields.items():
            if key == "field_principles" and isinstance(value, Mapping):
                artifact["field_principles"].update(dict(value))
            else:
                artifact[key] = value

    artifact["reproducibility_checksum"] = _sha256(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    issues = validate_arc_solve_artifact(artifact, allow_live=allow_live)
    if issues:
        raise ValueError("; ".join(f"{issue.kind}: {issue.detail}" for issue in issues))
    return artifact


def _is_finite_number(value: Any) -> bool:
    value = _unwrap_principle_value(value)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _unwrap_principle_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "AGGREGATION_SUBSTRATE",
    "ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE",
    "ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE",
    "ARC_LIVE_AGENT_NO_LLM_SUBSTRATE",
    "ARC_SUPERVISOR_RECEIPT_REPLAY_SUBSTRATE",
    "ArtifactDisciplineIssue",
    "FIELD_PRINCIPLES",
    "LIVE_LLM_SUBSTRATE",
    "SPEC_REFS",
    "SUBSTRATE_DURATION_FLOORS",
    "TERMINAL_VERDICT_PREFIXES",
    "VERIFIER_SCORING_SUBSTRATE",
    "build_arc_solve_artifact",
    "duration_floor_s",
    "terminal_prefixed",
    "validate_arc_solve_artifact",
]
