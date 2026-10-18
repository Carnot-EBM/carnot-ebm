"""Latent-vs-deterministic discipline gate policy helpers.

This module is deliberately small: Exp 1500 is an ops policy artifact, not a
new verifier stack.  The helper keeps the policy tables and terminal artifact
schema deterministic so future changes can be tested without turning latent
signals into another automatic decision surface.

Spec: REQ-VERIFY-1500, SCENARIO-VERIFY-1500.
"""

from __future__ import annotations

from typing import Any

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "discipline_gate_ready",
    "gated_inputs_present",
    "signal_classes_audited",
    "headline_allowed_signals",
    "auxiliary_allowed_signals",
    "retired_signals",
    "deterministic_first_rules",
    "superficial_baseline_required_rules",
    "ops_note_path",
    "blockers",
    "honest_verdict",
)

ALLOWED_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

SIGNAL_CLASSES_AUDITED: tuple[str, ...] = (
    "deterministic_executable_validators",
    "deterministic_bounds",
    "energy_ranking",
    "latent_energy_like_signals",
    "probabilistic_or_llm_derived_signals",
    "memory_policy_signals",
    "schema_record_signals",
    "retired_semantic_energy_telemetry",
    "retired_v1_pairwise_self_verification",
)

HEADLINE_ALLOWED_SIGNALS: tuple[str, ...] = (
    "deterministic_executable_validators",
    "conservative_deterministic_bounds",
)

AUXILIARY_ALLOWED_SIGNALS: tuple[str, ...] = (
    "carnot_energy_ranking_after_validator_comparison",
    "partial_trace_energy_localization_for_repair",
    "query_time_memory_policy_zero_soundness_gated",
    "calibrated_probabilistic_verifiers_after_all_checks",
)

TRIAGE_ALLOWED_SIGNALS: tuple[str, ...] = (
    "llm_or_latent_uncertainty_for_manual_review_priority",
    "uncalibrated_energy_like_scores_for_debugging_only",
    "structured_verdict_records_for_auditability_only",
)

DETERMINISTIC_FIRST_RULES: tuple[str, ...] = (
    "Run applicable deterministic validators before latent, energy-like, probabilistic, or LLM-derived signals.",
    "When deterministic validators address the same claim, their reject decision dominates any latent accept or rank.",
    "Use energy ranking only after deterministic validity is known, or for repair localization rather than answer acceptance.",
    "Do not let memory hits, schema records, or LLM judge votes bypass deterministic validator failures.",
)

SUPERFICIAL_BASELINE_REQUIRED_RULES: tuple[str, ...] = (
    "Any latent, energy-like, probabilistic, or LLM-derived signal must beat matched superficial baselines before supporting a claim.",
    "Matched superficial baselines must include response length, lexical overlap, format validity, or another task-specific cheap confound when applicable.",
    "Held-out calibration must be measured before a probabilistic or latent score can leave triage status.",
    "False accepts must be counted on the same decision surface before a latent signal can influence ranking or routing.",
)


def build_discipline_gate_artifact(
    *,
    exp1499: dict[str, Any],
    exp1481: dict[str, Any],
    exp1487: dict[str, Any],
    ops_note_path: str,
) -> dict[str, Any]:
    """Build the Exp 1500 terminal artifact from checked-in upstream evidence."""

    blockers = _gate_blockers(exp1499=exp1499, exp1481=exp1481, exp1487=exp1487)
    ready = not blockers
    status = "complete" if ready else "blocked"
    retired = _retired_signals(exp1481=exp1481, exp1487=exp1487)
    verdict = (
        "complete: latent deterministic discipline gate ready"
        if ready
        else "complete: latent deterministic discipline gate blocked on required inputs"
    )
    artifact = {
        "status": status,
        "discipline_gate_ready": ready,
        "gated_inputs_present": bool(exp1499) and bool(exp1481) and bool(exp1487),
        "signal_classes_audited": list(SIGNAL_CLASSES_AUDITED),
        "headline_allowed_signals": list(HEADLINE_ALLOWED_SIGNALS),
        "auxiliary_allowed_signals": list(AUXILIARY_ALLOWED_SIGNALS),
        "triage_allowed_signals": list(TRIAGE_ALLOWED_SIGNALS),
        "retired_signals": retired,
        "deterministic_first_rules": list(DETERMINISTIC_FIRST_RULES),
        "superficial_baseline_required_rules": list(SUPERFICIAL_BASELINE_REQUIRED_RULES),
        "ops_note_path": ops_note_path,
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    _validate_required_fields(artifact)
    return artifact


def render_policy_markdown(artifact: dict[str, Any]) -> str:
    """Render the ops note tables that explain how the artifact gates claims."""

    lines = [
        "# Latent vs Deterministic Discipline Gate 1500",
        "",
        "Spec: REQ-VERIFY-1500, SCENARIO-VERIFY-1500.",
        "",
        "Run date: 20260507.",
        "",
        "## Headline Evidence",
        "",
        "| Signal | Allowed use | Gate |",
        "|---|---|---|",
    ]
    for signal in artifact["headline_allowed_signals"]:
        lines.append(
            f"| `{signal}` | Headline acceptance or rejection evidence | Must be deterministic and directly applicable to the claim |"
        )
    lines.extend(
        [
            "",
            "## Auxiliary Ranking Evidence",
            "",
            "| Signal | Allowed use | Gate |",
            "|---|---|---|",
        ]
    )
    for signal in artifact["auxiliary_allowed_signals"]:
        lines.append(
            f"| `{signal}` | Ranking, repair localization, or opt-in routing | Requires deterministic validator comparison and false-accept accounting |"
        )
    lines.extend(
        [
            "",
            "## Triage Evidence",
            "",
            "| Signal | Allowed use | Gate |",
            "|---|---|---|",
        ]
    )
    for signal in artifact["triage_allowed_signals"]:
        lines.append(
            f"| `{signal}` | Queueing, debugging, or manual-review priority only | Cannot accept, reject, or headline a claim |"
        )
    lines.extend(
        [
            "",
            "## Retired / No-Claim Evidence",
            "",
            "| Signal | Status | Reason |",
            "|---|---|---|",
        ]
    )
    for signal in artifact["retired_signals"]:
        lines.append(
            f"| `{signal}` | Retired from active claims | Confounded, dominated, or missing the required acceptance checks |"
        )
    lines.extend(
        [
            "",
            "## Required Checks Before Latent Influence",
            "",
            "- deterministic validator comparison",
            "- superficial-baseline comparison",
            "- held-out calibration",
            "- false-accept accounting",
            "",
            "## Deterministic-First Rules",
            "",
        ]
    )
    lines.extend(f"- {rule}" for rule in artifact["deterministic_first_rules"])
    lines.extend(["", "## Superficial-Baseline Rules", ""])
    lines.extend(f"- {rule}" for rule in artifact["superficial_baseline_required_rules"])
    return "\n".join(lines) + "\n"


def _gate_blockers(
    *,
    exp1499: dict[str, Any],
    exp1481: dict[str, Any],
    exp1487: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not _has_orthogonality_matrix(exp1499):
        blockers.append("missing_exp1499_orthogonality_matrix")
    if not exp1481:
        blockers.append("missing_exp1481_semantic_energy_retirement")
    if not exp1487:
        blockers.append("missing_exp1487_v1_pairwise_retirement")
    return blockers


def _has_orthogonality_matrix(exp1499: dict[str, Any]) -> bool:
    return bool(
        exp1499.get("orthogonality_matrix_written")
        and (exp1499.get("conditional_acceptance_matrix") or {}).get("labels")
    )


def _retired_signals(*, exp1481: dict[str, Any], exp1487: dict[str, Any]) -> list[str]:
    retired = ["uncalibrated_latent_or_llm_scores_without_required_checks"]
    if exp1481.get("claim_allowed") is False:
        retired.extend(
            [
                "semantic_energy_headline_telemetry",
                "semantic_energy_logit_telemetry_headline",
            ]
        )
    if exp1487.get("improvement_allowed") is False:
        retired.append("v1_pairwise_self_verification_active_gate")
    return retired


def _validate_required_fields(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(ALLOWED_VERDICT_PREFIXES):
        raise ValueError("honest_verdict has a disallowed prefix")
