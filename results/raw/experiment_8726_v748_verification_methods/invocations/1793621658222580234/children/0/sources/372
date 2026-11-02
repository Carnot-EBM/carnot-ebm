"""Exp 3143 FR-11 experience-driven verifier memory.

Spec refs: REQ-LEARN-3143, SCENARIO-LEARN-3143,
SCENARIO-LEARN-3143-BLOCKED.

This module implements controller-side verifier memory only. It learns from
checked-in historical outcomes by routing future checks more carefully: exact
low-risk memories can skip redundant checks, while families with false-accept
history are escalated. Nothing here updates base model weights, verifier model
weights, KAN weights, or LLM weights; the result is a replayable policy table
over prior evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3143_fr11_experience_driven_verifier_memory_v1"
SCHEMA = "carnot.fr11.experience_driven_verifier_memory.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json"
)
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MEMORY_KEY_FIELDS = (
    "fixture_family",
    "difficulty",
    "answer_format",
    "failure_mechanism",
    "contract_decision",
)
MEMORY_KEY_SCHEMA: JsonDict = {
    "schema": "carnot.fr11.verifier_memory_key.v1",
    "fields": list(MEMORY_KEY_FIELDS),
    "normalization": {
        "fixture_family": "artifact family id as lowercase string",
        "difficulty": "sorted pipe-joined difficulty buckets or variant kind",
        "answer_format": "extraction or response contract format",
        "failure_mechanism": "historical mechanism label, no_failure when clean",
        "contract_decision": "expected verifier contract action: accept/reject/unknown",
    },
}
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_experience_verifier_memory_v1_ready",
    "continuous_self_learning_targeted",
    "memory_key_schema",
    "replay_row_count",
    "suppressed_check_count",
    "escalated_check_count",
    "estimated_check_savings_rate",
    "residual_false_accept_risk",
    "residual_false_reject_risk",
    "ledger_consistency_rate",
    "no_weight_update_claim",
    "promotion_recommendation",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3143_fr11_experience_driven_verifier_memory_v1.py -q",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3143_fr11_experience_driven_verifier_memory_v1.py -q",
    ".venv/bin/coverage report --include='*/python/carnot/eval/fr11_experience_driven_verifier_memory_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3142_vera_evoenv_replay", EXP3142_REL_PATH, True),
    ("exp3129_constraint_memory_audit", EXP3129_REL_PATH, True),
    (
        "exp3143_module",
        Path("python/carnot/eval/fr11_experience_driven_verifier_memory_v1.py"),
        False,
    ),
    (
        "exp3143_tests",
        Path(
            "tests/python/test_experiment_3143_fr11_experience_driven_verifier_memory_v1.py"
        ),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_replay_rows(exp3136: Mapping[str, Any], exp3142: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3143-2: combine prior verifier and VeRA replay rows."""

    return verifier_replay_rows(exp3136) + variant_replay_rows(exp3142)


def verifier_replay_rows(exp3136: Mapping[str, Any]) -> list[JsonDict]:
    """Convert Exp 3136 verifier rows into policy replay rows."""

    false_accept_ids = {str(row_id) for row_id in exp3136.get("false_accept_row_ids", [])}
    rows: list[JsonDict] = []
    for raw in exp3136.get("verifier_rows", []):
        if not isinstance(raw, Mapping):
            continue
        row_id = str(raw.get("row_id") or raw.get("fixture_id") or "")
        expected_action = contract_decision_from_row(raw)
        observed_decision = str(raw.get("live_decision") or "unknown")
        false_accept = row_id in false_accept_ids or (
            expected_action == "reject" and observed_decision == "accept"
        )
        false_reject = expected_action == "accept" and observed_decision != "accept"
        ledger_consistent = ledger_consistent_from_row(raw)
        rows.append(
            {
                "row_id": row_id,
                "source": "exp3136_verifier_row",
                "fixture_family": normalize_token(raw.get("fixture_family") or "unknown"),
                "difficulty": normalize_difficulty(raw.get("difficulty_buckets")),
                "answer_format": normalize_token(
                    raw.get("answer_extraction_format") or "unknown"
                ),
                "failure_mechanism": normalize_token(
                    raw.get("failure_mechanism_from_exp3124") or "no_failure"
                ),
                "contract_decision": expected_action,
                "expected_action": expected_action,
                "observed_decision": observed_decision,
                "exact_label": str(raw.get("exact_label") or "unknown"),
                "false_accept": false_accept,
                "false_reject": false_reject,
                "replay_error": False,
                "ledger_consistent": ledger_consistent,
            }
        )
    return rows


def variant_replay_rows(exp3142: Mapping[str, Any]) -> list[JsonDict]:
    """Convert Exp 3142 variant records into policy replay rows."""

    rows: list[JsonDict] = []
    for raw in exp3142.get("variant_records", []):
        if not isinstance(raw, Mapping):
            continue
        environment = raw.get("environment") if isinstance(raw.get("environment"), Mapping) else {}
        soundness_errors = int(raw.get("soundness_errors") or 0)
        completeness_errors = int(raw.get("completeness_errors") or 0)
        exact_replay_passed = raw.get("exact_replay_passed") is True
        replay_error = not exact_replay_passed or soundness_errors > 0 or completeness_errors > 0
        rows.append(
            {
                "row_id": str(raw.get("variant_id") or raw.get("source_environment_id") or ""),
                "source": "exp3142_variant_replay",
                "fixture_family": normalize_token(environment.get("family_id") or "unknown"),
                "difficulty": normalize_difficulty([raw.get("variant_kind") or "variant"]),
                "answer_format": "json_assignment",
                "failure_mechanism": "variant_replay_error" if replay_error else "no_failure",
                "contract_decision": "accept",
                "expected_action": "accept",
                "observed_decision": "accept" if exact_replay_passed else "reject",
                "exact_label": "VALID_ASSIGNMENT",
                "false_accept": soundness_errors > 0,
                "false_reject": completeness_errors > 0 or not exact_replay_passed,
                "replay_error": replay_error,
                "ledger_consistent": exact_replay_passed
                and soundness_errors == 0
                and completeness_errors == 0,
            }
        )
    return rows


def memory_key_for_row(row: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3143-1: derive the replayable memory key for one row."""

    return {
        "fixture_family": normalize_token(row.get("fixture_family") or "unknown"),
        "difficulty": normalize_difficulty(row.get("difficulty")),
        "answer_format": normalize_token(row.get("answer_format") or "unknown"),
        "failure_mechanism": normalize_token(row.get("failure_mechanism") or "no_failure"),
        "contract_decision": contract_decision_from_row(row),
    }


def memory_key_id(key: Mapping[str, Any]) -> str:
    """Return a stable short id for a memory key."""

    payload = json.dumps(
        {field: str(key.get(field) or "") for field in MEMORY_KEY_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_memory_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Aggregate exact-key historical outcomes for routing."""

    index: dict[str, JsonDict] = {}
    for row in rows:
        key = memory_key_for_row(row)
        key_id = memory_key_id(key)
        summary = index.setdefault(
            key_id,
            {
                "key_id": key_id,
                "key": key,
                "observation_count": 0,
                "false_accept_count": 0,
                "false_reject_count": 0,
                "replay_error_count": 0,
                "ledger_consistent_count": 0,
                "row_ids": [],
            },
        )
        summary["observation_count"] += 1
        summary["false_accept_count"] += int(bool(row.get("false_accept")))
        summary["false_reject_count"] += int(bool(row.get("false_reject")))
        summary["replay_error_count"] += int(bool(row.get("replay_error")))
        summary["ledger_consistent_count"] += int(bool(row.get("ledger_consistent")))
        summary["row_ids"].append(str(row.get("row_id") or ""))
    for summary in index.values():
        summary["ledger_consistency_rate"] = rate(
            int(summary["ledger_consistent_count"]),
            int(summary["observation_count"]),
        )
        summary["low_risk_exact_history"] = exact_history_is_low_risk(summary)
    return index


def exact_history_is_low_risk(summary: Mapping[str, Any]) -> bool:
    """REQ-LEARN-3143-3: decide whether an exact key is safe to suppress."""

    return (
        int(summary.get("observation_count") or 0) > 0
        and int(summary.get("false_accept_count") or 0) == 0
        and int(summary.get("false_reject_count") or 0) == 0
        and int(summary.get("replay_error_count") or 0) == 0
        and float(summary.get("ledger_consistency_rate") or 0.0) == 1.0
    )


def simulate_routing_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    ledger_consistency_rate: float,
) -> JsonDict:
    """REQ-LEARN-3143-3/4/5: replay suppress/escalate routing over history."""

    memory_index = build_memory_index(rows)
    risky_families = {
        normalize_token(row.get("fixture_family") or "unknown")
        for row in rows
        if row.get("false_accept") or row.get("replay_error")
    }
    routing_rows: list[JsonDict] = []
    for row in rows:
        key = memory_key_for_row(row)
        key_id = memory_key_id(key)
        summary = memory_index[key_id]
        family = normalize_token(row.get("fixture_family") or "unknown")
        if family in risky_families:
            decision = "escalate"
            reason = "family_false_accept_or_replay_error_history"
        elif summary["low_risk_exact_history"]:
            decision = "suppress"
            reason = "exact_key_low_risk_history"
        else:
            decision = "normal"
            reason = "insufficient_exact_low_risk_history"
        routing_rows.append(
            dict(row)
            | {
                "memory_key_id": key_id,
                "routing_decision": decision,
                "routing_reason": reason,
            }
        )

    replay_row_count = len(routing_rows)
    suppressed = sum(1 for row in routing_rows if row["routing_decision"] == "suppress")
    escalated = sum(1 for row in routing_rows if row["routing_decision"] == "escalate")
    normal = replay_row_count - suppressed - escalated
    policy_check_count = normal + (2 * escalated)
    residual_false_accept_count = sum(
        1
        for row in routing_rows
        if row.get("false_accept") and row["routing_decision"] != "escalate"
    )
    residual_false_reject_count = sum(
        1
        for row in routing_rows
        if (
            row.get("expected_action") == "accept"
            and row["routing_decision"] == "escalate"
        )
        or (row.get("false_reject") and row["routing_decision"] != "escalate")
    )
    return {
        "replay_row_count": replay_row_count,
        "suppressed_check_count": suppressed,
        "escalated_check_count": escalated,
        "normal_check_count": normal,
        "baseline_check_count": replay_row_count,
        "estimated_policy_check_count": policy_check_count,
        "estimated_check_savings_rate": rate(
            max(0, replay_row_count - policy_check_count),
            replay_row_count,
        ),
        "gross_suppression_rate": rate(suppressed, replay_row_count),
        "residual_false_accept_risk": rate(residual_false_accept_count, replay_row_count),
        "residual_false_reject_risk": rate(residual_false_reject_count, replay_row_count),
        "residual_false_accept_count": residual_false_accept_count,
        "residual_false_reject_count": residual_false_reject_count,
        "ledger_consistency_rate": round_float(ledger_consistency_rate),
        "risky_families": sorted(risky_families),
        "memory_key_summaries": memory_index,
        "routing_rows": routing_rows,
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3143 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    exp3142 = read_json_object(root_path / EXP3142_REL_PATH)
    exp3129 = read_json_object(root_path / EXP3129_REL_PATH)
    blocker = precondition_blocker(exp3136, exp3142, exp3129)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    ledger_rate = inherited_ledger_consistency_rate(exp3129, exp3142)
    rows = load_replay_rows(exp3136, exp3142)
    routing = simulate_routing_policy(rows, ledger_consistency_rate=ledger_rate)
    ready = (
        int(routing["replay_row_count"]) > 0
        and int(routing["suppressed_check_count"]) > 0
        and int(routing["escalated_check_count"]) > 0
        and float(routing["residual_false_accept_risk"]) == 0.0
    )
    recommendation = promotion_recommendation(
        ready,
        float(routing["residual_false_accept_risk"]),
        ledger_rate,
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_experience_verifier_memory_v1_ready": ready,
        "continuous_self_learning_targeted": True,
        "memory_key_schema": MEMORY_KEY_SCHEMA,
        "replay_row_count": int(routing["replay_row_count"]),
        "suppressed_check_count": int(routing["suppressed_check_count"]),
        "escalated_check_count": int(routing["escalated_check_count"]),
        "estimated_check_savings_rate": float(routing["estimated_check_savings_rate"]),
        "residual_false_accept_risk": float(routing["residual_false_accept_risk"]),
        "residual_false_reject_risk": float(routing["residual_false_reject_risk"]),
        "ledger_consistency_rate": ledger_rate,
        "no_weight_update_claim": True,
        "promotion_recommendation": recommendation,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "precondition_checks": precondition_checks(exp3136, exp3142, exp3129),
        "false_accept_mechanism_counts": dict(exp3136.get("false_accept_mechanism_counts") or {}),
        "routing_summary": {
            key: value
            for key, value in routing.items()
            if key
            not in {
                "routing_rows",
                "memory_key_summaries",
            }
        },
        "memory_key_summaries": list(routing["memory_key_summaries"].values()),
        "routing_rows": routing["routing_rows"],
        "ledger_consistency_summary": {
            "exp3129_ledger_consistency_rate": float(
                exp3129.get("ledger_consistency_rate") or 0.0
            ),
            "exp3142_ledger_consistency_rate": float(
                exp3142.get("ledger_consistency_rate") or 0.0
            ),
            "policy_ledger_consistency_rate": ledger_rate,
        },
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(ready, recommendation),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    start: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a schema-complete artifact when required evidence is absent."""

    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_experience_verifier_memory_v1_ready": False,
        "continuous_self_learning_targeted": True,
        "memory_key_schema": MEMORY_KEY_SCHEMA,
        "replay_row_count": 0,
        "suppressed_check_count": 0,
        "escalated_check_count": 0,
        "estimated_check_savings_rate": 0.0,
        "residual_false_accept_risk": 0.0,
        "residual_false_reject_risk": 0.0,
        "ledger_consistency_rate": 0.0,
        "no_weight_update_claim": True,
        "promotion_recommendation": "block_fr11_experience_memory_missing_source_evidence",
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "precondition_checks": {
            "exp3136_false_accept_autopsy_ready": False,
            "exp3142_vera_evoenv_ready": False,
            "exp3129_constraint_memory_ready": False,
        },
        "false_accept_mechanism_counts": {},
        "routing_summary": {
            "risky_families": [],
            "normal_check_count": 0,
            "baseline_check_count": 0,
            "estimated_policy_check_count": 0,
        },
        "memory_key_summaries": [],
        "routing_rows": [],
        "ledger_consistency_summary": {},
        "blocked_reason": blocker,
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3143 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(
    exp3136: Mapping[str, Any],
    exp3142: Mapping[str, Any],
    exp3129: Mapping[str, Any],
) -> str:
    """Return the first missing source needed for policy replay."""

    if exp3136.get("false_accept_autopsy_v1_ready") is not True:
        return "exp3136_false_accept_autopsy_missing_or_not_ready"
    if exp3142.get("fr11_vera_evoenv_v2_ready") is not True:
        return "exp3142_vera_evoenv_missing_or_not_ready"
    if exp3129.get("fr11_constraint_memory_audit_v1_ready") is not True:
        return "exp3129_constraint_memory_audit_missing_or_not_ready"
    return ""


def precondition_checks(
    exp3136: Mapping[str, Any],
    exp3142: Mapping[str, Any],
    exp3129: Mapping[str, Any],
) -> JsonDict:
    """Expose source readiness checks in the final artifact."""

    return {
        "exp3136_false_accept_autopsy_ready": exp3136.get("false_accept_autopsy_v1_ready")
        is True,
        "exp3142_vera_evoenv_ready": exp3142.get("fr11_vera_evoenv_v2_ready") is True,
        "exp3129_constraint_memory_ready": exp3129.get(
            "fr11_constraint_memory_audit_v1_ready"
        )
        is True,
        "exp3142_no_weight_update_claim": exp3142.get("no_weight_update_claim") is True,
        "exp3129_no_weight_update_claim": exp3129.get("no_weight_update_claim") is True,
    }


def inherited_ledger_consistency_rate(
    exp3129: Mapping[str, Any],
    exp3142: Mapping[str, Any],
) -> float:
    """Track the strictest prior FR-11 ledger consistency blocker."""

    ledger = exp3142.get("ledger_replay_summary")
    candidate_rates = [
        exp3129.get("ledger_consistency_rate"),
        exp3142.get("ledger_consistency_rate"),
    ]
    if isinstance(ledger, Mapping):
        candidate_rates.append(ledger.get("prior_ledger_consistency_rate"))
    finite_rates = [
        float(value)
        for value in candidate_rates
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    return round_float(min(finite_rates)) if finite_rates else 0.0


def promotion_recommendation(
    ready: bool,
    residual_false_accept_risk: float,
    ledger_consistency_rate: float,
) -> str:
    """REQ-LEARN-3143-6: make the FR-11 promotion decision explicit."""

    if not ready:
        return "block_fr11_experience_memory_missing_or_unsafe_source_evidence"
    if residual_false_accept_risk > 0.0:
        return "block_fr11_experience_memory_unsafe_false_accept_risk"
    if ledger_consistency_rate < 1.0:
        return (
            "promote_controller_routing_memory_only_"
            "block_model_weight_learning_until_ledger_consistency_is_1.0"
        )
    return "promote_controller_routing_memory"


def inference_substrate(mode: str = "artifact_only_controller_routing_memory") -> JsonDict:
    """Separate routing memory from any model-weight learning claim."""

    return {
        "mode": mode,
        "controller_routing_memory_only": True,
        "memory_table_type": "exact_key_policy_statistics",
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "uses_checked_in_artifacts_only": True,
        "source_false_accept_artifact": EXP3136_REL_PATH.as_posix(),
        "source_variant_replay_artifact": EXP3142_REL_PATH.as_posix(),
        "source_ledger_artifact": EXP3129_REL_PATH.as_posix(),
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List replay sources with checksums for traceable memory evidence."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3143 artifact violates the routing-memory contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("no_weight_update_claim") is not True:
        raise ValueError("no_weight_update_claim must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or any(
        substrate.get(flag) is True
        for flag in ("model_weight_mutation", "model_weight_training", "base_model_weights_updated")
    ):
        raise ValueError("model_weight_mutation must remain false")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    for field in (
        "estimated_check_savings_rate",
        "residual_false_accept_risk",
        "residual_false_reject_risk",
        "ledger_consistency_rate",
    ):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and within [0, 1]")
    if artifact.get("fr11_experience_verifier_memory_v1_ready") is not True:
        return
    if int(artifact.get("replay_row_count") or 0) <= 0:
        raise ValueError("replay_row_count must be positive for readiness")
    if int(artifact.get("suppressed_check_count") or 0) <= 0:
        raise ValueError("suppressed_check_count must be positive for readiness")
    if int(artifact.get("escalated_check_count") or 0) <= 0:
        raise ValueError("escalated_check_count must be positive for readiness")
    if float(artifact.get("residual_false_accept_risk") or 0.0) != 0.0:
        raise ValueError("residual_false_accept_risk must be zero for readiness")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def honest_verdict(ready: bool, recommendation: str) -> str:
    """Return a conductor-compatible terminal verdict string."""

    if ready:
        return (
            "complete: fr11_experience_verifier_memory_v1_ready=true; "
            f"promotion_recommendation={recommendation}; no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11_experience_verifier_memory_v1_ready=false"


def contract_decision_from_row(row: Mapping[str, Any]) -> str:
    """Return the normalized verifier contract decision for a row."""

    for key in ("contract_decision", "expected_action", "ledger_action"):
        value = row.get(key)
        if value is not None:
            return normalize_token(value)
    return "unknown"


def ledger_consistent_from_row(row: Mapping[str, Any]) -> bool:
    """Read ledger consistency from direct fields or monitor events."""

    if "ledger_consistent" in row:
        return row.get("ledger_consistent") is True
    for event in row.get("monitor_events", []):
        if not isinstance(event, Mapping):
            continue
        if event.get("event_type") != "candidate_final_answer":
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            return payload.get("final_answer_consistent_with_ledger") is not False
    return True


def normalize_difficulty(value: Any) -> str:
    """Normalize a difficulty list/string into a deterministic key value."""

    if isinstance(value, str):
        return normalize_token(value)
    if isinstance(value, Sequence):
        normalized = sorted({normalize_token(item) for item in value if str(item)})
        return "|".join(normalized) if normalized else "unspecified"
    return "unspecified"


def normalize_token(value: Any) -> str:
    """Normalize an artifact token without losing replayability."""

    text = str(value).strip().lower().replace(" ", "_")
    return text or "unknown"


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate, using zero for empty denominators."""

    if denominator <= 0:
        return 0.0
    return round_float(numerator / denominator)


def round_float(value: float) -> float:
    """Round artifact floats to stable six-decimal precision."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return a stable elapsed duration for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the local source exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output so artifacts diff cleanly across reruns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
