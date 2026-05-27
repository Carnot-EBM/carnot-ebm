"""Exp 3200 FR-11 VeriFY-style trace-memory controller.

Spec refs: REQ-LEARN-3200, SCENARIO-LEARN-3200,
SCENARIO-LEARN-3200-BLOCKED.

This module turns the Exp 3186/3187 controller-memory evidence into explicit
self-verification trace records.  It is deliberately a controller-memory replay
artifact: it reads checked-in JSON, records routing decisions, and builds an
experience pool for redundant-check suppression.  It does not update base model
weights, verifier weights, KAN weights, hidden states, or any LLM parameters.
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
RUN_DATE = "20260527"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3200_fr11_verify_trace_memory_controller_v1"
SCHEMA = "carnot.fr11.verify_trace_memory_controller.v1"
OUTPUT_REL_PATH = Path("results/experiment_3200_fr11_verify_trace_memory_controller_v1.json")
EXP3186_REL_PATH = Path("results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json")
EXP3187_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_verify_trace_memory_controller_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3200_fr11_verify_trace_memory_controller_v1.py")

TRACE_SCHEMA_FIELDS = (
    "trace_id",
    "row_id",
    "replay_role",
    "initial_answer",
    "verification_query",
    "consistency_judgment",
    "answer_abstain_decision",
    "exact_label",
    "routing_outcome",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "continuous_self_learning_task",
    "source_artifacts",
    "trace_schema",
    "trace_count",
    "heldout_row_count",
    "drift_row_count",
    "negative_control_regression_count",
    "redundant_check_suppression_count",
    "routing_accuracy_delta",
    "model_weight_update_performed",
    "promotion_allowed",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3200_fr11_verify_trace_memory_controller_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3200_fr11_verify_trace_memory_controller_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_verify_trace_memory_controller_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3200_fr11_verify_trace_memory_controller_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3186_controller_memory_promotion", EXP3186_REL_PATH, True),
    ("exp3187_cross_environment_drift_replay", EXP3187_REL_PATH, True),
    ("exp3200_module", MODULE_REL_PATH, False),
    ("exp3200_tests", TEST_REL_PATH, False),
)
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the checked-in controller-memory artifacts used by Exp 3200."""

    root_path = Path(root)
    return {
        "exp3186": read_json_object(root_path / EXP3186_REL_PATH),
        "exp3187": read_json_object(root_path / EXP3187_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a schema-complete Exp 3200 artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, sources, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    exp3186 = sources["exp3186"]
    exp3187 = sources["exp3187"]
    materialized = materialize_trace_memory(exp3187)
    negative_count = int(exp3187.get("negative_control_regression_count") or 0)
    blockers = promotion_blockers(exp3186, exp3187, materialized)
    promotion_allowed = not blockers
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_artifacts": source_artifacts(root_path),
        "trace_schema": trace_schema(),
        "trace_count": int(materialized["trace_count"]),
        "heldout_row_count": int(materialized["heldout_row_count"]),
        "drift_row_count": int(materialized["drift_row_count"]),
        "negative_control_row_count": int(materialized["negative_control_row_count"]),
        "negative_control_regression_count": negative_count,
        "redundant_check_suppression_count": int(materialized["redundant_check_suppression_count"]),
        "routing_accuracy_delta": routing_accuracy_delta(exp3187),
        "heldout_drift_accuracy_delta": heldout_drift_accuracy_delta(exp3187),
        "model_weight_update_performed": False,
        "promotion_allowed": promotion_allowed,
        "promotion_blockers": blockers,
        "trace_records": materialized["trace_records"],
        "experience_pool": materialized["experience_pool"],
        "experience_pool_rule": experience_pool_rule(),
        "inference_substrate": inference_substrate(),
        "source_preconditions": precondition_checks(sources),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(
            promotion_allowed,
            int(materialized["trace_count"]),
            int(materialized["redundant_check_suppression_count"]),
            negative_count,
        ),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    sources: Mapping[str, Any],
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a terminal artifact when the required sources are unsafe."""

    return {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_artifacts": source_artifacts(root),
        "trace_schema": trace_schema(),
        "trace_count": 0,
        "heldout_row_count": 0,
        "drift_row_count": 0,
        "negative_control_row_count": 0,
        "negative_control_regression_count": 0,
        "redundant_check_suppression_count": 0,
        "routing_accuracy_delta": None,
        "heldout_drift_accuracy_delta": None,
        "model_weight_update_performed": detected_model_weight_update(sources),
        "promotion_allowed": False,
        "promotion_blockers": [blocker],
        "trace_records": [],
        "experience_pool": [],
        "experience_pool_rule": experience_pool_rule(),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "source_preconditions": precondition_checks(sources) | {"blocked_reason": blocker},
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": f"complete: blocked fr11 verify trace memory controller; {blocker}",
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """Fail closed when source artifacts are missing or claim forbidden learning."""

    exp3186 = sources.get("exp3186", {})
    exp3187 = sources.get("exp3187", {})
    if (
        not isinstance(exp3186, Mapping)
        or exp3186.get("fr11_controller_memory_promotion_pack_v1_ready") is not True
    ):
        return "exp3186_missing_or_not_ready"
    if exp3186.get("promotion_allowed") is not True:
        return "exp3186_promotion_not_allowed"
    if exp3186.get("no_model_weight_update_claimed") is not True:
        return "exp3186_model_weight_update_claimed"
    if source_claims_live_or_mutation(exp3186):
        return "exp3186_live_inference_or_weight_update_claimed"
    if (
        not isinstance(exp3187, Mapping)
        or exp3187.get("fr11_cross_environment_drift_replay_v1_ready") is not True
    ):
        return "exp3187_missing_or_not_ready"
    if exp3187.get("no_model_weight_update_claimed") is not True or source_claims_live_or_mutation(
        exp3187
    ):
        return "exp3187_live_inference_or_weight_update_claimed"
    return ""


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether a source artifact claims live inference or weight mutation."""

    substrate = payload.get("inference_substrate", {})
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def detected_model_weight_update(sources: Mapping[str, Any]) -> bool:
    """Report whether any loaded source already claimed a model-weight update."""

    for payload in sources.values():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("no_model_weight_update_claimed") is False:
            return True
        substrate = payload.get("inference_substrate", {})
        if isinstance(substrate, Mapping) and any(
            substrate.get(flag) is True for flag in MUTATION_FLAGS if "weight" in flag
        ):
            return True
    return False


def materialize_trace_memory(exp3187: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3200-1/2/3: build VeriFY-style traces and experience memory."""

    selection = exp3187.get("row_selection", {})
    selection_map = selection if isinstance(selection, Mapping) else {}
    role_rows = (
        ("heldout", rows_from_selection(selection_map, "heldout_rows")),
        ("drift", rows_from_selection(selection_map, "cross_environment_rows")),
        ("negative_control", rows_from_selection(selection_map, "negative_control_rows")),
    )
    experience_pool: dict[str, JsonDict] = {}
    trace_records: list[JsonDict] = []
    for replay_role, rows in role_rows:
        for row in rows:
            record = trace_record(row, replay_role, experience_pool)
            trace_records.append(record)
            update_experience_pool(experience_pool, record)
    return {
        "trace_records": trace_records,
        "experience_pool": list(experience_pool.values()),
        "trace_count": len(trace_records),
        "heldout_row_count": len(role_rows[0][1]),
        "drift_row_count": len(role_rows[1][1]),
        "negative_control_row_count": len(role_rows[2][1]),
        "redundant_check_suppression_count": sum(
            1 for record in trace_records if record["redundant_check_suppressed"]
        ),
    }


def rows_from_selection(selection: Mapping[str, Any], field: str) -> list[JsonDict]:
    """Return row dictionaries from an Exp 3187 row-selection field."""

    rows = selection.get(field, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def trace_record(
    row: Mapping[str, Any],
    replay_role: str,
    experience_pool: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Build one structured self-verification trace record."""

    summary = row_summary(row)
    key = evidence_key(summary)
    judgment = consistency_judgment_for(summary)
    answer_decision = answer_decision_for(judgment, summary["expected_action"])
    suppressed = should_suppress_recheck(experience_pool.get(key), summary, judgment)
    routing = routing_outcome_for(suppressed, answer_decision)
    return {
        "trace_id": trace_id_for(summary["row_id"], replay_role, key),
        "row_id": summary["row_id"],
        "replay_role": replay_role,
        "fixture_family": summary["fixture_family"],
        "source_artifact": summary["source_artifact"],
        "initial_answer": initial_answer_for(summary),
        "verification_query": verification_query_for(summary),
        "consistency_judgment": judgment,
        "answer_abstain_decision": answer_decision,
        "exact_label": summary["exact_label"],
        "routing_outcome": routing,
        "expected_action": summary["expected_action"],
        "ledger_action": summary["ledger_action"],
        "observed_action": summary["observed_action"],
        "observed_action_changed": summary["observed_action_changed"],
        "historical_exact_evidence_key": key,
        "redundant_check_suppressed": suppressed,
        "suppression_reason": "exact_evidence_recheck_redundant" if suppressed else "",
    }


def row_summary(row: Mapping[str, Any]) -> JsonDict:
    """Normalize row fields before building a self-verification trace."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "fixture_family": str(row.get("fixture_family") or "unknown"),
        "source_artifact": str(row.get("source_artifact") or ""),
        "expected_action": expected,
        "ledger_action": ledger,
        "observed_action": observed,
        "exact_label": exact_label_for(row),
        "observed_action_changed": bool(row.get("observed_action_changed")),
    }


def evidence_key(summary: Mapping[str, Any]) -> str:
    """Return an exact evidence key for same-row redundant-check decisions."""

    payload = {
        "row_id": str(summary.get("row_id") or ""),
        "expected_action": normalize_action(summary.get("expected_action")),
        "ledger_action": normalize_action(summary.get("ledger_action")),
        "exact_label": str(summary.get("exact_label") or ""),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def trace_id_for(row_id: str, replay_role: str, key: str) -> str:
    """Return a stable trace id that remains readable in row-level artifacts."""

    safe_row = row_id.replace("/", "_") or "missing-row"
    return f"trace-{replay_role}-{safe_row}-{key[:8]}"


def consistency_judgment_for(summary: Mapping[str, Any]) -> str:
    """Judge whether expected, ledger, and observed actions agree."""

    expected = normalize_action(summary.get("expected_action"))
    ledger = normalize_action(summary.get("ledger_action"))
    observed = normalize_action(summary.get("observed_action"))
    if expected not in {"missing", "unknown"} and expected == ledger == observed:
        return "consistent"
    return "inconsistent"


def answer_decision_for(consistency_judgment: str, expected_action: str) -> str:
    """Map exact consistency to the VeriFY-style answer/abstain decision."""

    if consistency_judgment == "consistent" and normalize_action(expected_action) == "accept":
        return "answer"
    return "abstain"


def routing_outcome_for(suppressed: bool, answer_decision: str) -> str:
    """Return the controller routing outcome for a trace."""

    if suppressed:
        return "skip_redundant_recheck"
    if answer_decision == "abstain":
        return "abstain_or_escalate"
    return "verify_then_answer"


def should_suppress_recheck(
    prior: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
    consistency_judgment: str,
) -> bool:
    """REQ-LEARN-3200-3: suppress only after unchanged exact evidence."""

    return (
        prior is not None
        and int(prior.get("observation_count") or 0) > 0
        and int(prior.get("inconsistent_count") or 0) == 0
        and int(prior.get("observed_action_change_count") or 0) == 0
        and consistency_judgment == "consistent"
        and not bool(summary.get("observed_action_changed"))
    )


def update_experience_pool(pool: dict[str, JsonDict], record: Mapping[str, Any]) -> None:
    """Record one exact trace outcome after the routing decision is made."""

    key = str(record.get("historical_exact_evidence_key") or "")
    summary = pool.setdefault(
        key,
        {
            "evidence_key": key,
            "observation_count": 0,
            "consistent_count": 0,
            "inconsistent_count": 0,
            "observed_action_change_count": 0,
            "suppressed_recheck_count": 0,
            "row_ids": [],
            "evidence_says_recheck_unlikely_to_change_outcome": False,
        },
    )
    summary["observation_count"] += 1
    summary["consistent_count"] += int(record.get("consistency_judgment") == "consistent")
    summary["inconsistent_count"] += int(record.get("consistency_judgment") != "consistent")
    summary["observed_action_change_count"] += int(bool(record.get("observed_action_changed")))
    summary["suppressed_recheck_count"] += int(bool(record.get("redundant_check_suppressed")))
    row_id = str(record.get("row_id") or "")
    if row_id and row_id not in summary["row_ids"]:
        summary["row_ids"].append(row_id)
    summary["evidence_says_recheck_unlikely_to_change_outcome"] = (
        int(summary["observation_count"]) > 0
        and int(summary["inconsistent_count"]) == 0
        and int(summary["observed_action_change_count"]) == 0
    )


def initial_answer_for(summary: Mapping[str, Any]) -> str:
    """Create the trace's initial answer from exact replay evidence."""

    row_id = str(summary.get("row_id") or "unknown-row")
    if normalize_action(summary.get("expected_action")) == "accept":
        return f"Row {row_id} is answerable under exact controller replay."
    return f"Row {row_id} should abstain because exact replay requires rejection."


def verification_query_for(summary: Mapping[str, Any]) -> str:
    """Create the trace's explicit verification query."""

    return (
        f"Does row {summary.get('row_id')} satisfy expected="
        f"{summary.get('expected_action')}, ledger={summary.get('ledger_action')}, "
        f"and observed={summary.get('observed_action')} under exact replay?"
    )


def exact_label_for(row: Mapping[str, Any]) -> str:
    """Return an exact label, deriving one from exact action authority if needed."""

    label = str(row.get("exact_label") or "").strip()
    if label and label.lower() not in {"unknown", "missing", "none"}:
        return label.upper()
    expected = normalize_action(row.get("expected_action"))
    if expected == "accept":
        return "EXACT_ACCEPT"
    if expected == "reject":
        return "EXACT_REJECT"
    return "EXACT_UNKNOWN"


def trace_schema() -> JsonDict:
    """Return the machine-readable VeriFY-style trace schema."""

    return {
        "schema_id": "carnot.fr11.verify_trace_memory_record.v1",
        "schema_version": SCHEMA_VERSION,
        "style": "VeriFY-inspired controller-memory trace",
        "fields": list(TRACE_SCHEMA_FIELDS),
        "field_descriptions": {
            "initial_answer": "controller-side initial answer reconstructed from exact replay",
            "verification_query": "explicit consistency query over exact replay fields",
            "consistency_judgment": "consistent or inconsistent exact action judgment",
            "answer_abstain_decision": "answer only for exact accepted consistent rows",
            "exact_label": "checked-in exact label or action-derived exact authority label",
            "routing_outcome": "verify_then_answer, abstain_or_escalate, or skip_redundant_recheck",
        },
    }


def experience_pool_rule() -> JsonDict:
    """Describe when historical exact evidence may suppress a redundant check."""

    return {
        "rule_id": "exact_row_action_recheck_suppression_v1",
        "suppression_allowed_when": [
            "same historical_exact_evidence_key has at least one prior observation",
            "prior inconsistent_count == 0",
            "prior observed_action_change_count == 0",
            "current consistency_judgment == consistent",
            "current observed_action_changed == false",
        ],
        "not_authority_for": [
            "accepting answers without exact replay",
            "negative-control regression forgiveness",
            "model-weight update claims",
        ],
    }


def promotion_blockers(
    exp3186: Mapping[str, Any],
    exp3187: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> list[str]:
    """REQ-LEARN-3200-4: list promotion blockers for the trace controller."""

    blockers: list[str] = []
    if exp3186.get("promotion_allowed") is not True:
        blockers.append("exp3186_promotion_not_allowed")
    if exp3187.get("promotion_allowed") is not True:
        blockers.append("exp3187_promotion_not_allowed")
    if int(materialized.get("trace_count") or 0) <= 0:
        blockers.append("empty_trace_memory")
    if int(materialized.get("heldout_row_count") or 0) <= 0:
        blockers.append("missing_heldout_replay")
    if int(materialized.get("drift_row_count") or 0) <= 0:
        blockers.append("missing_drift_replay")
    if int(exp3187.get("negative_control_regression_count") or 0) > 0:
        blockers.append("negative_control_regression")
    if heldout_drift_accuracy_delta(exp3187) is None:
        blockers.append("missing_heldout_drift_accuracy")
    return blockers


def routing_accuracy_delta(exp3187: Mapping[str, Any]) -> float | None:
    """Return the inherited source before/after routing lift when available."""

    before_after = exp3187.get("before_after_consistency", {})
    source = before_after.get("source", {}) if isinstance(before_after, Mapping) else {}
    lift = source.get("lift") if isinstance(source, Mapping) else None
    if isinstance(lift, int | float) and math.isfinite(float(lift)):
        return round_float(float(lift))
    return None


def heldout_drift_accuracy_delta(exp3187: Mapping[str, Any]) -> float | None:
    """Measure held-out plus drift replay delta separately from source lift."""

    before_after = exp3187.get("before_after_consistency", {})
    if not isinstance(before_after, Mapping):
        return None
    heldout = before_after.get("heldout", {})
    drift = before_after.get("cross_environment", {})
    if not isinstance(heldout, Mapping) or not isinstance(drift, Mapping):
        return None
    before = (float(heldout.get("before_rate") or 0.0) + float(drift.get("before_rate") or 0.0)) / 2
    after = (float(heldout.get("after_rate") or 0.0) + float(drift.get("after_rate") or 0.0)) / 2
    return round_float(after - before)


def precondition_checks(sources: Mapping[str, Any]) -> JsonDict:
    """Expose source readiness and mutation boundaries in the final artifact."""

    exp3186 = sources.get("exp3186", {})
    exp3187 = sources.get("exp3187", {})
    return {
        "exp3186_present": bool(exp3186),
        "exp3186_ready": isinstance(exp3186, Mapping)
        and exp3186.get("fr11_controller_memory_promotion_pack_v1_ready") is True,
        "exp3186_promotion_allowed": isinstance(exp3186, Mapping)
        and exp3186.get("promotion_allowed") is True,
        "exp3187_present": bool(exp3187),
        "exp3187_ready": isinstance(exp3187, Mapping)
        and exp3187.get("fr11_cross_environment_drift_replay_v1_ready") is True,
        "exp3187_promotion_allowed": isinstance(exp3187, Mapping)
        and exp3187.get("promotion_allowed") is True,
        "source_model_weight_update_detected": detected_model_weight_update(sources),
    }


def inference_substrate(mode: str = "controller_memory_trace_replay") -> JsonDict:
    """REQ-LEARN-3200-6: declare replay-only controller memory."""

    return {
        "mode": mode,
        "controller_memory_replay_only": True,
        "trace_memory_policy_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "hidden_state_mutation_claimed": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for artifact lineage."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        exists = path.is_file()
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3200 artifact overclaims learning or promotion."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("model_weight_update_performed") is not False:
        raise ValueError("model_weight_update_performed must remain false")
    schema = artifact.get("trace_schema")
    if not isinstance(schema, Mapping) or not set(TRACE_SCHEMA_FIELDS) <= set(
        schema.get("fields", [])
    ):
        raise ValueError("trace_schema must contain all VeriFY-style fields")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    if any(substrate.get(flag) is True for flag in MUTATION_FLAGS):
        raise ValueError("live-inference and model mutation flags must remain false")
    if artifact.get("promotion_allowed") is True:
        if int(artifact.get("trace_count") or 0) <= 0:
            raise ValueError("trace_count must be positive when promotion is allowed")
        if int(artifact.get("heldout_row_count") or 0) <= 0:
            raise ValueError("heldout_row_count must be positive when promotion is allowed")
        if int(artifact.get("drift_row_count") or 0) <= 0:
            raise ValueError("drift_row_count must be positive when promotion is allowed")
        if int(artifact.get("negative_control_regression_count") or 0) != 0:
            raise ValueError("promotion requires zero negative-control regressions")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3200 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(
    promotion_allowed: bool,
    trace_count: int,
    suppression_count: int,
    negative_regression_count: int,
) -> str:
    """Return a truthful terminal verdict for the controller-memory artifact."""

    return (
        "complete: fr11 verify trace-memory controller v1 materialized; "
        f"promotion_allowed={str(promotion_allowed).lower()}; "
        f"trace_count={trace_count}; "
        f"redundant_check_suppression_count={suppression_count}; "
        f"negative_control_regression_count={negative_regression_count}; "
        "model_weight_update_performed=false"
    )


def normalize_action(value: Any) -> str:
    """Normalize small controller action labels."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def rate(numerator: int, denominator: int) -> float:
    """Return a stable fraction without implicit zero-division behavior."""

    if denominator <= 0:
        return 0.0
    return round_float(numerator / denominator)


def round_float(value: float) -> float:
    """Round floats so regenerated artifacts stay stable."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when the source file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
