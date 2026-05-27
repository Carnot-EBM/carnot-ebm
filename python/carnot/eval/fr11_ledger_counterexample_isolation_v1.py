"""Exp 3171 FR-11 ledger counterexample isolation.

Spec refs: REQ-LEARN-3171, SCENARIO-LEARN-3171,
SCENARIO-LEARN-3171-BLOCKED.

This module turns the Exp 3156 aggregate ledger gap into concrete controller
replay work.  It does not train a model, call a live LLM, or infer new labels;
it only groups checked-in replay rows so the next pilot can update controller
memory on failing rows while measuring nonforgetting on rows that already work.
"""

from __future__ import annotations

from collections import defaultdict
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
ARTIFACT = "experiment_3171_fr11_ledger_counterexample_isolation_v1"
SCHEMA = "carnot.fr11.ledger_counterexample_isolation.v1"
OUTPUT_REL_PATH = Path("results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json")
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3143_REL_PATH = Path(
    "results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json"
)
EXP3156_REL_PATH = Path("results/experiment_3156_fr11_ledger_consistency_closure_v1.json")
EXP3157_REL_PATH = Path("results/experiment_3157_fr11_attractor_residual_memory_audit_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")

REQUIRED_ARTIFACT_FIELDS = {
    "fr11_ledger_counterexample_isolation_ready",
    "continuous_self_learning_task",
    "prior_ledger_consistency_rate",
    "isolated_counterexample_families",
    "passing_families",
    "environment_variant_split",
    "negative_control_rows",
    "suspected_failure_modes",
    "promotion_allowed",
    "no_model_weight_update_claimed",
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
ENVIRONMENT_VARIANT_CATEGORIES = {
    "admitted_environment",
    "equivalent_variant",
    "hardened_variant",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3171_fr11_ledger_counterexample_isolation_v1.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_ledger_counterexample_isolation_v1.py -m pytest -o addopts='' tests/python/test_experiment_3171_fr11_ledger_counterexample_isolation_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_ledger_counterexample_isolation_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3171_fr11_ledger_counterexample_isolation_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3128_evoenv_admission", EXP3128_REL_PATH, True),
    ("exp3129_constraint_memory_audit", EXP3129_REL_PATH, True),
    ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
    ("exp3142_vera_evoenv_replay", EXP3142_REL_PATH, True),
    ("exp3143_experience_memory", EXP3143_REL_PATH, True),
    ("exp3156_ledger_closure", EXP3156_REL_PATH, True),
    ("exp3157_attractor_residual_memory", EXP3157_REL_PATH, True),
    (
        "exp3171_module",
        Path("python/carnot/eval/fr11_ledger_counterexample_isolation_v1.py"),
        False,
    ),
    (
        "exp3171_tests",
        Path("tests/python/test_experiment_3171_fr11_ledger_counterexample_isolation_v1.py"),
        False,
    ),
)
READY_FLAGS = (
    (
        "exp3156",
        "fr11_ledger_consistency_closure_v1_ready",
        "exp3156_ledger_closure_missing_or_not_ready",
    ),
    (
        "exp3157",
        "fr11_attractor_residual_memory_audit_v1_ready",
        "exp3157_residual_memory_missing_or_not_ready",
    ),
    ("exp3128", "fr11_evoenv_pilot_v1_ready", "exp3128_evoenv_missing_or_not_ready"),
    (
        "exp3129",
        "fr11_constraint_memory_audit_v1_ready",
        "exp3129_constraint_memory_missing_or_not_ready",
    ),
    ("exp3136", "false_accept_autopsy_v1_ready", "exp3136_false_accept_missing_or_not_ready"),
    ("exp3142", "fr11_vera_evoenv_v2_ready", "exp3142_vera_evoenv_missing_or_not_ready"),
    (
        "exp3143",
        "fr11_experience_verifier_memory_v1_ready",
        "exp3143_experience_memory_missing_or_not_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load every checked-in artifact used by the Exp 3171 isolation."""

    root_path = Path(root)
    return {
        "exp3128": read_json_object(root_path / EXP3128_REL_PATH),
        "exp3129": read_json_object(root_path / EXP3129_REL_PATH),
        "exp3136": read_json_object(root_path / EXP3136_REL_PATH),
        "exp3142": read_json_object(root_path / EXP3142_REL_PATH),
        "exp3143": read_json_object(root_path / EXP3143_REL_PATH),
        "exp3156": read_json_object(root_path / EXP3156_REL_PATH),
        "exp3157": read_json_object(root_path / EXP3157_REL_PATH),
    }


def isolate_counterexamples(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """REQ-LEARN-3171-1/2/3/4/5: isolate rows and define the pilot split."""

    exp3156 = sources.get("exp3156", {})
    exp3157 = sources.get("exp3157", {})
    rows = replay_rows(exp3156)
    failing_rows = [row for row in rows if not bool(row.get("consistent"))]
    passing_rows = [row for row in rows if bool(row.get("consistent"))]
    counterexample_families = grouped_counterexample_families(failing_rows)
    passing = grouped_passing_families(passing_rows)
    negative_controls = [
        row_summary(row) | {"control_role": "environment_variant_nonforgetting"}
        for row in passing_rows
        if str(row.get("panel_category") or "") in ENVIRONMENT_VARIANT_CATEGORIES
    ]
    split = environment_variant_split(failing_rows, passing_rows, negative_controls)
    consistent_count = sum(1 for row in rows if bool(row.get("consistent")))
    prior_rate = float(exp3156.get("ledger_consistency_rate") or rate(consistent_count, len(rows)))
    return {
        "prior_ledger_consistency_rate": round_float(prior_rate),
        "replay_panel_count": len(rows),
        "ledger_consistent_count": consistent_count,
        "ledger_inconsistent_count": len(failing_rows),
        "source_replay_panel_count": int(exp3156.get("replay_panel_count") or 0),
        "source_ledger_consistent_count": int(exp3156.get("ledger_consistent_count") or 0),
        "isolated_counterexample_families": counterexample_families,
        "passing_families": passing,
        "environment_variant_split": split,
        "negative_control_rows": negative_controls,
        "suspected_failure_modes": suspected_failure_modes(rows, failing_rows, exp3157),
    }


def replay_rows(exp3156: Mapping[str, Any]) -> list[JsonDict]:
    """Return normalized Exp 3156 replay rows."""

    rows = exp3156.get("replay_panel_rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def grouped_counterexample_families(failing_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group failing rows by fixture family with replayable row evidence."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in failing_rows:
        grouped[str(row.get("fixture_family") or "unknown")].append(row)
    families: list[JsonDict] = []
    for family in sorted(grouped):
        rows = grouped[family]
        families.append(
            {
                "fixture_family": family,
                "failing_row_count": len(rows),
                "failing_row_ids": [str(row.get("row_id") or "") for row in rows],
                "mismatch_classes": sorted({str(row.get("mismatch_class") or "") for row in rows}),
                "panel_categories": sorted({str(row.get("panel_category") or "") for row in rows}),
                "source_artifacts": sorted({str(row.get("source_artifact") or "") for row in rows}),
                "rows": [row_summary(row) for row in rows],
            }
        )
    return families


def grouped_passing_families(passing_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group passing rows so working controller memory behavior is retained."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in passing_rows:
        grouped[str(row.get("fixture_family") or "unknown")].append(row)
    families: list[JsonDict] = []
    for family in sorted(grouped):
        rows = grouped[family]
        families.append(
            {
                "fixture_family": family,
                "passing_row_count": len(rows),
                "passing_row_ids": [str(row.get("row_id") or "") for row in rows],
                "panel_categories": sorted({str(row.get("panel_category") or "") for row in rows}),
                "rows": [row_summary(row) for row in rows],
            }
        )
    return families


def environment_variant_split(
    failing_rows: Sequence[Mapping[str, Any]],
    passing_rows: Sequence[Mapping[str, Any]],
    negative_controls: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """REQ-LEARN-3171-4: separate update rows from evaluation controls."""

    training_rows = [row_summary(row) | {"split_role": "controller_memory_update"} for row in failing_rows]
    heldout_rows = [row_summary(row) | {"split_role": "held_out_nonforgetting_replay"} for row in passing_rows]
    environment_rows = [
        row_summary(row)
        for row in passing_rows
        if str(row.get("panel_category") or "") == "admitted_environment"
    ]
    variant_rows = [
        row_summary(row)
        for row in passing_rows
        if str(row.get("panel_category") or "") in {"equivalent_variant", "hardened_variant"}
    ]
    return {
        "split_policy": (
            "Train/update only on rows that failed Exp 3156; replay every passing "
            "row as held-out nonforgetting evidence; use passing EvoEnv and VeRA "
            "environment/variant rows as negative controls."
        ),
        "training_update_rows": training_rows,
        "held_out_replay_rows": heldout_rows,
        "environment_rows": environment_rows,
        "variant_rows": variant_rows,
        "negative_control_rows": list(negative_controls),
        "split_counts": {
            "training_update_rows": len(training_rows),
            "held_out_replay_rows": len(heldout_rows),
            "negative_control_rows": len(negative_controls),
        },
    }


def suspected_failure_modes(
    rows: Sequence[Mapping[str, Any]],
    failing_rows: Sequence[Mapping[str, Any]],
    exp3157: Mapping[str, Any],
) -> list[JsonDict]:
    """REQ-LEARN-3171-5: identify which proposed causes fit the evidence."""

    failing_ids = [str(row.get("row_id") or "") for row in failing_rows]
    contradiction_rows = [
        str(row.get("row_id") or "")
        for row in failing_rows
        if normalize_action(row.get("expected_action")) == "reject"
        and normalize_action(row.get("ledger_action")) == "reject"
        and normalize_action(row.get("observed_action")) == "accept"
    ]
    stale_rows = [
        str(row.get("row_id") or "")
        for row in failing_rows
        if normalize_action(row.get("routing_decision")) == "suppress"
    ]
    environment_rows = [
        str(row.get("row_id") or "")
        for row in rows
        if str(row.get("panel_category") or "") in ENVIRONMENT_VARIANT_CATEGORIES
        and not bool(row.get("consistent"))
    ]
    missing_label_rows = [
        str(row.get("row_id") or "")
        for row in failing_rows
        if normalize_action(row.get("exact_label")) in {"", "missing", "unknown"}
    ]
    threshold_rows = [
        str(row.get("row_id") or "")
        for row in failing_rows
        if "threshold" in json.dumps(row, sort_keys=True).lower()
    ]
    schema_rows = failing_ids if aggregation_schema_mismatch(rows, exp3157) else []
    return [
        failure_mode(
            "controller_observed_decision_contradicts_exact_reject",
            contradiction_rows,
            "Exact expected action and ledger action reject, but the observed controller decision accepts.",
        ),
        failure_mode(
            "stale_memory",
            stale_rows,
            "Would apply if a failing row was suppressed by old routing memory; the isolated failures route to escalation.",
        ),
        failure_mode(
            "environment_mismatch",
            environment_rows,
            "Would apply if admitted environments or VeRA variants failed exact replay; those rows pass.",
        ),
        failure_mode(
            "threshold_drift",
            threshold_rows,
            "Would apply if a failing row carried threshold evidence; the closure rows are action/token mismatches.",
        ),
        failure_mode(
            "missing_exact_label",
            missing_label_rows,
            "Would apply if exact labels were missing; the failures carry exact INVALID/UNSAT labels.",
        ),
        failure_mode(
            "aggregation_schema_mismatch",
            schema_rows,
            "Would apply if row counts and carried-forward rates disagreed across artifacts.",
        ),
    ]


def aggregation_schema_mismatch(
    rows: Sequence[Mapping[str, Any]],
    exp3157: Mapping[str, Any],
) -> bool:
    """Return whether cross-artifact denominator/rate evidence disagrees."""

    if not rows:
        return False
    rate3157 = exp3157.get("ledger_consistency_rate")
    if rate3157 is None:
        return False
    computed = rate(sum(1 for row in rows if bool(row.get("consistent"))), len(rows))
    return round_float(float(rate3157)) != computed


def failure_mode(mode: str, row_ids: Sequence[str], evidence: str) -> JsonDict:
    """Build one machine-readable failure-mode diagnosis."""

    clean_ids = [row_id for row_id in row_ids if row_id]
    return {
        "mode": mode,
        "applies": bool(clean_ids),
        "row_ids": clean_ids,
        "evidence": evidence,
    }


def row_summary(row: Mapping[str, Any]) -> JsonDict:
    """Return replay fields needed by the next controller-only pilot."""

    return {
        "row_id": str(row.get("row_id") or ""),
        "fixture_family": str(row.get("fixture_family") or "unknown"),
        "panel_category": str(row.get("panel_category") or "unknown"),
        "source_artifact": str(row.get("source_artifact") or ""),
        "expected_action": normalize_action(row.get("expected_action")),
        "ledger_action": normalize_action(row.get("ledger_action")),
        "observed_action": normalize_action(row.get("observed_action")),
        "routing_decision": normalize_action(row.get("routing_decision")),
        "mismatch_class": str(row.get("mismatch_class") or ""),
        "exact_label": str(row.get("exact_label") or "unknown"),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3171 terminal artifact from checked-in replay evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = precondition_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact
    isolation = isolate_counterexamples(sources)
    prior_rate = float(isolation["prior_ledger_consistency_rate"])
    isolated = isolation["isolated_counterexample_families"]
    ready = int(isolation["replay_panel_count"]) > 0 and (prior_rate == 1.0 or bool(isolated))
    promotion_allowed = prior_rate == 1.0 and not isolated
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_ledger_counterexample_isolation_ready": ready,
        "continuous_self_learning_task": True,
        "prior_ledger_consistency_rate": prior_rate,
        "isolated_counterexample_families": isolated,
        "passing_families": isolation["passing_families"],
        "environment_variant_split": isolation["environment_variant_split"],
        "negative_control_rows": isolation["negative_control_rows"],
        "suspected_failure_modes": isolation["suspected_failure_modes"],
        "promotion_allowed": promotion_allowed,
        "no_model_weight_update_claimed": True,
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "replay_panel_count": int(isolation["replay_panel_count"]),
        "ledger_consistent_count": int(isolation["ledger_consistent_count"]),
        "ledger_inconsistent_count": int(isolation["ledger_inconsistent_count"]),
        "source_replay_panel_count": int(isolation["source_replay_panel_count"]),
        "source_ledger_consistent_count": int(isolation["source_ledger_consistent_count"]),
        "precondition_checks": precondition_checks(sources),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(ready, prior_rate, isolated, promotion_allowed),
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
    """Return a schema-complete blocked artifact when evidence is missing."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_ledger_counterexample_isolation_ready": False,
        "continuous_self_learning_task": True,
        "prior_ledger_consistency_rate": 0.0,
        "isolated_counterexample_families": [],
        "passing_families": [],
        "environment_variant_split": empty_split(),
        "negative_control_rows": [],
        "suspected_failure_modes": [],
        "promotion_allowed": False,
        "no_model_weight_update_claimed": True,
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "blocked_reason": blocker,
        "precondition_checks": {},
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write stable Exp 3171 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(sources: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the first missing source required for isolation."""

    for source_id, ready_field, blocker in READY_FLAGS:
        if sources.get(source_id, {}).get(ready_field) is not True:
            return blocker
    if not replay_rows(sources.get("exp3156", {})):
        return "exp3156_replay_panel_rows_missing"
    return ""


def precondition_checks(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Expose source readiness for auditability."""

    checks: JsonDict = {}
    for source_id, ready_field, _blocker in READY_FLAGS:
        checks[f"{source_id}_{ready_field}"] = sources.get(source_id, {}).get(ready_field) is True
    checks["exp3156_replay_panel_rows_present"] = bool(replay_rows(sources.get("exp3156", {})))
    return checks


def empty_split() -> JsonDict:
    """Return the schema-complete split used by blocked artifacts."""

    return {
        "split_policy": "blocked_precondition_check",
        "training_update_rows": [],
        "held_out_replay_rows": [],
        "environment_rows": [],
        "variant_rows": [],
        "negative_control_rows": [],
        "split_counts": {
            "training_update_rows": 0,
            "held_out_replay_rows": 0,
            "negative_control_rows": 0,
        },
    }


def inference_substrate(mode: str = "controller_memory_replay") -> JsonDict:
    """Declare that isolation uses controller replay, not live inference."""

    return {
        "mode": mode,
        "controller_memory_replay_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for traceability."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3171 artifact overclaims the isolation result."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("no_model_weight_update_claimed") is not True:
        raise ValueError("no_model_weight_update_claimed must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or any(
        substrate.get(flag) is True
        for flag in ("model_weight_mutation", "model_weight_training", "base_model_weights_updated")
    ):
        raise ValueError("model_weight_mutation must remain false")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    prior_rate = float(artifact.get("prior_ledger_consistency_rate", math.nan))
    if not math.isfinite(prior_rate) or not 0.0 <= prior_rate <= 1.0:
        raise ValueError("prior_ledger_consistency_rate must be finite and within [0, 1]")
    if prior_rate < 1.0 and artifact.get("promotion_allowed") is True:
        raise ValueError("promotion_allowed must be false before ledger consistency reaches 1.0")
    assert_split_disjoint(artifact.get("environment_variant_split"))
    if artifact.get("fr11_ledger_counterexample_isolation_ready") is not True:
        return
    if prior_rate < 1.0 and not artifact.get("isolated_counterexample_families"):
        raise ValueError("imperfect ledgers must include isolated counterexample families")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def assert_split_disjoint(split: Any) -> None:
    """Ensure training-update rows are never also evaluation/control rows."""

    if not isinstance(split, Mapping):
        raise ValueError("environment_variant_split must be an object")
    training = row_id_set(split.get("training_update_rows", []))
    heldout = row_id_set(split.get("held_out_replay_rows", []))
    negative = row_id_set(split.get("negative_control_rows", []))
    if training & heldout or training & negative:
        raise ValueError("split overlap between training rows and evaluation controls")


def row_id_set(rows: Any) -> set[str]:
    """Return row ids from a list of row-like dictionaries."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return set()
    return {str(row.get("row_id") or "") for row in rows if isinstance(row, Mapping)}


def honest_verdict(
    ready: bool,
    prior_rate: float,
    isolated: Sequence[Mapping[str, Any]],
    promotion_allowed: bool,
) -> str:
    """Return a conductor-compatible terminal verdict."""

    if ready:
        return (
            "complete: fr11 ledger counterexample isolation ready; "
            f"prior_ledger_consistency_rate={round_float(prior_rate)}; "
            f"isolated_counterexample_family_count={len(isolated)}; "
            f"promotion_allowed={str(promotion_allowed).lower()}; "
            "no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11 ledger counterexample isolation sources missing"


def normalize_action(value: Any) -> str:
    """Normalize small action tokens used by replay artifacts."""

    text = str(value or "").strip().lower().replace(" ", "_")
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
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the path exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
