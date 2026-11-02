"""Exp 3187 FR-11 cross-environment drift replay v1.

Spec refs: REQ-LEARN-3187, SCENARIO-LEARN-3187,
SCENARIO-LEARN-3187-BLOCKED.

This module checks whether the controller-memory update packaged by Exp 3186
stays safe outside the original counterexample family.  It deliberately treats
the update as replay data: exact row-id routing overrides are applied to copied
row dictionaries only, so the experiment can produce drift evidence without
touching production controller state or model weights.
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
ARTIFACT = "experiment_3187_fr11_cross_environment_drift_replay_v1"
SCHEMA = "carnot.fr11.cross_environment_drift_replay.v1"
OUTPUT_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
EXP3186_REL_PATH = Path("results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json")
EXP3172_REL_PATH = Path("results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json")
EXP3171_REL_PATH = Path("results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_cross_environment_drift_replay_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3187_fr11_cross_environment_drift_replay_v1.py")

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_cross_environment_drift_replay_v1_ready",
    "continuous_self_learning_task",
    "replay_mode_only",
    "no_model_weight_update_claimed",
    "heldout_row_count",
    "cross_environment_row_count",
    "before_after_consistency",
    "negative_control_regression_count",
    "drift_cases",
    "rollback_triggered",
    "promotion_allowed",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3187_fr11_cross_environment_drift_replay_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3187_fr11_cross_environment_drift_replay_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_cross_environment_drift_replay_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3187_fr11_cross_environment_drift_replay_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, True),
    ("exp3186_promotion_pack", EXP3186_REL_PATH, True),
    ("exp3172_nonforgetting_replay", EXP3172_REL_PATH, True),
    ("exp3171_counterexample_isolation", EXP3171_REL_PATH, True),
    ("exp3187_module", MODULE_REL_PATH, False),
    ("exp3187_tests", TEST_REL_PATH, False),
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
    """Read checked-in JSON evidence and fail closed when it is absent.

    A replay artifact should never infer missing authority from filenames or
    partial text.  Returning an empty mapping keeps every caller on the same
    precondition path and makes blocked artifacts deterministic.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the promotion pack and replay splits used by Exp 3187."""

    root_path = Path(root)
    return {
        "exp3186": read_json_object(root_path / EXP3186_REL_PATH),
        "exp3172": read_json_object(root_path / EXP3172_REL_PATH),
        "exp3171": read_json_object(root_path / EXP3171_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a schema-complete Exp 3187 artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run, sources)
        validate_artifact(artifact)
        return artifact
    artifact = build_drift_replay(root_path, sources, start, now_s, tests_run)
    validate_artifact(artifact)
    return artifact


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3187-1: fail closed unless Exp 3186 can authorize replay."""

    exp3186 = sources.get("exp3186", {})
    if (
        not isinstance(exp3186, Mapping)
        or exp3186.get("fr11_controller_memory_promotion_pack_v1_ready") is not True
    ):
        return "exp3186_missing_or_not_ready"
    if exp3186.get("promotion_allowed") is not True:
        return "exp3186_promotion_not_allowed"
    if exp3186.get("no_model_weight_update_claimed") is not True:
        return "exp3186_model_weight_update_claimed"
    substrate = exp3186.get("inference_substrate", {})
    if not isinstance(substrate, Mapping) or int(substrate.get("fresh_live_inference_calls") or 0):
        return "exp3186_live_inference_claimed"
    manifest = exp3186.get("promotion_manifest")
    if not isinstance(manifest, Mapping):
        return "exp3186_promotion_manifest_missing"
    if not isinstance(manifest.get("activation_predicate"), Mapping):
        return "exp3186_activation_predicate_missing"
    return ""


def blocked_artifact(
    root: Path,
    blocker: str,
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
    sources: Mapping[str, Any],
) -> JsonDict:
    """Return a blocked artifact that performs no replay mutation."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_cross_environment_drift_replay_v1_ready": False,
        "continuous_self_learning_task": True,
        "replay_mode_only": True,
        "no_model_weight_update_claimed": True,
        "heldout_row_count": 0,
        "cross_environment_row_count": 0,
        "before_after_consistency": empty_consistency(),
        "negative_control_regression_count": 0,
        "drift_cases": [],
        "rollback_triggered": True,
        "promotion_allowed": False,
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "source_artifacts": source_artifacts(root),
        "precondition_checks": precondition_checks(sources) | {"blocked_reason": blocker},
        "row_selection": empty_row_selection(),
        "replay_rows": [],
        "rollback_triggers": ["blocked_precondition"],
        "matrix_v29_input": {
            "ready_for_matrix_v29": False,
            "blocked_reason": blocker,
            "artifact_path": OUTPUT_REL_PATH.as_posix(),
        },
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }


def build_drift_replay(
    root: Path,
    sources: Mapping[str, Any],
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """REQ-LEARN-3187-2/3/4/5/6: replay exact controller memory on controls."""

    exp3186 = sources["exp3186"]
    exp3172 = sources.get("exp3172", {})
    exp3171 = sources.get("exp3171", {})
    manifest = dict(exp3186["promotion_manifest"])
    overrides = row_overrides(manifest)
    selected = select_replay_rows(manifest, exp3172, exp3171)
    heldout = replay_panel(selected["heldout_rows"], overrides)
    cross_environment = replay_panel(selected["cross_environment_rows"], overrides)
    negative = replay_panel(selected["negative_control_rows"], overrides)
    negative_regressions = negative_control_regressions(negative)
    drift = drift_cases(heldout, "heldout") + drift_cases(cross_environment, "cross_environment")
    rollback = bool(drift or negative_regressions)
    before_after = before_after_consistency(manifest, heldout, cross_environment, negative)
    promotion_allowed = (
        not rollback
        and bool(heldout)
        and bool(cross_environment)
        and before_after["heldout"]["after_rate"] == 1.0
        and before_after["cross_environment"]["after_rate"] == 1.0
        and before_after["negative_control"]["regression_count"] == 0
    )
    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_cross_environment_drift_replay_v1_ready": True,
        "continuous_self_learning_task": True,
        "replay_mode_only": True,
        "no_model_weight_update_claimed": True,
        "heldout_row_count": len(heldout),
        "cross_environment_row_count": len(cross_environment),
        "before_after_consistency": before_after,
        "negative_control_regression_count": len(negative_regressions),
        "negative_control_regressions": negative_regressions,
        "drift_cases": drift,
        "rollback_triggered": rollback,
        "promotion_allowed": promotion_allowed,
        "inference_substrate": inference_substrate(),
        "source_artifacts": source_artifacts(root),
        "precondition_checks": precondition_checks(sources),
        "promotion_manifest_ref": {
            "source_artifact": EXP3186_REL_PATH.as_posix(),
            "update_id": str(manifest.get("update_id") or ""),
            "update_type": str(manifest.get("update_type") or ""),
            "row_action_override_count": len(overrides),
        },
        "row_selection": selected,
        "replay_rows": heldout + cross_environment + negative,
        "rollback_triggers": rollback_triggers(drift, negative_regressions),
        "matrix_v29_input": {
            "ready_for_matrix_v29": True,
            "artifact_path": OUTPUT_REL_PATH.as_posix(),
            "fr11_status": "promotion_allowed" if promotion_allowed else "rollback_triggered",
        },
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": honest_verdict(
            promotion_allowed, rollback, len(drift), len(negative_regressions)
        ),
    }


def select_replay_rows(
    manifest: Mapping[str, Any],
    exp3172: Mapping[str, Any],
    exp3171: Mapping[str, Any],
) -> JsonDict:
    """REQ-LEARN-3187-2: choose held-out controls and non-source environments."""

    source_families = sorted(
        {str(family) for family in manifest.get("source_counterexample_families", [])}
    )
    heldout = [
        row for row in rows_from_payload(exp3172, "heldout_replay_rows") if has_exact_authority(row)
    ]
    negative = [
        row
        for row in rows_from_payload(exp3172, "negative_control_replay_rows")
        if has_exact_authority(row)
    ]
    split = exp3171.get("environment_variant_split", {})
    split_map = split if isinstance(split, Mapping) else {}
    candidates = (
        rows_from_payload(split_map, "environment_rows")
        + rows_from_payload(split_map, "variant_rows")
        + rows_from_payload(split_map, "held_out_replay_rows")
    )
    cross_environment = [
        row
        for row in dedupe_rows(candidates)
        if has_exact_authority(row)
        and str(row.get("fixture_family") or "") not in source_families
        and str(row.get("panel_category") or "")
        in {"admitted_environment", "equivalent_variant", "hardened_variant"}
    ]
    return {
        "source_counterexample_families": source_families,
        "selection_policy": (
            "heldout and negative controls from Exp 3172; cross-environment "
            "rows from Exp 3171 environment/variant split excluding source "
            "counterexample families"
        ),
        "heldout_rows": annotate_authority(dedupe_rows(heldout)),
        "negative_control_rows": annotate_authority(dedupe_rows(negative)),
        "cross_environment_rows": annotate_authority(cross_environment),
    }


def rows_from_payload(payload: Mapping[str, Any], field: str) -> list[JsonDict]:
    """Return row dictionaries from a JSON payload field."""

    rows = payload.get(field, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep the first instance of each row id so denominators stay auditable."""

    by_id: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if row_id and row_id not in by_id:
            by_id[row_id] = dict(row)
    return list(by_id.values())


def annotate_authority(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Copy selected rows and expose why they can be replayed exactly."""

    return [
        row_summary(row)
        | {
            "exact_authority": has_exact_authority(row),
            "exact_authority_source": "expected_ledger_observed_action_consensus",
        }
        for row in rows
    ]


def row_overrides(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return exact row-id controller overrides from an Exp 3186 manifest."""

    activation = manifest.get("activation_predicate", {})
    overrides = (
        activation.get("row_action_overrides", {}) if isinstance(activation, Mapping) else {}
    )
    if not isinstance(overrides, Mapping):
        return {}
    return {str(row_id): normalize_action(action) for row_id, action in overrides.items()}


def replay_panel(
    rows: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, str],
) -> list[JsonDict]:
    """Replay rows before and after applying copied exact-row overrides."""

    return [replay_row(row, overrides) for row in rows]


def replay_row(row: Mapping[str, Any], overrides: Mapping[str, str]) -> JsonDict:
    """Apply the controller-memory update to one copied row only."""

    before = row_summary(row)
    override = overrides.get(before["row_id"])
    after = dict(before)
    if override is not None:
        after["observed_action"] = normalize_action(override)
    return {
        **after,
        "pre_update_observed_action": before["observed_action"],
        "post_update_observed_action": after["observed_action"],
        "controller_update_applied": override is not None,
        "before_consistent": row_consistent(before),
        "after_consistent": row_consistent(after),
        "observed_action_changed": after["observed_action"] != before["observed_action"],
        "exact_authority": has_exact_authority(before),
    }


def row_summary(row: Mapping[str, Any]) -> JsonDict:
    """Normalize replay fields before consistency checks."""

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


def has_exact_authority(row: Mapping[str, Any]) -> bool:
    """Return whether a row has enough checked-in authority for replay."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    return (
        expected not in {"missing", "unknown"}
        and ledger not in {"missing", "unknown"}
        and observed not in {"missing", "unknown"}
        and expected == ledger
    )


def row_consistent(row: Mapping[str, Any]) -> bool:
    """Judge consistency from expected, ledger, and observed actions."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    return observed not in {"missing", "unknown"} and expected == ledger == observed


def negative_control_regressions(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """REQ-LEARN-3187-6: list negative controls worsened by replayed memory."""

    regressions: list[JsonDict] = []
    for row in rows:
        if row.get("before_consistent") and (
            not row.get("after_consistent") or row.get("observed_action_changed")
        ):
            regressions.append(regression_case(row, "negative_control_regression"))
    return regressions


def drift_cases(rows: Sequence[Mapping[str, Any]], role: str) -> list[JsonDict]:
    """REQ-LEARN-3187-5: list held-out or transfer rows that drift."""

    cases: list[JsonDict] = []
    for row in rows:
        if row.get("before_consistent") and (
            not row.get("after_consistent") or row.get("observed_action_changed")
        ):
            cases.append(regression_case(row, f"{role}_drift_failure") | {"replay_role": role})
    return cases


def regression_case(row: Mapping[str, Any], reason: str) -> JsonDict:
    """Return the compact row evidence needed for rollback triage."""

    return {
        "row_id": str(row.get("row_id") or ""),
        "fixture_family": str(row.get("fixture_family") or "unknown"),
        "panel_category": str(row.get("panel_category") or "unknown"),
        "source_artifact": str(row.get("source_artifact") or ""),
        "reason": reason,
        "before_observed_action": normalize_action(row.get("pre_update_observed_action")),
        "after_observed_action": normalize_action(row.get("post_update_observed_action")),
        "expected_action": normalize_action(row.get("expected_action")),
        "ledger_action": normalize_action(row.get("ledger_action")),
    }


def before_after_consistency(
    manifest: Mapping[str, Any],
    heldout_rows: Sequence[Mapping[str, Any]],
    cross_environment_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """REQ-LEARN-3187-4: expose replay denominators and consistency rates."""

    evidence = manifest.get("evidence", {}) if isinstance(manifest.get("evidence"), Mapping) else {}
    before = round_float(float(evidence.get("before_ledger_consistency_rate") or 0.0))
    after = round_float(float(evidence.get("after_ledger_consistency_rate") or 0.0))
    return {
        "source": {
            "before_rate": before,
            "after_rate": after,
            "lift": round_float(after - before),
            "source_artifact": EXP3186_REL_PATH.as_posix(),
        },
        "heldout": panel_consistency(heldout_rows),
        "cross_environment": panel_consistency(cross_environment_rows),
        "negative_control": panel_consistency(negative_rows)
        | {"regression_count": len(negative_control_regressions(negative_rows))},
    }


def panel_consistency(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return row count and before/after consistency rates for one panel."""

    return {
        "row_count": len(rows),
        "before_rate": rate(sum(1 for row in rows if row.get("before_consistent")), len(rows)),
        "after_rate": rate(sum(1 for row in rows if row.get("after_consistent")), len(rows)),
    }


def rollback_triggers(
    drift: Sequence[Mapping[str, Any]],
    negative_regressions: Sequence[Mapping[str, Any]],
) -> list[str]:
    """List the exact rollback triggers activated by replay evidence."""

    triggers: list[str] = []
    if negative_regressions:
        triggers.append("negative_control_regression")
    if drift:
        triggers.append("cross_environment_drift_failure")
    return triggers


def empty_consistency() -> JsonDict:
    """Return schema-stable zero metrics for blocked artifacts."""

    empty = {"row_count": 0, "before_rate": 0.0, "after_rate": 0.0}
    return {
        "source": {"before_rate": 0.0, "after_rate": 0.0, "lift": 0.0},
        "heldout": dict(empty),
        "cross_environment": dict(empty),
        "negative_control": dict(empty) | {"regression_count": 0},
    }


def empty_row_selection() -> JsonDict:
    """Return schema-stable empty selection metadata for blocked artifacts."""

    return {
        "source_counterexample_families": [],
        "selection_policy": "blocked_precondition_check",
        "heldout_rows": [],
        "negative_control_rows": [],
        "cross_environment_rows": [],
    }


def precondition_checks(sources: Mapping[str, Any]) -> JsonDict:
    """Expose source gates so blocked artifacts explain what failed."""

    exp3186 = sources.get("exp3186", {})
    manifest = exp3186.get("promotion_manifest", {}) if isinstance(exp3186, Mapping) else {}
    return {
        "exp3186_present": bool(exp3186),
        "exp3186_ready": isinstance(exp3186, Mapping)
        and exp3186.get("fr11_controller_memory_promotion_pack_v1_ready") is True,
        "exp3186_promotion_allowed": isinstance(exp3186, Mapping)
        and exp3186.get("promotion_allowed") is True,
        "promotion_manifest_present": isinstance(manifest, Mapping) and bool(manifest),
        "exp3172_present": bool(sources.get("exp3172")),
        "exp3171_present": bool(sources.get("exp3171")),
    }


def inference_substrate(mode: str = "controller_memory_cross_environment_drift_replay") -> JsonDict:
    """REQ-LEARN-3187-3: declare exact/cached replay with no live inference."""

    return {
        "mode": mode,
        "aggregation_and_replay_only": True,
        "controller_memory_replay_only": True,
        "uses_checked_in_artifacts_only": True,
        "replay_mode_only": True,
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
    """List source files and checksums for matrix v29 traceability."""

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
    """Raise when the replay artifact overclaims promotion safety."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("replay_mode_only") is not True:
        raise ValueError("replay_mode_only must be true")
    if artifact.get("no_model_weight_update_claimed") is not True:
        raise ValueError("no_model_weight_update_claimed must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    if any(substrate.get(flag) is True for flag in MUTATION_FLAGS):
        raise ValueError("live-inference and model mutation flags must remain false")
    before_after = artifact.get("before_after_consistency")
    if not isinstance(before_after, Mapping):
        raise ValueError("before_after_consistency must be a mapping")
    if artifact.get("fr11_cross_environment_drift_replay_v1_ready") is not True:
        return
    if int(artifact.get("heldout_row_count") or 0) <= 0:
        raise ValueError("heldout_row_count must be positive when ready")
    if int(artifact.get("cross_environment_row_count") or 0) <= 0:
        raise ValueError("cross_environment_row_count must be positive when ready")
    drift = list(artifact.get("drift_cases") or [])
    negative_count = int(artifact.get("negative_control_regression_count") or 0)
    if artifact.get("promotion_allowed") is True and (
        drift or negative_count or artifact.get("rollback_triggered") is True
    ):
        raise ValueError("promotion_allowed requires no drift, regressions, or rollback")
    if (drift or negative_count) and artifact.get("rollback_triggered") is not True:
        raise ValueError("rollback_triggered must be true when regressions are present")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3187 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(
    promotion_allowed: bool,
    rollback_triggered: bool,
    drift_count: int,
    negative_regression_count: int,
) -> str:
    """Return a terminal verdict that states the promotion boundary."""

    return (
        "complete: fr11 cross-environment drift replay v1 finished; "
        f"promotion_allowed={str(promotion_allowed).lower()}; "
        f"rollback_triggered={str(rollback_triggered).lower()}; "
        f"drift_case_count={drift_count}; "
        f"negative_control_regression_count={negative_regression_count}; "
        "no model-weight update claimed"
    )


def normalize_action(value: Any) -> str:
    """Normalize small action tokens used by controller replay rows."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def rate(numerator: int, denominator: int) -> float:
    """Return a stable fraction and avoid hidden zero-division policy."""

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
    """Return a file checksum when the replay source exists."""

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
