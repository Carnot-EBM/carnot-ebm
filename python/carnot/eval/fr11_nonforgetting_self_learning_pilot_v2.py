"""Exp 3172 FR-11 nonforgetting self-learning pilot v2.

Spec refs: REQ-LEARN-3172, SCENARIO-LEARN-3172,
SCENARIO-LEARN-3172-BLOCKED.

The pilot tests the narrow learning mechanism that Exp 3171 prepared: a
controller can remember exact counterexample row ids whose ledger expected
rejection but whose observed decision accepted.  That is learning in controller
memory only.  This module deliberately avoids finetuning, hidden-state claims,
live model calls, and any use of held-out or negative-control rows to build the
update.
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
ARTIFACT = "experiment_3172_fr11_nonforgetting_self_learning_pilot_v2"
SCHEMA = "carnot.fr11.nonforgetting_self_learning_pilot.v2"
OUTPUT_REL_PATH = Path("results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json")
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3156_REL_PATH = Path("results/experiment_3156_fr11_ledger_consistency_closure_v1.json")
EXP3157_REL_PATH = Path("results/experiment_3157_fr11_attractor_residual_memory_audit_v1.json")
EXP3171_REL_PATH = Path("results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")

REQUIRED_ARTIFACT_FIELDS = {
    "fr11_nonforgetting_self_learning_pilot_v2_ready",
    "continuous_self_learning_task",
    "controller_memory_update_applied",
    "model_weight_update_claimed",
    "before_ledger_consistency_rate",
    "after_ledger_consistency_rate",
    "heldout_consistency_rate",
    "negative_control_regression_count",
    "nonforgetting_passed",
    "promotion_allowed",
    "promotion_recommendation",
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
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.py -q",
    ".venv/bin/coverage report --include='*/fr11_nonforgetting_self_learning_pilot_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3156_ledger_closure", EXP3156_REL_PATH, True),
    ("exp3157_attractor_residual_memory", EXP3157_REL_PATH, True),
    ("exp3171_counterexample_isolation", EXP3171_REL_PATH, True),
    (
        "exp3172_module",
        Path("python/carnot/eval/fr11_nonforgetting_self_learning_pilot_v2.py"),
        False,
    ),
    (
        "exp3172_tests",
        Path("tests/python/test_experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.py"),
        False,
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed when evidence is absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the checked-in artifacts used by the Exp 3172 replay."""

    root_path = Path(root)
    return {
        "exp3156": read_json_object(root_path / EXP3156_REL_PATH),
        "exp3157": read_json_object(root_path / EXP3157_REL_PATH),
        "exp3171": read_json_object(root_path / EXP3171_REL_PATH),
    }


def split_blocker(exp3171: Mapping[str, Any]) -> str:
    """Return why the Exp 3171 split cannot support nonforgetting replay."""

    if exp3171.get("fr11_ledger_counterexample_isolation_ready") is not True:
        return "exp3171_counterexample_split_missing_or_not_ready"
    split = exp3171.get("environment_variant_split")
    if not isinstance(split, Mapping):
        return "exp3171_environment_variant_split_missing"
    training = rows_from_split(split, "training_update_rows")
    heldout = rows_from_split(split, "held_out_replay_rows")
    negative = rows_from_split(split, "negative_control_rows")
    if not training:
        return "exp3171_training_update_rows_missing"
    if not heldout:
        return "exp3171_held_out_replay_rows_missing"
    if not negative:
        return "exp3171_negative_control_rows_missing"
    if row_id_set(training) & (row_id_set(heldout) | row_id_set(negative)):
        return "exp3171_training_rows_overlap_evaluation_controls"
    return ""


def rows_from_split(split: Mapping[str, Any], field: str) -> list[JsonDict]:
    """Return row dictionaries from one Exp 3171 split field."""

    rows = split.get(field, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def run_pilot(exp3171: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3172-2/3/4: update training rows and replay controls."""

    split = exp3171["environment_variant_split"]
    training = rows_from_split(split, "training_update_rows")
    heldout = rows_from_split(split, "held_out_replay_rows")
    negative = rows_from_split(split, "negative_control_rows")
    update = build_controller_memory_update(training)
    before_training = [replay_row(row, {}) for row in training]
    before_heldout = [replay_row(row, {}) for row in heldout]
    training_replay = [replay_row(row, update) for row in training]
    heldout_replay = [replay_row(row, update) for row in heldout]
    negative_before = [replay_row(row, {}) for row in negative]
    negative_replay = [replay_row(row, update) for row in negative]
    before_rows = before_training + before_heldout
    after_rows = training_replay + heldout_replay
    regressions = negative_control_regressions(negative_before, negative_replay)
    heldout_rate = consistency_rate(heldout_replay)
    after_rate = consistency_rate(after_rows)
    nonforgetting = heldout_rate == 1.0 and not regressions
    promotion_allowed = (
        bool(update["updated_rows"])
        and after_rate == 1.0
        and heldout_rate == 1.0
        and nonforgetting
        and not regressions
    )
    return {
        "controller_memory_update": update,
        "controller_memory_update_applied": bool(update["updated_rows"]),
        "before_ledger_consistency_rate": consistency_rate(before_rows),
        "after_ledger_consistency_rate": after_rate,
        "heldout_consistency_rate": heldout_rate,
        "negative_control_regression_count": len(regressions),
        "negative_control_regressions": regressions,
        "nonforgetting_passed": nonforgetting,
        "promotion_allowed": promotion_allowed,
        "promotion_recommendation": promotion_recommendation(
            True,
            after_rate,
            heldout_rate,
            len(regressions),
            nonforgetting,
            False,
        ),
        "training_replay_rows": training_replay,
        "heldout_replay_rows": heldout_replay,
        "negative_control_replay_rows": negative_replay,
        "replay_panel_count": len(after_rows),
        "training_update_row_count": len(training),
        "heldout_replay_row_count": len(heldout),
        "negative_control_row_count": len(negative),
    }


def build_controller_memory_update(training_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build row-bounded controller memory from training rows only."""

    overrides: dict[str, str] = {}
    updated_rows: list[JsonDict] = []
    for row in training_rows:
        summary = row_summary(row)
        row_id = summary["row_id"]
        expected = summary["expected_action"]
        if row_id and expected != "unknown" and expected == summary["ledger_action"]:
            overrides[row_id] = expected
            updated_rows.append(
                {
                    "row_id": row_id,
                    "fixture_family": summary["fixture_family"],
                    "update_type": "row_exact_action_override",
                    "before_observed_action": summary["observed_action"],
                    "after_observed_action": expected,
                    "bounded_to_training_row_id": True,
                    "model_weight_update": False,
                }
            )
    return {
        "update_policy": "exact_row_id_controller_memory_override_from_training_rows_only",
        "updated_row_count": len(updated_rows),
        "updated_rows": updated_rows,
        "row_action_overrides": overrides,
        "heldout_rows_used_for_update": False,
        "negative_control_rows_used_for_update": False,
        "model_weight_update": False,
    }


def replay_row(row: Mapping[str, Any], update: Mapping[str, Any]) -> JsonDict:
    """Replay one row after applying any exact-row controller override."""

    replay = row_summary(row)
    overrides = update.get("row_action_overrides", {})
    override = overrides.get(replay["row_id"]) if isinstance(overrides, Mapping) else None
    replay["pre_update_observed_action"] = replay["observed_action"]
    replay["controller_update_applied"] = override is not None
    if override is not None:
        replay["observed_action"] = normalize_action(override)
    replay["consistent"] = row_consistent(replay)
    return replay


def negative_control_regressions(
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """REQ-LEARN-3172-4: count negative controls worsened by the update."""

    regressions: list[JsonDict] = []
    for before, after in zip(before_rows, after_rows, strict=True):
        if before.get("consistent") and (
            not after.get("consistent")
            or normalize_action(after.get("observed_action"))
            != normalize_action(before.get("observed_action"))
        ):
            regressions.append(
                {
                    "row_id": str(after.get("row_id") or ""),
                    "before_observed_action": normalize_action(before.get("observed_action")),
                    "after_observed_action": normalize_action(after.get("observed_action")),
                }
            )
    return regressions


def row_summary(row: Mapping[str, Any]) -> JsonDict:
    """Return the replay fields needed to judge ledger consistency."""

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


def row_consistent(row: Mapping[str, Any]) -> bool:
    """Recompute consistency from expected, ledger, and observed actions."""

    expected = normalize_action(row.get("expected_action"))
    ledger = normalize_action(row.get("ledger_action"))
    observed = normalize_action(row.get("observed_action"))
    return observed not in {"missing", "unknown"} and expected == ledger == observed


def consistency_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the fraction of replay rows whose actions are consistent."""

    return rate(sum(1 for row in rows if row.get("consistent")), len(rows))


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3172 terminal artifact from checked-in replay evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = split_blocker(sources["exp3171"])
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run, sources)
        validate_artifact(artifact)
        return artifact
    pilot = run_pilot(sources["exp3171"])
    ready = bool(pilot["controller_memory_update_applied"]) and pilot["replay_panel_count"] > 0
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_nonforgetting_self_learning_pilot_v2_ready": ready,
        "continuous_self_learning_task": True,
        "controller_memory_update_applied": pilot["controller_memory_update_applied"],
        "model_weight_update_claimed": False,
        "before_ledger_consistency_rate": pilot["before_ledger_consistency_rate"],
        "after_ledger_consistency_rate": pilot["after_ledger_consistency_rate"],
        "heldout_consistency_rate": pilot["heldout_consistency_rate"],
        "negative_control_regression_count": pilot["negative_control_regression_count"],
        "nonforgetting_passed": pilot["nonforgetting_passed"],
        "promotion_allowed": pilot["promotion_allowed"],
        "promotion_recommendation": pilot["promotion_recommendation"],
        "controller_memory_update": pilot["controller_memory_update"],
        "training_replay_rows": pilot["training_replay_rows"],
        "heldout_replay_rows": pilot["heldout_replay_rows"],
        "negative_control_replay_rows": pilot["negative_control_replay_rows"],
        "negative_control_regressions": pilot["negative_control_regressions"],
        "replay_panel_count": pilot["replay_panel_count"],
        "training_update_row_count": pilot["training_update_row_count"],
        "heldout_replay_row_count": pilot["heldout_replay_row_count"],
        "negative_control_row_count": pilot["negative_control_row_count"],
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(),
        "precondition_checks": precondition_checks(sources),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(
            ready,
            pilot["promotion_allowed"],
            pilot["after_ledger_consistency_rate"],
            pilot["nonforgetting_passed"],
        ),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    start: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
    sources: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return a schema-complete blocked artifact when split evidence is missing."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_nonforgetting_self_learning_pilot_v2_ready": False,
        "continuous_self_learning_task": True,
        "controller_memory_update_applied": False,
        "model_weight_update_claimed": False,
        "before_ledger_consistency_rate": 0.0,
        "after_ledger_consistency_rate": 0.0,
        "heldout_consistency_rate": 0.0,
        "negative_control_regression_count": 0,
        "nonforgetting_passed": False,
        "promotion_allowed": False,
        "promotion_recommendation": "block_fr11_nonforgetting_pilot_precondition_failed",
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate(mode="blocked_precondition_check"),
        "precondition_checks": precondition_checks(sources),
        "controller_memory_update": {
            "update_policy": "blocked_precondition_check",
            "updated_row_count": 0,
            "updated_rows": [],
            "row_action_overrides": {},
            "model_weight_update": False,
        },
        "training_replay_rows": [],
        "heldout_replay_rows": [],
        "negative_control_replay_rows": [],
        "negative_control_regressions": [],
        "replay_panel_count": 0,
        "training_update_row_count": 0,
        "heldout_replay_row_count": 0,
        "negative_control_row_count": 0,
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
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
    """Build, validate, and write deterministic Exp 3172 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_checks(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Expose source readiness and split presence for auditability."""

    exp3171 = sources.get("exp3171", {})
    split = exp3171.get("environment_variant_split")
    return {
        "exp3156_present": bool(sources.get("exp3156")),
        "exp3157_present": bool(sources.get("exp3157")),
        "exp3171_counterexample_isolation_ready": exp3171.get(
            "fr11_ledger_counterexample_isolation_ready"
        )
        is True,
        "exp3171_environment_variant_split_present": isinstance(split, Mapping),
        "exp3171_split_blocker": split_blocker(exp3171),
    }


def promotion_recommendation(
    ready: bool,
    after_rate: float,
    heldout_rate: float,
    regression_count: int,
    nonforgetting_passed: bool,
    model_weight_update_claimed: bool,
) -> str:
    """REQ-LEARN-3172-5: apply the exact controller-memory promotion gate."""

    if not ready:
        return "block_fr11_nonforgetting_pilot_precondition_failed"
    if model_weight_update_claimed:
        return "block_fr11_promotion_model_weight_update_claimed"
    if after_rate < 1.0:
        return "block_fr11_promotion_until_ledger_consistency_reaches_1.0"
    if heldout_rate < 1.0:
        return "block_fr11_promotion_heldout_replay_regressed"
    if regression_count or not nonforgetting_passed:
        return "block_fr11_promotion_nonforgetting_failed"
    return "promote_controller_memory_update_only"


def inference_substrate(mode: str = "controller_memory_nonforgetting_replay") -> JsonDict:
    """Declare that the pilot uses replayed controller memory only."""

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
        "hidden_state_mutation_claimed": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for replay traceability."""

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
    """Raise when an Exp 3172 artifact violates the nonforgetting contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("model_weight_update_claimed") is not False:
        raise ValueError("model_weight_update_claimed must be false")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or any(
        substrate.get(flag) is True
        for flag in (
            "model_weight_mutation",
            "model_weight_training",
            "base_model_weights_updated",
            "kan_model_weight_training",
            "hidden_state_mutation_claimed",
        )
    ):
        raise ValueError("model_weight_mutation must remain false")
    if int(substrate.get("fresh_live_inference_calls") or 0) != 0:
        raise ValueError("fresh_live_inference_calls must remain zero")
    for field in (
        "before_ledger_consistency_rate",
        "after_ledger_consistency_rate",
        "heldout_consistency_rate",
    ):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and within [0, 1]")
    if artifact.get("fr11_nonforgetting_self_learning_pilot_v2_ready") is not True:
        return
    if artifact.get("controller_memory_update_applied") is not True:
        raise ValueError("controller_memory_update_applied must be true for readiness")
    if artifact.get("promotion_allowed") is True and not (
        float(artifact.get("after_ledger_consistency_rate", 0.0)) == 1.0
        and float(artifact.get("heldout_consistency_rate", 0.0)) == 1.0
        and int(artifact.get("negative_control_regression_count") or 0) == 0
        and artifact.get("nonforgetting_passed") is True
    ):
        raise ValueError("promotion_allowed requires perfect consistency and nonforgetting")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def honest_verdict(
    ready: bool,
    promotion_allowed: bool,
    after_rate: float,
    nonforgetting_passed: bool,
) -> str:
    """Return a conductor-compatible terminal verdict."""

    if ready:
        return (
            "complete: fr11 nonforgetting self-learning pilot v2 replay finished; "
            f"after_ledger_consistency_rate={round_float(after_rate)}; "
            f"nonforgetting_passed={str(nonforgetting_passed).lower()}; "
            f"promotion_allowed={str(promotion_allowed).lower()}; "
            "no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11 nonforgetting self-learning pilot split missing"


def row_id_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return non-empty row ids from row dictionaries."""

    return {str(row.get("row_id") or "") for row in rows if str(row.get("row_id") or "")}


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
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
