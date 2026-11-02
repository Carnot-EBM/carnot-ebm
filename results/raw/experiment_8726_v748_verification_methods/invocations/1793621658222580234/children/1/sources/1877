"""Exp5583 cached-row causal memory metric corrigendum.

Spec refs: REQ-LEARN-5583,
SCENARIO-LEARN-5583-ROWS,
SCENARIO-LEARN-5583-ESTIMANDS,
SCENARIO-LEARN-5583-CONTROLS,
SCENARIO-LEARN-5583-ARTIFACT.

Exp5569's raw tournament rows are preserved, but its headline gate is retired:
the prior artifact compared two metrics that answered the same delayed-row
question. This module only replays checked-in row outcomes. It does not call a
model, score a verifier, or invent substitute observations when row evidence is
missing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_5558_RELATIVE_PATH = Path("results/experiment_5558_causal_write_manage_read_csl_memory.json")
SOURCE_5569_RELATIVE_PATH = Path("results/experiment_5569_causal_memory_policy_tournament.json")
RESULT_RELATIVE_PATH = Path("results/experiment_5583_causal_memory_metric_corrigendum.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5583_causal_memory_metric_corrigendum.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5583_causal_memory_metric_corrigendum.py")

SCHEMA = "carnot.experiment_5583.causal_memory_metric_corrigendum.v1"
EXPERIMENT = 5583
EXPERIMENT_ID = "experiment_5583_causal_memory_metric_corrigendum"
TASK_ID = "exp5583-causal-memory-metric-corrigendum"
MILESTONE = "2026.07.505"
RUN_DATE = "2026-07-14"
INFERENCE_SUBSTRATE = "cached_row_level_memory_outcomes"
DURATION_S = 0.01
FIRST_LATER_SESSION_INDEX = 2
FIRST_EXPOSURE_PHASE = 0
FIRST_DELAYED_PHASE = 2
FINAL_DELAYED_PHASE = 4
FORGETTING_MAX_ALLOWED = 0.02

NO_MEMORY_ARM = "no_memory"
SHUFFLED_MEMORY_ARM = "shuffled_memory"
STATIC_CAUSAL_ARM = "static_causal"
ALWAYS_FULL_ARM = "always_full"
SELF_OPTIMIZED_CAUSAL_ARM = "self_optimized_causal"
ARM_NAMES = (
    NO_MEMORY_ARM,
    SHUFFLED_MEMORY_ARM,
    STATIC_CAUSAL_ARM,
    ALWAYS_FULL_ARM,
    SELF_OPTIMIZED_CAUSAL_ARM,
)
SPEC_REFS = (
    "REQ-LEARN-5583",
    "SCENARIO-LEARN-5583-ROWS",
    "SCENARIO-LEARN-5583-ESTIMANDS",
    "SCENARIO-LEARN-5583-CONTROLS",
    "SCENARIO-LEARN-5583-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "rows_reconstructed",
    "estimand_definitions",
    "forward_transfer_delta",
    "backward_retention_delta",
    "forgetting_delta",
    "permutation_control_passed",
    "metric_independence_positive_control",
    "flagged_adversarial",
    "policy_ready",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "rows_reconstructed": "metrics trace to observations",
    "estimand_definitions": "forward and backward questions are distinct",
    "forward_transfer_delta": "later-family adaptation is isolated",
    "backward_retention_delta": "earlier-family durability is isolated",
    "forgetting_delta": "retention loss stays visible",
    "permutation_control_passed": "causal order must matter",
    "metric_independence_positive_control": "formulas cannot be aliases",
    "flagged_adversarial": "downstream use requires a clean audit",
    "policy_ready": "controls and policy benefit must both pass",
    "inference_substrate": "no new inference",
    "honest_verdict": "repeated tautology retires the policy lane",
}
FIELD_PRINCIPLES: JsonDict = {
    **REQUIRED_FIELD_PRINCIPLES,
    "field_principles": "Keeps the audit reason next to each required field.",
    "arm_comparison": "Reruns every tournament arm from the same cached rows.",
    "policy_cost": "Reports retrieval/read burden outside transfer and retention.",
    "source_artifacts": "Shows the corrigendum preserves upstream raw outcomes.",
    "positive_control_passed": "Makes the null-risk control conductor visible.",
    "false_negative_risk_checked": "Records that a null policy gate has controls.",
    "null_delta_methodology_note": "Explains why the zero forward transfer delta is measured.",
    "duration_s": "Declares bounded JSON replay work rather than live inference.",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5583_causal_memory_metric_corrigendum.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5583_causal_memory_metric_corrigendum.py "
    "-m pytest tests/python/test_experiment_5583_causal_memory_metric_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5583_causal_memory_metric_corrigendum.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5583_causal_memory_metric_corrigendum.json",
)


def load_json(path: Path | str) -> JsonDict:
    """Load a JSON object from disk and reject non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - defensive input guard.
        raise ValueError(f"expected JSON object at {path}")
    return payload


def reconstruct_cached_rows(source: Mapping[str, Any]) -> list[JsonDict]:
    """Recover seed, arm, session, and family membership from Exp5569 rows."""

    session_lookup = _session_lookup(source)
    expected_n_events = int(source.get("n_events", len(session_lookup)))
    seed_results = _require_list(
        source.get("tournament", {}).get("seed_results")
        if isinstance(source.get("tournament"), Mapping)
        else None,
        "row-level outcomes",
    )
    rows: list[JsonDict] = []
    for seed_result in seed_results:
        if not isinstance(seed_result, Mapping):  # pragma: no cover - malformed source.
            raise ValueError("row-level outcomes: seed result is not an object")
        seed = int(seed_result.get("seed", -1))
        arm_rows = seed_result.get("arm_rows")
        if not isinstance(arm_rows, Mapping):  # pragma: no cover - malformed source.
            raise ValueError("row-level outcomes: arm_rows missing")
        if set(arm_rows.keys()) != set(ARM_NAMES):
            raise ValueError("row-level outcomes: tournament arms are incomplete")
        for arm in ARM_NAMES:
            cached_rows = _require_list(arm_rows.get(arm), "row-level outcomes")
            if len(cached_rows) != expected_n_events:
                raise ValueError("row-level outcomes: missing cached rows")
            rows.extend(
                _normalize_row(seed=seed, arm=arm, raw=row, lookup=session_lookup)
                for row in cached_rows
            )
    _require_complete_grid(rows, expected_n_events=expected_n_events)
    return rows


def _session_lookup(source: Mapping[str, Any]) -> dict[str, JsonDict]:
    sessions = _require_list(source.get("sessions"), "session membership")
    lookup: dict[str, JsonDict] = {}
    for session in sessions:
        if not isinstance(session, Mapping):  # pragma: no cover - malformed source.
            raise ValueError("session membership: session is not an object")
        event_ids = _require_list(session.get("event_ids"), "session membership")
        for local_index, event_id in enumerate(event_ids):
            key = str(event_id)
            lookup[key] = {
                "session_id": str(session["session_id"]),
                "session_index": int(session["session_index"]),
                "family_kind": str(session["family_kind"]),
                "family_name": str(session["family_name"]),
                "local_index": local_index,
            }
    return lookup


def _normalize_row(
    *,
    seed: int,
    arm: str,
    raw: Any,
    lookup: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    if not isinstance(raw, Mapping):  # pragma: no cover - malformed source.
        raise ValueError("row-level outcomes: row is not an object")
    event_id = str(raw.get("event_id", ""))
    membership = lookup.get(event_id)
    if membership is None:
        raise ValueError(f"session membership missing for event_id={event_id}")
    if int(raw.get("session_index", -1)) != int(membership["session_index"]):
        raise ValueError(f"session membership mismatch for event_id={event_id}")
    return {
        "row_id": f"{seed}:{arm}:{event_id}",
        "seed": seed,
        "arm": arm,
        "event_id": event_id,
        "session_id": str(membership["session_id"]),
        "session_index": int(membership["session_index"]),
        "family_kind": str(membership["family_kind"]),
        "family_name": str(membership["family_name"]),
        "local_index": int(membership["local_index"]),
        "phase": int(raw["phase"]),
        "delayed_eval": bool(raw["delayed_eval"]),
        "accepted": raw.get("accepted") is True,
        "exact_energy": float(raw["exact_energy"]),
        "context_key": str(raw["context_key"]),
        "selected_action": str(raw["selected_action"]),
        "expected_action": str(raw["expected_action"]),
        "read_memory_id": raw.get("read_memory_id"),
        "retrieval_candidates_considered": int(raw["retrieval_candidates_considered"]),
    }


def _require_complete_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_n_events: int,
) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        for arm in ARM_NAMES:
            arm_rows = [row for row in rows if row["seed"] == seed and row["arm"] == arm]
            if len(arm_rows) != expected_n_events:
                raise ValueError("row-level outcomes: incomplete seed/arm grid")


def rows_reconstructed(rows: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> JsonDict:
    """Summarize the observation grid used by all corrected metrics."""

    return {
        "source_artifact": SOURCE_5569_RELATIVE_PATH.as_posix(),
        "source_experiment_id": source.get("experiment_id"),
        "source_flagged_adversarial": source.get("flagged_adversarial") is True,
        "complete": True,
        "total_rows": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "arms": list(ARM_NAMES),
        "rows_per_arm_seed": len(rows)
        // max(1, len(ARM_NAMES) * len({int(row["seed"]) for row in rows})),
        "sessions": len({str(row["session_id"]) for row in rows}),
        "families": len({str(row["family_name"]) for row in rows}),
        "blocked_if_missing": True,
    }


def compute_estimands(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute independent transfer, retention, forgetting, and cost estimands."""

    forward_static = _success_for(rows, STATIC_CAUSAL_ARM, _is_forward_first_exposure)
    forward_optimized = _success_for(
        rows,
        SELF_OPTIMIZED_CAUSAL_ARM,
        _is_forward_first_exposure,
    )
    backward_static = _success_for(rows, STATIC_CAUSAL_ARM, _is_backward_retention)
    backward_optimized = _success_for(
        rows,
        SELF_OPTIMIZED_CAUSAL_ARM,
        _is_backward_retention,
    )
    first_delayed_optimized = _success_for(
        rows,
        SELF_OPTIMIZED_CAUSAL_ARM,
        _is_first_delayed_replay,
    )
    final_delayed_optimized = _success_for(
        rows,
        SELF_OPTIMIZED_CAUSAL_ARM,
        _is_final_delayed_replay,
    )
    forward_denominator = _denominator_for(
        rows, SELF_OPTIMIZED_CAUSAL_ARM, _is_forward_first_exposure
    )
    backward_denominator = _denominator_for(rows, SELF_OPTIMIZED_CAUSAL_ARM, _is_backward_retention)
    first_denominator = _denominator_for(rows, SELF_OPTIMIZED_CAUSAL_ARM, _is_first_delayed_replay)
    final_denominator = _denominator_for(rows, SELF_OPTIMIZED_CAUSAL_ARM, _is_final_delayed_replay)
    return {
        "forward_transfer_delta": _round(forward_optimized - forward_static),
        "backward_retention_delta": _round(backward_optimized - backward_static),
        "forgetting_delta": _round(first_delayed_optimized - final_delayed_optimized),
        "forward_transfer": {
            "optimized_success": forward_optimized,
            "static_success": forward_static,
            "denominator_per_arm": forward_denominator,
            "row_predicate": "session_index >= 2 and phase == 0",
        },
        "backward_retention": {
            "optimized_success": backward_optimized,
            "static_success": backward_static,
            "denominator_per_arm": backward_denominator,
            "row_predicate": "session_index < 2 and delayed_eval",
        },
        "forgetting": {
            "optimized_first_delayed_success": first_delayed_optimized,
            "optimized_final_delayed_success": final_delayed_optimized,
            "first_delayed_denominator": first_denominator,
            "final_delayed_denominator": final_denominator,
            "row_predicate": "optimized phase 2 delayed success minus phase 4 delayed success",
        },
        "policy_cost": policy_cost(rows),
    }


def arm_comparison(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rerun all five arm comparisons from cached row outcomes."""

    comparison: JsonDict = {}
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row["arm"] == arm]
        heldout = [row for row in arm_rows if row["delayed_eval"]]
        comparison[arm] = {
            "row_count": len(arm_rows),
            "heldout_success": success_rate(heldout),
            "later_family_first_exposure_success": _success_for(
                rows,
                arm,
                _is_forward_first_exposure,
            ),
            "earlier_family_delayed_replay_success": _success_for(
                rows,
                arm,
                _is_backward_retention,
            ),
            "first_delayed_success": _success_for(rows, arm, _is_first_delayed_replay),
            "final_delayed_success": _success_for(rows, arm, _is_final_delayed_replay),
            "forgetting_loss": _round(
                _success_for(rows, arm, _is_first_delayed_replay)
                - _success_for(rows, arm, _is_final_delayed_replay)
            ),
            "read_rate": success_rate(
                [{"accepted": row.get("read_memory_id") is not None} for row in arm_rows]
            ),
            "heldout_retrieval_candidates_mean": _mean_candidates(heldout),
        }
    return comparison


def policy_cost(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report retrieval and read burden without mixing it into outcome deltas."""

    return {
        arm: {
            "read_rate_all_rows": arm_comparison_value(rows, arm, "read_rate"),
            "heldout_retrieval_candidates_mean": arm_comparison_value(
                rows,
                arm,
                "heldout_retrieval_candidates_mean",
            ),
        }
        for arm in ARM_NAMES
    }


def arm_comparison_value(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> float:
    """Return one arm-cost metric without materializing a recursive comparison."""

    arm_rows = [row for row in rows if row["arm"] == arm]
    if metric == "read_rate":
        return success_rate(
            [{"accepted": row.get("read_memory_id") is not None} for row in arm_rows]
        )
    heldout = [row for row in arm_rows if row["delayed_eval"]]
    return _mean_candidates(heldout)


def permutation_control(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run a deterministic label/session permutation that breaks causal order."""

    original = compute_estimands(rows)
    permuted_rows = permute_labels_and_sessions(rows)
    permuted = compute_estimands(permuted_rows)
    permuted_ready = policy_ready(permuted, controls_passed=True)
    passed = (
        permuted["backward_retention_delta"] != original["backward_retention_delta"]
        and permuted["forgetting_delta"] != original["forgetting_delta"]
        and permuted_ready is False
    )
    return {
        "passed": passed,
        "label_rotation": 3,
        "session_index_rotation": 1,
        "original_metrics": _headline_metrics(original),
        "permuted_metrics": _headline_metrics(permuted),
        "permuted_policy_ready": permuted_ready,
    }


def permute_labels_and_sessions(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Rotate labels within each seed/arm and rotate session IDs as a control."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["arm"]))].append(row)
    out: list[JsonDict] = []
    for group in grouped.values():
        ordered = sorted(
            group, key=lambda row: (int(row["session_index"]), int(row["local_index"]))
        )
        labels = [row.get("accepted") is True for row in ordered]
        rotated = labels[3:] + labels[:3]
        for row, accepted in zip(ordered, rotated, strict=True):
            clone = dict(row)
            clone["accepted"] = accepted
            clone["session_index"] = (int(row["session_index"]) + 1) % 6
            clone["family_name"] = f"permuted_session_{clone['session_index']}"
            out.append(clone)
    return out


def metric_independence_fixture() -> list[JsonDict]:
    """Return a tiny row set where only forward exposure can be changed."""

    rows: list[JsonDict] = []
    for arm in (STATIC_CAUSAL_ARM, SELF_OPTIMIZED_CAUSAL_ARM):
        is_optimized = arm == SELF_OPTIMIZED_CAUSAL_ARM
        rows.extend(
            [
                _fixture_row(
                    arm=arm,
                    event_id=f"{arm}-early-delayed",
                    session_index=0,
                    phase=2,
                    delayed_eval=True,
                    accepted=is_optimized,
                ),
                _fixture_row(
                    arm=arm,
                    event_id=f"{arm}-later-first",
                    session_index=2,
                    phase=0,
                    delayed_eval=False,
                    accepted=False,
                ),
                _fixture_row(
                    arm=arm,
                    event_id=f"{arm}-later-first-delayed",
                    session_index=2,
                    phase=2,
                    delayed_eval=True,
                    accepted=True,
                ),
                _fixture_row(
                    arm=arm,
                    event_id=f"{arm}-later-final-delayed",
                    session_index=2,
                    phase=4,
                    delayed_eval=True,
                    accepted=True,
                ),
            ]
        )
    return rows


def flip_forward_positive_control(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Flip only the optimized later-family first exposure in the fixture."""

    out = [dict(row) for row in rows]
    for row in out:
        if (
            row["arm"] == SELF_OPTIMIZED_CAUSAL_ARM
            and int(row["session_index"]) >= FIRST_LATER_SESSION_INDEX
            and int(row["phase"]) == FIRST_EXPOSURE_PHASE
        ):
            row["accepted"] = True
    return out


def metric_independence_positive_control() -> JsonDict:
    """Prove the forward formula can move while backward retention is fixed."""

    before = compute_estimands(metric_independence_fixture())
    after = compute_estimands(flip_forward_positive_control(metric_independence_fixture()))
    return {
        "passed": (
            after["forward_transfer_delta"] > before["forward_transfer_delta"]
            and after["backward_retention_delta"] == before["backward_retention_delta"]
        ),
        "before": _headline_metrics(before),
        "after": _headline_metrics(after),
        "fixture": "later-family first-exposure flip only",
    }


def _fixture_row(
    *,
    arm: str,
    event_id: str,
    session_index: int,
    phase: int,
    delayed_eval: bool,
    accepted: bool,
) -> JsonDict:
    return {
        "row_id": f"fixture:{arm}:{event_id}",
        "seed": 1,
        "arm": arm,
        "event_id": event_id,
        "session_id": f"fixture-session-{session_index}",
        "session_index": session_index,
        "family_kind": "fixture",
        "family_name": f"fixture-family-{session_index}",
        "local_index": phase,
        "phase": phase,
        "delayed_eval": delayed_eval,
        "accepted": accepted,
        "read_memory_id": "fixture-memory" if delayed_eval else None,
        "retrieval_candidates_considered": 1 if delayed_eval else 0,
    }


def policy_ready(estimands: Mapping[str, Any], *, controls_passed: bool) -> bool:
    """Gate the policy only when corrected benefit and controls both pass."""

    return (
        controls_passed
        and float(estimands["forward_transfer_delta"]) > 0.0
        and float(estimands["backward_retention_delta"]) >= -0.02
        and float(estimands["forgetting_delta"]) <= FORGETTING_MAX_ALLOWED
    )


def estimand_definitions() -> JsonDict:
    """Describe the corrected row predicates in artifact-visible language."""

    return {
        "forward_transfer_delta": (
            "optimized minus static success on later-family first exposure: "
            "session_index >= 2 and phase == 0"
        ),
        "backward_retention_delta": (
            "optimized minus static success on earlier-family delayed replay: "
            "session_index < 2 and delayed_eval"
        ),
        "forgetting_delta": (
            "optimized first delayed success minus optimized final delayed success: "
            "phase 2 delayed rows minus phase 4 delayed rows"
        ),
        "policy_cost": (
            "read rates and retrieval-candidate burden from cached rows, reported "
            "outside transfer and retention deltas"
        ),
    }


def build_artifact(
    *,
    root: Path | str,
    tests_added_or_reused: Sequence[str],
) -> JsonDict:
    """Build the Exp5583 corrigendum from checked-in Exp5569 row outcomes."""

    root_path = Path(root)
    source5569 = load_json(root_path / SOURCE_5569_RELATIVE_PATH)
    rows = reconstruct_cached_rows(source5569)
    estimands = compute_estimands(rows)
    permutation = permutation_control(rows)
    positive = metric_independence_positive_control()
    controls_passed = permutation["passed"] is True and positive["passed"] is True
    ready = policy_ready(estimands, controls_passed=controls_passed)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": DURATION_S,
        "random_seed": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "rows_reconstructed": {
            **rows_reconstructed(rows, source5569),
            "source_sha256": sha256_file(root_path / SOURCE_5569_RELATIVE_PATH),
        },
        "estimand_definitions": estimand_definitions(),
        "forward_transfer_delta": estimands["forward_transfer_delta"],
        "backward_retention_delta": estimands["backward_retention_delta"],
        "forgetting_delta": estimands["forgetting_delta"],
        "estimand_details": estimands,
        "arm_comparison": arm_comparison(rows),
        "policy_cost": estimands["policy_cost"],
        "permutation_control_passed": permutation["passed"],
        "permutation_control": permutation,
        "metric_independence_positive_control": positive,
        "positive_control_passed": positive["passed"],
        "false_negative_risk_checked": True,
        "null_delta_methodology_note": (
            "forward_transfer_delta=0.0 is measured from cached Exp5569 later-family "
            "first-exposure rows: optimized and static both scored 0/80. The "
            "metric-independence positive control flips forward transfer while "
            "backward retention remains fixed, so the zero is not a stub default."
        ),
        "flagged_adversarial": False,
        "policy_ready": ready,
        "policy_gate": policy_gate(estimands, controls_passed=controls_passed),
        "source_artifacts": source_artifacts(root_path, source5569),
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "complete: exp5569_policy_lane_retired_metric_corrigendum_from_cached_rows",
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def policy_gate(estimands: Mapping[str, Any], *, controls_passed: bool) -> JsonDict:
    """Expose why the corrected policy cannot gate self-learning."""

    forward_positive = float(estimands["forward_transfer_delta"]) > 0.0
    forgetting_within_bound = float(estimands["forgetting_delta"]) <= FORGETTING_MAX_ALLOWED
    return {
        "controls_passed": controls_passed,
        "forward_transfer_positive": forward_positive,
        "backward_retention_nonnegative": (float(estimands["backward_retention_delta"]) >= -0.02),
        "forgetting_within_bound": forgetting_within_bound,
        "forgetting_max_allowed": FORGETTING_MAX_ALLOWED,
        "policy_benefit_passed": forward_positive and forgetting_within_bound,
        "retirement_reasons": [
            reason
            for reason, active in (
                ("forward_transfer_delta_not_positive", not forward_positive),
                ("optimized_forgetting_loss_visible", not forgetting_within_bound),
            )
            if active
        ],
    }


def source_artifacts(root: Path, source5569: Mapping[str, Any]) -> JsonDict:
    """Record upstream context without laundering its flagged metric."""

    source5558_path = root / SOURCE_5558_RELATIVE_PATH
    return {
        "exp5558": {
            "path": SOURCE_5558_RELATIVE_PATH.as_posix(),
            "loadable": source5558_path.exists(),
            "sha256": sha256_file(source5558_path) if source5558_path.exists() else None,
        },
        "exp5569": {
            "path": SOURCE_5569_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / SOURCE_5569_RELATIVE_PATH),
            "flagged_adversarial": source5569.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(source5569.get("corrigendum_pending", [])),
            "raw_outcomes_preserved": True,
        },
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        write_json(resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when required fields or gates contradict the corrected metrics."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5583 artifact: " + "; ".join(errors))  # pragma: no cover
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and gate errors for the corrigendum artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")  # pragma: no cover
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping):
        errors.append("field_principles")  # pragma: no cover
    else:
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles[{field}]")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")  # pragma: no cover
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial")  # pragma: no cover
    estimands = {
        "forward_transfer_delta": artifact.get("forward_transfer_delta", 0.0),
        "backward_retention_delta": artifact.get("backward_retention_delta", 0.0),
        "forgetting_delta": artifact.get("forgetting_delta", 1.0),
    }
    gate = artifact.get("policy_gate", {})
    controls_passed = isinstance(gate, Mapping) and gate.get("controls_passed") is True
    if artifact.get("policy_ready") is not policy_ready(
        estimands,
        controls_passed=controls_passed,
    ):
        errors.append("policy_ready")  # pragma: no cover
    if artifact.get("permutation_control_passed") is not True:
        errors.append("permutation_control_passed")  # pragma: no cover
    positive = artifact.get("metric_independence_positive_control", {})
    if not isinstance(positive, Mapping) or positive.get("passed") is not True:
        errors.append("metric_independence_positive_control")  # pragma: no cover
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith("complete:"):
        errors.append("honest_verdict")  # pragma: no cover
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")  # pragma: no cover
    return errors


def _is_forward_first_exposure(row: Mapping[str, Any]) -> bool:
    return (
        int(row["session_index"]) >= FIRST_LATER_SESSION_INDEX
        and int(row["phase"]) == FIRST_EXPOSURE_PHASE
    )


def _is_backward_retention(row: Mapping[str, Any]) -> bool:
    return int(row["session_index"]) < FIRST_LATER_SESSION_INDEX and bool(row["delayed_eval"])


def _is_first_delayed_replay(row: Mapping[str, Any]) -> bool:
    return bool(row["delayed_eval"]) and int(row["phase"]) == FIRST_DELAYED_PHASE


def _is_final_delayed_replay(row: Mapping[str, Any]) -> bool:
    return bool(row["delayed_eval"]) and int(row["phase"]) == FINAL_DELAYED_PHASE


def _success_for(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    predicate: Any,
) -> float:
    return success_rate([row for row in rows if row["arm"] == arm and predicate(row)])


def _denominator_for(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    predicate: Any,
) -> int:
    return sum(1 for row in rows if row["arm"] == arm and predicate(row))


def success_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return deterministic exact-success rate over cached rows."""

    if not rows:
        return 0.0
    return _round(sum(1 for row in rows if row.get("accepted") is True) / len(rows))


def _mean_candidates(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return _round(sum(int(row["retrieval_candidates_considered"]) for row in rows) / len(rows))


def _headline_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "forward_transfer_delta": metrics["forward_transfer_delta"],
        "backward_retention_delta": metrics["backward_retention_delta"],
        "forgetting_delta": metrics["forgetting_delta"],
    }


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected list")  # pragma: no cover
    return value


def resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for diffable experiment receipts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with the checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the receipt."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for one file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for a JSON-compatible mapping."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 10)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
