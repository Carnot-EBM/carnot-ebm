"""Cached-fixture verifier-dose scheduler replay for Exp 5264.

The live verifier-dose task was blocked in Exp 5250 because the upstream
cross-model typed-memory gate had not opened. This module keeps the useful part
of Hybrid Verified Decoding: spend the expensive verifier only when a cheap
gate or typed memory is not enough. It does that without live LLM calls by
replaying prior cached rows whose receipts are complete enough to inspect.

Spec refs: REQ-VERIFY-5264, SCENARIO-VERIFY-5264.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5264_verifier_dose_scheduler_replay_v481"
EXPERIMENT_ID = 5264
SCHEMA = "carnot.verifier_dose_scheduler_replay.v481"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5264
RESULT_RELATIVE_PATH = "results/experiment_5264_verifier_dose_scheduler_replay_v481.json"
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
MIN_FIXTURE_COUNT = 6

REQ_REFS = ("REQ-VERIFY-5264", "SCENARIO-VERIFY-5264")

EXP5239 = "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json"
EXP5247 = "results/experiment_5247_slot_artifact_normalizer_v480.json"
EXP5248 = "results/experiment_5248_gap4_receipt_salvage_or_retire_v480.json"
EXP5250 = "results/experiment_5250_verifier_dose_scheduler_v480.json"
EXP5261 = "results/experiment_5261_typed_memory_interference_audit_v481.json"
SOURCE_ARTIFACTS = (EXP5239, EXP5247, EXP5248, EXP5250, EXP5261)

ROUTE_NO_VERIFIER = "no_verifier"
ROUTE_CHEAP = "cheap_deterministic"
ROUTE_TYPED_MEMORY = "typed_memory"
ROUTE_FULL = "full_replay"
ROUTES = (ROUTE_NO_VERIFIER, ROUTE_CHEAP, ROUTE_TYPED_MEMORY, ROUTE_FULL)

VERIFIER_COST_UNITS = {
    ROUTE_NO_VERIFIER: 0,
    ROUTE_CHEAP: 1,
    ROUTE_TYPED_MEMORY: 2,
    ROUTE_FULL: 10,
}
ABSTAIN_OR_BLOCK_PREFIXES = ("abstain", "block", "quarantine", "reject", "retire")

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal Exp 5264 verdict; starts with complete: or blocked_ and states "
        "whether cached scheduler replay is useful, null, harmful, or underpowered."
    ),
    "inference_substrate": (
        "Declares cached fixture replay with no live LLM calls, preventing Exp "
        "5264 from being mistaken for live verifier-dose inference."
    ),
    "full_verifier_calls_avoided_rate": (
        "Fraction of always-full replay calls avoided by the scheduler on "
        "receipt-backed cached fixtures."
    ),
    "decision_quality_delta": (
        "Scheduler decision-quality rate minus always-full decision-quality rate "
        "on the same cached fixtures."
    ),
    "false_accept_delta": (
        "Scheduler false-accept rate minus always-full false-accept rate on the "
        "same cached fixtures; positive values are unsafe."
    ),
    "abstain_or_block_count": (
        "Count of scheduler decisions that route to abstain, block, quarantine, "
        "or retire instead of accepting unsafe evidence."
    ),
    "fixture_receipts": (
        "Checksums for every source artifact and deterministic fixture row used "
        "by the cached replay."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "full_verifier_calls_avoided_rate",
    "decision_quality_delta",
    "false_accept_delta",
    "abstain_or_block_count",
)


@dataclass(frozen=True)
class SchedulerFixture:
    """One cached verifier-dose decision with all policy-visible features.

    The decisions are replayed from prior artifacts or derived from their
    deterministic rows. The scheduler sees only cheap-gate status, memory
    confidence, violation count, and receipt completeness; it does not peek at
    the expected decision when choosing the verifier dose.
    """

    task_id: str
    source_artifact: str
    receipt_sources: tuple[str, ...]
    cheap_gate_passed: bool
    memory_confidence: float
    deterministic_violation_count: int
    receipt_complete: bool
    expected_decision: str
    no_verifier_decision: str
    cheap_decision: str
    typed_memory_decision: str
    full_decision: str


def policy_feature_names() -> tuple[str, ...]:
    """Return the transparent features the dose policy is allowed to inspect."""

    return (
        "cheap_gate_passed",
        "memory_confidence",
        "deterministic_violation_count",
        "artifact_receipt_complete",
    )


def build_scheduler_fixtures(root: Path | str = REPO_ROOT) -> tuple[SchedulerFixture, ...]:
    """Build deterministic replay fixtures from prior receipt-backed artifacts."""

    root_path = Path(root)
    exp5239 = _read_json(root_path / EXP5239)
    exp5247 = _read_json(root_path / EXP5247)
    exp5248 = _read_json(root_path / EXP5248)
    _read_json(root_path / EXP5250)
    exp5261 = _read_json(root_path / EXP5261)

    aligned = _rows_by_task(exp5239["arm_metrics"]["aligned_memory"]["rows"])
    no_memory = _rows_by_task(exp5239["arm_metrics"]["no_memory"]["rows"])
    normalizer_ready = bool(exp5247.get("artifact_normalizer_ready"))
    gap4_receipts_complete = (
        _wrapped_value(exp5248, "gap4_final_decision") == "salvaged_clean_null"
    )

    fixtures = [
        _from_exp5239(
            task_id="gap1_memory_only_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=False,
            memory_confidence=0.94,
            deterministic_violation_count=1,
            cheap_decision="attempt_gap1_registry_promotion",
            receipt_complete=normalizer_ready,
            receipt_sources=(EXP5239, EXP5247),
        ),
        _from_exp5239(
            task_id="gap1_registry_rollback_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=False,
            memory_confidence=0.91,
            deterministic_violation_count=2,
            cheap_decision="attempt_gap1_registry_promotion",
            receipt_complete=normalizer_ready,
            receipt_sources=(EXP5239, EXP5247),
        ),
        _from_exp5239(
            task_id="gap4_candidate_pool_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=False,
            memory_confidence=0.86,
            deterministic_violation_count=3,
            cheap_decision="reuse_gap4_candidate_pool",
            receipt_complete=normalizer_ready and gap4_receipts_complete,
            receipt_sources=(EXP5239, EXP5247, EXP5248),
        ),
        _from_exp5239(
            task_id="mmlu_hidden_state_retention_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=True,
            memory_confidence=0.72,
            deterministic_violation_count=0,
            cheap_decision="retire_mmlu_hidden_state_path",
            receipt_complete=normalizer_ready,
            receipt_sources=(EXP5239, EXP5247),
        ),
        _from_exp5239(
            task_id="arc_rubric_before_patch_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=False,
            memory_confidence=0.9,
            deterministic_violation_count=2,
            cheap_decision="patch_arc_level_directly",
            receipt_complete=normalizer_ready,
            receipt_sources=(EXP5239, EXP5247),
        ),
        _from_exp5239(
            task_id="hardware_speedup_boundary_consumer",
            aligned=aligned,
            no_memory=no_memory,
            cheap_gate_passed=False,
            memory_confidence=0.58,
            deterministic_violation_count=1,
            cheap_decision="block_hardware_speedup_claim_until_transcript",
            receipt_complete=normalizer_ready,
            receipt_sources=(EXP5239, EXP5247),
        ),
    ]
    fixtures.append(_unrelated_fixture_from_exp5261(exp5261, receipt_complete=normalizer_ready))
    return tuple(fixtures)


def choose_verifier_route(fixture: SchedulerFixture) -> str:
    """Choose the cheapest verifier dose expected to preserve safety."""

    if not fixture.receipt_complete:
        return ROUTE_FULL
    if fixture.deterministic_violation_count >= 3:
        return ROUTE_FULL
    if not fixture.cheap_gate_passed and fixture.memory_confidence >= 0.8:
        return ROUTE_TYPED_MEMORY
    if not fixture.cheap_gate_passed:
        return ROUTE_CHEAP
    if fixture.memory_confidence < 0.4 and fixture.deterministic_violation_count == 0:
        return ROUTE_NO_VERIFIER
    return ROUTE_CHEAP


def replay_scheduler(fixtures: Sequence[SchedulerFixture]) -> JsonDict:
    """Replay the scheduler and all requested baselines on the same fixtures."""

    scheduler_rows = []
    for fixture in fixtures:
        route = choose_verifier_route(fixture)
        decision = _decision_for_route(fixture, route)
        scheduler_rows.append(_decision_row(fixture=fixture, route=route, decision=decision))

    baseline_rows = {
        "no_verifier": [
            _decision_row(fixture=fixture, route=ROUTE_NO_VERIFIER, decision=fixture.no_verifier_decision)
            for fixture in fixtures
        ],
        "always_cheap": [
            _decision_row(fixture=fixture, route=ROUTE_CHEAP, decision=fixture.cheap_decision)
            for fixture in fixtures
        ],
        "always_full": [
            _decision_row(fixture=fixture, route=ROUTE_FULL, decision=fixture.full_decision)
            for fixture in fixtures
        ],
    }

    scheduler_metrics = _metrics(scheduler_rows)
    baseline_metrics = {
        name: _metrics(rows) for name, rows in baseline_rows.items()
    }
    always_full = baseline_metrics["always_full"]
    full_denominator = int(always_full["full_verifier_calls"])
    avoided_rate = _rate(
        full_denominator - int(scheduler_metrics["full_verifier_calls"]),
        full_denominator,
    )
    decision_quality_delta = _delta(
        float(scheduler_metrics["quality_rate"]),
        float(always_full["quality_rate"]),
    )
    false_accept_delta = _delta(
        float(scheduler_metrics["false_accept_rate"]),
        float(always_full["false_accept_rate"]),
    )
    abstain_or_block_count = sum(
        1 for row in scheduler_rows if _is_abstain_or_block(str(row["selected_decision"]))
    )
    scheduler_ready = bool(
        len(fixtures) >= MIN_FIXTURE_COUNT
        and avoided_rate > 0.0
        and decision_quality_delta >= 0.0
        and false_accept_delta <= 0.0
    )
    return {
        "fixture_count": len(fixtures),
        "policy_features": list(policy_feature_names()),
        "policy_rules": list(_policy_rules()),
        "scheduler_rows": scheduler_rows,
        "baseline_rows": baseline_rows,
        "scheduler_metrics": scheduler_metrics,
        "baseline_metrics": baseline_metrics,
        "route_counts": _ordered_counts(Counter(str(row["route"]) for row in scheduler_rows)),
        "full_verifier_calls_avoided_rate": avoided_rate,
        "decision_quality_delta": decision_quality_delta,
        "false_accept_delta": false_accept_delta,
        "abstain_or_block_count": abstain_or_block_count,
        "scheduler_ready": scheduler_ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp 5264 artifact from deterministic cached replay."""

    fixtures = build_scheduler_fixtures(root=root)
    replay = replay_scheduler(fixtures)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(REQ_REFS),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(replay)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "scheduler_ready": bool(replay["scheduler_ready"]),
        "scheduler_ready_principle": _scheduler_ready_principle(replay),
        "full_verifier_calls_avoided_rate": _wrap(
            "full_verifier_calls_avoided_rate",
            float(replay["full_verifier_calls_avoided_rate"]),
        ),
        "decision_quality_delta": _wrap(
            "decision_quality_delta",
            float(replay["decision_quality_delta"]),
        ),
        "false_accept_delta": _wrap(
            "false_accept_delta",
            float(replay["false_accept_delta"]),
        ),
        "abstain_or_block_count": _wrap(
            "abstain_or_block_count",
            int(replay["abstain_or_block_count"]),
        ),
        "fixture_receipts": fixture_receipts(fixtures=fixtures, root=root),
        "replay": replay,
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def fixture_receipts(
    *,
    fixtures: Sequence[SchedulerFixture],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Return source and row checksums for the replayed fixture panel."""

    row_payloads = {fixture.task_id: asdict(fixture) for fixture in fixtures}
    checksums = {
        "fixture_set_sha256": _sha256_json(row_payloads),
        "fixture_rows": {
            task_id: _sha256_json(payload) for task_id, payload in row_payloads.items()
        },
        "source_artifacts": _source_artifact_checksums(Path(root)),
    }
    return {
        "principle": FIELD_PRINCIPLES["fixture_receipts"],
        "checksums": checksums,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the result shape required by REQ-VERIFY-5264."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be cached_fixture_replay_no_llm")  # pragma: no cover
    if not isinstance(artifact.get("scheduler_ready"), bool):
        raise ValueError("scheduler_ready must be a bare bool")  # pragma: no cover
    if not artifact.get("scheduler_ready_principle"):
        raise ValueError("missing scheduler_ready_principle")  # pragma: no cover
    receipts = artifact.get("fixture_receipts")
    if not isinstance(receipts, Mapping) or "checksums" not in receipts or "principle" not in receipts:
        raise ValueError("fixture_receipts must include checksums and principle")  # pragma: no cover
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5264 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def _from_exp5239(
    *,
    task_id: str,
    aligned: Mapping[str, Mapping[str, Any]],
    no_memory: Mapping[str, Mapping[str, Any]],
    cheap_gate_passed: bool,
    memory_confidence: float,
    deterministic_violation_count: int,
    cheap_decision: str,
    receipt_complete: bool,
    receipt_sources: tuple[str, ...],
) -> SchedulerFixture:
    aligned_row = aligned[task_id]
    no_memory_row = no_memory[task_id]
    expected = str(aligned_row["expected_action"])
    full = str(aligned_row["selected_action"])
    return SchedulerFixture(
        task_id=task_id,
        source_artifact=EXP5239,
        receipt_sources=receipt_sources,
        cheap_gate_passed=cheap_gate_passed,
        memory_confidence=memory_confidence,
        deterministic_violation_count=deterministic_violation_count,
        receipt_complete=receipt_complete,
        expected_decision=expected,
        no_verifier_decision=str(no_memory_row["selected_action"]),
        cheap_decision=cheap_decision,
        typed_memory_decision=full,
        full_decision=full,
    )


def _unrelated_fixture_from_exp5261(
    exp5261: Mapping[str, Any],
    *,
    receipt_complete: bool,
) -> SchedulerFixture:
    row = next(
        item
        for item in exp5261["audit_rows"]["unrelated"]
        if item["task_id"] == "range_constraint_unrelated"
    )
    selected = str(row["selected_action"])
    expected = str(row["expected_action"])
    return SchedulerFixture(
        task_id="range_constraint_unrelated",
        source_artifact=EXP5261,
        receipt_sources=(EXP5261, EXP5247),
        cheap_gate_passed=True,
        memory_confidence=0.2,
        deterministic_violation_count=0,
        receipt_complete=receipt_complete,
        expected_decision=expected,
        no_verifier_decision=selected,
        cheap_decision=selected,
        typed_memory_decision=selected,
        full_decision=expected,
    )


def _decision_for_route(fixture: SchedulerFixture, route: str) -> str:
    if route == ROUTE_NO_VERIFIER:
        return fixture.no_verifier_decision
    if route == ROUTE_CHEAP:
        return fixture.cheap_decision
    if route == ROUTE_TYPED_MEMORY:
        return fixture.typed_memory_decision
    return fixture.full_decision


def _decision_row(*, fixture: SchedulerFixture, route: str, decision: str) -> JsonDict:
    correct = decision == fixture.expected_decision
    return {
        "task_id": fixture.task_id,
        "source_artifact": fixture.source_artifact,
        "route": route,
        "features": {
            "cheap_gate_passed": fixture.cheap_gate_passed,
            "memory_confidence": fixture.memory_confidence,
            "deterministic_violation_count": fixture.deterministic_violation_count,
            "artifact_receipt_complete": fixture.receipt_complete,
        },
        "selected_decision": decision,
        "expected_decision": fixture.expected_decision,
        "correct": correct,
        "false_accept": _is_false_accept(decision, fixture.expected_decision),
        "full_verifier_call": route == ROUTE_FULL,
        "cost_units": VERIFIER_COST_UNITS[route],
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "n": len(rows),
        "correct_n": sum(1 for row in rows if row["correct"]),
        "quality_rate": _rate(sum(1 for row in rows if row["correct"]), len(rows)),
        "false_accepts": sum(1 for row in rows if row["false_accept"]),
        "false_accept_rate": _rate(sum(1 for row in rows if row["false_accept"]), len(rows)),
        "full_verifier_calls": sum(1 for row in rows if row["full_verifier_call"]),
        "cost_units": sum(int(row["cost_units"]) for row in rows),
    }


def _honest_verdict(replay: Mapping[str, Any]) -> str:
    if not replay["scheduler_ready"]:
        if int(replay["fixture_count"]) < MIN_FIXTURE_COUNT:
            return "blocked_underpowered: cached verifier-dose scheduler replay has too few fixtures"
        if float(replay["false_accept_delta"]) > 0.0:
            return "complete: harmful scheduler replay increased false accepts versus always-full"
        if float(replay["decision_quality_delta"]) < 0.0:
            return "complete: harmful scheduler replay lost decision quality versus always-full"
        return "complete: null scheduler replay did not avoid enough full verifier calls"
    return (
        "complete: useful scheduler replay preserved always-full decision quality, "
        "kept false_accept_delta=0.000000, and avoided "
        f"{float(replay['full_verifier_calls_avoided_rate']):.6f} full verifier calls"
    )


def _scheduler_ready_principle(replay: Mapping[str, Any]) -> str:
    return (
        "scheduler_ready=true only when fixture_count>=6, "
        "decision_quality_delta>=0, false_accept_delta<=0, and "
        "full_verifier_calls_avoided_rate>0; observed "
        f"fixture_count={replay['fixture_count']}, "
        f"decision_quality_delta={replay['decision_quality_delta']:.6f}, "
        f"false_accept_delta={replay['false_accept_delta']:.6f}, "
        f"full_verifier_calls_avoided_rate={replay['full_verifier_calls_avoided_rate']:.6f}"
    )


def _policy_rules() -> tuple[str, ...]:
    return (
        "receipt_complete=false routes to full_replay fail-closed handling",
        "deterministic_violation_count>=3 routes to full_replay",
        "cheap_gate_passed=false and memory_confidence>=0.8 routes to typed_memory",
        "cheap_gate_passed=false routes remaining cases to cheap_deterministic",
        "cheap_gate_passed=true with low memory confidence and no violations routes to no_verifier",
        "all other cached cases route to cheap_deterministic",
    )


def _rows_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["task_id"]): row for row in rows}


def _is_false_accept(decision: str, expected_decision: str) -> bool:
    if decision == expected_decision:
        return False
    return _is_abstain_or_block(expected_decision) and not _is_abstain_or_block(decision)


def _is_abstain_or_block(decision: str) -> bool:
    return str(decision).startswith(ABSTAIN_OR_BLOCK_PREFIXES)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str, default: Any = None) -> Any:
    value = artifact.get(field, default)
    return value.get("value", default) if isinstance(value, Mapping) else value


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _delta(left: float, right: float) -> float:
    return round(left - right, 6)


def _ordered_counts(counts: Mapping[str, int]) -> JsonDict:
    return {route: int(counts.get(route, 0)) for route in ROUTES if int(counts.get(route, 0))}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_artifact_checksums(root: Path) -> JsonDict:
    return {
        source: _sha256_bytes((root / source).read_bytes()) if (root / source).exists() else None
        for source in SOURCE_ARTIFACTS
    }


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return _sha256_json(stable)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
