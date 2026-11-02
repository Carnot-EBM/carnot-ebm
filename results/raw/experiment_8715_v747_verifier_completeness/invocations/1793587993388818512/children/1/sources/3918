"""LTLZinc-style continual-learning adapter backed by the CerCE ledger.

Exp 1669 turns small finite temporal traces into MiniZinc-style constraint
cases and replays them through the pipeline CerCE ledger.  This is deliberate
bookkeeping only: it checks whether a proposed query-time memory update would
retain old temporal constraints without mutating model weights or training a
new verifier.

Spec: REQ-LEARN-1669, SCENARIO-LEARN-1669a.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

from carnot.pipeline.cerce_ledger import (
    MemoryPolicyUpdate,
    ReplayCase,
    evaluate_promotion_gate,
    stable_hash,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260510"
EXPERIMENT_ID = 1669
EXPERIMENT = "1669_ltlzinc_cerce_continual_learning_adapter"
OUTPUT_FILE = "experiment_1669_ltlzinc.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
SCHEMA = "carnot.pipeline_ltlzinc_adapter.v1"
POLICY_UPDATE_ID = "policy:fr11:ltlzinc:temporal-nonforgetting"
SPEC_TRACES = ("REQ-LEARN-1669", "SCENARIO-LEARN-1669a")
SUPPORTED_OPERATORS = ("always", "eventually", "next", "until")
DEFAULT_CASE_COUNT = 8

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "schema",
    "experiment_id",
    "ltlzinc_adapter_ready",
    "temporal_cases_generated",
    "cerce_ledger_ready",
    "promotion_gate_passed",
    "forgetting_rate",
    "cerce_nonforgetting_rate",
    "ledger_artifact",
    "case_results",
    "blockers",
    "tests_run",
    "honest_verdict",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path | PathLike[str], run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _write_json(path: str | Path | PathLike[str], artifact: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _normalize_trace(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, bool | int]]:
    normalized: list[dict[str, bool | int]] = []
    for index, step in enumerate(trace):
        row: dict[str, bool | int] = {"t": index}
        for key, value in step.items():
            if key != "t":
                row[str(key)] = bool(value)
        normalized.append(row)
    return normalized


def _ltl_formula(operator: str, signal: str, guard_signal: str | None) -> str:
    if operator == "always":
        return f"G {signal}"
    if operator == "eventually":
        return f"F {signal}"
    if operator == "next":
        return f"X {signal}"
    return f"{guard_signal} U {signal}"


def _minizinc_constraint(operator: str, signal: str, guard_signal: str | None) -> str:
    if operator == "always":
        return f"constraint forall(t in TIME)({signal}[t]);"
    if operator == "eventually":
        return f"constraint exists(t in TIME)({signal}[t]);"
    if operator == "next":
        return f"constraint {signal}[2];"
    return f"constraint exists(t in TIME)({signal}[t] /\\ forall(p in 1..t-1)({guard_signal}[p]));"


def make_temporal_case(
    case_id: str,
    operator: str,
    signal: str,
    trace: Sequence[Mapping[str, Any]],
    expected_satisfied: bool,
    *,
    guard_signal: str | None = None,
) -> JsonDict:
    """Build one deterministic LTLZinc-style temporal replay row."""

    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported temporal operator: {operator}")  # pragma: no cover
    guard = guard_signal if operator == "until" else None
    return {
        "case_id": case_id,
        "source": "experiment_1669_ltlzinc",
        "temporal_operator": operator,
        "signal": signal,
        "guard_signal": guard,
        "trace": _normalize_trace(trace),
        "expected_satisfied": bool(expected_satisfied),
        "ltl_formula": _ltl_formula(operator, signal, guard),
        "minizinc_constraint": _minizinc_constraint(operator, signal, guard),
        "certificate_state": "SAT" if expected_satisfied else "REPAIR_HINT",
    }


def _step_value(step: Mapping[str, Any], signal: str) -> bool:
    return bool(step.get(signal, False))


def _verify_until(
    trace: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    guard_signal: str,
) -> bool:
    for index, step in enumerate(trace):
        if _step_value(step, signal):
            return all(_step_value(trace[prior], guard_signal) for prior in range(index))
    return False


def verify_temporal_case(case: Mapping[str, Any]) -> bool:
    """Evaluate the finite trace for the supported temporal operators."""

    operator = str(case["temporal_operator"])
    signal = str(case["signal"])
    trace = case["trace"]
    if operator == "always":
        return all(_step_value(step, signal) for step in trace)
    if operator == "eventually":
        return any(_step_value(step, signal) for step in trace)
    if operator == "next":
        return len(trace) > 1 and _step_value(trace[1], signal)
    if operator == "until":
        return _verify_until(trace, signal=signal, guard_signal=str(case["guard_signal"]))
    raise ValueError(f"unsupported temporal operator: {operator}")  # pragma: no cover


def validate_temporal_case(case: Mapping[str, Any]) -> None:
    """Check the row contract before replay evidence is derived."""

    required = {
        "case_id",
        "temporal_operator",
        "trace",
        "expected_satisfied",
        "ltl_formula",
        "minizinc_constraint",
        "certificate_state",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise AssertionError(f"missing temporal case fields: {missing}")  # pragma: no cover
    if case["temporal_operator"] not in SUPPORTED_OPERATORS:
        raise AssertionError("unsupported temporal operator")  # pragma: no cover
    if not str(case["minizinc_constraint"]).startswith("constraint "):
        raise AssertionError("missing MiniZinc-style constraint")  # pragma: no cover
    if verify_temporal_case(case) is not bool(case["expected_satisfied"]):
        raise AssertionError("temporal verifier disagrees with expected label")  # pragma: no cover


def _template_rows() -> tuple[
    tuple[str, str, str | None, list[dict[str, bool]], list[dict[str, bool]]],
    ...,
]:
    return (
        (
            "always",
            "power_ok",
            None,
            [{"power_ok": True}, {"power_ok": True}],
            [{"power_ok": True}, {"power_ok": False}],
        ),
        (
            "eventually",
            "ready",
            None,
            [{"ready": False}, {"ready": True}],
            [{"ready": False}, {"ready": False}],
        ),
        (
            "next",
            "armed",
            None,
            [{"armed": False}, {"armed": True}],
            [{"armed": True}, {"armed": False}],
        ),
        (
            "until",
            "released",
            "locked",
            [{"locked": True, "released": False}, {"locked": False, "released": True}],
            [{"locked": False, "released": False}, {"locked": False, "released": True}],
        ),
    )


def generate_temporal_cases() -> list[JsonDict]:
    """REQ-LEARN-1669-1: generate paired SAT and REPAIR_HINT temporal cases."""

    cases: list[JsonDict] = []
    for index, (operator, signal, guard, sat_trace, repair_trace) in enumerate(_template_rows()):
        cases.append(
            make_temporal_case(
                f"ltlzinc1669-{operator}-{signal}-sat-{index:02d}",
                operator,
                signal,
                sat_trace,
                True,
                guard_signal=guard,
            )
        )
        cases.append(
            make_temporal_case(
                f"ltlzinc1669-{operator}-{signal}-repair-hint-{index:02d}",
                operator,
                signal,
                repair_trace,
                False,
                guard_signal=guard,
            )
        )
    return cases


def _case_result(case: Mapping[str, Any], forgotten_case_ids: set[str]) -> JsonDict:
    validate_temporal_case(case)
    case_id = str(case["case_id"])
    local_satisfied = verify_temporal_case(case)
    expected_satisfied = bool(case["expected_satisfied"])
    local_matches = local_satisfied is expected_satisfied
    forgotten = case_id in forgotten_case_ids
    pre_bound = 0.0 if local_matches else 1.0
    post_bound = 1.0 if forgotten else pre_bound
    retained = bool(local_matches and not forgotten)
    return {
        "case_id": case_id,
        "temporal_operator": str(case["temporal_operator"]),
        "expected_satisfied": expected_satisfied,
        "local_satisfied": local_satisfied,
        "local_verifier_matches_expected": local_matches,
        "retained": retained,
        "pre_violation_bound": pre_bound,
        "post_violation_bound": post_bound,
        "bound_worsened": post_bound > pre_bound,
    }


def build_replay_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    forgotten_case_ids: Sequence[str] = (),
) -> tuple[ReplayCase, ...]:
    """REQ-LEARN-1669-3: convert temporal cases into CerCE replay rows."""

    forgotten = {str(case_id) for case_id in forgotten_case_ids}
    replay_cases: list[ReplayCase] = []
    for case in cases:
        result = _case_result(case, forgotten)
        replay_cases.append(
            ReplayCase(
                case_id=str(result["case_id"]),
                pre_violation_bound=float(result["pre_violation_bound"]),
                post_violation_bound=float(result["post_violation_bound"]),
                retained=bool(result["retained"]),
                replay_failed=not bool(result["local_verifier_matches_expected"]),
                source="experiment_1669_ltlzinc",
            )
        )
    return tuple(replay_cases)


def build_memory_update(
    cases: Sequence[Mapping[str, Any]],
    *,
    forgotten_case_ids: Sequence[str] = (),
) -> MemoryPolicyUpdate:
    """Build the CerCE candidate update used for temporal non-forgetting."""

    case_ids = [str(case["case_id"]) for case in cases]
    replay_cases = build_replay_cases(cases, forgotten_case_ids=forgotten_case_ids)
    return MemoryPolicyUpdate(
        policy_update_id=POLICY_UPDATE_ID,
        prior_memory_hash=stable_hash({"stage": "before_ltlzinc_update", "case_ids": ()}),
        updated_memory_hash=stable_hash(
            {"stage": "after_ltlzinc_update", "case_ids": sorted(case_ids)}
        ),
        replay_cases=replay_cases,
        utility_delta=round(len(replay_cases) / 100.0, 6),
        no_model_weight_mutation=True,
        provenance=("REQ-LEARN-1669", "results/experiment_1669_ltlzinc.json"),
    )


def _case_results(
    cases: Sequence[Mapping[str, Any]],
    *,
    forgotten_case_ids: Sequence[str] = (),
) -> list[JsonDict]:
    forgotten = {str(case_id) for case_id in forgotten_case_ids}
    return [_case_result(case, forgotten) for case in cases]


def build_artifact(
    *,
    cases: Sequence[Mapping[str, Any]],
    forgotten_case_ids: Sequence[str] = (),
    project_root: str | Path | PathLike[str] = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1669-4/5: run CerCE and build the terminal benchmark artifact."""

    case_results = _case_results(cases, forgotten_case_ids=forgotten_case_ids)
    update = build_memory_update(cases, forgotten_case_ids=forgotten_case_ids)
    ledger_artifact = evaluate_promotion_gate(
        [update],
        project_root=project_root,
        run_date=run_date,
        tests_run=tests_run,
    )
    temporal_cases_generated = len(case_results)
    temporal_cases_retained = sum(1 for result in case_results if result["retained"])
    forgotten_cases = sum(1 for result in case_results if not result["retained"])
    forgetting_rate = (
        round(forgotten_cases / temporal_cases_generated, 6) if temporal_cases_generated else 1.0
    )
    blockers = sorted(
        set(str(blocker) for blocker in ledger_artifact["blockers"])
        | ({"no_temporal_cases"} if temporal_cases_generated == 0 else set())
    )
    ready = bool(
        temporal_cases_generated
        and temporal_cases_retained == temporal_cases_generated
        and ledger_artifact["cerce_ledger_ready"] is True
        and ledger_artifact["promotion_gate_passed"] is True
        and forgetting_rate == 0.0
        and not blockers
    )
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_traces": list(SPEC_TRACES),
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "continuous_self_learning_task": True,
        "ltlzinc_adapter_ready": ready,
        "temporal_cases_generated": temporal_cases_generated,
        "temporal_cases_retained": temporal_cases_retained,
        "supported_operators": list(SUPPORTED_OPERATORS),
        "temporal_cases": [dict(case) for case in cases],
        "cerce_ledger_ready": bool(ledger_artifact["cerce_ledger_ready"]),
        "promotion_gate_passed": bool(ledger_artifact["promotion_gate_passed"]),
        "policy_certificates_evaluated": int(ledger_artifact["policy_certificates_evaluated"]),
        "accepted_violation_count": int(ledger_artifact["accepted_violation_count"]),
        "replay_retention_rate": float(ledger_artifact["replay_retention_rate"]),
        "cerce_nonforgetting_rate": float(ledger_artifact["nonforgetting_rate"]),
        "forgetting_rate": forgetting_rate,
        "ledger_artifact": ledger_artifact,
        "case_results": case_results,
        "blockers": blockers,
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: ltlzinc_cerce_nonforgetting_passed"
            if ready
            else "blocked: ltlzinc_cerce_forgetting_detected"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields consumed by the Exp 1669 result gate."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"unsupported schema: {artifact['schema']}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    forgetting_rate = float(artifact["forgetting_rate"])
    cerce_nonforgetting_rate = float(artifact["cerce_nonforgetting_rate"])
    if not 0.0 <= forgetting_rate <= 1.0:
        raise AssertionError("forgetting_rate must be between 0 and 1")
    if not 0.0 <= cerce_nonforgetting_rate <= 1.0:
        raise AssertionError("cerce_nonforgetting_rate must be between 0 and 1")
    case_results = artifact["case_results"]
    if len(case_results) != int(artifact["temporal_cases_generated"]):
        raise AssertionError("case_results must match temporal_cases_generated")
    counted_retained = sum(1 for result in case_results if result.get("retained"))
    if counted_retained != int(artifact["temporal_cases_retained"]):
        raise AssertionError("temporal_cases_retained must match case_results")
    ledger_artifact = artifact["ledger_artifact"]
    if int(artifact["policy_certificates_evaluated"]) != int(
        ledger_artifact["policy_certificates_evaluated"]
    ):
        raise AssertionError("policy certificate count must match ledger artifact")
    if artifact["status"] == "complete":
        errors: list[str] = []
        if artifact["ltlzinc_adapter_ready"] is not True:
            errors.append("ltlzinc_adapter_ready must be true")
        if artifact["cerce_ledger_ready"] is not True:
            errors.append("cerce_ledger_ready must be true")
        if artifact["promotion_gate_passed"] is not True:
            errors.append("promotion_gate_passed must be true")
        if forgetting_rate != 0.0:
            errors.append("forgetting_rate must be zero")
        if cerce_nonforgetting_rate != 1.0:
            errors.append("cerce_nonforgetting_rate must be one")
        if artifact["blockers"]:
            errors.append("complete artifact cannot contain blockers")
        if errors:
            raise AssertionError(f"complete artifact is invalid: {errors}")


def run_experiment(
    *,
    output_path: str | Path | PathLike[str] = DEFAULT_OUTPUT_PATH,
    project_root: str | Path | PathLike[str] = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1669 and write `results/experiment_1669_ltlzinc.json`."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    artifact = build_artifact(
        cases=generate_temporal_cases(),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        tests_run=tests_run,
    )
    return _write_json(output_path, artifact)


def main() -> int:  # pragma: no cover
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASE_COUNT",
    "DEFAULT_OUTPUT_PATH",
    "EXPERIMENT_ID",
    "OUTPUT_FILE",
    "POLICY_UPDATE_ID",
    "REQUIRED_ARTIFACT_FIELDS",
    "SCHEMA",
    "SPEC_TRACES",
    "SUPPORTED_OPERATORS",
    "build_artifact",
    "build_memory_update",
    "build_replay_cases",
    "generate_temporal_cases",
    "make_temporal_case",
    "run_experiment",
    "validate_artifact",
    "validate_temporal_case",
    "verify_temporal_case",
]
