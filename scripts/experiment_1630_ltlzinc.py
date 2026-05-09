#!/usr/bin/env python3
"""Exp 1630 LTLZinc-style temporal retention benchmark.

The benchmark is deliberately local and deterministic.  It reuses the finite
trace checker from Exp 1449, converts each temporal row into the same
`CaseRecord` shape used by FR-11 CaseMemory, records extra update rows, and
then checks that every original temporal constraint is still retrievable.

Spec: REQ-LEARN-1630, SCENARIO-LEARN-1630.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.case_memory import CaseMemory, CaseQuery, CaseRecord
from carnot.reporting import ltlzinc_temporal_continual_learning_adapter as temporal


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = "experiment_1630_ltlzinc.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE

EXPERIMENT_ID = 1630
EXPERIMENT = "1630_ltlzinc_temporal_retention_benchmark"
SCHEMA = "ltlzinc_temporal_retention_benchmark_v1"
RUN_DATE = "20260509"
BENCHMARK_NAME = "ltlzinc_temporal_retention"
BENCHMARK_SLICE_PREFIX = "temporal_retention"
MODEL_NAME = "fr11-query-time-memory"
DEFAULT_BENCHMARK_SIZE = 24
SUPPORTED_OPERATORS = temporal.SUPPORTED_OPERATORS
VERIFIER_PATH = (
    "carnot.reporting.ltlzinc_temporal_continual_learning_adapter.verify_temporal_case"
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "benchmark_size",
    "pass_rate",
    "retained_case_count",
    "memory_entry_count",
    "case_results",
    "honest_verdict",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_temporal_case(case: Mapping[str, Any]) -> bool:
    """Use the existing local checker so this benchmark does not parse formulas twice."""

    return temporal.verify_temporal_case(case)


def validate_case_schema(case: Mapping[str, Any]) -> None:
    """Delegate schema checks to the adapter that owns the LTLZinc row contract."""

    temporal.validate_case_schema(case)


def generate_benchmark_cases() -> list[JsonDict]:
    """REQ-LEARN-1630-2: generate the deterministic 24-row temporal benchmark."""

    return [dict(case) for case in temporal.generate_temporal_cases()]


def generate_update_cases() -> list[JsonDict]:
    """Generate later query-time updates that should not evict prior cases.

    These rows use the same four temporal operators as the benchmark, but with
    fresh signals.  They simulate FR-11 learning new temporal constraints after
    the original retention anchors have already been stored.
    """

    templates: tuple[
        tuple[str, str, str | None, list[dict[str, bool]], list[dict[str, bool]]],
        ...,
    ] = (
        (
            "always",
            "coolant_ok",
            None,
            [{"coolant_ok": True}, {"coolant_ok": True}],
            [{"coolant_ok": True}, {"coolant_ok": False}],
        ),
        (
            "eventually",
            "commit_seen",
            None,
            [{"commit_seen": False}, {"commit_seen": True}],
            [{"commit_seen": False}, {"commit_seen": False}],
        ),
        (
            "next",
            "rollback_ready",
            None,
            [{"rollback_ready": False}, {"rollback_ready": True}],
            [{"rollback_ready": True}, {"rollback_ready": False}],
        ),
        (
            "until",
            "lease_released",
            "lease_held",
            [
                {"lease_held": True, "lease_released": False},
                {"lease_held": False, "lease_released": True},
            ],
            [
                {"lease_held": False, "lease_released": False},
                {"lease_held": False, "lease_released": True},
            ],
        ),
    )
    rows: list[JsonDict] = []
    for index, (operator, signal, guard, accepted_trace, rejected_trace) in enumerate(templates):
        rows.append(
            temporal.make_case(
                f"ltlzinc-update-{operator}-{signal}-sat-{index:02d}",
                operator,
                signal,
                accepted_trace,
                True,
                guard_signal=guard,
                split="update",
            )
        )
        rows.append(
            temporal.make_case(
                f"ltlzinc-update-{operator}-{signal}-repair-hint-{index:02d}",
                operator,
                signal,
                rejected_trace,
                False,
                guard_signal=guard,
                split="update",
            )
        )
    return rows


def case_to_record(case: Mapping[str, Any]) -> CaseRecord:
    """REQ-LEARN-1630-3: convert one temporal row into CaseMemory's shape."""

    validate_case_schema(case)
    operator = str(case["temporal_operator"])
    signal = str(case["signal"])
    guard_signal = case.get("guard_signal")
    certificate_state = str(case["certificate_state"]).lower()
    expected_satisfied = bool(case["expected_satisfied"])
    property_names = [
        "temporal_constraint_retention",
        f"temporal_operator_{operator}",
        f"signal_{signal}",
    ]
    if guard_signal:
        property_names.append(f"guard_{guard_signal}")
    return CaseRecord.normalize(
        benchmark=BENCHMARK_NAME,
        benchmark_slice=f"{BENCHMARK_SLICE_PREFIX}:{operator}",
        model_name=MODEL_NAME,
        case_id=str(case["case_id"]),
        violation_types=(f"temporal:{operator}:{signal}:{certificate_state}",),
        prompt_text=(
            f"Retain LTLZinc {operator} constraint for {signal} case {case['case_id']}"
        ),
        description_texts=(
            str(case["ltl_formula"]),
            str(case["minizinc_constraint"]),
            f"expected_satisfied={expected_satisfied}",
        ),
        property_names=tuple(property_names),
        baseline_success=expected_satisfied,
        repair_success=True,
        confidence=1.0,
        source_experiment=EXPERIMENT_ID,
        source_artifact=f"results/{OUTPUT_FILE}",
        response_mode="ltlzinc_temporal_retention",
        verifier_path=VERIFIER_PATH,
    )


def _record_cases(memory: CaseMemory, cases: Sequence[Mapping[str, Any]]) -> list[CaseRecord]:
    records = [case_to_record(case) for case in cases]
    for record in records:
        memory.record(record)
    return records


def _retention_result(
    memory: CaseMemory,
    case: Mapping[str, Any],
    record: CaseRecord,
) -> JsonDict:
    expected_satisfied = bool(case["expected_satisfied"])
    local_satisfied = verify_temporal_case(case)
    local_matches = local_satisfied is expected_satisfied
    matches = memory.retrieve(CaseQuery.from_record(record), limit=5)
    expected_fingerprint = record.key.fingerprint
    exact_match = next(
        (match for match in matches if match.entry.key.fingerprint == expected_fingerprint),
        None,
    )
    return {
        "case_id": str(case["case_id"]),
        "temporal_operator": str(case["temporal_operator"]),
        "expected_satisfied": expected_satisfied,
        "local_satisfied": local_satisfied,
        "local_verifier_matches_expected": local_matches,
        "retrieved_case_fingerprint": exact_match.entry.key.fingerprint if exact_match else None,
        "retrieved_score": exact_match.score if exact_match else 0,
        "retrieved_matched_fields": list(exact_match.matched_fields) if exact_match else [],
        "retained": bool(local_matches and exact_match is not None),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_artifact(
    *,
    benchmark_cases: Sequence[Mapping[str, Any]],
    update_cases: Sequence[Mapping[str, Any]] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """REQ-LEARN-1630-4/5: build the terminal retention benchmark artifact."""

    memory = CaseMemory()
    records = _record_cases(memory, benchmark_cases)
    _record_cases(memory, update_cases or ())
    case_results = [
        _retention_result(memory, case, record)
        for case, record in zip(benchmark_cases, records, strict=True)
    ]
    benchmark_size = len(case_results)
    retained_case_count = sum(1 for result in case_results if result["retained"])
    pass_rate = retained_case_count / benchmark_size if benchmark_size else 0.0
    complete = benchmark_size > 0 and pass_rate == 1.0
    artifact: JsonDict = {
        "status": "complete" if complete else "blocked",
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec": ["REQ-LEARN-1630", "SCENARIO-LEARN-1630"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_size": benchmark_size,
        "pass_rate": pass_rate,
        "retained_case_count": retained_case_count,
        "memory_entry_count": len(memory),
        "supported_operators": list(SUPPORTED_OPERATORS),
        "case_results": case_results,
        "honest_verdict": (
            "ltlzinc_temporal_retention_benchmark_passed"
            if complete
            else "ltlzinc_temporal_retention_benchmark_failed"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1630-4/5: enforce the JSON contract used by later gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact["schema"] == SCHEMA, "unsupported schema")
    status = str(artifact["status"])
    _require(status in {"complete", "blocked"}, "unsupported status")
    benchmark_size = int(artifact["benchmark_size"])
    retained_case_count = int(artifact["retained_case_count"])
    pass_rate = float(artifact["pass_rate"])
    case_results = artifact["case_results"]
    _require(isinstance(case_results, Sequence), "case_results must be a sequence")
    _require(not isinstance(case_results, (str, bytes)), "case_results must be rows")
    _require(benchmark_size >= 0, "benchmark_size must be non-negative")
    _require(0.0 <= pass_rate <= 1.0, "pass_rate must be between 0 and 1")
    _require(len(case_results) == benchmark_size, "case_results must match benchmark_size")
    counted_retained = sum(1 for result in case_results if result.get("retained"))
    _require(
        counted_retained == retained_case_count,
        "retained_case_count must match retained case results",
    )
    if status == "complete":
        _require(
            benchmark_size > 0
            and pass_rate == 1.0
            and retained_case_count == benchmark_size
            and all(
                result.get("temporal_operator") and result.get("retrieved_case_fingerprint")
                for result in case_results
            ),
            "complete artifact is invalid",
        )


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run Exp 1630 and write `results/experiment_1630_ltlzinc.json`."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    artifact = build_artifact(
        benchmark_cases=generate_benchmark_cases(),
        update_cases=generate_update_cases(),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )
    return _write_json(output_path, artifact)


def main() -> int:  # pragma: no cover
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
