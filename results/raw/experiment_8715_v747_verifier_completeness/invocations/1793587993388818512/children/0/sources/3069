"""Ship and measure an opt-in consumer for the Rust recalibration service.

The experiment requalifies changed worker bytes before reuse. It measures the
public client boundary and makes no calibration-quality or default-promotion
claim.

Spec: REQ-CL-7598, REQ-VERIFY-7598, and SCENARIO-CL-7598-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, TextIO

from carnot import experiment_7585_v662_portable_service as upstream
from carnot.experiment_7561_v661_recalibration_prototype import typed_decision
from carnot.pipeline.calibrated_decision_service import (
    DURABILITY_POLICY,
    CalibratedDecision,
    CalibratedDecisionService,
    FeedbackAcknowledgment,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.663"
EXPERIMENT_ID = "exp7598-v663-rust-consumer"
SCHEMA = "carnot.exp7598.v663.rust_consumer.v1"
RANDOM_SEED = 7_598_001
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7598_v663_rust_consumer.json")
RAW_DIR = Path("results/raw/experiment_7598_v663_rust_consumer")
MODULE_PATH = Path("python/carnot/experiment_7598_v663_rust_consumer.py")
CLIENT_PATH = Path("python/carnot/pipeline/calibrated_decision_service.py")
CALIBRATION_PATH = Path("python/carnot/pipeline/probability_calibration_verifier.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7598_v663_rust_consumer.py")
TEST_PATH = Path("tests/python/test_experiment_7598_v663_rust_consumer.py")
CALIBRATION_TEST_PATH = Path("tests/python/test_probability_calibration_verifier.py")
RUST_SOURCE_PATH = Path("crates/carnot-core/src/bin/portable-recalibration-service.rs")
RUST_BINARY = Path("target/release/portable-recalibration-service")
CL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
VERIFY_SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_PATH = Path("results/experiment_7585_v662_portable_service.json")
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "current_llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
ARMS = ("rust", "python_service", "python_inprocess")
MODES = ("cold", "warm")
BATCH_SIZES = (1, 8)
REPEATS = 30
REQUIRED_CHECK_NAMES = (
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "rust_binary_tests",
    "rust_fmt",
    "rust_clippy",
    "foreign_cwd_consumer_integration",
)
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object without inventing evidence for invalid bytes."""

    return upstream.load_object(path)


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind one exact source while keeping a stable worktree label."""

    try:
        return upstream.source_row(path, root)
    except UnicodeDecodeError:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            label = str(resolved)
        return {
            "path": label,
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "original_honest_verdict": None,
            "original_verdict_class": None,
            "original_flagged_adversarial": None,
        }


def _check(
    check: str,
    upstream_name: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "eq",
) -> JsonDict:
    passed = observed == expected if op == "eq" else bool(observed)
    return {
        "check": check,
        "upstream": upstream_name,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
    }


def _receipt_names(value: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("name"))
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is True
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate Exp7585 and identify whether current parity must rerun."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    required = (
        UPSTREAM_PATH,
        RUST_SOURCE_PATH,
        CL_SPEC_PATH,
        VERIFY_SPEC_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
    )
    for relative in required:
        path = repo / relative
        readable = path.is_file() and path.stat().st_size > 0
        rows.append(
            _check(
                f"source_readable:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if readable else None,
            )
        )
        if readable:
            sources[relative.as_posix()] = source_row(path, repo)

    exp7585 = load_object(repo / UPSTREAM_PATH)
    passed_receipts = _receipt_names(exp7585)
    historical_policies = {
        row.get("durability_policy")
        for row in exp7585.get("rows") or []
        if isinstance(row, Mapping) and row.get("row_type") == "service_trace"
    }
    terminal_valid = set(upstream.TERMINAL_CHECK_NAMES).issubset(passed_receipts)
    equal_durability = historical_policies == {DURABILITY_POLICY}
    rows.extend(
        [
            _check(
                "exp7585_portable_parity_score",
                "Exp7585",
                UPSTREAM_PATH.as_posix(),
                "portable_parity_score",
                1,
                exp7585.get("portable_parity_score"),
            ),
            _check(
                "exp7585_service_measurement_complete_score",
                "Exp7585",
                UPSTREAM_PATH.as_posix(),
                "service_measurement_complete_score",
                1,
                exp7585.get("service_measurement_complete_score"),
            ),
            _check(
                "exp7585_terminal_validation",
                "Exp7585",
                UPSTREAM_PATH.as_posix(),
                "validation_receipts",
                True,
                terminal_valid,
            ),
            _check(
                "exp7585_not_flagged",
                "Exp7585",
                UPSTREAM_PATH.as_posix(),
                "flagged_adversarial",
                False,
                exp7585.get("flagged_adversarial"),
            ),
            _check(
                "exp7585_equal_durability",
                "Exp7585",
                UPSTREAM_PATH.as_posix(),
                "rows[*].durability_policy",
                True,
                equal_durability,
            ),
        ]
    )
    for spec_path, requirement in (
        (CL_SPEC_PATH, "REQ-CL-7598"),
        (VERIFY_SPEC_PATH, "REQ-VERIFY-7598"),
    ):
        text = (
            (repo / spec_path).read_text(encoding="utf-8") if (repo / spec_path).is_file() else ""
        )
        rows.append(
            _check(
                f"driving_requirement:{requirement}",
                spec_path.as_posix(),
                spec_path.as_posix(),
                "REQ-*",
                requirement,
                requirement if requirement in text else None,
            )
        )
    rows.append(
        _check(
            "rust_toolchain_owned",
            "local toolchain",
            "PATH",
            "cargo",
            True,
            shutil.which("cargo") is not None,
        )
    )
    old_source = (
        exp7585.get("source_artifact_hashes", {}).get(RUST_SOURCE_PATH.as_posix(), {}).get("sha256")
    )
    current_source = (
        sha256_file(repo / RUST_SOURCE_PATH) if (repo / RUST_SOURCE_PATH).is_file() else None
    )
    historical_binary = exp7585.get("rust_binary_sha256")
    current_binary = sha256_file(repo / RUST_BINARY) if (repo / RUST_BINARY).is_file() else None
    failed = next((row for row in rows if row["passed"] is not True), None)
    blocker = (
        {
            key: deepcopy(failed[key])
            for key in ("check", "upstream", "path", "field", "op", "expected", "observed")
        }
        if failed
        else None
    )
    return {
        "rows": rows,
        "blocker": blocker,
        "exp7585": exp7585,
        "source_artifact_hashes": sources,
        "historical_equal_durability": equal_durability,
        "historical_rust_source_sha256": old_source,
        "current_rust_source_sha256": current_source,
        "historical_rust_binary_sha256": historical_binary,
        "current_rust_binary_sha256": current_binary,
        "requalification_required": old_source != current_source
        or historical_binary != current_binary,
        "resource_observations": {
            "cargo_path": shutil.which("cargo"),
            "tmp_free_bytes": os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize,
            "state_root_owner": "current_experiment_private_tmp",
            "process_owner": "current_client_only",
        },
    }


def run_python_service_request(request: Mapping[str, Any]) -> JsonDict:
    """Serve the extended protocol while reusing the qualified Python math."""

    if request.get("operation") != "predict":
        return upstream.run_python_service_request(request)
    try:
        state_path = Path(str(request["state_path"]))
        if not state_path.exists():
            upstream.initialize_state(state_path)
        started = time.perf_counter_ns()
        state = upstream._load_state(state_path)
        read_ns = time.perf_counter_ns() - started
        seen: set[str] = set()
        predictions: list[JsonDict] = []
        predict_ns = 0
        for raw in request.get("queries") or []:
            row = dict(raw)
            event_id = str(row["event_id"])
            if event_id in seen or event_id in state.processed_event_ids:
                raise ValueError(f"duplicate_feedback:{event_id}")
            seen.add(event_id)
            started = time.perf_counter_ns()
            probability = state.predict(float(row["probability"]))
            decision = typed_decision(probability)
            predict_ns += time.perf_counter_ns() - started
            predictions.append(
                {"event_id": event_id, "probability": probability, "action": decision["action"]}
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {"ok": False, "error": str(error)}
    return {
        "ok": True,
        "predictions": predictions,
        "acknowledgments": [],
        "acknowledged_release_count": 0,
        "processed_event_count": 0,
        "reloaded_state_matches": True,
        "state_bytes": state_path.stat().st_size,
        "state": state.to_payload(),
        "stage_ns": {"read": read_ns, "predict": predict_ns},
        "kernel_ns": predict_ns,
        "durability_policy": DURABILITY_POLICY,
    }


def python_service_worker_loop(stream_in: TextIO, stream_out: TextIO) -> int:  # pragma: no cover
    """Serve the matched Python JSON-lines arm in a fresh process."""

    for line in stream_in:
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("request_not_object")
            response = run_python_service_request(request)
        except (json.JSONDecodeError, ValueError) as error:
            response = {"ok": False, "error": str(error)}
        stream_out.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
        stream_out.flush()
    return 0


class _InProcessService:  # pragma: no cover - exercised by the declared benchmark.
    """Matched reference with process transport removed but durability retained."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        started = time.perf_counter_ns()
        upstream.initialize_state(state_path)
        self.setup_ns = time.perf_counter_ns() - started
        self._pending: dict[str, float] = {}

    def predict(self, event_id: str, probability: float) -> CalibratedDecision:
        started = time.perf_counter_ns()
        response = run_python_service_request(
            {
                "operation": "predict",
                "state_path": str(self.state_path),
                "queries": [{"event_id": event_id, "probability": probability}],
            }
        )
        elapsed = time.perf_counter_ns() - started
        if response.get("ok") is not True:
            return CalibratedDecision(
                event_id, 1.0, "escalate", False, error=str(response.get("error"))
            )
        row = response["predictions"][0]
        self._pending[event_id] = probability
        return CalibratedDecision(
            event_id,
            float(row["probability"]),
            row["action"],
            True,
            service_ns=elapsed,
            stage_ns=dict(response["stage_ns"]),
        )

    def release_feedback(self, event_id: str, label: int) -> FeedbackAcknowledgment:
        started = time.perf_counter_ns()
        response = run_python_service_request(
            {
                "operation": "trace",
                "state_path": str(self.state_path),
                "events": [
                    {
                        "event_id": event_id,
                        "probability": self._pending[event_id],
                        "label": label,
                    }
                ],
            }
        )
        elapsed = time.perf_counter_ns() - started
        durable = bool(
            response.get("ok") is True
            and response.get("acknowledgments") == [0]
            and response.get("reloaded_state_matches") is True
            and response.get("durability_policy") == DURABILITY_POLICY
        )
        if not durable:
            return FeedbackAcknowledgment(event_id, False, False, False, str(response.get("error")))
        self._pending.pop(event_id)
        return FeedbackAcknowledgment(
            event_id,
            True,
            True,
            True,
            durability_policy=DURABILITY_POLICY,
            service_ns=elapsed,
            kernel_ns=int(response["kernel_ns"]),
            stage_ns=dict(response["stage_ns"]),
        )

    def close(self) -> None:
        return None


def _python_worker_command(root: Path) -> tuple[str, ...]:  # pragma: no cover
    return (
        str(root / ".venv/bin/python"),
        "-u",
        str(root / WRAPPER_PATH),
        "--date",
        RUN_DATE,
        "--root",
        str(root),
        "--python-service-worker",
    )


def _open_service(
    root: Path, arm: str, state: Path, *, telemetry_enabled: bool = False
) -> Any:  # pragma: no cover
    if arm == "python_inprocess":
        return _InProcessService(state)
    started = time.perf_counter_ns()
    command = _python_worker_command(root) if arm == "python_service" else None
    service = CalibratedDecisionService(
        state_path=state,
        binary_path=root / RUST_BINARY,
        response_timeout_s=10.0,
        process_command=command,
        cwd=root,
        extra_env={
            "PYTHONPATH": f"{root / 'python'}:{root}",
            "PYTHONUNBUFFERED": "1",
        },
        telemetry_enabled=telemetry_enabled,
    )
    service.setup_ns = time.perf_counter_ns() - started
    return service


def _block_events(seed: int, batch_size: int) -> list[tuple[str, float, int]]:  # pragma: no cover
    rng = random.Random(seed)
    return [
        (f"event-{seed}-{index}", rng.uniform(0.001, 0.999), rng.randrange(2))
        for index in range(batch_size)
    ]


def _run_block(
    service: Any,
    events: Sequence[tuple[str, float, int]],
    *,
    include_setup: bool,
) -> JsonDict:  # pragma: no cover
    started = time.perf_counter_ns()
    decisions: list[tuple[str, float, str]] = []
    acknowledgments: list[FeedbackAcknowledgment] = []
    kernel_ns = 0
    for event_id, probability, label in events:
        decision = service.predict(event_id, probability)
        if not decision.available:
            raise RuntimeError(f"prediction_failed:{decision.error}")
        decisions.append((event_id, decision.error_probability, decision.action))
        ack = service.release_feedback(event_id, label)
        if not ack.durable:
            raise RuntimeError(f"feedback_failed:{ack.error}")
        acknowledgments.append(ack)
        kernel_ns += ack.kernel_ns + int(decision.stage_ns.get("predict", 0))
    resumed_id = f"{events[-1][0]}-resumed"
    resumed = service.predict(resumed_id, 0.37)
    if not resumed.available:
        raise RuntimeError(f"resumed_prediction_failed:{resumed.error}")
    request_ns = time.perf_counter_ns() - started
    setup_ns = int(getattr(service, "setup_ns", 0))
    return {
        "elapsed_ns": request_ns + (setup_ns if include_setup else 0),
        "setup_ns": setup_ns,
        "per_request_ns": request_ns / (len(events) * 2 + 1),
        "kernel_ns": kernel_ns + int(resumed.stage_ns.get("predict", 0)),
        "decisions": decisions,
        "resumed": (resumed.error_probability, resumed.action),
        "durable_acknowledgments": len(acknowledgments),
        "state": upstream._load_state(service.state_path).to_payload(),
    }


def _outcomes_match(
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, bool]:  # pragma: no cover
    reference = results["python_inprocess"]
    decision_parity = True
    for arm in ("rust", "python_service"):
        candidate = results[arm]
        if len(candidate["decisions"]) != len(reference["decisions"]):
            decision_parity = False
            continue
        for left, right in zip(candidate["decisions"], reference["decisions"], strict=True):
            decision_parity &= bool(
                left[0] == right[0]
                and abs(float(left[1]) - float(right[1])) <= upstream.PARITY_TOLERANCE
                and left[2] == right[2]
            )
        decision_parity &= bool(
            abs(float(candidate["resumed"][0]) - float(reference["resumed"][0]))
            <= upstream.PARITY_TOLERANCE
            and candidate["resumed"][1] == reference["resumed"][1]
        )
    reference_state = upstream._state_core(reference["state"])
    reload_parity = True
    for arm in ("rust", "python_service"):
        candidate_state = upstream._state_core(results[arm]["state"])
        numeric_error = max(
            upstream._numeric_nested_error(candidate_state.get(field), reference_state.get(field))
            for field in ("gram", "target", "theta")
        )
        exact_fields_match = all(
            candidate_state.get(field) == reference_state.get(field)
            for field in ("sample_count", "processed_event_ids", "solver_config", "constrained")
        )
        reload_parity &= numeric_error <= upstream.PARITY_TOLERANCE and exact_fields_match
    return decision_parity, reload_parity


def _comparison_row(
    *,
    arm: str,
    mode: str,
    batch_size: int,
    repeat: int,
    seed: int,
    order: Sequence[str],
    result: Mapping[str, Any],
    decision_parity: bool,
    reload_parity: bool,
    source_hashes: Mapping[str, str],
) -> JsonDict:  # pragma: no cover
    pair_id = f"{mode}:{batch_size}:{repeat}"
    return {
        "row_type": "consumer_comparison",
        "unit_id": pair_id,
        "pair_id": pair_id,
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "arm": arm,
        "arm_order": list(order),
        "seed": seed,
        "numerator": int(result["elapsed_ns"]),
        "denominator": batch_size * 2 + 1,
        "metric": int(result["elapsed_ns"]),
        "metric_name": "whole_consumer_ns",
        "metric_direction": "lower_is_better",
        "setup_ns": int(result["setup_ns"]),
        "per_request_ns": float(result["per_request_ns"]),
        "kernel_ns": int(result["kernel_ns"]),
        "censored": False,
        "missing": False,
        "provenance": "public_predict_release_durable_ack_resumed_prediction",
        "durability_policy": DURABILITY_POLICY,
        "decision_parity": decision_parity,
        "reload_parity": reload_parity,
        "durable_acknowledgments": int(result["durable_acknowledgments"]),
        "source_hashes": dict(source_hashes),
    }


def measure_consumer_rows(
    root: Path, scratch: Path, *, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Measure 30 paired blocks for every registered mode and batch size."""

    rows: list[JsonDict] = []
    scratch.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "rust_source": sha256_file(root / RUST_SOURCE_PATH),
        "rust_binary": sha256_file(root / RUST_BINARY),
    }
    completed = 0
    planned = len(MODES) * len(BATCH_SIZES) * REPEATS
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            warm_services: dict[str, Any] = {}
            if mode == "warm":
                for arm in ARMS:
                    warm_services[arm] = _open_service(
                        root,
                        arm,
                        scratch / f"warm-b{batch_size}-{arm}.json",
                    )
            try:
                for repeat in range(REPEATS):
                    seed = RANDOM_SEED + mode_index * 10_000 + batch_size * 100 + repeat
                    events = _block_events(seed, batch_size)
                    order = list(ARMS)
                    random.Random(seed ^ RANDOM_SEED).shuffle(order)
                    results: dict[str, JsonDict] = {}
                    for arm in order:
                        if mode == "cold":
                            service = _open_service(
                                root,
                                arm,
                                scratch / f"cold-b{batch_size}-r{repeat}-{arm}.json",
                            )
                        else:
                            service = warm_services[arm]
                        try:
                            results[arm] = _run_block(
                                service,
                                events,
                                include_setup=mode == "cold",
                            )
                        finally:
                            if mode == "cold":
                                service.close()
                    decision_parity, reload_parity = _outcomes_match(results)
                    for arm in ARMS:
                        rows.append(
                            _comparison_row(
                                arm=arm,
                                mode=mode,
                                batch_size=batch_size,
                                repeat=repeat,
                                seed=seed,
                                order=order,
                                result=results[arm],
                                decision_parity=decision_parity,
                                reload_parity=reload_parity,
                                source_hashes=source_hashes,
                            )
                        )
                    completed += 1
                    progress(
                        started, "benchmark", "unit_complete", completed=completed, planned=planned
                    )
            finally:
                for service in warm_services.values():
                    service.close()
    return rows


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile_requires_values")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _arm_latency(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "p50_ns": _percentile([float(row["metric"]) for row in rows], 0.50),
        "p95_ns": _percentile([float(row["metric"]) for row in rows], 0.95),
        "setup_p50_ns": _percentile([float(row["setup_ns"]) for row in rows], 0.50),
        "per_request_p50_ns": _percentile([float(row["per_request_ns"]) for row in rows], 0.50),
        "kernel_p50_ns": _percentile([float(row["kernel_ns"]) for row in rows], 0.50),
    }


def _paired_ratio_interval(ratios: Sequence[float], seed: int) -> JsonDict:
    """Return a deterministic paired bootstrap interval over independent blocks."""

    if len(ratios) != REPEATS:
        raise ValueError("paired_ratio_requires_30_blocks")
    rng = random.Random(seed)
    bootstraps = [
        statistics.median(ratios[rng.randrange(len(ratios))] for _ in ratios) for _ in range(2_000)
    ]
    return {
        "estimate": statistics.median(ratios),
        "lower95": _percentile(bootstraps, 0.025),
        "upper95": _percentile(bootstraps, 0.975),
        "pair_count": len(ratios),
        "positive_count": sum(value > 1.0 for value in ratios),
        "negative_count": sum(value < 1.0 for value in ratios),
        "tie_count": sum(value == 1.0 for value in ratios),
        "direction": "comparator_over_rust_higher_favors_rust",
        "bootstrap_seed": seed,
        "bootstrap_draws": 2_000,
    }


def summarize_consumer_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce complete triples and choose the fastest eligible comparator."""

    policies = {row.get("durability_policy") for row in rows}
    if policies != {DURABILITY_POLICY}:
        raise ValueError("durability_policy_mismatch")
    result: JsonDict = {}
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            selected = [
                row
                for row in rows
                if row.get("mode") == mode and row.get("batch_size") == batch_size
            ]
            pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
            for row in selected:
                pairs.setdefault(str(row.get("pair_id")), {})[str(row.get("arm"))] = row
            if len(pairs) != REPEATS or any(set(arms) != set(ARMS) for arms in pairs.values()):
                raise ValueError("consumer_pair_incomplete")
            by_arm = {arm: [arms[arm] for arms in pairs.values()] for arm in ARMS}
            eligible = [
                arm
                for arm in ("python_service", "python_inprocess")
                if all(
                    row.get("decision_parity") is True
                    and row.get("reload_parity") is True
                    and row.get("censored") is False
                    and row.get("missing") is False
                    and int(row.get("durable_acknowledgments") or 0) == batch_size
                    for row in by_arm[arm]
                )
            ]
            if not eligible:
                raise ValueError("eligible_comparator_missing")
            arm_summaries = {arm: _arm_latency(by_arm[arm]) for arm in ARMS}
            comparator = min(eligible, key=lambda arm: arm_summaries[arm]["p50_ns"])
            ratios = [
                float(arms[comparator]["metric"]) / float(arms["rust"]["metric"])
                for arms in pairs.values()
            ]
            key = f"{mode}:{batch_size}"
            result[key] = {
                **arm_summaries,
                "primary_comparator": comparator,
                "eligible_comparators": eligible,
                "paired_ratio": _paired_ratio_interval(
                    ratios, RANDOM_SEED + mode_index * 100 + batch_size
                ),
                "pair_count": len(pairs),
                "excluded_count": 0,
                "censored_count": 0,
                "decision_parity": all(row.get("decision_parity") is True for row in selected),
                "reload_parity": all(row.get("reload_parity") is True for row in selected),
            }
    return result


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    return set((*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(passed)


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute public-client readiness and comparator-relative benefit."""

    rows = value.get("consumer_comparison_rows")
    summary: JsonDict = {}
    measurement_complete = False
    if isinstance(rows, list) and rows:
        try:
            summary = summarize_consumer_rows(rows)
            measurement_complete = True
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            measurement_complete = False
    requalification = value.get("current_binary_requalification")
    parity = isinstance(requalification, Mapping) and requalification.get("passed") is True
    validation = _receipts_pass(list(value.get("validation_receipts") or []))
    ready = bool(measurement_complete and parity and validation)
    speed = bool(
        ready
        and len(summary) == 4
        and all(
            row.get("decision_parity") is True
            and row.get("reload_parity") is True
            and float(row.get("paired_ratio", {}).get("lower95", 0.0)) > 1.0
            for row in summary.values()
        )
    )
    return {
        "consumer_ready_score": int(ready),
        "consumer_speed_benefit_score": int(speed),
        "measurement_complete": measurement_complete,
        "current_binary_requalified": parity,
        "required_validation_passed": validation,
        "comparison_summary": summary,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream_name: str,
    path: str,
    field: str,
    op: str = "eq",
) -> JsonDict:
    if op == "gt":
        passed = observed is not None and float(observed) > float(expected)
    else:
        passed = observed == expected
    return {
        "check": check,
        "category": category,
        "upstream": upstream_name,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": "Keep validity, readiness, benefit, retention, and freshness independent.",
    }


def acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    lowers = [
        row.get("paired_ratio", {}).get("lower95")
        for row in reduction.get("comparison_summary", {}).values()
    ]
    minimum = min(lowers) if lowers and all(item is not None for item in lowers) else None
    return [
        _gate(
            "current_binary_parity",
            "validity",
            True,
            reduction["current_binary_requalified"],
            upstream_name="current Rust and Python workers",
            path="current_binary_requalification",
            field="passed",
        ),
        _gate(
            "public_consumer_ready",
            "readiness",
            1,
            reduction["consumer_ready_score"],
            upstream_name="current public-client rows",
            path="consumer_comparison_rows",
            field="consumer_ready_score",
        ),
        _gate(
            "strongest_comparator_lower95",
            "benefit",
            1.0,
            minimum,
            upstream_name="current paired blocks",
            path="comparison_summary[*].paired_ratio",
            field="lower95",
            op="gt",
        ),
        _gate(
            "empirical_head_not_installed",
            "retention",
            False,
            value.get("empirical_head_installed"),
            upstream_name="current client configuration",
            path="empirical_head_installed",
            field="empirical_head_installed",
        ),
        _gate(
            "fresh_calibration_claim_forbidden",
            "freshness",
            False,
            value.get("calibration_quality_claimed"),
            upstream_name="task claim boundary",
            path="calibration_quality_claimed",
            field="calibration_quality_claimed",
        ),
    ]


def field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Carry each required reporting rule beside the emitted value."""

    specific = {
        "honest_verdict": "Use a complete_ terminal prefix; execution does not prove policy benefit.",
        "verdict_class": "Use one closed class; only unfinished owned work is partial.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence opens no gate.",
        "gate_check_summary": "A block names check, upstream, path, field, operator, expected, and observed.",
        "acceptance_gate_results": "Keep validity, readiness, benefit, retention, and freshness separate.",
        "rows": "Keep one auditable arm row per independent paired block.",
        "sample_size_budget": "Count independent blocks, exclusions, and censoring without multiplying windows.",
        "inference_substrate": "State the real host process and reference execution mode.",
        "inference_substrate_class": "Record the actual no-model class separately from the planned class.",
        "MODEL_SPECS": "No current LLM call means an empty model list.",
        "invocation_counts": "Count current loads, forwards, generations, and tokens independently.",
        "duration_s": "Measure current monotonic work without inherited or padded time.",
        "random_seed": "Bind every randomized arm order, event block, and bootstrap stage.",
        "reproducibility_checksum": "Bind immutable evidence, configuration, and terminal reduction.",
        "source_artifact_hashes": "Distinguish authenticated producer bytes from missing evidence.",
        "validation_receipts": "Bind each command, worktree, exit code, and raw log hash.",
        "verifier_is_oracle": "Exact fixtures cannot support an oracle-distinct benefit claim.",
        "field_principles": "Keep each reporting principle in the artifact.",
        "consumer_ready_score": "One requires a real public call plus restart and error semantics.",
        "consumer_speed_benefit_score": "One requires lower95 above one against the fastest eligible comparator.",
        "consumer_comparison_rows": "Keep every paired block, batch size, arm, time, and exact outcome.",
        "production_default_changed": "The service remains explicit opt-in.",
        "calibration_quality_claimed": "This interface measurement does not measure calibration quality.",
    }
    return {
        key: specific.get(key, f"Retain {key} so terminal scope cannot disappear silently.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks and this self-reference."""

    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "started_at_utc",
        "completed_at_utc",
        "phase_spans",
    }
    stable = {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    return canonical_hash(stable)


def _base_artifact(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Mapping[str, Any]],
    duration_s: float,
    requalification: Mapping[str, Any],
    blocker: Mapping[str, Any] | None = None,
) -> JsonDict:
    comparison_rows = [deepcopy(dict(row)) for row in rows]
    provisional: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7598,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "honest_verdict": "complete_blocked_pending",
        "verdict_class": "blocked" if blocker else "null",
        "flagged_adversarial": False,
        "gate_check_summary": deepcopy(dict(blocker)) if blocker else None,
        "acceptance_gate_results": [],
        "rows": deepcopy(comparison_rows),
        "consumer_comparison_rows": comparison_rows,
        "sample_size_budget": {
            "paired_blocks": {
                "intended": REPEATS * len(MODES) * len(BATCH_SIZES),
                "observed": len({row.get("pair_id") for row in comparison_rows}),
                "excluded": 0,
                "censored": sum(bool(row.get("censored")) for row in comparison_rows),
                "arms_per_block": len(ARMS),
            }
        },
        "inference_substrate": "host_rust_jsonl_python_service_and_inprocess_reference",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "no_model_load": True,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": duration_s,
        "random_seed": {
            "event_and_arm_order": RANDOM_SEED,
            "paired_bootstrap_base": RANDOM_SEED,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "verifier_is_oracle": False,
        "consumer_ready_score": 0,
        "consumer_speed_benefit_score": 0,
        "production_default_changed": False,
        "calibration_quality_claimed": False,
        "empirical_head_installed": False,
        "arc_agent_changed": False,
        "pyo3_binding_claimed": False,
        "hardware_speed_claimed": False,
        "end_to_end_qwen_latency_claimed": False,
        "current_binary_requalification": deepcopy(dict(requalification)),
        "comparison_summary": {},
        "independent_reduction": {},
        "execution_venue": "host",
        "worktree": str(root.resolve()),
        "retire_if_same_verdict": True,
        "prior_verdict_retirement": {
            "literal_prior_verdict": "complete_blocked_exp7561_recalibration_ready_score",
            "repeated": False,
            "retired": False,
            "scientific_hypothesis_retired": False,
        },
    }
    reduction = independent_reduce(provisional)
    provisional["consumer_ready_score"] = reduction["consumer_ready_score"]
    provisional["consumer_speed_benefit_score"] = reduction["consumer_speed_benefit_score"]
    provisional["comparison_summary"] = reduction["comparison_summary"]
    provisional["independent_reduction"] = reduction
    provisional["acceptance_gate_results"] = acceptance_gates(provisional, reduction)
    if blocker:
        provisional["honest_verdict"] = f"complete_blocked_{blocker['check']}"
        provisional["verdict_class"] = "blocked"
    elif reduction["consumer_speed_benefit_score"] == 1:
        provisional["honest_verdict"] = "complete_positive_rust_consumer_speed_benefit"
        provisional["verdict_class"] = "positive"
    else:
        provisional["honest_verdict"] = "complete_null_rust_consumer_ready_speed_gate_failed"
        provisional["verdict_class"] = "null"
    provisional["field_principles"] = field_principles(
        [*provisional, "field_principles", "reproducibility_checksum"]
    )
    provisional["reproducibility_checksum"] = reproducibility_checksum(provisional)
    return provisional


def build_test_artifact(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build complete private evidence for reducer and mutation tests."""

    return _base_artifact(
        root=root,
        rows=rows,
        validation_receipts=validation_receipts,
        source_hashes={},
        duration_s=1.0,
        requalification={
            "required": True,
            "passed": True,
            "stream_count": 1_000,
            "current_rust_source_sha256": "sha256:" + "1" * 64,
            "current_rust_binary_sha256": "sha256:" + "2" * 64,
            "historical_rust_source_sha256": "sha256:" + "3" * 64,
            "historical_rust_binary_sha256": None,
            "equal_durability_policy": DURABILITY_POLICY,
        },
    )


def build_blocked_artifact(
    blocker: Mapping[str, Any],
    root: Path,
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Close an external pre-gate failure without calling the service."""

    return _base_artifact(
        root=root,
        rows=[],
        validation_receipts=validation_receipts,
        source_hashes={},
        duration_s=0.01,
        requalification={
            "required": True,
            "passed": True,
            "stream_count": 0,
            "current_rust_source_sha256": None,
            "current_rust_binary_sha256": None,
            "historical_rust_source_sha256": None,
            "historical_rust_binary_sha256": None,
            "equal_durability_policy": DURABILITY_POLICY,
        },
        blocker=blocker,
    )


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    for label, row in (value.get("source_artifact_hashes") or {}).items():
        if not isinstance(row, Mapping):
            errors.append(f"source_row_invalid:{label}")
            continue
        path = Path(str(row.get("path") or label))
        resolved = path if path.is_absolute() else root.resolve() / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != row.get("sha256"):
            errors.append(f"source_hash_invalid:{label}")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, root: Path, verify_sources: bool = True
) -> list[str]:
    """Reject identity, reduction, custody, claim, or receipt drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_mismatch")
    if value.get("run_date") != RUN_DATE or value.get("milestone") != MILESTONE:
        errors.append("task_binding_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if value.get("no_model_load") is not True:
        errors.append("no_model_load_invalid")
    counts = value.get("invocation_counts")
    if counts != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_claim_invalid")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class_invalid")
    if value.get("planned_inference_substrate_class") != "no_model_load":
        errors.append("planned_inference_substrate_class_invalid")
    if value.get("production_default_changed") is not False:
        errors.append("production_default_changed_invalid")
    if value.get("calibration_quality_claimed") is not False:
        errors.append("calibration_quality_claim_invalid")
    if value.get("empirical_head_installed") is not False:
        errors.append("empirical_head_install_invalid")
    if value.get("arc_agent_changed") is not False:
        errors.append("arc_default_changed_invalid")
    if value.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_declaration_invalid")
    if value.get("rows") != value.get("consumer_comparison_rows"):
        errors.append("comparison_rows_mismatch")
    try:
        reduction = independent_reduce(value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        reduction = {}
        errors.append("independent_reduction_failed")
    if value.get("independent_reduction") != reduction:
        errors.append("independent_reduction_mismatch")
    if value.get("comparison_summary") != reduction.get("comparison_summary"):
        errors.append("comparison_summary_mismatch")
    for field in ("consumer_ready_score", "consumer_speed_benefit_score"):
        observed = value.get(field)
        if not isinstance(observed, int) or isinstance(observed, bool):
            errors.append(f"score_not_bare_numeric:{field}")
        elif observed != reduction.get(field):
            errors.append(f"{field}_mismatch")
    expected_gates = acceptance_gates(value, reduction) if reduction else []
    if value.get("acceptance_gate_results") != expected_gates:
        errors.append("acceptance_gate_results_mismatch")
    blocker = value.get("gate_check_summary")
    blocked = value.get("verdict_class") == "blocked"
    if blocked:
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not isinstance(blocker, Mapping) or not required.issubset(blocker):
            errors.append("external_blocker_incomplete")
        expected_verdict = (
            f"complete_blocked_{blocker.get('check')}" if isinstance(blocker, Mapping) else None
        )
        if value.get("honest_verdict") != expected_verdict:
            errors.append("terminal_verdict_mismatch")
        if value.get("consumer_ready_score") != 0 or value.get("consumer_speed_benefit_score") != 0:
            errors.append("blocked_measurement_must_be_unready")
    else:
        if blocker is not None:
            errors.append("unblocked_gate_summary_must_be_null")
        speed = reduction.get("consumer_speed_benefit_score") == 1
        expected_class = "positive" if speed else "null"
        expected_verdict = (
            "complete_positive_rust_consumer_speed_benefit"
            if speed
            else "complete_null_rust_consumer_ready_speed_gate_failed"
        )
        if (
            value.get("verdict_class") != expected_class
            or value.get("honest_verdict") != expected_verdict
        ):
            errors.append("terminal_verdict_mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial_invalid")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        key not in principles for key in (*value.keys(), "reproducibility_checksum")
    ):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    requalification = value.get("current_binary_requalification")
    if not isinstance(requalification, Mapping) or (
        not blocked and requalification.get("passed") is not True
    ):
        errors.append("current_binary_requalification_invalid")
    if verify_sources:
        errors.extend(_verify_sources(value, root))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path, verify_sources: bool = True) -> list[str]:
    value = load_object(path)
    if not value:
        return ["artifact_not_object"]
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def independent_replay(path: Path, *, root: Path, verify_sources: bool = True) -> list[str]:
    value = load_object(path)
    if not value:
        return ["artifact_not_object"]
    errors = validate_artifact(value, root=root, verify_sources=verify_sources)
    if value.get("independent_reduction") != independent_reduce(value):
        errors.append("independent_reduction_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze affected Python, Rust, specification, and foreign-CWD checks."""

    basetemp = private_root / "basetemp"
    coverage_file = private_root / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(), CALIBRATION_TEST_PATH.as_posix()),
        (MODULE_PATH.as_posix(), CLIENT_PATH.as_posix(), CALIBRATION_PATH.as_posix()),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    smoke_code = (
        "import os,sys; from pathlib import Path; "
        "root=Path(sys.argv[1]).resolve(); os.chdir('/tmp'); "
        "sys.path[:0]=[str(root/'python'),str(root)]; "
        "from carnot.experiment_7598_v663_rust_consumer import consumer_smoke; "
        "raise SystemExit(consumer_smoke(root,Path(sys.argv[2])))"
    )
    commands.extend(
        [
            validation_scope.CommandSpec(
                "rust_binary_tests",
                (
                    "cargo",
                    "test",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                ),
                "changed_rust_binary",
                600.0,
            ),
            validation_scope.CommandSpec(
                "rust_fmt",
                (
                    "rustfmt",
                    "--edition",
                    "2021",
                    "--check",
                    RUST_SOURCE_PATH.as_posix(),
                ),
                "changed_rust_source",
                300.0,
            ),
            validation_scope.CommandSpec(
                "rust_clippy",
                (
                    "cargo",
                    "clippy",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                    "--",
                    "-D",
                    "warnings",
                    "-A",
                    "clippy::needless_late_init",
                ),
                "changed_rust_binary_with_unrelated_known_lint_isolated",
                900.0,
            ),
            validation_scope.CommandSpec(
                "foreign_cwd_consumer_integration",
                (
                    str(root / ".venv/bin/python"),
                    "-u",
                    "-c",
                    smoke_code,
                    str(root),
                    str(private_root / "foreign-cwd-smoke"),
                ),
                "public_client_foreign_cwd",
                300.0,
            ),
        ]
    )
    return commands


def private_basetemps(commands: Sequence[validation_scope.CommandSpec]) -> list[str]:
    paths = [
        argument.split("=", 1)[1]
        for command in commands
        for argument in command.argv
        if argument.startswith("--basetemp=")
    ]
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return paths


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    common = ("--date", RUN_DATE, "--root", str(root.resolve()))
    return [
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (
                str(root / ".venv/bin/python"),
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (
                str(root / ".venv/bin/python"),
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[2],
            (str(root / ".venv/bin/python"), "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[3],
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _provisional_terminal_receipts() -> list[JsonDict]:  # pragma: no cover
    return [
        {
            "name": name,
            "command": "pending exact terminal candidate check",
            "command_argv": ["pending", name],
            "scope": "private_provisional_candidate",
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
            "output_tail": "private provisional receipt; never published",
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _run_commands(
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
    *,
    coverage_file: Path | None = None,
) -> list[JsonDict]:  # pragma: no cover
    receipts: list[JsonDict] = []
    for index, command in enumerate(commands):
        private_basetemps([command])
        environment = {"COVERAGE_FILE": str(coverage_file)} if coverage_file else None
        receipts.extend(
            validation_scope.run_commands(
                root,
                [command],
                log_dir=log_dir / f"{index:02d}_{command.name}",
                extra_env=environment,
                heartbeat_s=60.0,
            )
        )
    return receipts


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7598] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def consumer_smoke(root: Path, scratch: Path) -> int:  # pragma: no cover
    """Exercise predict, release, durable reload, restart, and duplicate rejection."""

    scratch.mkdir(parents=True, exist_ok=True)
    state = scratch / "session.json"
    with CalibratedDecisionService(
        state_path=state,
        binary_path=root / RUST_BINARY,
        response_timeout_s=5.0,
        cwd=root,
    ) as service:
        prediction = service.predict("smoke-event", 0.12)
        acknowledgment = service.release_feedback("smoke-event", 1)
    with CalibratedDecisionService(
        state_path=state,
        binary_path=root / RUST_BINARY,
        response_timeout_s=5.0,
        cwd=root,
    ) as restarted:
        resumed = restarted.predict("smoke-resumed", 0.37)
        duplicate = restarted.release_feedback("smoke-event", 1)
    return int(
        not (
            prediction.available
            and acknowledgment.durable
            and resumed.available
            and duplicate.error == "duplicate_feedback:smoke-event"
            and upstream._load_state(state).sample_count == 1
        )
    )


def _extra_sources(
    root: Path, raw_paths: Sequence[Path]
) -> dict[str, JsonDict]:  # pragma: no cover
    sources: dict[str, JsonDict] = {}
    for relative in (
        MODULE_PATH,
        CLIENT_PATH,
        CALIBRATION_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        RUST_SOURCE_PATH,
        RUST_BINARY,
        CL_SPEC_PATH,
        VERIFY_SPEC_PATH,
        UPSTREAM_PATH,
    ):
        sources[relative.as_posix()] = source_row(root / relative, root)
    for path in raw_paths:
        sources[path.relative_to(root).as_posix()] = source_row(path, root)
    return sources


def _add_extras(value: JsonDict, extras: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    value.update(deepcopy(dict(extras)))
    value["field_principles"] = field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def run_experiment(  # pragma: no cover - the declared entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Requalify, benchmark, validate, and atomically publish the result."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    repo = root.resolve()
    destination = output_path or repo / RESULT_PATH
    raw_root = repo / RAW_DIR
    raw_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="carnot-exp7598-work-", dir="/tmp"))
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7598-validation-", dir="/tmp"))
    parity_path = raw_root / "current_binary_parity_rows.json"
    rows_path = raw_root / "consumer_comparison_rows.json"
    manifest_path = raw_root / "affected_validation_manifest.json"
    candidate_path = raw_root / "measured_terminal_candidate.json"
    exact_path = raw_root / "exact_terminal_candidate.json"
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    atomic_json(
        raw_root / "preconditions_checkpoint.json",
        {"rows": context["rows"], "blocker": context["blocker"]},
    )
    spans.append(_span("preconditions", phase_started, started, len(context["rows"])))
    progress(
        started,
        "preconditions",
        "complete",
        completed=len(context["rows"]),
        blocked=context["blocker"] is not None,
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after", completed=0)

    build_receipts: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    consumer_rows: list[JsonDict] = []
    parity_summary: JsonDict = {"passed": False, "stream_count": 0, "blocked": True}
    if context["blocker"] is None:
        build_commands = [
            validation_scope.CommandSpec(
                "rust_release_build",
                (
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                ),
                "current_rust_binary",
                900.0,
            )
        ]
        progress(started, "rust_build", "before_subprocess", planned=1)
        phase_started = time.monotonic()
        build_receipts = _run_commands(repo, build_commands, raw_root / "validation/build")
        spans.append(_span("rust_build", phase_started, started, len(build_receipts)))
        progress(
            started,
            "rust_build",
            "after_subprocess",
            completed=len(build_receipts),
            passed=all(row.get("passed") is True for row in build_receipts),
        )
        if not all(row.get("passed") is True for row in build_receipts):
            raise RuntimeError("rust_release_build_failed")

        progress(started, "requalification", "before", planned=upstream.PARITY_STREAM_COUNT)
        phase_started = time.monotonic()
        parity_rows, parity_summary = upstream.run_parity_suite(
            repo, work_root / "parity", started=started
        )
        spans.append(_span("requalification", phase_started, started, upstream.PARITY_STREAM_COUNT))
        progress(
            started,
            "requalification",
            "after",
            completed=upstream.PARITY_STREAM_COUNT,
            passed=parity_summary["passed"],
        )
        if parity_summary["passed"] is not True:
            raise RuntimeError("current_binary_parity_failed")
        atomic_json(parity_path, {"summary": parity_summary, "rows": parity_rows})

        progress(
            started,
            "benchmark",
            "before",
            planned=REPEATS * len(MODES) * len(BATCH_SIZES),
        )
        phase_started = time.monotonic()
        consumer_rows = measure_consumer_rows(repo, work_root / "benchmark", started=started)
        comparison_summary = summarize_consumer_rows(consumer_rows)
        spans.append(
            _span(
                "benchmark",
                phase_started,
                started,
                REPEATS * len(MODES) * len(BATCH_SIZES),
            )
        )
        progress(
            started,
            "benchmark",
            "after",
            completed=REPEATS * len(MODES) * len(BATCH_SIZES),
            minimum_lower95=min(
                row["paired_ratio"]["lower95"] for row in comparison_summary.values()
            ),
        )
        atomic_json(rows_path, {"summary": comparison_summary, "rows": consumer_rows})
    else:
        progress(started, "rust_build", "skipped", reason="external_blocker")
        progress(started, "requalification", "after", completed=0, blocked=True)
        progress(started, "benchmark", "after", completed=0, blocked=True)

    current_source_hash = sha256_file(repo / RUST_SOURCE_PATH)
    current_binary_hash = (
        sha256_file(repo / RUST_BINARY) if (repo / RUST_BINARY).is_file() else None
    )
    requalification = {
        "required": context["requalification_required"],
        "passed": parity_summary.get("passed") is True,
        "stream_count": parity_summary.get("stream_count", 0),
        "max_probability_absolute_error": parity_summary.get("max_probability_absolute_error"),
        "typed_decision_mismatch_count": parity_summary.get("typed_decision_mismatch_count"),
        "lifecycle_controls": parity_summary.get("lifecycle_controls", {}),
        "historical_rust_source_sha256": context["historical_rust_source_sha256"],
        "historical_rust_binary_sha256": context["historical_rust_binary_sha256"],
        "current_rust_source_sha256": current_source_hash,
        "current_rust_binary_sha256": current_binary_hash,
        "equal_durability_policy": DURABILITY_POLICY,
    }

    progress(started, "manifest", "start")
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": [TEST_PATH.as_posix(), CALIBRATION_TEST_PATH.as_posix()],
            "changed_modules": [
                MODULE_PATH.as_posix(),
                CLIENT_PATH.as_posix(),
                CALIBRATION_PATH.as_posix(),
            ],
            "python_static_paths": [WRAPPER_PATH.as_posix()],
            "rust_paths": [RUST_SOURCE_PATH.as_posix()],
            "capability_specs": [CL_SPEC_PATH.as_posix(), VERIFY_SPEC_PATH.as_posix()],
        },
    )
    progress(started, "manifest", "complete")

    commands = build_validation_commands(repo, private_root)
    coverage_file = private_root / "coverage" / ".coverage"
    progress(started, "scoped_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = _run_commands(
        repo,
        commands,
        raw_root / "validation/affected",
        coverage_file=coverage_file,
    )
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=all(row.get("passed") is True for row in affected),
    )
    if not all(row.get("passed") is True for row in affected):
        raise RuntimeError("required_scoped_validation_failed")

    raw_sources = [path for path in (parity_path, rows_path, manifest_path) if path.is_file()]
    sources = deepcopy(context["source_artifact_hashes"])
    sources.update(_extra_sources(repo, raw_sources))
    comparison_summary = summarize_consumer_rows(consumer_rows) if consumer_rows else {}
    common_extras: JsonDict = {
        "preconditions_checked": deepcopy(context["rows"]),
        "resource_ownership": deepcopy(context["resource_observations"]),
        "exact_worker_identity": {
            "rust_source_sha256": current_source_hash,
            "rust_binary_sha256": current_binary_hash,
            "python_reference_sha256": sha256_file(
                repo / "python/carnot/experiment_7585_v662_portable_service.py"
            ),
        },
        "upstream_authentication": {
            "experiment": "Exp7585",
            "artifact_sha256": sha256_file(repo / UPSTREAM_PATH),
            "portable_parity_score": context["exp7585"].get("portable_parity_score"),
            "terminal_validation": set(upstream.TERMINAL_CHECK_NAMES).issubset(
                _receipt_names(context["exp7585"])
            ),
            "equal_durability": context["historical_equal_durability"],
        },
        "phase_spans": spans,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "raw_measurement_receipts": {
            path.relative_to(repo).as_posix(): source_row(path, repo) for path in raw_sources
        },
        "e2e_applicability": {
            "subprocess_service_integration": "required_and_run",
            "E2E-003": "not_applicable_no_pyo3_binding",
            "E2E-004": "not_applicable_no_safetensors_cross_language_serialization",
            "E2E-009_through_013": "not_applicable_arc_agent_unchanged",
            "immutable_evidence_output_boundary": "unchanged_not_exercised_by_non_arc_client",
        },
        "kernel_and_consumer_scope": {
            "local_kernel_reported_separately": True,
            "whole_consumer_reported": True,
            "board_claim": False,
            "hundred_x_hardware_claim": False,
        },
        "comparison_summary": comparison_summary,
    }

    def build_current(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
        artifact = _base_artifact(
            root=repo,
            rows=consumer_rows,
            validation_receipts=receipts,
            source_hashes=sources,
            duration_s=time.monotonic() - started,
            requalification=requalification,
            blocker=context["blocker"],
        )
        return _add_extras(artifact, common_extras)

    provisional = build_current([*build_receipts, *affected, *_provisional_terminal_receipts()])
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, provisional)
    progress(started, "candidate", "after_serialization", bytes=candidate_path.stat().st_size)

    terminal_plan = terminal_commands(candidate_path, repo)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = _run_commands(repo, terminal_plan, raw_root / "validation/terminal_provisional")
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_candidate_validation_failed")

    common_extras["phase_spans"] = spans
    common_extras["completed_at_utc"] = datetime.now(UTC).isoformat()
    common_extras["terminal_reader_outcomes"] = {
        row["name"]: {"passed": row["passed"], "log_sha256": row["log_sha256"]} for row in terminal
    }
    final = build_current([*build_receipts, *affected, *terminal])
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(exact_path, final)

    exact_plan = terminal_commands(exact_path, repo)
    progress(started, "exact_candidate_validation", "before_subprocesses", planned=len(exact_plan))
    exact = _run_commands(repo, exact_plan, raw_root / "validation/terminal_exact")
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed=len(exact),
        passed=all(row.get("passed") is True for row in exact),
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(raw_root / "exact_terminal_validation_receipts.json", {"receipts": exact})

    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact_path):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        bytes=destination.stat().st_size,
        verdict=final["honest_verdict"],
    )
    shutil.rmtree(work_root)
    shutil.rmtree(private_root)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed producer, worker, and read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--python-service-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    args.root = args.root.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode and keep the repository wrapper thin."""

    args = parse_args(argv)
    if args.python_service_worker:
        return python_service_worker_loop(sys.stdin, sys.stdout)
    verify_sources = not args.no_source_check
    if args.cold_replay is not None:
        errors = cold_replay(
            args.cold_replay.resolve(), root=args.root, verify_sources=verify_sources
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(
            args.independent_reduce.resolve(), root=args.root, verify_sources=verify_sources
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    output = args.output if args.output.is_absolute() else args.root / args.output
    run_experiment(args.root, args.date, output_path=output)
    return 0
