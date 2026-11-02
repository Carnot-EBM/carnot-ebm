"""REQ-REPORT-8533: primitive native measurements determine a finite qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np

from carnot.reporting import nine_input_inputs_8533 as inputs
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.verify import native_nine_input_8533 as build
from carnot.verify import nine_input_8533 as numeric

Json = dict[str, Any]
ROOT = inputs.ROOT
NAME = "experiment_8533_v733_nine_input_native_qualification"
TASK = "exp8533-nine-input-native-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_nine_input_8533.py"
OWNED = [
    f"python/carnot/{p}.py"
    for p in [
        "verify/nine_input_8533",
        "verify/native_nine_input_8533",
        "reporting/nine_input_inputs_8533",
        "reporting/nine_input_qualification_8533",
        "reporting/nine_input_runner_8533",
        "reporting/nine_input_validation_8533",
    ]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries and actual completed counts."""
    print(f"[exp8533] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """The terminal envelope binds every field except the checksum itself."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def diagnostic(path: Path, field: str, observed: Any, *, expected: Any = 1) -> Json:
    """Missing is null, while measured zero remains an explicit observation."""
    return dict(
        check=field,
        upstream="exp8526-contract-methods",
        path=str(path),
        sha256=inputs.reference(path)["sha256"] if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def measure(root: Path, raw: Path) -> Json:
    """Authenticate first, then run actual native controls and complete request smoke arms."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions")
    tools = {n: shutil.which(n) for n in ["cargo", "rustc", "llvm-cov", "llvm-profdata"]}
    tools.update(
        {
            n: shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        }
    )
    memory = (
        int(
            next(
                l.split()[1]
                for l in Path("/proc/meminfo").read_text().splitlines()
                if l.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    mount = max(
        (
            line.split()
            for line in Path("/proc/mounts").read_text().splitlines()
            if Path("/var/tmp").resolve().is_relative_to(Path(line.split()[1]))
        ),
        key=lambda row: len(row[1]),
    )
    pre: Json = dict(
        tool_paths=tools,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage("/var/tmp").free,
        writable_scratch=os.access("/var/tmp", os.W_OK),
        disk_backed_scratch="/var/tmp",
        scratch_filesystem=mount[2],
        scratch_is_disk_backed=mount[2] not in {"tmpfs", "ramfs"},
        task_cap_s=4800,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        CUDA_required=False,
        board_required=False,
    )
    work: Json = dict(
        rows=[],
        failures=[],
        owned_failure=False,
        preconditions=pre,
        source_refs=[],
        started_monotonic_ns=start,
        raw=str(raw.absolute()),
        extension={},
        timing_consumer_rows=[],
        historical_model_provenance=[],
    )
    try:
        head, tape, refs, upstream = inputs.operands(root, raw)
        work.update(
            source_refs=refs,
            historical_model_provenance=upstream["historical_model_provenance"],
            head_reference=inputs.reference(raw / "head.json"),
            request_tape_reference=inputs.reference(raw / "request_tape.json"),
        )
    except (OSError, ValueError, KeyError, StopIteration) as error:
        path = (
            Path(error.filename)
            if isinstance(error, OSError) and error.filename
            else root / inputs.UPSTREAM
        )
        fields = (
            ["methods_ready_score", "nine_input_inputs_ready_score"]
            if path == root / inputs.UPSTREAM
            else ["original_sealed_input_available"]
        )
        for field in fields:
            observed = None if not path.exists() else str(error)
            if path == root / inputs.UPSTREAM and path.exists():
                observed = json.loads(path.read_bytes()).get(field)
            work["failures"].append(dict(diagnostic(path, field, observed), reason=str(error)))
    if (
        not all(tools.values())
        or memory < 268435456
        or int(pre["free_disk_bytes"]) < 104857600
        or not pre["writable_scratch"]
        or not pre["scratch_is_disk_backed"]
    ):
        work["failures"].append(
            diagnostic(Path("/var/tmp"), "tools_memory_scratch", pre, expected=True)
        )
    if not work["failures"]:
        progress("before_native_build")
        try:
            with TemporaryDirectory(prefix="carnot-8533-", dir="/var/tmp") as directory:
                native, ref = build.extension(Path(directory), raw / "native_logs")
                work["extension"] = ref
                progress("after_native_build")
                matrix = np.asarray([r["original_features"] for r in tape["rows"]], dtype=float)
                work["rows"] = numeric.qualify(head, matrix, native)
                progress("before_complete_request_smoke", 0, 128)
                timed = set(tape["timed_ids"])
                for row in [r for r in tape["rows"] if r["unit_id"] in timed]:
                    pair = []
                    for arm, provider in [("python", None), ("native", native)]:
                        began = time.perf_counter_ns()
                        response = numeric.request(head, row, provider)
                        pair.append(json.loads(response)["decision"])
                        work["timing_consumer_rows"].append(
                            dict(
                                unit_id=row["unit_id"],
                                arm=arm,
                                response=response,
                                duration_ns=time.perf_counter_ns() - began,
                            )
                        )
                    if pair[0] != pair[1]:
                        raise ValueError("downstream_typed_decision_mismatch")
                    if len(work["timing_consumer_rows"]) % 32 == 0:
                        progress(
                            "complete_request_smoke",
                            len(work["timing_consumer_rows"]),
                            128 - len(work["timing_consumer_rows"]),
                        )
                progress("after_complete_request_smoke", 128)
        except (OSError, ValueError, KeyError, FloatingPointError) as error:
            work["owned_failure"] = True
            work["failures"].append(
                diagnostic(raw, "native_measurement", str(error), expected="finite_parity")
            )
    work.update(
        ended_monotonic_ns=time.monotonic_ns(),
        code_refs=[
            inputs.reference(ROOT / p)
            for p in OWNED
            + build.SOURCES
            + [
                TEST,
                "python/carnot/verify/sparse_energy_7996.py",
                "python/carnot/verify/evidence_features_7980.py",
                "python/carnot/verify/v726_decision_metrics.py",
                "python/carnot/reporting/primary_publication.py",
                "crates/carnot-core/src/lib.rs",
                "crates/carnot-python/src/lib.rs",
                "crates/carnot-python/Cargo.toml",
            ]
        ],
    )
    atomic_json(raw / "primitive_rows.json", dict(rows=work["rows"]))
    work["primitive_rows_reference"] = inputs.reference(raw / "primitive_rows.json")
    atomic_json(raw / "measurement.json", work)
    return work


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Only complete owned validation and measured finite parity can release readiness."""
    rows = work["rows"]
    failed = sum(not r["passed"] for r in rows)
    owned = work["owned_failure"] or failed > 0 or any(not r["passed"] for r in receipts)
    blocked = bool(work["failures"]) and not owned
    ready = bool(rows) and not owned and not blocked and bool(receipts)
    verdict = (
        "disqualified"
        if owned or not blocked and not ready
        else "blocked"
        if blocked
        else "circular_positive"
    )
    finite = [r for r in rows if "residual_error" in r]
    reported = [
        dict(
            r,
            family="finite_numerical_control",
            status="completed" if r["passed"] else "failed",
            absolute_metric=int(r["passed"]),
            raw_numerator=int(r["passed"]),
            raw_denominator=1,
            censored=False,
            excluded=False,
        )
        for r in rows
    ]
    if not reported:
        reported = [
            dict(
                unit_id=f"unavailable-release-slot-{i}",
                family="unavailable_original_identity",
                status="unstarted",
                absolute_metric=None,
                raw_numerator=None,
                raw_denominator=1,
                censored=True,
                excluded=False,
                missing_reason="required_input_or_native_measurement_unavailable",
            )
            for i in range(900)
        ]
    value = dict(
        experiment_id=8533,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v733_nine_input_native_qualification",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        current_llm_invocations=0,
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        historical_model_provenance=work["historical_model_provenance"],
        rows=reported,
        intended_count=len(rows) or 900,
        completed_count=len(rows) - failed,
        failed_count=failed,
        censored_count=0 if rows else 900,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            released_answers=900,
            exposed_questions=300,
            timed_requests=64,
            machines=1,
            independence="finite deterministic numerical controls; no independent factual evaluation",
        ),
        verifier_is_oracle=True,
        exposure_scope="all 900 answers and 300 questions are exposed development history",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            finite_parity=bool(rows) and failed == 0,
            original_inputs=not blocked,
            complete_request_consumer=len(work["timing_consumer_rows"]) == 128,
            owned_validation=bool(receipts) and all(r["passed"] for r in receipts),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                name="authentication_native_parity_and_complete_request_smoke",
                duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
            )
        ],
        random_seed=7338533,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[work["primitive_rows_reference"]],
        cited_upstream_artifacts=work["source_refs"],
        primitive_reference=inputs.reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output.absolute()),
        nine_input_native_ready_score=int(ready),
        loaded_extension_reference=work["extension"],
        native_execution_count=sum(r["native_execution_count"] for r in rows)
        + sum(r["arm"] == "native" for r in work["timing_consumer_rows"]),
        finite_parity_rows=len(finite),
        boundary_fallback_rows=[r["unit_id"] for r in finite if r["boundary_fallback"]],
        max_residual_error=max((r["residual_error"] for r in finite), default=None),
        max_probability_error=max((r["probability_error"] for r in finite), default=None),
        typed_decision_mismatch_count=sum(r["decision_mismatch"] for r in finite),
        request_tape_reference=work.get("request_tape_reference"),
        head_reference=work.get("head_reference"),
        compiler_flags=build.FLAGS,
        numerical_scope="Finite binary64 panel only; no universal error certificate, speed or Rust/Python NFR-01 claim.",
        measurement_kernel_reference=work["code_refs"][0],
        complete_request_rows=work["timing_consumer_rows"],
        historical_five_input_kernel="preserved, separately versioned 5-input/34-coefficient store; consumer mismatch and append result unchanged",
        methodology_note="Original V726 nine-input head and fit-only scaler. Dense independent SciPy versus actual Rust residual; Python retains policy. Cached source text is recomputed per request. No fitting, sampling, model loads or new labels. Exposed finite numerical qualification does not establish factual generalization or speed benefit.",
    )
    value["field_principles"] = {
        k: "Bind observed invocation evidence and finite numerical scope; imported provenance is not current generation."
        for k in value
    }
    value["field_principles"].update(
        nine_input_native_ready_score="One only after native calls, finite controls, complete request consumer and all required validation pass; no speed or learning claim.",
        native_execution_count="Actual residual binding calls in the frozen finite controls and native request smoke; validation/replay calls are not added to scientific sample size.",
        finite_parity_rows="Finite answer, knot, decay and threshold comparisons; malformed rejection controls are counted separately in rows.",
        boundary_fallback_rows="IDs using the Python reference within 1e-10 of .2 or .8, including explicit controlled threshold neighbors.",
        max_residual_error="Maximum absolute dense-SciPy versus native residual difference; every row must satisfy 1e-12*(1+abs(reference)).",
        max_probability_error="Maximum absolute stable two-energy probability difference under unchanged Python policy.",
        typed_decision_mismatch_count="Accept/reject/escalate disagreements after the threshold guard; readiness requires zero.",
        head_reference="Exact frozen seed17 V726 109-coefficient spline with fitting-only scaler; original Exp8431 producer seal remains in source_artifact_hashes.",
        request_tape_reference="All900 original stable-order text/p2 records; first64 test answer IDs define future complete timing requests.",
        loaded_extension_reference="Exact module.__file__, binary/source hashes, compiler flags and real build receipts; no Python mirror.",
        complete_request_rows="128 feature-through-serialization smoke spans from64 requests in two arms; no speed benefit estimate.",
        acceptance_gates="All finite controls, independently sealed original inputs, complete request consumer and required owned validation must pass together.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold computation uses original independent seals, so repaired local hashes do not help."""
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        work = inputs.read(value["primitive_reference"])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_sha256" in receipt
                    and inputs.reference(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        for ref in work["code_refs"] + work["source_refs"] + [work["primitive_rows_reference"]]:
            if inputs.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                return False
        if work["rows"]:
            with TemporaryDirectory(prefix="exp8533-replay-", dir="/var/tmp") as directory:
                head, tape, _, _ = inputs.operands(ROOT, Path(directory))
                if head != inputs.read(work["head_reference"]) or tape != inputs.read(
                    work["request_tape_reference"]
                ):
                    return False
            ref = work["extension"]
            if inputs.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                return False
            native = build.load(Path(ref["path"]))
            rows = numeric.qualify(
                head,
                np.asarray([r["original_features"] for r in tape["rows"]], dtype=float),
                native,
            )
            if (
                rows != work["rows"]
                or inputs.read(work["primitive_rows_reference"])["rows"] != rows
            ):
                return False
            timed = set(tape["timed_ids"])
            responses = [
                (r["unit_id"], arm, numeric.request(head, r, provider))
                for r in tape["rows"]
                if r["unit_id"] in timed
                for arm, provider in [("python", None), ("native", native)]
            ]
            if responses != [
                (r["unit_id"], r["arm"], r["response"]) for r in work["timing_consumer_rows"]
            ]:
                return False
        expected = artifact(work, value["validation_receipts"], Path(value["publication_output"]))
        return bool(expected == value)
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def consume(path: Path) -> Json:
    """The timing consumer freezes exact native bytes, complete request tape and original head."""
    value = json.loads(path.read_bytes())
    ready = (
        value["nine_input_native_ready_score"] == 1
        and value["required_checks_passed"]
        and replay(path)
    )
    if not ready:
        raise ValueError("nine_input_native_not_ready")
    return dict(
        ready=True,
        extension=value["loaded_extension_reference"],
        request_tape=value["request_tape_reference"],
        head=value["head_reference"],
        measurement_kernel=value["measurement_kernel_reference"],
    )
