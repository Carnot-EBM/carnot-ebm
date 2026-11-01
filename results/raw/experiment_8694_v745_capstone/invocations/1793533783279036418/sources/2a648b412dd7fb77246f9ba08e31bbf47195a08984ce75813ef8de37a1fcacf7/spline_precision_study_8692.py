"""REQ-REPORT-8692: original checkpoint custody and finite CPU arithmetic controls."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from carnot.verify import spline_precision_8692 as numeric
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8692_v745_spline_precision_feasibility"
TASK = "exp8692-spline-precision-feasibility"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_spline_precision_8692.py",
    "tests/python/test_spline_precision_runner_8692.py",
]
OWNED = [
    "python/carnot/verify/spline_precision_8692.py",
    "python/carnot/reporting/spline_precision_study_8692.py",
    "python/carnot/reporting/spline_precision_runner_8692.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v744-methods-protocol.json"
PROTOCOL_PIN = "sha256:842e28d27098b7dadf41290c407e72cfa4019a2cbb09429c492355aa0f55cb67"
__all__ = [
    "ROOT",
    "NAME",
    "TASK",
    "CLI",
    "TESTS",
    "OWNED",
    "MODEL_SPECS",
    "PROTOCOL",
    "numeric",
    "progress",
    "reference",
    "read",
    "check",
    "authenticate",
    "evaluate",
    "build",
    "sha256_file",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Give the supervisor completed counts without manufacturing elapsed time."""
    print(f"[exp8692] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Record missing paths explicitly so absence cannot be mistaken for zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def read(ref: Json) -> Json:
    """A hash authenticates original bytes before any structured field is consumed."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("changed_bytes")
    return dict(json.loads(path.read_bytes()))


def check(
    ref: Json, field: str, expected: Any, upstream: str = "exp8431-criterion-energy-fit"
) -> Json:
    """Retain the exact operand state at this gate, including observed numeric zero."""
    observed: Any = None
    state = "missing_file"
    passed = False
    try:
        value = read(ref)
        state = (
            "missing_field"
            if field not in value
            else "observed_zero"
            if value[field] == 0
            else "observed"
        )
        observed = value.get(field)
        passed = field in value and type(observed) is type(expected) and observed == expected
    except ValueError:
        state = "changed_bytes"
    except OSError:
        pass
    return dict(
        upstream=upstream,
        check="authenticated_field",
        **ref,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        operand_state=state,
        passed=passed,
    )


def authenticate() -> Json:
    """Admit the numerical producer independently of the disqualified V744 contract."""
    from carnot.verify import v726_criterion_energy_fit as producer
    from carnot.reporting.primary_publication import validate_primary

    refs = [dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)]
    failures: list[Json] = []
    heads: list[Json] = []
    provenance: Json = {}
    try:
        protocol = read(refs[0])
        upstream_ref = protocol["original_producers"]["8431"]
        refs.extend([upstream_ref, protocol["head_checkpoint_reference"]])
        upstream = read(upstream_ref)
        gates = [
            check(upstream_ref, k, v)
            for k, v in [
                ("head_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]
        ]
        failures.extend(row for row in gates if not row["passed"])
        validate_primary(upstream, Path(upstream_ref["source_path"]))
        frozen = read(protocol["head_checkpoint_reference"])
        heads = frozen["heads"]["spline9"]
        if len(heads) != 5 or any(
            h["parameter_count"] != 109
            or h["arm"] != "spline"
            or len(h["parameters"]) != 109
            or len(h["scaler"]["minimum"]) != 9
            or len(h["scaler"]["maximum"]) != 9
            for h in heads
        ):
            raise ValueError("original_nine_input_head")
        work = read(upstream["primitive_reference"])
        refs.append(upstream["primitive_reference"])
        if (
            work["result"] != frozen
            or producer.build(
                work,
                upstream["validation_receipts"],
                Path(upstream["primitive_reference"]["path"]).parent,
                Path(upstream["publication_output"]),
            )
            != upstream
        ):
            raise ValueError("declared_producer_construction")
        terminal_ref = reference(Path(upstream["terminal_validation_sidecar_path"]))
        terminal = read(terminal_ref)
        bound = read(reference(Path(terminal["publication"]["sidecar_path"])))
        refs.extend([terminal_ref, reference(Path(terminal["publication"]["sidecar_path"]))])
        if (
            terminal["primary_sha256"] != upstream_ref["sha256"]
            or not terminal["consumer"]["passed"]
            or not bound["report"]["passed"]
            or bound["primary_sha256"] != upstream_ref["sha256"]
        ):
            raise ValueError("terminal_validation_binding")
        rule = numeric.original.RULE
        if (
            protocol["objective_updates"]["brier_derivative"] != rule["data"]
            or protocol["objective_updates"]["learning_rate"] != rule["rate"]
            or protocol["objective_updates"]["l2"] != rule["l2"]
            or not rule["bias_regularized"]
        ):
            raise ValueError("update_rule")
        provenance = upstream["historical_model_provenance"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream="exp8431-criterion-energy-fit",
                check="numerical_checkpoint_authentication",
                path=refs[-1]["path"],
                sha256=refs[-1].get("sha256"),
                field="producer_checkpoint_rule_terminal",
                operator="==",
                expected="authenticated",
                observed=str(error),
                passed=False,
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "changed_or_invalid_operand",
            )
        )
    return dict(
        heads=[] if failures else heads,
        refs=refs,
        failures=failures,
        historical_model_provenance=provenance,
    )


def evaluate(heads: list[Json], frozen: Json) -> Json:
    """Retain every intended control, including descriptive failures in smaller formats."""
    precision: list[Json] = []
    trajectories: list[Json] = []
    summaries: list[Json] = []
    fallbacks: list[Json] = []
    spans: list[Json] = []
    final_states: list[Json] = []
    for hi, head in enumerate(heads):
        for bits, fractional in numeric.FORMATS:
            progress(
                f"benchmark_before_head{hi}_{bits}_{fractional}",
                hi * 3 + numeric.FORMATS.index((bits, fractional)),
                len(heads) * 3 - hi * 3 - numeric.FORMATS.index((bits, fractional)),
            )
            f = numeric.Fixed(bits, fractional)
            initial = numeric.encode(head, f)
            rows: list[Json] = []
            for sequence, labels in [("one_step", [0, 1]), *sorted(frozen["sequences"].items())]:
                state, reference_head = deepcopy(initial), deepcopy(head)
                count = 512 if sequence == "one_step" else 256
                for index in range(count):
                    vector_id = index // 2 if sequence == "one_step" else index
                    label = labels[index % len(labels)]
                    x = frozen["vectors"][vector_id]["x"]
                    if sequence == "one_step":
                        state, reference_head = deepcopy(initial), deepcopy(head)
                    previous_probability, previous_receipt = numeric.probability(state, x, f)
                    spans.extend(previous_receipt["spans"])
                    start = time.perf_counter_ns()
                    previous_float = float(
                        numeric.original.sparse.predict(reference_head, np.array([x]))[0]
                    )
                    reference_head, _ = numeric.original.update(reference_head, np.array(x), label)
                    probability_float = float(
                        numeric.original.sparse.predict(reference_head, np.array([x]))[0]
                    )
                    end = time.perf_counter_ns()
                    spans.append(
                        dict(
                            operation="float64_reference",
                            start_ns=start,
                            end_ns=end,
                            duration_ns=end - start,
                        )
                    )
                    state, receipt = numeric.step(state, x, label, f)
                    probability, prediction = numeric.probability(state, x, f)
                    spans.extend(receipt["spans"] + prediction["spans"])
                    error = max(
                        abs(probability - probability_float),
                        abs(previous_probability - previous_float),
                    )
                    margin = min(abs(probability_float - 0.2), abs(probability_float - 0.8))
                    fallback = (
                        min(margin, abs(previous_float - 0.2), abs(previous_float - 0.8)) <= 0.001
                    )
                    start = time.perf_counter_ns()
                    action = numeric.decision(probability)
                    if fallback:
                        probability_fallback = float(
                            numeric.original.sparse.predict(reference_head, np.array([x]))[0]
                        )
                        action = numeric.decision(probability_fallback)
                    end = time.perf_counter_ns()
                    spans.append(
                        dict(
                            operation="host_float64_fallback_and_decision",
                            start_ns=start,
                            end_ns=end,
                            duration_ns=end - start,
                        )
                    )
                    row = dict(
                        head_index=hi,
                        bits=bits,
                        fractional=fractional,
                        sequence=sequence,
                        vector_id=vector_id,
                        update_index=index,
                        label=label,
                        probability=probability,
                        float64_probability=probability_float,
                        probability_error=error,
                        effective_coefficient_error=float(
                            np.max(
                                np.abs(
                                    np.array(numeric.effective(state, f))
                                    - numeric.original.sparse.parameters(reference_head)
                                )
                            )
                        ),
                        decay_scale_error=abs(
                            state["decay_scale"] / f.unit - reference_head["decay_scale"]
                        ),
                        local_writes=receipt["local_writes"],
                        global_decay_writes=receipt["global_decay_writes"],
                        renormalization_writes=receipt["renormalization_writes"],
                        fallback=fallback,
                        decision=action,
                        float64_decision=numeric.decision(probability_float),
                        decision_parity=action == numeric.decision(probability_float),
                        finite=all(
                            math.isfinite(v)
                            for v in [
                                probability,
                                probability_float,
                                previous_probability,
                                previous_float,
                                *numeric.effective(state, f),
                            ]
                        ),
                        state_hash=canonical_hash(state),
                        effective_coefficient_abs_max=max(
                            abs(v) for v in numeric.effective(state, f)
                        ),
                        stored_coefficient_abs_max=max(abs(v) for v in state["parameters"]),
                        residual=prediction["residual"],
                    )
                    rows.append(row)
                    (precision if sequence == "one_step" else trajectories).append(row)
                    if fallback:
                        fallbacks.append(
                            dict(row, fallback_cost_ns=end - start, fallback="original_float64")
                        )
                    if (index + 1) % 128 == 0:
                        progress(f"{bits}_{fractional}_{sequence}", index + 1, count - index - 1)
                start = time.perf_counter_ns()
                encoded = numeric.serialize(state, f)
                end = time.perf_counter_ns()
                spans.append(
                    dict(
                        operation="host_serialization",
                        start_ns=start,
                        end_ns=end,
                        duration_ns=end - start,
                    )
                )
                final_states.append(
                    dict(
                        head_index=hi,
                        bits=bits,
                        sequence=sequence,
                        encoded_hex=encoded.hex(),
                        bytes=len(encoded),
                    )
                )
            summaries.append(
                dict(
                    head_index=hi,
                    bits=bits,
                    fractional=fractional,
                    **f.counts,
                    operations=f.operations,
                    accumulator_abs_max=f.maximum,
                    accumulator_limit=2**63 - 1,
                    conservative_product_sum_bound=37 * (2 ** (bits - 1)) ** 2,
                    maximum_probability_error=max(r["probability_error"] for r in rows),
                    maximum_coefficient_error=max(r["effective_coefficient_error"] for r in rows),
                    maximum_decay_error=max(r["decay_scale_error"] for r in rows),
                    maximum_local_writes=max(r["local_writes"] for r in rows),
                    fallback_count=sum(r["fallback"] for r in rows),
                    fallback_rate=sum(r["fallback"] for r in rows) / len(rows),
                    serialized_state_bytes=111 * (bits // 8) + 18 * 8,
                    coefficient_abs_max=max(r["effective_coefficient_abs_max"] for r in rows),
                    stored_coefficient_integer_abs_max=max(
                        r["stored_coefficient_abs_max"] for r in rows
                    ),
                    host_snapshot_coefficient_copies=109 * len(rows),
                    temperature=state["temperature"] / f.unit,
                    final_decay_scale=state["decay_scale"] / f.unit,
                    status="completed",
                    censored=False,
                    excluded=False,
                    state_encoding="111 signed little-endian fixed-point integers plus18 binary64 scaler bounds; schema external",
                    passed=f.counts["saturation"] == 0
                    and f.counts["overflow"] == 0
                    and all(
                        r["finite"]
                        and r["probability_error"] <= 0.001
                        and r["decision_parity"]
                        and r["local_writes"] <= 37
                        for r in rows
                    ),
                )
            )
            progress(
                f"benchmark_after_head{hi}_{bits}_{fractional}",
                hi * 3 + numeric.FORMATS.index((bits, fractional)) + 1,
                len(heads) * 3 - hi * 3 - numeric.FORMATS.index((bits, fractional)) - 1,
            )
    return dict(
        precision_rows=precision,
        trajectory_rows=trajectories,
        rows=summaries,
        initial_head_count=len(heads) or 5,
        serialized_final_states=final_states,
        primary_passed=bool(heads) and all(r["passed"] for r in summaries if r["bits"] == 32),
        saturation_counts=[
            {
                k: r[k]
                for k in [
                    "head_index",
                    "bits",
                    "saturation",
                    "overflow",
                    "decay_underflow",
                    "renormalization",
                ]
            }
            for r in summaries
        ],
        accumulator_bounds=[
            {
                k: r[k]
                for k in [
                    "head_index",
                    "bits",
                    "accumulator_abs_max",
                    "accumulator_limit",
                    "conservative_product_sum_bound",
                ]
            }
            for r in summaries
        ],
        threshold_fallback_rows=fallbacks,
        serialized_state_bytes=[
            dict(bits=b, bytes=111 * (b // 8) + 144) for b, _ in numeric.FORMATS
        ],
        cpu_operation_spans=spans,
        operation_inventory=[
            dict(
                operation=k,
                device="cpu",
                duration_ns=sum(v["duration_ns"] for v in spans if v["operation"] == k),
                calls=sum(v["operation"] == k for v in spans),
            )
            for k in sorted({v["operation"] for v in spans})
        ],
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """A pure reduction keeps raw measurements immutable during report construction."""
    external_bad = any(not r["passed"] and r.get("coverage_role") == "consumers" for r in receipts)
    owned_bad = any(
        not r["passed"]
        and r.get("coverage_role") != "consumers"
        and not (external_bad and r.get("name") == "original_disjoint_union")
        for r in receipts
    )
    validation = not owned_bad
    external_failures = [
        dict(
            upstream="existing_private_e2e",
            check=r.get("name"),
            path=r.get("stdout_path"),
            sha256=r.get("stdout_sha256"),
            field="actual_exit",
            operator="==",
            expected=r.get("expected_exit", 0),
            observed=r.get("actual_exit"),
            passed=False,
            operand_state="unchanged_external_validation_failure",
        )
        for r in receipts
        if not r["passed"] and r.get("coverage_role") == "consumers"
    ]
    verdict = (
        "disqualified"
        if not validation
        else "blocked"
        if work["failures"] or external_bad
        else "circular_positive"
        if work["primary_passed"]
        else "null"
    )
    ready = int(validation and not external_bad and not work["failures"] and work["primary_passed"])
    result = {
        k: deepcopy(work[k])
        for k in [
            "rows",
            "precision_rows",
            "trajectory_rows",
            "saturation_counts",
            "accumulator_bounds",
            "threshold_fallback_rows",
            "serialized_state_bytes",
            "cpu_operation_spans",
            "operation_inventory",
            "started_at_utc",
            "ended_at_utc",
            "duration_s",
            "phase_spans",
            "historical_model_provenance",
            "serialized_final_states",
        ]
    }
    result.update(
        experiment_id=8692,
        task_id=TASK,
        milestone="2026.11.745",
        run_date=work["run_date"],
        honest_verdict="complete_"
        + (
            "blocked_numerical_checkpoint_or_host"
            if verdict == "blocked"
            else "null_primary_integer_format"
            if verdict == "null"
            else verdict + "_spline_precision_feasibility"
        ),
        verdict_class=verdict,
        gate_check_summary=deepcopy(work["failures"]) + external_failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        substrate_roles=dict(
            reader="aggregation_from_upstream_artifacts",
            controls="deterministic_verifier",
            numerical_head="verifier_ensemble_against_cached_candidates",
        ),
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        target_model=None,
        current_llm_invocations=0,
        intended_count=work.get("initial_head_count", 5) * 3,
        completed_count=len(work["rows"]),
        failed_count=0,
        censored_count=work.get("initial_head_count", 5) * 3 - len(work["rows"]),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            initial_heads=work.get("initial_head_count", 5),
            formats=3,
            vectors=256,
            one_step_labels=2,
            sequences=4,
            updates_per_sequence=256,
            independent_questions=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_label_independent_numerical_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=validation,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates=dict(
            primary_32_24=work["primary_passed"],
            owned_validation=validation,
            external_validation=not external_bad,
            authenticated_inputs=not work["failures"],
        ),
        validation_receipts=deepcopy(receipts),
        preconditions_checked=work["preconditions"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        random_seed=numeric.SEED,
        source_artifact_hashes=deepcopy(work["source_refs"]),
        code_config_hashes=deepcopy(work["code_refs"]),
        raw_shard_hashes=[
            deepcopy(s["reference"]) for receipt in receipts for s in receipt.get("shards", [])
        ],
        cited_upstream_artifacts=deepcopy(work["source_refs"]),
        precision_mapping_ready_score=ready,
        branch_gate_values=dict(precision_mapping_ready_score=ready),
        physical_device_executed=False,
        publication_output=str(output.absolute()),
        invocation_id=Path(work["raw"]).name,
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        methodology_note="CPU fixed-point numerical controls; no causal stream, retention labels, FPGA synthesis, deployment or device performance measured.",
        primary_format=[32, 24],
        rounding="nearest_ties_to_even",
        arithmetic="explicit_saturation_checked_signed64",
        host_operations=[
            "scaling",
            "logit_offset",
            "sigmoid",
            "threshold_fallback",
            "serialization",
            "durable_storage",
        ],
        literature=dict(
            url="https://arxiv.org/html/2602.02056v4",
            revision="2026-06-19",
            adopted="local cubic support and explicit fixed-point rounding/saturation",
            limit="Published FPGA measurements do not certify this CPU emulator.",
        ),
        preserved_scope=[
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
            "Exp8667 historical disqualification",
            "all stamped quarantines",
        ],
    )
    if not result["rows"]:
        result["rows"] = [
            dict(
                head_index=i,
                bits=b,
                fractional=f,
                status="censored",
                censored=True,
                excluded=False,
                observed_updates=0,
                expected_updates=1536,
            )
            for i in range(5)
            for b, f in numeric.FORMATS
        ]
    result["field_principles"] = {
        k: "Bind this field to immutable task-specific numerical operands; no independent learning or device inference."
        for k in result
    }
    result["field_principles"].update(
        reproducibility_checksum="Bind the complete reduction to this immutable invocation.",
        precision_mapping_ready_score="Primary32/24 parity and owned validation must both pass.",
        gate_check_summary="Missing paths, missing fields, changed bytes and observed zero remain distinct.",
        cpu_operation_spans="Nonoverlapping monotonic CPU spans cannot establish a physical device latency.",
        rows="Heads, widths and repeated controls are dependent numerical units, never independent questions.",
    )
    result["reproducibility_checksum"] = canonical_hash(result)
    return result
