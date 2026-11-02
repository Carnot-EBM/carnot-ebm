"""REQ-REPORT-8719: bind a language-port parity claim to qualified original operands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import spline_lifecycle_study_8706 as previous
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import compiled_sparse_8719 as numerical

Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8719_v747_compiled_sparse_update", "exp8719-compiled-sparse-update"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/verify/compiled_sparse_8719.py",
    "python/carnot/reporting/compiled_sparse_study_8719.py",
    "python/carnot/reporting/compiled_sparse_runner_8719.py",
    CLI,
]
TESTS = [
    "tests/python/test_compiled_sparse_8719.py",
    "tests/python/test_compiled_sparse_runner_8719.py",
]
RUST = [
    "crates/carnot-python/src/spline_fixed_8719.rs",
    "crates/carnot-python/src/spline_fixed_8719_tests.rs",
    "crates/carnot-python/src/lib.rs",
    "crates/carnot-python/Cargo.toml",
    "Cargo.lock",
]
PRODUCER = "results/experiment_8706_v746_spline_lifecycle_stress.json"
PIN = "sha256:306fde7e7afccef1f9a079ead7a1bb26226008e0126bfccbc837a5afe172e0f5"
reference, original_verify, read, progress = (
    previous.reference,
    previous.verify,
    previous.read,
    numerical.progress,
)
MODEL_SPECS: list[Json] = []
__all__ = ["numerical"]


def verify(ref: Json) -> None:
    """Bind observed absence for blocked reports without confusing it with required bytes."""
    if ref["sha256"] is None:
        if Path(ref["path"]).is_file():
            raise ValueError("changed_absence:" + ref["path"])
    else:
        original_verify(ref)


def gate(ref: Json, field: str, expected: Any) -> Json:
    """Keep file absence, changed bytes, missing fields and observed zero distinct."""
    return dict(previous.gate(ref, field, expected), upstream="exp8706-spline-lifecycle-stress")


def inputs() -> Json:
    """Authenticate original authority without requiring any unrelated V747 producer."""
    result: Json = dict(previous.authenticate())
    result["contract"] = contract.authority(ROOT)
    result["failures"] += result["contract"]["gate_check_summary"]
    result["refs"] += result["contract"]["references"]
    result.update(jobs=[], historical_refs=[], original_summaries=[])
    ref = dict(path=str(ROOT / PRODUCER), sha256=PIN)
    result["refs"].append(ref)
    try:
        value = read(ref)
        for field, expected in [
            ("experiment_id", 8706),
            ("numerical_lifecycle_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = gate(ref, field, expected)
            if not row["passed"]:
                result["failures"].append(row)
        result["refs"] += previous.terminal(ref, value)
        for operand in value["code_config_hashes"] + [
            value["primitive_reference"],
            value["validation_reference"],
        ]:
            verify(operand)
            result["refs"].append(operand)
        work, validation = read(value["primitive_reference"]), read(value["validation_reference"])
        if (
            not all(r["passed"] for r in validation["receipts"])
            or validation["receipts"] != value["validation_receipts"]
            or work["private_control"]
            or len(work["summaries"]) != 25
            or work["authority"]["canonical_tasks_sha256"] != previous.PIN
        ):
            raise ValueError("qualified_lifecycle_primitives")
        for index, summary in enumerate(work["summaries"]):
            job_ref, rows_ref = summary["job_reference"], summary["rows_reference"]
            job = read(job_ref)
            verify(rows_ref)
            hi = index // 4 if index < 20 else index - 20
            sequence = (
                sorted(result["frozen"]["sequences"])[index % 4] if index < 20 else "boundary"
            )
            expected_job = dict(
                head=result["heads"][hi],
                head_index=hi,
                sequence=sequence,
                labels=result["frozen"]["sequences"][sequence if index < 20 else "alternating"],
                vectors=result["frozen"]["vectors"],
                boundary=index >= 20,
                updates=4096 if index < 20 else 64,
            )
            if index >= 20:
                expected_job["boundary_offset"] = hi - 2
            if job != expected_job or not summary["passed"]:
                raise ValueError("original_job_identity")
            result["jobs"].append(job)
            result["historical_refs"].append(rows_ref)
            result["original_summaries"].append(summary)
            result["refs"] += [job_ref, rows_ref]
        result["refs"] += [dict(path=str(ROOT / previous.PROTOCOL), sha256=previous.PROTOCOL_PIN)]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["failures"].append(
            dict(
                upstream="exp8706-spline-lifecycle-stress",
                **ref,
                check="producer_primitives_terminal",
                field="qualified_original_operands",
                operator="==",
                expected="authenticated",
                observed=str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
                passed=False,
            )
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness requires all exact rows and validation; repeats provide no new science."""
    bad = any(not r["passed"] for r in receipts)
    failures = work["failures"]
    rows = work["summaries"]
    numerical_ok = (
        bool(rows)
        and all(r["passed"] for r in rows)
        and all(r["passed"] for r in work["restart_rows"] + work["rejection_rows"])
    )
    complete = len(rows) == (2 if work["private_control"] else 25)
    ready = int(
        not work["private_control"] and not failures and not bad and numerical_ok and complete
    )
    verdict = (
        "disqualified"
        if bad or (rows and not numerical_ok)
        else "blocked"
        if failures
        else "circular_positive"
    )
    parity: list[Json] = []
    boundary: list[Json] = []
    for summary in rows:
        verify(summary["rows_reference"])
        target = boundary if summary["boundary"] else parity
        target.extend(
            dict(json.loads(line), head_index=summary["head_index"], sequence=summary["sequence"])
            for line in Path(summary["rows_reference"]["path"]).read_text().splitlines()
        )
    count = 2 if work["private_control"] else 25
    unit_rows = [
        dict(
            unit_id=str(i),
            head_index=i // 4 if i < 20 else i - 20,
            status="completed" if i < len(rows) else "censored",
            censored=i >= len(rows),
            excluded=False,
            passed=rows[i]["passed"] if i < len(rows) else False,
            observed_updates=rows[i]["observed_updates"] if i < len(rows) else 0,
        )
        for i in range(count)
    ]
    value = dict(
        experiment_id=8719,
        task_id=TASK,
        milestone="2026.11.747",
        run_date=work["run_date"],
        invocation_id=Path(work["raw"]).name,
        honest_verdict="complete_" + verdict + "_compiled_sparse_update",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        target_model=None,
        historical_model_provenance=work["historical_model_provenance"],
        rows=unit_rows,
        intended_count=count,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=count - len(rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            initial_heads=5,
            sequences=4,
            vectors=256,
            updates_per_sequence=4096,
            ordinary_pairs=81920,
            boundary_controls=320,
            restart_replay_updates=81920,
            cold_replay_pairs=82240,
            independent_questions=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="same_original_exposed_lifecycle_inputs_language_port_parity_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        acceptance_gates=dict(
            owned_validation=not bad,
            exact_native_parity=numerical_ok,
            authenticated_originals=not failures,
            complete_rows=complete,
            loaded_extension=work["extension"]["actual_loaded"],
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=numerical.numeric.SEED,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"]
        + [s["reference"] for r in receipts for s in r.get("shards", [])],
        cited_upstream_artifacts=work["upstream_refs"],
        branch_gate_values=dict(native_update_ready_score=ready),
        native_update_ready_score=ready,
        native_parity_rows=parity,
        boundary_rows=boundary,
        rejection_rows=work["rejection_rows"],
        restart_rows=work["restart_rows"],
        loaded_extension_reference=work["extension"],
        supported_domain=previous.numerical.envelope(
            dict(
                accumulator_abs_max=max((s["checkpoint"]["maximum"] for s in rows), default=0),
                maximum_basis_integer=(1 << 24) + 16,
            )
        ),
        operation_partition_rows=[
            dict(head_index=s["head_index"], sequence=s["sequence"], **r)
            for s in rows
            for r in s["operation_partition_rows"]
        ],
        host_call_rows=[
            dict(head_index=s["head_index"], sequence=s["sequence"], **s["host_calls"])
            for s in rows
        ],
        build_host_reference=work["host"],
        task_authority=work["contract"],
        publication_output=str(output),
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        speedup_claim=None,
        physical_fpga_execution=False,
        private_control=work["private_control"],
    )
    value["field_principles"] = {
        k: "Bind this field to sealed exact parity evidence; CPU port readiness grants no learning, hardware or throughput claim."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(
        dict(code=work["code_refs"], sources=work["source_refs"], seed=value["random_seed"])
    )
    return value
