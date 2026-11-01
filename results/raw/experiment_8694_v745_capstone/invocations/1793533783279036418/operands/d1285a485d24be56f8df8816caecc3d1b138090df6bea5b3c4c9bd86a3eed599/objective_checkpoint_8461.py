"""REQ-REPORT-8461: engineering qualification stays separate from measured learning benefit."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import block_checkpoint_qualification_8433 as shipped
from carnot.reporting import objective_checkpoint_evidence_8461 as evidence
from carnot.reporting import v727_historical_inputs as historical
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.human_label_custody_8389 import gate
from carnot.verify import objective_checkpoint_8461 as adapter
from carnot.verify import objective_derivatives_8461 as derivatives

Json = dict[str, Any]
ROOT, NAME, TASK, CLI = evidence.ROOT, evidence.NAME, evidence.TASK, evidence.CLI
TEST = "tests/python/test_objective_checkpoint_8461.py"
EXTRA_TEST = "tests/python/test_objective_reporting_8461.py"
OWNED = [
    "python/carnot/verify/objective_checkpoint_8461.py",
    "python/carnot/verify/objective_derivatives_8461.py",
    "python/carnot/reporting/objective_checkpoint_evidence_8461.py",
    "python/carnot/reporting/objective_checkpoint_8461.py",
    "python/carnot/reporting/objective_checkpoint_runner_8461.py",
    CLI,
]
REQUIRED_CHECKS = shipped.REQUIRED_CHECKS | {"historical_store_regressions", "coverage_archive"}
MODEL_SPECS: list[Json] = []
read, reference, checksum, progress = (
    evidence.read,
    evidence.reference,
    evidence.methods.checksum,
    evidence.progress,
)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Retain executable bytes and a constructed fixture receipt for a later measurement task."""
    work = evidence.measure(root, raw, private)
    shutil.copyfile(ROOT / OWNED[0], raw / "update_adapter.py")
    atomic_json(
        raw / "fixture_receipt.json",
        dict(
            update_rule=adapter.RULE,
            update_rule_hash=adapter.RULE_HASH,
            update_adapter_reference=reference(raw / "update_adapter.py"),
            primitive_reference=work["primitive_reference"],
            released_evaluation_labels=0,
            objective_stream_ready_score=0,
            physical_power_loss_recertified=False,
        ),
    )
    work["code_refs"] = [reference(ROOT / p) for p in [*OWNED, TEST, EXTRA_TEST]]
    work["fixture_reference"] = reference(raw / "fixture_receipt.json")
    atomic_json(raw / "measurement.json", work)
    return work


def manifest(private: Path) -> list[Json]:
    """Reuse frozen serial coverage and unchanged private authority, hard-exit and replay tests."""
    with (
        patch.object(shipped, "OWNED", OWNED),
        patch.object(shipped, "TEST", TEST),
        patch.object(shipped, "CLI", CLI),
    ):
        plan = shipped.manifest(private)
    owned = next(r for r in plan if r["name"] == "owned_tests")
    owned["argv"].extend(
        [
            EXTRA_TEST,
        ]
    )
    owned["deadline"] = 900
    for row in plan:
        if row["name"] in ("ruff_check", "ruff_format", "spec_coverage"):
            row["argv"].append(EXTRA_TEST)
    history_program = """import sys,pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from carnot.reporting import v727_historical_inputs as h
from carnot.reporting.v721_capstone_evidence import frozen_inputs
with TemporaryDirectory(prefix='8461-history-tests-',dir='/var/tmp') as directory:
    private=Path(directory)
    refs=h.custody(h.ROOT,private/'authority')
    with frozen_inputs(refs,private/'consumer'):
        status=pytest.main(['-n','0','-o','addopts=','--no-cov','-q',
            'tests/python/test_block_feedback_state_8433.py',
            'tests/python/test_capacity_online_learning_8434.py',
            '--basetemp='+str(private/'pytest')])
    sys.exit(status)
"""
    plan.insert(
        1,
        dict(
            name="historical_store_regressions",
            argv=[sys.executable, "-u", "-c", history_program],
            expected=0,
            deadline=600,
            scope="owned",
        ),
    )
    archive_program = "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); d=p/'uncombined_shards'; d.mkdir(exist_ok=True); files=list(p.glob('.coverage.*')); [shutil.copyfile(f,d/f.name) for f in files]; print('genuine_shards_archived',len(files),flush=True); sys.exit(0 if files else 1)"
    index = next(i for i, row in enumerate(plan) if row["name"] == "coverage_combine")
    plan.insert(
        index,
        dict(
            name="coverage_archive",
            argv=[sys.executable, "-u", "-c", archive_program, str(private)],
            expected=0,
            deadline=30,
            scope="owned",
        ),
    )
    return list(plan)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Require every prerequisite before readiness; all fixtures have zero generalization support."""
    primitive = read(work["primitive_reference"])
    derivatives_ok = len(primitive["derivative_rows"]) == 90 and all(
        r["passed"] for r in primitive["derivative_rows"]
    )
    controls_ok = bool(primitive["transition_rows"]) and all(
        r["passed"] for r in primitive["transition_rows"]
    )
    derivative_controls_ok = (
        derivatives_ok
        and all(
            r["negative_controls"]["reverse_gradient_rejected"]
            for r in primitive["derivative_rows"]
            if not r["saturated"]
        )
        and all(
            r["negative_controls"]["omit_temperature_rejected"]
            for r in primitive["derivative_rows"]
            if r["temperature"] != 1 and abs(r["logit"]) <= 4
        )
    )
    crashes_ok = (
        len(primitive["restart_rows"]) == 36
        and len(primitive["process_exit_rows"]) == 32
        and all(r["passed"] for r in primitive["restart_rows"] + primitive["process_exit_rows"])
    )
    qualified = (
        bool(receipts)
        and REQUIRED_CHECKS <= {r["name"] for r in receipts}
        and all(r["passed"] for r in receipts)
        and not work.get("adversarial_findings")
    )
    ready = (
        qualified
        and derivatives_ok
        and derivative_controls_ok
        and controls_ok
        and crashes_ok
        and not work["failures"]
    )
    verdict = (
        "disqualified"
        if not qualified
        or any(
            not r["passed"]
            for r in primitive["derivative_rows"]
            + primitive["transition_rows"]
            + primitive["restart_rows"]
        )
        or any(f.get("check") == "measurement_cap_s" for f in work["failures"])
        else "circular_positive"
        if ready
        else "blocked"
    )
    rows = []
    for r in primitive["derivative_rows"] + primitive["restart_rows"]:
        rows.append(
            dict(
                r,
                status="completed" if r["passed"] else "failed",
                absolute_metric=int(r["passed"]),
                raw_numerator=int(r["passed"]),
                raw_denominator=1,
                censored=False,
                excluded=False,
                missing_reason=None,
            )
        )
    if not primitive["derivative_rows"]:
        rows.extend(
            dict(
                unit_id=f"{support}:T{t}:z{z}:y{y}",
                family="constructed_derivatives",
                arm="brier",
                status="censored",
                absolute_metric=None,
                raw_numerator=None,
                raw_denominator=1,
                passed=None,
                censored=True,
                excluded=False,
                missing_reason="required_cpu_operand_unavailable",
            )
            for support in ("lower", "interior", "upper")
            for t in (0.5, 1.0, 2.0)
            for z in (-20.0, -4.0, 0.0, 4.0, 20.0)
            for y in (0, 1)
        )
    present_ids = {r["unit_id"] for r in rows}
    if len(primitive["restart_rows"]) < 36:
        rows.extend(
            dict(
                unit_id=f"c{c}-{p}:{case}",
                family="constructed_recovery",
                arm=p,
                capacity=c,
                status="censored",
                absolute_metric=None,
                raw_numerator=None,
                raw_denominator=1,
                passed=None,
                censored=True,
                excluded=False,
                missing_reason="required_historical_head_unavailable",
            )
            for c in (8, 16)
            for p in ("fifo", "uniform")
            for case in [
                "none",
                *[f"{t}:0:{b}" for t in ("seal", "release") for b in adapter.legacy.BARRIERS],
            ]
            if f"c{c}-{p}:{case}" not in present_ids
        )
    gates = work["failures"] + [
        gate(str(raw), "required_owned_check", True, r["passed"], upstream=TASK)
        for r in receipts
        if not r["passed"]
    ]
    if not REQUIRED_CHECKS <= {r["name"] for r in receipts}:
        gates.append(
            gate(
                str(raw / "execution_manifest.json"),
                "required_validation_manifest",
                sorted(REQUIRED_CHECKS),
                sorted(r["name"] for r in receipts),
                upstream=TASK,
            )
        )
    value = dict(
        experiment_id=8461,
        task_id=TASK,
        milestone="2026.10.728",
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v728_objective_checkpoint_qualification",
        verdict_class=verdict,
        gate_check_summary=gates,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if primitive["derivative_rows"]
        else "aggregation_from_upstream_artifacts",
        reader_inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=126,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            constructed_derivative_cells=90, dependent_recovery_cases=36, independent_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed gradients and state fixtures; no released evaluation labels",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            derivatives=derivatives_ok,
            negative_derivative_controls=derivative_controls_ok,
            sealed_transition_controls=controls_ok,
            durable_recovery=crashes_ok,
            finite_state=crashes_ok,
            owned_validation=qualified,
            current_authority=work["authority"]["activated"],
            required_inputs=not work["failures"],
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(name=r["name"], duration_s=r["duration_s"])
            for r in work["process_receipts"] + receipts
        ],
        random_seed=17,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[work["primitive_reference"], work["fixture_reference"]]
        + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields=[
                    "head",
                    "historical_model_provenance",
                    "original_producer_qualification",
                ],
            )
            for r in work["source_refs"]
        ],
        objective_kernel_ready_score=int(ready),
        objective_stream_ready_score=0,
        stream_rows_reference=None,
        retention_rows=[],
        coefficient_change_count=sum(
            r["coefficient_change_count"] for r in primitive["restart_rows"]
        ),
        admission_identity_checks=[
            dict(unit_id=r["unit_id"], passed=r["passed"]) for r in primitive["restart_rows"]
        ],
        gradient_checks=primitive["derivative_rows"],
        derivative_rows=primitive["derivative_rows"],
        restart_checks=primitive["restart_rows"],
        process_exit_rows=primitive["process_exit_rows"],
        transition_controls=primitive["transition_rows"],
        update_rule_hash=adapter.RULE_HASH,
        update_adapter_reference=reference(raw / "update_adapter.py"),
        fixture_receipt_reference=work["fixture_reference"],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        physical_power_loss_recertified=False,
        historical_limits=dict(
            V717_H1=-0.00390625,
            V717_H2=0,
            V726_admission_audit="null",
            kernel_service_percent=0.093,
            persistence_service_percent=96.44,
        ),
        methodology_note="Independent dense Brier derivatives and centered differences precede real two-commit child exits; no online benefit measured.",
    )
    value["field_principles"] = {
        k: "Invocation-bound constructed qualification; no learning-benefit or generalization claim."
        for k in value
    }
    value["field_principles"].update(
        objective_kernel_ready_score="All derivative, negative control, recovery, authority and owned checks must pass; no benefit is measured.",
        objective_stream_ready_score="Zero because the separate objective-learning stream has not run.",
        derivative_rows="Independent individual bases and centered differences retain both absolute and relative error limits.",
        transition_controls="Constructed future, early and wrong-block labels must fail before updates.",
        coefficient_change_count="Changes in constructed fixture coefficients; dependent repeats do not count as independent questions.",
        update_rule_hash="Version, formula, temperature, local support, bias and regularizer conventions bind checkpoint resumes.",
        raw_shard_hashes="Exact primitive operands and genuine invocation Coverage.py shards remain available after private scratch cleanup.",
        reproducibility_checksum="A canonical aggregate hash is checked against reconstructed primitives, never trusted alone.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild derivatives and states from fixed operands; self-consistent outer hashes are insufficient."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        primitive = read(work["primitive_reference"])
        for ref in (
            work["source_refs"]
            + work["code_refs"]
            + work.get("coverage_shards", [])
            + [work["fixture_reference"]]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if work["code_refs"] != [reference(ROOT / p) for p in [*OWNED, TEST, EXTRA_TEST]]:
            return False
        with TemporaryDirectory(prefix="8461-cold-", dir="/var/tmp") as scratch:
            try:
                authority = evidence.methods.authority(Path(work["root"]), Path(scratch))
                activated = authority["activated"]
            except (OSError, ValueError, KeyError):
                activated = False
        if activated != work["authority"]["activated"]:
            return False
        if activated and authority["tasks"] != work["authority"]["tasks"]:
            return False
        head = primitive["head"]
        if head is not None:
            primary = next(
                r
                for r in work["source_refs"]
                if r["path"].endswith("experiment_8434_v726_capacity_online_learning.json")
            )
            if primary["sha256"] != "sha256:" + historical.PINS[8434]:
                return False
            original = read(read(primary)["primitive_reference"])
            head_ref = original["cells"][0]["configuration"]
            if head != read(head_ref)["head"]:
                return False
        if primitive["derivative_rows"] and primitive["derivative_rows"] != derivatives.checks(
            head
        ):
            return False
        if primitive["transition_rows"] and primitive[
            "transition_rows"
        ] != derivatives.transition_controls(head):
            return False
        for row in primitive["restart_rows"]:
            cfg, state = read(row["configuration"]), read(row["final_state"])
            if cfg != adapter.config(
                head, row["capacity"], row["arm"], shuffled=True
            ) or not adapter.replay_state(state, cfg):
                return False
            if row["logical_state_hash"] != canonical_hash(state) or row[
                "prediction_hash"
            ] != canonical_hash(state["issued"]):
                return False
        for row in primitive["process_exit_rows"]:
            state = read(row["checkpoint"])
            if (
                row["actual_exit"] != 73
                or state["version"] != row["expected_version"]
                or not adapter.replay_state(state, read(row["configuration"]))
            ):
                return False
        for receipt in work["process_receipts"] + value["validation_receipts"]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if any(
                receipt[k] != saved[k]
                for k in (
                    "name",
                    "argv",
                    "exit_code",
                    "expected_exit",
                    "timed_out",
                    "duration_s",
                    "stdout_sha256",
                    "stderr_sha256",
                )
            ):
                return False
            expected_pass = (
                receipt["exit_code"] == receipt["expected_exit"] and not receipt["timed_out"]
            )
            if receipt["name"] == "adversarial":
                report = json.loads(Path(receipt["stdout_path"]).read_bytes())
                flags = [flag for row in report["reports"] for flag in row.get("flags", [])]
                expected_pass = expected_pass and not flags
            if receipt["passed"] != expected_pass:
                return False
            if (
                sha256_file(Path(receipt["stdout_path"])) != receipt["stdout_sha256"]
                or sha256_file(Path(receipt["stderr_path"])) != receipt["stderr_sha256"]
            ):
                return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
