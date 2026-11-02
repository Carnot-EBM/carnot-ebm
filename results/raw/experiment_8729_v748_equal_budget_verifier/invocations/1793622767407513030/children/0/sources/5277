"""REQ-REPORT-8706: bind lifecycle conclusions to original numerical authority."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.primary_publication import validate_primary
from carnot.verify import spline_lifecycle_8706 as numerical

__all__ = ["numerical"]

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8706_v746_spline_lifecycle_stress"
TASK = "exp8706-spline-lifecycle-stress"
MILESTONE = "2026.11.746"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_spline_lifecycle_8706.py",
    "tests/python/test_spline_lifecycle_runner_8706.py",
]
OWNED = [
    "python/carnot/verify/spline_lifecycle_8706.py",
    "python/carnot/reporting/spline_lifecycle_study_8706.py",
    "python/carnot/reporting/spline_lifecycle_runner_8706.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
PIN = "43132b7a0dad4165413d7bcce82bdc9aa5602d17fc1a5144a960edb9d01c3c97"
UPSTREAM = "results/experiment_8692_v745_spline_precision_feasibility.json"
UPSTREAM_PIN = "sha256:bb960180068d3f6960df74f7f0dfeb440ed64ac4951071674e820a0d912eaa33"
PROTOCOL = "openspec/change-proposals/v744-methods-protocol.json"
PROTOCOL_PIN = "sha256:842e28d27098b7dadf41290c407e72cfa4019a2cbb09429c492355aa0f55cb67"
progress = numerical.progress


def reference(path: Path) -> Json:
    """Record absence explicitly; an absent artifact has no digest or numeric score."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def verify(ref: Json) -> None:
    """The byte check runs before JSON interpretation or score consumption."""
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("changed_bytes:" + ref["path"])


def read(ref: Json) -> Json:
    """Read only the exact bytes named by the immutable evidence reference."""
    verify(ref)
    return dict(json.loads(Path(ref["path"]).read_bytes()))


def gate(ref: Json, field: str, expected: Any) -> Json:
    """Preserve exact operands so a missing field cannot be confused with observed zero."""
    observed, state = None, "missing_file"
    passed = False
    try:
        value = read(ref)
        observed = value.get(field)
        state = (
            "missing_field"
            if field not in value
            else "observed_zero"
            if observed == 0
            else "observed"
        )
        passed = field in value and type(observed) is type(expected) and observed == expected
    except ValueError:
        state = "changed_bytes"
    except OSError:
        pass
    return dict(
        upstream="exp8692-spline-precision-feasibility",
        check="authenticated_field",
        **ref,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        operand_state=state,
        passed=passed,
    )


def authority(root: Path) -> Json:
    """The full-object digest also binds prompts, lineage and gate edges outside the table."""
    text = (root / DESIGN).read_text()
    shown, tasks = parse_design(text, milestone=MILESTONE)
    actual = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    comparison = compare_contract(text, actual, actual, milestone=MILESTONE, first_id=8695)
    envelope = json.loads(
        text.split("<!-- V746_TASK_CONTRACT_START -->", 1)[1]
        .split("```json", 1)[1]
        .split("```", 1)[0]
    )
    if not (
        comparison["passed"]
        and actual["tasks"] == tasks
        and canonical_hash(tasks) == "sha256:" + PIN
        and envelope == dict(milestone=MILESTONE, tasks=tasks, canonical_tasks_sha256=PIN)
        and f"Canonical task SHA256: `{PIN}`" in text
        and "<!-- V746_TASK_CONTRACT_END -->" in text
    ):
        raise ValueError("v746_full_object_authority")
    return dict(
        ready=True,
        canonical_tasks_sha256=PIN,
        visible_table=shown,
        task=tasks[11],
        full_objects_sha256=canonical_hash(tasks),
        comparison=comparison,
    )


def terminal(ref: Json, value: Json) -> list[Json]:
    """A passing score must have both producer identity and hash-bound terminal acceptance."""
    validate_primary(value, Path(value["publication_output"]))
    side = reference(Path(value["terminal_validation_sidecar_path"]))
    doc = read(side)
    bound_ref = reference(Path(doc["publication"]["sidecar_path"]))
    bound = read(bound_ref)
    if not (
        doc["primary_sha256"] == ref["sha256"] == bound["primary_sha256"]
        and doc["consumer"]["passed"] is True
        and bound["report"]["passed"] is True
    ):
        raise ValueError("producer_terminal_binding")
    return [side, bound_ref]


def authenticate() -> Json:
    """Pinned originals grant numerical authority without using flagged planning producers."""
    refs = [
        dict(path=str(ROOT / UPSTREAM), sha256=UPSTREAM_PIN),
        dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN),
    ]
    failures: list[Json] = []
    heads: list[Json] = []
    frozen: Json = {}
    provenance: Json = {}
    try:
        upstream, protocol = read(refs[0]), read(refs[1])
        for field, expected in [
            ("experiment_id", 8692),
            ("precision_mapping_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = gate(refs[0], field, expected)
            if not row["passed"]:
                failures.append(row)
        refs.extend(terminal(refs[0], upstream))
        refs.extend(
            [
                upstream["primitive_reference"],
                upstream["validation_reference"],
                protocol["head_checkpoint_reference"],
                protocol["original_producers"]["8431"],
            ]
        )
        work, validation = (
            read(upstream["primitive_reference"]),
            read(upstream["validation_reference"]),
        )
        checkpoint = read(protocol["head_checkpoint_reference"])
        producer = read(protocol["original_producers"]["8431"])
        refs.extend(terminal(protocol["original_producers"]["8431"], producer))
        for field, expected in [
            ("head_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = gate(protocol["original_producers"]["8431"], field, expected)
            row["upstream"] = "exp8431-criterion-energy-fit"
            if not row["passed"]:
                failures.append(row)
        for ref in upstream["code_config_hashes"]:
            verify(ref)
            refs.append(ref)
        heads = checkpoint["heads"]["spline9"]
        frozen_ref = next(
            r for r in work["source_refs"] if r["path"].endswith("/frozen_vectors.json")
        )
        initial_ref = next(
            r for r in work["source_refs"] if r["path"].endswith("/initial_heads.json")
        )
        refs.extend([frozen_ref, initial_ref])
        frozen = read(frozen_ref)
        if not (
            len(heads) == 5
            and heads == read(initial_ref)["heads"]
            and frozen == numerical.numeric.vectors(heads)
            and upstream["primary_format"] == [32, 24]
            and work["primary_passed"]
            and len(work["trajectory_rows"]) == 15360
            and len(work["precision_rows"]) == 7680
            and validation["receipts"] == upstream["validation_receipts"]
            and all(r["passed"] for r in validation["receipts"])
            and protocol["objective_updates"]["brier_derivative"]
            == numerical.numeric.original.RULE["data"]
            and protocol["objective_updates"]["learning_rate"] == 0.01
            and protocol["objective_updates"]["l2"] == 0.001
        ):
            raise ValueError("original_heads_geometry_rule_or_validation")
        for receipt in validation["receipts"]:
            for shard in receipt.get("shards", []):
                verify(shard["reference"])
                refs.append(shard["reference"])
        preserved = dict(
            path=str(
                ROOT / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
            ),
            sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
        )
        verify(preserved)
        refs.append(preserved)
        provenance = upstream["historical_model_provenance"]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        failures.append(
            dict(
                upstream="exp8692_original_numerical_authority",
                check="producer_primitives_terminal_checkpoint",
                path=refs[-1]["path"],
                sha256=refs[-1]["sha256"],
                field="original_numerical_authority",
                operator="==",
                expected="authenticated",
                observed=str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
                passed=False,
            )
        )
    return dict(
        heads=[] if failures else heads,
        frozen=frozen,
        refs=refs,
        failures=failures,
        historical_model_provenance=provenance,
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness describes complete evidence; numerical failures retain a qualified null."""
    external_bad = [
        r for r in receipts if not r["passed"] and r.get("coverage_role") == "consumers"
    ]
    owned_bad = any(
        not r["passed"]
        and r.get("coverage_role") != "consumers"
        and not (external_bad and r.get("name") == "original_disjoint_union")
        for r in receipts
    )
    failures = deepcopy(work["failures"])
    failures += [
        dict(
            upstream="unchanged_private_e2e",
            check=r["name"],
            path=r["stdout_path"],
            sha256=r["stdout_sha256"],
            field="actual_exit",
            operator="==",
            expected=0,
            observed=r["actual_exit"],
            operand_state="unchanged_external_validation_failure",
            passed=False,
        )
        for r in external_bad
    ]
    numerical_ok = (
        bool(work["summaries"])
        and all(s["passed"] for s in work["summaries"])
        and all(r["passed"] for r in work["restart_rows"])
        and all(numerical.envelope(s)["observed_inside_envelope"] for s in work["summaries"])
    )
    production = not work["private_control"]
    qualified = not owned_bad and not failures
    complete = len(work["summaries"]) == (25 if production else 2) and all(
        r["passed"] for r in work["worker_receipts"]
    )
    verdict = (
        "disqualified"
        if owned_bad
        else "blocked"
        if failures
        else "circular_positive"
        if numerical_ok
        else "null"
    )
    rows = [
        dict(
            head_index=s["head_index"],
            sequence=s["sequence"],
            status="completed",
            censored=False,
            excluded=False,
            passed=s["passed"],
            observed_updates=s["observed_updates"],
            first_failure=s["first_failure"],
        )
        for s in work["summaries"]
    ]
    if not rows:
        rows = [
            dict(
                head_index=i // 4,
                sequence=["alternating", "constant_one", "constant_zero", "seeded_bernoulli"][
                    i % 4
                ],
                status="censored",
                censored=True,
                excluded=False,
                passed=False,
                observed_updates=0,
            )
            for i in range(20)
        ] + [
            dict(
                head_index=i,
                sequence="boundary",
                status="censored",
                censored=True,
                excluded=False,
                passed=False,
                observed_updates=0,
            )
            for i in range(5)
        ]
    ordinary: list[Json] = []
    boundary: list[Json] = []
    for summary in work["summaries"]:
        target = boundary if summary["boundary"] else ordinary
        verify(summary["rows_reference"])
        target.extend(
            json.loads(line)
            for line in Path(summary["rows_reference"]["path"]).read_text().splitlines()
        )
    value = dict(
        experiment_id=8706,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        invocation_id=Path(work["raw"]).name,
        prior_invocation_dispositions=work["prior_invocation_dispositions"],
        honest_verdict="complete_" + verdict + "_spline_lifecycle_stress",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        target_model=None,
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=25 if production else 2,
        completed_count=len(work["summaries"]),
        failed_count=sum(not s["passed"] for s in work["summaries"]),
        censored_count=(25 if production else 2) - len(work["summaries"]),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            initial_heads=5,
            vectors=256,
            sequences=4,
            updates_per_sequence=4096,
            ordinary_pairs=81920,
            boundary_controls=320,
            restarted_integer_replay_updates=81920,
            cold_numerical_replay_pairs=82240,
            independent_questions=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_frozen_vectors_and_label_orders_no_independent_questions",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=not owned_bad,
        flagged_adversarial=owned_bad,
        acceptance_gates=dict(
            owned_validation=not owned_bad,
            authenticated_inputs=not failures,
            complete_lifecycle_rows=complete,
            numerical_criteria=numerical_ok,
        ),
        validation_receipts=deepcopy(receipts),
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
        branch_gate_values=dict(
            lifecycle_rows_ready_score=int(production and qualified and complete),
            numerical_lifecycle_ready_score=int(
                production and qualified and complete and numerical_ok
            ),
        ),
        lifecycle_rows_ready_score=int(production and qualified and complete),
        numerical_lifecycle_ready_score=int(production and qualified and complete and numerical_ok),
        update_rows=ordinary,
        boundary_rows=boundary,
        restart_rows=work["restart_rows"],
        accumulator_bound_rows=[
            dict(head_index=s["head_index"], sequence=s["sequence"], **numerical.envelope(s))
            for s in work["summaries"]
        ],
        decay_error_rows=[
            dict(
                head_index=s["head_index"],
                sequence=s["sequence"],
                maximum_decay_error=s["maximum_decay_error"],
                maximum_coefficient_error=s["maximum_coefficient_error"],
                renormalizations=s["counts"]["renormalization"],
            )
            for s in work["summaries"]
        ],
        complete_operation_cost_rows=[
            dict(head_index=s["head_index"], sequence=s["sequence"], device="cpu", operation=k, **v)
            for s in work["summaries"]
            for k, v in s["costs"].items()
        ]
        + work["restart_cost_rows"]
        + [
            dict(
                operation="complete_worker_wall",
                device="cpu",
                duration_ns=r["ended_monotonic_ns"] - r["started_monotonic_ns"],
                calls=1,
                aggregation_role="envelope_not_additive_with_inner_spans",
                command_receipt=r["stdout_path"],
            )
            for r in work["worker_receipts"]
        ],
        physical_device_executed=False,
        authority=work["authority"],
        primitive_reference=reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output.absolute()),
        qualification_scope="production" if production else "private_control_only",
        worker_receipts=work["worker_receipts"],
        first_failing_state=next(
            (s["first_failure"] for s in work["summaries"] if s["first_failure"]), None
        ),
        primary_format=[32, 24],
        rounding="nearest_ties_to_even",
        arithmetic="explicit_saturation_checked_signed64",
        methodology_note="CPU emulation with original Brier equation, scaling and temperature; marked boundary states are constructed, not learned. No generalized learning, FPGA execution or acceleration measured.",
        literature=dict(
            url="https://arxiv.org/html/2602.02056v4",
            limit="Format/locality motivation only; paper timings do not qualify this CPU emulator.",
        ),
        preserved_scope=[
            "Exp8681 and Exp8691 historical disqualifications",
            "all historical quarantines",
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
        ],
        interval_assumptions=dict(
            format=[32, 24],
            decay_trigger="rounded_next_scale < 65536",
            probability_error_max=0.001,
            continuity_error_max=0.5 / (1 << 24),
            supported_domain_only=True,
            global_overflow_proof=False,
        ),
    )
    value["field_principles"] = {
        k: "Bind exact invocation evidence; constructed CPU arithmetic grants neither useful learning nor physical acceleration."
        for k in value
    }
    value["field_principles"].update(
        lifecycle_rows_ready_score="Complete authenticated rows and owned validation, independent of numerical success.",
        numerical_lifecycle_ready_score="All numerical criteria plus complete qualified lifecycle evidence.",
        update_rows="One status per intended update; replay, seeds and controls add no independent questions.",
        complete_operation_cost_rows="Retain dense renormalization, host copies, nonlinear CPU work, serialization, durable writes and restart work.",
        reproducibility_checksum="Stable digest of the immutable report reduction.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
