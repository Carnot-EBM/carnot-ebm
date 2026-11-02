"""REQ-REPORT-8701: distinguish lexical capacity from faithful symbolic records.

The supplied worlds are exposed oracle controls. Missing p2 probabilities cannot
be inferred from truth labels or silently replaced with a neutral constant.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

from carnot.reporting import relational_evidence_8698 as upstream
from carnot.reporting import v746_contract_authority as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.verify import streaming_evidence as stream
from carnot.verify.v726_criterion_energy_fit import predict

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8701_v746_relational_feature_audit", "exp8701-relational-feature-audit"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/relational_feature{s}_8701.py" for s in ("", "_runner")] + [CLI]
TESTS = ["tests/python/test_relational_feature_8701.py"]
CORPUS = "results/experiment_8698_v746_relational_evidence_controls.json"
CORPUS_PIN = "sha256:2d32e3129192287b372ea342b3e56b9139d18ea5a7eae23347151cf822df4d8a"
MODEL_SPECS: list[Json] = []
reference, load = upstream.reference, upstream.load


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the supervisor can distinguish work from a stall."""
    print(f"[exp8701] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path) -> Json:
    """Reauthenticate original operands independently of historical global readiness."""
    branch = upstream.inputs(root)
    refs, failures = branch["references"], branch["failures"]
    ref = dict(path=str(root / CORPUS), sha256=CORPUS_PIN)
    custody = upstream.custody(ref, "corpus_ready_score", 8698)
    refs += custody["references"]
    failures += custody["gate_check_summary"]
    head: Json = {}
    head_ref = None
    try:
        protocol = load(dict(path=str(root / upstream.PROTOCOL), sha256=upstream.PROTOCOL_PIN))
        head_ref = protocol["head_checkpoint_reference"]
        refs.append(head_ref)
        head = load(head_ref)
        terminal = upstream.custody(protocol["current_producers"]["8431"], "head_ready_score", 8431)
        refs += terminal["references"]
        failures += terminal["gate_check_summary"]
        registered = protocol["relational_evidence_protocol"]
        if not (
            registered["chunk_sizes"] == [1, 32, 128]
            and registered["fitting"] is False
            and registered["feature_changes"] is False
            and len(protocol["features"]["names"]) == 9
        ):
            failures.append(
                authority.gate(root / upstream.PROTOCOL, "registered_method_values", True, False)
            )
        corpus = load(ref)
        if corpus["rows"] != upstream.qualify(branch["worlds"])["rows"]:
            failures.append(
                authority.gate(root / CORPUS, "original_480_rows", True, False, "exp8698")
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                upstream.authenticate(ref),
                check="original_operands",
                passed=False,
                diagnostic=str(error),
            )
        )
    failures += [g for ref in refs if not (g := upstream.authenticate(ref))["passed"]]
    return dict(
        branch,
        passed=not failures,
        references=refs,
        failures=failures,
        head=head,
        head_reference=head_ref,
    )


def sql_decision(path: Path, facts: list[list[int]], query: Any) -> int:
    """Execute only fixed SQL with bound integers; absent premises always escalate."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (a INTEGER, b INTEGER, value INTEGER, PRIMARY KEY(a,b))"
        )
        conn.execute("DELETE FROM facts")
        conn.executemany("INSERT INTO facts VALUES (?, ?, ?)", facts)
        if query is None:
            return -1
        predicates = query if isinstance(query[0], list) else [query]
        actual = [
            conn.execute("SELECT value FROM facts WHERE a=? AND b=?", (a, b)).fetchone()
            for a, b, _ in predicates
        ]
        if any(row is None for row in actual):
            return -1
        return int(all(row[0] == p[2] for row, p in zip(actual, predicates, strict=True)))


def diagnose(
    original: list[list[int]],
    records: list[list[int]],
    query: Any,
    actual_query: Any,
    decision: int,
    oracle: int,
) -> list[str]:
    """Separate translation, query and decision faults rather than assigning one accuracy."""
    categories = []
    if original != records:
        categories.append(
            "extraction_omission" if all(r in original for r in records) else "wrong_relation"
        )
    if query != actual_query:
        categories.append("wrong_query")
    if decision != oracle:
        categories.append("wrong_decision")
    return categories


def numerical(values: list[float], q: float | None, checkpoint: Json) -> Json:
    """A checkpoint authenticates coefficients, not a missing candidate probability."""
    if q is None:
        return dict(
            probability=None,
            decision=None,
            operand_state="missing_field",
            field="p2_hallucination_probability",
            numerical_calls=0,
        )
    probabilities = [
        float(predict(head, np.asarray([[q, *values]]))[0])
        for head in checkpoint["heads"]["spline9"]
    ]
    probability = sum(probabilities) / len(probabilities)
    return dict(
        probability=probability,
        decision=0 if probability >= 0.8 else 1 if probability <= 0.2 else -1,
        operand_state="observed_value",
        field="p2_hallucination_probability",
        numerical_calls=len(probabilities),
    )


def measure(worlds: list[Json], checkpoint: Json, raw: Path) -> Json:
    """Retain each view and arm; chunk repeats never multiply independent worlds."""
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[Json] = []
    symbolic, extraction, query_errors, alignment, collisions = [], [], [], [], []
    for index, world in enumerate(worlds):
        queries = upstream.ground(upstream.question(world))
        query_seal = canonical_hash(queries)
        local = []
        facts = world["facts"]
        for view in world["views"]:
            uid = world["world_id"] + ":" + view["view"]
            query = upstream.parse_claim(view["claim"], world)
            arms = []
            for chunk in [1, 32, 128]:
                path = raw / f"{uid}-{chunk}.jsonl"
                public = dict(
                    family_id=uid,
                    source_bytes=view["source"].encode().hex(),
                    answer_bytes=view["claim"].encode().hex(),
                )
                result = stream.extract(public, path, chunk_size=chunk)
                arms.append(
                    dict(
                        chunk_size=chunk,
                        values=result["features"]["values"],
                        latency_s=result["latency_s"],
                        complete=result["complete"],
                        windows=result["total_window_count"],
                        window_reference=reference(path),
                    )
                )
            head = numerical(
                arms[0]["values"], view.get("p2_hallucination_probability"), checkpoint
            )
            for arm in arms:
                arm.update(decision=head["decision"], decision_operand_state=head["operand_state"])
            ss, cs = view["source_span"], view["claim_span"]
            document = view["document"].encode()
            spans_ok = (
                document[slice(*ss)] == view["source"].encode()
                and document[slice(*cs)] == view["claim"].encode()
            )
            expected = upstream.evaluate(facts, query)
            predicates = query if query is None or isinstance(query[0], list) else [query]
            valid_query = query is None or all(p in queries for p in predicates)
            record = dict(
                unit_id=uid,
                world_id=world["world_id"],
                view=view["view"],
                truth=view["truth"],
                query=query,
                query_seal=query_seal,
                source_span=ss,
                claim_span=cs,
                spans_aligned=spans_ok,
                lexical_arms=arms,
                head=head,
                absolute_metric=int(expected == view["truth"] and spans_ok and valid_query),
                status="completed",
                censored=False,
                excluded=False,
            )
            rows.append(record)
            record["status"] = "completed" if record["absolute_metric"] else "failed"
            local.append(record)
            deleted = [r for r in facts if query is not None and r[:2] != predicates[0][:2]]
            reversed_facts = [[b, a, v] for a, b, v in facts]
            for arm, records in [
                ("complete_records", facts),
                ("deleted_required_records", deleted),
                ("reversed_relations", reversed_facts),
            ]:
                started = time.monotonic_ns()
                path = raw / f"{uid}-{arm}.sqlite"
                decision = sql_decision(path, records, query)
                oracle = upstream.evaluate(records, query)
                categories = diagnose(facts, records, query, query, decision, oracle)
                symbolic.append(
                    dict(
                        unit_id=uid,
                        arm=arm,
                        records=records,
                        query=query,
                        decision=decision,
                        expected=oracle,
                        original_truth=view["truth"],
                        passed=decision == oracle,
                        escalated=decision == -1,
                        latency_s=(time.monotonic_ns() - started) / 1e9,
                        categories=categories,
                        database_reference=reference(path),
                    )
                )
                if categories:
                    extraction.append(dict(unit_id=uid, arm=arm, categories=categories))
            wrong = None if query is not None else [0, 1, 1]
            for arm, candidate_query, candidate_decision, oracle in [
                (
                    "wrong_query_control",
                    wrong,
                    upstream.evaluate(facts, wrong),
                    upstream.evaluate(facts, wrong),
                ),
                ("wrong_decision_control", query, 0 if expected == -1 else 1 - expected, expected),
            ]:
                query_errors.append(
                    dict(
                        unit_id=uid,
                        arm=arm,
                        query=candidate_query,
                        decision=candidate_decision,
                        expected=oracle,
                        categories=diagnose(
                            facts, facts, query, candidate_query, candidate_decision, oracle
                        ),
                    )
                )
        indexed = {v["view"]: v for v in local}
        supported, contradicted, equivalent = (
            indexed[v] for v in ["supported", "contradicted", "equivalent_wording"]
        )
        a, b, c = (v["lexical_arms"][0]["values"] for v in [supported, contradicted, equivalent])
        collisions.append(
            dict(
                world_id=world["world_id"],
                left=supported["unit_id"],
                right=contradicted["unit_id"],
                left_values=a,
                right_values=b,
                exact_equal=a == b,
                opposite_truth=supported["truth"] != contradicted["truth"],
                scope="representational_counterexample_only",
            )
        )
        alignment.append(
            dict(
                world_id=world["world_id"],
                truth_equivalent=supported["truth"] == equivalent["truth"] == 1,
                query_equivalent=sorted(
                    supported["query"]
                    if isinstance(supported["query"][0], list)
                    else [supported["query"]]
                )
                == sorted(
                    equivalent["query"]
                    if isinstance(equivalent["query"][0], list)
                    else [equivalent["query"]]
                ),
                lexical_equal=a == c,
                chunk_parity=all(
                    all(
                        v["values"] == row["lexical_arms"][0]["values"] and v["complete"]
                        for v in row["lexical_arms"]
                    )
                    for row in local
                ),
            )
        )
        progress("worlds", index + 1, len(worlds) - index - 1)
    if not worlds:
        rows = [
            dict(
                unit_id=f"{family}:{i:02d}:{view}",
                world_id=f"{family}:{i:02d}",
                view=view,
                truth=None,
                status="censored",
                censored=True,
                excluded=False,
                absolute_metric=0,
                lexical_arms=[],
                head=numerical([], None, {}),
            )
            for family in upstream.FAMILIES
            for i in range(40)
            for view in ["supported", "equivalent_wording", "contradicted", "explicit_uncertainty"]
        ]
    complete = [r for r in symbolic if r["arm"] == "complete_records"]
    deleted_rows = [r for r in symbolic if r["arm"] == "deleted_required_records"]
    capacity = (
        len(rows) == 480
        and len({r["world_id"] for r in rows}) == 120
        and all(r["absolute_metric"] for r in rows)
        and all(r["passed"] and r["decision"] == r["original_truth"] for r in complete)
        and all(r["escalated"] for r in deleted_rows)
        and all(
            r["chunk_parity"] and r["truth_equivalent"] and r["query_equivalent"] for r in alignment
        )
    )
    return dict(
        rows=rows,
        collision_rows=collisions,
        symbolic_control_rows=symbolic,
        extraction_error_rows=extraction,
        query_error_rows=query_errors,
        view_alignment_rows=alignment,
        symbolic_capacity_passed=capacity,
        numerical_completed_count=sum(r["head"]["numerical_calls"] > 0 for r in rows),
    )


def checksum(value: Json) -> str:
    """Bind every report field except its checksum to immutable invocation primitives."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def build(work: Json, output: Path) -> Json:
    """A successful oracle control cannot clear a missing numerical comparison operand."""
    data, validation = deepcopy(work["measurement"]), work["validation"]
    owned = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    owned = owned and validation["coverage"]["complete"] and validation["custody"]["passed"]
    numerical_complete = data["numerical_completed_count"] == 480
    ready = int(
        owned
        and work["inputs"]["passed"]
        and work["contract"]["ready"]
        and data["symbolic_capacity_passed"]
        and numerical_complete
        and not work["adversarial_findings"]
    )
    verdict = (
        "null"
        if work["private_control"]
        else "disqualified"
        if not owned or work["adversarial_findings"]
        else "circular_positive"
        if ready
        else "blocked"
    )
    failures = deepcopy(work["inputs"]["failures"] + work["contract"]["gate_check_summary"])
    if not work["resources"]["ready"]:
        failures.append(
            authority.gate(
                Path(work["resources"]["scratch"]), "disk_memory_scratch", True, work["resources"]
            )
        )
    if not numerical_complete:
        failures.append(
            dict(
                authority.gate(
                    Path(
                        work["inputs"]["protocol"]
                        .get("worlds_reference", {})
                        .get("path", str(ROOT / upstream.PROTOCOL))
                    ),
                    "views[*].p2_hallucination_probability",
                    "480 authenticated candidate probabilities",
                    None,
                    "exp8698",
                ),
                operand_state="missing_field",
                missing_count=480 - data["numerical_completed_count"],
            )
        )
    if not owned:
        failures.append(authority.gate(Path(work["raw"]), "owned_validation", True, False))
    if not data["symbolic_capacity_passed"]:
        failures.append(authority.gate(Path(work["raw"]), "all_480_capacity_controls", True, False))
    value: Json = dict(
        experiment_id=8701,
        task_id=TASK,
        milestone="2026.11.746",
        run_date="20261101",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_"
        + verdict
        + (
            "_missing_numerical_head_operand"
            if verdict == "blocked" and not numerical_complete
            else "_v746_relational_feature_audit"
        ),
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier"
        if not numerical_complete
        else "verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=dict(
            name="Exp8431 frozen spline9 ensemble",
            checkpoint=work["head_reference"],
            evaluated_this_invocation=numerical_complete,
            unavailable_operand="p2_hallucination_probability" if not numerical_complete else None,
        ),
        historical_model_provenance=dict(
            scope="Historical numerical fit only; no current language-model invocation."
        ),
        rows=data["rows"],
        intended_count=480,
        completed_count=sum(r["status"] == "completed" for r in data["rows"]),
        failed_count=sum(r["status"] == "failed" for r in data["rows"]),
        censored_count=sum(r["censored"] for r in data["rows"]),
        excluded_count=0,
        independent_count=len(data["view_alignment_rows"]),
        sample_size_budget=dict(
            constructed_worlds=120,
            repeated_views=480,
            lexical_chunk_arms=3,
            symbolic_record_arms=3,
            natural_questions=0,
            repeats_multiply_evidence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="Fully exposed constructed oracle records; no natural-language extractor is evaluated.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(
            not work["private_control"] and (not owned or work["adversarial_findings"])
        ),
        acceptance_gates=dict(
            original_branch_inputs=work["inputs"]["passed"],
            full_v746_contract=work["contract"]["ready"],
            symbolic_capacity=data["symbolic_capacity_passed"],
            numerical_operands_complete=numerical_complete,
            owned_validation=bool(owned),
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        retained_child_custody=validation["custody"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["resources"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7448670,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"] + validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=dict(symbolic_control_ready_score=ready),
        symbolic_control_ready_score=ready,
        symbolic_capacity_passed=data["symbolic_capacity_passed"],
        numerical_completed_count=data["numerical_completed_count"],
        numerical_head_invocations=sum(r["head"]["numerical_calls"] for r in data["rows"]),
        collision_rows=data["collision_rows"],
        symbolic_control_rows=data["symbolic_control_rows"],
        extraction_error_rows=data["extraction_error_rows"],
        query_error_rows=data["query_error_rows"],
        view_alignment_rows=data["view_alignment_rows"],
        historical_quarantines=work["inputs"]["historical_quarantines"],
        preserved_nulls=[
            "V733_source_information",
            "V735_cold_cost",
            "V737_supervisor",
            "V743_batch_one",
        ],
        physical_fpga_execution=False,
        execution_authority=work["contract"],
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology_note="Compare unchanged complete-source lexical vectors at three chunks. Fixed parameterized SQL checks supplied integer records. Fault injections distinguish omission, relation, query and decision errors. Unknown always escalates. Exact collisions are representational counterexamples, not population hallucination accuracy. Complete-record agreement is circular_positive capacity only, not natural extraction, independent generalization, learning or physical FPGA evidence. The nine-input head cannot run without authenticated p2.",
    )
    value["field_principles"] = {
        k: "Retain the exact invocation evidence and its constructed scope for " + k + "."
        for k in value
    }
    value["field_principles"].update(
        symbolic_control_ready_score="One requires all intended arms and qualified original authority; absent p2 keeps this zero.",
        independent_count="120 world IDs; views, chunks and record arms do not multiply independence.",
        field_principles="Explain every field's evidence scope.",
        reproducibility_checksum="Hash all report fields except this checksum.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value
