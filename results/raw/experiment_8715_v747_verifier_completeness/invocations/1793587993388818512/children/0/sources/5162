"""REQ-REPORT-8712: measure information loss without inventing a ninth model input.

The worlds and records are exposed constructed controls. Equal lexical vectors
with opposite truths refute sufficiency; they do not estimate population accuracy.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from carnot.reporting import relational_evidence_8698 as upstream
from carnot.reporting import v747_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.relational_feature_8701 import sql_decision
from carnot.verify import streaming_evidence as stream

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8712_v747_relational_capacity", "exp8712-relational-capacity"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/relational_capacity{s}_8712.py" for s in ("", "_runner")] + [CLI]
TESTS = ["tests/python/test_relational_capacity_8712.py"]
CORPUS = "results/experiment_8698_v746_relational_evidence_controls.json"
CORPUS_PIN = "sha256:2d32e3129192287b372ea342b3e56b9139d18ea5a7eae23347151cf822df4d8a"
MODEL_SPECS: list[Json] = []
CHUNKS = [1, 32, 128]
ARMS = ["complete_records", "deleted_required_records", "reversed_relations"]
reference, load, checksum = upstream.reference, upstream.load, upstream.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a supervisor can distinguish active work from a stall."""
    print(f"[exp8712] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path) -> Json:
    """Authenticate original scientific operands without requiring another current branch."""
    branch = upstream.inputs(root)
    refs, failures = branch["references"], branch["failures"]
    ref = dict(path=str(root / CORPUS), sha256=CORPUS_PIN)
    custody = upstream.custody(ref, "corpus_ready_score", 8698)
    refs += custody["references"]
    failures += custody["gate_check_summary"]
    try:
        corpus = load(ref)
        original = upstream.qualify(branch["worlds"])
        if not original["ready"] or corpus["rows"] != original["rows"]:
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
    return dict(branch, passed=not failures, references=refs, failures=failures)


def measure(worlds: list[Json], raw: Path) -> Json:
    """Keep every view and arm; SQL receives supplied records, never inferred lexical facts."""
    raw.mkdir(parents=True, exist_ok=True)
    rows, lexical, symbolic, collisions, wording = [], [], [], [], []
    for index, world in enumerate(worlds):
        queries = upstream.ground(upstream.question(world))
        query_seal = canonical_hash(queries)
        local = {}
        facts = world["facts"]
        for view in world["views"]:
            uid = world["world_id"] + ":" + view["view"]
            query = upstream.parse_claim(view["claim"], world)
            predicates = query if query is None or isinstance(query[0], list) else [query]
            features = []
            for chunk in CHUNKS:
                path = raw / f"{uid}-{chunk}.jsonl"
                public = dict(
                    family_id=uid,
                    source_bytes=view["source"].encode().hex(),
                    answer_bytes=view["claim"].encode().hex(),
                )
                result = stream.extract(public, path, chunk_size=chunk)
                feature_row = dict(
                    unit_id=uid,
                    world_id=world["world_id"],
                    chunk_size=chunk,
                    values=result["features"]["values"],
                    complete=result["complete"],
                    total_window_count=result["total_window_count"],
                    latency_s=result["latency_s"],
                    window_reference=reference(path),
                    status="completed" if result["complete"] else "failed",
                )
                lexical.append(feature_row)
                features.append(feature_row)
            document = view["document"].encode()
            aligned = (
                document[slice(*view["source_span"])] == view["source"].encode()
                and document[slice(*view["claim_span"])] == view["claim"].encode()
            )
            passed = (
                upstream.evaluate(facts, query) == view["truth"]
                and aligned
                and (query is None or all(p in queries for p in predicates))
                and all(
                    x["complete"] and x["values"] == features[0]["values"] and len(x["values"]) == 8
                    for x in features
                )
            )
            row = dict(
                unit_id=uid,
                world_id=world["world_id"],
                family=world["family"],
                view=view["view"],
                truth=view["truth"],
                query=query,
                query_seal=query_seal,
                source_span=view["source_span"],
                claim_span=view["claim_span"],
                document_sha256=canonical_hash(view["document"]),
                spans_aligned=aligned,
                values=features[0]["values"],
                absolute_metric=int(passed),
                status="completed" if passed else "failed",
                censored=False,
                excluded=False,
            )
            rows.append(row)
            local[view["view"]] = row
            deleted = [r for r in facts if query is not None and r[:2] != predicates[0][:2]]
            reversed_facts = [[b, a, v] for a, b, v in facts]
            for arm, records in zip(ARMS, [facts, deleted, reversed_facts], strict=True):
                started = time.monotonic_ns()
                path = raw / f"{uid}-{arm}.sqlite"
                decision, expected = (
                    sql_decision(path, records, query),
                    upstream.evaluate(records, query),
                )
                passed = (
                    decision == expected
                    and (arm != "complete_records" or decision == view["truth"])
                    and (arm != "deleted_required_records" or decision == -1)
                )
                symbolic.append(
                    dict(
                        unit_id=uid,
                        world_id=world["world_id"],
                        arm=arm,
                        records=records,
                        query=query,
                        query_seal=query_seal,
                        decision=decision,
                        expected=expected,
                        original_truth=view["truth"],
                        passed=passed,
                        absolute_metric=int(passed),
                        status="completed" if passed else "failed",
                        latency_s=(time.monotonic_ns() - started) / 1e9,
                        database_reference=reference(path),
                    )
                )
        a, b, c = (local[v] for v in ["supported", "contradicted", "equivalent_wording"])
        collisions.append(
            dict(
                world_id=world["world_id"],
                family=world["family"],
                left=a["unit_id"],
                right=b["unit_id"],
                left_truth=a["truth"],
                right_truth=b["truth"],
                left_values=a["values"],
                right_values=b["values"],
                exact_equal=a["values"] == b["values"],
                opposite_truth={a["truth"], b["truth"]} == {0, 1},
                scope="representational_counterexample_only",
            )
        )
        wording.append(
            dict(
                world_id=world["world_id"],
                left=a["unit_id"],
                right=c["unit_id"],
                same_truth=a["truth"] == c["truth"] == 1,
                query_equivalent=canonical_hash(sorted(a["query"]))
                == canonical_hash(sorted(c["query"])),
                lexical_equal=a["values"] == c["values"],
                left_values=a["values"],
                right_values=c["values"],
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
            )
            for family in upstream.FAMILIES
            for i in range(40)
            for view in ["supported", "equivalent_wording", "contradicted", "explicit_uncertainty"]
        ]
        lexical = [
            dict(unit_id=r["unit_id"], chunk_size=c, values=None, status="censored")
            for r in rows
            for c in CHUNKS
        ]
        symbolic = [
            dict(
                unit_id=r["unit_id"],
                arm=a,
                decision=None,
                expected=None,
                status="censored",
                passed=False,
            )
            for r in rows
            for a in ARMS
        ]
    relational = (
        len(rows) == 480
        and len(collisions) == 120
        and all(r["absolute_metric"] for r in rows)
        and all(w["same_truth"] and w["query_equivalent"] for w in wording)
    )
    return dict(
        rows=rows,
        lexical_rows=lexical,
        symbolic_rows=symbolic,
        collision_witnesses=collisions,
        wording_changes=wording,
        relational_passed=relational,
        symbolic_passed=len(symbolic) == 1440 and all(r["passed"] for r in symbolic),
    )


def build(work: Json, output: Path) -> Json:
    """Qualify exposed capacity independently of the absent numerical comparison and CUDA."""
    data, validation = deepcopy(work["measurement"]), work["validation"]
    owned = bool(validation["receipts"]) and all(r["passed"] for r in validation["receipts"])
    owned = owned and validation["coverage"]["complete"] and validation["custody"]["passed"]
    eligible = work["inputs"]["passed"] and work["contract"]["ready"] and work["resources"]["ready"]
    incorrect = eligible and not (data["relational_passed"] and data["symbolic_passed"])
    clean = owned and not work["private_control"] and not work["adversarial_findings"]
    relational = int(clean and eligible and data["relational_passed"] and not incorrect)
    symbolic = int(clean and eligible and data["symbolic_passed"] and not incorrect)
    verdict = (
        "null"
        if work["private_control"]
        else "disqualified"
        if not owned or work["adversarial_findings"] or incorrect
        else "circular_positive"
        if relational and symbolic
        else "blocked"
    )
    failures = deepcopy(work["inputs"]["failures"] + work["contract"]["gate_check_summary"])
    for check, passed in [
        ("disk_memory_scratch", work["resources"]["ready"]),
        ("owned_checks_and_coverage", owned),
        ("relational_controls", data["relational_passed"]),
        ("symbolic_controls", data["symbolic_passed"]),
    ]:
        if not passed:
            failures.append(authority.gate(Path(work["raw"]), check, True, False, TASK))
    values: Json = dict(
        experiment_id=8712,
        task_id=TASK,
        milestone="2026.11.747",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_"
        + (
            "external_operand_or_precondition"
            if verdict == "blocked"
            else "v747_relational_capacity"
        ),
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        numerical_head_calls=0,
        target_model=None,
        historical_model_provenance=dict(
            scope="Historical nine-input learned-head comparison remains blocked; no current model or numerical head execution."
        ),
        rows=data["rows"],
        intended_count=480,
        completed_count=sum(r["status"] == "completed" for r in data["rows"]),
        failed_count=sum(r["status"] == "failed" for r in data["rows"]),
        censored_count=sum(r["censored"] for r in data["rows"]),
        excluded_count=0,
        independent_count=len(data["collision_witnesses"]),
        sample_size_budget=dict(
            constructed_worlds=120,
            repeated_views=480,
            lexical_rows=1440,
            symbolic_rows=1440,
            natural_questions=0,
            repeats_multiply_evidence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="Fully exposed constructed worlds and independently supplied oracle records.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(
            not work["private_control"] and (not owned or work["adversarial_findings"] or incorrect)
        ),
        acceptance_gates=dict(
            original_branch_inputs=work["inputs"]["passed"],
            full_v747_contract=work["contract"]["ready"],
            preconditions=work["resources"]["ready"],
            relational_rows=data["relational_passed"],
            symbolic_controls=data["symbolic_passed"],
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
        random_seed=7478712,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["raw_shard_hashes"] + work["raw_references"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=dict(
            relational_rows_ready_score=relational,
            symbolic_control_ready_score=symbolic,
            runtime_repair_ready_score=work["runtime"]["ready"],
        ),
        relational_rows_ready_score=relational,
        symbolic_control_ready_score=symbolic,
        runtime_repair_ready_score=work["runtime"]["ready"],
        runtime_gate_consumer="exp8718",
        optional_runtime_diagnostics=work["runtime"],
        lexical_rows=data["lexical_rows"],
        symbolic_rows=data["symbolic_rows"],
        collision_witnesses=data["collision_witnesses"],
        equivalent_wording_changes=data["wording_changes"],
        missing_probability_count=480,
        numerical_head_comparison_status="blocked_missing_ninth_input",
        numerical_head_gate_check_summary=[
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
                missing_count=480,
            )
        ],
        estimand="Information retained by eight complete-source lexical features and exact symbolic capacity on constructed controls.",
        representation_sufficiency_refuted=any(
            w["exact_equal"] and w["opposite_truth"] for w in data["collision_witnesses"]
        ),
        current_device_probe_count=0,
        physical_fpga_execution=False,
        execution_authority=work["contract"],
        historical_quarantines=work["inputs"]["historical_quarantines"],
        preserved_nulls=[
            "V733_source_information",
            "V735_cold_cost",
            "V737_supervisor",
            "V743_batch_one",
        ],
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology_note="Freeze question predicates before candidate parsing. Unchanged lexical extraction processes every source window at chunks 1/32/128. Independently supplied complete, deleted and reversed records drive fixed parameterized SQL. Collisions are representation counterexamples, not population error estimates. Oracle capacity grants no natural extraction, learning benefit or physical acceleration claim.",
    )
    values["field_principles"] = dict.fromkeys(
        values,
        "Bind this invocation to authenticated original operands; constructed capacity grants no independent scientific benefit.",
    )
    values["field_principles"].update(
        relational_rows_ready_score="One after 480 original views and 1440 chunk rows pass owned validation.",
        symbolic_control_ready_score="One after all 1440 SQL controls match independent truth and intended unknown behavior.",
        runtime_repair_ready_score="Independent supplied causal runtime gate; only Exp8718 consumes it.",
        missing_probability_count="Missing ninth inputs remain absent; never supply constants, labels or fitted proxies.",
        collision_witnesses="Exact opposite-truth equal-feature pairs refute sufficiency of this representation only.",
        field_principles="Explain evidence limits for each field.",
        reproducibility_checksum="Bind every field except this checksum.",
    )
    values["reproducibility_checksum"] = checksum(values)
    return values
