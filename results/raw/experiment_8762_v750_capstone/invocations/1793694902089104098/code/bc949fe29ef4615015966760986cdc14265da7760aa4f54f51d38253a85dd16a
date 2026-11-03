"""REQ-REPORT-8715: separate constructed verifier coverage from scientific benefit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import relational_evidence_8698 as upstream
from carnot.reporting import v747_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8715_v747_verifier_completeness", "exp8715-verifier-completeness"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/verifier_completeness{s}_8715.py" for s in ("", "_runner")] + [
    CLI
]
TESTS = ["tests/python/test_verifier_completeness_8715.py"]
PRODUCER = "results/experiment_8712_v747_relational_capacity.json"
PRODUCER_PIN = "sha256:70d65744e10471aa67a77671dbb894b766c587bd59988e0600766cc43765829f"
VARIANTS = (
    "correct",
    "reject_all",
    "accept_all",
    "omitted_required_predicate",
    "reversed_arguments",
    "missing_as_false",
)
BASELINE = ("truth", "equivalent", "invalid", "unknown", "deleted_records")
MODEL_SPECS: list[Json] = []
reference, load, checksum = upstream.reference, upstream.load, upstream.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual progress so a supervisor can distinguish a stall from work."""
    print(f"[exp8715] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path) -> Json:
    """Authenticate original labels and the declared producer rather than a scheduling skip."""
    branch = upstream.inputs(root)
    ref = dict(path=str(root / PRODUCER), sha256=PRODUCER_PIN)
    custody = upstream.custody(ref, "symbolic_control_ready_score", 8712)
    branch["references"] += custody["references"]
    branch["failures"] += custody["gate_check_summary"]
    branch["failures"] += [
        g for ref in branch["references"] if not (g := upstream.authenticate(ref))["passed"]
    ]
    branch["passed"] = not branch["failures"]
    return branch


def oracle(facts: list[list[int]], query: Any) -> int:
    """Derive labels by explicit membership without calling any evaluated variant."""
    if query is None or any(not any(r[:2] == q[:2] for r in facts) for q in query):
        return -1
    return int(all(q in facts for q in query))


def verifier(variant: str, facts: list[list[int]], query: Any) -> int:
    """Inject one frozen defect while keeping the correct variant on the shipped evaluator."""
    if variant not in VARIANTS:
        raise ValueError("variant")
    if variant == "reject_all":
        return 0
    if variant == "accept_all":
        return 1
    if query is not None:
        query = query[:1] if variant == "omitted_required_predicate" else query
        query = [[b, a, v] for a, b, v in query] if variant == "reversed_arguments" else query
    actual = upstream.evaluate(facts, query)
    return 0 if variant == "missing_as_false" and actual == -1 else actual


def freeze(worlds: list[Json]) -> Json:
    """Freeze answer-independent questions, transformations and independent expected labels."""
    rows = []
    for index, world in enumerate(worlds):
        question = upstream.question(world)
        grounded = upstream.ground(question)
        query = [[0, 1, 1], [0, 2, 0]] if world["family"] == "negation_attachment" else [[0, 1, 1]]
        assert all(p in grounded for p in query)
        facts = deepcopy(world["facts"])
        invalid = [[a, b, 1 - v] for a, b, v in query]
        probes = []

        def add(name: str, records: list[list[int]], predicates: Any) -> None:
            probes.append(
                dict(
                    transform=name,
                    facts=records,
                    query=predicates,
                    expected=oracle(records, predicates),
                    suite="baseline" if name in BASELINE else "extended",
                )
            )

        add("truth", facts, query)
        add("equivalent", facts, list(reversed(query)))
        add("invalid", facts, invalid)
        add("unknown", facts, None)
        add("deleted_records", [], query)
        add("record_order", list(reversed(facts)), query)
        rename = {0: 13, 1: 7, 2: 29}
        add(
            "entity_bijection",
            [[rename[a], rename[b], v] for a, b, v in facts],
            [[rename[a], rename[b], v] for a, b, v in query],
        )
        add("irrelevant_fact", facts + [[31, 32, 1]], query)
        last = query[-1]
        add("required_premise", [r for r in facts if r[:2] != last[:2]], query)
        add(
            "required_role",
            [[a, b, 1 - v] if [a, b] == last[:2] else [a, b, v] for a, b, v in facts],
            query,
        )
        add("reversed_records", [[b, a, v] for a, b, v in facts], query)
        rows.append(
            dict(
                world_id=world["world_id"],
                partition="sealed_evaluation" if (index + 1) % 4 == 0 else "implementation",
                question=question,
                grounded_queries=grounded,
                query_sha256=canonical_hash(grounded),
                candidate_read_before_grounding=False,
                probes=probes,
            )
        )
    return dict(
        variants=list(VARIANTS),
        worlds=rows,
        label_source="independent explicit-membership oracle",
        split_rule="every fourth original roster ID; no fitting",
    )


def measure(frozen: Json, raw: Path) -> Json:
    """Execute both suites without short circuiting; repeats never add independent worlds."""
    raw.mkdir(parents=True, exist_ok=True)
    rows, probes, defects, alternatives, costs, witnesses = [], [], [], [], [], []
    for index, world in enumerate(frozen["worlds"]):
        for variant in frozen["variants"]:
            local = []
            for probe in world["probes"]:
                actual = verifier(variant, probe["facts"], probe["query"])
                row = dict(
                    world_id=world["world_id"],
                    partition=world["partition"],
                    variant=variant,
                    **deepcopy(probe),
                    actual=actual,
                    passed=actual == probe["expected"],
                    probe_cost=1,
                )
                probes.append(row)
                local.append(row)
            baseline = all(p["passed"] for p in local if p["suite"] == "baseline")
            extended = all(p["passed"] for p in local)
            identity = dict(
                world_id=world["world_id"], partition=world["partition"], variant=variant
            )
            rows.append(
                dict(
                    identity,
                    status="completed",
                    absolute_metric=int(extended),
                    baseline_passed=baseline,
                    extended_passed=extended,
                    censored=False,
                    excluded=False,
                )
            )
            defects.append(
                dict(
                    identity,
                    intended_defect=variant != "correct",
                    baseline_detected=not baseline,
                    extended_detected=not extended,
                )
            )
            costs.append(
                dict(
                    identity,
                    baseline_probes=len(BASELINE),
                    extended_probes=len(local),
                    shared_baseline_probes=len(BASELINE),
                    actual_executed_probes=len(local),
                )
            )
            alternatives.append(
                dict(
                    identity,
                    valid=sum(p["expected"] == 1 for p in local),
                    accepted_valid=sum(p["expected"] == 1 and p["actual"] == 1 for p in local),
                    invalid=sum(p["expected"] == 0 for p in local),
                    rejected_invalid=sum(p["expected"] == 0 and p["actual"] == 0 for p in local),
                    unknown=sum(p["expected"] == -1 for p in local),
                    preserved_unknown=sum(p["expected"] == -1 and p["actual"] == -1 for p in local),
                )
            )
            if variant != "correct" and baseline:
                failures = [p for p in local if not p["passed"]]
                witnesses.append(
                    dict(
                        identity,
                        missed_by="baseline" if failures else "both",
                        witness=next(
                            (p for p in failures if p["transform"] == "required_role"),
                            failures[0] if failures else None,
                        ),
                        explanation="Omitted predicate accepts a present but wrong relation."
                        if failures
                        else "Single-predicate query cannot expose conjunction omission; no semantic completeness claim.",
                    )
                )
        progress("worlds", index + 1, len(frozen["worlds"]) - index - 1)
    atomic_json(raw / "probes.json", dict(frozen=frozen, probes=probes))
    return dict(
        rows=rows,
        variant_world_rows=rows,
        probe_rows=probes,
        defect_detection_rows=defects,
        valid_alternative_rows=alternatives,
        unresolved_defect_witnesses=witnesses,
        probe_cost_rows=costs,
        heldout_world_ids=[
            w["world_id"] for w in frozen["worlds"] if w["partition"] == "sealed_evaluation"
        ],
    )


def build(work: Json, output: Path) -> Json:
    """Keep readiness independent of defect gains and deny all unqualified positive claims."""
    measurement, validation = work["measurement"], work["validation"]
    rows = deepcopy(measurement["rows"])
    if not rows:
        rows = [
            dict(
                world_id=f"unavailable:{i}",
                variant=v,
                status="censored",
                absolute_metric=0,
                censored=True,
                excluded=False,
            )
            for i in range(120)
            for v in VARIANTS
        ]
    owned = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
        and validation["custody"]["passed"]
    )
    complete = (
        len(measurement["rows"]) == 720
        and len(measurement["heldout_world_ids"]) == 30
        and all(r["extended_passed"] for r in measurement["rows"] if r["variant"] == "correct")
    )
    complete = complete and all(
        not r["baseline_passed"]
        for r in measurement["rows"]
        if r["variant"] in {"reject_all", "accept_all"}
    )
    heldout = [
        r
        for r in measurement["defect_detection_rows"]
        if r["partition"] == "sealed_evaluation" and r["intended_defect"]
    ]
    gain = sum(r["extended_detected"] and not r["baseline_detected"] for r in heldout)
    failures = deepcopy(work["inputs"]["failures"] + work["contract"]["gate_check_summary"])
    for field, passed in [
        ("disk_memory_scratch", work["resources"]["ready"]),
        ("owned_checks_and_coverage", owned),
        ("complete_720_units", complete),
    ]:
        if not passed:
            failures.append(authority.gate(Path(work["raw"]), field, True, False))
    ready = int(
        owned
        and complete
        and work["inputs"]["passed"]
        and work["contract"]["ready"]
        and work["resources"]["ready"]
        and not work["adversarial_findings"]
        and not work["private_control"]
    )
    verdict = (
        "null"
        if work["private_control"]
        else "disqualified"
        if not owned or work["adversarial_findings"]
        else "blocked"
        if not ready
        else "circular_positive"
        if gain
        else "null"
    )
    value: Json = dict(
        experiment_id=8715,
        task_id=TASK,
        milestone="2026.11.747",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_"
        + ("missing_external_operand" if verdict == "blocked" else "v747_verifier_completeness"),
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        target_model=None,
        historical_model_provenance=dict(
            scope="Original constructed worlds only; no current or cached language-model predictions."
        ),
        intended_count=720,
        completed_count=len(measurement["rows"]),
        failed_count=0,
        censored_count=720 - len(measurement["rows"]),
        excluded_count=0,
        independent_count=len(work["frozen"]["worlds"]),
        sample_size_budget=dict(
            independent_constructed_worlds=120,
            implementation_worlds=90,
            sealed_worlds=30,
            variant_world_units=720,
            sealed_variant_world_units=180,
            natural_questions=0,
            repeats_multiply_evidence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="Constructed controls; every fourth original world reserved until frozen evaluation. No fitting.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(
            not work["private_control"] and (not owned or work["adversarial_findings"])
        ),
        acceptance_gates=dict(
            original_operands=work["inputs"]["passed"],
            full_v747_contract=work["contract"]["ready"],
            complete_constructed_controls=complete,
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
        random_seed=7478715,
        source_artifact_hashes=work["inputs"]["references"] + work["contract"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["raw_shard_hashes"] + work["raw_references"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=dict(verifier_audit_ready_score=ready),
        verifier_audit_ready_score=ready,
        execution_authority=work["contract"],
        canonical_tasks_sha256=authority.PIN,
        sealed_defect_count=len(heldout),
        baseline_detected_defect_count=sum(r["baseline_detected"] for r in heldout),
        extended_detected_defect_count=sum(r["extended_detected"] for r in heldout),
        incremental_detected_defect_count=gain,
        heldout_valid_alternative_rows=[
            r
            for r in measurement["valid_alternative_rows"]
            if r["partition"] == "sealed_evaluation"
        ],
        heldout_probe_cost_rows=[
            r for r in measurement["probe_cost_rows"] if r["partition"] == "sealed_evaluation"
        ],
        probe_cost_comparison="Extended suite includes baseline probes and costs more; no equal-budget superiority claim.",
        physical_fpga_execution=False,
        historical_quarantines=work["inputs"]["historical_quarantines"],
        preserved_nulls=[
            "V733_source_information",
            "V735_cold_cost",
            "V737_supervisor",
            "V743_batch_one",
        ],
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology_note="Adapt VerifierBench valid alternatives and ReLoop behavioral perturbations to explicit constructed relations. Labels are independent of tested variants. All faults are constructed, not LLM-produced error frequencies. Passing perturbations cannot prove natural-language correctness, learning, generalization or physical acceleration. No production verification or canary change.",
        **{k: v for k, v in measurement.items() if k != "rows"},
        rows=rows,
    )
    value["field_principles"] = dict.fromkeys(
        value,
        "Bind exact constructed evidence and invocation; readiness does not grant independent generalization or physical execution.",
    )
    value["field_principles"].update(
        field_principles="Explain scope for every field.",
        reproducibility_checksum="Bind all fields except this checksum.",
        verifier_audit_ready_score="One only after complete qualified evidence, independent of incremental defect gain.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value
