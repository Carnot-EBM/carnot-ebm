"""REQ-REPORT-8442: continuation requires changed evidence rather than scheduling completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v726_capstone_evidence as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]
CLAIMS = {
    "retrospective_criterion": (
        8432,
        "continue_changed_independent_cohort",
        "Release disjoint independently labeled question clusters with resolved exposure and sealed extraction truth; preserve the current public split.",
    ),
    "bounded_feedback_learning": (
        8435,
        "retire_unchanged_measured_null",
        "Change the criterion or admission/update mechanism before a new preregistered delayed-feedback stream; preserve all five retention windows.",
    ),
    "arithmetic_service_acceleration": (
        8437,
        "retire_unchanged_arithmetic_only_attempt",
        "Change the measured nonkernel service path or workload and collect complete paired clocks before proposing native arithmetic or hardware acceleration.",
    ),
    "current_qwen": (
        8438,
        "unmeasured_pre_gate",
        "Supply a hash-bound causal runtime repair and successful bounded context/copy probe before the optional current-Qwen canary.",
    ),
    "arc_cross_game_supervision": (
        8439,
        "blocked_independent_support",
        "Resolve the original publication seal discrepancy and supply three adapter-withheld overlapping games with five resolved firings per shared cell.",
    ),
    "board_operations": (
        8440,
        "blocked_execution_evidence",
        "Resolve original board-local evidence discrepancies; supply supported installed operations and complete device/transport clocks. KV260 quadratic Ising k<=5; PolarFire Linux CPU dispatch only.",
    ),
    "gatemate_execution": (
        8441,
        "blocked_physical_and_source_evidence",
        "Recover the exact missing source hashes plus dated physical cable/port/power change, IDCODE0x20000001, n16 flash and device sample/hash smoke.",
    ),
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualification and benefit are separate so a measured null survives administrative failure."""
    rows = list(work["slots"])
    own = dict(
        task_id=e.TASK,
        experiment_id=8442,
        unit_id=e.TASK,
        family=e.TASK,
        arm="scheduling_slot",
        seed=None,
        disposition="current_reconciliation",
        status="completed",
        censored=False,
        excluded=False,
        absolute_metric=1,
        raw_numerator=1,
        raw_denominator=1,
        effective_independent_groups=0,
        missing_reason=None,
        producer_executed=False,
        eligible=False,
    )
    rows.append(own)
    checks = list(work["checks"])
    for row in rows[:-1]:
        if row["verdict_class"] == "blocked":
            source_gates = row.get("source_gates", [])
            checks.append(
                e.gate(
                    row["primary_reference"],
                    "upstream_scientific_verdict",
                    "measured_qualified_branch",
                    row["honest_verdict"],
                    row["task_id"],
                )
            )
            checks.extend(
                g for g in source_gates if isinstance(g, dict) and g.get("passed") is not True
            )
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work["memory"]["passed"]
        and not work.get("adversarial_findings")
    )
    qualified = qualified and all(r["passed"] for r in work["branch_receipts"])
    verdict = "disqualified" if not qualified else "blocked" if checks else "null"
    claims, provenance, conditions, retirements = {}, {}, {}, []
    for claim, (number, decision, condition) in CLAIMS.items():
        source = next(r for r in rows if r["experiment_id"] == number)
        measured = source.get("producer_executed", False) and source.get("eligible", False)
        claims[claim] = dict(
            decision=decision if measured or number == 8438 else "unmeasured_or_disqualified",
            original_verdict=source.get("honest_verdict"),
            qualified_producer=measured,
            metrics=source.get("selected", {}),
            independent_generalization_score=0,
            generalized_learning_benefit_score=0,
            evidence_scope="retrospective exposed public labels or engineering feasibility only",
        )
        provenance[claim] = dict(
            producer=source["task_id"],
            primary=source.get("primary_reference"),
            imported_fields=list(source.get("selected", {})),
            primitive_reference=source.get("primitive_reference"),
            cold_replay=source.get("branch_replay"),
        )
        conditions[claim] = condition
        if number in [8435, 8437, 8439, 8440, 8441]:
            retirements.append(
                dict(
                    task_id=source["task_id"],
                    predecessor=source.get("primary_reference"),
                    exact_verdict=source.get("honest_verdict"),
                    recommendation="retire_exact_unchanged_repeat",
                    scope=claim,
                    reopening_condition=condition,
                    hypothesis_retired=False,
                )
            )
    for predecessor in work["predecessors"][:2]:
        old = e.load(predecessor)
        if old["experiment_id"] == 8415:
            claims["V717_closed"] = dict(
                H1=old["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"],
                H2=old["H2"]["block_bootstrap_summary"]["mean_gain"],
                decision="keep_closed",
                source=predecessor,
            )
            provenance["V717_closed"] = dict(
                primary=predecessor,
                imported_fields=[
                    "H1.bootstrap_summary.all_intended.mean_gain",
                    "H2.block_bootstrap_summary.mean_gain",
                ],
            )
        retirements.append(
            dict(
                task_id=old["task_id"],
                predecessor=predecessor,
                exact_verdict=old["honest_verdict"],
                recommendation="retire_exact_unchanged_repeat",
                reopening_condition="Supply a genuinely changed qualified producer or repair the named historical validation defect; preserve predecessor bytes.",
                hypothesis_retired=False,
            )
        )
    for receipt in [*receipts, *work["branch_receipts"]]:
        if not receipt["passed"]:
            checks.append(
                e.gate(
                    dict(
                        path=receipt.get("stdout_path", str(raw)),
                        sha256=receipt.get("stdout_sha256"),
                    ),
                    "required_owned_check:" + receipt.get("name", "branch"),
                    True,
                    False,
                )
            )
    checks = [
        dict(
            check=g.get("check", g.get("artifact_field", g.get("field"))),
            upstream=g.get("upstream", g.get("upstream_id", e.TASK)),
            path=g.get("path", g.get("artifact_path")),
            hash=g.get("hash", g.get("sha256", g.get("artifact_hash"))),
            field=g.get("field", g.get("artifact_field")),
            operator=g.get("operator", g.get("op", "==")),
            expected=g["expected"],
            observed=g.get("observed"),
            passed=False,
        )
        for g in checks
    ]
    publication = work["publication"]
    completed = sum(r["status"] == "completed" for r in rows)
    value = dict(
        experiment_id=8442,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date="20261012",
        honest_verdict="complete_disqualified_owned_validation"
        if verdict == "disqualified"
        else f"complete_{verdict}_v726_capstone",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=[
            dict(
                producer=r["task_id"],
                provenance=r.get("historical_model_provenance", []),
                scope="imported historical outputs; zero current model invocations",
            )
            for r in rows[:-1]
            if r.get("historical_model_provenance")
        ],
        rows=rows,
        milestone_outcomes=rows,
        intended_count=14,
        completed_count=completed,
        failed_count=0,
        censored_count=14 - completed,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            intended_scheduling_slots=14,
            independent_scientific_units_added=0,
            public_answer_rows=900,
            public_question_clusters=300,
            older_exposure="unknown",
            seeds_and_siblings_multiply_independence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="public-label retrospective and historical engineering evidence; independent factuality and hidden-game generalization remain open",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(qualified),
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            full_current_authority=work["authority"]["activated"],
            all_slots_accounted=completed == 14,
            required_owned_checks=bool(qualified),
            scientific_claims_independently_generalize=False,
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions"],
        memory_bounds=work["memory"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=726,
        source_artifact_hashes=[*work["refs"], *work["predecessors"]],
        code_config_hashes=[
            ref for ref in work["refs"] if "/code/" in str(ref.get("snapshot_path"))
        ],
        raw_shard_hashes=[e.reference(raw / "measurement.json"), *work.get("coverage_shards", [])],
        primitive_reference=e.reference(raw / "measurement.json"),
        publication_output=str(output),
        cited_upstream_artifacts=[
            dict(r["primary_reference"], producer=r["task_id"], imported_fields=list(r["selected"]))
            for r in rows[:-1]
        ],
        task_count=14,
        producer_count=sum(r.get("producer_executed", False) for r in rows[:-1]),
        pregate_count=sum(r["disposition"] == "pre_gate_receipt" for r in rows),
        cascade_skip_count=sum(r["disposition"] == "cascade_skip" for r in rows),
        absent_count=sum(r["disposition"] == "absent_artifact" for r in rows),
        disqualified_producer_count=sum(r["disposition"] == "disqualified_producer" for r in rows),
        contract_digest=work["contract_digest"],
        scientific_claims=claims,
        per_claim_provenance=provenance,
        next_evidence_conditions=conditions,
        retirement_recommendations=retirements,
        g1=publication["gates"]["G1"],
        g2=publication["gates"]["G2"],
        g3=publication["gates"]["G3"],
        g4=publication["gates"]["G4"],
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        publication_gate_receipt=work["publication_receipt"],
        methodology_note="Read pinned producer bytes and cold-replay shipped reducers in separate bounded CPU processes. Retain all scheduling slots, primitive measures and disqualifications. No model, device or game is run; FoVer publication readiness is independent of V726 scientific success.",
    )
    value["field_principles"] = {
        key: "Invocation-bound aggregation; imported evidence preserves its original scope and source hash."
        for key in value
    }
    value["field_principles"].update(
        acceptance_gates="Authority, complete slot accounting and owned validation qualify the reconciliation; none establishes independent scientific benefit.",
        scientific_claims="Positive public-label contrasts earn only changed independent evidence; measured nulls retire exact unchanged mechanisms.",
        sample_size_budget="Scheduling slots are administrative units; 900 sibling answers contain 300 question clusters, with unknown historical exposure.",
        retirement_recommendations="Recommendations bind exact predecessor bytes; exclusion history is unchanged.",
        paper_ready="Unmodified publication_gate.py G1 through G4; historical FoVer readiness is separate from V726.",
        reproducibility_checksum="Hash of the complete artifact excluding this checksum.",
        field_principles="Every extra field and acceptance gate retains its evidence scope.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value
