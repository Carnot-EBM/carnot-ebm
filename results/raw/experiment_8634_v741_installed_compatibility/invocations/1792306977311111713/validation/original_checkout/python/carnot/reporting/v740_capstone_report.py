"""REQ-REPORT-8626: current qualification cannot rewrite an older failed study."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v739_capstone_report as previous
from carnot.reporting import v740_capstone_evidence as e
from carnot.reporting import v740_capstone_samples as samples

Json = dict[str, Any]
_BUILD, _REPLAY = previous.build, previous.replay


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Reuse established field accounting while binding exactly twelve current slots."""
    adapted = dict(
        work, protocol=dict(work["protocol"], task_map_v739=work["protocol"]["task_map_v740"])
    )
    with patch.object(previous, "e", e):
        value = _BUILD(adapted, receipts, output)
    checks_ok = all(r["matches_original"] for r in work["samples"]["complete_lean_cost"]["rows"])
    checks_ok = (
        checks_ok
        and work["samples"]["triviaplus_coverage"]["matches_original"]
        and work["samples"]["current_arc_horizons"]["matches_original"]
        and work["samples"]["released_feedback"]["matches_original"]
    )
    owned = value["required_checks_passed"] and checks_ok
    verdict = "blocked" if owned else "disqualified"
    value.update(
        experiment_id=8626,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        honest_verdict=f"complete_{verdict}_v740_capstone",
        verdict_class=verdict,
        random_seed=7408626,
        intended_count=12,
        task_count=12,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        archive_authority=dict(
            completed_archive_ends_at="2026.10.738",
            during_planning=True,
            current_evidence="V740 original producers and conductor receipts",
            successful_studies_not_implied=True,
        ),
        sample_size_budget=dict(
            task_slots=12,
            independent_questions=0,
            fixed_diagnostic_only=True,
            repetitions_are_not_independent=True,
        ),
    )
    value["rows"][-1].update(
        experiment_id=8626,
        task_id=e.TASK,
        unit_id=e.TASK,
        eligible=owned,
        verdict_class=verdict,
        honest_verdict=value["honest_verdict"],
    )
    value["failed_count"] = sum(r["verdict_class"] == "disqualified" for r in value["rows"])
    value["acceptance_gates"].update(
        owned_validation=owned,
        all_slots_accounted=len(value["rows"]) == 12,
        qualified_lean_cost=any(
            r["eligible"] and r["experiment_id"] == 8622 for r in work["slots"]
        ),
    )
    value["qualified_scientific_results"] += [r for r in work["slots"] if r["eligible"]]
    value["quarantined_inputs"] = [
        r for r in value["quarantined_inputs"] if r["experiment_id"] != 8614
    ]
    value["cited_upstream_artifacts"] = work["source_refs"]
    value["scheduling_completion_count"] = value["completed_count"]
    value["executed_producer_count"] = sum(r["producer_executed"] for r in value["rows"])
    value["conditional_exposed_learning_benefit_score"] = 0
    value["learning_execution_ready_score"] = int(
        work["samples"]["source_bound_deadline"]["current_qualified"]
    )
    value["complete_cost_benefit_score"] = int(
        owned
        and value["acceptance_gates"]["qualified_lean_cost"]
        and work["samples"]["complete_lean_cost"]["lean_service_benefit_score"]
    )
    value["nfr01_score"] = int(work["samples"]["complete_lean_cost"]["nfr01_score"])
    value["branch_gate_values"] = {
        k: value[k]
        for k in [
            "learning_execution_ready_score",
            "conditional_exposed_learning_benefit_score",
            "independent_generalization_score",
            "generalized_learning_benefit_score",
            "complete_cost_benefit_score",
            "nfr01_score",
        ]
    }
    value["complete_cost_findings"] = work["samples"]["complete_lean_cost"]
    value["preserved_v739_authority"] = dict(
        missing_design_authority=True,
        source=e.reference(e.ROOT / "results/experiment_8614_v739_capstone.json"),
    )
    value["next_evidence_requirements"] = dict(
        deadline="Retain original child execution shards, exact historical test union and semantic deadline cases before fresh learning; normalization94=92 alone does not qualify Exp8616.",
        learning="Fresh qualified causal predictions, released feedback and simultaneous retention audit under sealed methods; unknown prior exposure stays unknown.",
        labels="New factual correctness or independent question labels; Trivia+ lexical/source faithfulness coverage is not factuality or a Qwen result.",
        lean="Changed registered lifecycle or native arithmetic mechanism and fresh complete matched costs; NFR01 remains unmet.",
        qwen="Supplied causal CUDA library/environment repair hashes and genuine context/copy success before a new leased canary.",
        arc="Changed supervisor with causal cross-game support under original floors; observer fixtures justify mechanics only.",
        hardware="Compatible measured physical execution, qualified learning costs and retained full host/lifecycle work; no repeated unchanged board probes.",
    )
    prior = e.read(e.reference(e.ROOT / "results/experiment_8614_v739_capstone.json"))
    quarantined = {r["experiment_id"] for r in value["quarantined_inputs"]}
    value["quarantined_inputs"] += [
        r for r in prior["quarantined_inputs"] if r["experiment_id"] not in quarantined
    ]
    for task, decision, row in zip(
        work["protocol"]["task_map_v740"][:-1],
        value["continuation_decisions"],
        work["slots"],
        strict=True,
    ):
        old: list[Json] = []
        for registration in task.get("prior_failures", []):
            number = int(registration["experiment_id"][3:7])
            old.extend(r for r in prior["rows"] if r["experiment_id"] == number)
        decision.update(
            actual_prior_outcomes=old,
            changed_prerequisite=task.get("prior_failures", []),
            decision="close_qualified_null"
            if row["eligible"] and row["verdict_class"] == "null"
            else "retain_qualified_result"
            if row["eligible"]
            else "retire_exact_unchanged_scope",
        )
        decision["reopen"] = deepcopy(value["next_evidence_requirements"])
    value["continuation_decisions"].append(
        dict(
            task_id=e.TASK,
            prior_failures=work["protocol"]["task_map_v740"][-1]["prior_failures"],
            actual_prior_outcomes=[prior["rows"][-1]],
            observed_verdict=value["honest_verdict"],
            decision="retire_exact_unchanged_scope",
            reopen=deepcopy(value["next_evidence_requirements"]),
        )
    )
    value["retirement_recommendations"] = deepcopy(value["continuation_decisions"])
    value["methodology_note"] = (
        "Twelve scheduling dispositions are separate from executed science. V739 authority failure and Exp8603 disqualification remain. Source-bound accounting diagnoses94=92 but missing Exp8616 original shards still disqualify. Preserve V733 source, V735 cold-cost and V737 ARC nulls. V738 native CI95[1.0740525,1.1057774] is distinct from lifecycle reuse and new lean gains. Exposed learning, independent evaluation, complete costs and NFR01 have separate gates. Trivia+ faithfulness is not factuality. Observer fixtures change no supervisor priorities or solve counts. G1-G4 concern the older FoVer headline and authorize no publication."
    )
    value["field_principles"] = {
        k: "Bind original evidence to this invocation; scheduling, eligibility and numerical fixtures do not establish independent scientific benefit."
        for k in [*value, "field_principles"]
    }
    for k in value["branch_gate_values"]:
        value["field_principles"][k] = (
            "Numeric0/1 reports only the named qualified branch; missing operands remain named failures, never measured zero outcomes."
        )
    value["field_principles"].update(
        administrative_complete="All twelve scheduling dispositions have terminal accounting; this does not count successful experiments.",
        executed_producer_count="Count original producer executions separately from conductor pre-gates and cascade skips.",
        learning_execution_ready_score="One requires source-bound original executable coverage and all deadline qualification assertions; missing retained shards keep this zero.",
        conditional_exposed_learning_benefit_score="One requires qualified current causal learning and a separate retention audit; historical exposed arithmetic adds no new benefit.",
        independent_generalization_score="Zero because this aggregation creates no independent evaluation or known unexposed question sample.",
        generalized_learning_benefit_score="Zero because exposed learning and fixture qualification cannot establish generalized learning benefit.",
        complete_cost_benefit_score="One only for qualified inclusive lean session costs and the registered lifecycle benefit gate; native arithmetic is a separate contrast.",
        nfr01_score="Preserve the producer's unchanged complete-cost NFR01 gate, even when a lean lifecycle speedup is present.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return dict(value)


def replay(path: Path) -> None:
    """The existing replay checks hashes, clocks, receipts and reconstructed full claims."""
    with patch.multiple(previous, e=e, build=build):
        _REPLAY(path)
