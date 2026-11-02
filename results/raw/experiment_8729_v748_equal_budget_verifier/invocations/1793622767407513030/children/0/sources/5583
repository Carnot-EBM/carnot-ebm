"""REQ-REPORT-8601: qualification, scientific benefit and administration remain separate."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_capstone_report as base
from carnot.reporting import v738_capstone_evidence as e
from carnot.reporting.v736_historical_inputs import checked

Json = dict[str, Any]
DECISIONS: Json = dict(
    fixed_brier_learning=dict(
        decision="retire_exact_scope_after_repeated_qualification_failure",
        qualified_benefit=None,
        prior_honest_verdict="complete_disqualified_v737_causal_brier_stream",
        diagnosed_cause="Exp8582 test_production_dispatch_and_deadline exhausted the mocked time.monotonic_ns iterator (StopIteration). Exp8592 required_owned_checks == True observed False: owned coverage assertion failed because python/carnot/reporting/causal_brier_runner_8582.py line94 (second deadline_clock read on the measurement-deadline path) remained uncovered; 3/4 changed runner statements were covered. No unchanged run is recommended.",
        reopen="A separately diagnosed changed qualification mechanism with every original assertion and 100 percent owned coverage; only then register fresh learning and retention, without repairing old hashes.",
    ),
    resident_native_boundary=dict(
        decision="close_qualified_cost_study_separate_native_and_lifecycle",
        qualified_benefit=True,
        prior_honest_verdict="complete_disqualified_v737_resident_complete_cost",
        diagnosed_cause="Exp8584 METHODOLOGY_MISSING model_specs/target_model; Exp8593 qualified explicit spline9 provenance and Exp8596 fresh inclusive clocks. Historical Exp8584 speed stays quarantined.",
        reopen="Exp8596 separately measured resident_native versus resident_python: throughput-ratio CI95 [1.0740525244779107, 1.105777420052135]. Close this unchanged study. A changed mechanism must target the still-unmet NFR01 latency gate in a new registration; lifecycle reuse does not substitute for the native control or revise Exp8559 cold null.",
    ),
    fixed_arc_panel=dict(
        decision="close_qualified_null",
        qualified_benefit=False,
        prior_honest_verdict="complete_null_v737_arc_capability_panel",
        diagnosed_cause="Exp8588 qualified null; Exp8599 refinement_eligible_score=0 and unresolved support do not identify an update.",
        reopen="Changed causal supervisor mechanism and a new independently registered study with resolved cross-game control support; never rerun the unchanged closed panel.",
    ),
)


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Reuse stable row semantics and retain every older quarantine and evidence boundary."""
    with patch.object(base, "e", e):
        value = dict(base.build(work, validation, raw, output))
    prior = e.read(work["protocol"]["original_producers"]["8590"])
    slots = work["slots"]
    by = {r["experiment_id"]: r for r in slots}
    clock = e.read(by[8592]["primary_reference"])
    clock_work = e.read(clock["primitive_reference"])
    clock_coverage = e.read(clock_work["coverage_reference"])
    runner_coverage = clock_coverage["files"]["python/carnot/reporting/causal_brier_runner_8582.py"]
    value.update(
        qualification_failure_evidence=dict(
            primary=by[8592]["primary_reference"],
            primitive=clock["primitive_reference"],
            coverage_reference=clock_work["coverage_reference"],
            failed_assertion="all(not f[summary][missing_lines] for f in coverage[files].values())",
            field="files.python/carnot/reporting/causal_brier_runner_8582.py.summary.missing_lines",
            operator="==",
            expected_missing_statements=0,
            observed_missing_statements=runner_coverage["summary"]["missing_lines"],
            missing_lines=runner_coverage["missing_lines"],
        ),
        prior_owned_attempts=work["prior_owned_attempts"],
        experiment_id=8601,
        run_date=work["run_date"],
        random_seed=7388601,
        task_count=11,
        intended_count=11,
        milestone_status="eleven_slots_reconciled_not_eleven_successful_studies",
        preserved_nulls=deepcopy(prior["preserved_nulls"]),
        quarantined_inputs=deepcopy(prior["quarantined_inputs"])
        + [r for r in slots if r["verdict_class"] == "disqualified"],
        qualified_scientific_results=deepcopy(prior["qualified_scientific_results"])
        + [r for r in slots if r["eligible"] and r["experiment_id"] in {8596, 8599}],
        hallutruth_release_eligibility=deepcopy(prior["hallutruth_release_eligibility"]),
        continuation_decisions=DECISIONS,
        retirement_recommendations=[dict(scope=k, **v) for k, v in DECISIONS.items()],
        conditional_exposed_benefits=dict(
            learning=None,
            resident_lifecycle=True,
            resident_native=True,
            nfr01=False,
            arc=False,
            generalization_credit=0,
        ),
        count_revision_chain=deepcopy(prior["count_revision_chain"]),
        v737_scheduling_record=dict(
            task_count=12,
            successful_studies_not_implied=True,
            primary=work["protocol"]["original_producers"]["8590"],
        ),
        medexpert_evidence_eligibility=dict(
            primary_reference=by[8597]["primary_reference"],
            audit_completed=True,
            evaluation_eligible=False,
            release_available=False,
            independent_truth=False,
        ),
        methodology_note="Eleven V738 scheduling dispositions preserve original evidence, missing cascades and quarantines. Numerical spline9 arithmetic is diagnostic inference, with no language-model load or generation. Exposed learning is at most circular_positive. Resident lifecycle and native arithmetic gains use separate paired controls. NFR01 remains unmet. MedExpert audit completion grants no independent evaluation. Qualified source, cold-cost and ARC nulls remain closed. Original G1-G4 grant no publication authority.",
    )
    value["next_evidence_requirements"].update(
        learning_qualification="Diagnose and cover causal_brier_runner_8582.py line94 under original authority; every original test and all changed statements must qualify before a fresh learning/retention study.",
        medexpert="Authenticated original and assisted sentence bytes, explicit assisted-use rights, blinded-label independence and a sealed disjoint question inventory; audit completion alone cannot reopen evaluation.",
        arc="For each held-out game require three overlapping training games and five resolved observations in every shared arm/mode/game cell; zero support forbids refinement or unchanged panel retries.",
        resident_latency="A separately registered changed numerical or service mechanism that passes the original NFR01 latency condition; preserve both current paired native and lifecycle contrasts.",
    )
    owned = (
        value["required_checks_passed"]
        and all(
            r["matches_original"]
            for r in work["samples"]["retention_sample"] + work["samples"]["learning_losses"]
        )
        and work["samples"]["qualified_resident_sample"]["matches_original"]
    )
    value["required_checks_passed"], value["flagged_adversarial"] = bool(owned), not owned
    value["verdict_class"] = "blocked" if owned else "disqualified"
    value["honest_verdict"] = f"complete_{value['verdict_class']}_v738_capstone"
    value["rows"][-1].update(
        experiment_id=8601,
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        eligible=bool(owned),
    )
    value["sample_size_budget"]["intended_scheduling_slots"] = 11
    value["acceptance_gates"].update(
        owned_validation=bool(owned),
        all_slots_accounted=len(value["rows"]) == 11,
        qualified_resident_cost=by[8596]["eligible"],
        qualified_arc_panel=False,
    )
    value["cited_upstream_artifacts"] += [r["primary_reference"] for r in slots]
    value["field_principles"].update(
        {
            k: "Bind original producer qualification and exact invocation operands; completion supplies no independent scientific credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct original claims so repaired envelope hashes cannot manufacture evidence."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.checksum(value) != value["reproducibility_checksum"]
            or e.methods.protocol() != work["protocol"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] < 3600
        ):
            return False
        for ref in work["source_refs"] + work["code_refs"]:
            if ref.get("exists", True) and not checked(
                dict(ref, path=ref.get("snapshot_path", ref["path"]))
            ):
                return False
        if e.methods.authority(Path(work["root"])) != work["contract"]:
            return False
        if (
            json.loads(Path(work["publication_receipt"]["stdout_path"]).read_bytes())
            != work["publication"]
        ):
            return False
        if value != build(
            work,
            value["validation_reference"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        ):
            return False
        tasks = work["protocol"]["task_map_v738"]
        log = next(
            r
            for r in work["source_refs"]
            if r["path"].endswith("ops/conductor-log.md") and "snapshot_path" in r
        )
        lines = Path(log["snapshot_path"]).read_text().splitlines()
        for row, task in zip(work["slots"], tasks[:-1], strict=True):
            ref = row["primary_reference"]
            summary = e.outcome(task, ref, Path(ref["path"]).parent) if ref["exists"] else None
            cascade = next(
                (
                    s
                    for s in lines
                    if "GATE_BLOCK" in s and s.split("|")[2].strip() == task["title"][:50].strip()
                ),
                None,
            )
            if row != e.old.slot(task, ref, cascade, summary):
                return False
        with TemporaryDirectory(prefix="exp8601-replay-", dir="/var/tmp") as directory:
            diagnostic, receipt = e.samples(Path(directory))
        if not receipt["passed"] or work["samples"] != diagnostic:
            return False
        for receipt in [
            *work["branch_receipts"],
            *value["validation_receipts"],
            work["publication_receipt"],
        ]:
            original = Path(receipt["stdout_path"]).with_name(receipt["name"] + ".receipt.json")
            if json.loads(original.read_bytes()) != receipt:
                return False
            for stream in ["stdout", "stderr"]:
                if not checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def consumer(output: Path) -> Json:
    """Both unchanged readers must select exactly the published primary bytes."""
    with patch.object(base, "e", e):
        return dict(base.consumer(output))
