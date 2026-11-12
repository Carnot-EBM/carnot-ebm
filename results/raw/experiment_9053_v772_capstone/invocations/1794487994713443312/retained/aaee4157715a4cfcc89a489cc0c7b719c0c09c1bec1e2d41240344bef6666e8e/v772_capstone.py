"""REQ-REPORT-9053-V772: complete thirteen outcomes without inventing science."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v771_capstone as old
from carnot.reporting import v772_branch_contract as authority
from carnot.reporting import v767_capstone as accounting
from carnot.reporting import identity_obstruction_runner_9043 as identity
from carnot.reporting import annotation_feasibility_9044 as annotation_source
from carnot.reporting import identity_obstruction_9043 as identity_source
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v771_branch_evidence as history
from carnot.reporting import v770_capstone as lifecycle
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v768_capstone import edge
from carnot.reporting.v766_branch_evidence import arc_support
from carnot.reporting.invocation_provenance_9000 import environment

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9053_v772_capstone", "exp9053-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v772_capstone.py", CLI]
TEST = "tests/python/test_v772_capstone_9053.py"
DIRECT = [*old.DIRECT, annotation_source.TEST]
REUSED = list(
    dict.fromkeys(
        [
            *old.REUSED,
            *old.OWNED,
            *authority.OWNED,
            *identity_source.OWNED,
            *annotation_source.OWNED,
        ]
    )
)
CHECKS, checksum = old.CHECKS, old.checksum
MODEL_SPECS: list[Json] = []
PINS = dict(
    zip(
        [9041, 9042, 9043, 9044, 9045, 9046, 9052],
        [
            "sha256:1837f2e88e964d63bb25c890dc3b4bad0292df90e4690c6c28d54c803476aba2",
            "sha256:1cb85452aabf4e851823a675f6137ff5f51571768cae6881b8ba2b47c8ae7a12",
            "sha256:644e3db049e45b9fb9d80984a7083bf4afd32b53a0cb7eb4b9214227a758e324",
            "sha256:288ad02b0b47ff820a9afef66269a640a42a6465d389af002368fe6e9cfeec3a",
            "sha256:2fc94ea71538440a38e03c4e679ce446767dd07e81238d8f3248da4e202566f4",
            "sha256:da6a56e3bc89675c59c7ef8857fb563203f6e9c12689c695f2ad31d9f965d8bf",
            "sha256:a093ff28e7c43bce8628e794837a0d4eb16c33a19e7bac9644ea41a0ee250f33",
        ],
        strict=True,
    )
)
AUTHORITY_PINS = [
    "sha256:b8114003aa7be73393edd3e70a731df947e89176f69fbce764cb4be8014897fc",
    "sha256:5cf26fc9e212725527f9243707a92e9f2b7257944de19aef9f522e1f0695f6ab",
]
LOG_PIN = "sha256:6bd330458ead50adb59aecc650987d8022d1d31ddb2a226d176f53c70ec5608c"
ALTERNATES = {
    n: f"experiment_{n}_{s}.json"
    for n, s in [
        (9045, "fresh_corpus_a"),
        (9046, "fresh_corpus_b"),
        (9052, "arc_transfer_evidence"),
    ]
}
_plan, _scope, _receipts, _validate, _terminal, _main = (
    old.validation_plan,
    old.scope_manifest,
    old.receipts_pass,
    old.validate,
    old.terminal,
    old.main,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep actual counts visible while the reused lifecycle supervises its children."""
    print(
        f"[exp9053] phase={phase} completed={min(13, completed)} pending={min(13, pending)}",
        flush=True,
    )


@contextmanager
def binding() -> Iterator[None]:
    """Reuse qualified execution without editing historical consumers or their assertions."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        authority=authority,
        PINS=PINS,
        AUTHORITY_PINS=AUTHORITY_PINS,
        LOG_PIN=LOG_PIN,
        progress=progress,
        validation_plan=validation_plan,
        scope_manifest=scope_manifest,
        receipts_pass=receipts_pass,
        collect=collect,
        reduce=reduce,
        build=build,
        replay=replay,
        validate=validate,
        terminal=terminal,
        verify_measured=verify_measured,
        measured=measured,
    ):
        yield


def validation_plan(private: Path) -> list[CommandSpec]:
    """Retain the complete original private consumer scope and historical authority adapter."""
    with binding():
        return _plan(private)


def scope_manifest(private: Path) -> Json:
    """Freeze owned paths, original assertions and exact commands before accounting."""
    with binding():
        return _scope(private)


def receipts_pass(work: Json) -> bool:
    """Only current normal exits bound to the frozen source and log bytes qualify."""
    with binding():
        return _receipts(work)


def contract(saved: Json) -> Json:
    """Full objects and an independent digest prevent authority substitution after rehashing."""
    text = Path(saved["design_reference"]["path"]).read_text()
    queue = yaml.safe_load(Path(saved["queue_reference"]["path"]).read_bytes())
    tasks = parse_design(text, milestone=MILESTONE)[1]
    compared = compare_contract(text, queue, queue, milestone=MILESTONE, first_id=9041, count=13)
    ready = (
        compared["passed"]
        and tasks == queue["tasks"] == saved["tasks"]
        and canonical_hash(tasks) == authority.PIN
        and [saved[k]["sha256"] for k in ["design_reference", "queue_reference"]] == AUTHORITY_PINS
        and all(
            reference(Path(saved[k]["path"]))["sha256"] == saved[k]["sha256"]
            for k in ["design_reference", "queue_reference"]
        )
    )
    for row in compared["rows"]:
        row["checks"]["full_object"] = tasks == queue["tasks"] == saved["tasks"]
        row["matched"] = row["matched"] and ready
    return dict(saved, ready=bool(ready), rows=compared["rows"])


def slots(work: Json) -> list[Json]:
    """Scheduler messages account for absence and never supply a missing producer verdict."""
    result = []
    with patch.object(accounting, "MILESTONE", MILESTONE):
        for task in work["tasks"][:-1]:
            entry = work["values"].get(task["id"], {})
            original = entry.get("original", {})
            row = accounting.slot(
                task,
                entry.get("value", {}),
                entry.get("reference"),
                [
                    line
                    for line in work["scheduler_lines"]
                    if task["title"][:48] in line and "GATE_BLOCK" in line
                ],
                work["values"],
            )
            scheduler = original.get("blocked_at_layer") == "conductor_pre_gate"
            row.update(
                primary_path=task["deliverable"],
                primary_status=entry.get("primary_status", "missing_file"),
                honest_verdict=None if scheduler else original.get("honest_verdict"),
                verdict_class=None if scheduler else original.get("verdict_class"),
                scheduler_honest_verdict=original.get("honest_verdict") if scheduler else None,
            )
            row["accounted"] = row["disposition"] != "missing_primary"
            drift = [
                dict(r, observed=reference(Path(r["path"]))["sha256"])
                for r in original.get("code_config_hashes", [])
                if reference(Path(r["path"]))["sha256"] != r["sha256"]
            ]
            row["current_code_drift"] = drift
            row["current_evidence_verdict_class"] = (
                "disqualified" if drift else row["verdict_class"]
            )
            if drift:
                row.update(qualified=False, disposition="quarantined_primary")
                row["gate_check_summary"].extend(
                    dict(
                        check="producer_code_drift",
                        upstream=task["id"],
                        path=r["path"],
                        sha256=r["sha256"],
                        artifact_field="code_config_hashes.sha256",
                        op="==",
                        expected=r["sha256"],
                        observed=r["observed"],
                        passed=False,
                    )
                    for r in drift
                )
            for gate in row["gate_check_summary"]:
                gate.update(op=gate.get("operator", "=="), passed=gate.get("passed", False))
            result.append(row)
    return result


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Retain accepted and rejected input bytes before opening any reduction operands."""
    progress("preconditions_before", 0, 13)
    started = time.monotonic_ns()
    private = Path(tempfile.mkdtemp(prefix="carnot-9053-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen = time.monotonic_ns()
    preconditions = environment(raw)
    require(run_date == "20261112", "run_date")
    branch_ref = retain(ROOT / "results/experiment_9041_v772_branch_contract.json", raw, PINS[9041])
    branch = load(branch_ref)
    branch_primitive = branch["raw_shard_hashes"][0]
    branch_primitive_ref = retain(Path(branch_primitive["path"]), raw, branch_primitive["sha256"])
    refs = [branch_ref, branch_primitive_ref]
    authorities = [
        retain(Path(r["path"]), raw, r["sha256"]) for r in branch["authority_manifest"][:2]
    ]
    refs.extend(authorities)
    bound = contract(
        dict(
            tasks=branch["full_task_objects"],
            design_reference=authorities[0],
            queue_reference=authorities[1],
        )
    )
    require(bound["ready"], "full_object_contract")
    # The active queue encodes prompts; the retained original is the decoded authority.
    require(authority.authority(ROOT)["ready"], "current_full_object_contract")
    log = retain(ROOT / "ops/conductor-log.md", raw, LOG_PIN)
    refs.append(log)
    lines = (
        Path(log["path"]).read_text().split(f"Milestone {MILESTONE} activated", 1)[-1].splitlines()
    )
    values: Json = {}
    checks = list(preconditions["checks"])
    primitives: Json = {}
    for index, task in enumerate(branch["full_task_objects"][:-1]):
        n = int(task["id"][3:7])
        primary = sources / Path(task["deliverable"]).name
        selected = primary if primary.is_file() else sources / ALTERNATES.get(n, primary.name)
        if selected.is_file():
            ref = retain(selected, raw, reference(selected)["sha256"])
            original = load(ref)
            passed = ref["sha256"] == PINS.get(n)
            checks.append(
                dict(
                    check="input_bytes",
                    upstream=task["id"],
                    **ref,
                    artifact_field="sha256",
                    op="==",
                    expected=PINS.get(n),
                    observed=ref["sha256"],
                    passed=passed,
                )
            )
            values[task["id"]] = dict(
                original=original,
                value=original if passed else dict(original, flagged_adversarial=True),
                reference=ref,
                primary_status="present" if primary.is_file() else "missing_file",
            )
            refs.append(ref)
            for r in original.get("raw_shard_hashes", []):
                refs.append(retain(Path(r["path"]), raw, r["sha256"]))
            if original.get("raw_shard_hashes"):
                primitives[task["id"]] = refs[-len(original["raw_shard_hashes"])]
            for receipt in original.get("validation_receipts", []):
                if not receipt.get("passed", True) and receipt.get("log_sha256"):
                    refs.append(retain(ROOT / receipt["log_path"], raw, receipt["log_sha256"]))
        progress("producer_authenticated", index + 1, 12 - index)
    historical: Json = {}
    for task in branch["full_task_objects"]:
        for prior in task["prior_failures"]:
            identity_key = prior["experiment_id"]
            paths = list((ROOT / "results").glob(f"experiment_{identity_key[3:7]}_*.json"))
            if identity_key not in historical and len(paths) == 1:
                ref = retain(paths[0], raw, reference(paths[0])["sha256"])
                historical[identity_key] = ref
                refs.append(ref)
    health = history.health(ROOT)
    require(all(r["passed"] for r in health["checks"]), "historical_health_custody")
    for r in health["references"]:
        refs.append(retain(ROOT / r["path"], raw, r["sha256"]))
    for name in ["research-roadmap.yaml", "research-complete.yaml", "ops/exclusion_manifest.yaml"]:
        refs.append(
            retain(
                ROOT / name,
                raw,
                AUTHORITY_PINS[1]
                if name == "research-roadmap.yaml"
                else reference(ROOT / name)["sha256"],
            )
        )
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    config = manifest["coverage_config_reference"]
    code.append(dict(retain(Path(config["path"]), raw, config["sha256"]), executed_source=False))
    with binding(), old.binding(), lifecycle.binding():
        receipt = runner.run_check(
            CommandSpec(
                "publication_gate",
                (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                "historical_FoVer",
                60,
            ),
            raw / "publication",
            manifest,
        )
    require(receipt["passed"] and receipt["normal_exit"], "publication_gate_execution")
    gate = json.loads(Path(receipt["log_path"]).read_bytes())
    refs.append(reference(Path(receipt["log_path"])))
    for name in [
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "results/" + gate["gates"]["G1"]["source"],
    ]:
        refs.append(retain(ROOT / name, raw, reference(ROOT / name)["sha256"]))
    work = dict(
        raw=str(raw.absolute()),
        private=str(private),
        run_date=run_date,
        started_ns=started,
        frozen_ns=frozen,
        ended_ns=time.monotonic_ns(),
        validation_scope_manifest=manifest,
        preconditions=preconditions,
        tasks=branch["full_task_objects"],
        contract=bound,
        branch_reference=branch_ref,
        branch_primitive_reference=branch_primitive_ref,
        references=refs,
        code=code,
        values=values,
        scheduler_lines=lines,
        history=historical,
        repository_health=health,
        primitive_references=primitives,
        publication_gate=gate,
        publication_receipts=[receipt],
        checks=checks,
        validation=dict(receipts=[], coverage=dict(complete=False)),
    )
    work["slots"] = slots(work)
    atomic_json(raw / "work.json", work)
    progress("preconditions_after", 13, 0)
    return work


def reduce(work: Json) -> Json:
    """Recount retained units; absent prediction and timing primaries yield no effect estimates."""
    for task_id, primitive_ref in work["primitive_references"].items():
        require(
            primitive_ref["sha256"]
            == work["values"][task_id]["original"]["raw_shard_hashes"][0]["sha256"],
            "primitive_binding",
        )
    branch = load(work["branch_reference"])
    result: Json = dict(
        natural_feasibility_summary={}, fresh_source_summary=dict(status="missing_file")
    )
    for name, policy, index in [
        ("seven_bound_rows", "seven_bound_contract", 7),
        ("ten_bound_rows", "ten_bound_contract", 8),
    ]:
        slot = work["slots"][index]
        require(not slot["qualified"], "registered_raw_reducer_required")
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **bound,
                value=None,
                numerator=None,
                denominator=0,
                observed="missing_file" if slot["reference"] is None else "unqualified_rows",
                status="unmeasured",
                passed=False,
                upstream=slot["task_id"],
                path=slot["primary_path"],
            )
            for i, bound in enumerate(branch["frozen_science"][policy]["bounds"])
        ]
        result[policy] = branch["frozen_science"][policy]
    for index, key in [(2, "identity"), (3, "annotation")]:
        task = work["tasks"][index]
        entry = work["values"].get(task["id"])
        summary: Json = dict(status="missing_file")
        if entry and entry["reference"]["sha256"] == PINS[9041 + index]:
            source_primitive = load(work["primitive_references"][task["id"]])
            primitive = source_primitive["reduction"]
            require(
                all(
                    entry["original"][k] == primitive[k]
                    for k in ["rows", "intended_count", "completed_count"]
                ),
                "natural_primitive_rows",
            )
            rows = (
                primitive["obstruction_rows"]
                if key == "identity"
                else primitive["annotation_schema_rows"]
            )
            require(len({r["unit_id"] for r in rows}) == len(rows), "natural_unit_identity")
            summary = dict(
                status=entry["original"]["honest_verdict"],
                record_count=len(rows),
                primitive_reference=work["primitive_references"][task["id"]],
            )
            if key == "identity":
                require(
                    identity.replay(Path(entry["reference"]["path"]))["passed"],
                    "natural_primitive_replay",
                )
                summary.update(
                    category_counts=dict(Counter(r["category"] for r in rows)),
                    source_identity_ready_score=primitive["source_identity_ready_score"],
                    reopening_evidence=primitive["reopening_evidence"],
                )
            else:
                current = annotation_source.inspect(
                    Path(source_primitive["inspected_manifest"]["path"])
                )
                require(
                    [r["unit_id"] for r in current["annotation_schema_rows"]]
                    == [r["unit_id"] for r in rows],
                    "annotation_identity_drift",
                )
                summary.update(
                    reported_annotation_structure_ready_score=primitive[
                        "annotation_structure_ready_score"
                    ],
                    current_participation_authority_ready_score=current[
                        "annotation_structure_ready_score"
                    ],
                    participation_mismatch_count=sum(
                        current["participation_mismatch_counts"].values()
                    ),
                    current_code_drift=work["slots"][index]["current_code_drift"],
                    natural_pilot_ready_score=0,
                    test_outcome_values_accessed=False,
                    original_criterion=primitive["original_criterion"],
                    missing_authority=primitive["missing_authority"],
                )
        result["natural_feasibility_summary"][key] = summary
    entry = work["values"].get(work["tasks"][1]["id"])
    if entry and work["slots"][1]["qualified"]:
        rows = entry["original"]["rows"]
        primitive = load(work["primitive_references"][work["tasks"][1]["id"]])
        count = len(primitive["result"]["graph_manifest"])
        counts = Counter(r["status"] for r in rows)
        require(len({r["unit_id"] for r in rows}) == len(rows) == 1024, "fresh_unit_identity")
        require(
            counts["completed"] == count and counts["censored"] == 1024 - count,
            "fresh_primitive_counts",
        )
        result["fresh_source_summary"] = dict(
            status=entry["original"]["honest_verdict"],
            intended_count=len(rows),
            completed_count=counts["completed"],
            censored_count=counts["censored"],
            fresh_source_ready_score=0,
        )
    result["cost_quality_summary"] = dict(
        status="missing_file",
        measured_process_units=0,
        cost_ratio_upper95=None,
        demand_measured=False,
        hardware_deployment_ready=False,
        path=work["tasks"][10]["deliverable"],
    )
    arc_ref = work["branch_primitive_reference"]
    require(arc_ref["sha256"] == branch["raw_shard_hashes"][0]["sha256"], "arc_primitive_binding")
    arc = load(arc_ref)["arc"]
    support = arc_support(arc["new_event_rows"])
    require(
        support == arc["support"] == branch["arc_frontier"]["support"]
        and canonical_hash(arc["new_event_rows"]) == arc["delta_sha256"],
        "arc_primitive_support",
    )
    result["arc_summary"] = dict(
        support,
        source_event_count=len(arc["new_event_rows"]),
        source_reference=arc_ref,
        qualified_association=False,
        frontier_delta_rows=branch["frontier_delta_rows"],
        new_gameplay_executed=False,
    )
    result["archive_lag"] = branch["archive_lag"]
    return result


def build(work: Json) -> Json:
    """Scoped acceptance qualifies accounting while unavailable external science stays blocked."""
    require(
        canonical_hash(work["tasks"]) == authority.PIN
        and work["contract"]["tasks"] == work["tasks"],
        "task_authority",
    )
    science = reduce(work)
    validation = work["validation"]
    owned = (
        receipts_pass(work)
        and verify_measured(validation["coverage"], OWNED)
        and not validation.get("terminal_failed", False)
        and all(s["accounted"] for s in work["slots"])
        and all(g["passed"] for g in work["checks"])
    )
    verdict = (
        "complete_blocked_declared_gate" if owned else "complete_disqualified_owned_validation"
    )
    cls = "blocked" if owned else "disqualified"
    all_slots = [
        *work["slots"],
        dict(
            task_id=TASK,
            experiment_id=9053,
            disposition="self_slot",
            accounted=True,
            qualified=bool(owned),
            honest_verdict=verdict,
            verdict_class=cls,
        ),
    ]
    gates = [edge(g, work["values"]) for task in work["tasks"] for g in task["gated_on"]]
    retirement = []
    for task, slot in zip(work["tasks"], all_slots, strict=True):
        for prior in task["prior_failures"]:
            ref = work["history"].get(prior["experiment_id"])
            actual = load(ref).get("honest_verdict") if ref else None
            unchanged = actual is not None and actual == slot["honest_verdict"]
            retirement.append(
                dict(
                    task_id=task["id"],
                    prior_experiment_id=prior["experiment_id"],
                    planned_prior_verdict=prior["verdict"],
                    actual_prior_verdict=actual,
                    actual_current_verdict=slot["honest_verdict"],
                    unchanged_verdict=unchanged,
                    prior_reference=ref,
                    retire_if_same_verdict=prior["retire_if_same_verdict"],
                    outcome="retire_unchanged_scope"
                    if unchanged
                    else "preserve_original_and_require_changed_evidence",
                    reopening_evidence=prior["addressed_by"],
                    broader_hypothesis_retired=False,
                )
            )
    continuation = [
        dict(scope=s, status=status, next_evidence=evidence)
        for s, status, evidence in [
            (
                "old_A_B_reuse",
                "closed_unchanged",
                "Exposed A and unknown B remain ineligible; do not relabel their roles as fresh.",
            ),
            (
                "missing_QID_acquisition",
                "closed_unchanged",
                "Authenticated changed source metadata or author cross-language mappings for all 215 identities; identical retained queries cannot close missing pages or absent QIDs.",
            ),
            (
                "graph_comparison",
                "blocked_unmeasured",
                "A stated new mechanism or preregistered independent replication with a complete fresh roster and prediction-before-truth seals; no favorable-cohort hunt. If the frozen spline has no benefit, prefer converged linear and stop this unchanged comparison.",
            ),
            (
                "human_pilot",
                "not_authorized",
                "Exact original annotation authority, full source identity and exposure closure, frozen eligible cohort and original label support; diagnosis alone authorizes no pilot.",
            ),
            (
                "cost_and_hardware",
                "blocked_unmeasured",
                "Measured useful quality and workload demand, complete paired service costs including guards, durable recovery and data movement, and supported hardware operations.",
            ),
            (
                "ARC_transfer",
                "blocked_no_new_support",
                "New uncensored held-family live outcomes with matched arm support; public replay is not hidden-game generalization.",
            ),
        ]
    ]
    elapsed = (work["ended_ns"] - work["started_ns"]) / 1e9
    publication = work["publication_gate"]
    value: Json = dict(
        experiment_id=9053,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=verdict,
        verdict_class=cls,
        verifier_is_oracle=True,
        claim_scope="Thirteen administrative outcomes; no measured detector, learning, independent-label, cost or deployment benefit.",
        exposure_scope="Historical graph A exposed and B unknown; natural source identity/exposure unresolved; protected TEST outcomes withheld.",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        intended_count=13,
        completed_count=sum(s["accounted"] for s in all_slots),
        independent_count=0,
        excluded_count=0,
        censored_count=sum(not s["accounted"] for s in all_slots),
        failed_count=0,
        rows=[
            dict(
                unit_id=s["task_id"],
                group_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                metric="task_slot_accounted",
                value=int(s["accounted"]),
                numerator=int(s["accounted"]),
                denominator=1,
                status="completed" if s["accounted"] else "censored",
                exclusion_reason=None,
            )
            for s in all_slots
        ],
        reconciliation_ready_score=int(owned),
        slot_rows=all_slots,
        full_contract_rows=work["contract"]["rows"],
        gate_edge_rows=gates,
        **science,
        retirement_scope_rows=retirement,
        continuation_evidence=continuation,
        gate_check_summary=[
            *work["checks"],
            *[g for s in all_slots for g in s.get("gate_check_summary", [])],
            *gates,
        ],
        required_checks_passed=bool(owned),
        flagged_adversarial=not owned,
        validation_receipts=[
            *work["publication_receipts"],
            *validation["receipts"],
            *validation.get("terminal_receipts", []),
        ],
        validation_scope_manifest=work["validation_scope_manifest"],
        owned_statement_coverage=validation["coverage"],
        repository_health=work["repository_health"],
        repository_health_passed=False,
        preconditions_checked=work["preconditions"],
        duration_s=elapsed,
        random_seed=7729053,
        phase_spans=work.get(
            "phase_spans",
            [dict(phase="authentication_and_reconciliation", start_s=0, end_s=elapsed)],
        ),
        acceptance_gates=dict(
            owned_checks=bool(owned),
            complete_accounting=all(s["accounted"] for s in all_slots),
            independent_verifier_benefit=False,
            graph_benefit=False,
            learning_benefit=False,
            deployment_benefit=False,
        ),
        source_artifact_hashes=[
            dict(
                s["reference"],
                experiment_id=s["experiment_id"],
                fields_imported=["original_outcome", "gate_operands"],
            )
            for s in all_slots
            if s.get("reference")
        ],
        raw_shard_hashes=work["references"],
        code_config_hashes=work["code"],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        cited_upstream_artifacts=[
            dict(
                ref,
                experiment_id=int(identity_key[3:7]),
                fields_imported=["original_honest_verdict", "historical_work_by_reference_only"],
            )
            for identity_key, ref in sorted(work["history"].items())
        ],
        publication_scope="Historical FoVer only; BFS graph positives are circular_positive and cannot close the no-cheap-oracle north star or paper readiness.",
        **{k: publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=False,
        historical_FoVer_paper_ready=publication["paper_ready"],
        historical_publication_gate=publication,
        paper_readiness_scope="V772 combined claim; historical FoVer G1-G4 are preserved separately and unchanged.",
        unmet_gates=[
            "V772_independent_verifier_benefit",
            "V772_sealed_detector_and_learning_measurements",
            "V772_complete_quality_and_cost",
        ],
        activation_authorized=False,
        methodology_note="No model, generation, training or benchmark is run. Original fixed seven detector and ten learning contracts remain unmeasured without fresh sealed predictions. Recount actual retained natural structure and identity obstructions, preserve rejected primaries, failed broad health and original FoVer gates. Authenticate all thirteen full tasks and exact scheduler operands. Current scoped normal exits and zero-exclusion owned statements qualify accounting only.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Immutable operands and independent recounts bind claims; administrative completion cannot grant scientific benefit, original annotation validity or repository health.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild every field from pinned operands so rehashing cannot create a win."""
    try:
        value = json.loads(path.read_bytes())
        require(checksum(value) == value["reproducibility_checksum"], "checksum_drift")
        work = load(value["work_reference"])
        for ref in work["references"] + work["code"]:
            require(reference(Path(ref["path"]))["sha256"] == ref["sha256"], "operand_drift")
        for ref in work["code"]:
            if ref.get("executed_source", True):
                require(
                    reference(Path(ref["original_path"]))["sha256"] == ref["sha256"],
                    "executed_code_drift",
                )
        require(work["branch_reference"]["sha256"] == PINS[9041], "branch_pin")
        require(work["contract"] == contract(work["contract"]), "authority_drift")
        log = next(
            r
            for r in work["references"]
            if r.get("original_path") == str(ROOT / "ops/conductor-log.md")
        )
        require(
            log["sha256"] == LOG_PIN
            and work["scheduler_lines"]
            == Path(log["path"])
            .read_text()
            .split(f"Milestone {MILESTONE} activated", 1)[-1]
            .splitlines(),
            "scheduler_drift",
        )
        require(
            work["validation_scope_manifest"] == scope_manifest(Path(work["private"])),
            "scope_drift",
        )
        for task in work["tasks"][:-1]:
            entry = work["values"].get(task["id"])
            if entry:
                original = load(entry["reference"])
                require(
                    entry["reference"]["sha256"] == PINS.get(int(task["id"][3:7]))
                    and entry["original"] == entry["value"] == original,
                    "producer_drift",
                )
        require(work["slots"] == slots(work), "slot_drift")
        require(value == build(work), "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=13)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Run the frozen scope once and reopen the actual SQLite statement measurements."""
    with binding():
        return _validate(work)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged scanners and the real foreign-directory replay qualify publication bytes."""
    with binding():
        return _terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Use the qualified lifecycle, deadlines and atomic publication unchanged."""
    with binding():
        return _main(argv)
