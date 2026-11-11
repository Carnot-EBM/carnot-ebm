"""REQ-REPORT-9026-SCOPED: complete accounting cannot promote unavailable science."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from collections.abc import Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v767_capstone as accounting
from carnot.reporting import v770_branch_contract as authority
from carnot.reporting import v770_branch_evidence as history
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v768_capstone import edge

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9026_v770_capstone", "exp9026-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v770_capstone.py", CLI]
TEST = "tests/python/test_v770_capstone_9026.py"
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7837_v681_contract_methods.py",
    "tests/python/test_v769_capstone_9012.py",
    "tests/python/test_v770_branch_contract_9013.py",
]
REUSED = [
    "python/carnot/reporting/" + p + ".py"
    for p in [
        "scoped_provenance_9014",
        "invocation_provenance_9000",
        "v767_capstone",
        "v768_capstone",
        "v770_branch_contract",
        "v770_branch_evidence",
        "v769_branch_evidence",
        "current_work_receipt",
        "primary_publication",
        "roadmap_contract",
        "v746_retained_evidence",
        "v756_branch_qualification",
        "v763_capstone_reduction",
        "v764_capstone_sources",
        "v709_execution",
        "experiment_7303_validation_scope",
    ]
] + [
    "scripts/" + p + ".py"
    for p in ["publication_gate", "adversarial_verify", "verdict_row_consistency_lint"]
]
CHECKS = runner.CHECKS
MODEL_SPECS: list[Json] = []
LOG_PIN = "sha256:220a88ebf3a770ec81600ebe6b71e8b7085a7bd570d56d279208b32a37f2fac3"
AUTHORITY_PINS = [
    "sha256:7f07a9250aa8cba0ad21ce9164f87fc380c069dc7e1c26430534cfe8cb28ecfb",
    "sha256:4b6641d72a5a0878d5d3098e61d715385f23413a5eb58ac0c9879401fe9dab53",
]
PINS = {
    9013: "sha256:dc94513d17071575dcb0d2f93610b7978bda98b361b00e82764a49174ed6dbe8",
    9014: "sha256:ac6037edc4bcb9fad3d7d62610153962090da35bd1b0cc597c68e7b3bd41dcdb",
    9015: "sha256:94fefec6e66dbbca4974fbfdab9f5c470bfc515a62d1d0314f1793169872d96a",
    9016: "sha256:9f2261110a2b0d5ccd6057493eebe327ecaa2fdda7457e81c432989cf54dcdcf",
    9017: "sha256:d8a527874de2bf0ccbff209ac9114cc592713c15ca9f5a761482e5165fa46650",
    9019: "sha256:5debe8677a611b99b366266b2a445cf17b90e61ad6c9bf6d5bfdf09cbfc53b4c",
    9025: "sha256:09602df2f9b520dfc91ed81aed434a5f85677a2ede63794c4cf4cca8d2e58b24",
}
ALTERNATES = {
    n: f"experiment_{n}_{s}.json"
    for n, s in [
        (9016, "retained_shard_a"),
        (9017, "graph_corpus_b"),
        (9019, "human_pilot_methods"),
        (9025, "arc_transfer_evidence"),
    ]
}
checksum = accounting.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual flushed counts distinguish a bounded wait from unfinished work."""
    print(f"[exp9026] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Use the existing scoped runner without changing historical producer acceptance."""
    with patch.multiple(
        runner,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        progress=progress,
    ):
        yield


def scope_manifest(private: Path) -> Json:
    """Freeze exact current commands and sources before opening measurement inputs."""
    with binding():
        return dict(
            runner.scope_manifest(private),
            coverage_config_reference=reference(private / "coverage.ini"),
            scientific_contract_sha256=authority.PIN,
            full_suite_policy="Historical failed health stays separate from current scoped acceptance.",
        )


def receipts_pass(work: Json) -> bool:
    """An optimistic scalar cannot replace normal exits, exact commands and log bytes."""
    with binding():
        return runner.receipts_pass(work)


def contract(saved: Json) -> Json:
    """Compare complete retained task objects as well as the visible design table."""
    text = Path(saved["design_reference"]["path"]).read_text()
    queue = yaml.safe_load(Path(saved["queue_reference"]["path"]).read_bytes())
    tasks = parse_design(text, milestone=MILESTONE)[1]
    compared = compare_contract(text, queue, queue, milestone=MILESTONE, first_id=9013)
    ready = (
        compared["passed"]
        and [saved[k]["sha256"] for k in ["design_reference", "queue_reference"]] == AUTHORITY_PINS
        and all(
            reference(Path(saved[k]["path"]))["sha256"] == saved[k]["sha256"]
            for k in ["design_reference", "queue_reference"]
        )
        and tasks == queue["tasks"] == saved["tasks"]
        and canonical_hash(tasks) == authority.PIN
    )
    return dict(saved, ready=bool(ready), rows=compared["rows"])


def slots(work: Json) -> list[Json]:
    """Scheduler events account for absence but cannot supply a producer's verdict."""
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
            row.update(
                primary_path=task["deliverable"],
                primary_status=entry.get("primary_status", "missing_file"),
                honest_verdict=original.get("honest_verdict"),
                verdict_class=original.get("verdict_class"),
                planned_status="planned",
                implemented_status=row["disposition"],
            )
            for gate in row["gate_check_summary"]:
                gate.update(op=gate.get("operator", "=="), passed=gate.get("passed", False))
            result.append(row)
    return result


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Authenticate and retain rejected evidence before any scientific reduction."""
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic_ns()
    private = Path(tempfile.mkdtemp(prefix="carnot-9026-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen = time.monotonic_ns()
    preconditions = environment(raw)
    require(run_date == "20261111", "run_date")
    branch_path = ROOT / "results/experiment_9013_v770_branch_contract.json"
    branch_ref = retain(branch_path, raw, PINS[9013])
    branch = load(branch_ref)
    refs = [branch_ref]
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
    log_path = ROOT / "ops/conductor-log.md"
    log = retain(log_path, raw, LOG_PIN)
    refs.append(log)
    lines = (
        Path(log["path"]).read_text().split(f"Milestone {MILESTONE} activated", 1)[-1].splitlines()
    )
    values: Json = {}
    checks = list(preconditions["checks"])
    for index, task in enumerate(branch["full_task_objects"][:-1]):
        n = 9013 + index
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
        progress("producer_authenticated", index + 1, 13 - index)
    historical: Json = {}
    for task in branch["full_task_objects"]:
        for prior in task["prior_failures"]:
            identity = prior["experiment_id"]
            paths = list((ROOT / "results").glob(f"experiment_{identity[3:7]}_*.json"))
            if identity not in historical and len(paths) == 1:
                ref = retain(
                    paths[0],
                    raw,
                    history.PINS.get(str(paths[0].relative_to(ROOT)))
                    or reference(paths[0])["sha256"],
                )
                historical[identity] = ref
                refs.append(ref)
    health = history.health(ROOT)
    require(all(r["passed"] for r in health["checks"]), "historical_health_custody")
    for r in health["references"]:
        path = ROOT / r["path"]
        refs.append(retain(path, raw, r["sha256"]))
    observation = (
        ROOT / "results/raw" / NAME / "repository_health/full_python_suite_once.receipt.json"
    )
    health["current_observations"] = []
    if observation.is_file():
        ref = retain(observation, raw, reference(observation)["sha256"])
        health["current_observations"].append(ref)
        refs.append(ref)
        receipt_value = load(ref)
        for kind in ["stdout", "stderr"]:
            refs.append(
                retain(ROOT / receipt_value[kind + "_path"], raw, receipt_value[kind + "_sha256"])
            )
    for name in ["research-complete.yaml", "ops/exclusion_manifest.yaml"]:
        path = ROOT / name
        refs.append(retain(path, raw, reference(path)["sha256"]))
    human = values.get("exp9015-human-disagreement-source", {}).get("original", {})
    primitive = None
    if human:
        r = human["primitive_reference"]
        primitive = retain(Path(r["path"]), raw, r["sha256"])
        refs.append(primitive)
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    config = manifest["coverage_config_reference"]
    code.append(dict(retain(Path(config["path"]), raw, config["sha256"]), executed_source=False))
    with binding():
        receipt = runner.run_check(
            CommandSpec(
                "publication_gate",
                (sys.executable, "-u", "scripts/publication_gate.py", "--json"),
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
        path = ROOT / name
        refs.append(retain(path, raw, reference(path)["sha256"]))
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
        references=refs,
        code=code,
        values=values,
        scheduler_lines=lines,
        history=historical,
        repository_health=health,
        human_primitive=primitive,
        publication_gate=gate,
        publication_receipts=[receipt],
        checks=checks,
        validation=dict(receipts=[], coverage=dict(complete=False)),
    )
    work["slots"] = slots(work)
    atomic_json(raw / "work.json", work)
    progress("preconditions_after", 14, 0)
    return work


def reduce(work: Json) -> Json:
    """Recompute support by independent unit; missing scientific rows are not null effects."""
    branch = load(work["branch_reference"])
    frozen = branch["frozen_science"]
    human_contract = dict(
        draws=10000,
        seed=770019,
        family_alpha=0.05,
        tail=0.05 / 4,
        unit="paired TEST source group",
        intended_test_groups=200,
        bounds=[
            dict(metric="brier_delta", control=a, criterion="upper<0")
            for a in ["matched_linear", "full_embedding_linear", "prevalence"]
        ]
        + [
            dict(
                metric="hard_error_delta",
                control="CAL_selected_best_linear",
                criterion="upper<=0.02",
            )
        ],
    )
    result: Json = dict(
        scientific_benefit=False,
        administrative_readiness=branch["contract_ready_score"],
        frozen_graph_method_readiness=branch["frozen_graph_authority_ready_score"],
        online_method_readiness=branch["online_protocol_ready_score"],
    )
    for name, index, policy in [
        ("graph_detector_bounds", 7, frozen["seven_bound_contract"]),
        ("causal_learning_bounds", 9, frozen["ten_bound_contract"]),
        ("human_disagreement_bounds", 10, human_contract),
    ]:
        slot = work["slots"][index]
        require(not slot["qualified"], "registered_raw_reducer_required")
        result[name + "_contract"] = policy
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **bound,
                numerator=None,
                denominator=0,
                observed="missing_file" if slot["reference"] is None else "unqualified_rows",
                status="unmeasured",
                passed=False,
                upstream=slot["task_id"],
                path=slot["primary_path"],
            )
            for i, bound in enumerate(policy["bounds"])
        ]
    human = work["values"].get("exp9015-human-disagreement-source", {}).get("original", {})
    result["human_source_audit"] = dict(status="missing_file", source_group_count=None)
    if human:
        primitive = load(work["human_primitive"])["reduction"]
        require(
            all(
                primitive[k] == human[k]
                for k in [
                    "rows",
                    "source_group_rows",
                    "role_manifest",
                    "raw_annotator_rows",
                    "annotation_disagreement_rows",
                ]
            ),
            "human_primitive_rows",
        )
        groups = primitive["source_group_rows"]
        require(len({g["group_id"] for g in groups}) == len(groups), "human_source_group_integrity")
        result["human_source_audit"] = dict(
            status=human["honest_verdict"],
            source_group_count=len(groups),
            eligible_source_group_count=sum(g["exclusion_reason"] is None for g in groups),
            unresolved_source_group_count=len(primitive["unresolved_source_groups"]),
            selected_role_count=len(primitive["role_manifest"]),
            intended_role_count=400,
            original_annotation_count=len(primitive["raw_annotator_rows"]),
            disagreement_row_count=len(primitive["annotation_disagreement_rows"]),
            independent_label_ready_score=human["independent_label_ready_score"],
            natural_pilot_ready_score=human["natural_pilot_ready_score"],
            criterion_policy=human["criterion_policy"],
            primitive_reference=work["human_primitive"],
            limit="Source overlap must resolve before roles or labels qualify. Human judgments are fallible; public pretrained-model contamination is unknown.",
        )
    original = load(work["history"]["exp8990-graph-corpus-a"])
    require(
        original["honest_verdict"] == "complete_disqualified_graph_corpus_a"
        and original["flagged_adversarial"] is True,
        "original_disqualification",
    )
    ledger = original["invocation_ledger"]
    calls = [r for r in ledger if r["operation"] == "generation" and r["status"] == "completed"]
    require(
        len({r["call_id"] for r in ledger}) == len(ledger)
        and len(calls) == original["model_invocation_counts"]["generation_calls_completed"] == 512,
        "historical_call_denominator",
    )
    result.update(
        historical_generation_calls_counted_once=len(calls),
        historical_generation_verdict=original["honest_verdict"],
        historical_generation_reference=work["history"]["exp8990-graph-corpus-a"],
        complete_costs=[
            dict(
                arm=a,
                intended_paired_process_units=24,
                independent_units=0,
                complete_process_s=None,
                cost_ratio_upper95=None,
                status="missing_file",
            )
            for a in ["frozen_spline", "adaptive_spline", "frozen_linear", "adaptive_linear"]
        ],
        hardware=dict(
            status="unmeasured", deployment_ready=False, maximum_serial_fraction_for_100x=0.01
        ),
        arc=dict(
            optional=True,
            ready=bool(branch["arc_new_outcomes_ready_score"]),
            support=branch["arc_frontier"]["support"],
            frontier_delta_rows=branch["frontier_delta_rows"],
            qualified_association=False,
            claim_scope="Observational held-family support only; no causal transfer or hidden-game generalization.",
        ),
        graph_claim="BFS is a cheap oracle. A graph benefit would be circular_positive and cannot close the no-cheap-oracle target.",
        human_claim="No independent-label benefit measured. Source-group integrity, original annotator authority and unknown public pretraining exposure remain mandatory.",
    )
    return result


def build(work: Json) -> Json:
    """Passing current checks qualify accounting while external unavailability stays blocked."""
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
    )
    cls = "blocked" if owned else "disqualified"
    all_slots = [
        *work["slots"],
        dict(
            task_id=TASK,
            experiment_id=9026,
            disposition="self_slot",
            accounted=True,
            qualified=bool(owned),
            honest_verdict=f"complete_{cls}_v770_capstone",
            verdict_class=cls,
            planned_status="planned",
            implemented_status="completed_owned_accounting",
        ),
    ]
    gates = [edge(g, work["values"]) for task in work["tasks"] for g in task["gated_on"]]
    checks = [
        *work["checks"],
        *[g for s in all_slots for g in s.get("gate_check_summary", [])],
        *gates,
    ]
    retirement = []
    for task, slot in zip(work["tasks"], all_slots, strict=True):
        for item in task["prior_failures"]:
            ref = work["history"].get(item["experiment_id"])
            actual = load(ref).get("honest_verdict") if ref else None
            retirement.append(
                dict(
                    task_id=task["id"],
                    prior_experiment_id=item["experiment_id"],
                    planned_prior_verdict=item["verdict"],
                    actual_prior_verdict=actual,
                    actual_current_verdict=slot["honest_verdict"],
                    unchanged_verdict=actual is not None and actual == slot["honest_verdict"],
                    prior_reference=ref,
                    scope_unchanged=True,
                    broader_hypothesis_retired=False,
                    reopening_evidence=item["addressed_by"],
                    action="preserve_original_scope_and_require_changed_evidence",
                )
            )
    failed = next((g for g in gates if not g["passed"]), dict(check="upstream_science_unavailable"))
    elapsed = (work["ended_ns"] - work["started_ns"]) / 1e9
    gate = work["publication_gate"]
    value: Json = dict(
        experiment_id=9026,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict="complete_blocked_" + failed["check"]
        if owned
        else "complete_disqualified_owned_validation",
        verdict_class=cls,
        verifier_is_oracle=True,
        claim_scope="Fourteen administrative outcomes; graph, independent human annotation, causal learning and deployment claims are separate and unmeasured.",
        exposure_scope="Exposed graph development; rejected historical responses; unresolved public human-source groups and unknown pretraining contamination.",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        intended_count=14,
        completed_count=14,
        independent_count=0,
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        rows=[
            dict(
                unit_id=s["task_id"],
                group_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                metric="task_slot_accounted",
                value=1,
                numerator=1,
                denominator=1,
                status="completed",
                exclusion_reason=None,
            )
            for s in all_slots
        ],
        reconciliation_ready_score=int(owned),
        slot_rows=all_slots,
        gate_edge_rows=gates,
        full_contract_rows=work["contract"]["rows"],
        v770_science_summary=science,
        retirement_scope_rows=retirement,
        gate_check_summary=checks,
        continuation_evidence=[
            dict(
                upstream=g["upstream"],
                failed_field=g["artifact_field"],
                observed=g["observed"],
                next_evidence="Qualify this exact gate with new normal-exit evidence; preserve original rejections, thresholds and unseen-source and annotation requirements.",
            )
            for g in gates
            if not g["passed"]
        ],
        required_checks_passed=bool(owned),
        flagged_adversarial=not owned,
        validation_receipts=[
            *work["publication_receipts"],
            *validation["receipts"],
            *validation.get("terminal_receipts", []),
        ],
        owned_statement_coverage=validation["coverage"],
        validation_scope_manifest=work["validation_scope_manifest"],
        preconditions_checked=work["preconditions"],
        repository_health=work["repository_health"],
        repository_health_passed=False,
        duration_s=elapsed,
        random_seed=7709026,
        phase_spans=work.get(
            "phase_spans",
            [dict(phase="authentication_and_reconciliation", start_s=0, end_s=elapsed)],
        ),
        acceptance_gates=dict(
            owned_checks=bool(owned),
            complete_accounting=True,
            graph_benefit=False,
            human_label_benefit=False,
            causal_learning_benefit=False,
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
                experiment_id=int(identity[3:7]),
                fields_imported=["original_honest_verdict", "historical_model_work_only"],
            )
            for identity, ref in sorted(work["history"].items())
        ],
        publication_scope="Historical FoVer G1-G4 only; no V770 science certification or external publication.",
        **{k: gate["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=gate["paper_ready"],
        unmet_gates=gate["unmet_gates"],
        activation_authorized=False,
        methodology_note="Retain original bytes and independently count each scientific unit. Frozen seven graph, ten learning and four human bounds remain unmeasured. Only current scoped normal exits and measured new statements qualify accounting. Historical failed health and model work remain separate. No model, generation, training, benchmark or board call occurs.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Retained primitives and independent denominators bind each claim. Accounting and historical publication readiness grant no new scientific benefit or repository health.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from authenticated operands so rehashing an aggregate cannot create readiness."""
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
        require(work["branch_reference"]["sha256"] == PINS[9013], "branch_pin")
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
        for n, task in enumerate(work["tasks"][:-1], 9013):
            entry = work["values"].get(task["id"], {})
            if entry:
                original = load(entry["reference"])
                expected = (
                    original
                    if entry["reference"]["sha256"] == PINS.get(n)
                    else dict(original, flagged_adversarial=True)
                )
                require(
                    entry["original"] == original and entry["value"] == expected, "producer_drift"
                )
        require(work["slots"] == slots(work), "slot_drift")
        original_health = history.health(ROOT)
        require(
            all(work["repository_health"][k] == v for k, v in original_health.items()),
            "historical_health_drift",
        )
        require(value == build(work), "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Run the frozen scoped commands once and reopen their actual statement data."""
    private, raw = Path(work["private"]), Path(work["raw"])
    manifest = work["validation_scope_manifest"]
    with binding(), patch.dict(os.environ, COVERAGE_RCFILE=str(private / "coverage.ini")):
        plan = runner.validation_plan(private)
        require(
            [dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in plan]
            == manifest["commands"],
            "frozen_commands",
        )
        receipts = []
        for i, spec in enumerate(plan):
            receipts.append(runner.run_check(spec, raw / "validation", manifest))
            progress("current_validation", i + 1, len(plan) - i - 1)
    coverage = (
        measured([private / ".coverage"], OWNED, raw / "coverage")
        if (private / ".coverage").is_file()
        else dict(complete=False, files=[])
    )
    coverage.pop("rows", None)
    return dict(receipts=receipts, coverage=coverage)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged scanners and foreign-directory replay check the exact candidate bytes."""
    with binding():
        return runner.terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Expose a primary only after owned checks pass; retain every rejected candidate."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261111"], default="20261111")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--check-receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.from_work and args.output.absolute().is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    try:
        if args.cold_replay or args.check_receipts:
            report = (
                replay(args.cold_replay)
                if args.cold_replay
                else dict(passed=receipts_pass(json.loads(args.check_receipts.read_bytes())))
            )
            print(json.dumps(report), flush=True)
            return 0 if report["passed"] else 1
        raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
        if args.from_work:
            work = json.loads(args.from_work.read_bytes())
            work["raw"] = str(raw)
        else:
            work = collect(raw, args.date, args.sources)
            progress("reductions_before", 0, 14)
            reduce(work)
            progress("reductions_after", 14, 0)
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3300, "measurement_deadline")
            work["validation"] = validate(work)
        work["ended_ns"] = time.monotonic_ns()
        work["phase_spans"] = [
            dict(phase="measurement", duration_s=(work["frozen_ns"] - work["started_ns"]) / 1e9),
            dict(
                phase="reconciliation_and_validation",
                duration_s=(work["ended_ns"] - work["frozen_ns"]) / 1e9,
            ),
        ]
        atomic_json(raw / "work.json", work)
        candidate = raw / "candidate.json"
        value = build(work)
        atomic_json(candidate, value)
        require(value["required_checks_passed"], "owned_validation_missing_or_failed")
        report = terminal(candidate, raw / "preflight_checks", work["validation_scope_manifest"])
        work["validation"].update(
            terminal_failed=not report["passed"], terminal_receipts=report["receipts"]
        )
        atomic_json(raw / "work.json", work)
        value = build(work)
        atomic_json(candidate, value)
        require(report["passed"], "owned_terminal_checks")
        require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4800, "hard_deadline")
        published = publish_primary(
            args.output,
            value,
            lambda p: terminal(p, raw / "publication_checks", work["validation_scope_manifest"]),
        )
        progress("complete", 14, 0)
        print(
            json.dumps(
                dict(
                    published,
                    honest_verdict=value["honest_verdict"],
                    reconciliation_ready_score=value["reconciliation_ready_score"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
