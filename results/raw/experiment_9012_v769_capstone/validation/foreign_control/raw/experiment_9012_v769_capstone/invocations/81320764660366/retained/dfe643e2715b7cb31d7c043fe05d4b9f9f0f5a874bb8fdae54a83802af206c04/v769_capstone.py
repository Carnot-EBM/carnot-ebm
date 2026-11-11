"""REQ-REPORT-9012: account for scheduled work without promoting missing science.

The source branches have terminal qualification failures. This reduction preserves
their actual bytes and the frozen scientific questions; it does not rerun them.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v767_capstone as prior
from carnot.reporting import v768_capstone as previous
from carnot.reporting import v768_branch_contract as environment
from carnot.reporting import v769_branch_contract as authority
from carnot.reporting import v769_branch_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v756_branch_qualification import verify_measured
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9012_v769_capstone", "exp9012-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v769_capstone.py", CLI]
TEST = "tests/python/test_v769_capstone_9012.py"
CHECKS = {
    "owned_unit_child",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018_021",
    "coverage_report",
}
PINS = {
    8999: "sha256:6e60393a99efb680765b8d1f6cd5993efdf4717ec0a091da6624a0cf4d9997d7",
    9000: "sha256:e92dddafa07eb2b28911b7ece011db3b58607f7e4050e5dd3ee26a2ab2558679",
    9001: "sha256:9a6d75d73f3bcce3d2f57da61bd679d5f49a70785648a434f418b1fd4ae0256f",
    9002: "sha256:c05a8dcec3fc405284b2ff15dcc069d36fd92d08ed6c1c8ffb2bf660ddf84a31",
    9003: "sha256:85c25c0038c5a621a41eb27b95c2e1c2b9c85975d283e95f081f5685a9fed531",
    9005: "sha256:8b82afb294a9826cbd75381a871065cb83ebfe7f4613163c2fb502079f104577",
    9011: "sha256:d8258dc0de24c197108945e0a915385b6109ec4c1de575a25797fb4806dfe594",
}
ALTERNATES = {
    n: f"experiment_{n}_{s}.json"
    for n, s in [
        (9002, "retained_shard_a"),
        (9003, "graph_corpus_b"),
        (9005, "natural_pilot_methods"),
        (9011, "arc_transfer_evidence"),
    ]
}
checksum, edge = prior.checksum, previous.edge


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts because an idle child must not look like completed work."""
    print(f"[exp9012] phase={phase} completed={completed} pending={pending}", flush=True)


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Retain original and rejected bytes before reducing their independent scopes."""
    progress("preconditions_before", 0, 14)
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = environment.resources(raw, run_date)
    branch_ref = retain(ROOT / "results/experiment_8999_v769_branch_contract.json", raw, PINS[8999])
    branch = load(branch_ref)
    tasks = branch["full_task_objects"]
    require(canonical_hash(tasks) == authority.PIN, "task_authority")
    refs = [branch_ref]
    for r in branch["authority_manifest"][:2]:
        saved = retain(Path(r["path"]), raw, r["sha256"])
        target = raw / "authority" / Path(r["source"]["path"]).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(saved["path"]).read_bytes())
        refs.extend([saved, reference(target)])
    contract = authority.authority(raw / "authority")
    require(contract["ready"] and contract["tasks"] == tasks, "full_object_contract")
    log = retain(ROOT / "ops/conductor-log.md", raw, sha256_file(ROOT / "ops/conductor-log.md"))
    refs.append(log)
    for name in ["research-complete.yaml", "ops/exclusion_manifest.yaml", *evidence.PINS]:
        refs.append(retain(ROOT / name, raw, evidence.PINS.get(name, sha256_file(ROOT / name))))
    lines = Path(log["path"]).read_text().splitlines()
    values: Json = {}
    failures = environment.resource_failures(preconditions, raw)
    for index, task in enumerate(tasks[:-1]):
        n = 8999 + index
        primary = sources / Path(task["deliverable"]).name
        selected = primary if primary.is_file() else sources / ALTERNATES.get(n, primary.name)
        if selected.is_file():
            ref = retain(selected, raw, sha256_file(selected))
            original = load(ref)
            passed = ref["sha256"] == PINS.get(n)
            failures.append(
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
            value = original if passed else dict(original, flagged_adversarial=True)
            values[task["id"]] = dict(value=value, reference=ref, original=original)
            refs.append(ref)
        progress("producer_authenticated", index + 1, 13 - index)
    with patch.object(prior, "MILESTONE", MILESTONE):
        slots = [
            prior.slot(
                t,
                values.get(t["id"], {}).get("value", {}),
                values.get(t["id"], {}).get("reference"),
                [line for line in lines if t["title"][:48] in line and "GATE_BLOCK" in line],
                values,
            )
            for t in tasks[:-1]
        ]
    for task, s in zip(tasks, slots):
        entry = values.get(task["id"], {})
        s.update(
            primary_status="present"
            if (sources / Path(task["deliverable"]).name).is_file()
            else "missing_file",
            primary_path=task["deliverable"],
            planned_status="planned",
            implemented_status=s["disposition"],
            honest_verdict=entry.get("original", {}).get("honest_verdict"),
            verdict_class=entry.get("original", {}).get("verdict_class"),
        )
        for g in s["gate_check_summary"]:
            g.update(op=g.get("operator", "=="), passed=g.get("passed", False))
    history: Json = {}
    for task in tasks:
        for item in task["prior_failures"]:
            identity = item["experiment_id"]
            paths = list((ROOT / "results").glob(f"experiment_{identity[3:7]}_*.json"))
            if identity not in history and len(paths) == 1:
                ref = retain(paths[0], raw, sha256_file(paths[0]))
                history[identity] = ref
                refs.append(ref)
    natural = values.get("exp9001-human-source-authority", {}).get("original", {})
    primitive = None
    if natural:
        r = natural["primitive_reference"]
        primitive = retain(Path(r["path"]), raw, r["sha256"])
        refs.append(primitive)
    progress("publication_gate_before", 0, 1)
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                "publication_gate",
                (sys.executable, "-u", "scripts/publication_gate.py", "--json"),
                "historical_FoVer",
                60,
            )
        ],
        log_dir=raw / "publication",
        heartbeat_s=15,
    )
    require(receipts[0]["passed"], "publication_gate_execution")
    gate = json.loads((ROOT / receipts[0]["log_path"]).read_bytes())
    refs.append(reference(ROOT / receipts[0]["log_path"]))
    code = [
        retain(ROOT / p, raw, sha256_file(ROOT / p))
        for p in [
            *OWNED,
            TEST,
            "python/carnot/reporting/v769_branch_contract.py",
            "python/carnot/reporting/v768_capstone.py",
            "python/carnot/reporting/v767_capstone.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/roadmap_contract.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    ]
    progress("authentication_after", 14, 0)
    return dict(
        tasks=tasks,
        slots=slots,
        values=values,
        references=refs,
        code=code,
        contract=contract,
        branch_reference=branch_ref,
        history=history,
        natural_primitive=primitive,
        publication_gate=gate,
        publication_receipts=receipts,
        preconditions=preconditions,
        failures=failures,
        raw=str(raw.absolute()),
        run_date=run_date,
        started_ns=start,
        ended_ns=time.monotonic_ns(),
    )


def reduce(work: Json) -> Json:
    """Reduce each scientific denominator separately; absent observations are not zero effects."""
    branch = load(work["branch_reference"])
    contracts = branch["frozen_science"]
    result: Json = dict(
        scientific_benefit=False,
        administrative_readiness=branch["contract_ready_score"],
        frozen_graph_method_readiness=branch["frozen_graph_authority_ready_score"],
        online_method_readiness=branch["online_protocol_ready_score"],
    )
    natural_contract = dict(
        bootstrap_draws=10000,
        seed=769008,
        family_alpha=0.05,
        tail=0.05 / 4,
        bootstrap_unit="paired source group",
        bounds=[
            dict(metric="brier_improvement", control=a, criterion="lower>0")
            for a in ["shared_linear", "lexical", "prevalence"]
        ]
        + [dict(metric="accepted_error_delta_50", control="shared_linear", criterion="upper<=0")],
    )
    for name, number, contract in [
        ("graph_detector_bounds", 9006, contracts["seven_bound_contract"]),
        ("causal_learning_bounds", 9009, contracts["ten_bound_contract"]),
        ("natural_label_bounds", 9008, natural_contract),
    ]:
        slot = work["slots"][number - 8999]
        require(not slot["qualified"], "new_science_requires_registered_raw_reducer")
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **b,
                observed="missing_file" if slot["reference"] is None else "unqualified_rows",
                status="unmeasured",
                passed=False,
                numerator=None,
                denominator=0,
                upstream=slot["task_id"],
                path=slot["primary_path"],
            )
            for i, b in enumerate(contract["bounds"])
        ]
        result[name + "_contract"] = contract
    natural = work["values"].get("exp9001-human-source-authority", {}).get("original", {})
    result["natural_source_audit"] = dict(status="missing_file", unseen_source_group_count=None)
    if natural:
        raw = load(work["natural_primitive"])["reduction"]
        require(
            raw["rows"] == natural["rows"]
            and raw["source_group_rows"] == natural["source_group_rows"],
            "natural_primitive_rows",
        )
        groups = raw["source_group_rows"]
        require(len({g["group_id"] for g in groups}) == len(groups), "source_group_integrity")
        unseen = sum(not g["exposed_source_ids"] for g in groups)
        result["natural_source_audit"] = dict(
            status="retained_rejected_audit",
            qualified=False,
            source_group_count=len(groups),
            unseen_source_group_count=unseen,
            exposed_source_group_count=len(groups) - unseen,
            human_authority_group_count=raw["human_authority_group_count"],
            original_annotation_origin_rows=raw["annotation_origin_rows"],
            source_group_rows=groups,
            intended_pilot_groups=600,
            observed_pilot_groups=len(raw["role_manifest"]),
            independent_label_ready_score=natural["independent_label_ready_score"],
            natural_pilot_ready_score=natural["natural_pilot_ready_score"],
            criterion_policy=natural["criterion_policy"],
            pretrained_model_contamination="Public release may occur in pretraining; no contamination-free or unseen-generator claim.",
            authority_limit="Manual source-support judgments are fallible; annotator independence and independent adjudication are not recorded.",
            reference=work["natural_primitive"],
        )
    historical = load(work["history"]["exp8990-graph-corpus-a"])
    require(
        historical["honest_verdict"] == "complete_disqualified_graph_corpus_a"
        and historical["flagged_adversarial"] is True,
        "original_disqualification",
    )
    ledger = historical["invocation_ledger"]
    calls = [r for r in ledger if r["operation"] == "generation" and r["status"] == "completed"]
    require(
        len({r["call_id"] for r in ledger}) == len(ledger)
        and len(calls)
        == historical["model_invocation_counts"]["generation_calls_completed"]
        == 512,
        "historical_call_denominator",
    )
    result["historical_generation_calls_counted_once"] = len(calls)
    result["historical_invocation_rows"] = ledger
    result["historical_generation_reference"] = work["history"]["exp8990-graph-corpus-a"]
    result["complete_costs"] = [
        dict(
            arm=a,
            intended_paired_process_units=24,
            independent_units=0,
            complete_process_s=None,
            cost_ratio_upper95=None,
            status="missing_file",
        )
        for a in ["frozen_spline", "adaptive_spline", "frozen_linear", "adaptive_linear"]
    ]
    result["hardware"] = dict(
        status="unmeasured",
        deployment_ready=False,
        maximum_serial_fraction_for_100x=0.01,
        claim_scope="No board calls or measured accelerator speedup; quality and complete cost gates remain open.",
    )
    result["arc"] = dict(
        optional=True,
        ready=bool(branch["arc_new_outcomes_ready_score"]),
        support=branch["arc_frontier"]["support"],
        frontier_delta_rows=branch["frontier_delta_rows"],
        qualified_association=False,
        claim_scope="Observational held-family support only; no causal policy or hidden-game generalization.",
    )
    result["graph_claim"] = (
        "BFS oracle-based; any future graph positive is circular_positive and cannot close the no-cheap-oracle target."
    )
    result["natural_claim"] = (
        "No held-out human-label benefit measured; source custody, exposure, annotation authority and pretrained-model contamination limits remain mandatory."
    )
    return result


def build(work: Json, validation: Json) -> Json:
    """Readiness certifies complete reconciliation, while unavailable science remains blocked."""
    require(
        canonical_hash(work["tasks"]) == authority.PIN
        and work["contract"]["tasks"] == work["tasks"],
        "task_authority",
    )
    require(work["branch_reference"]["sha256"] == PINS[8999], "branch_authority_pin")
    science = reduce(work)
    owned = bool(validation["passed"] and validation["coverage"].get("complete") is True)
    verdict = "blocked" if owned else "disqualified"
    slots = [
        *work["slots"],
        dict(
            task_id=TASK,
            experiment_id=9012,
            disposition="self_slot",
            accounted=True,
            qualified=owned,
            verdict_class=verdict,
            honest_verdict=f"complete_{verdict}_v769_capstone",
            planned_status="planned",
            implemented_status="completed_owned_accounting",
        ),
    ]
    failures = [
        r
        for r in [*work["failures"], *[g for s in slots for g in s.get("gate_check_summary", [])]]
        if not r["passed"]
    ]
    gates = [edge(g, work["values"]) for t in work["tasks"] for g in t["gated_on"]]
    retirement = []
    for task in work["tasks"]:
        for item in task["prior_failures"]:
            ref = work["history"].get(item["experiment_id"])
            actual = load(ref).get("honest_verdict") if ref else None
            current = next(s for s in slots if s["task_id"] == task["id"])
            retirement.append(
                dict(
                    task_id=task["id"],
                    prior_experiment_id=item["experiment_id"],
                    planned_prior_verdict=item["verdict"],
                    actual_prior_verdict=actual,
                    actual_current_verdict=current["honest_verdict"],
                    unchanged_verdict=actual is not None and actual == current["honest_verdict"],
                    prior_reference=ref,
                    scope_unchanged=True,
                    broader_hypothesis_retired=False,
                    reopening_evidence=item["addressed_by"],
                    action="preserve_original_scope_and_require_changed_evidence",
                )
            )
    elapsed = (work["ended_ns"] - work["started_ns"]) / 1e9
    gate = work["publication_gate"]
    value: Json = dict(
        experiment_id=9012,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_"
        + ("upstream_science_unavailable" if owned else "owned_validation"),
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
            generated_tokens=0,
        ),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        claim_scope="Fourteen administrative outcomes; graph-oracle, independent human-label, learning and hardware claims remain separate and unmeasured.",
        exposure_scope="Frozen exposed graph development; original disqualified graph evidence; public human annotations with no unseen source groups.",
        independent_generalization_score=0,
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
                cluster_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                metric="task_slot_accounted",
                value=1,
                numerator=1,
                denominator=1,
                status="completed",
                exclusion_reason=None,
            )
            for s in slots
        ],
        reconciliation_ready_score=int(owned),
        slot_rows=slots,
        gate_edge_rows=gates,
        full_contract_rows=work["contract"]["rows"],
        v769_science_summary=science,
        retirement_scope_rows=retirement,
        gate_check_summary=failures + [g for g in gates if not g["passed"]],
        continuation_evidence=[
            dict(
                upstream=g["upstream"],
                failed_field=g["artifact_field"],
                observed=g["observed"],
                next_evidence="Qualify the named original producer and its exact frozen gate; missing downstream science needs original candidate, causal-feedback or complete-process rows. Source authority additionally needs600 unseen human-source groups, and ARC needs30 new uncensored episodes in three families with both arms.",
            )
            for g in gates
            if not g["passed"]
        ],
        duration_s=elapsed,
        random_seed=7699012,
        phase_spans=[dict(phase="authentication_and_reconciliation", start_s=0, end_s=elapsed)],
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        owned_statement_coverage=validation["coverage"],
        validation_receipts=[*work["publication_receipts"], *validation["receipts"]],
        validation_state=validation,
        preconditions_checked=work["preconditions"],
        acceptance_gates=dict(
            owned_checks=owned,
            complete_accounting=True,
            graph_benefit=False,
            causal_learning_benefit=False,
            natural_label_benefit=False,
            deployment_benefit=False,
        ),
        source_artifact_hashes=[
            dict(
                s["reference"],
                experiment_id=s["experiment_id"],
                fields_imported=["original_outcome", "gate_operands"],
            )
            for s in slots
            if s.get("reference")
        ],
        raw_shard_hashes=work["references"],
        code_config_hashes=work["code"],
        cited_upstream_artifacts=[
            dict(
                ref,
                experiment_id=int(identity[3:7]),
                fields_imported=["original_honest_verdict", "historical_model_work_only"],
            )
            for identity, ref in sorted(work["history"].items())
        ],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        repository_health=validation.get("repository_health", []),
        publication_scope="Unchanged historical FoVer only; no V769 science certification or external publication.",
        **{k: gate["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=gate["paper_ready"],
        unmet_gates=gate["unmet_gates"],
        activation_authorized=False,
        methodology_note="Authenticate the frozen full-object contract and original producer or exact scheduler bytes. Recompute each source-group and administrative denominator separately. Preserve seven graph, ten causal and four human-label thresholds with missing operands explicit. No model load, training, benchmark, graph oracle evaluation or hardware call occurs. Historical invocation evidence is counted once and never treated as current work. Scoped normal-exit checks and newly owned statement coverage determine accounting readiness independently of the historical FoVer publication gate.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Retained bytes and independent denominators bind each scope; accounting and historical publication readiness cannot certify new scientific benefit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from retained primitives so a changed aggregate cannot create readiness."""
    value = json.loads(path.read_bytes())
    require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    for ref in work["references"] + work["code"]:
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
    for ref in work["code"]:
        require(sha256_file(Path(ref["original_path"])) == ref["sha256"], "executed_code_drift")
    require(
        work["contract"] == authority.authority(Path(work["raw"]) / "authority"), "authority_drift"
    )
    lines = (
        Path(
            next(
                r["path"]
                for r in work["references"]
                if r.get("original_path") == str(ROOT / "ops/conductor-log.md")
            )
        )
        .read_text()
        .splitlines()
    )
    with patch.object(prior, "MILESTONE", MILESTONE):
        for task, slot in zip(work["tasks"][:-1], work["slots"], strict=True):
            entry = work["values"].get(task["id"], {})
            original = load(entry["reference"]) if entry else {}
            require(not entry or entry["original"] == original, "original_bytes")
            expected = prior.slot(
                task,
                entry.get("value", {}),
                entry.get("reference"),
                [line for line in lines if task["title"][:48] in line and "GATE_BLOCK" in line],
                work["values"],
            )
            require(
                all(slot[k] == expected[k] for k in ["qualified", "disposition", "conductor_rows"]),
                "slot_custody_drift",
            )
            require(
                slot["honest_verdict"] == original.get("honest_verdict")
                and slot["verdict_class"] == original.get("verdict_class"),
                "original_verdict_drift",
            )
            if slot["qualified"]:
                require(
                    entry["reference"]["sha256"] == PINS[slot["experiment_id"]]
                    and entry["value"] == original,
                    "producer_bytes",
                )
    require(value == build(work, value["validation_state"]), "reduction_drift")
    return dict(passed=True, replay_passed=True, slots=14)


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged scanners and a fresh CLI process to check the exact candidate bytes."""
    with patch.multiple(prior, CLI=CLI, progress=progress):
        return prior.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Publish checked accounting; retain disqualified private bytes when receipts are absent."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261111"], default="20261111")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = args.output.absolute()
        receipt_path = args.receipts or output.parent / "raw" / NAME / "validation.json"
        raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
        work = collect(raw, args.date, args.sources)
        validation: Json = (
            json.loads(receipt_path.read_bytes())
            if receipt_path.is_file()
            else dict(passed=False, coverage=dict(complete=False), receipts=[])
        )
        progress("reductions_before", 0, 14)
        reduce(work)
        progress("reductions_after", 14, 0)
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        candidate = raw / "candidate.json"
        atomic_json(candidate, build(work, validation))
        require(
            bool(validation.get("owned_passed", validation["passed"])),
            "owned_validation_missing_or_failed",
        )
        require(verify_measured(validation["coverage"], OWNED), "owned_coverage_bytes")
        receipts = validation["receipts"]
        require(
            bool(receipts)
            and all(
                r["passed"]
                and r["exit_code"] == 0
                and not r.get("timed_out", False)
                and sha256_file(ROOT / r["log_path"]) == r["log_sha256"]
                for r in receipts
            ),
            "owned_receipts",
        )
        plan = load(validation["plan_reference"])
        require([r["command_argv"] for r in receipts] == plan["argv"], "frozen_validation_argv")
        require(
            {r["name"] for r in receipts} == CHECKS
            and plan["code_hashes"] == {p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
            "validation_scope_or_source_drift",
        )
        report = terminal(candidate, raw / "preflight_checks")
        require(report["passed"], "owned_terminal_checks")
        validation["receipts"].extend(report["receipts"])
        require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3600, "measurement_deadline")
        published = publish_primary(
            output, build(work, validation), lambda p: terminal(p, raw / "terminal_checks")
        )
        print(json.dumps(published), flush=True)
        progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
