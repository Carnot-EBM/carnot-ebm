"""REQ-REPORT-8998: complete accounting preserves qualified methods and absent science."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v767_capstone as prior
from carnot.reporting import v768_branch_contract as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v763_capstone import references, resolve
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8998_v768_capstone", "exp8998-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v768_capstone.py", CLI]
TEST = "tests/python/test_v768_capstone_8998.py"
CHECKS = {
    "owned_unit_child",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018_021",
    "coverage_combine",
    "coverage_report",
}
PINS = {
    8985: "sha256:e1a2c0669ab34f6fbd3ede92fc27fe3019b880659764d41ff2b7edd1ab78d45b",
    8986: "sha256:bf91b5f34041dcb903d48c477920836d1c98a17e381bf71c7dd19b8d7eaaee8c",
    8987: "sha256:32fbf7d2e347f56cf3baf5d92a73b7d62ab8bd68373518d0e5346a2ac485e920",
    8988: "sha256:6f8cde0ffece06efe64743dc72705fd189dd2376fcb2a4b21fe9e5fe3de733e2",
    8989: "sha256:bdbdc46d94d0dc6d0d3667c72ab1a9e47f4b4a60cb98cabea4cb2015623571ff",
    8990: "sha256:70882818afd83e69a4ea72a8530f99ce28d71ee8c87ffd3aa5b8f8a69c05c2bd",
    8992: "sha256:3c5d00c77e907719490026479ecae181a3bdb569fe8f3bec02a1d42d687db606",
    8997: "sha256:76299450edbd259a0aaf671a1a20a8d356127829e68cf10979f4f478709afbc4",
}
CONTRACT_PINS = {n: PINS[n] for n in [8987, 8988]}
HISTORY = "results/experiment_8984_v767_capstone.json"
HISTORY_PIN = "sha256:df2cfe785ec32fe95e0d29c2a49d4a5fad4d19c337dfaae3598574586d0bc5e9"
checksum = prior.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so a supervisor can distinguish work from a stall."""
    print(f"[exp8998] phase={phase} completed={completed} pending={pending}", flush=True)


def edge(gate: Json, values: Json) -> Json:
    """Preserve null, absence and observed zero as different operands."""
    row = prior.edge(gate, values)
    observed = row["observed"]
    state = (
        "scientific_null"
        if observed is None
        else "observed_zero"
        if observed == 0
        else "observed_value"
    )
    return dict(
        row,
        op=gate["op"],
        operand_status=observed if observed in ("missing_file", "missing_field") else state,
    )


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Snapshot observed bytes before reduction; unavailable producers remain unavailable."""
    progress("authentication_before", 0, 14)
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = authority.resources(raw, run_date)
    branch_path = ROOT / "results/experiment_8985_v768_branch_contract.json"
    branch = load(retain(branch_path, raw, PINS[8985]))
    tasks = branch["full_task_objects"]
    require(canonical_hash(tasks) == authority.PIN, "task_authority")
    refs = [
        retain(ROOT / p, raw, sha256_file(ROOT / p))
        for p in ["ops/conductor-log.md", "research-complete.yaml", "ops/exclusion_manifest.yaml"]
    ]
    history = retain(ROOT / HISTORY, raw, HISTORY_PIN)
    refs.append(history)
    contracts = []
    for number in [8987, 8988]:
        path = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        ref = retain(path, raw, CONTRACT_PINS[number])
        refs.append(ref)
        contracts.append(load(ref))
    lines = Path(refs[0]["path"]).read_text().splitlines()
    values: Json = {}
    failures = authority.resource_failures(preconditions, raw)
    for index, task in enumerate(tasks[:-1]):
        n = int(task["id"][3:7])
        candidates = list(sources.glob(f"experiment_{n}_*.json"))
        if len(candidates) == 1:
            path = candidates[0]
            ref = retain(path, raw, sha256_file(path))
            value = load(ref)
            if ref["sha256"] != PINS.get(n):
                failures.append(
                    dict(
                        check="input_bytes",
                        upstream=task["id"],
                        **ref,
                        artifact_field="sha256",
                        op="==",
                        expected=PINS.get(n),
                        observed=ref["sha256"],
                        passed=False,
                    )
                )
                value = dict(value, flagged_adversarial=True)
            values[task["id"]] = dict(value=value, reference=ref)
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
    catalog = {
        r["sha256"]: r
        for e in values.values()
        for r in references(e["value"])
        if r.get("path")
        and r.get("sha256")
        and Path(r["path"]).is_file()
        and sha256_file(Path(r["path"])) == r["sha256"]
    }
    for s in slots:
        for g in s["gate_check_summary"]:
            g.update(op=g.get("operator", "=="), passed=g.get("passed", False))
        s["planned_status"] = "planned"
        s["implemented_status"] = s["disposition"]
        entry = values.get(s["task_id"], {}).get("value", {})
        if entry.get("required_checks_passed") is False:
            failures.append(
                dict(
                    check="producer_validation",
                    upstream=s["task_id"],
                    **s["reference"],
                    artifact_field="required_checks_passed",
                    op="==",
                    expected=True,
                    observed=False,
                    passed=False,
                )
            )
        if s["qualified"]:
            for r in entry.get("raw_shard_hashes", []) + entry.get("source_artifact_hashes", []):
                if r.get("path") and r.get("sha256"):
                    bound = resolve(r, catalog)
                    refs.append(retain(Path(bound["path"]), raw, r["sha256"]))
        for r in entry.get("validation_receipts", []):
            if not r.get("passed", True) and r.get("log_sha256"):
                refs.append(retain(ROOT / r["log_path"], raw, r["log_sha256"]))
        progress("operands_retained", s["experiment_id"] - 8984, 8997 - s["experiment_id"])
    progress("publication_gate_before", 13, 1)
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
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v767_capstone.py",
            "python/carnot/reporting/v768_branch_contract.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/experiment_7303_validation_scope.py",
            "python/carnot/reporting/spline_development_admission_8987.py",
            "openspec/change-proposals/v768-spline-protocol.json",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    ]
    progress("authentication_after", 14, 0)
    return dict(
        detector_contract=contracts[0]["seven_bound_contract"],
        learning_contract=contracts[1]["ten_bound_contract"],
        tasks=tasks,
        slots=slots,
        values=values,
        references=refs,
        history=history,
        code=code,
        publication_gate=gate,
        publication_receipts=receipts,
        preconditions=preconditions,
        failures=failures,
        started_ns=start,
        ended_ns=time.monotonic_ns(),
        owner_pid=os.getpid(),
        raw=str(raw.absolute()),
        run_date=run_date,
    )


def reduce(work: Json) -> Json:
    """Recompute primitive development losses and annotation support without fitting heads."""
    from carnot.reporting.spline_development_admission_8987 import admission

    admitted = {s["experiment_id"]: load(s["reference"]) for s in work["slots"] if s["qualified"]}
    result: Json = dict(
        development=dict(status="missing_file", collection_permitted=False),
        natural=dict(status="missing_file", independent_human_groups=None),
        scientific_benefit=False,
    )
    if 8987 in admitted:
        value = admitted[8987]
        raw_ref = next(
            r
            for r in work["references"]
            if r.get("original_path") == value["raw_shard_hashes"][0]["path"]
        )
        primitive = load(raw_ref)
        require(primitive["rows"] == value["rows"], "development_primitive_rows")
        metrics, checks = admission(primitive["rows"])
        require(
            metrics == value["development_metrics"]
            and checks == value["development_admission_rows"],
            "development_reduction",
        )
        result["development"] = dict(
            status="exposed_development_only",
            metrics=metrics,
            admission_rows=checks,
            collection_permitted=all(r["passed"] for r in checks),
            reference=work["values"][value["task_id"]]["reference"],
        )
    if 8989 in admitted:
        value = admitted[8989]
        raw_ref = next(
            r
            for r in work["references"]
            if r.get("original_path") == value["raw_shard_hashes"][0]["path"]
        )
        primitive = load(raw_ref)["reduction"]
        require(
            primitive["rows"] == value["rows"]
            and primitive["annotation_origin_rows"] == value["annotation_origin_rows"],
            "natural_primitive_rows",
        )
        rows = primitive["rows"]
        groups = {r["group_id"] for r in rows}
        humans = sum(r["independent_human_decision"] for r in value["annotation_origin_rows"])
        require(
            len(rows) == value["completed_count"]
            and len(groups) == value["independent_count"]
            and humans == value["independent_human_label_group_count"] == 0,
            "natural_reduction",
        )
        result["natural"] = dict(
            status="complete_null_machine_annotation_only",
            response_rows=len(rows),
            question_groups=len(groups),
            machine_positive_rows=sum(r["value"] for r in rows),
            independent_human_groups=humans,
            annotation_origin_rows=value["annotation_origin_rows"],
            gate_operands=value["independent_label_gate_operands"],
            independent_label_ready=False,
            reference=work["values"][value["task_id"]]["reference"],
        )
    for name, number, contract in [
        ("detector_bounds", 8993, work["detector_contract"]),
        ("learning_bounds", 8995, work["learning_contract"]),
    ]:
        require(number not in admitted, "unregistered_new_science_requires_raw_reducer")
        task = work["tasks"][number - 8985]
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **r,
                observed="missing_file",
                status="unmeasured",
                passed=False,
                numerator=None,
                denominator=0,
                upstream=task["id"],
                path=task["deliverable"],
                next_evidence="Qualified real candidate/feedback rows and unchanged registered paired bootstrap; fixtures and prior results cannot fill this bound.",
            )
            for i, r in enumerate(contract["bounds"])
        ]
    result["complete_service_costs"] = [
        dict(
            arm=a,
            independent_units=0,
            complete_process_s=None,
            cost_ratio_upper95=None,
            status="missing_file",
            upstream="exp8996-complete-spline-cost",
            next_evidence="24 paired fresh processes including startup, updates, guards, fsync, restart, final drain and shutdown.",
        )
        for a in ["frozen_spline", "adaptive_spline", "frozen_linear", "adaptive_linear"]
    ]
    result["online_methods"] = dict(
        qualified=8988 in admitted,
        scope="method fixtures only; no real-stream benefit",
        ready_score=admitted.get(8988, {}).get("online_methods_ready_score", "missing_file"),
    )
    result["arc"] = dict(
        observed=admitted.get(8985, {}).get("arc_new_outcomes_ready_score", "missing_file"),
        scope="optional observational support only",
        minimum="30 new uncensored episodes, three families, both arms per held family",
        qualified_association=False,
    )
    return result


def build(work: Json, validation: Json) -> Json:
    """Complete owned accounting can be ready while upstream science is terminal blocked."""
    require(canonical_hash(work["tasks"]) == authority.PIN, "task_authority")
    science = reduce(work)
    owned = validation["passed"] and validation["coverage"].get("complete") is True
    verdict = "blocked" if owned else "disqualified"
    slots = [
        *work["slots"],
        dict(
            task_id=TASK,
            experiment_id=8998,
            disposition="self_slot",
            accounted=True,
            qualified=bool(owned),
            verdict_class=verdict,
            planned_status="planned",
            implemented_status="completed_owned_accounting",
        ),
    ]
    elapsed = (work["ended_ns"] - work["started_ns"]) / 1e9
    gates = [edge(g, work["values"]) for t in work["tasks"] for g in t["gated_on"]]
    failures = [
        *work["failures"],
        *[r for s in slots for r in s.get("gate_check_summary", []) if not r["passed"]],
    ]
    if science["natural"].get("independent_human_groups") == 0:
        failures.append(
            dict(
                check="independent_natural_annotation",
                upstream="exp8989-natural-annotation-authority",
                **science["natural"]["reference"],
                artifact_field="independent_label_ready_score",
                op="==",
                expected=1,
                observed=0,
                passed=False,
            )
        )
    gate = work["publication_gate"]
    first = (
        failures[0]
        if failures
        else dict(
            upstream="exp8993-spline-calibrated-decisions",
            artifact_field="rows",
            observed="missing_file",
            check="fresh_science",
        )
    )
    value: Json = dict(
        experiment_id=8998,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_"
        + ("upstream_science_unavailable" if owned else "owned_validation"),
        verdict_class=verdict,
        blocked_at=dict(
            upstream=first["upstream"],
            artifact_field=first["artifact_field"],
            observed=first["observed"],
        ),
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
        claim_scope="Fourteen administrative outcomes and exposed development reductions; no qualified fresh detector, online benefit or independent verifier claim.",
        exposure_scope="Exposed BFS graphs, constructed method fixtures and machine annotated public release; old science remains historical.",
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
        slot_rows=slots,
        gate_edge_rows=gates,
        v768_science_summary=science,
        gate_check_summary=failures,
        retirement_scope_rows=[
            dict(
                scope="V767 fixed common-feature Gibbs head",
                action="preserve_three_failed_development_gates",
                prior_reference=work["history"],
                prior_verdict=load(work["history"])["honest_verdict"],
                prior_failed_development_rows=load(work["history"])["branch_outcomes"][
                    "shared_feature"
                ]["admission_rows"],
                changed_spline_scope_retired=False,
            ),
            dict(
                scope="V768 convex spline",
                action="development_admitted; fresh_science_unmeasured",
                broader_hypothesis_retired=False,
            ),
            dict(
                scope="RAGognize independent evidence",
                action="machine_annotation_null; not_independent_human_truth",
            ),
            dict(scope="Exp8990", action="quarantine_validation_failure; no_scientific_null"),
        ],
        continuation_evidence=[
            dict(
                upstream=r["upstream"],
                failed_field=r["artifact_field"],
                observed=r["observed"],
                next_evidence="Thirty new uncensored live episodes, three families and both arms per held family, newer than the authenticated V767 frontier."
                if r["artifact_field"] == "arc_new_outcomes_ready_score"
                else "Independent documented human decisions for at least200 source groups with both labels, authorized original text and no cheap oracle."
                if r["upstream"] == "exp8989-natural-annotation-authority"
                else "Correct producer inference provenance and its owned validator failure; obtain qualified original shard A and missing shard B before corpus admission, seven detector bounds, 24 streams, ten learning bounds and24 complete-cost process pairs.",
            )
            for r in failures
        ],
        duration_s=elapsed,
        phase_spans=[dict(phase="authentication_and_reconciliation", start_s=0, end_s=elapsed)],
        random_seed=7688998,
        required_checks_passed=bool(owned),
        flagged_adversarial=not owned,
        reconciliation_ready_score=int(owned),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=[*work["publication_receipts"], *validation["receipts"]],
        validation_state=validation,
        preconditions_checked=work["preconditions"],
        acceptance_gates=dict(
            owned_checks=bool(owned),
            complete_accounting=True,
            method_readiness=science["online_methods"]["qualified"],
            observational_support=False,
            scientific_benefit=False,
        ),
        source_artifact_hashes=[
            dict(
                s["reference"],
                experiment_id=s["experiment_id"],
                fields_imported=["original_outcome", "rows", "gate_operands"],
            )
            for s in slots
            if s.get("reference")
        ],
        raw_shard_hashes=work["references"],
        code_config_hashes=work["code"],
        cited_upstream_artifacts=[
            dict(
                s["reference"],
                experiment_id=s["experiment_id"],
                fields_imported=[
                    "historical_model_provenance_only"
                    if s["experiment_id"] == 8990
                    else "original_outcome_and_rows"
                ],
            )
            for s in slots
            if s.get("reference")
        ],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        repository_health=validation.get("repository_health", []),
        source_limitations=dict(
            natural="Machine annotations have no documented independent human decisions.",
            graph="Exp8990 owned validation failed; shard B absent; dependent science unmeasured.",
            arc="No new qualifying live outcome support.",
            historical="FoVer readiness and V767 failures retain their original scopes.",
        ),
        publication_scope="Unchanged historical FoVer only; no V768 graph, natural-evidence or external publication certification.",
        **{k: gate["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=gate["paper_ready"],
        unmet_gates=gate["unmet_gates"],
        activation_authorized=False,
        methodology_note="Authenticate original fourteen-task authority and producer bytes. Retain rejected corpus and scheduler skip paths. Reduce fixed exposed-development predictions and public machine-label rows, without a model load or head fit. Seven detector and ten learning bounds remain unmeasured because downstream science never qualified. Missing service costs supply no efficiency claim. Administrative readiness needs normal-exit owned validation and measured unit/child coverage; historical FoVer readiness is independent.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Retained exact bytes and local denominators; administrative and historical readiness cannot certify new science.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from retained operands so a rehashed aggregate cannot create readiness."""
    value = json.loads(path.read_bytes())
    require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    for ref in work["references"] + work["code"]:
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
    for ref in work["code"]:
        require(sha256_file(Path(ref["original_path"])) == ref["sha256"], "code_drift")
    lines = Path(work["references"][0]["path"]).read_text().splitlines()
    with patch.object(prior, "MILESTONE", MILESTONE):
        for task, s in zip(work["tasks"][:-1], work["slots"], strict=True):
            entry = work["values"].get(task["id"], {})
            original = load(entry["reference"]) if entry else {}
            expected = prior.slot(
                task,
                original,
                entry.get("reference"),
                [line for line in lines if task["title"][:48] in line and "GATE_BLOCK" in line],
                work["values"],
            )
            require(
                s["qualified"] == expected["qualified"]
                and s["disposition"] == expected["disposition"],
                "slot_custody_drift",
            )
            if s["qualified"]:
                require(
                    entry["reference"]["sha256"] == PINS[s["experiment_id"]]
                    and entry["value"] == original,
                    "producer_bytes",
                )
    require(value == build(work, value["validation_state"]), "reduction_drift")
    return dict(passed=True, replay_passed=True, slots=14)


def terminal(candidate: Path, raw: Path) -> Json:
    """Reuse unchanged adversarial, strict-row and fresh-process replay checks."""
    with patch.multiple(prior, CLI=CLI, progress=progress):
        return prior.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Expose only a checked primary; absent owned receipts remain disqualified privately."""
    progress("start")
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
        validation = (
            json.loads(receipt_path.read_bytes())
            if receipt_path.is_file()
            else dict(passed=False, coverage=dict(complete=False), receipts=[])
        )
        if validation["passed"]:
            require(
                authority.prior.verify_measured(validation["coverage"], OWNED),
                "owned_coverage_bytes",
            )
            receipts = validation["receipts"]
            require(
                bool(receipts)
                and all(
                    r["passed"]
                    and r["exit_code"] == 0
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
        progress("reductions_before", 0, 14)
        reduce(work)
        progress("reductions_after", 14, 0)
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        candidate = raw / "candidate.json"
        atomic_json(candidate, build(work, validation))
        require(validation["passed"], "owned_validation_missing_or_failed")
        report = terminal(candidate, raw / "preflight_checks")
        require(report["passed"], "owned_terminal_checks")
        validation["receipts"].extend(report["receipts"])
        require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3900, "measurement_deadline")
        result = build(work, validation)
        published = publish_primary(output, result, lambda p: terminal(p, raw / "terminal_checks"))
        print(json.dumps(published), flush=True)
        progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
