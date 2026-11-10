"""REQ-REPORT-8984: complete accounting preserves scientific failures and source blocks."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import v767_branch_contract as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, validate_primary
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8984_v767_capstone", "exp8984-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v767_capstone.py", CLI]
TEST = "tests/python/test_v767_capstone_8984.py"
PINS = {
    8971: "sha256:5d42cd9cec9edd0339d367b3352af0f1b05392562497c9d2a0a194c7077d4212",
    8972: "sha256:aca5e51be78c64c911d49e6ce3cf3e036393ff59f1a300af29486f91733c56f4",
    8973: "sha256:3870b7055fbff26bc4a35316f60555abe99c0f2326c14156540af78a2e5b8fa6",
    8975: "sha256:4203efad6223b48bd6627a0953a289c26ce9a7df0832e878cf37317f84bb0089",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so the parent can detect stalled work."""
    print(f"[exp8984] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """Keep the digest outside its own input so replay can verify complete bytes."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def edge(gate: Json, values: Json) -> Json:
    """Missing files and missing fields must not become observed zeroes."""
    entry = values.get(gate["upstream"], {})
    ref = entry.get("reference", {})
    observed = (
        entry.get("value", {}).get(gate["artifact_field"], "missing_field")
        if entry
        else "missing_file"
    )
    return dict(
        upstream=gate["upstream"],
        check="declared_gate",
        path=ref.get("path"),
        sha256=ref.get("sha256"),
        artifact_field=gate["artifact_field"],
        operator=gate["op"],
        expected=gate["value"],
        observed=observed,
        passed=observed == gate["value"],
    )


def slot(task: Json, value: Json, ref: Json | None, logs: list[str], values: Json) -> Json:
    """Account for scheduler completion without importing its missing scientific result."""
    checks = [edge(g, values) for g in task["gated_on"]]
    reason = None
    if ref is None:
        disposition = "conductor_skip_without_primary" if logs else "missing_primary"
        checks.append(
            dict(
                upstream=task["id"],
                check="missing_file",
                path=task.get("deliverable"),
                sha256=None,
                artifact_field="primary",
                operator="exists",
                expected=True,
                observed="missing_file",
            )
        )
    elif value.get("blocked_at_layer") == "conductor_pre_gate":
        matches = [
            r
            for r in checks
            if r["upstream"] == value.get("failed_upstream")
            and r["artifact_field"] == value.get("failed_field")
        ]
        valid = value.get("experiment") == int(task["id"][3:7]) and len(matches) == 1
        valid = valid and all(
            value.get("failed_" + key) == matches[0][target]
            for key, target in [
                ("observed", "observed"),
                ("expected", "expected"),
                ("operator", "operator"),
                ("evidence_sha256", "sha256"),
            ]
        )
        disposition = (
            "authenticated_scheduler_skip"
            if valid and not matches[0]["passed"]
            else "quarantined_primary"
        )
    else:
        try:
            validate_primary(value, Path(task["deliverable"]))
            require(
                value["task_id"] == task["id"]
                and value["milestone"] == MILESTONE
                and value["required_checks_passed"] is True
                and value["flagged_adversarial"] is False
                and value["verdict_class"] != "disqualified"
                and not (value["verdict_class"] == "positive" and value.get("verifier_is_oracle"))
                and value["owned_statement_coverage"].get(
                    "complete", value["owned_statement_coverage"].get("percent_covered") == 100
                ),
                "producer_qualification",
            )
            disposition = "qualified_primary"
        except (ValueError, KeyError, TypeError) as error:
            disposition, reason = "quarantined_primary", str(error)
    if disposition == "quarantined_primary":
        checks.append(
            dict(
                upstream=task["id"],
                check="producer_qualification",
                **(ref or {}),
                artifact_field="producer_contract",
                operator="==",
                expected="qualified_original",
                observed=reason or "scheduler_binding_rejected",
            )
        )
    return dict(
        task_id=task["id"],
        experiment_id=int(task["id"][3:7]),
        reference=ref,
        disposition=disposition,
        accounted=True,
        qualified=disposition == "qualified_primary",
        honest_verdict=value.get("honest_verdict"),
        verdict_class=value.get("verdict_class", "blocked"),
        conductor_rows=logs,
        conductor_gate_check_summary=value.get("gate_check_summary"),
        gate_check_summary=checks,
        producer_counts={
            k: value.get(k)
            for k in [
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
            ]
        },
    )


def collect(raw: Path, run_date: str, sources: Path = ROOT / "results") -> Json:
    """Retain original producer bytes and the exact authority before any numerical reduction."""
    progress("start_authentication", 0, 14)
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = authority.resources(raw, run_date)
    contract = authority.authority(ROOT)
    require(contract["ready"], "task_authority")
    tasks, values, refs = contract["tasks"], {}, []
    for name in [
        authority.DESIGN,
        "ops/conductor-log.md",
        "research-complete.yaml",
        "research-references.md",
    ]:
        refs.append(retain(ROOT / name, raw, sha256_file(ROOT / name)))
    log = Path(refs[1]["path"]).read_text().splitlines()
    for task in tasks[:-1]:
        n = int(task["id"][3:7])
        candidates = list(sources.glob(f"experiment_{n}_*.json"))
        if len(candidates) == 1:
            path = candidates[0]
            ref = retain(path, raw, sha256_file(path))
            value = load(ref)
            if n in PINS and ref["sha256"] != PINS[n]:
                value = dict(value, flagged_adversarial=True)
            values[task["id"]] = dict(value=value, reference=ref)
            refs.append(ref)
    slots = [
        slot(
            t,
            values.get(t["id"], {}).get("value", {}),
            values.get(t["id"], {}).get("reference"),
            [line for line in log if t["title"][:48] in line and "GATE_BLOCK" in line],
            values,
        )
        for t in tasks[:-1]
    ]
    historical: Json = {}
    from carnot.reporting.v767_branch_evidence import PINS as HISTORY_PINS

    for name, pin in HISTORY_PINS.items():
        if any(name.startswith(f"results/experiment_{n}_") for n in (8963, 8964, 8967, 8968, 8969)):
            ref = retain(ROOT / name, raw, pin)
            historical[str(load(ref).get("experiment_id", load(ref).get("experiment")))] = ref
            refs.append(ref)
    for entry in values.values():
        for method in entry["value"].get("method_source_rows", []):
            refs.append(retain(Path(method["path"]), raw, method["sha256"]))
    corpus = load(historical["8963"])
    labels = [
        retain(Path(r["label_reference"]["path"]), raw, r["label_reference"]["sha256"])
        for r in corpus["broker_manifest"].values()
    ]
    refs.extend(labels)
    code = [
        retain(ROOT / p, raw, sha256_file(ROOT / p))
        for p in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/current_work_receipt.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    ]
    progress("publication_gate_before", 13, 1)
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                "publication_gate",
                (sys.executable, "-u", "scripts/publication_gate.py", "--json"),
                "historic_publication",
                60,
            )
        ],
        log_dir=raw / "publication",
        heartbeat_s=15,
    )
    require(receipts[0]["passed"], "publication_gate_execution")
    gate = json.loads(Path(receipts[0]["log_path"]).read_bytes())
    refs.append(reference(Path(receipts[0]["log_path"])))
    for name in [
        authority.ACTIVE,
        "scripts/publication_gate.py",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "results/" + gate["gates"]["G1"]["source"],
    ]:
        refs.append(retain(ROOT / name, raw, sha256_file(ROOT / name)))
    progress("authentication_complete", 14, 0)
    return dict(
        tasks=tasks,
        slots=slots,
        values=values,
        references=refs,
        code=code,
        historical=historical,
        labels=labels,
        publication_gate=gate,
        publication_receipts=receipts,
        preconditions=preconditions,
        resource_failures=authority.resource_failures(preconditions, raw),
        started_ns=start,
        ended_ns=time.monotonic_ns(),
        owner_pid=os.getpid(),
        raw=str(raw),
        run_date=run_date,
    )


def reduce(work: Json) -> Json:
    """Reduce recorded predictions only; never refit a head or call a generator."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from carnot.reporting import relational_decisions_8964 as detector
    from carnot.reporting import online_benefit_audit_8967 as learning
    from carnot.reporting.shared_feature_methods_8973 import admission

    qualified = {s["task_id"]: load(s["reference"]) for s in work["slots"] if s["qualified"]}
    shared = next((v for v in qualified.values() if v["experiment_id"] == 8973), {})
    result: Json = dict(
        shared_feature=dict(status="unavailable", collection_permitted=False),
        natural_evidence=dict(status="unavailable", joined_pairs=0),
    )
    if shared:
        arms = {
            a: [r for r in shared["rows"] if r["arm"] == a and r.get("metric") == "brier"]
            for a in ("shared_energy", "shared_logistic", "prevalence")
        }
        metrics = {
            k: float(np.mean([(r["p_error"] - r["error"]) ** 2 for r in arms[a]]))
            for k, a in [
                ("energy_brier", "shared_energy"),
                ("logistic_brier", "shared_logistic"),
                ("prevalence_brier", "prevalence"),
            ]
        }
        metrics["auroc"] = float(
            roc_auc_score(
                [r["error"] for r in arms["shared_energy"]],
                [r["p_error"] for r in arms["shared_energy"]],
            )
        )
        gates = admission(metrics)
        require(gates == shared["development_admission_rows"], "development_reduction")
        result["shared_feature"] = dict(
            status="exposed_development_only",
            metrics=metrics,
            admission_rows=gates,
            collection_permitted=all(g["passed"] for g in gates),
            fixed_method="logistic encoder; identical27 features; eight-unit Gibbs head; T=1; seeds71/72/73",
            fresh_scientific_benefit=False,
        )
    natural = next((v for v in qualified.values() if v["experiment_id"] == 8975), {})
    if natural:
        joined = sum(r["source_joined"] for r in natural["rows"])
        require(joined == natural["completed_count"], "natural_join_reduction")
        result["natural_evidence"] = dict(
            status=natural["honest_verdict"],
            joined_pairs=joined,
            intended_pairs=len(natural["rows"]),
            metadata_audited=sum(r["annotation_present"] for r in natural["rows"]),
            ready=joined == 800,
            retry_requires="changed authorized source access and model terms; no unchanged retry",
        )
    old = {n: load(ref) for n, ref in work["historical"].items()}
    rows = old["8964"]["rows"]
    ids = [r["unit_id"] for r in rows if r["arm"] == "relational_energy"]
    truth = [r["error"] for r in rows if r["arm"] == "relational_energy"]
    probabilities = {
        a: [r["p_error"] for r in rows if r["arm"] == a] for a in detector.methods.ARMS
    }
    measured = detector.reduce(probabilities, truth, ids)
    require(
        measured["paired_intervals"] == old["8964"]["paired_intervals"],
        "historical_detector_reduction",
    )
    learned = learning.summarize(old["8967"]["rows"], old["8967"]["multiplicity_ledger"])
    require(
        learned["acceptance_gate_rows"] == old["8967"]["acceptance_gate_rows"],
        "historical_learning_reduction",
    )
    labels = [row for ref in work["labels"] for row in load(ref)["rows"]]
    costs = old["8968"]["rows"]
    result["v766"] = dict(
        answers=len({r["unit_id"] for r in labels}),
        errors=sum(r["label"] for r in labels),
        detector={k: measured[k] for k in ("descriptive_rows", "paired_intervals")},
        learning=dict(acceptance_gate_rows=learned["acceptance_gate_rows"]),
        costs=[
            dict(
                arm=a,
                units=len(group),
                mean_complete_process_s=sum(r["value"] for r in group) / len(group),
                scope="historical cached service; no V767 cost or deployment benefit",
            )
            for a in sorted({r["arm"] for r in costs})
            if (group := [r for r in costs if r["arm"] == a])
        ],
        archive_state="V766 now archived; planning archive lag was historical, not a missing scientific result",
        exp8969_state="authenticated_scheduler_skip",
        exp8970_state="absent_primary",
    )
    result["literature_mapping"] = [
        r for v in qualified.values() for r in v.get("method_source_rows", [])
    ]
    return result


def build(work: Json, validation: Json) -> Json:
    """Readiness measures complete accounting; external missing science stays terminal blocked."""
    require(canonical_hash(work["tasks"]) == authority.PIN, "task_authority")
    science = reduce(work)
    owned = validation["passed"] and validation["coverage"].get("complete") is True
    blocked = bool(
        work["resource_failures"]
        or any(not s["qualified"] for s in work["slots"])
        or not science["natural_evidence"].get("ready", False)
        or not science["shared_feature"]["collection_permitted"]
    )
    verdict = "disqualified" if not owned else "blocked" if blocked else "null"
    slots = [
        *work["slots"],
        dict(
            task_id=TASK,
            experiment_id=8984,
            disposition="self_slot",
            accounted=True,
            qualified=bool(owned),
            verdict_class=verdict,
        ),
    ]
    receipt = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            operation="retained prediction and scheduler row reduction"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        phase_spans=[
            dict(
                phase="authentication_and_reconciliation",
                start_s=0,
                end_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
            )
        ],
    )
    gate = work["publication_gate"]
    failures = [
        *work["resource_failures"],
        *[g for s in slots for g in s.get("gate_check_summary", []) if not g.get("passed", False)],
    ]
    for entry in work["values"].values():
        if isinstance(entry["value"].get("gate_check_summary"), list):
            failures.extend(entry["value"]["gate_check_summary"])
    value = dict(
        experiment_id=8984,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_"
        + ("external_science_and_source_access" if blocked else "v767_capstone"),
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        random_seed=7678984,
        verifier_is_oracle=True,
        exposure_scope="Exposed development graphs and historical BFS truth; no fresh scientific benefit or no-cheap-oracle claim.",
        independent_generalization_score=0,
        intended_count=14,
        completed_count=14,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        rows=[
            dict(
                unit_id=s["task_id"],
                cluster_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                metric="task_slot_accounted",
                value=1,
                numerator=1,
                denominator=1,
                status="completed",
                censor_reason=None,
            )
            for s in slots
        ],
        slot_rows=slots,
        actual_task_order=[t["id"] for t in work["tasks"]],
        gate_edge_rows=[edge(g, work["values"]) for t in work["tasks"] for g in t["gated_on"]],
        branch_outcomes=science,
        scientific_reductions=science,
        gate_check_summary=failures,
        retirement_scope_rows=[
            dict(
                scope="fixed_shared_feature_Gibbs_head",
                action="close_collection",
                evidence=science["shared_feature"],
                broader_hypothesis_retired=False,
                exclusion_manifest_edit_authorized=False,
            ),
            dict(scope="natural_labels", action="await_changed_authorized_source_access"),
            dict(
                scope="fresh_energy_online_cost",
                action="unmeasured_scheduler_or_absent; no scientific null inferred",
            ),
        ],
        continuation_evidence=dict(
            strongest_next_experiment="Only a changed head objective or optimization with a fixed representation development gate; no new collection before it passes.",
            evidence_changes_prior_odds="Common features did not rescue this fixed energy head. Coordinate mismatch alone does not explain its failure.",
            natural_evidence="Obtain already-authorized original dialogue access before a grouped annotation-relative evaluation.",
            activation_authorized=False,
        ),
        publication_gate=dict(
            gate,
            scope="Historic FoVer only; does not certify this milestone or authorize publication.",
        ),
        **{k: gate["gates"][k] for k in ("G1", "G2", "G3", "G4")},
        paper_ready=gate["paper_ready"],
        unmet_gates=gate["unmet_gates"],
        reconciliation_ready_score=int(owned and all(s["accounted"] for s in slots)),
        required_checks_passed=bool(owned),
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_checks=bool(owned),
            complete_accounting=True,
            fresh_scientific_benefit=False,
            natural_evidence_access=science["natural_evidence"].get("ready", False),
        ),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        validation_state=validation,
        preconditions_checked=work["preconditions"],
        source_artifact_hashes=[
            dict(
                s["reference"],
                experiment_id=s["experiment_id"],
                fields_imported=["original_terminal_outcome", "rows", "gate_check_summary"],
            )
            for s in work["slots"]
            if s["reference"]
        ],
        code_config_hashes=work["code"],
        raw_shard_hashes=work["references"],
        cited_upstream_artifacts=[
            dict(r, experiment_id=n, fields_imported=["historical_rows_only"])
            for n, r in work["historical"].items()
        ],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        repository_health=validation.get("repository_health", []),
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Exact retained inputs and local denominators; accounting readiness and historical publication gates grant no fresh scientific credit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from pinned rows so a rehashed summary cannot authorize publication."""
    value = json.loads(path.read_bytes())
    require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    for ref in work["references"] + work["code"]:
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
    for s in work["slots"]:
        if s["qualified"] and s["experiment_id"] in PINS:
            require(s["reference"]["sha256"] == PINS[s["experiment_id"]], "producer_pin")
    require(value == build(work, value["validation_state"]), "reduction_drift")
    return dict(passed=True, replay_passed=True, slots=14)


def terminal(candidate: Path, raw: Path) -> Json:
    """Keep actual unchanged validator findings bound to the private candidate bytes."""
    progress("terminal_validators_before", 0, 3)
    commands = [
        CommandSpec(name, (sys.executable, "-u", *args), "owned_terminal", 180)
        for name, args in [
            ("cold_replay", [CLI, "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]
    ]
    receipts = run_commands(ROOT, commands, log_dir=raw, heartbeat_s=15)
    progress("terminal_validators_after", 3, 0)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Publish only checked bytes; absent validation never grants readiness."""
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261110"], default="20261110")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--receipts", type=Path, default=ROOT / "results/raw" / NAME / "validation.json"
    )
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
        work = collect(raw, args.date, args.sources)
        validation = (
            json.loads(args.receipts.read_bytes())
            if args.receipts.is_file()
            else dict(passed=False, coverage=dict(complete=False), receipts=[])
        )
        if validation["passed"]:
            require(
                authority.verify_measured(validation["coverage"], OWNED), "owned_coverage_bytes"
            )
            require(all(r["passed"] for r in validation["receipts"]), "owned_receipts")
        progress("row_reductions_before", 0, 4)
        reduce(work)
        progress("row_reductions_after", 4, 0)
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        candidate = raw / "preflight.json"
        atomic_json(candidate, build(work, validation))
        report = terminal(candidate, raw / "preflight_checks")
        require(report["passed"], "owned_terminal_checks")
        validation["receipts"].extend(report["receipts"])
        require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4200, "fixed_deadline")
        result = build(work, validation)
        published = publish_primary(output, result, lambda p: terminal(p, raw / "terminal_checks"))
        print(json.dumps(published), flush=True)
        progress("complete", 14, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
