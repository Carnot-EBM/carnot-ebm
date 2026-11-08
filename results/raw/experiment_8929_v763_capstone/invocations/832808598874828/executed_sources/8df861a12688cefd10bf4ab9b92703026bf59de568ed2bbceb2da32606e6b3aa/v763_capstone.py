"""REQ-REPORT-8929: reconcile thirteen slots without granting scientific success.

Original producer bytes and execution authority are pinned. Administrative
receipts stay outside scientific comparisons, so neither counts nor timing
repeats can become an independent generalization claim.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_runner_8825 as checks
from carnot.reporting import v763_branch_contract as authority
from carnot.reporting import v763_capstone_reduction as science
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_8929_v763_capstone", "exp8929-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v763_capstone.py",
    "python/carnot/reporting/v763_capstone_reduction.py",
    CLI,
]
TEST = "tests/python/test_v763_capstone_8929.py"
PINS = {
    8917: (
        "experiment_8917_v763_branch_contract.json",
        "cc2da034ec3708d6b45ea6f4520e7df05a06957119589bea473ceae833d01524",
        "contract_ready_score",
    ),
    8918: (
        "experiment_8918_v763_bfs_error_pilot.json",
        "b2d62a2d78535ad3bf5ca9aa83c330f2490ed7d40941a7a42b18a489aafd800e",
        "pilot_error_gate_score",
    ),
    8919: (
        "experiment_8919_v763_interpolated_update_methods.json",
        "ddc6d3cc5aecffabe0e99079833361e98f5085925ea6f30f3115b6109f7ab2ec",
        "candidate_ready_score",
    ),
    8920: (
        "experiment_8920_v763_label_demand_methods.json",
        "6fa9b66aa01b265ec5097f0ab8910e0281b42a3f4eda686aec3b17493d4691a3",
        "label_demand_ready_score",
    ),
    8921: (
        "experiment_8921_v763_inline_clock_methods.json",
        "93d1af0f09146a17914d9b3730ff84e3d6289d30c38dec9bb49470900e97fc19",
        "inline_clock_ready_score",
    ),
    8922: (
        "experiment_8922_v763_interpolated_causal_streams.json",
        "53a63467e2cbad29541ba7f3b3bb3b8aec5fa091ca54a1df26aed63e106c710b",
        "causal_rows_ready_score",
    ),
    8923: (
        "experiment_8923_label_demand_evaluation.json",
        "7a86cdde9ed000881c9ee292f3223e2c529fb8177f8ffbfcd013ca581574b99b",
        "failed_observed",
    ),
    8924: (
        "experiment_8924_v763_qwen_natural_canary.json",
        "2abdf8028dee3078c1978d8082cb34c88c099b626a3cf767b40f3f23cab332cb",
        "natural_canary_ready_score",
    ),
    8925: (
        "experiment_8925_v763_interpolated_benefit_audit.json",
        "88f6d845ac54d3ff1b553e135b4d8bf416cf714bbbbd1e73bcc0acb1635c7269",
        "interpolated_benefit_score",
    ),
    8926: (
        "experiment_8926_v763_durable_cost_qualification.json",
        "1c5f1969924341ce5416db9c649265dac03985946d722e111792b03372cb5a24",
        "adaptation_cost_ready_score",
    ),
    8927: (
        "experiment_8927_inline_request_cost.json",
        "353ea1de2edc7f3a85e6ceebdc7598434d0ae4cfea9703c00307b6cfddac1399",
        "failed_observed",
    ),
    8928: (
        "experiment_8928_v763_hardware_operation_budget.json",
        "efc2dabc7531fce61a6150657593d53f325656355669df28593be0af7d788230",
        "hardware_map_ready_score",
    ),
}
NEXT = {
    8917: "Retain original authority; any future task requires a new frozen full contract",
    8918: "Propose the rate-based collection only after freezing fresh graph splits and a k sensitivity control",
    8919: "Change representation or promotion mechanism; preserve all nine prospective and retention thresholds",
    8920: "Repair the failed owned adversarial acceptance gate without relaxing validators; requalify label access",
    8921: "Change observer implementation and pass every simultaneous overhead upper bound <=0.05 on new development worlds",
    8922: "Require a changed informative candidate and fresh sealed streams; unchanged interpolation is retired",
    8923: "Require label_demand_ready_score=1 before any fresh 40-draw evaluation; charge all-policy union and selection labels",
    8924: "Require independent blind source-semantic labels and broader source clusters; assisted criteria remain dependent",
    8925: "Retire fixed alpha=0.25 beta=0.5 eta=0.1 with the unchanged guard/delay/query schedule; new mechanism must pass all nine bounds",
    8926: "Retain qualified historical guarded costs; measure matched current optimizer total-service costs before a speed-quality claim",
    8927: "Require inline_clock_ready_score=1 before new full-request worlds; no qualified component fractions exist",
    8928: "Require supported equivalent hardware kernel, parity and measured complete-service host/transfer/durability fractions",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush progress without changing the fixed invocation deadline."""
    print(f"[exp8929] phase={phase} completed={completed} pending={pending}", flush=True)


def references(value: Json) -> list[Json]:
    """Support original list and mapping schemas without changing producer fields."""
    result = []
    for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"):
        refs = value.get(key, [])
        result.extend(refs.values() if isinstance(refs, dict) else refs)
    return result


def resolve(ref: Json, catalog: dict[str, Json]) -> Json:
    """An old path may resolve only to retained bytes with its original hash."""
    path = Path(ref["path"])
    actual = sha256_file(path) if path.is_file() else None
    bound = ref if actual == ref["sha256"] else catalog.get(ref["sha256"], ref)
    science.require(
        Path(bound["path"]).is_file() and sha256_file(Path(bound["path"])) == ref["sha256"],
        "operand_drift:" + str(path),
    )
    return dict(path=bound["path"], sha256=ref["sha256"], original_path=ref["path"])


def admission(value: Json, n: int, task: str) -> str | None:
    """Readiness zero is evidence; failed owned controls stay disqualified."""
    field = PINS[n][2]
    if field not in value:
        return "missing_field"
    if (
        value.get("verdict_class") == "disqualified"
        and value.get("required_checks_passed") is False
    ):
        return "disqualified_primary"
    coverage = value.get("owned_statement_coverage", {})
    measured = coverage.get("complete") is True and coverage.get("percent") == 100
    measured |= bool(
        coverage.get("num_statements", 0) > 0
        and coverage.get("covered_lines") == coverage.get("num_statements")
        and coverage.get("percent_covered") == 100
        and coverage.get("missing_lines") == 0
        and coverage.get("excluded_lines") == 0
    )
    measured |= bool(
        coverage.get("total_statements", 0) > 0
        and coverage.get("covered_statements") == coverage.get("total_statements")
        and coverage.get("percent_covered") == 100
        and coverage.get("missing_lines") == []
        and coverage.get("excluded_lines") == []
    )
    if (
        value.get("experiment_id") != n
        or value.get("task_id") != task
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != "20261108"
        or type(value[field]) is not int
        or value[field] not in (0, 1)
        or value.get("required_checks_passed") is not True
        or not measured
        or not value.get("validation_receipts")
        or value.get("verdict_class") not in {"positive", "circular_positive", "null", "blocked"}
        or value.get("verifier_is_oracle") is True
        and value["verdict_class"] == "positive"
        or any(
            not r["passed"]
            for r in value["validation_receipts"]
            if r.get("scope") != "repository_health"
        )
    ):
        return "rejected_producer"
    return None


def scheduler(value: Json, n: int, task: Json, evidence: dict[int, Json]) -> bool:
    """A skip needs both original scheduler bytes and the actual failed upstream gate."""
    edge = task["gated_on"][0]
    upstream = next((v for v in evidence.values() if v.get("task_id") == edge["upstream"]), {})
    source = ROOT / "results" / PINS[8920 if n == 8923 else 8921][0]
    return bool(
        value.get("schema") == "blocked_gate_check_v1"
        and value.get("experiment") == n
        and value.get("blocked_at_layer") == "conductor_pre_gate"
        and value.get("failed_upstream") == edge["upstream"]
        and value.get("failed_field") == edge["artifact_field"]
        and value.get("failed_evidence_path") == str(source)
        and value.get("failed_evidence_sha256") == reference(source)["sha256"]
        and type(value.get("failed_observed")) is int
        and value["failed_observed"] == 0
        and type(value.get("failed_expected")) is int
        and value["failed_expected"] == 1
        and type(upstream.get(edge["artifact_field"])) is int
        and upstream[edge["artifact_field"]] == value["failed_observed"]
    )


def checksum(value: Json) -> str:
    """The checksum covers every assertion except its own recursive field."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def collect(raw: Path, sources: Path = ROOT / "results") -> Json:
    """Authenticate all slots before reduction; absent external work remains blocked."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("authentication_before", 0, 12)
    evidence, primaries, catalog, hashes = {}, {}, {}, {}
    for n, (name, pin, _) in PINS.items():
        path = sources / name
        if path.is_file() and sha256_file(path) == "sha256:" + pin:
            value = json.loads(path.read_bytes())
            target = raw / "primaries" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            primaries[n], evidence[n] = reference(target), value
    for value in evidence.values():
        for ref in references(value):
            path = Path(ref["path"])
            if path not in hashes:
                hashes[path] = sha256_file(path) if path.is_file() else None
            if hashes[path] == ref["sha256"] and ref["sha256"]:
                catalog[ref["sha256"]] = ref
        progress("index_producer_operands", len(hashes), 0)
    contract: Json = dict(ready=False, tasks=[], failures=[])
    if 8917 in evidence:
        for ref in evidence[8917]["execution_snapshot"]:
            bound = resolve(ref, catalog)
            target = raw / "authority" / Path(ref["source"]["path"]).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(bound["path"]).read_bytes())
        contract = authority.authority(raw / "authority")
        science.require(
            contract["tasks"] == evidence[8917]["full_task_objects"], "original_authority_drift"
        )
    tasks = {8917 + i: task for i, task in enumerate(contract["tasks"])}
    slots, refs = [], list(primaries.values())
    for n, (name, pin, field) in PINS.items():
        value = evidence.get(n, {})
        task = tasks.get(n, dict(id=f"exp{n}-unavailable", gated_on=[], prior_failures=[]))
        reason = "missing_file" if not (sources / name).is_file() else "rejected_producer"
        bound = []
        if value:
            try:
                bound = [resolve(ref, catalog) for ref in references(value) if ref.get("sha256")]
                reason = (
                    (
                        "authenticated_scheduler_skip"
                        if scheduler(value, n, task, evidence)
                        else "rejected_producer"
                    )
                    if n in (8923, 8927)
                    else admission(value, n, task["id"])
                )
            except (OSError, ValueError, KeyError) as error:
                reason = "rejected_producer"
                progress("rejected_operand_" + str(error), len(slots), 12 - len(slots))
        refs.extend(bound)
        accounted = n in primaries and reason in (
            None,
            "disqualified_primary",
            "authenticated_scheduler_skip",
        )
        diagnostic = dict(
            check="scheduler_skip" if n in (8923, 8927) else "producer_gate",
            upstream=value.get("failed_upstream", task["id"]),
            artifact_field=value.get("failed_field", field),
            expected=1,
            observed=value.get("failed_observed", value.get(field)),
            path=str(sources / name),
            hash=primaries.get(n, {}).get("sha256"),
            missing_reason="rejected_producer"
            if reason == "disqualified_primary"
            else "observed_zero"
            if value.get(field) == 0
            else reason,
        )
        slots.append(
            dict(
                experiment_id=n,
                task_id=task["id"],
                accounted=accounted,
                disposition=reason or "qualified_primary",
                primary_reference=primaries.get(n),
                producer=value,
                task=task,
                gate_check_summary=([diagnostic] if reason or value.get(field) == 0 else [])
                + (
                    value.get("gate_check_summary", [])
                    if isinstance(value.get("gate_check_summary", []), list)
                    else []
                ),
            )
        )
        progress("slot_authenticated", len(slots), 12 - len(slots))
    admitted = {
        s["experiment_id"]: s["producer"]
        for s in slots
        if s["accounted"] and s["disposition"] != "authenticated_scheduler_skip"
    }
    progress("scientific_reductions_before", 0, len(admitted))
    reductions = science.reduce(admitted, raw)
    progress("scientific_reductions_after", len(admitted), 0)
    publication = run_commands(
        ROOT,
        [
            CommandSpec(
                "historical_publication_gates",
                (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                "owned",
                60,
            )
        ],
        log_dir=raw / "publication_gates",
        heartbeat_s=30,
    )[0]
    science.require(publication["passed"], "publication_gate_command")
    gates = json.loads((ROOT / publication["log_path"]).read_bytes())
    refs.append(reference(ROOT / publication["log_path"]))
    code = [
        reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            "scripts/experiment_template.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/publication_gate.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "openspec/capabilities/research-reporting/spec.md",
        ]
    ]
    sealed = []
    for ref in code:
        source = Path(ref["path"])
        target = raw / "executed_sources" / ref["sha256"].split(":")[1] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sealed.append(dict(reference(target), original_path=ref["path"]))
    code = sealed
    work = dict(
        slots=slots,
        contract=contract,
        reductions=reductions,
        raw=str(raw),
        sources=str(sources),
        references=list({(r["path"], r["sha256"]): r for r in refs}.values()),
        code=code,
        publication_gates=gates,
        publication_receipt=publication,
        started_ns=start,
        ended_ns=time.monotonic_ns(),
        owner_pid=os.getpid(),
        invocation=dict(
            argv=sys.orig_argv,
            cwd=os.getcwd(),
            pid=os.getpid(),
            executed_sources=[reference(ROOT / p) for p in OWNED],
        ),
        preconditions=dict(
            no_model_loads=True,
            no_device_probes=True,
            hard_cap_s=4800,
            planned_budget_s=4200,
            run_date="20261108",
            original_authority_sha256=authority.PIN,
        ),
    )
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """Thirteen accounted outcomes can be ready while scientific branches stay blocked."""
    owned = validation["passed"] and validation["coverage"].get("complete") is True
    ready = bool(
        owned
        and work["contract"]["ready"]
        and len(work["slots"]) == 12
        and all(s["accounted"] for s in work["slots"])
    )
    verdict = "blocked" if owned else "disqualified"
    rows, branches, continuations, retirements, denominators = [], [], [], [], []
    for slot in work["slots"]:
        n, producer, task = slot["experiment_id"], slot["producer"], slot["task"]
        rows.append(
            dict(
                unit_id=slot["task_id"],
                cluster_id="administrative_slots",
                arm=slot["disposition"],
                seed=7638929,
                metric=int(slot["accounted"]),
                denominator=1,
                status="completed" if slot["accounted"] else "censored",
                missing_reason=None if slot["accounted"] else "external_evidence_unavailable",
                experiment_id=n,
                disposition=slot["disposition"],
                prior_verdict=producer.get("verdict_class", "blocked"),
                original_honest_verdict=producer.get("honest_verdict"),
                primary_reference=slot["primary_reference"],
            )
        )
        branches.append(
            dict(
                task_id=slot["task_id"],
                disposition=slot["disposition"],
                verdict_class=producer.get("verdict_class", "blocked"),
                gate_check_summary=slot["gate_check_summary"],
                original_assertions={
                    k: producer.get(k)
                    for k in (
                        "honest_verdict",
                        "verdict_class",
                        "required_checks_passed",
                        "acceptance_gates",
                        "gate_check_summary",
                        "model_invocation_counts",
                        "numerical_training_calls",
                        "inference_substrate",
                        "inference_substrate_class",
                        "MODEL_SPECS",
                        "verifier_is_oracle",
                    )
                },
                original_producer_reference=slot["primary_reference"],
                independent_generalization_score=0,
            )
        )
        mechanism = NEXT[n]
        continuations.append(
            dict(
                task_id=slot["task_id"],
                continuation="repair_owned_controls"
                if slot["disposition"] == "disqualified_primary"
                else "require_changed_external_evidence"
                if slot["gate_check_summary"]
                else "retain_qualified_scope",
                changed_mechanism_requirement=mechanism,
                original_changed_mechanisms=[
                    p["addressed_by"] for p in task.get("prior_failures", [])
                ],
                activation_authorized=False,
            )
        )
        retirements.append(
            dict(
                task_id=slot["task_id"],
                prior_verdicts=task.get("prior_failures", []),
                current_verdict=producer.get("verdict_class", "blocked"),
                exact_scope=task.get("title", task["id"]),
                required_changed_mechanism=mechanism,
                original_retirement_contract=task.get("prior_failures", []),
                broader_hypothesis_retired=False,
                exclusion_manifest_edit_authorized=False,
            )
        )
        denominators.append(
            dict(
                task_id=slot["task_id"],
                graph_seeds=producer.get("independent_count", 0) if n == 8918 else 0,
                development_streams=8 if n == 8919 and producer else 0,
                learning_streams=producer.get("independent_count", 0) if n in (8922, 8925) else 0,
                certificate_draws=0,
                source_clusters=producer.get("source_document_coverage", {}).get("completed", 0),
                timing_worlds=producer.get("independent_count", 0) if n in (8921, 8926) else 0,
                producer_counts={
                    k: producer.get(k)
                    for k in (
                        "intended_count",
                        "completed_count",
                        "failed_count",
                        "censored_count",
                        "excluded_count",
                        "independent_count",
                    )
                },
                scope="branch-local units; related learning audit reuses streams; never sum these denominators",
            )
        )
    rows.append(
        dict(
            unit_id=TASK,
            cluster_id="administrative_slots",
            arm="self_slot",
            seed=7638929,
            metric=int(ready),
            denominator=1,
            status="completed",
            missing_reason=None,
            experiment_id=8929,
            disposition="self_slot",
        )
    )
    receipt = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            operation="authenticate and reconcile retained measurements"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        phase_spans=[
            dict(
                phase="reconciliation_and_checks",
                start_s=0,
                end_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
            )
        ],
    )
    gates = work["publication_gates"]
    value = dict(
        experiment_id=8929,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261108",
        honest_verdict="complete_" + verdict + "_v763_capstone",
        verdict_class=verdict,
        rows=rows,
        outcome_rows=rows,
        branch_conclusions=branches,
        continuation_rows=continuations,
        retirement_recommendations=retirements,
        independent_denominator_rows=denominators,
        reconciliation_ready_score=int(ready),
        scientific_reductions=work["reductions"],
        intended_count=13,
        completed_count=sum(s["accounted"] for s in work["slots"]) + 1,
        failed_count=0,
        censored_count=sum(not s["accounted"] for s in work["slots"]),
        excluded_count=0,
        independent_count=0,
        independent_generalization_score=0,
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        execution_venue="host",
        verifier_is_oracle=True,
        exposure_scope="Retained BFS final-answer oracle, constructed learning streams, dependent natural annotations and historical CPU clocks; no localization or independent generalization",
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        current_work_receipt=receipt,
        preconditions_checked=work["preconditions"],
        invocation=work["invocation"],
        random_seed=7638929,
        source_artifact_hashes=[
            s["primary_reference"] for s in work["slots"] if s["primary_reference"]
        ],
        code_config_hashes=work["code"],
        raw_shard_hashes=work["references"],
        cited_upstream_artifacts=[
            s["primary_reference"] for s in work["slots"] if s["primary_reference"]
        ],
        gate_check_summary=[d for s in work["slots"] for d in s["gate_check_summary"]],
        acceptance_gates=dict(
            complete_thirteen_slot_accounting=ready,
            scientific_benefit=False,
            cold_replay_required=True,
            owned_statement_percent=100,
            zero_exclusions=True,
        ),
        required_checks_passed=bool(owned),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        validation_state=validation,
        work_reference=reference(Path(work["raw"]) / "work.json"),
        flagged_adversarial=False,
        publication_gate_rows=[
            dict(
                gate=k,
                historical_result=v,
                scope="historical FoVer only",
                milestone_independent_publication_evidence=False,
            )
            for k, v in gates["gates"].items()
        ],
        paper_ready=gates["paper_ready"],
        paper_ready_scope="Historical FoVer only",
        milestone_paper_ready=False,
        milestone_unmet_gates=[
            "independent_natural_truth",
            "nine_bound_future_learning",
            "qualified_label_demand_evaluation",
            "qualified_observer_and_request_attribution",
        ],
        unmet_gates=gates["unmet_gates"],
        methodology_note="Unconditional thirteen-slot reconciliation. Pilot labels never train a detector. Administrative completeness does not establish scientific benefit. ARC remains suspended; no roadmap activation, external publication or exclusion edit.",
        repository_health=[
            r for r in validation["receipts"] if r.get("scope") == "repository_health"
        ],
    )
    value["field_principles"] = {
        k: "Preserve original byte identity and branch scope for "
        + k
        + "; administrative completion is not scientific success."
        for k in [*value, "reproducibility_checksum", "field_principles"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild claims from pinned producers so self-consistent rehashing cannot qualify drift."""
    value = json.loads(path.read_bytes())
    science.require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    expected = build(work, value["validation_state"])
    science.require(value == expected, "artifact_reduction_drift")
    for i, ref in enumerate(work["references"] + work["code"]):
        science.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
        if i % 500 == 0:
            progress("replay_authentication", i, len(work["references"]) + len(work["code"]) - i)
    evidence = {}
    for slot in work["slots"]:
        n, ref = slot["experiment_id"], slot["primary_reference"]
        if ref:
            science.require(ref["sha256"] == "sha256:" + PINS[n][1], "primary_pin")
            science.require(load(ref) == slot["producer"], "original_producer_assertions")
            evidence[n] = slot["producer"]
    if 8917 in evidence:
        original = authority.authority(Path(work["raw"]) / "authority")
        science.require(
            original == work["contract"]
            and original["tasks"] == evidence[8917]["full_task_objects"],
            "authority_drift",
        )
    for index, slot in enumerate(work["slots"]):
        if work["contract"]["tasks"]:
            task = work["contract"]["tasks"][index]
            science.require(
                slot["task"] == task
                and slot["task_id"] == task["id"]
                and slot["experiment_id"] == 8917 + index,
                "ordered_slot_authority",
            )
        n = slot["experiment_id"]
        valid = n in evidence and (
            scheduler(evidence[n], n, slot["task"], evidence)
            if n in (8923, 8927)
            else admission(evidence[n], n, slot["task_id"]) in (None, "disqualified_primary")
        )
        science.require(slot["accounted"] == valid, "accounting_drift")
    measured = science.reduce(
        {
            n: v
            for n, v in evidence.items()
            if n not in (8923, 8927)
            and any(s["experiment_id"] == n and s["accounted"] for s in work["slots"])
        },
        Path(work["raw"]),
    )
    science.require(measured == work["reductions"], "scientific_reduction_drift")
    return dict(passed=True, status="replay_passed", slots=13)


checker = namespace(checks, "capstone8929_checks")
checker.m = sys.modules[__name__]


def main(argv: list[str] | None = None) -> int:
    """Publish after owned validation; private execution tests plumbing without qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
            return 1
    output = (args.private_output or ROOT / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8929, "V763 thirteen-slot capstone", output.name, repo_root=ROOT, seed=7638929
    )
    progress("reconciliation_before", 0, 13)
    with template.phase("authenticate_and_reduce"):
        work = collect(raw, sources=args.sources)
    progress("owned_validation_before", 0, 1)
    validation = dict(passed=False, coverage=dict(complete=False), receipts=[])
    if not args.private_output:
        validation = checker.validate(raw / "validation")
    progress("owned_validation_after", 1, 0)
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    value = build(work, validation)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        progress("terminal_checks_before", 0, 3)
        terminal = checker.terminal(candidate, raw)
        validation["receipts"] += terminal["receipts"]
        validation["passed"] &= terminal["passed"]
        work["ended_ns"] = time.monotonic_ns()
        work["references"] += [reference(p) for p in (raw / "validation").rglob("*") if p.is_file()]
        work["references"] += [reference(p) for p in (raw / "terminal").rglob("*") if p.is_file()]
        atomic_json(raw / "work.json", work)
        value = build(work, validation)
        progress("terminal_checks_after", 3, 0)
    science.require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4200, "fixed_deadline")
    publish_primary(output, value, replay)
    progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0
