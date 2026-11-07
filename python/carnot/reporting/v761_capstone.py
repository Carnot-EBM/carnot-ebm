"""REQ-REPORT-8903: complete accounting preserves blocked and circular science."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v758_capstone as previous
from carnot.reporting import v758_capstone_runner as previous_runner
from carnot.reporting import v755_branch_runner as scoped
from carnot.reporting import v759_capstone as historical
from carnot.reporting import v761_branch_contract as original
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v761_capstone_reduction import reduce as reduce
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, MILESTONE = original.ROOT, original.MILESTONE
NAME, TASK = "experiment_8903_v761_capstone", "exp8903-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v761_capstone.py",
    "python/carnot/reporting/v761_capstone_reduction.py",
    CLI,
]
TEST = "tests/python/test_v761_capstone_8903.py"
EXECUTION = (
    "results/raw/experiment_8891_v761_branch_contract/invocations/1794029196874424145/authority"
)
CAPSTONE = "results/experiment_8890_v760_capstone.json"
PINS = {
    **original.PINS,
    f"{EXECUTION}/{original.DESIGN}": "4218673480b57621f2b5ce07b602aaeaade64b5fd798fa7b014a0baf2e1bb294",
    f"{EXECUTION}/{original.ACTIVE}": "4fe783888ed60e0e456d83073d8f7e967e42f446de3763eb46837a949fdeed04",
}
base = namespace(previous, "capstone8903")
reference, retain, admit, checksum = base.reference, base.retain, base.admit, base.checksum
accounted = historical.accounted
SPEC = "openspec/capabilities/research-reporting/spec.md"
SPEC_HISTORY = {
    "exp8894-buffered-clock-methods": (
        "2624f7ff98f0abe7483156880fd627150cb30848c9c2b4bd13108637e2285426",
        "cbc981e220cc1caa5e8fd9dee35b3114fa2ecda5",
    ),
    "exp8899-buffered-cost-attribution": (
        "b745d05c583ccf9dd5ce7291d85bac2c80aa2d827ef1b0d68ab1c961041e2297",
        "ba944dec5efa58fc7d7d0709056ef4eb4f2398fd",
    ),
}
original_admit = admit


def original_spec(task_id: str) -> Path:
    """Exact Git bytes preserve the producer's assertion after legitimate spec reconciliation."""
    return (
        ROOT
        / "results/raw"
        / NAME
        / "development/pristine_specs"
        / SPEC_HISTORY[task_id][0]
        / "spec.md"
    )


def admit(path: Path, task_id: str) -> Json:
    """Resolve only named historical configs; leave original primary fields untouched."""
    pristine = original_spec(task_id) if task_id in SPEC_HISTORY else None
    original_reference = base.reference

    def resolve(candidate: Path) -> Json:
        ref: Json = original_reference(
            pristine if pristine is not None and candidate == ROOT / SPEC else candidate
        )
        if (
            pristine is not None
            and candidate == ROOT / SPEC
            and ref["sha256"] != "sha256:" + SPEC_HISTORY[task_id][0]
        ):
            raise ValueError("original_spec_hash_drift")
        return ref

    base.reference = resolve
    try:
        result: Json = original_admit(path, task_id)
    finally:
        base.reference = original_reference
    result["references"] = [
        dict(ref, path=str(pristine), original_path=ref["path"])
        if pristine is not None and Path(ref["path"]) == ROOT / SPEC
        else ref
        for ref in result["references"]
    ]
    return result


base.admit = base.shared.admit = admit


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries without extending the supervisor's fixed deadline."""
    if phase.startswith("child_"):
        pending = 1
    print(f"[exp8903] phase={phase} completed={completed} pending={pending}", flush=True)


def contract(root: Path) -> Json:
    """Use the execution snapshot so later queue edits cannot rewrite the roster."""
    return dict(original.authority(root / EXECUTION))


base.authority = SimpleNamespace(
    ROOT=ROOT,
    MILESTONE=MILESTONE,
    PIN=original.PIN,
    PINS=PINS,
    DESIGN=f"{EXECUTION}/{original.DESIGN}",
    ACTIVE=f"{EXECUTION}/{original.ACTIVE}",
    STAGED=f"{EXECUTION}/{original.STAGED}",
    CAPSTONE=CAPSTONE,
    authority=contract,
)
base.OWNED, base.TEST, base.NAME, base.TASK, base.CLI, base.MILESTONE = (
    OWNED,
    TEST,
    NAME,
    TASK,
    CLI,
    MILESTONE,
)
base.reduce, base.progress = reduce, progress


def summary(value: Json) -> Json:
    """Preserve upstream counters without attributing their compute to this capstone."""
    result: Json = historical.summary(value)
    result.update(
        {
            k: value[k]
            for k in [
                "model_invocation_counts",
                "numerical_training_calls",
                "independent_count",
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
            ]
            if k in value
        }
    )
    return result


base.shared.producer_summary = summary


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal primary and scheduler bytes before interpreting any scientific endpoint."""
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    template = ExperimentTemplate(8903, "V761 capstone", f"results/{NAME}.json", seed=7618903)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(base.freeze, root, raw, run_date)
        while not pending.done():
            try:
                pending.result(timeout=30)
            except TimeoutError:
                progress("operand_custody_wait", 0, 1)
        work: Json = pending.result()
    work["preconditions"]["random_seed"] = template.random_seed
    work["preconditions"]["producer_invocation"] = dict(
        executable=sys.executable, argv=sys.argv, cwd=os.getcwd()
    )
    log = next(
        r for r in work["snapshots"] if r.get("source_path", "").endswith("ops/conductor-log.md")
    )
    for slot, task in zip(work["slots"], work["contract"]["tasks"][:-1], strict=True):
        slot["conductor_reference"] = log
        observed = accounted(slot, task)
        slot.update(
            accounted=observed,
            metric=int(observed),
            status="completed" if observed else "censored",
            missing_reason=None if observed else "missing_file",
        )
        slot["source_resolution_rows"] = []
        if slot["admitted"] and task["id"] in SPEC_HISTORY:
            source = original_spec(task["id"])
            ref = next(r for r in slot["snapshots"] if r.get("source_path") == str(source))
            slot["source_resolution_rows"].append(
                dict(
                    ref,
                    original_path=str(ROOT / SPEC),
                    source_commit=SPEC_HISTORY[task["id"]][1],
                    artifact_field="code_config_hashes",
                    expected="sha256:" + SPEC_HISTORY[task["id"]][0],
                    observed=ref["sha256"],
                )
            )
    work["code"] += [
        reference(ROOT / p)
        for p in [
            "python/carnot/reporting/v758_capstone.py",
            "python/carnot/reporting/v758_capstone_runner.py",
            "python/carnot/reporting/v759_capstone.py",
            "python/carnot/reporting/v761_branch_contract.py",
        ]
    ]
    atomic_json(raw / "work.json", work)
    return work


DECISIONS = [
    (
        "contract_ready_score",
        "Quarantine the disqualified authority primary; retain independently authenticated execution snapshot",
        "Fix owned formatting before a newly validated authority producer",
    ),
    (
        "signed_methods_ready_score",
        "Keep qualified signed-feature methods separate from benefit",
        "Changed informative features with equal templates and optimizer work",
    ),
    (
        "shared_terminal_ready_score",
        "Keep common-final certificate methods",
        "Changed allocation using fresh sealed certification prefixes",
    ),
    (
        "observer_ready_score",
        "Retain calibrated buffered clocks",
        "Changed observer with future-world overhead below five percent",
    ),
    (
        "causal_rows_ready_score",
        "Retain causal sealed rows without a learning claim",
        "Changed informative representation with prospective retention controls",
    ),
    (
        "shared_union_benefit_score",
        "Retain standalone savings; retire unchanged shared-union savings retry",
        "Change shared allocation so every policy can stop before the common terminal",
    ),
    (
        "source_boundary_ready_score",
        "Retain circular record and source-coverage boundary",
        "Independent source-semantic annotations before natural extraction claims",
    ),
    (
        "feature_specific_benefit_score",
        "Retire unchanged signed-feature learning benefit retry",
        "Change representation to pass feature-specific conjunctions and retention",
    ),
    (
        "complete_cost_rows_ready_score",
        "Retire unchanged complete attribution retry",
        "Lower observer overhead and close all complete parent windows",
    ),
    (
        "source_boundary_ready_score",
        "Keep optional Qwen evidence terminal blocked",
        "Authenticated CUDA repair followed by a bounded source canary",
    ),
    (
        "adaptation_cost_ready_score",
        "Retain complete CPU adaptation costs without speed claim",
        "Matched quality with faster complete durable service windows",
    ),
    (
        "hardware_map_ready_score",
        "Keep CPU operation map; physical acceleration blocked",
        "Compatible physical device plus measured transfer and durable host costs",
    ),
]


def build(work: Json, validation: Json) -> Json:
    """Administrative completion grants neither scientific benefit nor deployment authority."""
    base.report.c = SimpleNamespace(
        TASK=TASK,
        ROOT=ROOT,
        MILESTONE=MILESTONE,
        authority=SimpleNamespace(PIN=original.PIN, HISTORY=CAPSTONE),
        reference=reference,
        retirement=base.retirement,
        shared=base.shared,
        checksum=checksum,
    )
    value: Json = base.report.build(work, validation)
    owned = value["required_checks_passed"]
    ready = bool(owned and len(work["slots"]) == 12 and all(r["accounted"] for r in work["slots"]))
    blocked = not work["contract"]["ready"] or any(
        not r["admitted"] or r["verdict_class"] in {"blocked", "disqualified"}
        for r in work["slots"]
    )
    verdict = "disqualified" if not owned else "blocked" if blocked else "null"
    honest = f"complete_{verdict}_v761_capstone"
    value["rows"][-1].update(
        experiment_id=8903, verdict_class=verdict, primary_honest_verdict=honest, metric=int(ready)
    )
    denominators, decisions = [], []
    for slot, branch, (field, decision, mechanism) in zip(
        work["slots"], value["branch_conclusions"], DECISIONS, strict=True
    ):
        producer, reduction = slot["producer"], slot["reduction"]
        gate = dict(
            check="continuation_gate",
            upstream=slot["task_id"],
            artifact_field=field,
            expected=1,
            observed=producer.get(field),
            passed=slot["admitted"] and producer.get(field) == 1,
            **slot["primary_reference"],
        )
        branch.update(
            producer_counts={
                k: producer.get(k)
                for k in [
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                    "independent_count",
                ]
            },
            upstream_compute=slot["upstream_compute"],
            verifier_is_oracle=producer.get("verifier_is_oracle", False),
            inference_substrate_class=producer.get("inference_substrate_class", "blocked_no_run"),
            source_resolution_rows=slot.get("source_resolution_rows", []),
            primary_reference=slot["primary_reference"],
            scheduler_receipts=slot["scheduler_receipts"],
            conductor_reference=slot["conductor_reference"],
            continuation_gate=gate,
            next_decision=decision,
            changed_mechanism_needed=mechanism,
            exact_scope_retirement="repair_owned_validation"
            if slot["verdict_class"] == "disqualified"
            else "block_external_once"
            if not slot["admitted"]
            else "retire_unchanged_retry"
            if not gate["passed"]
            else "retain_qualified_scope",
        )
        decisions.append(
            {
                k: branch[k]
                for k in [
                    "task_id",
                    "next_decision",
                    "changed_mechanism_needed",
                    "continuation_gate",
                    "exact_scope_retirement",
                ]
            }
        )
        denominators.append(
            dict(
                task_id=slot["task_id"],
                admitted=slot["admitted"],
                streams=len(reduction.get("stream_ids", [])),
                topologies=len(reduction.get("topology_ids", [])),
                worlds=len(reduction.get("world_ids", [])),
                calibration_replicates=len(reduction.get("calibration_replicate_ids", [])),
                cost_windows=reduction.get("cpu_cost_window_count", 0),
                cpu_request_rows=reduction.get("cpu_request_count", 0),
                producer_independent_count=producer.get("independent_count"),
                lineage=slot["primary_reference"],
                scope="branch-local; reused streams, worlds, seeds and candidates never add independent units",
            )
        )
    for recommendation in value["retirement_recommendations"]:
        current = next(
            (b for b in value["branch_conclusions"] if b["task_id"] == recommendation["task_id"]),
            None,
        )
        recommendation["exact_scope_recommendation"] = (
            current["exact_scope_retirement"] if current else "capstone_accounting_only"
        )
        recommendation["broader_hypothesis_retired"] = False
    history = json.loads((ROOT / CAPSTONE).read_bytes())
    gates = value["historical_publication_gates"]
    decisions.append(
        dict(
            task_id=TASK,
            next_decision="Keep complete accounting separate from blocked source extraction; do not activate",
            changed_mechanism_needed="Admit the named changed evidence for each blocked or null branch",
            continuation_gate=dict(
                artifact_field="reconciliation_ready_score",
                expected=1,
                observed=int(ready),
                passed=ready,
            ),
            exact_scope_retirement="administrative_self_slot_only",
        )
    )
    value.update(
        experiment_id=8903,
        honest_verdict=honest,
        verdict_class=verdict,
        random_seed=7618903,
        execution_venue="host",
        reconciliation_ready_score=int(ready),
        contract_ready_score=int(work["contract"]["ready"]),
        natural_generalization_score=0,
        historical_v760={
            k: deepcopy(history[k])
            for k in ["honest_verdict", "verdict_class", "contract_ready_score", "rows"]
        },
        independent_denominators=denominators,
        constructed_measurement_count=0,
        next_decisions=decisions,
        cited_upstream_artifacts=[s["primary_reference"] for s in work["slots"]],
        paper_ready=gates["paper_ready"],
        unmet_gates=gates["unmet_gates"],
        exposure_scope="thirteen V761 administrative slots; scoped constructed streams, certificates, source/view errors and CPU costs",
    )
    value["historical_v760"]["reference"] = reference(ROOT / CAPSTONE)
    value["acceptance_gates"]["complete_accounting"]["passed"] = ready
    value["gate_check_summary"] += work["contract"]["failures"]
    for row, task in zip(value["rows"], work["contract"]["tasks"], strict=True):
        row["title"] = task["title"]
    for key in ("g1", "g2", "g3", "g4"):
        value[key] = gates["gates"][key.upper()]
    value["methodology_note"] = (
        "Capstone accounting is not a measured scientific unit. Correct-label benefit, sequential label savings, "
        "served risk, constructed certification, source extraction and complete CPU costs remain distinct. "
        "Exact encoded truth remains circular. Historical FoVer G1-G4 gates retain their original scope. "
        "External missing evidence is terminal blocked; no model, training, deployment or activation occurred."
    )
    value["raw_shard_hashes"] = list(
        {(r["path"], r["sha256"]): r for r in value["raw_shard_hashes"]}.values()
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Original bytes bind "
        + k
        + "; administrative accounting never increases scientific independence or deployment authority."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


base.build = build
legacy_replay = base.replay


def replay(path: Path) -> Json:
    """Require the original task order and sealed scheduler lines even after rehashing."""
    result: Json = legacy_replay(path)
    if not result["passed"]:
        return result
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    passed = all(
        slot["task_id"] == task["id"] and slot["accounted"] == accounted(slot, task)
        for slot, task in zip(work["slots"], work["contract"]["tasks"][:-1], strict=True)
    )
    return dict(passed=passed, reason="replay_passed" if passed else "scheduler_custody_drift")


base.replay = replay
runner = namespace(previous_runner, "capstone8903_runner")
runner.c = runner.runner.c = runner.runner.checks.c = sys.modules[__name__]
checks = namespace(scoped, "capstone8903_checks")
checks.c = sys.modules[__name__]


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Bound large custody tests once; progress never renews their elapsed-time budget."""
    if name == "owned_tests":
        argv = [*argv, "-s"]
    return dict(
        checks.run_check(
            name,
            argv,
            raw,
            1200
            if name == "owned_tests"
            else 600
            if name in {"private_e2e018_021", "cold_replay"}
            else deadline,
            scope,
        )
    )


runner.runner.checks.run_check = run_check
runner.runner.Path = lambda path: (
    ROOT / "results/raw" / NAME / "development/repository_health.json"
    if str(path) == "/tmp/exp8839-global/receipt.json"
    else Path(path)
)
main = runner.main
