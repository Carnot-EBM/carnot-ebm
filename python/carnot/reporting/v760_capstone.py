"""REQ-REPORT-8890: complete accounting preserves blocked and circular science."""

from copy import deepcopy
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
from carnot.reporting import v760_branch_contract as original
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v760_capstone_reduction import reduce as reduce
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, MILESTONE = original.ROOT, original.MILESTONE
NAME, TASK = "experiment_8890_v760_capstone", "exp8890-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v760_capstone.py",
    "python/carnot/reporting/v760_capstone_reduction.py",
    CLI,
]
TEST = "tests/python/test_v760_capstone_8890.py"
EXECUTION = (
    "results/raw/experiment_8878_v760_branch_contract/invocations/1793989376413282562/authority"
)
CAPSTONE = "results/experiment_8877_v759_capstone.json"
PINS = {
    **original.PINS,
    f"{EXECUTION}/{original.DESIGN}": "972d1520b2a07c48551b1dbf6afc0a9f17bb568fb3cde4696792aef8fd2aa1f1",
    f"{EXECUTION}/{original.ACTIVE}": "4509ec1aa2c1ad4204a91cfae9f76b1ac8ba5b65f72cb776a8c02b6d4cf44761",
}
base = namespace(previous, "capstone8890")
reference, retain, admit, checksum = base.reference, base.retain, base.admit, base.checksum
accounted = historical.accounted


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries without extending the supervisor's fixed deadline."""
    if phase.startswith("child_"):
        pending = 1
    print(f"[exp8890] phase={phase} completed={completed} pending={pending}", flush=True)


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
    work: Json = base.freeze(root, raw, run_date)
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
    work["code"] += [
        reference(ROOT / p)
        for p in [
            "python/carnot/reporting/v758_capstone.py",
            "python/carnot/reporting/v758_capstone_runner.py",
            "python/carnot/reporting/v759_capstone.py",
            "python/carnot/reporting/v760_branch_contract.py",
        ]
    ]
    atomic_json(raw / "work.json", work)
    return work


DECISIONS = [
    (
        "contract_ready_score",
        "Retain complete authority; optional CUDA remains blocked",
        "Dated causal CUDA repair with measured offload",
    ),
    (
        "diagnosis_ready_score",
        "Keep qualified diagnosis separate from label benefit",
        "Changed informative feedback with independently sealed controls",
    ),
    (
        "sequential_methods_ready_score",
        "Keep finite-look certification methods",
        "Changed allocation with fresh independent certification draws",
    ),
    (
        "topology_methods_ready_score",
        "Keep matched topology methods separate from benefit",
        "Changed label information with equal template and optimizer work",
    ),
    (
        "cost_attribution_ready_score",
        "Retire unchanged attribution retry",
        "Close every non-overlapping clock and bound observer overhead at five percent",
    ),
    (
        "source_fidelity_established_score",
        "Retain circular source/view boundary findings",
        "Independent full-source annotations before extraction claims",
    ),
    (
        "independent_benefit_score",
        "Retire unchanged topology learning retry",
        "Changed informative labels with frozen future and retention contrasts",
    ),
    (
        "label_savings_score",
        "Retire unchanged sequential label-saving retry",
        "Changed stopping allocation with positive savings and safe served risk",
    ),
    (
        "label_benefit_score",
        "Retire unchanged correct-label benefit retry",
        "Changed mechanism passing all eight simultaneous benefit and retention gates",
    ),
    (
        "source_fidelity_established_score",
        "Block external Qwen scope once; no unchanged retry",
        "Authenticated CUDA repair and source canary prerequisites",
    ),
    (
        "adaptation_cost_ready_score",
        "Retain complete CPU costs without speed benefit",
        "Matched scientific quality and faster complete durable windows",
    ),
    (
        "hardware_map_ready_score",
        "Keep CPU operation map; defer physical deployment",
        "Compatible measured device, transfer and durable host costs",
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
    honest = f"complete_{verdict}_v760_capstone"
    value["rows"][-1].update(
        experiment_id=8890, verdict_class=verdict, primary_honest_verdict=honest, metric=int(ready)
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
            primary_reference=slot["primary_reference"],
            scheduler_receipts=slot["scheduler_receipts"],
            conductor_reference=slot["conductor_reference"],
            continuation_gate=gate,
            next_decision=decision,
            changed_mechanism_needed=mechanism,
            exact_scope_retirement="block_external_once"
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
        experiment_id=8890,
        honest_verdict=honest,
        verdict_class=verdict,
        random_seed=7608890,
        execution_venue="host",
        reconciliation_ready_score=int(ready),
        contract_ready_score=int(work["contract"]["ready"]),
        natural_generalization_score=0,
        historical_v759={
            k: deepcopy(history[k])
            for k in ["honest_verdict", "verdict_class", "contract_ready_score", "rows"]
        },
        independent_denominators=denominators,
        constructed_measurement_count=0,
        next_decisions=decisions,
        cited_upstream_artifacts=[s["primary_reference"] for s in work["slots"]],
        paper_ready=gates["paper_ready"],
        unmet_gates=gates["unmet_gates"],
        exposure_scope="thirteen V760 administrative slots; scoped constructed streams, certificates, source/view errors and CPU costs",
    )
    value["historical_v759"]["reference"] = reference(ROOT / CAPSTONE)
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
runner = namespace(previous_runner, "capstone8890_runner")
runner.c = runner.runner.c = runner.runner.checks.c = sys.modules[__name__]
checks = namespace(scoped, "capstone8890_checks")
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
