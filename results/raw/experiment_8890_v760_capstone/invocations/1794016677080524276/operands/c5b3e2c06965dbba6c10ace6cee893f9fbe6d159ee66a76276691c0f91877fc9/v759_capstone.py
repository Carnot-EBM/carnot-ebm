"""REQ-REPORT-8877: reuse historical custody while preserving twelve distinct outcomes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from carnot.reporting import v758_capstone as previous
from carnot.reporting import v758_capstone_runner as previous_runner
from carnot.reporting import v759_branch_contract as original
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, MILESTONE = original.ROOT, original.MILESTONE
NAME, TASK = "experiment_8877_v759_capstone", "exp8877-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v759_capstone.py", CLI]
TEST = "tests/python/test_v759_capstone_8877.py"
EXECUTION = (
    "results/raw/experiment_8866_v759_branch_contract/invocations/1793960942212453784/authority"
)
PINS = {
    **original.PINS,
    f"{EXECUTION}/{original.DESIGN}": "c0550b31623ec711c5cb833b1c6fcb8aa986ae77ce9cb03293dbb30ca68d2407",
    f"{EXECUTION}/{original.ACTIVE}": "e1eb2c3ca1487fa4efc9acef930c1ecb45ed497d3fc8de87da597290b59b02ad",
}
base = namespace(previous, "capstone8877")
reference, retain, admit, checksum = (
    base.reference,
    base.retain,
    base.admit,
    base.checksum,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show actual producer identity and completed work without extending deadlines."""
    if phase in {"preconditions_before", "slot_before", "slot_after"}:
        pending = max(0, pending - 1)
    if phase in {"custody_after", "publication_after"}:
        completed -= 1
    print(f"[exp8877] phase={phase} completed={completed} pending={pending}", flush=True)


def contract(root: Path) -> Json:
    """Read original execution bytes so a later roadmap cannot rewrite this task."""
    result: Json = original.authority(root / EXECUTION)
    for failure in result["failures"]:
        failure["path"] = str(ROOT / EXECUTION / original.DESIGN)
    return result


def reduce(value: Json) -> Json:
    """Rebuild endpoints from raw rows and keep each statistical unit in its own scope."""
    n, rows = value["experiment_id"], value.get("rows", [])
    progress(f"reduction_{n}_before", 0, len(rows))
    result: Json = previous.reduce(dict(experiment_id=n, rows=rows))
    estimands: Json = {}
    worlds, replicates = set(), set()
    if n == 8868:
        trace = json.loads(Path(value["draw_manifest"]["path"]).read_bytes())
        replicates = {f"8868/rep/{r['replicate']}" for r in trace["replicates"]}
        population = json.loads(Path(value["population_manifest"]["path"]).read_bytes())
        worlds = {r["cluster_id"] for r in population["inputs"] if r["split"] == "population"}
    if n == 8870:
        from carnot.reporting.conditional_risk_evaluation_8870 import summarize

        raw = json.loads(Path(value["measurement_manifest"]["path"]).read_bytes())
        if raw["rows"] != rows:
            raise ValueError("certificate_rows_drift")
        primary = [r for r in rows if r["scenario"] != "evidence_corruption"]
        if raw["group_count_rows"] != value["group_count_rows"]:
            raise ValueError("group_rows_drift")
        estimands = summarize(primary, raw["group_count_rows"])
        replicates = {f"8868/rep/{r['replicate']}" for r in primary}
    if n in {8871, 8874}:
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        if work["rows"] != rows:
            raise ValueError("cost_rows_drift")
        worlds = {f"8871/{r['cluster_id']}" for r in rows}
        if n == 8874:
            from carnot.reporting.size_aware_complete_cost_8874 import reduce as cost

            estimands = cost(rows, value["installation_accounting"]["installation_s"])
    for key, observed in estimands.items():
        if key in value and value[key] != observed:
            raise ValueError("registered_estimand_drift:" + key)
    result.update(
        verdict_class=value.get("verdict_class", "blocked"),
        verifier_is_oracle=value.get("verifier_is_oracle", False),
        registered_estimands=estimands,
        world_ids=sorted(worlds),
        calibration_replicate_ids=sorted(replicates),
        constructed_measurement_count=len(rows) if n in {8868, 8870, 8871} else 0,
        cpu_request_count=len(rows) if n == 8874 else 0,
    )
    progress(f"reduction_{n}_after", len(rows), 0)
    return result


base.authority = SimpleNamespace(
    ROOT=ROOT,
    MILESTONE=MILESTONE,
    PIN=original.PIN,
    PINS=PINS,
    DESIGN=f"{EXECUTION}/{original.DESIGN}",
    ACTIVE=f"{EXECUTION}/{original.ACTIVE}",
    STAGED=f"{EXECUTION}/{original.STAGED}",
    CAPSTONE=original.CAPSTONE,
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
original_summary = base.shared.producer_summary


def summary(value: Json) -> Json:
    """Scheduler diagnostic strings cannot be consumed as scientific gate objects."""
    result: Json = original_summary(value)
    if not isinstance(result.get("gate_check_summary", []), list):
        result["gate_check_summary"] = []
    return result


base.shared.producer_summary = summary


def accounted(slot: Json, task: Json) -> bool:
    """A terminal skip needs original task identity and preserved scheduler bytes."""
    if slot["producer"].get("task_id") == task["id"]:
        return True
    for receipt in slot["scheduler_receipts"]:
        value = json.loads(Path(receipt["reference"]["path"]).read_bytes())
        if (
            value.get("blocked_at_layer") == "conductor_pre_gate"
            and value.get("title") == task["title"]
        ):
            return True
    log = Path(slot["conductor_reference"]["path"])
    lines = log.read_text().splitlines() if log.is_file() else []
    return bool(slot["conductor_lines"]) and all(
        line in lines
        and "GATE_BLOCK" in line
        and task["title"].startswith(line.split("|")[2].strip())
        for line in slot["conductor_lines"]
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze authenticated producers and exact scheduler receipts before interpreting them."""
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
            conductor_reference=log,
        )
    work["code"] += [
        reference(ROOT / p)
        for p in [
            "python/carnot/reporting/v758_capstone.py",
            "python/carnot/reporting/v758_capstone_reduction.py",
            "python/carnot/reporting/v758_capstone_runner.py",
            "python/carnot/reporting/v759_branch_contract.py",
        ]
    ]
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json, validation: Json) -> Json:
    """Complete accounting grants no scientific benefit or deployment permission."""
    base.report.c = SimpleNamespace(
        TASK=TASK,
        ROOT=ROOT,
        MILESTONE=MILESTONE,
        authority=SimpleNamespace(PIN=original.PIN, HISTORY=original.CAPSTONE),
        reference=reference,
        retirement=base.retirement,
        shared=base.shared,
        checksum=checksum,
    )
    value: Json = base.report.build(work, validation)
    owned = value["required_checks_passed"]
    ready = bool(owned and len(work["slots"]) == 11 and all(r["accounted"] for r in work["slots"]))
    blocked = not work["contract"]["ready"] or any(
        not r["admitted"] or r["verdict_class"] in {"blocked", "disqualified"}
        for r in work["slots"]
    )
    verdict = "disqualified" if not owned else "blocked" if blocked else "null"
    honest = f"complete_{verdict}_v759_capstone"
    value["rows"][-1].update(
        experiment_id=8877, verdict_class=verdict, primary_honest_verdict=honest, metric=int(ready)
    )
    decisions = [
        (
            "Keep CPU work independent; original contract and CUDA remain blocked",
            "Original complete design/table/JSON and dated causal CUDA repair with actual offload",
        ),
        (
            "Quarantine the disqualified label diagnosis",
            "Correct the owned validation failure before interpreting changed label controls",
        ),
        (
            "Keep fixed-group certificate methods separate from deployment utility",
            "Fresh disjoint certification after a changed selection rule",
        ),
        (
            "Keep matched learning methods blocked",
            "Admitted label diagnosis and query-only, equal-template controls",
        ),
        (
            "Retain conditional utility and pooled parity without robustness advantage",
            "Positive registered robustness contrast on new independent calibration draws",
        ),
        (
            "Keep exact structured joins opt-in; no learned semantic claim",
            "Oracle-distinct natural inputs and matched complete deployment costs",
        ),
        (
            "Keep continuous label-learning unmeasured",
            "Admitted matched methods and sealed future predictions with retention controls",
        ),
        (
            "Keep correct-label specificity and retention unmeasured",
            "Admitted streams with correct-label versus prefix-shuffled paired contrasts",
        ),
        (
            "Retain size-aware installed benefit null",
            "Complete mixture speed interval above one and size8/16 non-regression",
        ),
        (
            "Keep Qwen fidelity blocked; do not retry unchanged runtime",
            "Authenticated changed CUDA and installed readiness before the bounded canary",
        ),
        (
            "Keep CPU mapping; no hardware deployment or purchase",
            "Supported parity and measured disjoint host/transfer/device/durable fractions",
        ),
    ]
    for slot, branch, (decision, evidence) in zip(
        work["slots"], value["branch_conclusions"], decisions, strict=True
    ):
        branch.update(
            verifier_is_oracle=slot["producer"].get("verifier_is_oracle", False),
            primary_reference=slot["primary_reference"],
            next_decision=decision,
            evidence_needed_to_reopen=evidence,
            scheduler_receipts=slot["scheduler_receipts"],
            conductor_reference=slot["conductor_reference"],
        )
    for recommendation in value["retirement_recommendations"]:
        ref = recommendation["prior_reference"]
        earlier = json.loads(Path(ref["path"]).read_bytes()) if ref.get("sha256") else {}
        current = next(r for r in value["rows"] if r["task_id"] == recommendation["task_id"])
        authenticated = bool(
            ref.get("sha256")
            and reference(Path(ref["path"]))["sha256"] == ref["sha256"]
            and earlier.get("honest_verdict") == recommendation["prior_verdict"]
            and recommendation["prior_verdict"] is not None
        )
        recommendation.update(
            prior_verdict_class=earlier.get("verdict_class"),
            prior_verifier_is_oracle=earlier.get("verifier_is_oracle"),
            verdict_class=current["verdict_class"],
            verifier_is_oracle=next(
                (
                    s["producer"].get("verifier_is_oracle", False)
                    for s in work["slots"]
                    if s["task_id"] == current["task_id"]
                ),
                False,
            ),
        )
        recommendation["prior_authenticated"] = authenticated
        recommendation["retire_exact_scope"] &= authenticated
        recommendation["evidence_needed_to_reopen"] = next(
            (
                b["evidence_needed_to_reopen"]
                for b in value["branch_conclusions"]
                if b["task_id"] == recommendation["task_id"]
            ),
            "Original complete authority and all twelve authenticated outcomes",
        )
    gates = value["historical_publication_gates"]
    value.update(
        experiment_id=8877,
        honest_verdict=honest,
        verdict_class=verdict,
        random_seed=7598877,
        execution_venue="host",
        intended_count=12,
        reconciliation_ready_score=int(ready),
        contract_ready_score=int(work["contract"]["ready"]),
        historical_v758=dict(
            reference=reference(ROOT / original.CAPSTONE),
            honest_verdict=work["historical"]["honest_verdict"],
            verdict_class=work["historical"]["verdict_class"],
            rows=deepcopy(json.loads((ROOT / original.CAPSTONE).read_bytes())["rows"]),
        ),
        exposure_scope="twelve V759 slots; fixed constructed groups, oracle joins and installed Python CPU costs",
        paper_ready=gates["paper_ready"],
        unmet_gates=gates["unmet_gates"],
        next_decisions=[
            dict(
                task_id=b["task_id"],
                decision=b["next_decision"],
                evidence_needed_to_reopen=b["evidence_needed_to_reopen"],
            )
            for b in value["branch_conclusions"]
        ],
        independent_calibration_replicate_count=len(
            {
                r
                for s in work["slots"]
                if s["admitted"]
                for r in s["reduction"]["calibration_replicate_ids"]
            }
        ),
        independent_world_count=len(
            {r for s in work["slots"] if s["admitted"] for r in s["reduction"]["world_ids"]}
        ),
        independent_stream_count=0,
    )
    value["acceptance_gates"]["complete_accounting"].update(passed=ready, threshold=12)
    for row, task in zip(value["rows"], work["contract"]["tasks"], strict=True):
        row["title"] = task["title"]
    value["next_decisions"].append(
        dict(
            task_id=TASK,
            decision="Keep unresolved external scope terminal blocked; do not activate or deploy",
            evidence_needed_to_reopen="Complete original authority plus each branch's named changed evidence",
        )
    )
    value["gate_check_summary"] += work["contract"]["failures"]
    for key in ("g1", "g2", "g3", "g4"):
        value[key] = gates["gates"][key.upper()]
    value["methodology_note"] = (
        "Independent label information, structural utility, retention, certificate robustness, "
        "installed costs and bounded Qwen fidelity remain separate endpoints. Fixed-group mixtures "
        "give no guarantee under within-group corruption or natural text. Query-only exact joins "
        "are execution-grounded circular_positive evidence. Python CPU gains establish neither "
        "native10x nor physical speedup. External absence is terminal blocked. No new model loads, "
        "training, deployment, publication or activation occurred."
    )
    value["raw_shard_hashes"] = list(
        {(r["path"], r["sha256"]): r for r in value["raw_shard_hashes"]}.values()
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Original bytes bind each scoped conclusion; accounting completion does not establish science or authorize deployment.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


base.build = build
legacy_replay = base.replay


def replay(path: Path) -> Json:
    """Rehashed work still needs the original scheduler identity and exact slot roster."""
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
runner = namespace(previous_runner, "capstone8877_runner")
runner.c = runner.runner.c = runner.runner.checks.c = sys.modules[__name__]
runner.runner.Path = lambda path: (
    ROOT / "results/raw" / NAME / "development/repository_health.json"
    if str(path) == "/tmp/exp8839-global/receipt.json"
    else Path(path)
)
main = runner.main
