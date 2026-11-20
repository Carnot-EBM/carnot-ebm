"""REQ-REPORT-9096-V776: bind planning to original evidence without running science.

The older collection and validation helpers keep their existing checks. Temporary
bindings give them this experiment's paths without changing historical producers.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import scoped_provenance_9014 as validation
from carnot.reporting import v775_contract as previous
from carnot.reporting import v775_capstone_evidence as history
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, LOG = (
    previous.ROOT,
    previous.DESIGN,
    previous.ACTIVE,
    previous.STAGED,
    previous.LOG,
)
NAME, TASK, MILESTONE = "experiment_9096_v776_contract", "exp9096-contract", "2026.11.776"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v776_contract_9096.py"
OWNED = ["python/carnot/reporting/v776_contract.py", CLI]
DIRECT = previous.DIRECT
REUSED = list(
    dict.fromkeys(
        [
            *previous.REUSED,
            *previous.OWNED,
            "python/carnot/reporting/v775_capstone_evidence.py",
            "python/carnot/reporting/service_cost_9093.py",
            "python/carnot/reporting/susceptibility_9094.py",
        ]
    )
)
CAPSTONE = "results/experiment_9095_v775_capstone.json"
HUMAN = "results/experiment_9084_v775_human_methods.json"
ARCHIVE = "openspec/change-proposals/research-roadmap-v775-preserved-20261115.md"
QUEUE = "openspec/change-proposals/research-roadmap-v775-queue-preserved-20261115.yaml"
REGISTRY, FRONTIER = "ops/arc_solve_registry.yaml", "ops/arc_supervisor_refinement_ledger.json"
PINS = {
    DESIGN: "sha256:9636b61f464b350f38e2317f2e197ddc79f22b4ed4c0f34d13099c36c5e26700",
    ARCHIVE: "sha256:ce767f80004cb4b9aef48f4b49c46ce024f06e8bea251a4cf9aa4d33c631ab1b",
    QUEUE: "sha256:10caaae6c06ef27417685d999ca8ff6293e275409eb08a75746b32d99502df37",
    CAPSTONE: "sha256:e3b8e64fb3aad034ab99a10a0985b07b55522dcc1606ec3b4b80bc73e950f100",
    **{p: pin for p, pin in history.PINS.items() if p.startswith("results/")},
}
AUTHORITIES = [DESIGN, ACTIVE, LOG, "research-complete.yaml", ARCHIVE, QUEUE, REGISTRY, FRONTIER]
MODEL_SPECS: list[Json] = []
reference, checksum, check = previous.reference, previous.checksum, previous.check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase changes while the shared supervisor reports long child progress."""
    print(f"[exp9096] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def validation_binding() -> Iterator[None]:
    """Keep the proven command policy while owning only this reader's statements."""
    with patch.multiple(
        validation,
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


@contextmanager
def binding() -> Iterator[None]:
    """Reuse immutable collection without editing a historical producer or conductor."""
    with patch.multiple(
        previous,
        NAME=NAME,
        OWNED=OWNED,
        TEST=TEST,
        AUTHORITIES=AUTHORITIES,
        binding=validation_binding,
        scope_manifest=scope_manifest,
        authority=authority,
        input_pins=input_pins,
        reduce=reduce,
        progress=progress,
    ):
        yield


def scope_manifest(private: Path) -> Json:
    """Freeze exact tests, code and deadlines before any historical reduction."""
    with validation_binding():
        manifest = validation.scope_manifest(private)
    manifest.update(
        activation_commands=[],
        global_suite_command=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        claim_scope="V776 administrative contract only; scientific protocols are frozen, not executed.",
    )
    return manifest


def authority(root: Path) -> Json:
    """Compare complete objects so a matching title cannot hide a changed prompt or gate."""
    tasks, rows, keys, checks = [], [], [], []
    mode = "unavailable"
    try:
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        path = root / (ACTIVE if mode == "activated" else STAGED)
        queue = yaml.safe_load(path.read_bytes())
        Roadmap.model_validate(queue)
        tasks = queue["tasks"]
        table, machine = parse_design((root / DESIGN).read_text(), milestone=MILESTONE)
        checks = [
            check(
                root / DESIGN,
                "sha256",
                PINS[DESIGN],
                reference(root / DESIGN)["sha256"],
                "design_bytes",
            ),
            check(path, "milestone", MILESTONE, queue["milestone"], "lifecycle"),
            check(
                path, "task_count", [14] * 3, [len(tasks), len(table), len(machine)], "task_count"
            ),
        ]
        completed = {
            p[2].strip()
            for line in (root / LOG).read_text().splitlines()
            if len(p := line.split("|")) >= 4 and p[3].strip() in {"OK", "FLAGGED"}
        }
        archived = (root / "research-complete.yaml").read_text()
        titles = [t["title"][:50].strip() for t in tasks]
        earlier: dict[str, set[str]] = {}
        for i, task in enumerate(tasks):
            identity, key = task["id"], titles[i]
            predicates = dict(
                full_object=i < len(machine)
                and task in (machine[i], dict(machine[i], prompt=encode(machine[i]["prompt"]))),
                visible_table=i < len(table)
                and all(table[i][k] == task[k] for k in TABLE_FIELDS)
                and table[i]["order"] == i + 1,
                sequence=identity.startswith(f"exp{9096 + i}-"),
                gates=all(
                    set(g) == {"upstream", "artifact_field", "op", "value"}
                    and g["upstream"] in earlier
                    and g["artifact_field"] in earlier[g["upstream"]]
                    for g in task.get("gated_on", [])
                ),
                unique_key=titles.count(key) == 1,
                uncompleted_key=key not in completed,
                unused_id=re.search(rf"\bexp{9096 + i}[-_]", archived) is None
                and all(
                    identity == TASK and p.name == NAME + ".json"
                    for p in (root / "results").glob(f"experiment_{9096 + i}_*.json")
                ),
            )
            matched = all(predicates.values())
            rows.append(
                dict(
                    unit_id=identity,
                    original_qid=None,
                    group_id=identity,
                    family=identity,
                    arm="full_task_contract",
                    condition="exact_authority",
                    seed=None,
                    metric="ordered_full_object_match",
                    value=int(matched),
                    absolute_metric=int(matched),
                    numerator=int(matched),
                    denominator=1,
                    status="completed" if matched else "failed",
                    exclusion_reason=None,
                    censor_reason=None,
                    task=task,
                    checks=predicates,
                    matched=matched,
                )
            )
            keys.append(
                dict(
                    unit_id=identity,
                    dispatch_key=key,
                    passed=predicates["unique_key"] and predicates["uncompleted_key"],
                )
            )
            earlier[identity] = set(
                re.findall(
                    r"\b[A-Za-z][A-Za-z0-9_]*\b",
                    task["prompt"].split("REQUIRED ARTIFACT FIELDS:", 1)[-1],
                )
            )
        checks.append(
            check(
                path,
                "all_complete_objects",
                True,
                len(rows) == 14 and all(r["matched"] for r in rows),
                "full_contract",
            )
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        yaml.YAMLError,
    ) as error:
        checks.append(
            check(
                root / ACTIVE,
                "full_task_objects",
                "14 exact ordered objects",
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else str(error),
                "authority",
            )
        )
    return dict(
        ready=bool(checks) and all(r["passed"] for r in checks),
        tasks=tasks,
        rows=rows,
        title_key_rows=keys,
        checks=checks,
        comparison_mode=mode,
    )


def input_pins(root: Path) -> dict[str, str]:
    """Follow only authenticated historical references; never bless current producer bytes."""
    pins = dict(PINS)
    if reference(root / CAPSTONE)["sha256"] == PINS[CAPSTONE]:
        capstone = json.loads((root / CAPSTONE).read_bytes())
        ref = capstone["work_reference"]
        relative = str(Path(ref["path"]).relative_to(ROOT))
        pins[relative] = ref["sha256"]
        if reference(root / relative)["sha256"] == ref["sha256"]:
            work = json.loads((root / relative).read_bytes())
            for saved in [*work["snapshots"].values(), *work["operands"].values()]:
                pins[str(Path(saved["path"]).relative_to(ROOT))] = saved["sha256"]
    return pins


def reduce(work: Json) -> Json:
    """Recompute historical summaries while keeping absent science and scheduler skips separate."""
    root, raw = Path(work["retained"]), Path(work["evidence_raw"])
    capstone = json.loads((root / CAPSTONE).read_bytes())
    saved = root / Path(capstone["work_reference"]["path"]).relative_to(ROOT)
    historical = json.loads(saved.read_bytes())
    for ref in [*historical["snapshots"].values(), *historical["operands"].values()]:
        ref["path"] = str(root / Path(ref["path"]).relative_to(ROOT))
    science = history.reduce(historical)
    outcomes = history.outcomes(historical)
    outcomes.append(
        dict(
            capstone["task_outcome_rows"][-1],
            task=historical["tasks"][-1],
            primary_status="present",
            primary_path=CAPSTONE,
            selected_path=CAPSTONE,
            reference=reference(root / CAPSTONE),
            scheduler_honest_verdict=None,
        )
    )
    human = historical["values"]["exp9084-human-methods"]["value"]
    custody = historical["values"]["exp9083-prospective-custody"]["value"]
    text = (root / DESIGN).read_text()
    section = text.split("## Frozen scientific protocols\n", 1)[1].split("## Dependency graph", 1)[
        0
    ]
    controls = section.split("### Signal controls and fixed Qwen comparator\n", 1)[1].split(
        "### Complete costs and hardware", 1
    )[0]
    protocols: Json = dict(
        static=dict(
            protocol=human["protocol"],
            original_reference=historical["operands"][human["human_protocol_reference"]["path"]],
        ),
        online=section.split("### Continuous self-learning\n", 1)[1].split(
            "### Signal controls", 1
        )[0],
        signal_controls=controls.split("Exp9107", 1)[0],
        judge="Exp9107" + controls.split("Exp9107", 1)[1],
        timing=section.split("### Complete costs and hardware\n", 1)[1],
    )
    refs = []
    for name, protocol in protocols.items():
        target = raw / "frozen_protocols" / (name + ".json")
        value = dict(
            name=name,
            protocol=protocol["protocol"] if name == "static" else protocol,
            source_reference=protocol["original_reference"]
            if name == "static"
            else reference(root / DESIGN),
            measured=False,
        )
        if target.exists() and json.loads(target.read_bytes()) != value:
            raise ValueError("immutable_protocol_drift")
        atomic_json(target, value)
        refs.append(reference(target))
    archived = yaml.load((root / "research-complete.yaml").read_bytes(), Loader=yaml.CSafeLoader)[
        "milestones"
    ]
    registry = yaml.safe_load((root / REGISTRY).read_bytes())
    frontier = json.loads((root / FRONTIER).read_bytes())
    return dict(
        contract=authority(root),
        historical_model_provenance=capstone["historical_model_provenance"],
        previous_outcome_rows=outcomes,
        protocol_references=refs,
        preserved_results=dict(
            custody={
                k: custody[k]
                for k in ["fixture_custody_ready_score", "custody_ready_score", "honest_verdict"]
            },
            human=science["human_summary"],
            cost=science["cost_summary"],
            susceptibility=science["susceptibility_summary"],
        ),
        deferred_graph_bounds={k: science[k] for k in ["seven_bound_rows", "ten_bound_rows"]},
        closed_scope_rows=science["continuation_decisions"],
        archive_lag=dict(
            planning_archive_end=capstone["archive_lag"]["observed_archive_end"],
            observed_archive_end=archived[-1]["id"],
            completed_scheduling_frontier="2026.11.775",
            archive_lagged=archived[-1]["id"] != "2026.11.775",
            archive_modified=False,
            scientific_result_from_scheduler=False,
        ),
        arc_summary=dict(
            registry_reference=reference(root / REGISTRY),
            frontier_reference=reference(root / FRONTIER),
            registry_parsed=isinstance(registry, dict),
            frontier_parsed=isinstance(frontier, dict),
            new_gameplay_executed=False,
            new_solve_count=0,
            new_receipt_created=False,
            claim_scope="Current registry/frontier context for the capstone boundary only.",
        ),
    )


def collect(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Reuse source authentication, private storage and scope freezing before reduction."""
    with binding():
        return previous.collect(root, raw, private, run_date)


def validate(work: Json) -> Json:
    """Run frozen normal-exit checks and retain actual owned SQLite statement coverage."""
    with binding():
        return previous.validate(work)


def build(work: Json) -> Json:
    """Administrative readiness requires current checks; no outcome here measures useful learning."""
    reduction = work["reduction"]
    checks = work["preconditions_checked"]["checks"]
    blocked = not all(r["passed"] for r in checks)
    with validation_binding():
        qualified = validation.receipts_pass(work) and verify_measured(
            work["validation"]["coverage"], OWNED
        )
    ready = bool(qualified and not blocked and reduction and reduction["contract"]["ready"])
    cls = "blocked" if blocked else "null" if ready else "disqualified"
    failed = next((r["check"] for r in checks if not r["passed"]), "v776_contract")
    rows = (
        reduction["contract"]["rows"]
        if reduction
        else [
            dict(
                unit_id=f"exp{9096 + i}",
                original_qid=None,
                group_id=f"exp{9096 + i}",
                family=f"exp{9096 + i}",
                arm="full_task_contract",
                condition="preconditions_blocked",
                seed=None,
                metric="ordered_full_object_match",
                value=None,
                absolute_metric=None,
                numerator=0,
                denominator=0,
                status="censored",
                exclusion_reason=None,
                censor_reason=failed,
            )
            for i in range(14)
        ]
    )
    end = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in work["validation"]["receipts"]],
        ]
    )
    current = build_current_work_receipt(
        run_id=str(work["start_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            scope="Planning and byte custody; no current model operations"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["start_ns"],
        ended_monotonic_ns=end,
    )
    value: Json = dict(
        experiment_id=9096,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_{failed if blocked else 'v776_contract'}",
        verdict_class=cls,
        verifier_is_oracle=False,
        claim_scope="Administrative human-decision contract only; historical exact-oracle success remains circular_positive. CPU fixtures establish no scientific benefit; no current ARC solve is claimed.",
        exposure_scope="LPHB exposed development; no independent generalization; no new scientific measurements.",
        independent_generalization_score=0,
        MODEL_SPECS=[],
        model_specs=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        current_llm_invocations=0,
        model_invocation_counts=dict(model_loads=0, generation_calls=0, llm=0),
        numerical_training_calls=0,
        current_work_receipts=[current],
        historical_model_provenance=reduction.get("historical_model_provenance", []),
        required_checks_passed=bool(qualified),
        flagged_adversarial=cls == "disqualified",
        gate_check_summary=checks,
        rows=rows,
        intended_count=14,
        attempted_count=sum(r["status"] != "censored" for r in rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        effective_independent_groups=0,
        sample_size_budget=dict(administrative_tasks=14, scientific_units=0),
        contract_ready_score=int(ready),
        full_task_objects=reduction.get("contract", {}).get("tasks", []),
        canonical_task_hash=canonical_hash(reduction.get("contract", {}).get("tasks", [])),
        authority_comparison_mode=reduction.get("contract", {}).get(
            "comparison_mode", "unavailable"
        ),
        authority_snapshots=work["snapshots"],
        title_key_rows=reduction.get("contract", {}).get("title_key_rows", []),
        previous_outcome_rows=reduction.get("previous_outcome_rows", []),
        protocol_references=reduction.get("protocol_references", []),
        closed_scope_rows=reduction.get("closed_scope_rows", []),
        archive_lag=reduction.get("archive_lag", {}),
        preserved_results=reduction.get("preserved_results", {}),
        deferred_graph_bounds=reduction.get("deferred_graph_bounds", {}),
        graph_study_status="deferred_unchanged_seven_and_ten_bounds",
        arc_summary=reduction.get("arc_summary", {}),
        open_gaps=["GAP-ORACLE-DISTINCT", "GAP-DETECTOR-AUROC-4208"],
        activation_authorized=False,
        external_publication_authorized=False,
        generator_weights_trained=False,
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=work["validation"]["receipts"],
        owned_statement_coverage=work["validation"]["coverage"],
        repository_health=work.get("repository_health", dict(passed=False, reason="not_recorded")),
        code_config_hashes=work["validation_scope_manifest"]["code_test_hashes"],
        source_references=work["snapshots"],
        raw_shard_hashes=[
            reference(Path(work["raw"]) / "work.json"),
            *work["snapshots"],
            *reduction.get("protocol_references", []),
        ],
        duration_s=(end - work["start_ns"]) / 1e9,
        random_seed=7769096,
        acceptance_gates=dict(
            authenticated_inputs=not blocked,
            owned_checks=bool(qualified),
            exact_task_authority=ready,
        ),
        methodology_note="Fourteen administrative task units, zero scientific units. Recompute historical inventories from retained primitives; seeds and siblings add no independent units. No model loads, inference, training, games or scientific benefit measurement.",
    )
    value["field_principles"] = {
        **dict.fromkeys(
            [*value, "reproducibility_checksum"],
            "Exact bytes and frozen scope make administrative custody falsifiable; no scientific benefit follows.",
        ),
        "verdict_class": "Terminal external absence is blocked; owned-check failure disqualifies; administrative completion is null.",
        "contract_ready_score": "All task objects and current affected checks must agree; readiness does not measure prediction benefit.",
        "previous_outcome_rows": "Missing declared science and scheduler alternatives remain separate; original failed outcomes cannot be repaired retroactively.",
        "protocol_references": "Freeze target, cohort, arms, thresholds and timing boundaries before scientific work to prevent outcome-driven selection.",
        "effective_independent_groups": "Administrative tasks and repeated controls supply zero independent scientific observations.",
        "current_work_receipts": "Only operations owned by this invocation count as current; archived generation is historical provenance.",
        "archive_lag": "Scheduler completion cannot fill missing archive entries or scientific results.",
        "preserved_results": "Historical reported aggregates are recomputed from authenticated retained per-unit primitives without new experiments.",
        "honest_verdict": "A complete prefix records terminal accounting, not a scientific win.",
        "verifier_is_oracle": "This administrative checker is not a scientific verifier; preserved graph truth remains an executable oracle.",
        "claim_scope": "A qualified contract supports planning only, never prediction accuracy or physical acceleration.",
        "exposure_scope": "Prior researcher exposure remains visible even when a causal execution protocol is qualified.",
        "independent_generalization_score": "An exposed development panel cannot establish performance on independent natural questions.",
        "MODEL_SPECS": "An empty list declares that no model is required or loaded by this producer.",
        "inference_substrate_class": "no_model_load describes current execution rather than models cited in historical tasks.",
        "current_llm_invocations": "Current invocation counts come from the owned empty event ledger, not archived generation.",
        "model_invocation_counts": "Loading, generation and language-model calls stay separate and zero for this administrative run.",
        "historical_model_provenance": "Archived model work retains its original scope and contributes no current compute or scientific credit.",
        "rows": "Each task retains its identity, condition, absolute metric and denominator so reductions are reproducible.",
        "sample_size_budget": "Fourteen planned administrative units are distinct from a scientific sample size of zero.",
        "gate_check_summary": "Exact failed operands distinguish absent files, absent fields and observed numeric zeros.",
        "canonical_task_hash": "Stable JSON hashing binds full prompts, routing and gates rather than titles alone.",
        "authority_snapshots": "Retained original bytes permit cold verification after mutable roadmap authorities advance.",
        "title_key_rows": "The actual truncated dispatch key must be unique and unused to prevent accidental scheduler skips.",
        "closed_scope_rows": "Unchanged external failures need changed evidence before reopening; missing results are not scientific nulls.",
        "validation_receipts": "Exact current commands, normal exits, scope hashes and durable logs establish that checks actually ran.",
        "required_checks_passed": "Every affected frozen check must pass regardless of separate repository history.",
        "owned_statement_coverage": "Real SQLite execution data must cover all owned statements with zero exclusions.",
        "duration_s": "Elapsed monotonic work is measured without padding or borrowed historical runtime.",
        "random_seed": "A fixed administrative seed records reproducibility without creating independent observations.",
        "reproducibility_checksum": "Every published field is bound so rehashed tampering still faces primitive replay.",
        "code_config_hashes": "Executed reader code and original test assertions remain bound to validation scope.",
        "source_references": "Source byte pins establish custody; custody alone does not establish scientific independence.",
        "acceptance_gates": "Input authenticity, owned checks and exact task authority are separate necessary conditions for readiness.",
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reject rehashed summaries by rebuilding them from independently pinned retained inputs."""
    try:
        value = json.loads(path.read_bytes())
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        root = Path(work["retained"])
        if work["reduction"] and any(
            reference(root / name)["sha256"] != pin for name, pin in input_pins(root).items()
        ):
            raise ValueError("pinned_authority_drift")
        if work["reduction"] and work["reduction"] != reduce(work):
            raise ValueError("reduction_drift")
        if work["validation_scope_manifest"] != scope_manifest(Path(work["private"])):
            raise ValueError("scope_drift")
        if value != build(work) or value["reproducibility_checksum"] != checksum(value):
            raise ValueError("aggregate_drift")
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """The unchanged scanners and foreign cold replay must accept the exact candidate bytes."""
    with validation_binding():
        return validation.terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Publish checked bytes once; private replays cannot rewrite a live primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    if args.date != "20261115":
        parser.error("this contract requires --date 20261115")
    output = args.output.absolute()
    if args.from_work and output.is_relative_to(ROOT):
        parser.error("private work publication requires output outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    if args.from_work:
        work = json.loads(args.from_work.read_bytes())
        work["raw"] = str(raw)
    else:
        private = Path(tempfile.mkdtemp(prefix="carnot9096-", dir="/var/tmp"))
        work = collect(args.root, raw, private, args.date)
        if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
            raise TimeoutError("measurement_deadline")
        progress("validation_before")
        work["validation"] = validate(work)
        progress("validation_after")
        if not build(work)["required_checks_passed"]:
            atomic_json(raw / "work.json", work)
            progress("publication_disqualified", 0, 1)
            return 1
        receipt = validation.run_check(
            CommandSpec(
                "global_python_suite",
                tuple(work["validation_scope_manifest"]["global_suite_command"]),
                "broader_history",
                900,
            ),
            raw / "global_suite",
            work["validation_scope_manifest"],
        )
        work["repository_health"] = dict(
            passed=receipt["passed"],
            receipt=receipt,
            affects_owned_readiness=False,
            historical_failures_preserved=True,
        )
        work["end_ns"] = time.monotonic_ns()
        if (work["end_ns"] - work["start_ns"]) / 1e9 >= 4800:
            raise TimeoutError("hard_deadline")
    atomic_json(raw / "work.json", work)
    value = build(work)
    progress("publication_before")
    receipt = publish_primary(
        output,
        value,
        lambda path: terminal(path, raw / "terminal", work["validation_scope_manifest"]),
    )
    atomic_json(raw / "publication.json", receipt)
    progress("publication_after", 1, 0)
    print(json.dumps(dict(receipt, honest_verdict=value["honest_verdict"])), flush=True)
    return 0
