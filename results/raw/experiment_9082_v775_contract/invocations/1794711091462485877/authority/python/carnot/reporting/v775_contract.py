"""REQ-REPORT-9082-V775: freeze administrative custody without claiming new science."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
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
from carnot.reporting import v774_dispatch_contract as previous
from carnot.reporting import v769_branch_evidence as evidence
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    build_current_work_receipt,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design, TABLE_FIELDS
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from scripts import research_conductor as conductor
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = previous.ROOT, previous.DESIGN, previous.ACTIVE, previous.STAGED
NAME, TASK, MILESTONE = "experiment_9082_v775_contract", "exp9082-contract", "2026.11.775"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v775_contract_9082.py"
OWNED = ["python/carnot/reporting/v775_contract.py", CLI]
DIRECT = [
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_conductor_deliverable_status.py",
    "tests/python/test_roadmap_schema.py",
    "tests/python/test_audit_roadmap_gates.py",
]
REUSED = list(
    dict.fromkeys(
        [
            *validation.REUSED,
            *previous.OWNED,
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/v769_branch_evidence.py",
            "scripts/conductor_gates.py",
            "scripts/research_conductor.py",
            "scripts/experiment_template.py",
            "scripts/exclusion_manifest_lint.py",
            "scripts/overdue_priority_lint.py",
            "scripts/harness_fit_lint.py",
            "scripts/audit_roadmap_gates.py",
            "ops/exclusion_manifest.yaml",
            "scripts/conductor_exclusion_manifest.json",
        ]
    )
)
LOG = previous.LOG
ARCHIVE = "openspec/change-proposals/research-roadmap-v774-preserved-20261114.md"
QUEUE = "openspec/change-proposals/research-roadmap-v774-queue-preserved-20261114.yaml"
BRANCH = "results/experiment_9068_v774_dispatch_contract.json"
CAPSTONE = "results/experiment_9081_v774_capstone.json"
AUTHORITIES = [
    DESIGN,
    ACTIVE,
    LOG,
    "research-complete.yaml",
    ARCHIVE,
    QUEUE,
    previous.QUEUE,
    "scripts/conductor_exclusion_manifest.json",
]
PINS = {
    DESIGN: "sha256:ce767f80004cb4b9aef48f4b49c46ce024f06e8bea251a4cf9aa4d33c631ab1b",
    ARCHIVE: "sha256:9737d705002f952c03fc9f0a57e3404e8ad8efc5a7119af10d891d967fad8b10",
    QUEUE: "sha256:1d87c5c7f1e4ab81c81333f67c6266d1e78f290ba0b28abc130d93c5326847bd",
    BRANCH: "sha256:fdb97d887079941cfa17e89146cf6097deb2396d2007a44565020a755f10f090",
    CAPSTONE: "sha256:f6097c48c395164b77ef62a87f3274e492badc99bc9fd1fde0f0f277ae37f704",
    **{
        p: evidence.PINS[p]
        for p in [evidence.GRAPH, evidence.ONLINE, evidence.PROTOCOL, evidence.ONLINE_PROTOCOL]
    },
}
reference, checksum, check = previous.reference, previous.checksum, evidence.check
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries expose progress while the shared supervisor heartbeats children."""
    print(f"[exp9082] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the proven command and receipt policy with this consumer's exact scope."""
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


def scope_manifest(private: Path) -> Json:
    """Freeze tests, commands and code before reading dependent scientific evidence."""
    with binding():
        manifest = validation.scope_manifest(private)
    py = str(ROOT / ".venv/bin/python")
    manifest["activation_commands"] = [
        [py, "scripts/exclusion_manifest_lint.py", ACTIVE],
        [py, "scripts/overdue_priority_lint.py", "--strict"],
        [py, "scripts/harness_fit_lint.py", ACTIVE],
        [py, "scripts/audit_roadmap_gates.py", ACTIVE],
    ]
    manifest["global_suite_command"] = [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    return manifest


def authority(root: Path) -> Json:
    """Bind every full task object and exact producer gate without inventing prior failures."""
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
        checks += [
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
        log = (root / LOG).read_text()
        completed = {
            p[2].strip()
            for line in log.splitlines()
            if len(p := line.split("|")) >= 4 and p[3].strip() in {"OK", "FLAGGED"}
        }
        archived = (root / "research-complete.yaml").read_text()
        titles = [t["title"][:50].strip() for t in tasks]
        earlier: Json = {}
        for i, task in enumerate(tasks):
            identity, key = task["id"], titles[i]
            declared = task["prompt"].split("REQUIRED ARTIFACT FIELDS:", 1)[-1]
            gate_ok = all(
                set(g) == {"upstream", "artifact_field", "op", "value"}
                and g["upstream"] in earlier
                and g["artifact_field"] in earlier[g["upstream"]]
                for g in task.get("gated_on", [])
            )
            predicates = dict(
                full_object=i < len(machine) and task == machine[i],
                visible_table=i < len(table)
                and all(table[i][k] == task[k] for k in TABLE_FIELDS)
                and table[i]["order"] == i + 1,
                sequence=identity.startswith(f"exp{9082 + i}-"),
                gates=gate_ok,
                unique_key=titles.count(key) == 1,
                uncompleted_key=key not in completed,
                unused_id=re.search(rf"\bexp{9082 + i}[-_]", archived) is None
                and not list((root / "results").glob(f"experiment_{9082 + i}_*.json")),
            )
            matched = all(predicates.values())
            rows.append(
                dict(
                    unit_id=identity,
                    group_id=identity,
                    family=identity,
                    arm="full_task_contract",
                    condition="exact_authority",
                    seed=None,
                    status="completed" if matched else "failed",
                    metric="ordered_full_object_match",
                    value=int(matched),
                    absolute_metric=int(matched),
                    numerator=int(matched),
                    denominator=1,
                    exclusion_reason=None,
                    censor_reason=None,
                    task=task,
                    task_sha256=canonical_hash(task),
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
            earlier[identity] = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", declared))
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


def scheduler_replay(root: Path, tasks: list[Json], directory: Path) -> list[Json]:
    """Run the unchanged selector with private queues, A receipts and absent B primaries."""
    historical = yaml.safe_load((root / previous.QUEUE).read_bytes())["tasks"]
    pairs = [
        ([t for t in historical if t["id"].startswith(("exp9058-", "exp9059-"))], "old"),
        ([t for t in tasks if t["id"] in {"exp9085-collect-a", "exp9086-collect-b"}], "new"),
    ]
    rows = []
    for pair, label in pairs:
        a, b = pair
        private = directory / label
        roadmap, artifact = private / ACTIVE, private / a["deliverable"]
        log = f"| date | {a['title'][:50].strip()} | OK | private A only |\n"
        files = {
            roadmap: yaml.safe_dump(dict(milestone=MILESTONE, tasks=pair)),
            artifact: json.dumps(
                dict(
                    experiment_id=int(a["id"][3:7]),
                    honest_verdict="complete_null_fixture",
                    verdict_class="null",
                )
            ),
            private / "log.md": log,
            private / "scripts/conductor_exclusion_manifest.json": (
                root / "scripts/conductor_exclusion_manifest.json"
            ).read_text(),
        }
        for path, content in files.items():
            if path.exists() and path.read_text() != content:
                raise ValueError("fixture_drift")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        with patch.multiple(
            conductor,
            PROJECT_ROOT=private,
            ROADMAP_FILE=roadmap,
            RESEARCH_TASKS=[],
            _tasks_loaded=False,
            _roadmap_mtime=0,
            _EXCLUSION_MANIFEST=None,
            _EXCLUSION_MANIFEST_MTIME=0,
        ):
            reason = conductor._artifact_unfinished_reason(b)
            selected = conductor.pick_next_task(log)
        selected_id = selected["id"] if selected else None
        success = selected_id == b["id"]
        rows.append(
            dict(
                condition=label,
                expected_id=b["id"],
                selected_id=selected_id,
                B_primary_absent=not (private / b["deliverable"]).exists(),
                missing_primary_reason=reason,
                expected_failure_observed=label == "old" and not success,
                passed=success if label == "new" else not success,
                fixture_only=True,
                references=[reference(p) for p in files],
            )
        )
    return rows


def reduce(work: Json) -> Json:
    """Frozen historical methods support readiness only; missing scientific primaries stay absent."""
    root, raw = Path(work["retained"]), Path(work["evidence_raw"])
    contract = authority(root)
    science = evidence.science(root)
    frozen = []
    for name in [evidence.PROTOCOL, evidence.ONLINE_PROTOCOL]:
        method = json.loads((root / name).read_bytes())
        wrapper = dict(
            method=method,
            original_reference=reference(root / name),
            corpus_ids=["exp9085-collect-a", "exp9086-collect-b"],
            access_authority="exp9083-prospective-custody",
        )
        target = raw / "frozen_protocols" / Path(name).name.replace("v768", "v775")
        if target.exists() and json.loads(target.read_bytes()) != wrapper:
            raise ValueError("immutable_protocol_drift")
        if not target.exists():
            atomic_json(target, wrapper)
        frozen.append(dict(reference(target), original_reference=reference(root / name)))
    branch = json.loads((root / BRANCH).read_bytes())
    capstone = json.loads((root / CAPSTONE).read_bytes())
    archived = yaml.load((root / "research-complete.yaml").read_bytes(), Loader=yaml.CSafeLoader)[
        "milestones"
    ]
    return dict(
        contract=contract,
        science=science,
        frozen_protocol_references=frozen,
        scheduler_rows=scheduler_replay(root, contract["tasks"], raw / "scheduler"),
        previous_outcome_inventory=capstone["slot_rows"],
        historical_model_provenance=[
            dict(
                reference=reference(root / p),
                model_invocation_counts=json.loads((root / p).read_bytes()).get(
                    "model_invocation_counts", {}
                ),
            )
            for p in [evidence.GRAPH, evidence.ONLINE]
        ],
        archive_lag=dict(
            planning_archive_end="2026.11.773",
            observed_archive_end=archived[-1]["id"],
            completed_scheduling_frontier="2026.11.774",
            archive_lagged=archived[-1]["id"] != "2026.11.774",
            archive_modified=False,
            scientific_result_from_scheduler=False,
            reference=reference(root / "research-complete.yaml"),
        ),
        arc_summary=dict(
            arc_new_outcomes_ready_score=branch["arc_new_outcomes_ready_score"],
            new_solve_count=0,
            added_dependency=False,
            reference=reference(root / BRANCH),
        ),
    )


def input_pins(root: Path) -> dict[str, str]:
    """Original primary pins authenticate the producer and state hashes they declare."""
    pins = dict(PINS, **{previous.QUEUE: previous.PINS[previous.QUEUE]})
    for name in [evidence.GRAPH, evidence.ONLINE]:
        if reference(root / name)["sha256"] == PINS[name]:
            value = json.loads((root / name).read_bytes())
            refs = [
                *value["code_config_hashes"],
                *value.get("feature_references", []),
                *value["raw_shard_hashes"][:1],
            ]
            if "trained_state_reference" in value:
                refs.append(value["trained_state_reference"])
            for ref in refs:
                relative = str(Path(ref["path"]).relative_to(ROOT))
                # The evolving capability spec is historical context, not executable producer code.
                if relative != "openspec/capabilities/research-reporting/spec.md":
                    pins[relative] = ref["sha256"]
    return pins


def collect(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate inputs and storage before dependent work; keep all failed operands visible."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private.mkdir(parents=True, mode=0o700, exist_ok=True)
    progress("preconditions_before")
    manifest = scope_manifest(private)
    env = environment(private)
    checks = [
        *env["checks"],
        check(raw, "run_date", "20261115", run_date, "date"),
        check(raw, "private_storage_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage"),
        check(
            private,
            "private_storage_mode",
            0o700,
            private.stat().st_mode & 0o777,
            "private_storage",
        ),
    ]
    pins = input_pins(root)
    paths = list(
        dict.fromkeys(
            [
                *pins,
                *AUTHORITIES,
                "AGENTS.md",
                "CODEX.md",
                "CLAUDE.md",
                "research-program.md",
                "ops/north-star.md",
                "ops/e2e-test-plan.md",
                "research-references.md",
                "openspec/capabilities/research-reporting/spec.md",
                *[str(Path(r["path"]).relative_to(ROOT)) for r in manifest["code_test_hashes"]],
            ]
        )
    )
    if (root / STAGED).is_file():
        paths.append(STAGED)
    snapshots = []
    for i, name in enumerate(paths):
        source = ROOT / name if name in OWNED + [TEST] else root / name
        target = raw / "authority" / name
        ref = reference(source)
        checks.append(
            check(
                source,
                "sha256",
                pins.get(name, ref["sha256"] or "required_file"),
                ref["sha256"] or "missing_file",
                "source_bytes",
            )
        )
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("immutable_snapshot_drift")
            target.write_bytes(source.read_bytes())
            snapshots.append(dict(reference(target), source=ref))
        if i % 25 == 0:
            progress("authenticate_and_freeze", i + 1, len(paths) - i - 1)
    checks += authority(raw / "authority")["checks"]
    work: Json = dict(
        raw=str(raw),
        evidence_raw=str(raw),
        retained=str(raw / "authority"),
        private=str(private),
        start_ns=started,
        frozen_ns=time.monotonic_ns(),
        run_date=run_date,
        owner_pid=os.getpid(),
        snapshots=snapshots,
        validation_scope_manifest=manifest,
        preconditions_checked=dict(env, checks=checks),
        reduction={},
        activation_receipts=[],
        validation=dict(receipts=[], coverage=dict(complete=False, percent=0)),
    )
    progress("preconditions_after", len(checks), sum(not r["passed"] for r in checks))
    if all(r["passed"] for r in checks):
        for i, argv in enumerate(manifest["activation_commands"]):
            work["activation_receipts"].append(
                validation.run_check(
                    CommandSpec("activation_" + str(i), tuple(argv), "activation_read_only", 120),
                    raw / "activation",
                    manifest,
                )
            )
        progress("reduction_before")
        work["reduction"] = reduce(work)
        progress("reduction_after", 14, 0)
    work["end_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    return work


def activation_pass(work: Json) -> bool:
    """Passing flags need exact current commands, scope, timing and durable log hashes."""
    try:
        manifest, receipts = work["validation_scope_manifest"], work["activation_receipts"]
        commands = manifest["activation_commands"]
        return len(receipts) == len(commands) and all(
            r["name"] == "activation_" + str(i)
            and r["command_argv"] == commands[i]
            and r["passed"] is True
            and r["normal_exit"] is True
            and r["timed_out"] is False
            and type(r["exit_code"]) is int
            and r["exit_code"] == 0
            and r["ended_monotonic_ns"] >= r["started_monotonic_ns"] >= work["frozen_ns"]
            and r["scope_sha256"] == canonical_hash(manifest)
            and r["log_sha256"] is not None
            and reference(Path(r["log_path"]))["sha256"] == r["log_sha256"]
            for i, r in enumerate(receipts)
        )
    except (KeyError, TypeError, OSError, ValueError):
        return False


def build(work: Json) -> Json:
    """Recompute independent readiness from primitive task rows and current normal-exit checks."""
    reduction, valid = work["reduction"], work["validation"]
    preconditions = work["preconditions_checked"]["checks"]
    blocked = not all(r["passed"] for r in preconditions)
    with binding():
        checked = validation.receipts_pass(work) and verify_measured(valid["coverage"], OWNED)
    checked = bool(checked and activation_pass(work))
    contract = bool(
        checked
        and not blocked
        and reduction
        and reduction["contract"]["ready"]
        and all(r["passed"] for r in reduction["scheduler_rows"])
    )
    methods = bool(
        checked
        and not blocked
        and reduction
        and reduction["science"]["frozen_graph_authority_ready_score"]
        and reduction["science"]["online_protocol_ready_score"]
    )
    cls = "blocked" if blocked else "null" if checked else "disqualified"
    failed = next((r["check"] for r in preconditions if not r["passed"]), "v775_contract")
    rows = (
        reduction["contract"]["rows"]
        if reduction
        else [
            dict(
                unit_id=f"exp{9082 + i}",
                group_id=f"exp{9082 + i}",
                family=f"exp{9082 + i}",
                seed=None,
                arm="full_task_contract",
                condition="preconditions_blocked",
                status="censored",
                metric="ordered_full_object_match",
                value=None,
                absolute_metric=None,
                numerator=0,
                denominator=0,
                exclusion_reason=None,
                censor_reason=failed,
            )
            for i in range(14)
        ]
    )
    end = max(
        [work["end_ns"], *[r.get("ended_monotonic_ns", work["end_ns"]) for r in valid["receipts"]]]
    )
    current = build_current_work_receipt(
        run_id=str(work["start_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            scope="Administrative contract only; no current model operations"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["start_ns"],
        ended_monotonic_ns=end,
    )
    value: Json = dict(
        experiment_id=9082,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_{failed if blocked else 'v775_contract'}",
        verdict_class=cls,
        verifier_is_oracle=False,
        claim_scope="Administrative contract and frozen-method custody only; graph success is circular_positive with verifier_is_oracle=true; no scientific benefit measured.",
        exposure_scope="Historical development states and metadata; human LPHB remains exposed development; protected TEST outcomes stay sealed.",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        current_llm_invocations=0,
        model_invocation_counts=dict(model_loads=0, generation_calls=0, llm=0),
        numerical_training_calls=0,
        current_work_receipts=[current],
        historical_model_provenance=reduction.get("historical_model_provenance", []),
        required_checks_passed=checked,
        flagged_adversarial=not checked and not blocked,
        gate_check_summary=[
            *preconditions,
            check(
                Path(work["raw"]) / "work.json",
                "required_checks_passed",
                True,
                checked,
                "current_qualification",
            ),
            *reduction.get("science", {}).get("checks", []),
        ],
        rows=rows,
        intended_count=14,
        attempted_count=sum(r["status"] != "censored" for r in rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        effective_independent_groups=0,
        administrative_groups=len(rows),
        contract_ready_score=int(contract),
        frozen_methods_ready_score=int(methods),
        canonical_tasks_sha256=canonical_hash(reduction.get("contract", {}).get("tasks", [])),
        full_task_objects=reduction.get("contract", {}).get("tasks", []),
        authority_comparison_mode=reduction.get("contract", {}).get(
            "comparison_mode", "unavailable"
        ),
        authority_snapshots=work["snapshots"],
        title_key_rows=reduction.get("contract", {}).get("title_key_rows", []),
        scheduler_rows=reduction.get("scheduler_rows", []),
        frozen_protocol_references=reduction.get("frozen_protocol_references", []),
        frozen_science=reduction.get("science", {}),
        archive_lag=reduction.get("archive_lag", {}),
        previous_outcome_inventory=reduction.get("previous_outcome_inventory", []),
        arc_summary=reduction.get("arc_summary", {}),
        open_gaps=[
            "Graph decisions remain circular; no oracle-distinct generalization evidence.",
            "Exposed LPHB cannot supply fresh generalization; independent useful learning remains open.",
            "Physical hardware claims need operation-specific evidence.",
        ],
        closed_scopes=[
            "No unchanged MDH-Bench or MDHJudgments source retry.",
            "No ARC re-solve or added ARC dependency; no newly qualified frontier.",
            "No generator training, roadmap activation or external publication.",
        ],
        activation_authorized=False,
        external_publication_authorized=False,
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=valid["receipts"],
        activation_receipts=work["activation_receipts"],
        owned_statement_coverage=valid["coverage"],
        repository_health=work.get("repository_health", dict(passed=False, reason="not_recorded")),
        repository_health_passed=False,
        preconditions_checked=work["preconditions_checked"],
        code_config_hashes=work["validation_scope_manifest"]["code_test_hashes"],
        source_references=work["snapshots"],
        raw_shard_hashes=[
            reference(Path(work["raw"]) / "work.json"),
            *work["snapshots"],
            *reduction.get("frozen_protocol_references", []),
        ],
        duration_s=(end - work["start_ns"]) / 1e9,
        random_seed=7759082,
        acceptance_gates=dict(
            authenticated_inputs=not blocked,
            owned_checks=checked,
            contract=contract,
            frozen_methods=methods,
        ),
        methodology_note="Fourteen administrative units; no scientific measurements, model loads, generation, training or independent claims. Seeds and repeats add no independent units.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Authenticate bytes and primitive rows; current work stays separate from historical model provenance and scheduler completion grants no scientific credit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def validate(work: Json) -> Json:
    """Run each frozen check once, retaining real SQLite coverage with zero exclusions."""
    receipts = []
    private, raw = Path(work["private"]), Path(work["raw"])
    with binding(), patch.dict(os.environ, COVERAGE_RCFILE=str(private / "coverage.ini")):
        for command in work["validation_scope_manifest"]["commands"]:
            receipts.append(
                validation.run_check(
                    CommandSpec(
                        command["name"],
                        tuple(command["argv"]),
                        "current_scoped",
                        command["timeout_s"],
                    ),
                    raw / "validation",
                    work["validation_scope_manifest"],
                )
            )
    coverage = measured([private / ".coverage"], OWNED, raw / "coverage")
    coverage.pop("rows", None)
    return dict(receipts=receipts, coverage=coverage)


def replay(path: Path) -> Json:
    """Rebuild frozen reductions so rehashed task edits cannot invent a qualified contract."""
    try:
        value = json.loads(path.read_bytes())
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        if work["reduction"] and any(
            reference(Path(work["retained"]) / name)["sha256"] != pin
            for name, pin in input_pins(Path(work["retained"])).items()
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
    """Unchanged adversarial and strict-row scanners plus foreign replay guard publication."""
    with binding():
        return validation.terminal(candidate, raw, manifest)


def main(argv: list[str] | None = None) -> int:
    """Publish one validated primary; private diagnostic invocations cannot alter live results."""
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
        private = Path(tempfile.mkdtemp(prefix="carnot9082-", dir="/var/tmp"))
        work = collect(args.root, raw, private, args.date)
        if work["reduction"]:
            if (time.monotonic_ns() - work["start_ns"]) / 1e9 >= 3300:
                raise TimeoutError("measurement_deadline")
            progress("validation_before")
            work["validation"] = validate(work)
            progress("validation_after")
    atomic_json(raw / "work.json", work)
    value = build(work)
    if not args.from_work and work["reduction"] and not value["required_checks_passed"]:
        progress("publication_disqualified", 0, 1)
        return 1
    if not args.from_work and work["reduction"]:
        progress("global_suite_before")
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
        progress("global_suite_after", 1, 0)
        work["end_ns"] = time.monotonic_ns()
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
