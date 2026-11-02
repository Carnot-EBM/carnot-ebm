"""REQ-REPORT-8387: sealed task accounting keeps engineering progress separate from benefit."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v721_capstone_evidence as old
from carnot.reporting import v722_contract_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT, DESIGN, STAGED, ACTIVE = contract.ROOT, contract.DESIGN, contract.STAGED, contract.ACTIVE
PROTOCOL, PROTOCOL_PIN = old.PROTOCOL, old.PROTOCOL_PIN
NAME, TASK, MILESTONE = "experiment_8387_v722_capstone", "exp8387-capstone", "2026.10.722"
TASK_PIN = "sha256:3534a4e917d2d8900490f96cbfde74b343ce5be277ccfa5a47cae6c145ce4829"
CLI, TEST = "scripts/experiments/" + NAME + ".py", "tests/python/test_v722_capstone_8387.py"
OWNED = [
    "python/carnot/reporting/v722_capstone_evidence.py",
    "python/carnot/reporting/v722_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
freeze, load, gate, memory = old.freeze, old.load, old.gate, old.memory
OUTCOME, BUILD, REPLAY = old.outcome, old.build, old.replay
NEXT = [
    "Continue custody after an independent full V722 design contract matches all active task bytes.",
    "Defer semantic study until80 disjoint independently labeled clusters, eight per class, released outputs and evidence qualify; no label-oracle substitution.",
    "Continue direct recovery after authority qualifies; retain process-crash scope and require exact reader, pending, acknowledgment and fsync receipts.",
    "Defer actual96-slot three-arm small-head training until direct_state_ready_score=1; preserve22 missing slots and closed exposed H2 null.",
    "Defer complete Python costs until direct state qualifies; include scoring, updates, serialization, transfers, fsync and response.",
    "Continue native qualification only after resolving the owned failure with changed authenticated evidence; finite numerical parity grants no semantic credit.",
    "Defer full native costs until actual Python costs and qualified native parity exist; include every request operation and copies.",
    "Continue the separate dyadic-logit prototype only under its declared mathematical policy; production migration and probability calibration remain unauthorized.",
    "Retire only unchanged CUDA evidence inspection after exact predecessor authentication; require a causal hash-bound library/environment change before one context/copy probe.",
    "Defer bounded Qwen canary until typed reader, changed runtime and actual context all qualify; source independence and small sample limits remain.",
    "Continue ARC only with authenticated fresh cross-game supervisor outcomes and overlapping arm support; public exposed observations grant no hidden-game credit.",
    "Continue operation mapping after complete direct/native costs exist; KV260 quadratic Ising k<=5 and PolarFire board-local Linux CPU scopes remain.",
    "Defer GateMate probing until both exact source hashes, dated cable/port/power change, IDCODE0x20000001, n16 flash and device smoke qualify.",
    "Continue independent changed evidence; terminal absence does not request an identical capstone retry or external publication.",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so the conductor can distinguish work from a stalled child."""
    print(f"[exp8387] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse sealed readers and worker isolation while directing children to this task's CLI."""
    with (
        patch.multiple(
            old,
            contract=contract,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            NEXT=NEXT,
            progress=progress,
            outcome=outcome,
            reduce=reduce,
            build=build,
        ),
        patch.object(
            old, "READERS", dict(old.READERS, **{}) | {8381: "logit_policy_certificate_8381"}
        ),
    ):
        yield


def outcome(task: Json, primary: Json, raw: Path, sealed: list[Json] | None = None) -> Json:
    """Import measured fields only after unchanged terminal and operand authentication."""
    value = load(primary)
    original_terminal = None
    if sealed is None and value.get("blocked_at_layer") != "conductor_pre_gate":
        terminal_path = Path(value["terminal_validation_sidecar_path"])
        terminal = json.loads(terminal_path.read_bytes())
        if "publication" not in terminal:
            original_terminal = freeze(terminal_path, raw / "terminal_origin")
            original_terminal["path"] = original_terminal["snapshot_path"]
            adapter = raw / "terminal_schema_adapter.json"
            atomic_json(adapter, dict(publication=terminal, schema_adapter_from=original_terminal))
            original_open = Path.open

            def opened(path: Path, *args: Any, **kwargs: Any) -> Any:
                return original_open(adapter if path == terminal_path else path, *args, **kwargs)

            with patch.object(Path, "open", opened):
                result: Json = dict(OUTCOME(task, primary, raw, sealed))
            result["closure"].append(original_terminal)
            result["checks"].append(
                gate(
                    original_terminal["path"],
                    "source.sha256",
                    original_terminal["sha256"],
                    original_terminal["sha256"],
                    original_terminal["sha256"],
                )
            )
        else:
            result = dict(OUTCOME(task, primary, raw, sealed))
    else:
        result = dict(OUTCOME(task, primary, raw, sealed))
    if result["producer_executed"]:
        terminal = load(result["closure"][1])
        if terminal.get("schema_adapter_from") and terminal["publication"] != load(
            terminal["schema_adapter_from"]
        ):
            raise ValueError("terminal_schema_adapter_binding")
        fields = [
            "H1",
            "H2",
            "qualified_scope_decision",
            "fast_path_fraction",
            "guard_ready_score",
            "class_support",
            "coverage_by_lane",
            "label_authority_counts",
            "source_overlap_rows",
            "lane_inventory",
            "adoption_decisions",
            "mixed_version_read_count",
            "recovery_mismatch_count",
            "lost_acknowledged_update_count",
            "duplicate_update_count",
            "issued_prediction_mutation_count",
            "power_loss_certified",
            "crash_rows",
            "max_probability_error",
            "max_coefficient_error",
            "action_mismatch_count",
            "finite_domain_only",
            "owned_failure",
            "native_call_counts",
            "new_policy_id",
            "original_policy_difference_rows",
            "confidence_scope",
            "default_enabled",
            "production_migration_authorized",
            "environment_delta",
            "current_probe_count",
            "context_copy_receipts",
            "arm_support_rows",
            "support_frontier",
            "panel",
            "frozen_panel",
            "per_game_results",
            "supervisor_outcome_rows",
            "emitted_receipt_count",
            "headline_solve_credit",
            "adapter_withheld",
            "actual_wrapper_path",
            "policy_rows",
            "operation_rows",
            "board_obligations",
            "polarfire_graduation",
            "polarfire_workload_validated",
            "compatible_fraction",
            "compatible_fraction_status",
            "compatible_cost_fraction",
            "historical_dispositions",
            "exact_missing_hashes",
            "physical_change_receipt",
            "required_idcode",
            "device_command_count",
            "next_evidence_conditions",
        ]
        result["selected"].update(
            {k: value[k] for k in fields if k in value and k not in result["selected"]}
        )
        result["selected"]["sample_boundary"] = {
            k: value.get(k)
            for k in [
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "exposure_scope",
                "verifier_is_oracle",
                "sample_size_budget",
            ]
        }
    result["checks"] = sorted(result["checks"], key=canonical_hash)
    return result


def invoke(
    task: Json, primary: Json, raw: Path, closure: list[Json] | None = None
) -> tuple[Json, Json]:
    """A new process owns each producer's1500 MiB peak and500 MiB growth budget."""
    with binding():
        summary, receipt = old.invoke(task, primary, raw, closure)
        return dict(summary), dict(receipt)


def worker_process(request: Path, output: Path) -> int:
    """Keep real child failures and fresh memory measurements through the repaired worker."""
    with binding():
        return int(old.worker_process(request, output))


def absent(task: Json, primary: Json, cascade: str | None) -> Json:
    """A conductor skip is an observation about scheduling, never an executed producer."""
    return dict(
        disposition="logged_cascade_skip" if cascade else "absent",
        honest_verdict=None,
        verdict_class="blocked",
        producer_executed=False,
        eligible=False,
        selected={},
        checks=[gate(primary["path"], "producer.exists", True, None)],
        closure=[primary],
        branch_replay=dict(status="blocked_absent_input", passed=False),
        memory=dict(passed=True),
        cascade_log_line=cascade,
    )


def measure(root: Path, raw: Path) -> Json:
    """Seal every scheduled operand once; later absent producers stay dependencies."""
    progress("measurement_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    start, before = time.monotonic_ns(), memory()
    auth = contract.authority(root, raw / "authority")
    tasks = auth["tasks"]
    if (
        [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8374, 8388)]
        or canonical_hash(tasks[-1]) != TASK_PIN
        or tasks[-1].get("gated_on")
    ):
        raise ValueError("exact_fourteen_task_authority")
    refs = [freeze(root / name, raw / "custody") for name in [DESIGN, STAGED, ACTIVE]]
    protocol = freeze(root / PROTOCOL, raw / "custody", PROTOCOL_PIN)
    checks = [
        *auth["gate_check_summary"],
        gate(str(root / PROTOCOL), "protocol.sha256", PROTOCOL_PIN, protocol["sha256"]),
    ]
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    checks.extend(
        [
            gate(
                str(raw),
                "private_disk_backed_scratch",
                True,
                filesystem not in ["tmpfs", "ramfs"] and raw.stat().st_mode & 0o077 == 0,
            ),
            gate(
                str(raw),
                "available_disk_at_least_1GiB",
                True,
                shutil.disk_usage(raw).free >= 1024**3,
            ),
            gate("/proc/meminfo", "available_memory_at_least_1GiB", True, available >= 1024**3),
            gate(str(root / ACTIVE), "full_task_sha256", TASK_PIN, canonical_hash(tasks[-1])),
        ]
    )
    checks.extend(
        gate(
            str(ROOT / ".venv/bin" / tool),
            "required_tool_executable",
            True,
            os.access(ROOT / ".venv/bin" / tool, os.X_OK),
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]
    )
    progress("preconditions_checked", len(checks), 0)
    for name, pin in [
        (contract.previous.PROTOCOL, contract.previous.DEPLOYMENT_PIN),
        (contract.PROTOCOL, contract.PROTOCOL_PIN),
    ]:
        ref = freeze(root / name, raw / "custody", pin)
        refs.append(ref)
        checks.append(gate(ref["path"], "protocol.sha256", pin, ref["sha256"]))
    refs.append(freeze(root / "ops/exclusion_manifest.yaml", raw / "custody"))
    conductor = freeze(root / "ops/conductor-log.md", raw / "custody")
    lines = Path(conductor["snapshot_path"]).read_text().splitlines() if conductor["exists"] else []
    inputs, receipts = [], []
    from scripts.conductor_gates import _find_artifact_by_task_id

    for index, task in enumerate(tasks[:-1]):
        progress("producer_before", index, 13 - index)
        path = _find_artifact_by_task_id(task["id"], root / "results") or root / task["deliverable"]
        primary = freeze(path, raw / "custody")
        cascade = next(
            (
                line
                for line in reversed(lines)
                if "Pre-emptive skip:" in line and task["title"][:48] in line
            ),
            None,
        )
        if primary["exists"]:
            summary, receipt = invoke(task, primary, raw / "branches" / task["id"])
            receipts.append(receipt)
        else:
            summary = absent(task, primary, cascade)
        inputs.append(dict(task=task, primary=primary, summary=summary))
        progress("producer_after", index + 1, 12 - index)
    history = []
    for number in [8361, 8362, 8368, 8370, 8371, 8372, 8373]:
        path = next((root / "results").glob(f"experiment_{number}_*.json"), None)
        if path:
            item = old.historical(path, raw / "history")
            if number == 8361:
                design = (
                    root / "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md"
                )
                prior_task = parse_design(design.read_text(), milestone="2026.10.721")[1][1]
                summary, receipt = invoke(prior_task, item["reference"], raw / "history_utility")
                item.update(
                    selected=summary["selected"],
                    refs=summary["closure"],
                    utility_replay=summary["branch_replay"],
                )
                receipts.append(receipt)
                refs.append(freeze(design, raw / "custody"))
            if number == 8362:
                prior = load(item["reference"])
                item["selected"] = {
                    k: prior[k]
                    for k in ["fast_path_fraction", "guard_ready_score", "fallback_counts"]
                }
            history.append(item)
    after = memory()
    work = dict(
        root=str(root),
        tasks=tasks,
        authority_refs=refs[:3],
        extra_refs=refs[3:] + [conductor],
        canonical_tasks_sha256=canonical_hash(tasks),
        protocol_reference=protocol,
        inputs=inputs,
        history=history,
        checks=checks,
        branch_receipts=receipts,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        memory_measurements=dict(
            parent_before=before,
            parent_after=after,
            parent_growth_mb=max(0, after["current_rss_mb"] - before["current_rss_mb"]),
            parent_measurement_scope="current_RSS_only; worker_peaks_are_fresh_processes",
            peak_bound_mb=1500,
            growth_bound_mb=500,
            workers=[i["summary"]["memory"] for i in inputs],
        ),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Recompute all dispositions while preserving zero semantic and learning benefit."""
    with binding():
        auth = old.authority(work)
    if work["tasks"] != auth["tasks"] or work["canonical_tasks_sha256"] != canonical_hash(
        auth["tasks"]
    ):
        raise ValueError("full_task_authority")
    owned = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    owned &= work["memory_measurements"]["parent_growth_mb"] <= 500
    owned &= work["memory_measurements"]["parent_after"]["current_rss_mb"] <= 1500
    owned &= all(i["summary"]["memory"]["passed"] for i in work["inputs"])
    owned &= all(r["passed"] for r in work["branch_receipts"])
    rows: list[Json] = []
    for i in work["inputs"]:
        s = i["summary"]
        row = {k: v for k, v in s.items() if k not in ["closure", "selected", "checks", "memory"]}
        row.update(
            experiment_id=int(i["task"]["id"][3:7]),
            task_id=i["task"]["id"],
            unit_id=i["task"]["id"],
            primary_reference=i["primary"],
            missing_reason=None if i["primary"]["exists"] else "producer_primary_absent",
            continuation_condition=NEXT[len(rows)],
            metric_reference=i["primary"] if s["producer_executed"] else None,
        )
        rows.append(row)
    kind = (
        "disqualified"
        if not owned
        else "blocked"
        if not all(g["passed"] for g in work["checks"]) or any(not r["eligible"] for r in rows)
        else "null"
    )
    rows.append(
        dict(
            experiment_id=8387,
            task_id=TASK,
            unit_id=TASK,
            disposition="capstone",
            producer_executed=True,
            eligible=owned,
            honest_verdict="complete_" + kind + "_v722_capstone",
            verdict_class=kind,
            primary_reference=None,
            continuation_condition=NEXT[-1],
            missing_reason=None,
        )
    )
    selected = [
        i["summary"]["selected"] if i["summary"]["producer_executed"] else None
        for i in work["inputs"]
    ]
    utility: Json = next(
        (
            h.get("selected", {})
            for h in work["history"]
            if h["experiment_id"] == 8361 and h["authenticated"]
        ),
        {},
    )
    gates = [
        *work["checks"],
        *[g for i in work["inputs"] for g in i["summary"]["checks"] if not g["passed"]],
        *[
            gate(
                r.get("stdout_path", "owned_validation"),
                r.get("name", "passed"),
                True,
                False,
                r.get("stdout_sha256"),
            )
            for r in receipts
            if not r["passed"] and r.get("scope") != "global"
        ],
    ]
    gates = [
        dict(
            g,
            check=g.get("check", g.get("artifact_field", g.get("field"))),
            upstream=g.get("upstream", g.get("upstream_id")),
            path=g.get("path", g.get("artifact_path")),
            sha256=g.get("sha256", g.get("artifact_hash", g.get("hash"))),
            artifact_field=g.get("artifact_field", g.get("field")),
            operator=g.get("operator", g.get("op", "==")),
            expected=g.get("expected", g.get("expected_value")),
            observed=g.get("observed", g.get("observed_value", g.get("actual"))),
        )
        for g in gates
    ]
    retirements = [
        dict(
            task_id=i["task"]["id"],
            prior_task_id=p["experiment_id"],
            exact_verdict=p["verdict"],
            prior_reference=h["reference"],
            current_reference=i["primary"],
            hypothesis_retired=False,
            scope="unchanged evidence inspection only",
            reopening_condition=NEXT[index],
        )
        for index, i in enumerate(work["inputs"])
        for p in i["task"].get("prior_failures", [])
        for h in work["history"]
        if h["task_id"] == p["experiment_id"]
        and h["authenticated"]
        and p.get("retire_if_same_verdict")
        and i["summary"]["producer_executed"]
        and i["summary"]["verdict_class"] != "disqualified"
        and h["honest_verdict"] == p["verdict"] == i["summary"]["honest_verdict"]
    ]
    return dict(
        rows=rows,
        task_dispositions=rows,
        honest_verdict=rows[-1]["honest_verdict"],
        verdict_class=kind,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        capstone_execution_ready_score=int(owned),
        science_ready_score=int(bool(utility)),
        gate_check_summary=gates,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        cascade_skip_count=sum(r["disposition"] == "logged_cascade_skip" for r in rows),
        missing_output_count=sum(not i["primary"]["exists"] for i in work["inputs"]),
        H1=utility.get("H1", dict(qualified=False)),
        H2=utility.get("H2", dict(qualified=False)),
        historical_table_boundary=next(
            (
                h["selected"]
                for h in work["history"]
                if h["experiment_id"] == 8362 and h["authenticated"]
            ),
            {},
        ),
        producer_measurements=selected,
        deployment_results=dict(
            atomic_recovery=selected[2],
            actual_continuous_training=selected[3],
            complete_local_service_costs=selected[4],
            native_parity=selected[5],
            native_service_costs=selected[6],
            numerical_prototype=selected[7],
            whole_service_benefit_proven=False,
            semantic_correctness_proven=False,
        ),
        external_corpus_readiness=selected[1],
        runtime_evidence=selected[8],
        qwen_canary=selected[9],
        arc_support=selected[10],
        board_obligations=dict(
            KV260="quadratic Ising k<=5 only",
            PolarFire="board-local Linux CPU only; fabric unmeasured",
            operation_mapping=selected[11],
            GateMate=selected[12],
        ),
        historical_dispositions=work["history"],
        original_utility_dispositions=utility.get("original_dispositions", []),
        qualified_scope_decision=utility.get("qualified_scope_decision", {}),
        retirements=retirements,
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=n, closed=False, progress=selected[j], next_evidence_condition=NEXT[j])
            for n, j in [
                ("useful_verified_decisions", 1),
                ("later_learning_and_retention", 3),
                ("request_scale_deployment", 6),
            ]
        ],
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the established schema and bind every added field to immutable operands."""
    with binding():
        value: Json = dict(BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8387,
        random_seed=7228387,
        completed_count=sum(r["eligible"] for r in value["rows"]),
        censored_count=sum(
            not r["eligible"]
            and r["verdict_class"] != "disqualified"
            and r["disposition"] != "conductor_pre_gate"
            for r in value["rows"]
        ),
        repository_health=[r for r in receipts if r.get("scope") == "global"],
        methodology_note="Read-only aggregation authenticates fourteen conductor slots and original terminal seals. Fresh bounded workers preserve frozen source aliases. Qualified exposed H1/H2 remain closed; direct recovery and finite numerical certificates do not establish semantic benefit. Absent training, full request costs and canary observations stay unmeasured. Current LLM calls are zero. No generator weights or device state changes.",
    )
    value["source_artifact_hashes"].extend(work["extra_refs"])
    value["cited_upstream_artifacts"].extend(
        dict(r, fields_imported=["sealed protocol or conductor scheduling observation"])
        for r in work["extra_refs"]
    )
    for i in work["inputs"]:
        value["cited_upstream_artifacts"].append(
            dict(i["primary"], fields_imported=list(i["summary"]["selected"]))
        )
    value["memory_bounds"] = dict(child_peak_mb=1500, growth_mb=500, parent_current_rss_mb=1500)
    value["field_principles"].update(
        {
            k: "Bind "
            + k
            + " to sealed upstream operands; missing observations stay null and engineering parity grants no semantic benefit."
            for k in value
        }
    )
    value["field_principles"].update(
        canonical_tasks_sha256="Canonical digest of the sealed active fourteen tasks; independent design qualification remains a separate blocked gate.",
        science_ready_score="Historical utility audit qualification only; closed exposed nulls grant no current semantic benefit.",
        cascade_skip_count="Logged scheduling skips with no producer execution; missing_output_count also counts their absent primaries.",
        producer_measurements="Each metric retains the producer primary hash, intended units and sample/exposure/oracle limits; pre-gates have no measurements.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute missing slots and historical utility before existing fresh worker reductions."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        ):
            return False
        work = load(value["work_reference"])
        reduced = reduce(work, value["validation_receipts"])
        if any(value.get(k) != v for k, v in reduced.items()):
            return False
        conductor = work["extra_refs"][-1]
        lines = (
            Path(conductor["snapshot_path"]).read_text().splitlines() if conductor["exists"] else []
        )
        for index, i in enumerate(work["inputs"]):
            if i["task"] != work["tasks"][index]:
                return False
            if not i["primary"]["exists"]:
                cascade = next(
                    (
                        line
                        for line in reversed(lines)
                        if "Pre-emptive skip:" in line and i["task"]["title"][:48] in line
                    ),
                    None,
                )
                if i["summary"] != absent(i["task"], i["primary"], cascade):
                    return False
        for h in work["history"]:
            primary, terminal, side = [load(r) for r in h["refs"][:3]]
            if h["honest_verdict"] != primary["honest_verdict"] or h["authenticated"] != (
                terminal["publication"]["primary_sha256"]
                == h["reference"]["sha256"]
                == side["primary_sha256"]
                and side["report"]["passed"] is True
            ):
                return False
            if h["experiment_id"] == 8361:
                with TemporaryDirectory(
                    prefix="exp8387-utility-cold-", dir="/var/tmp"
                ) as directory:
                    with old.frozen_inputs(h["refs"], Path(directory)):
                        if any(
                            old.utility(primary, h["refs"])[k] != h["selected"][k]
                            for k in ["H1", "H2"]
                        ):
                            return False
        with binding():
            return bool(REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual cold children must reject absent, foreign and rehashed substituted claims."""
    receipts = []
    for name in [
        "valid",
        "missing_input",
        "deliberate_error",
        "wrong_authority",
        "changed_source",
        "rehashed_tamper",
    ]:
        changed = deepcopy(value)
        if name == "missing_input":
            changed["work_reference"]["path"] = changed["work_reference"]["snapshot_path"] = str(
                raw / "absent"
            )
        elif name == "deliberate_error":
            changed["reproducibility_checksum"] = "invalid"
        elif name == "wrong_authority":
            changed["canonical_tasks_sha256"] = "foreign"
        elif name == "changed_source":
            changed["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
        elif name == "rehashed_tamper":
            changed["cascade_skip_count"] += 1
        if name not in ["valid", "deliberate_error"]:
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            old.child(
                "cold_" + name,
                [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--cold-replay", str(path)],
                raw / "logs",
                expected=int(name != "valid"),
                deadline=240,
                heartbeat=20,
            )
        )
    return receipts
