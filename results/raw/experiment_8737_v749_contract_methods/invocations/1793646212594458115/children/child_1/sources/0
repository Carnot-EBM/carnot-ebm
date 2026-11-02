"""REQ-REPORT-8737: register methods without promoting historical execution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v744_contract_methods as original
from carnot.reporting import v748_collection_contract as historical
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v746_retained_evidence import reference, load, aggregate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8737_v749_contract_methods",
    "exp8737-contract-methods",
    "2026.11.749",
)
DESIGN, ACTIVE, STAGED = historical.m.DESIGN, historical.m.ACTIVE, historical.m.STAGED
PIN = "4a6d677f74b3db5988f4f2dc2aee1ce023d2d132e9aac39ccd645c7fc2611787"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so a parent can distinguish custody work from a stall."""
    print(f"[exp8737] phase={phase} completed={completed} pending={pending}", flush=True)


def digest(tasks: list[Json]) -> str:
    """Use the design's exact canonical encoding, independently of mutable copies."""
    return hashlib.sha256(
        json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def encode(prompt: str) -> str:
    """Only conductor literal braces may change; substitution fields stay literal."""
    return re.sub(
        r"(?<!\{)\{([^{}]+)\}(?!\})",
        lambda x: x[0] if x[1] in {"project_root", "date"} else "{{" + x[1] + "}}",
        prompt,
    )


def inspect(path: Path, field: str, expected: Any, upstream: str = TASK) -> Json:
    """Missing operands and measured zero must produce different gate evidence."""
    value = json.loads(path.read_bytes()) if path.is_file() else None
    observed = value.get(field, "missing_field") if value is not None else "missing_file"
    return dict(
        upstream=upstream,
        path=str(path),
        sha256=reference(path)["sha256"] if path.is_file() else None,
        check="required_field",
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=observed == expected,
    )


def authority(root: Path) -> Json:
    """Compare complete sealed tasks and the visible table without changing a queue."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        stage = root / STAGED
        planned = yaml.safe_load(stage.read_bytes()) if stage.is_file() else active
        supplied = [planned] if active["milestone"] == "2026.11.748" else [planned, active]
        copies = []
        if active["milestone"] == "2026.11.748":
            copies.append(digest(active["tasks"]) == historical.m.PIN)
        for roadmap in supplied:
            normalized = deepcopy(roadmap["tasks"])
            for actual, expected in zip(normalized, tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
            copies.append(roadmap["milestone"] == MILESTONE and normalized == tasks)
        rows = []
        for i, task in enumerate(tasks):
            matched = (
                i < len(table)
                and table[i]
                == dict(
                    order=i + 1, **{k: task[k] for k in ["id", "title", "phase", "deliverable"]}
                )
                and task["id"].startswith(f"exp{8737 + i}-")
                and all(copies)
            )
            rows.append(
                dict(
                    unit_id=task["id"],
                    arm="task_contract",
                    status="completed",
                    absolute_metric=int(matched),
                    raw_numerator=int(matched),
                    raw_denominator=1,
                    censored=False,
                    excluded=False,
                    effective_independent_groups=0,
                )
            )
        ready = (
            len(rows) == len(table) == 12
            and all(r["absolute_metric"] for r in rows)
            and digest(tasks) == PIN
            and f"Canonical full-task SHA-256: `{PIN}`" in text
            and "<!-- V749_TASK_CONTRACT_END -->" in text
        )
        refs = [
            reference(root / p) for p in [DESIGN, ACTIVE] + ([STAGED] if stage.is_file() else [])
        ]
        return dict(ready=ready, rows=rows, tasks=tasks, visible_table=table, references=refs)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return dict(ready=False, rows=[], tasks=[], visible_table=[], references=[])


def mutations() -> list[Json]:
    """Fresh full-task mutations must fail even when a design is rehashed."""
    text = (ROOT / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    rows = []
    with TemporaryDirectory(prefix="exp8737-contract-") as directory:
        root = Path(directory)
        path = root / DESIGN
        path.parent.mkdir(parents=True)
        for mode in [
            "valid",
            "encoded",
            "prompt",
            "agent_type",
            "model",
            "routing",
            "gated_on",
            "deliverable",
            "omission",
            "reorder",
            "table",
            "digest",
            "rehashed",
        ]:
            changed, body = deepcopy(tasks), text
            if mode == "encoded":
                for task in changed:
                    task["prompt"] = encode(task["prompt"])
            elif mode in {"prompt", "agent_type", "model", "routing", "gated_on", "deliverable"}:
                changed[0][mode] = "forged"
            elif mode == "omission":
                changed.pop()
            elif mode == "reorder":
                changed.reverse()
            elif mode == "table":
                body = body.replace("| 1 |", "| 99 |", 1)
            elif mode == "digest":
                body = body.replace(PIN, "0" * 64)
            elif mode == "rehashed":
                changed[0]["prompt"] += "forged"
                body = re.sub(
                    r"(```json\s*).*?(\s*```)",
                    lambda x: x[1] + json.dumps(dict(milestone=MILESTONE, tasks=changed)) + x[2],
                    body,
                    count=1,
                    flags=re.S,
                ).replace(PIN, digest(changed))
            path.write_text(body)
            (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
            observed, expected = authority(root)["ready"], mode in {"valid", "encoded"}
            rows.append(
                dict(
                    mutation=mode, expected=expected, observed=observed, passed=observed == expected
                )
            )
            progress("contract_mutation", len(rows), 13 - len(rows))
    return rows


def learning() -> Json:
    """Authenticate unchanged numerical methods and inputs, independently of execution."""
    p = original.protocol()
    partition = load(p["cohort_partition_reference"])
    inventory = load(p["test_inventory_reference"])
    operands = load(p["complete_learning_source_reference"])
    history = load(partition["historical_primitive_reference"])
    refs = (
        [reference(ROOT / original.PROTOCOL)]
        + [
            p[k]
            for k in [
                "learning_protocol_reference",
                "test_inventory_reference",
                "cohort_partition_reference",
                "complete_learning_source_reference",
                "head_checkpoint_reference",
                "learning_data_reference",
            ]
        ]
        + p["learning_label_references"]
        + operands["source_refs"]
        + partition["source_references"]
        + [x["source"] for x in inventory["assertion_sources"].values()]
    )
    checked = [
        dict(
            r,
            passed=Path(r["path"]).is_file()
            and reference(Path(r["path"]))["sha256"] == r["sha256"],
        )
        for r in refs
    ]
    anchors = history["mapping"]["executable_statements"]
    return dict(
        ready=all(r["passed"] for r in checked)
        and len(partition["original_node_ids"]) == 64
        and len(partition["original_runner_node_ids"]) == 27
        and len(anchors) == 83,
        references=refs,
        input_checks=checked,
        original_node_count=len(partition["original_node_ids"]),
        original_node_ids=partition["original_node_ids"],
        original_runner_node_ids=partition["original_runner_node_ids"],
        statement_anchor_count=len(anchors),
        original_statement_anchors=anchors,
        original_inventory=inventory,
        current_source_dependencies=partition["source_references"],
        features=p["features"],
        coefficient_count=109,
        objective_updates=p["objective_updates"],
        objective_audit=p["objective_audit"],
        historical_quarantines=p["historical_quarantines"],
        execution_authorized=False,
        numerical_training_this_invocation=False,
        delayed_feedback_observations=[
            "pending_occupancy",
            "admission",
            "realized_delay",
            "lost_feedback",
        ],
        scheduler_changed=False,
    )


def upstreams() -> Json:
    """Authenticate raw successful children separately from a failed parent disposition."""
    from carnot.reporting import invocation_clock_core_8724 as clock
    from carnot.reporting.v748_capstone import PINS

    values, refs = {}, []
    for number in [8723, 8724, 8729, 8731, 8732, 8733, 8736]:
        path = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        pin = (
            "0920d4fc3ce9c76c2287a8b6eb43a548fee95dba85eccadd834c879fcd7c081e"
            if number == 8736
            else PINS[number]
        )
        ref = dict(path=str(path), sha256="sha256:" + pin)
        values[number] = load(ref)
        refs += [ref, values[number]["primitive_reference"]]
        load(refs[-1])
    collection = values[8723]
    validation = load(collection["validation_reference"])
    refs.append(collection["validation_reference"])
    custody = aggregate(
        validation["retained_child_rows"], validation["all_child_aggregate"]["node_ids"]
    )
    core = values[8724]
    children = [r for r in core["retained_child_rows"] if r["kind"] == "A"]
    lines = clock.e.core.attest(children, [s for r in children for s in r["sources"]])
    for row in children:
        refs += [f["reference"] for f in row["files"]] + [row["seal_reference"]] + row["sources"]
    smoke = next(r for r in core["validation_receipts"] if r["name"] == "deadline_smoke")
    for stream in ["stdout", "stderr"]:
        ref = dict(path=smoke[stream + "_path"], sha256=smoke[stream + "_sha256"])
        assert reference(Path(ref["path"]))["sha256"] == ref["sha256"]
        refs.append(ref)
    preserved = ROOT / "openspec/change-proposals/research-roadmap-v748-preserved-20261102.md"
    historical_tasks = parse_design(preserved.read_text(), milestone="2026.11.748")[1]
    ready = (
        custody["passed"]
        and collection["contract_ready_score"] == 1
        and len({n for r in children for n in r["node_ids"]}) == 39
        and not smoke["passed"]
        and smoke["timed_out"]
        and smoke["exit_code"] == -9
        and digest(historical_tasks) == historical.m.PIN
    )
    refs.append(reference(preserved))
    return dict(
        ready=ready,
        references=refs,
        collection_custody=custody,
        core=dict(
            passed_node_count=39,
            raw_successful_children=children,
            attested_executed_lines=lines,
            deadline_smoke_passed=smoke["passed"],
            deadline_smoke=smoke,
            learning_execution_ready_score=core["learning_core_ready_score"],
            primary_verdict=core["honest_verdict"],
        ),
        resident_parity=values[8731]["resident_update_ready_score"],
        durability_limit=values[8732]["durability_probe"],
        installed_profile=values[8733]["next_optimization_evidence"],
        preserved_nulls=values[8736]["preserved_nulls"],
    )


def cost_methods(tasks: list[Json], prior: Json) -> Json:
    """Bind fixed future comparisons to observations; register no new performance."""
    return dict(
        task_objects=[t for t in tasks if t["id"].startswith(("exp8743-", "exp8744-", "exp8745-"))],
        evidence=dict(
            durability_limit=prior["durability_limit"], installed_profile=prior["installed_profile"]
        ),
        feature_reuse=dict(
            batch_sizes=[1, 4, 16],
            providers=["python", "native"],
            ownership="request-local validated feature vector reused for score and digest",
            preserve=[
                "mutation rejection",
                "configuration identity",
                "malformed frames",
                "stale success rejection",
                "installed extension",
                "original imports",
            ],
        ),
        installed_cost=dict(
            batch=16,
            panels=16,
            requests_per_panel=64,
            repeats=10,
            cells=40,
            instrumentation=False,
            boundary="parent startup through shutdown",
            bootstrap_draws=10000,
            repeat_cluster="paired process repeat",
            benefit_lower95_gt=1.05,
            p95_ms_lte=50,
            p95_regression_lte=0.05,
            nfr01_lower95_gte=10,
            compare_matching_optimization=True,
        ),
        durable_cost=dict(
            offsets=[0, 2032, 4064],
            window_updates=32,
            arms=3,
            repeats=10,
            processes=90,
            measured_updates=2880,
            budget_s=3000,
            reserve_fraction=0.25,
            admission_probe_updates=4,
            prefix_costs_excluded_but_reported=True,
            start_states="qualified original 4096-update parity trajectory",
            restart_after_update=2048,
            durability=[
                "file fsync",
                "atomic replace",
                "directory fsync",
                "acknowledgment after commit",
            ],
            bootstrap_draws=10000,
            cluster="ten paired repeats retaining all arms and windows",
            equal_window_weight=True,
            throughput_lower95_gt=1.05,
            p95_regression_lte=0.05,
            complete_122880_update_workload="blocked; bounded study cannot replace it",
        ),
        measured_this_invocation=False,
    )
