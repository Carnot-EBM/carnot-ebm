"""REQ-REPORT-8517: measure the preserved four-arm protocol using qualified original operands.

Readiness describes durable causal measurements. Historical labels cannot prove
independent generalization, and parameter activity alone cannot prove benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import brier_online_learning_8490 as old
from carnot.reporting import capacity_learning_report_8434 as report
from carnot.reporting import objective_publication_8516 as upstream
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.primary_publication import read_bound_sidecar
from tempfile import TemporaryDirectory
from carnot.verify import objective_stream_8517 as stream

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8517_v732_brier_online_learning", "exp8517-brier-online-learning"
CLI, TEST = stream.CLI, "tests/python/test_brier_online_learning_8517.py"
OWNED = [
    "python/carnot/verify/objective_stream_8517.py",
    "python/carnot/reporting/brier_online_learning_8517.py",
    "python/carnot/reporting/brier_online_runner_8517.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ARMS, SEEDS = stream.ARMS, stream.SEEDS
UPSTREAM_PIN = "sha256:2aa6420f606e6eea82afa9ee8d1bd62e149efd2b608df3ecae1a1f68a82b69df"
read, reference, checksum, gate = old.read, old.reference, old.checksum, old.gate
REQUIRED_CHECKS = upstream.REQUIRED_CHECKS
_REDUCE = report.reductions


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so the supervisor can distinguish work from a stalled process."""
    print(f"[exp8517] phase={phase} completed={completed} pending={pending}", flush=True)


def source() -> Json:
    """Original V726 seals supply all heads/features; no V732 source fit enters learning."""
    plan = read(dict(path=str(old.PLAN_PATH), sha256=old.PLAN_PIN))
    work = read(plan["primitive_reference"])
    for ref in work["source_refs"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_source_hash")
    configs = [
        read(c["configuration"])
        for c in work["cells"]
        if c["arm"] == "frozen" and c["capacity"] == 8
    ]
    registered = methods.protocol()["objective_updates"]
    first = configs[0]
    if (
        [[q["id"] for q in b] for b in first["blocks"]] != registered["blocks"]
        or registered["arms"] != list(ARMS)
        or registered["seeds"] != list(SEEDS)
        or registered["capacity"] != 8
        or registered["learning_rate"] != 0.01
        or registered["l2"] != 0.001
    ):
        raise ValueError("registered_stream_operands")
    return dict(
        heads=[c["head"] for c in configs],
        blocks=first["blocks"],
        panel=dict(
            evaluator_custody=first["evaluator_labels"], trainer_view=first["retention_labels"]
        ),
        source_refs=[
            dict(path=str(old.PLAN_PATH), sha256=old.PLAN_PIN),
            plan["primitive_reference"],
            *work["source_refs"],
        ],
    )


def inspect(root: Path, raw: Path, private: Path) -> Json:
    """Current authority and original historical authority have separate, unchanged consumers."""
    raw.mkdir(parents=True, exist_ok=True)
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        methods=methods,
        upstream=upstream,
        UPSTREAM_TASK=upstream.TASK,
        UPSTREAM_PIN=UPSTREAM_PIN,
    ):
        result = dict(old.inspect(root, raw, private))
    path = root / "results" / (upstream.NAME + ".json")
    if path.is_file():
        value = json.loads(path.read_bytes())
        if value.get("learning_qualification_ready_score") != 1:
            result["failures"].append(
                gate(
                    path,
                    sha256_file(path),
                    "learning_qualification_ready_score",
                    1,
                    value.get("learning_qualification_ready_score"),
                    upstream.TASK,
                )
            )
    for snapshot in result["authority"].get("authority_snapshots", {}).values():
        if snapshot.get("snapshot_path"):
            result["source_refs"].append(
                dict(path=snapshot["snapshot_path"], sha256=snapshot["sha256"])
            )
    return result


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Gate before labels, then retain every intended cell and genuine restart receipt."""
    progress("preconditions", 0, 20)
    started = time.monotonic_ns()
    private.mkdir(parents=True, exist_ok=True)
    checked = inspect(root, raw, private)
    try:
        original = source()
    except (OSError, ValueError, KeyError) as error:
        checked["failures"].append(
            gate(old.PLAN_PATH, old.PLAN_PIN, "original_primitives", True, str(error), TASK)
        )
        raise ValueError("original_primitives_unavailable") from error
    checked["source_refs"].extend(original["source_refs"])
    cells, restarts, receipts = [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            identity = f"s{seed}-c8-{arm}"
            cell: Json = dict(
                identity=identity,
                seed=seed,
                capacity=8,
                arm=arm,
                status="censored",
                missing_reason="required_input_unavailable",
                steps=[],
            )
            cells.append(cell)
            if checked["failures"]:
                continue
            remaining = 3600 - (time.monotonic_ns() - started) / 1e9
            cfg = stream.configuration(
                next(h for h in original["heads"] if h["seed"] == seed),
                original["blocks"],
                seed,
                arm,
                original["panel"]["evaluator_custody"],
                original["panel"]["trainer_view"],
            )
            config_path, directory = (
                raw / "configurations" / (identity + ".json"),
                raw / "cells" / identity,
            )
            atomic_json(config_path, cfg)
            argv = [
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--configuration",
                str(config_path),
                "--worker",
                str(directory),
            ]
            receipt = child(
                identity, argv, raw / "processes", deadline=max(0.01, min(240, remaining))
            )
            receipts.append(receipt)
            cell.update(
                configuration=reference(config_path),
                steps=[reference(p) for p in sorted((directory / "steps").glob("*.json"))],
                status="completed" if receipt["passed"] else "failed",
                missing_reason=None if receipt["passed"] else "owned_child_failed",
            )
            for key, path in [
                ("final", directory / "final.json"),
                ("costs", directory / "costs.json"),
            ]:
                if path.exists():
                    cell[key] = reference(path)
            retention = directory / "retention.json"
            receipts.append(
                child(
                    identity + "-retention",
                    [*argv[:5], "--retention", str(directory), "--operand-output", str(retention)],
                    raw / "processes",
                    deadline=60,
                )
            )
            if retention.exists():
                cell["retention"] = reference(retention)
            if seed == 17:
                resumed = raw / "restarts" / arm
                restart_argv = [*argv[:5], "--worker", str(resumed)]
                for suffix, extras, expected in [("exit", ["--crash"], 73), ("resume", [], 0)]:
                    receipts.append(
                        child(
                            identity + "-" + suffix,
                            [*restart_argv, *extras],
                            raw / "processes",
                            expected=expected,
                            deadline=240,
                        )
                    )
                final = resumed / "final.json"
                equal = (
                    final.exists()
                    and "final" in cell
                    and read(reference(final)) == read(cell["final"])
                )
                restarts.append(
                    dict(
                        arm=arm,
                        seed=17,
                        capacity=8,
                        actual_exit=receipts[-2]["actual_exit"],
                        passed=bool(equal and all(r["passed"] for r in receipts[-2:])),
                        steps=[reference(p) for p in sorted((resumed / "steps").glob("*.json"))],
                        final=reference(final) if equal else None,
                    )
                )
            progress("cell_complete", len(cells), 20 - len(cells))
    work = dict(
        checked,
        cells=cells,
        restart_rows=restarts,
        process_receipts=receipts,
        root=str(root),
        raw=str(raw),
        code_refs=[reference(ROOT / p) for p in OWNED],
        source=original,
        independent_update=report.e.independent_update(),
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        adversarial_findings=[],
    )
    work["process_receipts"] += [
        json.loads(p.read_bytes())
        for p in sorted(raw.rglob("*.receipt.json"))
        if "evaluator_processes" in p.parts
    ]
    work["primitive_files"] = [
        reference(p)
        for p in sorted(raw.rglob("*.json"))
        if any(part in {"cells", "restarts", "configurations", "processes"} for part in p.parts)
    ]
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "stream_rows.json", dict(rows=reductions(work)["rows"]))
    return work


def reductions(work: Json) -> Json:
    """The shipped evaluator keeps sibling and repeated-seed counts distinct from independent questions."""
    with (
        patch.object(report, "operands", lambda root: work["source"]),
        patch.object(report.e, "ARMS", ()),
    ):
        return dict(_REDUCE(work))


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualify completeness independently of a favorable objective contrast."""
    with (
        patch.multiple(
            report,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            OWNED=OWNED,
            TEST=TEST,
            reductions=reductions,
            REQUIRED_CHECKS=REQUIRED_CHECKS,
        ),
        patch.object(report.e, "SEEDS", SEEDS),
    ):
        value = dict(report.build(work, receipts, raw, output))
    reduced = reductions(work)
    admissions = []
    for seed in SEEDS:
        matched = [c for c in work["cells"] if c["seed"] == seed and len(c["steps"]) == 5]
        sequences = [
            [read(p)["sealed_state"]["selected_cluster_ids"] for p in c["steps"]] for c in matched
        ]
        admissions.append(
            dict(seed=seed, passed=len(matched) == 4 and all(s == sequences[0] for s in sequences))
        )
    ready = (
        value["required_checks_passed"]
        and not work["failures"]
        and work["authority"].get("activated") is True
        and len(reduced["rows"]) == 6000
        and len(reduced["retention_rows"]) == 15000
        and all(r["status"] == "completed" for r in reduced["rows"])
        and len(work["restart_rows"]) == 4
        and all(r["passed"] for r in work["restart_rows"])
        and all(r["passed"] for r in admissions)
    )
    verdict = (
        "disqualified"
        if not value["required_checks_passed"]
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    costs = [
        dict(identity=c["identity"], **read(c["costs"])) for c in work["cells"] if c.get("costs")
    ]
    value.update(
        experiment_id=8517,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict=f"complete_{verdict}_v732_brier_online_learning",
        verdict_class=verdict,
        intended_count=6000,
        independent_count=100 if ready else 0,
        objective_stream_ready_score=int(ready),
        learning_stream_ready_score=int(ready),
        stream_rows_reference=reference(raw / "stream_rows.json"),
        restart_checks=work["restart_rows"],
        admission_identity_checks=admissions,
        admission_count=sum(r["admitted"] is True for r in reduced["admission_rows"]),
        unadmitted_count=sum(r["admitted"] is False for r in reduced["admission_rows"]),
        label_updates=sum(len(c["updates"]) for c in costs),
        released_feedback_count=len(reduced["release_rows"]),
        duplicate_feedback_count=0,
        pending_feedback_count=reduced["lost_feedback_count"],
        update_rule_hash=stream.adapter.RULE_HASH,
        update_cost_s=costs,
        external_validation_receipts=work["external_receipts"],
        hardware_path=dict(
            current="sparse CPU updates",
            GPU_NPU="future batching subject to operation support",
            FPGA="future coefficient arithmetic subject to operation support",
            measured_acceleration=None,
        ),
        cited_upstream_artifacts=[
            dict(ref, imported_fields=["original sealed authority, primitives or qualification"])
            for ref in work["source_refs"]
        ],
        future_dependencies=[dict(experiment_id=8518, status="future_dependency_not_observation")],
        methodology_note="Uniform capacity8 admission; registered rate0.01/L2.001; original V726 heads. Predictions precede admitted release. Real exit73 after block2 seed17 every arm. Read-only retention children. Readiness is mechanics; later losses are descriptive exposed-history science, not independent benefit.",
    )
    value["sample_size_budget"].update(
        intended_answer_rows=6000,
        intended_question_units=2000,
        intended_retention_rows=15000,
        capacities=[8],
        arms=list(ARMS),
    )
    value["acceptance_gates"].update(
        qualified_causal_measurements=ready, admission_identity=all(r["passed"] for r in admissions)
    )
    value["field_principles"].update(
        {
            k: "Bound to original operands and actual invocation; mechanics do not establish independent benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        update_cost_s="Actual numerical updates and shipped durable-write barrier spans; audit reconstructions excluded.",
        objective_stream_ready_score="All intended rows and causal/restart checks qualify irrespective of effect sign.",
    )
    archive = raw / "collection_archive.json"
    if archive.exists():
        value["collection_archive_reference"] = reference(archive)
        value["raw_shard_hashes"].append(reference(archive))
        value["field_principles"]["collection_archive_reference"] = (
            "Original node union and collection commands retain exits, durations and log hashes in immutable raw custody."
        )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh replay derives numerical effects from original seals, defeating locally consistent forged hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [
            *work["source_refs"],
            *work["code_refs"],
            *work["primitive_files"],
            *work.get("coverage_shards", []),
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        snapshots = work["authority"].get("authority_snapshots", {})
        if snapshots:
            with TemporaryDirectory(prefix="exp8517-authority-", dir="/var/tmp") as directory:
                cold = Path(directory)
                for role, relative in [
                    ("design", methods.DESIGN),
                    ("active", methods.ACTIVE),
                    ("staged", methods.STAGED),
                ]:
                    ref = snapshots[role]
                    if ref.get("snapshot_path"):
                        target = cold / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
                authority = methods.authority(cold, cold / "authority")
                if authority["activated"] != work["authority"]["activated"]:
                    return False
        upstream_refs = [
            ref for ref in work["source_refs"] if Path(ref["path"]).name == upstream.NAME + ".json"
        ]
        for ref in upstream_refs:
            if ref["sha256"] != UPSTREAM_PIN:
                return False
            original = read(ref)
            required = old.qualification_failures(Path(ref["path"]), ref["sha256"], original)
            if any(f["field"] not in {x["field"] for x in work["failures"]} for f in required):
                return False
            terminal = json.loads(Path(original["terminal_validation_sidecar_path"]).read_bytes())
            if not read_bound_sidecar(
                Path(ref["path"]), Path(terminal["publication"]["sidecar_path"])
            )["report"]["passed"]:
                return False
        if work["source"] != source():
            return False
        if [(c["seed"], c["arm"]) for c in work["cells"]] != [(s, a) for s in SEEDS for a in ARMS]:
            return False
        for cell in work["cells"]:
            if not cell.get("configuration"):
                continue
            cfg = read(cell["configuration"])
            original = work["source"]
            expected = stream.configuration(
                next(h for h in original["heads"] if h["seed"] == cell["seed"]),
                original["blocks"],
                cell["seed"],
                cell["arm"],
                original["panel"]["evaluator_custody"],
                original["panel"]["trainer_view"],
            )
            if cfg != expected:
                return False
            with stream.binding(cfg):
                previous = stream.shipped.state.initial(cfg)
            steps = []
            for ref in cell["steps"]:
                step = read(ref)
                if step["before_state"] != previous:
                    return False
                previous = stream.check_step(cfg, step)
                steps.append(step)
            if cell.get("final") and read(cell["final"]) != previous:
                return False
            if cell.get("retention") and read(cell["retention"]) != stream.shipped.retention(
                cfg, steps
            ):
                return False
            for restart in work["restart_rows"]:
                if cell["seed"] == 17 and restart["arm"] == cell["arm"]:
                    resumed = [read(ref) for ref in restart["steps"]]
                    equal = (
                        [stream.shipped.semantic_step(s) for s in steps]
                        == [stream.shipped.semantic_step(s) for s in resumed]
                        and restart["final"] is not None
                        and read(restart["final"]) == previous
                        and restart["actual_exit"] == 73
                    )
                    if restart["passed"] != equal:
                        return False
            progress(
                "cold_replay_cells", SEEDS.index(cell["seed"]) * 4 + ARMS.index(cell["arm"]) + 1, 0
            )
        if not report.receipts_valid(
            [*work["process_receipts"], *work["external_receipts"], *value["validation_receipts"]],
            work["adversarial_findings"],
        ):
            return False
        raw = Path(work["raw"])
        if read(reference(raw / "stream_rows.json")) != dict(rows=reductions(work)["rows"]):
            return False
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError):
        return False
