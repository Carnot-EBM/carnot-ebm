"""REQ-REPORT-8779: compare frozen sufficiency rules without truth-bearing inputs."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import evidence_set_corpus_8768 as corpus
from carnot.reporting import evidence_set_analysis_8768 as analysis
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v746_retained_evidence import load, reference

Json = dict[str, Any]
ROOT = q.ROOT
NAME, TASK = "experiment_8779_v752_evidence_set_evaluation", "exp8779-evidence-set-evaluation"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/{n}_8779.py"
    for n in ["evidence_set_evaluation", "evidence_evaluation_report", "evidence_evaluation_runner"]
] + [CLI]
TESTS = ["tests/python/test_evidence_set_evaluation_8779.py"]
MODEL_SPECS: list[Json] = []
SEED = original.SEED
UPSTREAM = ROOT / "results" / (q.NAME + ".json")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed phase counts distinguish actual progress from an unresponsive child."""
    print(f"[exp8779] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(path: Path) -> Json:
    """Bind exact producer and sealed operands; rejected historical producers cannot grant readiness."""
    result: Json = dict(passed=False, inputs=[], labels=[], references=[], diagnostics=[])
    diagnostic: Json = dict(
        upstream=q.TASK,
        path=str(path),
        sha256=None,
        check="missing_file",
        field="evidence_methods_ready_score",
        operator="==",
        expected=1,
        observed=None,
    )
    try:
        ref = reference(path)
        diagnostic["sha256"] = ref["sha256"]
        value = load(ref)
        result["references"].append(ref)
        diagnostic.update(check="missing_field", observed=value.get("evidence_methods_ready_score"))
        if "evidence_methods_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["evidence_methods_ready_score"] != 1:
            raise ValueError("readiness")
        diagnostic.update(
            check="rejected_producer",
            field="producer_identity_and_acceptance",
            expected=[8777, q.TASK, True, False],
            observed=[
                value.get(k)
                for k in [
                    "experiment_id",
                    "task_id",
                    "required_checks_passed",
                    "flagged_adversarial",
                ]
            ],
        )
        if diagnostic["observed"] != diagnostic["expected"] or value["verdict_class"] not in {
            "null",
            "circular_positive",
            "positive",
        }:
            raise ValueError("producer")
        diagnostic.update(
            check="frozen_operands",
            field="authenticated_method_and_inputs",
            expected=True,
            observed=False,
        )
        work = load(value["primitive_reference"])
        method = load(value["frozen_method_reference"])
        expected_method = dict(
            original_method=original.method(),
            maximum_evaluator_invocations=5,
            local_over_budget="first five premise checks; any refutation refutes; otherwise insufficient",
            MODEL_SPECS=[],
        )
        if {k: method.get(k) for k in expected_method} != expected_method:
            raise ValueError("method_drift")
        refs = [
            value["primitive_reference"],
            value["frozen_method_reference"],
            work["corpus_reference"],
            work["oracle_reference"],
        ]
        refs += [
            next(r for r in value["code_config_hashes"] if Path(r["path"]) == ROOT / p)
            for p in original.OWNED + q.OWNED
        ]
        for source in refs:
            if reference(Path(source["path"])) != source:
                raise ValueError("source_drift")
        inputs = [
            p
            for p in load(work["corpus_reference"])["rows"]
            if p["split"] in {"replication", "evaluation"}
        ]
        by_id = {p["unit_id"]: p for p in load(work["oracle_reference"])["rows"]}
        labels = [
            dict(unit_id=p["unit_id"], decision=by_id[p["unit_id"]]["decision"]) for p in inputs
        ]
        if len(inputs) != 2880 or len({p["unit_id"] for p in inputs}) != 2880:
            raise ValueError("incomplete_inputs")
        worlds = {(p["world_id"], p["structure"], p["split"]) for p in inputs}
        if len(worlds) != 240 or len({p[1] for p in worlds}) != 240:
            raise ValueError("world_overlap")
        if any(
            corpus.enumerate_truth(p["input"])["decision"] != label["decision"]
            for p, label in zip(inputs, labels, strict=True)
        ):
            raise ValueError("truth_drift")
        result.update(
            passed=True,
            inputs=inputs,
            labels=labels,
            method=method,
            references=result["references"] + refs,
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        result["diagnostics"].append(diagnostic)
    return result


def measure(payloads: list[Json]) -> list[Json]:
    """Only facts/query enter each arm; CPU and transport size describe actual computation."""
    rows = []
    progress("before_measurement", 0, len(payloads) * 3)
    for index, payload in enumerate(payloads):
        original.parse(payload)
        size = len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        for arm in original.ARMS:
            start, cpu = time.monotonic_ns(), time.process_time_ns()
            decision, calls = q.bounded(arm, payload)
            cpu_end, end = time.process_time_ns(), time.monotonic_ns()
            rows.append(
                dict(
                    input_index=index,
                    arm=arm,
                    decision=decision,
                    evaluator_invocations=calls,
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    wall_time_s=(end - start) / 1e9,
                    cpu_time_s=(cpu_end - cpu) / 1e9,
                    input_size_bytes=size,
                    input_sha256=canonical_hash(payload),
                )
            )
        if (index + 1) % 120 == 0:
            progress("measurement", (index + 1) * 3, (len(payloads) - index - 1) * 3)
    progress("after_measurement", len(rows), 0)
    return rows


def audit(inputs: list[Json], decisions: list[Json]) -> bool:
    """Recompute decisions independently of labels; never trust repaired summary hashes."""
    return q.audit(inputs, decisions) and all(
        r["cpu_time_s"] >= 0
        and r["input_sha256"] == canonical_hash(inputs[r["input_index"]]["input"])
        and r["input_size_bytes"]
        == len(
            json.dumps(
                inputs[r["input_index"]]["input"], sort_keys=True, separators=(",", ":")
            ).encode()
        )
        for r in decisions
    )


def score(inputs: list[Json], labels: list[Json], decisions: list[Json]) -> list[Json]:
    """Errors remain complete rows; valid renamed witnesses share their world's denominator."""
    rows = q.score(inputs, labels, decisions)
    for row in rows:
        row.update(
            surface=row["view"],
            independent_truth=row["expected"],
            false_acceptance=row["decision"] == "support" and row["expected"] != "support",
            valid_alternative_rejection=row["expected"] == "support"
            and row["decision"] != "support"
            and row["view"] == "renamed_ordered",
            missing_hop_error=row["condition"] == "missing_required_hop"
            and bool(row["raw_numerator"]),
            conflict_error=row["condition"] == "explicit_conflict" and bool(row["raw_numerator"]),
        )
    return rows


def summarize(rows: list[Json]) -> Json:
    """Analyze strata separately under the unchanged paired world bootstrap and safety rule."""
    result: Json = dict(
        paired_world_rows=[],
        paired_intervals=[],
        confusion_rows=[],
        valid_alternative_rows=[],
        cost_rows=[],
        scientific_promotion=False,
    )
    for split, field in [
        ("replication", "exposed_replication_metrics"),
        ("evaluation", "new_family_metrics"),
    ]:
        local = [r for r in rows if r["split"] == split]
        summary = analysis.summarize([dict(r, split="evaluation") for r in local]) if local else {}
        result[field] = summary
        for key in ["paired_world_rows", "paired_intervals"]:
            result[key] += [dict(r, stratum=split) for r in summary.get(key, [])]
        for arm in original.ARMS:
            group = [r for r in local if r["arm"] == arm]
            result["cost_rows"].append(
                dict(
                    stratum=split,
                    arm=arm,
                    denominator=len(group),
                    evaluator_invocations=sum(r["evaluator_invocations"] for r in group),
                    cpu_time_s=sum(r["cpu_time_s"] for r in group),
                    wall_time_s=sum(r["wall_time_s"] for r in group),
                    input_size_bytes=sum(r["input_size_bytes"] for r in group),
                )
            )
            result["valid_alternative_rows"].append(
                dict(
                    stratum=split,
                    arm=arm,
                    denominator=sum(
                        r["expected"] == "support" and r["view"] == "renamed_ordered" for r in group
                    ),
                    rejections=sum(r["valid_alternative_rejection"] for r in group),
                )
            )
            for truth in original.Decision:
                for decision in original.Decision:
                    result["confusion_rows"].append(
                        dict(
                            stratum=split,
                            arm=arm,
                            truth=truth,
                            decision=decision,
                            numerator=sum(
                                r["expected"] == truth and r["decision"] == decision for r in group
                            ),
                            denominator=sum(r["expected"] == truth for r in group),
                        )
                    )
    result["scientific_promotion"] = bool(
        result["new_family_metrics"].get("scientific_promotion", False)
    )
    return result
