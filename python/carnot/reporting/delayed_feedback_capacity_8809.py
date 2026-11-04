"""REQ-REPORT-8809: qualify measured queue interventions before interpreting their loss."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from types import FunctionType
from typing import Any

from carnot.learning.delayed_feedback_capacity_8809 import (
    arms,
    cell,
    census,
    controls,
    interval,
    manifest,
    semantic,
    summarize,
)
from carnot.learning import causal_template_streams_8804 as s
from carnot.reporting import causal_template_streams_8804 as base
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8809_v754_delayed_feedback_capacity"
CLI = f"scripts/experiments/{NAME}.py"
UPSTREAM = base.UPSTREAM
OWNED = [
    "python/carnot/learning/delayed_feedback_capacity_8809.py",
    "python/carnot/reporting/delayed_feedback_capacity_8809.py",
    CLI,
]
TESTS = ["tests/python/test_delayed_feedback_capacity_8809.py"]
reference = base.reference
validate = FunctionType(base.validate.__code__, dict(vars(base), OWNED=OWNED, TESTS=TESTS))


def terminal_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Keep the shared validator intact while allowing its full census scan to finish."""
    if name == "adversarial":
        kwargs["deadline"] = 300
    return dict(base.child(name, argv, logs, **kwargs))


terminal_checks = FunctionType(
    base.terminal_checks.__code__, dict(vars(base), CLI=CLI, child=terminal_child)
)


def preflight(raw: Path, plan: Json) -> Json:
    """Reuse exact source and authority checks; project one complete warm cell, not repeated imports."""
    started = time.monotonic()
    checker = FunctionType(base.preflight.__code__, dict(vars(base), UPSTREAM=UPSTREAM))
    pre: Json = checker(raw, intended_count=0, workers=1)
    raw.chmod(0o700)
    if pre["passed"]:
        s.progress("capacity_probe_before", 0, 1)
        head = s.heads(pre["upstream"], [11])[0][0]
        begin = time.monotonic()
        probe = cell(30, 11, 4, 8, "FIFO", head, plan["queries"])
        elapsed = time.monotonic() - begin
        atomic_json(raw / "capacity_probe.json", probe)
        pre["probe_refs"].append(reference(raw / "capacity_probe.json"))
        pre["numerical_training_calls"] += len(probe["applied_rows"])
        pre["measured_cheap_cell_s"] = elapsed
        pre["projected_work_s"] = elapsed * plan["streams"] * len(plan["seeds"]) * len(arms())
        pre["checked"]["projected_time"] = (
            time.monotonic() - started + pre["projected_work_s"] + 900 < 4800
        )
        pre["passed"] = all(pre["checked"].values())
        s.progress("capacity_probe_after", 1, 0)
    return pre


def run(raw: Path, date: str, *, plan: Json | None = None) -> Json:
    """Retain every scheduled outcome and all primitive causal events before qualification."""
    started = time.monotonic()
    frozen = plan or manifest()
    s.progress("capacity_preconditions_before")
    pre = preflight(raw, frozen)
    s.progress("capacity_preconditions_after")
    work: Json = dict(plan=frozen, cells=[])
    if pre["passed"]:
        heads, calls = s.heads(pre["upstream"], frozen["seeds"])
        assert calls == 0
        total = frozen["streams"] * len(heads) * len(arms())
        for stream in range(frozen["streams"]):
            for head in heads:
                for d, c, policy in arms():
                    work["cells"].append(
                        cell(stream, head["seed"], d, c, policy, head, frozen["queries"])
                    )
            s.progress("capacity_stream_complete", len(work["cells"]), total - len(work["cells"]))
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "preconditions.json", {k: v for k, v in pre.items() if k != "upstream"})
    rows = (
        [q for c in work["cells"] for q in c["per_query_rows"]]
        if pre["passed"]
        else census(frozen, "external_precondition_failed")
    )
    gates = [pre["gate"]] + [
        dict(
            check=k,
            upstream="exp8809-preconditions",
            artifact_field="checked." + k,
            expected=True,
            observed=v,
            passed=v,
            **reference(raw / "preconditions.json"),
        )
        for k, v in pre["checked"].items()
    ]
    measured = summarize(work["cells"])
    value = dict(
        experiment_id=8809,
        task_id="exp8809-delayed-feedback-capacity",
        milestone="2026.11.754",
        run_date=date,
        verdict_class="disqualified" if pre["passed"] else "blocked",
        honest_verdict="complete_disqualified_pending_validation"
        if pre["passed"]
        else "complete_blocked_preconditions",
        inference_substrate="cpu_energy_training",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        verifier_is_oracle=True,
        numerical_training_calls=pre["numerical_training_calls"]
        + sum(len(c["applied_rows"]) for c in work["cells"]),
        duration_s=time.monotonic() - started,
        random_seed=8809,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=pre["sources"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            reference(raw / "preconditions.json"),
            pre["authority_reference"],
        ]
        + pre["probe_refs"],
        preconditions_checked={k: v for k, v in pre.items() if k != "upstream"},
        phase_spans=[
            dict(phase="custody_and_capacity_streams", duration_s=time.monotonic() - started)
        ],
        rows=rows,
        per_query_rows=rows,
        intended_count=len(census(frozen, "pending")),
        completed_count=len(rows) if pre["passed"] else 0,
        censored_count=0 if pre["passed"] else len(rows),
        failed_count=0,
        excluded_count=0,
        independent_count=frozen["streams"] if pre["passed"] else 0,
        n_samples=frozen["streams"] if pre["passed"] else 0,
        gate_check_summary=gates,
        acceptance_gates=dict(
            preconditions=pre["passed"],
            complete_rows=pre["passed"]
            and len(rows) == frozen["streams"] * len(frozen["seeds"]) * 15 * frozen["queries"],
            causal_controls=bool(controls(work["cells"])["negative_control"]),
            owned_validation=False,
        ),
        required_checks_passed=False,
        owned_statement_coverage=dict(complete=False),
        validation_receipts=[],
        exposure_scope="constructed_signed_graphs_exact_verifier_labels; fresh resets; no independent generalization",
        independent_generalization_score=0,
        capacity_rows_ready_score=0,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8802,
                sha256=pre["gate"]["sha256"],
                fields_imported=[
                    "memory_execution_ready_score",
                    "calibrated_head_parameters",
                    "code_config_hashes",
                    "source_artifact_hashes",
                    "raw_shard_hashes",
                ],
            )
        ],
        theorem_scope="heuristic comparison; no OCO regret guarantee",
        field_principles={},
        protocol_scope="production" if plan is None else "private_fixture",
        queue_controls=controls(work["cells"]),
        numerical_training_calls_scope="actual preflight compilation/probes and admitted feedback; tests and cold replay recompute separately",
        methodology_note="Unweighted FIFO and uniform random eviction are heuristic comparators. Release precedes arrival. Predictions pay five checks and templates are bounded at32. Exact verifier labels make any favorable effect circular_positive.",
        global_reconciliation_debt="Conductor owns ops and BMAD reconciliation; whole-suite health remains separate from owned qualification.",
        **measured,
    )
    for key in [
        "overflow_event_rows",
        "admission_rows",
        "feedback_loss_rows",
        "release_rows",
        "applied_rows",
        "retention_rows",
        "queue_cost_rows",
    ]:
        value[key] = [q for c in work["cells"] for q in c[key]]
    value["field_principles"] = dict.fromkeys(
        value,
        "Exact frozen or measured operand; nested conditions add no independent streams and exact labels establish no independent skill.",
    )
    return value


def replay(path: Path) -> Json:
    """Rebuild each queue and numerical update; rehashing a false reduction cannot qualify it."""
    try:
        value = json.loads(path.read_bytes())
        refs = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        if not all(sha256_file(Path(ref["path"])) == ref["sha256"] for ref in refs):
            return dict(passed=False, reason="source_hash_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        plan = work["plan"]
        expected = (
            manifest()
            if value["protocol_scope"] == "production"
            else manifest(plan["streams"], plan["queries"], plan["seeds"])
        )
        passed = plan == expected
        rows = (
            [q for c in work["cells"] for q in c["per_query_rows"]]
            if value["preconditions_checked"]["passed"]
            else census(plan, "external_precondition_failed")
        )
        passed &= value["rows"] == value["per_query_rows"] == rows
        passed &= value["reproducibility_checksum"] == canonical_hash(work)
        passed &= value["intended_count"] == len(census(plan, "pending"))
        passed &= value["completed_count"] == sum(q["status"] == "completed" for q in rows)
        passed &= value["censored_count"] == sum(q["status"] == "censored" for q in rows)
        passed &= value["failed_count"] == value["excluded_count"] == 0
        passed &= value["independent_count"] == (plan["streams"] if work["cells"] else 0)
        passed &= value["numerical_training_calls"] == value["preconditions_checked"][
            "numerical_training_calls"
        ] + sum(len(c["applied_rows"]) for c in work["cells"])
        passed &= all(value[k] == v for k, v in summarize(work["cells"]).items())
        passed &= value["queue_controls"] == controls(work["cells"])
        for key in [
            "overflow_event_rows",
            "admission_rows",
            "feedback_loss_rows",
            "release_rows",
            "applied_rows",
            "retention_rows",
            "queue_cost_rows",
        ]:
            passed &= value[key] == [q for c in work["cells"] for q in c[key]]
        ready = int(
            all(value["acceptance_gates"].values())
            and value["owned_statement_coverage"]["complete"]
        )
        passed &= (
            value["capacity_rows_ready_score"] == ready == int(value["required_checks_passed"])
        )
        if ready:
            coverage = value["owned_statement_coverage"]
            passed &= coverage == base.coverage_summary(
                Path(coverage["combined_reference"]["path"]), OWNED
            )
            passed &= {r["name"] for r in value["validation_receipts"]} == {
                "owned_pytest",
                "private_E2E_018_021",
                "ruff_check",
                "ruff_format",
                "strict_mypy",
                "spec_coverage",
            }
            for receipt in value["validation_receipts"]:
                saved = json.loads(
                    Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
                )
                passed &= receipt == saved and receipt["passed"]
                passed &= all(
                    sha256_file(Path(receipt[k + "_path"])) == receipt[k + "_sha256"]
                    for k in ("stdout", "stderr")
                )
        if not value["preconditions_checked"]["passed"]:
            passed &= (
                not work["cells"]
                and value["verdict_class"] in {"blocked", "disqualified"}
                and not ready
            )
        else:
            upstream = json.loads(Path(value["source_artifact_hashes"][0]["path"]).read_bytes())
            passed &= value["source_artifact_hashes"][0]["sha256"] == base.UPSTREAM_PIN
            heads = {h["seed"]: h for h in upstream["calibrated_head_parameters"]}
            roster = [
                (i, seed, d, c, p)
                for i in range(plan["streams"])
                for seed in plan["seeds"]
                for d, c, p in arms()
            ]
            passed &= [
                (
                    int(c["cluster_id"].split("-")[-1]),
                    c["seed"],
                    c["delay"],
                    c["capacity"],
                    c["policy"],
                )
                for c in work["cells"]
            ] == roster
            for i, (saved, (stream, seed, d, c, policy)) in enumerate(zip(work["cells"], roster)):
                reconstructed = cell(stream, seed, d, c, policy, saved["head"], plan["queries"])
                passed &= saved["head"] == heads[seed]
                passed &= semantic(saved) == semantic(reconstructed)
                passed &= all(q["sealed_ns"] <= q["scored_ns"] for q in saved["per_query_rows"])
                cost = saved["queue_cost_rows"][0]
                passed &= cost["duration_s"] == (cost["ended_ns"] - cost["started_ns"]) / 1e9
                passed &= (
                    cost["tail_drain_s"] == (cost["tail_ended_ns"] - cost["tail_started_ns"]) / 1e9
                )
                passed &= 0 <= cost["queue_service_s"] <= cost["duration_s"]
                passed &= all(
                    cost[k] == reconstructed["queue_cost_rows"][0][k]
                    for k in (
                        "peak_pending_count",
                        "pending_after_drain",
                        "template_count",
                        "tail_applied",
                    )
                )
                if (i + 1) % 15 == 0:
                    s.progress("capacity_cold_replay", i + 1, len(roster) - i - 1)
        return dict(passed=bool(passed), reconstructed_rows=len(rows))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_failure")


def qualify(value: Json, raw: Path, output: Path) -> None:
    """Owned failures disqualify even blocked work; favorable exact-label effects stay circular."""
    started = time.monotonic()
    checked = validate(raw)
    value["validation_receipts"] = checked["receipts"]
    value["owned_statement_coverage"] = checked["coverage"]
    value["acceptance_gates"]["owned_validation"] = checked["passed"]
    ready = all(value["acceptance_gates"].values())
    value["capacity_rows_ready_score"] = int(ready)
    value["required_checks_passed"] = ready
    verdict = (
        "disqualified"
        if not checked["passed"]
        else "blocked"
        if not value["acceptance_gates"]["preconditions"]
        else "circular_positive"
        if ready and value["useful_pending_feedback_policy_score"]
        else "null"
        if ready
        else "disqualified"
    )
    value["verdict_class"] = verdict
    suffix = (
        "useful_pending_feedback_policy"
        if value["useful_pending_feedback_policy_score"]
        else "no_corrected_useful_policy_effect"
        if value["queue_controls"]["overflow_positive"]
        else "uninformative_capacity"
    )
    value["honest_verdict"] = f"complete_{verdict}_{suffix}"
    value["phase_spans"].append(
        dict(phase="owned_validation", duration_s=time.monotonic() - started)
    )
    value["duration_s"] += time.monotonic() - started
    health = ROOT / "results/raw" / NAME / "global-health/full_python_once.receipt.json"
    if health.is_file():
        value["repository_health_reference"] = reference(health)
    value["field_principles"] = dict.fromkeys(
        value,
        "Exact measured or frozen operand; heuristic policies and exact labels imply no OCO guarantee or independent generalization.",
    )
    atomic_json(raw / "prepublication.json", value)
    try:
        report = publish_primary(output, value, lambda candidate: terminal_checks(candidate, raw))
    except ValueError:
        atomic_json(raw / "rejected_candidate.json", value)
        raise
    atomic_json(raw / "publication.json", report)
    s.progress("capacity_publication_complete", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Keep the standalone entry point small and replay independent of its working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        result = replay(args.replay)
        print(json.dumps(result), flush=True)
        return int(not result["passed"])
    raw = args.raw or ROOT / "results/raw" / NAME / "invocations" / str(time.time_ns())
    value = run(raw, args.date)
    qualify(value, raw, args.output)
    print(
        json.dumps(
            {k: value[k] for k in ["verdict_class", "completed_count", "capacity_rows_ready_score"]}
        ),
        flush=True,
    )
    return 0
