"""REQ-REPORT-8592: qualify execution mechanics while keeping old evidence intact."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import time
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from carnot.reporting import causal_brier_runner_8582 as runner
from carnot.reporting import causal_brier_stream_8582 as stream
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT = stream.ROOT
NAME = "experiment_8592_v738_learning_clock_qualification"
TASK = "exp8592-learning-clock-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_learning_clock_8592.py"]
OWNED = [
    "python/carnot/reporting/learning_clock_8592.py",
    "python/carnot/reporting/learning_clock_validation_8592.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = {
    8580: "sha256:7656f6f49b69237528263cd6dc38ac9503e853b1f53d1b5b1037f6510d2892f4",
    8581: "sha256:f60e202794f220440094c940ea5d5e84d3a70b7fc8ba840ecf09fb915bf1896c",
    8582: "sha256:4f7ec92ec04fb3cecc16af7e4c8a87682c63c2e1d575bfbeccdc8b02f1470c78",
}
REQUIRED = {"collection", "ruff_check", "ruff_format", "mypy", "spec", "combine", "coverage"}
ref = stream.old.reference


def digest(value: Json) -> str:
    """Exclude the checksum itself so replay can compare independently rebuilt fields."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def prepare(raw: Path) -> Json:
    """Authenticate old bytes before making a separate current mechanics fixture."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="clock8592-", dir="/var/tmp") as scratch:
        checked = stream.inspect(ROOT, raw, Path(scratch))
    sources = []
    for identity, pin in PINS.items():
        path = next((ROOT / "results").glob(f"experiment_{identity}_*.json"))
        reference = dict(path=str(path), sha256=pin)
        upstream = stream.old.read(reference)
        sources.extend([reference, upstream["primitive_reference"]])
    previous = stream.old.read(sources[-1])
    fixture = deepcopy(previous)
    fixture_raw = raw / "fixture"
    fixture.update(
        raw=str(fixture_raw),
        owned_ready=False,
        receipts=[],
        coverage_complete=False,
        union_passed=False,
    )
    fixture["code_refs"] = [ref(Path(r["path"])) for r in previous["code_refs"]]
    for name in (
        "stream_rows.json",
        "released_feedback.json",
        "retention_rows.json",
        "coefficient_transitions.json",
    ):
        atomic_json(fixture_raw / name, json.loads((Path(previous["raw"]) / name).read_bytes()))
    atomic_json(fixture_raw / "measurement.json", fixture)
    historical = ROOT / "results/raw" / NAME / "failure_reproduction.json"
    reproduction = json.loads(historical.read_bytes())
    head = stream.source()["heads"][0]
    before = runner.deadline_clock
    try:
        runner.deadline_clock = lambda: -100.0
        changed, kernel = stream.stream.adapter.sparse.update(head, np.asarray([0.4] * 9), 1, 0.01)
    finally:
        runner.deadline_clock = before
    rows = [
        dict(
            unit="kernel_clock_independence",
            passed=changed != head and kernel["duration_s"] >= 0,
            duration_s=kernel["duration_s"],
            parameters=109,
            missing_reason=None,
        )
    ]
    paths = [
        *OWNED,
        *TESTS,
        *stream.OWNED,
        *stream.TESTS,
        "python/carnot/verify/sparse_energy_7996.py",
    ]
    work = dict(
        raw=str(raw),
        run_date=datetime.now().strftime("%Y%m%d"),
        started_monotonic_ns=started,
        source_refs=sources,
        current_source_reference=[ref(ROOT / p) for p in paths],
        original_authority=previous["authority"],
        original_nodes=previous["node_union"]["original_node_ids"][:-2],
        historical_code_refs=previous["code_refs"],
        prior_failed_attempts=[
            ref(p)
            for p in (ROOT / "results/raw" / NAME / "invocations").glob("*/aborted_attempt.json")
        ],
        mechanics_fixture=ref(fixture_raw / "measurement.json"),
        failure_reproduction=reproduction,
        clock_independence_rows=rows,
        preconditions=checked["preconditions"],
        failures=checked["failures"],
        receipts=[],
        node_union={},
        coverage_reference=None,
        validator_reference=stream.old.read(sources[0])["current_validator_reference"],
        checkpoint_reference=ref(
            Path(
                stream.old.read(stream.methods.protocol()["original_producers"]["8431"])[
                    "frozen_heads_path"
                ]
            )
        ),
    )
    atomic_json(raw / "work.json", work)
    return work


def qualified(work: Json) -> bool:
    """Require real terminal nodes, covered new statements and durable passing child logs."""
    union = work["node_union"]
    receipts = work["receipts"]
    coverage = stream.old.read(work["coverage_reference"]) if work["coverage_reference"] else {}
    return bool(
        work["failure_reproduction"]["passed"]
        and all(r["passed"] for r in work["clock_independence_rows"])
        and REQUIRED <= {r["name"] for r in receipts if r["passed"]}
        and all(r["passed"] and not r.get("timed_out") for r in receipts)
        and stream.validation.exact_union(
            work["original_nodes"], union.get("original_children", [])
        )
        and stream.validation.exact_union(
            union.get("collected", []), [[r["node_id"] for r in union.get("outcomes", [])]]
        )
        and bool(union.get("outcomes"))
        and all(r["outcome"] == "passed" for r in union["outcomes"])
        and all(not f["summary"]["missing_lines"] for f in coverage.get("files", {}).values())
        and set(OWNED) <= set(coverage.get("files", {}))
        and not any(f["severity"] == "critical" for f in work.get("adversarial_findings", []))
    )


def build(work: Json, output: Path) -> Json:
    """Readiness is an execution qualification and carries no learning benefit."""
    ready = qualified(work) and not work["failures"]
    verdict = (
        "blocked" if qualified(work) and work["failures"] else "null" if ready else "disqualified"
    )
    value = dict(
        experiment_id=8592,
        task_id=TASK,
        milestone="2026.10.738",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v738_learning_clock_qualification",
        verdict_class=verdict,
        inference_substrate="deterministic_cpu_small_energy_head_no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(current_llm_calls=0, model_loads=0, generations=0),
        historical_model_provenance=[
            dict(
                models=["GPT-5-mini", "gemma-2-9b", "llama3-8b", "mistral-7b"],
                scope="imported historical cached labels only",
            )
        ],
        numerical_inference=dict(
            head="spline9", coefficients=109, checkpoint=work["checkpoint_reference"]
        ),
        rows=work["clock_independence_rows"],
        intended_count=1,
        completed_count=1,
        failed_count=int(not all(r["passed"] for r in work["clock_independence_rows"])),
        censored_count=0,
        excluded_count=0,
        independent_count=1,
        sample_size_budget=dict(
            unit="clock isolation control", intended=1, repeated_history_independent_units=0
        ),
        verifier_is_oracle=True,
        exposure_scope="historical exposed mechanics fixture; no benefit measurement",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        learning_execution_ready_score=int(ready),
        required_checks_passed=qualified(work),
        flagged_adversarial=not qualified(work),
        gate_check_summary=work["failures"]
        + (
            []
            if qualified(work)
            else [
                stream.accounting.old.gate(
                    Path(work["raw"]) / "work.json",
                    "required_owned_checks",
                    True,
                    False,
                    upstream=TASK,
                )
            ]
        ),
        acceptance_gates=dict(
            execution_qualification=qualified(work), upstream_authenticated=not work["failures"]
        ),
        validation_receipts=work["receipts"],
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work.get("duration_s", 0),
        phase_spans=work.get("phase_spans", []),
        random_seed=17,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["current_source_reference"],
        raw_shard_hashes=[work["mechanics_fixture"]],
        prior_failed_attempts=work["prior_failed_attempts"],
        cited_upstream_artifacts=work["source_refs"],
        isolated_clock_reference=dict(
            path=str(ROOT / stream.OWNED[2]),
            callable="deadline_clock",
            supervision="real monotonic_ns and SIGALRM",
        ),
        failure_reproduction=work["failure_reproduction"],
        original_node_union=work["node_union"],
        original_authority=work["original_authority"],
        historical_code_refs=work["historical_code_refs"],
        clock_independence_rows=work["clock_independence_rows"],
        current_source_reference=work["current_source_reference"],
        current_validator_reference=work["validator_reference"],
        private_cli_receipts=work.get("private_cli_receipts", []),
        primitive_reference=ref(Path(work["raw"]) / "work.json"),
        publication_output=str(output),
        historical_quarantines=dict(
            exp8582="unchanged clock-regression quarantine",
            exp8584="unchanged methodology quarantine",
            source_null="preserved",
            cold_cost_null="preserved",
            label_rejections="preserved",
            all_prior="unchanged",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8593, 8602)
        ],
        methodology_note="FR-11 continuous self-learning infrastructure qualification. The original sealed spline operands and causal accounting stay fixed. This invocation qualifies clock isolation and existing exposed-history mechanics; it measures no new learning benefit.",
    )
    value["field_principles"] = {
        key: "Bound to this invocation; qualification is separate from scientific benefit."
        for key in value
    }
    for keys, principle in [
        (
            "experiment_id task_id milestone run_date",
            "Bind the producer and planning slot to the supplied calendar date.",
        ),
        (
            "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget",
            "Count the actual clock control once; replayed historical observations add no independent learning samples.",
        ),
        (
            "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts numerical_inference historical_model_provenance",
            "Separate zero current language-model work from real CPU energy-head evaluation and imported generator names.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score",
            "Exposed and oracle-dependent mechanics cannot establish unseen-data or learning benefit; both scores stay zero.",
        ),
        (
            "learning_execution_ready_score required_checks_passed flagged_adversarial acceptance_gates gate_check_summary",
            "Readiness requires all owned checks and coverage; failed ownership disqualifies and unchanged upstream absence blocks.",
        ),
        (
            "duration_s phase_spans preconditions_checked isolated_clock_reference",
            "Use real clocks, writable disk scratch and independent supervision; only the orchestration deadline is injectable.",
        ),
        (
            "source_artifact_hashes code_config_hashes raw_shard_hashes current_source_reference current_validator_reference historical_code_refs cited_upstream_artifacts primitive_reference reproducibility_checksum",
            "Authenticate exact current operands while retaining original validator, accounting and quarantined historical hashes.",
        ),
        (
            "validation_receipts private_cli_receipts original_node_union failure_reproduction terminal_validation_sidecar_path adversarial_findings prior_failed_attempts",
            "Retain original identities, commands, actual exits and log hashes; killed or failed attempts are diagnostics and cannot qualify.",
        ),
    ]:
        value["field_principles"].update({key: principle for key in keys.split()})
    value["reproducibility_checksum"] = digest(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild summaries from pinned operands; repaired outer hashes cannot add readiness."""
    try:
        value = json.loads(path.read_bytes())
        work = stream.old.read(value["primitive_reference"])
        expected_sources = [
            dict(
                path=str(next((ROOT / "results").glob(f"experiment_{identity}_*.json"))), sha256=pin
            )
            for identity, pin in PINS.items()
        ]
        if work["source_refs"][::2] != expected_sources:
            return False
        for reference in [
            *work["source_refs"],
            *work["current_source_reference"],
            work["mechanics_fixture"],
        ]:
            stream.old.read(reference) if reference in work["source_refs"] else ref(
                Path(reference["path"])
            )
            if ref(Path(reference["path"])) != reference:
                return False
        previous = stream.old.read(work["source_refs"][-1])
        fixture = deepcopy(previous)
        fixture_raw = Path(work["mechanics_fixture"]["path"]).parent
        fixture.update(
            raw=str(fixture_raw),
            owned_ready=False,
            receipts=[],
            coverage_complete=False,
            union_passed=False,
        )
        fixture["code_refs"] = [ref(Path(r["path"])) for r in previous["code_refs"]]
        if stream.old.read(work["mechanics_fixture"]) != fixture:
            return False
        for name in (
            "stream_rows.json",
            "released_feedback.json",
            "retention_rows.json",
            "coefficient_transitions.json",
        ):
            if (fixture_raw / name).read_bytes() != (Path(previous["raw"]) / name).read_bytes():
                return False
        if work["original_nodes"] != previous["node_union"]["original_node_ids"][:-2]:
            return False
        if any(not r["passed"] for r in work["clock_independence_rows"]):
            return False
        for receipt in work["receipts"]:
            if not receipt["passed"] or receipt.get("timed_out"):
                return False
            for key in ("stdout", "stderr"):
                if ref(Path(receipt[key + "_path"]))["sha256"] != receipt[key + "_sha256"]:
                    return False
        return bool(value == build(work, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Keep actual warning exits and findings from the unchanged external consumers."""
    import sys

    from scripts.adversarial_verify import verify_artifact

    flags = verify_artifact(path)["flags"]
    commands = [
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)], int(bool(flags))),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [CLI, "--cold-replay", str(path)], 0),
    ]
    receipts = [
        child(
            name,
            [sys.executable, "-u", *argv],
            raw / "terminal_logs",
            expected=expected,
            deadline=180,
        )
        for name, argv, expected in commands
    ]
    return dict(
        passed=all(r["passed"] for r in receipts)
        and not any(f["severity"] == "critical" for f in flags),
        adversarial_findings=flags,
        receipts=receipts,
    )


def run(output: Path, run_date: str, *, work_path: Path | None = None) -> int:
    """Run bounded qualification once; private fixture publication never reruns training."""
    from carnot.reporting.learning_clock_validation_8592 import execute
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = json.loads(work_path.read_bytes()) if work_path else prepare(raw)
    if not work_path:
        execute(work)
    work["run_date"] = run_date
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [dict(phase="qualification", duration_s=work["duration_s"])]
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    preliminary = raw / "preterminal.json"
    atomic_json(preliminary, build(work, output))
    work["adversarial_findings"] = verify_artifact(preliminary)["flags"]
    atomic_json(raw / "work.json", work)
    value = build(work, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    selected = reader_receipt(
        TASK,
        output.parent,
        field="learning_execution_ready_score",
        expected=value["learning_execution_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected),
    )
    progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Validate calendar dates and keep the independent wall deadline outside injected clocks."""
    print("[exp8592] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8 or (
            args.private_output and args.private_output.resolve().is_relative_to(ROOT)
        ):
            return 2
    except ValueError:
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or ROOT / "results" / (NAME + ".json"),
            args.date,
            work_path=args.work,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
