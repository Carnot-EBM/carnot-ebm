"""REQ-REPORT-8641: qualify a new execution without changing historical failures."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import deadline_statement_8616 as original
from carnot.reporting import learning_clock_8592 as base
from carnot.reporting import learning_custody_data_8641 as custody
from carnot.reporting.v742_contract_methods import protocol
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT, ref, read = base.ROOT, custody.ref, base.stream.old.read
NAME = "experiment_8641_v742_learning_execution_custody"
TASK = "exp8641-learning-execution-custody"
METHODS = "experiment_8640_v742_contract_methods"
METHODS_PIN = "sha256:7abca18566c145b3b197d017fb6a955bf047952ed4325460ba2147c749b33608"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/{p}_8641.py"
    for p in ("learning_custody_data", "learning_execution_custody", "learning_custody_validation")
] + [CLI]
TESTS = [
    f"tests/python/test_{p}_8641.py"
    for p in ("learning_custody_data", "learning_execution_custody", "learning_custody_validation")
]
MODEL_SPECS: list[Json] = []


def upstream(root: Path) -> Json:
    """Separate absent fields from observed zero and authenticate publication independently."""
    path = root / "results" / (METHODS + ".json")
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    reference = ref(path) if path.is_file() else dict(path=str(path), sha256=None)
    failures = []
    if path.is_file() and reference["sha256"] != METHODS_PIN:
        failures.append(
            dict(
                check="upstream_sha256",
                upstream="exp8640-contract-methods",
                **reference,
                field="sha256",
                operator="==",
                expected=METHODS_PIN,
                observed=reference["sha256"],
                passed=False,
            )
        )
    for field in ("methods_ready_score", "learning_operands_ready_score"):
        if value.get(field) != 1:
            failures.append(
                dict(
                    check=field,
                    upstream="exp8640-contract-methods",
                    **reference,
                    field=field,
                    operator="==",
                    expected=1,
                    observed=value.get(field),
                    passed=False,
                    missing_reason="missing input or field" if field not in value else None,
                )
            )
    if value.get("methods_ready_score") == value.get("learning_operands_ready_score") == 1:
        try:
            sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
            if not read_bound_sidecar(path, Path(sidecar["publication"]["sidecar_path"]))["report"][
                "passed"
            ]:
                raise ValueError("publication")
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                dict(
                    check="upstream_publication",
                    upstream="exp8640-contract-methods",
                    **reference,
                    field="publication",
                    operator="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                )
            )
    return dict(reference=reference, failures=failures)


def prepare(raw: Path) -> Json:
    """Reuse authenticated mechanics; actual current numerical work stays explicit."""
    progress("8641_preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    work = base.prepare(raw)
    checked = upstream(ROOT)
    work["precondition_failures"] = list(work["failures"])
    work["failures"] += checked["failures"]
    work.update(
        upstream_reference=checked["reference"],
        run_date="20261018",
        statement_diagnosis=original.diagnosis(),
        child_shard_manifest=[],
        original_report={},
        owned_report={},
        deadline_case_rows=[],
        semantic_mutation_rows=[],
        custody_mutation_rows=[],
        second_combine_invariance={},
        receipts=[],
        node_union={},
        completed_execution=False,
    )
    work["current_source_reference"] += [ref(ROOT / p) for p in [*OWNED, *TESTS]]
    work["historical_failure_reference"] = ref(
        ROOT / "results/experiment_8616_v740_deadline_statement_qualification.json"
    )
    work["preconditions"]["optional_gpu_and_boards_required"] = False
    atomic_json(raw / "work.json", work)
    return work


def qualified(work: Json) -> bool:
    """A score is earned only by cold reconstruction of coverage and semantic controls."""
    from carnot.reporting.learning_custody_validation_8641 import attest

    try:
        attest(work)
        return bool(work["completed_execution"] and all(r["passed"] for r in work["receipts"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build(work: Json, output: Path) -> Json:
    """Keep the established schema while granting qualification no scientific benefit."""
    with patch.object(base, "qualified", qualified):
        value = base.build(work, output)
    if work["failures"] and not work["completed_execution"]:
        value["verdict_class"] = "blocked"
        value["gate_check_summary"] = list(work["failures"])
    failure_reference = work.get("execution_failure_reference")
    if failure_reference:
        failure = read(failure_reference)
        value["verdict_class"] = "disqualified"
        value["gate_check_summary"].append(
            dict(
                check="original_execution_completed",
                upstream=TASK,
                **failure_reference,
                field="execution_qualification",
                operator="==",
                expected=True,
                observed=False,
                passed=False,
                missing_reason=failure["error"],
                actual_exit=failure["last_receipt"].get("exit_code"),
            )
        )
    mapped = work.get("current_mapping") or work["statement_diagnosis"]
    union = work["node_union"]
    if not union.get("outcomes") and work["child_shard_manifest"]:
        outcomes = [
            row
            for manifest in work["child_shard_manifest"]
            if custody.verify({k: v for k, v in manifest.items() if k != "scope"})
            for row in terminal_nodes(Path(manifest["junit"]["path"]))
        ]
        collected = read(work["collection_reference"])["nodes"]
        union = dict(
            collected=collected,
            outcomes=outcomes,
            original_nodes=work["original_nodes"],
            pending_node_ids=[n for n in collected if n not in {r["node_id"] for r in outcomes}],
            status="unfinished_owned_validation",
        )
    value.update(
        experiment_id=8641,
        task_id=TASK,
        milestone="2026.10.742",
        honest_verdict=f"complete_{value['verdict_class']}_v742_learning_execution_custody",
        target_model=value["numerical_inference"],
        independent_count=0,
        rows=work["deadline_case_rows"],
        intended_count=3,
        completed_count=len(work["deadline_case_rows"]),
        failed_count=sum(not r["passed"] for r in work["deadline_case_rows"]),
        sample_size_budget=dict(
            unit="deterministic execution controls", intended=3, independent_questions=0
        ),
        original_node_union=union,
        executed_node_ids=[r["node_id"] for r in union.get("outcomes", [])],
        child_shard_manifest=work["child_shard_manifest"],
        original_aggregate_reference=work["original_report"].get("aggregate_reference"),
        owned_aggregate_reference=work["owned_report"].get("aggregate_reference"),
        coverage_version=mapped["coverage_version"],
        source_bytecode_hashes=mapped["source_bytecode_hashes"],
        normalized_statement_rows=mapped["normalized_statement_rows"],
        deadline_case_rows=work["deadline_case_rows"],
        semantic_mutation_rows=work["semantic_mutation_rows"],
        custody_mutation_rows=work["custody_mutation_rows"],
        second_combine_invariance=work["second_combine_invariance"],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8642, 8653)
        ],
    )
    value["branch_gate_values"] = dict(
        learning_execution_ready_score=value["learning_execution_ready_score"]
    )
    value["execution_failure_reference"] = failure_reference
    value["source_artifact_hashes"] = [
        *work["source_refs"],
        work["upstream_reference"],
        work["historical_failure_reference"],
    ]
    value["raw_shard_hashes"] += [
        r["reference"] for m in work["child_shard_manifest"] for r in m["data"]
    ]
    value["historical_quarantines"].update(
        exp8616="preserved original lost-shard failure; overwritten current measurements are not original evidence",
        exp8628="retired gate record; never a prerequisite",
    )
    value["retirement_recommendation"] = (
        None
        if qualified(work)
        else "Retire this exact custody qualification if owned execution repeats the lost-shard failure; no scientific null follows."
    )
    value["field_principles"].update(
        {
            k: "Authenticate original source, per-child bytes and actual semantics; execution qualification supplies no learning benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = base.digest(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from original authority and sealed children; repaired envelopes grant nothing."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        expected_sources = []
        for identity, pin in base.PINS.items():
            primary = dict(
                path=str(next((ROOT / "results").glob(f"experiment_{identity}_*.json"))), sha256=pin
            )
            expected_sources.extend([primary, read(primary)["primitive_reference"]])
        checkpoint = protocol()["head_checkpoint_reference"]
        if (
            work["source_refs"] != expected_sources
            or work["checkpoint_reference"] != checkpoint
            or ref(Path(checkpoint["path"])) != checkpoint
        ):
            return False
        prior = read(work["source_refs"][-1])
        if (
            work["original_authority"] != prior["authority"]
            or work["original_nodes"] != prior["node_union"]["original_node_ids"][:-2]
        ):
            return False
        for reference in [
            *work["source_refs"],
            *work["current_source_reference"],
            work["mechanics_fixture"],
            work["upstream_reference"],
            work["historical_failure_reference"],
        ]:
            if ref(Path(reference["path"])) != reference:
                return False
        fixture = read(work["mechanics_fixture"])
        expected = dict(
            prior,
            raw=fixture["raw"],
            owned_ready=False,
            receipts=[],
            coverage_complete=False,
            union_passed=False,
            code_refs=[ref(Path(r["path"])) for r in prior["code_refs"]],
        )
        if fixture != expected or work["statement_diagnosis"] != original.diagnosis():
            return False
        for name in (
            "stream_rows.json",
            "released_feedback.json",
            "retention_rows.json",
            "coefficient_transitions.json",
        ):
            if (Path(fixture["raw"]) / name).read_bytes() != (
                Path(prior["raw"]) / name
            ).read_bytes():
                return False
        if work["completed_execution"] and not qualified(work):
            return False
        for manifest in work["child_shard_manifest"]:
            if not custody.verify({k: v for k, v in manifest.items() if k != "scope"}):
                return False
        if work.get("execution_failure_reference"):
            failure = read(work["execution_failure_reference"])
            receipt = failure["last_receipt"]
            if (
                not failure["error"]
                or receipt not in work["receipts"]
                or any(
                    ref(Path(receipt[k + "_path"]))["sha256"] != receipt[k + "_sha256"]
                    for k in ("stdout", "stderr")
                )
            ):
                return False
        if work["failures"] != work["precondition_failures"] + upstream(ROOT)["failures"]:
            return False
        return bool(value == build(work, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Preserve actual scanner severity and cold replay exits beside the candidate."""
    with patch.object(base, "CLI", CLI):
        return dict(base.terminal(path, raw))


def run(output: Path, run_date: str, *, work_path: Path | None = None) -> int:
    """Measure once, then publish checked bytes through the existing atomic writer."""
    from carnot.reporting.learning_custody_validation_8641 import execute
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = json.loads(work_path.read_bytes()) if work_path else prepare(raw)
    if not work_path and not work["failures"]:
        try:
            execute(work)
        except (OSError, ValueError, TimeoutError) as error:
            raw = Path(work["raw"])
            atomic_json(
                raw / "execution_failure.json",
                dict(
                    error=type(error).__name__ + ":" + str(error), last_receipt=work["receipts"][-1]
                ),
            )
            work["execution_failure_reference"] = ref(raw / "execution_failure.json")
            work["completed_execution"] = False
    raw = Path(work["raw"])
    work["run_date"] = run_date
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [dict(phase="execution_and_validation", duration_s=work["duration_s"])]
    atomic_json(raw / "work.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, build(work, output))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
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
    progress("8641_normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Use real wall supervision; calendar validation cannot change learner clocks."""
    print("[exp8641] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
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
