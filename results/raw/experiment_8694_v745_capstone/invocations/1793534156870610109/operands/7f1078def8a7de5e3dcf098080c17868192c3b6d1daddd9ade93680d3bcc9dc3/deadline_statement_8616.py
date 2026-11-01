"""REQ-REPORT-8616: source-bound accounting preserves original deadline semantics."""

from __future__ import annotations

import hashlib
import json
import marshal
from pathlib import Path
import time
from types import CodeType
from typing import Any
from unittest.mock import patch

import coverage
from coverage.parser import PythonParser

from carnot.reporting import deadline_edge_8603 as edge
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
base = edge.old
ROOT, stream, ref, read = edge.ROOT, edge.stream, base.ref, edge.stream.old.read
RUNNER = "python/carnot/reporting/causal_brier_runner_8582.py"
NAME = "experiment_8616_v740_deadline_statement_qualification"
TASK = "exp8616-deadline-statement-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_deadline_statement_8616.py"]
OWNED = [
    "python/carnot/reporting/deadline_statement_8616.py",
    "python/carnot/reporting/deadline_statement_validation_8616.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PIN_SOURCE = "sha256:efc6c18ee2afa3edda16714b944e4b4dcec999265648ad0a1f98b7231f6881cf"
PIN_8603 = "sha256:e23c4f3823f971b10ec82ce55efb388bb644d99fee40485c5dc0bdd3581d9b10"
ANCHORS = [30, 74, 91, 92, 94]


def map_statements(source: Json, report: Json, *, anchors: list[int] = ANCHORS) -> Json:
    """Ask Coverage.py which statement owns each anchor; never invent execution."""
    if source["sha256"] != PIN_SOURCE or ref(Path(source["path"])) != source:
        raise ValueError("source_bytes_changed")
    data = Path(source["path"]).read_bytes()
    parser = PythonParser(text=data.decode(), filename=source["path"])
    parser.parse_source()
    measured = report["files"][RUNNER]
    executed = set(measured["executed_lines"])
    rows = []
    for anchor in anchors:
        normalized = parser.first_line(anchor)
        if normalized not in parser.statements:
            raise ValueError("anchor_not_executable_statement")
        if normalized not in executed:
            raise ValueError("unexecuted_real_statement")
        rows.append(dict(raw_anchor=anchor, normalized_statement=normalized, executed=True))
    code = compile(data, RUNNER, "exec")
    codes = [code, *(c for c in code.co_consts if isinstance(c, CodeType))]
    return dict(
        coverage_version=coverage.__version__,
        source_reference=source,
        executable_statements=sorted(parser.statements),
        raw_statement_anchor_rows=[dict(raw_anchor=a, raw_executed=a in executed) for a in anchors],
        normalized_statement_rows=rows,
        source_bytecode_hashes=dict(
            source_sha256=source["sha256"],
            bytecode_sha256="sha256:" + hashlib.sha256(marshal.dumps(code)).hexdigest(),
            line_tables={c.co_name: list(map(list, c.co_lines())) for c in codes},
        ),
    )


def semantic_cases(rows: list[Json], *, complete: bool = True) -> bool:
    """A covered append cannot prove that the separate diagnostic clock was read."""
    if complete and [r["predicate_elapsed_s"] for r in rows] != [3599.0, 3600.0, 3601.0]:
        return False
    for row in rows:
        elapsed = row["predicate_elapsed_s"]
        expired = elapsed >= 3600
        value = read(row["output_reference"])
        gates = [r for r in value["gate_check_summary"] if r["field"] == "measurement_deadline"]
        if not (
            row["passed"]
            and row["normal_cleanup"]
            and row["kernel_clock_unchanged"]
            and row["positive_class_absent"]
            and row["exit_code"] == 0
            and row["clock_reads"] == (3 if expired else 2)
            and row["observed_elapsed_s"] == (elapsed + 0.25 if expired else None)
            and row["deadline_failed"] == expired
            and bool(gates) == expired
            and (not expired or gates[0]["observed"] == elapsed + 0.25)
            and json.loads(
                (Path(row["output_reference"]["path"]).parent / "case.json").read_bytes()
            )
            == row
        ):
            return False
    return bool(rows)


def diagnosis() -> Json:
    """Authenticate sealed raw and projected evidence before running costly old tests."""
    path = ROOT / "results/experiment_8603_v739_deadline_edge_qualification.json"
    prior = read(dict(path=str(path), sha256=PIN_8603))
    work = read(prior["primitive_reference"])
    raw = Path(work["raw"])
    full, owned = ref(raw / "coverage_full.json"), work["coverage_reference"]
    # These hashes are immutable original operands, not newly repaired summaries.
    if (
        full["sha256"] != "sha256:8afd42e372cab7ac82694427cdb362f7cd511a55152644206d33267987ebe65a"
        or owned["sha256"]
        != "sha256:3f42796cc7ce93146894607e6572cb8e58e3be43305c8875640cb70d80452628"
    ):
        raise ValueError("historical_coverage_changed")
    report = read(full)
    runner = report["files"][RUNNER]
    if (
        runner["summary"]["num_statements"] != 83
        or runner["missing_lines"]
        or runner["excluded_lines"]
        or read(owned)["files"][RUNNER]["missing_lines"] != [94]
        or not semantic_cases(work["deadline_case_rows"])
    ):
        raise ValueError("diagnosis_contradicted")
    return dict(
        map_statements(ref(ROOT / RUNNER), report),
        historical_raw_statement_count=83,
        historical_projected_missing=[94],
        raw_coverage_reference=full,
        projected_coverage_reference=owned,
        historical_primary_reference=ref(path),
        historical_work_reference=prior["primitive_reference"],
    )


def prepare(raw: Path) -> Json:
    """Reuse sealed mechanics while retaining independent resource and authority checks."""
    started = time.monotonic_ns()
    progress("source_bound_prequalification")
    proof = diagnosis()
    work = edge.prepare(raw)
    work["started_monotonic_ns"] = started
    work.update(
        statement_diagnosis=proof,
        historical_work_reference=proof["historical_work_reference"],
        current_mapping={},
        shards=[],
        semantic_mutation_rows=[],
        revision=ref(ROOT / OWNED[0]),
    )
    work["current_source_reference"] += [ref(ROOT / p) for p in [*OWNED, *TESTS]]
    atomic_json(raw / "work.json", work)
    return work


_BASE_BUILD, _BASE_QUALIFIED, _EDGE_RUN = base.build, base.qualified, edge.run


def qualified(work: Json) -> bool:
    """Require durable child evidence as well as source mapping and semantic controls."""
    from carnot.reporting.deadline_statement_validation_8616 import attest

    try:
        if not work["current_mapping"]:
            return False
        attest(work)
        with patch.object(base, "OWNED", OWNED):
            return bool(_BASE_QUALIFIED(work) and semantic_cases(work["deadline_case_rows"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def build(work: Json, output: Path) -> Json:
    """Use the established artifact schema while separating qualification from benefit."""
    with patch.object(base, "qualified", qualified):
        value = _BASE_BUILD(work, output)
    proof = work["statement_diagnosis"]
    mapped = work["current_mapping"] or proof
    value.update(
        experiment_id=8616,
        task_id=TASK,
        milestone="2026.10.740",
        honest_verdict=f"complete_{value['verdict_class']}_v740_deadline_statement_qualification",
        target_model=value["numerical_inference"],
        rows=work["deadline_case_rows"],
        intended_count=3,
        completed_count=len(work["deadline_case_rows"]),
        failed_count=sum(not r["passed"] for r in work["deadline_case_rows"]),
        independent_count=0,
        sample_size_budget=dict(
            unit="deterministic deadline control", intended=3, independent_questions=0
        ),
        deadline_case_rows=work["deadline_case_rows"],
        original_node_ids=work["original_nodes"],
        executed_node_ids=[r["node_id"] for r in work["node_union"].get("outcomes", [])],
        exact_union_receipt=work["node_union"],
        measured_statement_anchors=ANCHORS,
        kernel_clock_independence=work["clock_independence_rows"],
        current_revision_reference=work["revision"],
        raw_statement_anchor_rows=mapped["raw_statement_anchor_rows"],
        normalized_statement_rows=mapped["normalized_statement_rows"],
        coverage_version=mapped["coverage_version"],
        source_bytecode_hashes=mapped["source_bytecode_hashes"],
        raw_coverage_reference=work.get("full_coverage_reference", proof["raw_coverage_reference"]),
        semantic_mutation_rows=work["semantic_mutation_rows"],
        statement_diagnosis=proof,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8617, 8627)
        ],
    )
    value["raw_shard_hashes"] += work["shards"]
    value["branch_gate_values"] = dict(
        learning_execution_ready_score=value["learning_execution_ready_score"]
    )
    value["historical_quarantines"].update(
        exp8592="preserved disqualification",
        exp8603="preserved disqualification; original projection demanded continuation94",
    )
    value["field_principles"].update(
        {
            key: "Bind source bytes, actual child execution and original semantic assertions; instrumentation qualification grants no learning benefit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"]["learning_execution_ready_score"] = (
        "Numeric0/1; one requires exact original union, real deadline diagnostics, source-bound execution, full owned coverage and accepted current invocation."
    )
    value["reproducibility_checksum"] = base.digest(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from pinned source, raw child shards and original authority after rehashing."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if (
            work["statement_diagnosis"] != diagnosis()
            or work["original_authority"] != read(work["source_refs"][-1])["authority"]
        ):
            return False
        if work["current_mapping"] and not qualified(work):
            return False
        if work["deadline_case_rows"] and (
            not semantic_cases(work["deadline_case_rows"])
            or not all(
                stream.replay(Path(r["output_reference"]["path"]))
                for r in work["deadline_case_rows"]
            )
        ):
            return False
        with patch.object(base, "build", build):
            return bool(base.replay(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Keep unchanged scanner and row-consumer findings beside a fresh bounded replay."""
    with patch.object(base, "CLI", CLI):
        return base.terminal(path, raw)


def run(output: Path, run_date: str, *, work_path: Path | None = None) -> int:
    """Run only this qualification controller, then validate exact bytes for publication."""
    from carnot.reporting.deadline_statement_validation_8616 import execute

    if work_path is None:
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        work = prepare(raw)
        execute(work)
        work_path = raw / "work.json"
        atomic_json(work_path, work)
    with (
        patch.object(edge, "build", build),
        patch.object(edge, "terminal", terminal),
        patch.object(edge, "NAME", NAME),
        patch.object(edge, "TASK", TASK),
    ):
        return int(_EDGE_RUN(output, run_date, work_path=work_path))


def main(argv: list[str] | None = None) -> int:
    """Share the calendar and real wall deadline controls without changing historical code."""
    print("[exp8616] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    with (
        patch.object(edge, "run", run),
        patch.object(edge, "replay", replay),
        patch.object(edge, "NAME", NAME),
    ):
        return int(edge.main(argv))
