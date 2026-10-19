"""REQ-REPORT-8641: execution qualification never rewrites failed history or claims benefit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import coverage
from coverage.parser import PythonParser
from carnot.reporting import learning_custody_8641 as c
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = c.ROOT
NAME = "experiment_8641_v742_learning_execution_custody"
TASK = "exp8641-learning-execution-custody"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/learning_custody_8641.py",
    "python/carnot/reporting/learning_execution_report_8641.py",
    "python/carnot/reporting/learning_execution_8641.py",
    CLI,
]
TESTS = ["tests/python/test_learning_execution_custody_8641.py"]
UPSTREAM = ROOT / "results/experiment_8640_v742_contract_methods.json"
RUNNER = "python/carnot/reporting/causal_brier_runner_8582.py"
TEST_PINS = {
    "test_learning_clock_8592.py": "471c965729bb77b8ab83c4213d9b791969c2dc142dbd4dda914023995b6a504b",
    "test_causal_brier_stream_8582.py": "33122f4b4c12c5ae5c08768a630926f2f2d585a034d528e2cb8d65868ea350b5",
    "test_objective_checkpoint_8461.py": "dd6fb400f2bd1d4a677cb78f07cbc19223f76c7b52722f984540674029ba6bd2",
    "test_deadline_edge_8603.py": "58041e124a9a1cd09aaeee2d1a978bac841abe29dac14cb3e447306c7834d933",
    "test_deadline_statement_8616.py": "92d09f06fe05a95b22ace5c03ece3a4f5f690335259dbacde63513f8c8be147a",
}
E2E_PINS = {
    "test_experiment_7891_v685_authority_lifecycle.py": "2d67e9cb874659e87788a9202b9d513c6b13e7edb2387e624a6c6dfea35dd157",
    "test_restricted_decision_audit_8210.py": "b390da1c764ab1c83b14d2343956fc35cc02c10ae164f0a10745661f5a38d908",
}


def gate(path: Path, field: str, expected: Any, observed: Any, *, upstream: str = TASK) -> Json:
    """Missing bytes and observed zero have different evidence and different reopening actions."""
    return dict(
        check=field,
        field=field,
        expected=expected,
        observed=observed,
        operator="==",
        upstream=upstream,
        path=str(path),
        sha256=c.ref(path)["sha256"] if path.is_file() else None,
        passed=False,
        missing_reason="missing input" if observed is None else None,
    )


def inputs() -> tuple[Json, list[Json]]:
    """Allow original execution to fail while independently authenticated operands remain usable."""
    value = json.loads(UPSTREAM.read_bytes()) if UPSTREAM.is_file() else {}
    return value, [
        gate(UPSTREAM, k, 1, value.get(k), upstream="exp8640-contract-methods")
        for k in ("methods_ready_score", "learning_operands_ready_score")
        if value.get(k) != 1
    ]


def statement_report(lines: dict[str, list[int]], sources: list[Json]) -> Json:
    """Coverage owns executable statements; continuation lines cannot create an extra statement."""
    files = {}
    for source in sources:
        path = source["path"]
        parser = PythonParser(filename=path)
        parser.parse_source()
        executed = set(parser.translate_lines(lines.get(path, [])))
        files[path] = dict(
            statements=sorted(parser.statements),
            executed=sorted(executed),
            missing=sorted(parser.statements - executed),
            excluded=sorted(parser.excluded),
        )
    return dict(
        files=files,
        complete=bool(files)
        and all(not f["missing"] and not f["excluded"] for f in files.values()),
    )


def digest(value: Json) -> str:
    """Keep the checksum outside its own input so a fresh process can reproduce the report."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def build(work: Json, output: Path) -> Json:
    """Report failed checks as qualification failures while preserving all scientific limits."""
    original = work.get("original_coverage", {})
    owned = work.get("owned_coverage", {})
    required = {"ruff_check", "ruff_format", "mypy", "spec", "owned_tests"}
    receipts = work["receipts"]
    checks = dict(
        inputs=not work["input_failures"],
        child_custody=bool(work["children"]),
        original_union=work.get("union_passed", False),
        original_statements=original.get("complete", False),
        owned_statements=owned.get("complete", False),
        semantic_deadlines=work.get("semantics_passed", False),
        custody_controls=work.get("custody_controls_passed", False),
        validation=required <= {r["name"] for r in receipts if r["passed"]}
        and all(r["passed"] for r in receipts),
    )
    ready = all(checks.values())
    verdict = (
        "circular_positive" if ready else "blocked" if work["input_failures"] else "disqualified"
    )
    failures = list(work["input_failures"]) + [
        gate(Path(work["raw"]) / "work.json", k, True, v)
        for k, v in checks.items()
        if not v and k != "inputs"
    ]
    rows = work.get("outcomes", [])
    upstream = work.get("upstream", {})
    value = dict(
        experiment_id=8641,
        task_id=TASK,
        milestone="2026.10.742",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_v742_learning_execution_custody",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_cpu_small_energy_head_no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(current_llm_calls=0, model_loads=0, generations=0),
        historical_model_provenance=upstream.get("historical_model_provenance", []),
        target_model=work.get("numerical_inference"),
        small_head_numerical_inference=work.get("numerical_inference"),
        rows=rows,
        intended_count=len(work.get("nodes", [])),
        completed_count=len(rows),
        failed_count=sum(r["outcome"] != "passed" for r in rows),
        censored_count=len(work.get("pending_nodes", [])),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(unit="original assertion node", independent_questions=0),
        verifier_is_oracle=True,
        exposure_scope="exposed historical mechanics and deterministic controls; no benefit measurement",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Actual source-bound execution is required; historical failure cannot qualify it.",
            )
            for k, v in checks.items()
        },
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7428641,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[f["reference"] for m in work["children"] for f in m["files"]],
        cited_upstream_artifacts=work["input_references"],
        learning_execution_ready_score=int(ready),
        methods_ready_score=int(upstream.get("methods_ready_score", 0)),
        learning_operands_ready_score=int(upstream.get("learning_operands_ready_score", 0)),
        original_node_union=work.get("node_union", {}),
        executed_node_ids=[r["node_id"] for r in rows],
        pending_node_ids=work.get("pending_nodes", []),
        child_shard_manifest=work["children"],
        original_aggregate_reference=work.get("original_aggregate"),
        owned_aggregate_reference=work.get("owned_aggregate"),
        coverage_version=coverage.__version__,
        source_bytecode_hashes=work.get("mapping", {}).get("source_bytecode_hashes", {}),
        normalized_statement_rows=work.get("mapping", {}).get("normalized_statement_rows", []),
        deadline_case_rows=work.get("deadline_case_rows", []),
        semantic_mutation_rows=work.get("semantic_mutation_rows", []),
        custody_mutation_rows=work.get("custody_mutation_rows", []),
        second_combine_invariance=work.get("second_combine_invariance", {}),
        original_statement_coverage=original,
        owned_statement_coverage=owned,
        primitive_reference=c.ref(Path(work["raw"]) / "work.json"),
        invocation_seal_reference=work.get("invocation_seal_reference"),
        historical_quarantines=dict(
            exp8616="preserved complete_disqualified; original shards lost",
            exp8628="preserved blocked input gate; retired qualification",
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8642, 8653)
        ],
        methodology_note="New actual execution with immediate immutable per-child coverage; qualification grants no scientific benefit.",
        failure_reproduction=work.get("failure_reproduction"),
        publication_output=str(output),
    )
    value["branch_gate_values"] = {
        k: value[k]
        for k in (
            "methods_ready_score",
            "learning_operands_ready_score",
            "learning_execution_ready_score",
        )
    }
    value["field_principles"] = {
        k: "Bind this invocation's authenticated operands, real child bytes and original assertions; qualification is separate from scientific benefit."
        for k in value
    }
    value["reproducibility_checksum"] = digest(value)
    return value
