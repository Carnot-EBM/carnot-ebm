"""REQ-REPORT-8541: reconstruct qualification from primitive terminal evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import historical_learning_8541 as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def complete_coverage(ref: Json | None) -> bool:
    """An unrelated small trace cannot stand in for all statements in the owned implementation."""
    import coverage

    if ref is None:
        return False
    value = e.read(ref)
    files = {str((e.ROOT / name).resolve()): row for name, row in value.get("files", {}).items()}
    expected = {str((e.ROOT / name).resolve()) for name in e.OWNED}
    if set(files) != expected:
        return False
    analyzer = coverage.Coverage(config_file=False, data_file=None)
    return all(
        not analyzer.analysis2(name)[2]
        and set(row["executed_lines"]) == set(analyzer.analysis2(name)[1])
        and not row["missing_lines"]
        and row["summary"]["num_statements"] == len(analyzer.analysis2(name)[1])
        for name, row in files.items()
    )


def check_contract(work: Json, raw: Path) -> bool:
    """Every required command and child clock must match this invocation's plan."""
    from carnot.reporting.historical_learning_runner_8541 import required_commands

    expected = required_commands(work, raw)
    checks = work["checks"]
    if {r["name"] for r in checks} != set(expected) or len(checks) != len(expected):
        return False
    return all(
        r["argv"] == expected[r["name"]][0]
        and r["actual_exit"] == r["exit_code"] == r["expected_exit"] == expected[r["name"]][1]
        and r["passed"]
        and not r["timed_out"]
        and 0 < r["deadline_s"] <= 480
        and work["start_ns"]
        <= r["started_monotonic_ns"]
        <= r["ended_monotonic_ns"]
        <= work["end_ns"]
        for r in checks
    )


def checksum(value: Json) -> str:
    """The checksum excludes itself so identical primitive reductions have one seal."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def gate(path: str, field: str, expected: Any, observed: Any, digest: str | None = None) -> Json:
    """Missing resources remain distinct from a measured zero or failed child."""
    return dict(
        check=field,
        upstream=e.TASK,
        path=path,
        sha256=digest,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def build(work: Json, raw: Path, output: Path) -> Json:
    """A passed outer process cannot replace missing original terminal assertions."""
    union = e.union(work["nodes"], work["receipts"])
    coverage = e.read(work["coverage"]) if work["coverage"] else {}
    covered = coverage.get("totals", {})
    coverage_ok = complete_coverage(work["coverage"])
    authority_ok = (
        bool(work["nodes"])
        and work["nodes"] == e.original_union()[1]
        and work["authorities"] == e.authorities(work["nodes"])
    )
    clocks_ok = all(
        work["start_ns"] <= r["started_monotonic_ns"] <= r["ended_monotonic_ns"] <= work["end_ns"]
        for r in work["receipts"]
    )
    required = (
        check_contract(work, raw) and authority_ok and clocks_ok and coverage_ok and union["passed"]
    )
    owned_failure = any(not r["passed"] for r in [*work["receipts"], *work["checks"]])
    kind = (
        "disqualified"
        if owned_failure or (not required and not work["failures"])
        else "blocked"
        if work["failures"]
        else "circular_positive"
    )
    by_node = {r["node_id"]: r["outcome"] for r in union["terminal_node_outcomes"]}
    rows = [
        dict(
            unit_id=n,
            family=n.split("::")[0],
            arm="original_authority",
            seed=None,
            status="completed"
            if by_node.get(n) == "passed"
            else "failed"
            if n in by_node
            else "censored",
            outcome=by_node.get(n),
            censored=n not in by_node,
            excluded=False,
            missing_reason=None if n in by_node else "no_passing_terminal_outcome",
            absolute_metric=int(by_node[n] == "passed") if n in by_node else None,
        )
        for n in work["nodes"]
    ]
    failures = list(work["failures"])
    failures += [
        gate(str(raw), field, True, observed)
        for field, observed in [
            ("required_command_contract", check_contract(work, raw)),
            ("original_authority", authority_ok),
            ("invocation_child_clocks", clocks_ok),
        ]
        if not observed
    ]
    for r in [*work["receipts"], *work["checks"]]:
        if not r["passed"]:
            failures.append(
                gate(r["stdout_path"], r["name"], 0, r["exit_code"], r["stdout_sha256"])
            )
    if not union["passed"]:
        failures.append(
            gate(str(raw / "measurement.json"), "exact_original_terminal_union", True, union)
        )
    if not coverage_ok:
        failures.append(
            gate(str(raw), "new_statement_coverage", 100, covered.get("percent_covered"))
        )
    diagnostics = []
    for r in work["receipts"]:
        if not r["passed"]:
            lines = Path(r["stdout_path"]).read_text().splitlines()
            pending = [n for n in r["collected_node_ids"] if n not in by_node]
            diagnostics.append(
                dict(
                    child=r["name"],
                    timed_out=r["timed_out"],
                    pending_node_ids=pending,
                    actual_timed_out_node=pending[0] if r["timed_out"] and pending else None,
                    log_tail=lines[-20:],
                )
            )
    value = dict(
        experiment_id=8541,
        task_id=e.TASK,
        milestone="2026.10.734",
        run_date="20261015",
        honest_verdict="complete_" + kind + "_v734_historical_learning_qualification",
        verdict_class=kind,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        current_llm_invocations=0,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work.get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            original_nodes=len(rows), independent_scientific_sources=0, repeats_add_evidence=False
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed historical tests and sealed fixtures only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=required,
        flagged_adversarial=not required or bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            required_command_contract=check_contract(work, raw),
            original_authority=authority_ok,
            exact_original_union=union["passed"],
            new_statement_coverage=coverage_ok,
            scientific_benefit=False,
        ),
        validation_receipts=[*work["receipts"], *work["checks"]],
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=work["preconditions"],
        duration_s=(work["end_ns"] - work["start_ns"]) / 1e9,
        phase_spans=work.get("phase_spans", []),
        random_seed=7348541,
        source_artifact_hashes=work["sources"],
        code_config_hashes=[e.reference(e.ROOT / p) for p in [*e.OWNED, e.TEST]],
        raw_shard_hashes=[
            e.reference(raw / "measurement.json"),
            *[e.reference(p) for p in sorted(raw.glob(".coverage*"))],
        ],
        cited_upstream_artifacts=[
            dict(
                e.primary(8530),
                imported_fields=["collected_node_ids", "validation_receipts"],
                prior_quarantine_preserved=True,
            )
        ],
        historical_learning_validation_ready_score=int(
            required and not work["failures"] and not work.get("adversarial_findings")
        ),
        original_authority_by_suite=work["authorities"],
        collected_node_ids=work["nodes"],
        shard_node_ids={r["name"]: r["collected_node_ids"] for r in work["receipts"]},
        terminal_node_outcomes=union["terminal_node_outcomes"],
        exact_union_receipt=union,
        failed_family_diagnostics=diagnostics,
        historical_failure_reproduction=work["historical_failure_reproduction"],
        preserved_quarantines=[8530, 8533, 8537],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_input") for n in range(8542, 8552)
        ],
        owned_coverage_reference=work["coverage"],
        primitive_reference=e.reference(raw / "measurement.json"),
        publication_output=str(output),
        methodology_note="Exact original terminal tests qualify exposed mechanics only. Historical failed logs and quarantines persist. No learning, native speed or capability benefit is measured.",
    )
    value["field_principles"] = {
        k: "Bind invocation, immutable source bytes and actual terminal outcomes; qualification grants no independent scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold reconstruction rejects changed operands even after summary hashes are repaired."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if work["nodes"] and work["nodes"] != e.original_union()[1]:
            return False
        for ref in [*value["source_artifact_hashes"], *value["code_config_hashes"]]:
            if e.reference(Path(ref["path"])) != ref:
                return False
        for r in [*work["receipts"], *work["checks"]]:
            for stream in ("stdout", "stderr"):
                if e.reference(Path(r[stream + "_path"]))["sha256"] != r[stream + "_sha256"]:
                    return False
            stored = json.loads(Path(r["stdout_path"]).with_suffix(".receipt.json").read_bytes())
            if any(stored[k] != r[k] for k in stored):
                return False
        if work["authorities"] != e.authorities(work["nodes"]):
            return False
        return bool(
            value
            == build(
                work,
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
