"""REQ-VERIFY-8257: publish only checked current invocation bytes.

The process supervisor keeps exact logs and terminates only its own child group.
Private fixture outcomes qualify mechanics without becoming live measurements.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import arc_authoritative_execution_8215 as qualified
from carnot.reporting import arc_outcome_frontier_8257 as reader
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import coverage_complete, dependency_hashes
from carnot.reporting.v709_execution import child, progress
from scripts.experiment_template import normalize_artifact_for_template_write

ROOT = reader.ROOT
CLI = "scripts/experiments/experiment_8257_v713_arc_outcome_frontier.py"
OUTPUT = ROOT / "results/experiment_8257_v713_arc_outcome_frontier.json"
TEST = "tests/python/test_arc_outcome_frontier_8257.py"
OWNED = [
    "python/carnot/reporting/arc_outcome_frontier_8257.py",
    "python/carnot/reporting/arc_outcome_execution_8257.py",
    CLI,
]
Json = dict[str, Any]


def commands(private: Path) -> list[Json]:
    """Reuse the qualified command protocol but measure only this task's statements."""
    plan = qualified.commands(private)
    replacements = dict(zip(qualified.OWNED, OWNED, strict=True)) | {qualified.TEST: TEST}
    for spec in plan:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] == "affected_consumers":
            spec["argv"].insert(-1, "tests/python/test_arc_outcome_delta_8229.py")
    config = private / "coverage.ini"
    config.write_text(
        config.read_text()
        .replace("arc_authoritative_frontier_8215.py", "arc_outcome_frontier_8257.py")
        .replace("arc_authoritative_execution_8215.py", "arc_outcome_execution_8257.py")
        .replace(
            "experiment_8215_v709_arc_authoritative_frontier.py",
            "experiment_8257_v713_arc_outcome_frontier.py",
        )
    )
    for name, test in [
        ("e2e_015", "tests/python/test_source_boundary_7852.py"),
        ("e2e_019", "tests/python/test_experiment_7942_v689_sentence_labels.py"),
    ]:
        plan.insert(
            -1,
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    test,
                    "--basetemp=" + str(private / name),
                ],
                deadline_s=60,
                expected_exit=0,
                classification="required",
            ),
        )
    return plan


def preconditions(private: Path) -> Json:
    """Check named inputs and private writable storage without requiring model resources."""
    available = qualified.preconditions(private)
    for label in [
        *OWNED,
        TEST,
        *[".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")],
        "results/experiment_8243_v712_arc_supervisor_frontier.json",
        "ops/north-star.md",
        "python/carnot/reporting/arc_authoritative_frontier_8215.py",
        "python/carnot/reporting/arc_authoritative_execution_8215.py",
        "openspec/capabilities/arc-world-model-trust-energy/spec.md",
        "python/carnot/agentic/arc_solver_kit.py",
        "results/experiment_8229_v711_arc_outcome_delta.json",
        "python/carnot/reporting/arc_outcome_delta_8229.py",
        "python/carnot/reporting/arc_outcome_execution_8229.py",
    ]:
        path = ROOT / label
        digest = sha256_file(path) if path.is_file() else None
        available["checks"].append(
            dict(
                reader.authority.operand(path, "is_file", True, path.is_file(), digest),
                required=True,
                passed=path.is_file(),
            )
        )
        if digest:
            available["hashes"][str(path)] = digest
    # Missing optional old inputs remain separate from required current evidence.
    available["failures"] = [r for r in available["checks"] if not r["passed"]]
    return available


def replay(value: Json) -> list[str]:
    """Recompute from primitive authority and reject altered receipts or execution claims."""
    errors = []
    for group in ("source_artifact_hashes", "raw_shard_hashes", "code_config_hashes"):
        for label, digest in value.get(group, {}).items():
            path = Path(label)
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file() or sha256_file(path) != digest:
                errors.append("sha256:" + label)
    primitive = Path(value["primitive_path"])
    if not primitive.is_file():
        return errors + ["missing_primitive"]
    stored = json.loads(primitive.read_text())
    reduced = reader.inspect(
        Path(value["locator_path"]), Path(value["frontier_path"]), Path(value["adapter_path"])
    )
    if stored != reduced or value["reproducibility_checksum"] != canonical_hash(reduced):
        errors.append("primitive_reduction_drift")
    expected = (
        reader.summarize([], stored["prior_frontier"]) if value["fixture_claim_scope"] else stored
    )
    for key in reader.summarize([], {}):
        if value.get(key) != expected.get(key):
            errors.append("reduction_drift:" + key)
    for row in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            path = Path(row[stream + "_path"])
            if not path.is_file() or sha256_file(path) != row[stream + "_sha256"]:
                errors.append("validation_stream:" + row["name"])
        original = json.loads(
            Path(row["stdout_path"]).with_name(row["name"] + ".receipt.json").read_text()
        )
        if any(row.get(k) != v for k, v in original.items()):
            errors.append("validation_receipt:" + row["name"])
    owned = [r for r in value["validation_receipts"] if r["classification"] == "required"]
    ready = value["acceptance_gates"]["owned_coverage_100"] and all(r["passed"] for r in owned)
    blocked = stored["failures"] or [r for r in value["preconditions_checked"] if not r["passed"]]
    state = "disqualified" if not ready else "blocked" if blocked else "null"
    assertions = dict(
        experiment_id=8257,
        task_id=reader.TASK_ID,
        run_date="20261007",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        solve_claims=[],
        credited_new_levels=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        verdict_class=state,
        arc_delta_ready_score=int(ready and not blocked),
        arc_evidence_ready_score=int(
            ready and not blocked and bool(expected["proposed_generalization_change"])
        ),
        frontier_sha256=stored["frontier_sha256"],
        receipt_frontier=stored["receipt_frontier"],
        proposed_arm_change=expected["proposed_arm_change"],
        frontier_hashes=stored["frontier_hashes"],
        solve_provenance="live_agent_self_discovery" if expected["new_outcome_count"] else None,
    )
    for key, expected_value in assertions.items():
        if value.get(key) != expected_value:
            errors.append("execution_claim:" + key)
    return errors


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged public checks and a fresh replay process on identical bytes."""
    reports = [
        child(name, argv, raw / "terminal_logs", deadline=60)
        for name, argv in [
            (
                "terminal_adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "terminal_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
            (
                "cold_replay",
                [str(ROOT / ".venv/bin/python"), str(ROOT / CLI), "--cold-replay", str(candidate)],
            ),
        ]
    ]
    return dict(passed=all(r["passed"] for r in reports), reports=reports)


def execute(
    locator: Path, frontier: Path, output: Path, private: Path, *, fixture: bool = False
) -> int:
    """Read once, retain all evidence, then atomically publish checked terminal bytes."""
    start = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    private.mkdir(parents=True, exist_ok=True)
    raw = output.parent / "raw" / output.stem / "invocations" / str(start)
    raw.mkdir(parents=True, exist_ok=True)
    progress("exp8257_preconditions", 0, 1)
    available = preconditions(private)
    specs = [] if fixture else commands(private)
    code = dependency_hashes(ROOT, paths=[*OWNED, TEST])
    manifest = raw / "validation_command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=specs,
            code_config_hashes=code,
            owned_files=OWNED,
            applicable_e2e=["E2E-015", "E2E-019"],
            fixture=fixture,
            heartbeat_s=30,
        ),
    )
    measured = time.monotonic_ns()
    progress("exp8257_frontier_delta", 0, 1)
    reduced = reader.inspect(locator, frontier, raw / "adapter")
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, reduced)
    reduced_at = time.monotonic_ns()
    source_hashes = dict(reduced["source_artifact_hashes"], **available["hashes"])
    snapshots = {str(primitive): sha256_file(primitive), str(manifest): sha256_file(manifest)}
    for label, digest in source_hashes.items():
        original = Path(label)
        copy = raw / "inputs" / digest[7:] / original.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copy)
        snapshots[str(copy)] = sha256_file(copy)
    receipts = []
    progress("exp8257_validation", 0, len(specs))
    for index, spec in enumerate(specs):
        progress("exp8257_check", index, len(specs) - index)
        row = dict(
            child(
                spec["name"],
                spec["argv"],
                raw / "logs",
                deadline=spec["deadline_s"],
                expected=spec["expected_exit"],
                scope=spec["classification"],
            ),
            classification=spec["classification"],
        )
        receipts.append(row)
        receipt_path = raw / "logs" / (spec["name"] + ".receipt.json")
        snapshots[str(receipt_path)] = sha256_file(receipt_path)
    owned = [r for r in receipts if r["classification"] == "required"]
    covered = fixture or coverage_complete(private / "coverage.json", includes=OWNED)
    coverage = {}
    if (private / "coverage.json").is_file():
        copy = raw / "coverage.json"
        shutil.copyfile(private / "coverage.json", copy)
        snapshots[str(copy)] = sha256_file(copy)
        coverage = {p: v["summary"] for p, v in json.loads(copy.read_text())["files"].items()}
    ready = covered and all(r["passed"] for r in owned)
    gates = [*available["failures"], *reduced["failures"]]
    for row in owned:
        if not row["passed"]:
            gates.append(
                reader.authority.operand(
                    Path(row["stdout_path"]),
                    "exit_code",
                    row["expected_exit"],
                    row["exit_code"],
                    row["stdout_sha256"],
                )
            )
    if not covered:
        gates.append(
            reader.authority.operand(
                private / "coverage.json", "owned_statement_coverage", 100, "incomplete", None
            )
        )
    blocked = bool(available["failures"] or reduced["failures"])
    state = "disqualified" if not ready else "blocked" if blocked else "null"
    honest = (
        "complete_disqualified_owned_checks"
        if not ready
        else "complete_blocked_" + gates[0]["artifact_field"].replace(".", "_")
        if blocked
        else "complete_null_descriptive_outcomes"
        if reduced["new_outcome_count"] and not fixture
        else "complete_null_no_new_outcomes"
    )
    live = reader.summarize([], reduced["prior_frontier"]) if fixture else reduced
    finished = time.monotonic_ns()
    value = dict(
        live,
        experiment_id=8257,
        experiment=8257,
        task_id=reader.TASK_ID,
        milestone="2026.10.713",
        run_date="20261007",
        schema="arc-outcome-frontier-v713",
        invocation_argv=[sys.executable, *sys.argv],
        title="Environment-grounded ARC supervisor outcome delta",
        status="complete",
        honest_verdict=honest,
        verdict_class=state,
        gate_check_summary=gates,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        duration_s=(finished - start) / 1e9,
        random_seed=0,
        reproducibility_checksum=canonical_hash(reduced),
        source_artifact_hashes=source_hashes,
        code_config_hashes=code,
        raw_shard_hashes=snapshots,
        phase_spans=[
            dict(
                phase="preconditions_and_freeze",
                started_monotonic_ns=start,
                ended_monotonic_ns=measured,
            ),
            dict(
                phase="authenticated_delta",
                started_monotonic_ns=measured,
                ended_monotonic_ns=reduced_at,
            ),
            dict(phase="validation", started_monotonic_ns=reduced_at, ended_monotonic_ns=finished),
        ],
        preconditions_checked=[*available["checks"], *reduced["authority_locator_rows"]],
        locator_path=str(locator),
        frontier_path=str(frontier),
        adapter_path=str(raw / "adapter"),
        receipt_frontier=reduced["receipt_frontier"],
        frontier_sha256=reduced["frontier_sha256"],
        frontier_hashes=reduced["frontier_hashes"],
        primitive_path=str(primitive),
        historical_excluded_count=reduced["historical_excluded_count"],
        unchanged_authority=reduced["unchanged_authority"],
        cited_upstream_artifacts=[
            dict(
                path=p,
                sha256=h,
                fields_imported=[
                    "experiment_id",
                    "task_id",
                    "schema",
                    "required_checks_passed",
                    "arc_delta_ready_score",
                    "verdict_class",
                    "finished_at",
                    "current_frontier",
                    "qualified_authority_signature",
                    "code_config_hashes",
                    "excluded_count",
                ]
                if p == str(frontier)
                else ["authenticated_file_bytes"],
            )
            for p, h in reduced["source_artifact_hashes"].items()
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_model_invocation_count=0,
        current_game_execution_count=0,
        generator_pin="unsloth/Qwen3.8-27B-GGUF",
        historical_model_provenance="Inherited generator pin only; zero current neural calls.",
        supervision_memory_rationale="AVO-style stalled-trajectory supervision supplies observed feedback; authenticated outcome memory may inform future curated arms only when cross-game support exists.",
        verifier_is_oracle=False,
        exposure_scope="exposed_development_cached_supervisor_receipts",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        solve_claims=[],
        credited_new_levels=0,
        solve_provenance="live_agent_self_discovery" if live["new_outcome_count"] else None,
        defaults_changed=False,
        arc_delta_ready_score=int(ready and not blocked),
        arc_evidence_ready_score=int(
            ready and not blocked and bool(live["proposed_generalization_change"])
        ),
        required_checks_passed=ready,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_coverage_100=covered,
            owned_checks=ready,
            authority_authenticated=not blocked,
            supported_transferable_evidence=bool(live["proposed_generalization_change"]),
        ),
        coverage_statement_counts=coverage,
        validation_receipts=receipts,
        repository_health=[r for r in receipts if r["classification"] == "repository_health"],
        fixture_claim_scope="development_proxy" if fixture else None,
        fixture_result=reduced if fixture else None,
        terminal_validation_sidecar_path=str(raw / "terminal_reports.json"),
        methodology="Authenticate Exp8243 bytes and adapt its frontier to the unchanged qualified reader, then count only new environment resolutions; observational overlap cannot establish causal arm superiority.",
        methodology_reference="https://arxiv.org/abs/2609.00652",
    )
    value["field_principles"] = {
        k: "Bind this invocation to authenticated bytes; execution readiness does not establish scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        receipt_frontier="Actual Exp8243 hash, clock and event IDs bound extraction of changed source bytes.",
        adapter_path="A labelled qualification projection adapts interface identity only; it is not an independent outcome source.",
        proposed_arm_change="Insufficient support remains null; a supported future comparison never changes live priority.",
        arc_delta_ready_score="Owned checks and authenticated authority qualify the delta read, including an empty frontier.",
        historical_excluded_count="Old excluded inventory is cited once and never becomes current observed units.",
        selection_recommendations="Unknown prior selection propensity prevents a transferable priority recommendation.",
        solve_provenance="Null without eligible live receipts; no game-level solve is credited.",
        terminal_validation_sidecar_path="A stable report points to the publisher's hash-bound terminal sidecar.",
    )
    value = normalize_artifact_for_template_write(value)
    candidate = private / output.name
    atomic_json(candidate, value)
    report = terminal(candidate, raw)
    if not report["passed"]:
        raise ValueError("terminal_candidate_rejected")
    published = publish_primary(
        output, value, lambda p: dict(report, passed=sha256_file(p) == sha256_file(candidate))
    )
    atomic_json(
        raw / "terminal_reports.json",
        dict(published, report=report, invocation_duration_s=(time.monotonic_ns() - start) / 1e9),
    )
    progress("exp8257_published", live["new_outcome_count"], 0)
    print(honest, flush=True)
    return int(state != "null")
