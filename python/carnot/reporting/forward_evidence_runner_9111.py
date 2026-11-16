"""REQ-REPORT-9111-V777: qualify evidence readers while preserving the quarantined judge.

All current work is CPU artifact validation. Old model work stays in authenticated
snapshots and diagnostics, and cannot become a claim about this process.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import difflib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any

from coverage import Coverage

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.forward_evidence_9111 import panel_count_derivations, semantic_control_rows
from carnot.reporting.natural_label_eligibility_8975 import ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.v709_execution import child, execute
from scripts import adversarial_verify as av

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9111_v777_forward_evidence", "exp9111-forward-evidence"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_forward_evidence_9111.py",
    "tests/python/test_forward_evidence_runner_9111.py",
]
OWNED = [
    "python/carnot/reporting/forward_evidence_9111.py",
    "python/carnot/reporting/forward_evidence_runner_9111.py",
    CLI,
]
CHANGED = ["scripts/adversarial_verify.py", "python/carnot/reporting/current_work_receipt.py"]
CONTRACT = "results/experiment_9110_v777_contract.json"
PINS = {
    CONTRACT: "sha256:63d99c27406fe67e603977f89bae2f80f24d504a327672dfa9dd913864f084f6",
    "judge": "sha256:1a8053cdf2fb2a5f111e97bfe888cdf4200d638131903d97e210dbdf993ac617",
    "old_validator": "sha256:18dfabaaed325c91bd03cd584c8f7b945a068529f00cbd68e45c0e0c013d49d2",
    "old_receipt": "sha256:428d1696eef0820ce9df519532743aeccb8ec1156eb8c1b52fa0c8f72bf16a10",
    "producer": "sha256:05c1d05bebcb5b5aa0950932caf3afd9f02dff7d32fe337b33d0966407018801",
}
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush phase boundaries while the existing supervisor supplies child heartbeats."""
    print(f"[exp9111] phase={phase}", flush=True)


def bound(reference: Json) -> Path:
    """A saved path is usable only while its exact bytes match the authenticated pin."""
    path = Path(reference["path"])
    if not reference.get("sha256") or ref(path)["sha256"] != reference["sha256"]:
        raise ValueError("reference_drift:" + str(path))
    return path


def authenticate(root: Path, raw: Path) -> Json:
    """Check gates before dependent reads and retain only authenticated immutable inputs."""
    progress("authentication_before")
    checks: list[Json] = []
    refs: list[Json] = []
    evidence: Json = dict(checks=checks, references=refs, snapshots={}, roster_rows=[])

    def retain(reference: Json, name: str) -> Path | None:
        path = Path(reference["path"])
        observed = ref(path)["sha256"] or "missing_file"
        checks.append(gate(path, "sha256", reference["sha256"], observed, name, "exp9110-contract"))
        if not checks[-1]["passed"]:
            return None
        target = raw / "inputs" / str(reference["sha256"]).split(":")[1] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha256_file(target) != reference["sha256"]:
            raise ValueError("input_changed_during_retention")
        refs.append(dict(ref(target), source=reference))
        evidence["snapshots"][name] = ref(target)
        return target

    path = retain(dict(path=str(root / CONTRACT), sha256=PINS[CONTRACT]), "contract_bytes")
    if path is not None:
        contract = json.loads(path.read_bytes())
        observed = contract.get("contract_ready_score", "missing_field")
        checks.append(
            gate(
                path,
                "contract_ready_score",
                1,
                observed,
                "contract_ready_score",
                "exp9110-contract",
            )
        )
        if checks[-1]["passed"]:
            for key in ["judge", "old_validator", "old_receipt", "producer"]:
                candidates = [
                    r for r in contract["authority_snapshots"] if r["sha256"] == PINS[key]
                ]
                if not candidates:
                    checks.append(
                        gate(path, key, PINS[key], "missing_field", key, "exp9110-contract")
                    )
                    continue
                retain(candidates[0], key)
            if all(c["passed"] for c in checks):
                judge = json.loads(bound(evidence["snapshots"]["judge"]).read_bytes())
                for key in [
                    "primitive_reference",
                    "judge_protocol_reference",
                    "judge_prediction_reference",
                    "request_clock_reference",
                ]:
                    retain(judge[key], key)
                authority_ref = next(
                    r
                    for r in contract.get("authority_snapshots", [])
                    if r["path"].endswith(
                        "/authority/results/experiment_9097_v776_human_authority.json"
                    )
                )
                authority_path = retain(authority_ref, "human_authority")
                if authority_path is not None:
                    authority = json.loads(authority_path.read_bytes())
                    for key in ["shard_a_roster_reference", "shard_b_roster_reference"]:
                        roster = retain(authority[key], key)
                        if roster is not None:
                            evidence["roster_rows"].extend(
                                r
                                for r in json.loads(roster.read_bytes())["rows"]
                                if r["role"] in {"CAL", "TEST"}
                            )
    progress("authentication_after")
    return evidence


def historical_report(evidence: Json) -> Json:
    """Execute the pinned historical checker instead of a mutable checkout replacement."""
    original_sys_path = sys.path[:]
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        old = runpy.run_path(str(bound(evidence["snapshots"]["old_validator"])))
    finally:
        sys.path[:] = original_sys_path
    report: Json = old["_verify_artifact_impl"](bound(evidence["snapshots"]["judge"]))
    return report


def measure(evidence: Json, raw: Path) -> Json:
    """Reproduce original critical flags before diagnosing the repaired reader privately."""
    work = dict(
        current_owner_pid=os.getpid(),
        evidence=evidence,
        original_flag_rows=[],
        semantic_control_rows=[],
        preserved_detection_rows=[],
        structural_count_derivations={},
        historical_judge_diagnostic={},
    )
    if not all(c["passed"] for c in evidence["checks"]):
        return work
    progress("historical_reproduction_before")
    snapshots = evidence["snapshots"]
    judge_path = bound(snapshots["judge"])
    judge = json.loads(judge_path.read_bytes())
    report = historical_report(evidence)
    work["original_flag_rows"] = report["flags"]
    atomic_json(raw / "original_flag_report.json", report)
    progress("historical_reproduction_after")
    expected = [
        ("TAUTOLOGY", "critical"),
        ("INFERENCE_PROVENANCE_CONTRADICTION", "critical"),
        ("SUBSTRATE_CLASS_MISMATCH", "critical"),
    ]
    if [(r["kind"], r["severity"]) for r in report["flags"]] != expected:
        raise ValueError("original_flag_reproduction")
    identity = lambda r: (r["unit_id"], r["qid"], r["group_id"], r["role"])
    if sorted(map(identity, judge["rows"])) != sorted(map(identity, evidence["roster_rows"])):
        raise ValueError("historical_roster_drift")
    diagnostic = dict(judge, structural_count_evidence="qid-panel-v1")
    work["structural_count_derivations"] = structural_derivations(diagnostic, evidence)
    candidate = raw / "historical_judge_diagnostic_input.json"
    atomic_json(candidate, diagnostic)
    work["diagnostic_input_reference"] = ref(candidate)
    after = av._verify_artifact_impl(candidate)
    work["historical_judge_diagnostic"] = dict(
        original_quarantine_preserved=True,
        judge_ready_score=0,
        scientific_use_permitted=False,
        original_primary=snapshots["judge"],
        repaired_reader_report=after,
        original_receipt_errors=[
            validate_current_work_receipt(r) for r in judge["current_work_receipts"]
        ],
        claim_scope="Receipt and structural arithmetic audit only; judge accuracy remains unaudited.",
    )
    work["semantic_control_rows"] = semantic_control_rows()
    work["preserved_detection_rows"] = structural_controls(diagnostic)
    atomic_json(raw / "historical_judge_diagnostic.json", work["historical_judge_diagnostic"])
    work["historical_judge_diagnostic_reference"] = ref(raw / "historical_judge_diagnostic.json")
    progress("semantic_measurement_after")
    return work


def structural_derivations(diagnostic: Json, evidence: Json) -> Json:
    """Retain distinct formulas and roster pins for the coincident original integers."""
    derived = panel_count_derivations(diagnostic)
    derived["proof_rows"] = [
        dict(
            field=field,
            value=derived[field],
            formula=formula,
            roster_references=[
                evidence["snapshots"][k]
                for k in ["shard_a_roster_reference", "shard_b_roster_reference"]
            ],
        )
        for field, formula in [
            (
                "effective_independent_groups",
                "count distinct original qid/group pairs across CAL and TEST; siblings count once",
            ),
            (
                "denominator",
                "count retained TEST answer rows, matched to TEST predictions and summed per-qid denominators",
            ),
            (
                "half_denominator",
                "floor(retained TEST answer rows / 2); independent of the CAL plus TEST qid cardinality",
            ),
        ]
    ]
    return derived


def structural_controls(diagnostic: Json) -> list[Json]:
    """Unsupported count assertions and copied outcome metrics must remain critical."""
    cases: list[tuple[str, Json]] = []
    for name in [
        "unbacked_count",
        "noninteger_sample",
        "contradictory_denominator",
        "copied_accuracy_auroc",
        "suffix_only_count",
    ]:
        value = deepcopy(diagnostic)
        if name == "unbacked_count":
            value.pop("rows")
        elif name == "noninteger_sample":
            value["effective_independent_groups"] = 150.0
        elif name == "contradictory_denominator":
            value["qid_rows"][0]["denominator"] += 1
        elif name == "copied_accuracy_auroc":
            value.update(accuracy=0.7654321, auroc=0.7654321)
        else:
            value = dict(effective_independent_groups=150, half_denominator=150)
        cases.append((name, value))
    rows = []
    for name, value in cases:
        flags: list[Any] = []
        av.check_tautology(av._flatten_metrics(value), flags)
        passed = any(f.severity == "critical" for f in flags)
        rows.append(
            dict(
                unit_id=name,
                qid=None,
                group_id=name,
                arm="structural_control",
                condition="rejecting_diagnostic",
                seed=7779111,
                absolute_metric=int(passed),
                numerator=int(passed),
                denominator=1,
                status="completed" if passed else "failed",
                exclusion_reason=None,
                censor_reason=None,
                passed=passed,
                flags=[f.to_dict() for f in flags],
            )
        )
    return rows


def validation_commands(raw: Path) -> list[Json]:
    """Recompute the command contract so a rehashed scope cannot shrink it."""
    config = raw / "coverage.ini"
    py = str(ROOT / ".venv/bin/python")
    test_argv = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q"]
    direct = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests/python").glob("test_adversarial_verify*.py")
    )
    direct += [
        "tests/python/test_current_work_receipt.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_qwen_judge_9107.py",
        "tests/python/test_invocation_provenance_9000.py",
        "tests/python/test_count_validator_correction_8567.py",
    ]
    commands = [
        (
            "owned_tests",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *TESTS,
            ],
        ),
        ("direct_consumers", test_argv + direct),
        ("e2e018", test_argv + ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]),
        ("e2e021", test_argv + ["tests/python/test_restricted_decision_audit_8210.py"]),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *CHANGED, *TESTS]),
        (
            "ruff_format",
            [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *CHANGED, *TESTS],
        ),
        (
            "strict_mypy",
            [
                str(ROOT / ".venv/bin/mypy"),
                "--strict",
                "--follow-imports=silent",
                *OWNED,
                CHANGED[1],
            ],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS]),
    ]
    return [dict(name=n, argv=a, expected=0, deadline=600, scope="owned") for n, a in commands]


def plan(raw: Path) -> Json:
    """Freeze exact paths and commands before outcomes; repository health is separate."""
    raw.mkdir(parents=True, exist_ok=True)
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED + CHANGED)
        + "[report]\nexclude_lines=\nomit=\n"
    )
    manifest = dict(
        commands=validation_commands(raw),
        code_config_hashes=[ref(ROOT / p) for p in OWNED + CHANGED + TESTS],
        repository_health_command=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        coverage_configuration=ref(config),
        frozen_ns=time.monotonic_ns(),
        e2e020_applicability="Not applicable: no learning or recovery implementation.",
        cli_commands=[
            dict(name=n, argv=a, expected=e) for n, a, e in cli_plan(raw / "private_cli")
        ],
    )
    atomic_json(raw / "validation_plan.json", manifest)
    return manifest


def statement_coverage(raw: Path, evidence: Json) -> Json:
    """Count new statements only, using immutable pre-repair source for shared readers."""
    tracer = Coverage(config_file=str(raw / "coverage.ini"), data_file=str(raw / ".coverage"))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    files = coverage_rows(tracer, evidence)
    total = sum(len(r["statements"]) for r in files)
    return dict(
        statement_percent=100 * (total - sum(len(r["missing"]) for r in files)) / total,
        excluded_statements=sum(len(r["excluded"]) for r in files),
        files=files,
        data_reference=ref(raw / ".coverage"),
    )


def coverage_rows(tracer: Coverage, evidence: Json) -> list[Json]:
    """Use actual executed lines and pinned pre-repair code, with no exclusions."""
    files = []
    for name in OWNED + CHANGED:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        selected = set(statements)
        if name in CHANGED:
            key = "old_validator" if name == CHANGED[0] else "old_receipt"
            before = bound(evidence["snapshots"][key]).read_text().splitlines()
            after = (ROOT / name).read_text().splitlines()
            changed = {
                i + 1
                for tag, _, _, start, end in difflib.SequenceMatcher(
                    a=before, b=after
                ).get_opcodes()
                if tag in {"insert", "replace"}
                for i in range(start, end)
            }
            selected &= changed
        files.append(
            dict(
                path=name,
                statements=sorted(selected),
                missing=sorted(selected & set(missing)),
                excluded=sorted(selected & set(excluded)),
            )
        )
    return files


def validation_qualified(work: Json, receipts: list[Json], coverage: Json) -> bool:
    """Reopen logs and coverage so passing summary fields cannot manufacture readiness."""
    try:
        scope = work["scope"]
        raw = bound(scope["coverage_configuration"]).parent
        if scope["commands"] != validation_commands(raw) or scope["cli_commands"] != [
            dict(name=n, argv=a, expected=e) for n, a, e in cli_plan(raw / "private_cli")
        ]:
            return False
        commands = scope["commands"] + scope["cli_commands"]
        if len(receipts) != len(commands) or not all(
            r["name"] == c["name"]
            and r["argv"] == c["argv"]
            and r["expected_exit"] == c["expected"]
            and r["passed"] is True
            and r["normal_exit"] is True
            and r["timed_out"] is False
            for r, c in zip(receipts, commands, strict=True)
        ):
            return False
        for reference in scope["code_config_hashes"]:
            bound(reference)
        for row in receipts:
            for stream in ["stdout", "stderr"]:
                bound(dict(path=row[stream + "_path"], sha256=row[stream + "_sha256"]))
        tracer = Coverage(config_file=False, data_file=str(bound(coverage["data_reference"])))
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        files = coverage_rows(tracer, work["evidence"])
        return files == coverage["files"] and all(
            r["statements"] and not r["missing"] and not r["excluded"] for r in files
        )
    except (OSError, ValueError, KeyError):
        return False


def build(
    work: Json, validation: list[Json], coverage: Json, date: str, start: int, end: int
) -> Json:
    """Administrative readiness needs every affected check, and never repairs old science."""
    checks = work["evidence"]["checks"]
    failed = [c for c in checks if not c["passed"]]
    controls = work["semantic_control_rows"] + work["preserved_detection_rows"]
    qualified = validation_qualified(work, validation, coverage)
    diagnostic = work["historical_judge_diagnostic"]
    semantic_pass = (
        bool(controls)
        and all(r["passed"] for r in controls)
        and (
            diagnostic["repaired_reader_report"]["flag_count"] == 0
            and not any(diagnostic["original_receipt_errors"])
        )
    )
    ready = qualified and semantic_pass and not failed
    receipt = build_current_work_receipt(
        run_id="exp9111",
        owner_pid=work["current_owner_pid"],
        events=[],
        inference_substrate="deterministic_no_model_load",
        inference_substrate_details=dict(
            work="CPU evidence audit",
            historical_model_work="Opaque pinned upstream provenance only",
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    value = dict(
        experiment_id=9111,
        task_id=TASK,
        milestone="2026.11.777",
        schema="carnot.forward_evidence.v1",
        run_date=date,
        honest_verdict="complete_blocked_" + failed[0]["check"]
        if failed
        else "complete_circular_positive_forward_evidence_qualified"
        if ready
        else "complete_disqualified_owned_checks",
        verdict_class="blocked" if failed else "circular_positive" if ready else "disqualified",
        forward_evidence_ready_score=int(ready),
        required_checks_passed=bool(qualified and semantic_pass),
        flagged_adversarial=not ready and not failed,
        gate_check_summary=checks,
        rows=controls,
        original_flag_rows=work["original_flag_rows"],
        semantic_control_rows=work["semantic_control_rows"],
        preserved_detection_rows=work["preserved_detection_rows"],
        structural_count_derivations=work["structural_count_derivations"],
        historical_judge_diagnostic_reference=work.get("historical_judge_diagnostic_reference"),
        changed_validator_paths=[*CHANGED, OWNED[0]],
        validator_hashes=[ref(ROOT / p) for p in [*CHANGED, OWNED[0]]],
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="no_model_load",
        inference_substrate="deterministic_no_model_load",
        current_llm_invocations=0,
        model_invocation_counts=receipt["invocation_counts"],
        generated_tokens=0,
        duration_s=(end - start) / 1e9,
        random_seed=7779111,
        current_work_receipts=[receipt],
        intended_count=19,
        attempted_count=len(controls),
        completed_count=sum(r["passed"] for r in controls),
        failed_count=sum(not r["passed"] for r in controls),
        censored_count=19 - len(controls),
        sample_size_budget=dict(semantic_controls=14, rejecting_structural_controls=5),
        effective_independent_groups=0,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Evidence semantics and administrative readiness only; no human benefit or ARC solve.",
        exposure_scope="LPHB exposed development; archived representations are historical provenance.",
        validation_receipts=validation,
        owned_statement_coverage=coverage,
        code_config_hashes=work.get("scope", {}).get("code_config_hashes", []),
        source_references=work["evidence"]["references"],
        acceptance_gates=dict(
            immutable_inputs=not failed,
            semantic_controls=bool(semantic_pass),
            owned_validation=bool(qualified),
        ),
    )
    replay_input = dict(
        work=work,
        validation=validation,
        coverage=coverage,
        date=date,
        start=start,
        end=end,
        owner_pid=receipt["current_owner_pid"],
    )
    value["replay_input"] = replay_input
    value["reproducibility_checksum"] = canonical_hash(replay_input)
    value["field_principles"] = principles()
    return value


def principles() -> dict[str, str]:
    """Name the separate scientific constraint each important field protects."""
    return dict(
        honest_verdict="Terminal wording prevents missing inputs from becoming retryable partial work.",
        verdict_class="Exact administrative controls confer circular validation, not an empirical benefit.",
        gate_check_summary="Typed gate operands distinguish absent files, absent fields and observed zero.",
        forward_evidence_ready_score="Every semantic control and affected validation must pass before reuse.",
        original_flag_rows="Immutable original detections remain visible after a reader is corrected.",
        semantic_control_rows="Positive and rejecting fixtures test operational meaning before natural use.",
        structural_count_derivations="Unique original question groups and floor(TEST/2) prove separate counts.",
        preserved_detection_rows="Rejecting controls prevent a targeted repair from weakening real detection.",
        historical_judge_diagnostic_reference="A diagnostic cannot republish or qualify quarantined science.",
        changed_validator_paths="The repair stays at shared readers instead of hiding producer fields.",
        validator_hashes="Reader identities make a later code change detectable.",
        MODEL_SPECS="An empty requested model list prevents administrative work from claiming a new model.",
        requested_inference_substrate_class="The declared budget forbids loading a model during this audit.",
        inference_substrate_class="Measured operation type remains separate from historical model activity.",
        inference_substrate="The CPU artifact audit is explicit and cannot borrow a live model duration floor.",
        current_llm_invocations="Only current owned model operations count as current invocations.",
        model_invocation_counts="Separate load, forward and sampling counters prevent cross-operation negation.",
        generated_tokens="Forward scoring and artifact validation do not imply token sampling.",
        duration_s="Monotonic elapsed time prevents padding a cheap check to resemble model work.",
        random_seed="Fixed control identities prevent outcome-dependent fixture selection.",
        reproducibility_checksum="The replay inputs bind reductions to retained evidence.",
        code_config_hashes="Validation is tied to the exact code and test assertions that ran.",
        source_references="Authenticated snapshots preserve historical authority despite checkout drift.",
        current_work_receipts="An empty owned ledger cannot absorb upstream positive model counters.",
        validation_receipts="Normal exits and durable command logs distinguish checks from assertions of checks.",
        required_checks_passed="Failed affected checks cannot be excused by successful artifact writing.",
        acceptance_gates="Source custody, semantic behavior and execution validation remain separate gates.",
        rows="Original control identities retain each attempted administrative unit and its outcome.",
        effective_independent_groups="Scripted controls provide no new independent human observations.",
        sample_size_budget="The fixed control roster prevents shrinking the panel after failures.",
        verifier_is_oracle="Exact fixture expectations are circular administrative evidence.",
        claim_scope="Evidence semantics establishes no judge accuracy, human benefit or ARC result.",
        exposure_scope="Previously exposed LPHB rows remain development evidence.",
        independent_generalization_score="Reusing exposed evidence cannot establish independent generalization.",
        owned_statement_coverage="All new executable statements must be measured with zero exclusions.",
    )


def replay(path: Path) -> Json:
    """Rebuild totals from pinned inputs and reject rehashed aggregate tampering."""
    value = json.loads(path.read_bytes())
    saved = value["replay_input"]
    work = saved["work"]
    for reference in value["source_references"] + value["code_config_hashes"]:
        bound(reference)
    if value["reproducibility_checksum"] != canonical_hash(saved):
        raise ValueError("replay_hash")
    for row in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            bound(dict(path=row[stream + "_path"], sha256=row[stream + "_sha256"]))
    rebuilt = build(
        work, saved["validation"], saved["coverage"], saved["date"], saved["start"], saved["end"]
    )
    if rebuilt != value:
        raise ValueError("reduction_drift")
    if work.get("historical_judge_diagnostic_reference"):
        bound(work["historical_judge_diagnostic_reference"])
        evidence = work["evidence"]
        judge = json.loads(bound(evidence["snapshots"]["judge"]).read_bytes())
        diagnostic_input = dict(judge, structural_count_evidence="qid-panel-v1")
        diagnostic_path = bound(work["diagnostic_input_reference"])
        if json.loads(diagnostic_path.read_bytes()) != diagnostic_input:
            raise ValueError("diagnostic_input_drift")
        roster_rows = [
            r
            for key in ["shard_a_roster_reference", "shard_b_roster_reference"]
            for r in json.loads(bound(evidence["snapshots"][key]).read_bytes())["rows"]
            if r["role"] in {"CAL", "TEST"}
        ]
        if (
            evidence["roster_rows"] != roster_rows
            or historical_report(evidence)["flags"] != value["original_flag_rows"]
        ):
            raise ValueError("historical_evidence_drift")
        actual_diagnostic = av._verify_artifact_impl(diagnostic_path)
        if actual_diagnostic != work["historical_judge_diagnostic"]["repaired_reader_report"]:
            raise ValueError("diagnostic_report_drift")
        derived = structural_derivations(diagnostic_input, evidence)
        if (
            derived != value["structural_count_derivations"]
            or semantic_control_rows() != value["semantic_control_rows"]
        ):
            raise ValueError("primitive_reduction_drift")
        if structural_controls(diagnostic_input) != value["preserved_detection_rows"]:
            raise ValueError("structural_control_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Both real checkers must accept the candidate bytes before atomic publication."""
    replay(path)
    receipts = [
        child(
            name,
            [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)],
            raw,
            deadline=120,
            heartbeat=30,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def foreign_argv(*args: str) -> list[str]:
    """The real CLI must not depend on the caller's working directory or import path."""
    return [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        "/tmp",
        str(ROOT / ".venv/bin/python"),
        str(ROOT / CLI),
        *args,
    ]


def cli_plan(raw: Path) -> list[tuple[str, list[str], int]]:
    """Freeze all actual success, blocked and tamper commands before measurement."""
    success = raw / "success" / (NAME + ".json")
    blocked = raw / "blocked" / (NAME + ".json")
    return [
        ("cli_success", foreign_argv("--diagnostic-only", "--output", str(success)), 0),
        (
            "cli_blocked",
            foreign_argv(
                "--diagnostic-only", "--input-root", str(raw / "absent"), "--output", str(blocked)
            ),
            0,
        ),
        ("cli_cold_replay", foreign_argv("--replay", str(success)), 0),
        ("cli_blocked_replay", foreign_argv("--replay", str(blocked)), 0),
        ("cli_tamper", foreign_argv("--replay", str(raw / "tampered.json")), 1),
    ]


def cli_controls(raw: Path) -> list[Json]:
    """Use actual archived evidence in private outputs and refuse changed aggregates."""
    raw.mkdir(parents=True, exist_ok=True)
    commands = cli_plan(raw)
    receipts = [child(n, a, raw / "logs", expected=e, deadline=240) for n, a, e in commands[:4]]
    success = raw / "success" / (NAME + ".json")
    value = json.loads(success.read_bytes())
    value["completed_count"] += 1
    value["reproducibility_checksum"] = canonical_hash(value["replay_input"])
    tampered = raw / "tampered.json"
    atomic_json(tampered, value)
    n, a, e = commands[-1]
    receipts.append(child(n, a, raw / "logs", expected=e, deadline=240))
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Keep model-free measurement bounded and publish only a checked terminal artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--schedule-v779", type=Path)
    parser.add_argument("--schedule-invocation")
    args = parser.parse_args(argv)
    try:
        if args.schedule_v779:
            from carnot.reporting.validation_schedule_9139 import orchestration

            print(
                json.dumps(orchestration(args.schedule_v779, args.schedule_invocation)), flush=True
            )
            return 0
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.date != "20261115":
            raise ValueError("run_date")
        start = time.monotonic_ns()
        raw = args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        scope = plan(raw)
        evidence = authenticate(args.input_root, raw)
        work = measure(evidence, raw)
        work["scope"] = scope
        validation, coverage = [], {}
        if not args.diagnostic_only:
            progress("validation_before")
            with ThreadPoolExecutor(max_workers=1) as pool:
                health_future = pool.submit(
                    child,
                    "repository_health",
                    scope["repository_health_command"],
                    raw / "health",
                    deadline=900,
                    scope="repository_health",
                )
                validation = execute(scope["commands"], raw / "validation")
                coverage = (
                    statement_coverage(raw, evidence)
                    if all(c["passed"] for c in evidence["checks"])
                    else dict(
                        statement_percent=0,
                        excluded_statements=0,
                        blocker="authenticated_baseline_unavailable",
                    )
                )
                validation.extend(cli_controls(raw / "private_cli"))
                health = health_future.result()
            progress("validation_after")
            atomic_json(raw / "repository_health.json", health)
        value = build(work, validation, coverage, args.date, start, time.monotonic_ns())
        atomic_json(raw / "work.json", work)
        progress("publication_before")
        publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        progress("publication_after")
        print(
            json.dumps(
                dict(
                    path=str(args.output),
                    honest_verdict=value["honest_verdict"],
                    forward_evidence_ready_score=value["forward_evidence_ready_score"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print("[exp9111] refused=" + str(error), flush=True)
        return 1
