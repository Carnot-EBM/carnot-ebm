"""Keep accepted ARC evidence distinct from seen receipts. REQ-REPORT-7924-V687."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting import v686_contract_validation as validation
from scripts.experiments import experiment_7911_v686_arc_supervisor_delta as previous

ROOT = Path(__file__).resolve().parents[3]
HISTORY = ROOT / "results/experiment_7911_v686_arc_supervisor_delta.json"
EXPECTED_HISTORY = "sha256:512dc8b230d54f37918d2acc535a92f909b5e03ffbfad91347cd1a48924ab197"
MODULE = "python/carnot/reporting/arc_supervisor_v687_delta.py"
CLI = "scripts/experiments/experiment_7924_v687_arc_supervisor_delta.py"
TEST = "tests/python/test_arc_supervisor_delta_7924.py"
INCLUDE = f"*/{MODULE},*/{CLI}"


def progress(started: float, phase: str, units: int = 0) -> None:
    """Print real phase boundaries so silence cannot hide unfinished work."""
    print(
        f"[exp7924] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def inputs() -> dict[str, Any]:
    """Authenticate the accepted cutoff and require every historical input ledger."""
    prior, checks, failures = previous.precheck()
    history: dict[str, Any] = {}

    def check(path: Path, expected: Any, role: str) -> bool:
        actual = sha256_file(path) if path.is_file() else "missing"
        row = dict(
            upstream_id=path.stem,
            path=str(path),
            sha256=None if actual == "missing" else actual,
            artifact_field="sha256",
            op="==",
            expected=expected,
            observed=actual,
            role=role,
            exposure_status="exposed_development",
        )
        checks.append(row)
        if expected != actual:
            failures.append(row)
        return expected == actual

    if check(HISTORY, EXPECTED_HISTORY, "disqualified_seen_only"):
        history = json.loads(HISTORY.read_text())
        ledger = Path(history["receipt_inventory_path"])
        expected = history.get("source_artifact_hashes", {}).get(
            str(ledger), sha256_file(ledger) if ledger.is_file() else "required_ledger"
        )
        check(ledger, expected, "historical_seen_inventory")
    for path in sorted((ROOT / "results").glob("experiment_*arc*.json")):
        number = path.name.split("_")[1]
        if number.isdecimal() and 7899 < int(number) != 7924:
            document = json.loads(path.read_text())
            check(path, sha256_file(path), "current_producer")
            for label, digest in document.get("source_artifact_hashes", {}).items():
                raw = Path(label)
                raw = raw if raw.is_absolute() else ROOT / raw
                if raw.is_relative_to(ROOT / "results/raw") and raw.suffix == ".json":
                    check(
                        raw,
                        digest.get("sha256") if isinstance(digest, dict) else digest,
                        "required_raw_ledger",
                    )
    return dict(prior=prior, history=history, checks=checks, failures=failures)


def audit(checked: dict[str, Any]) -> dict[str, Any]:
    """Use the old reducer with a seen-only inventory; no rejected headline is accepted."""
    prior, history = checked["prior"], checked["history"]
    accepted = dict(prior.get("cutoff_receipt_hashes", {}))
    seen = dict(accepted)
    for row in history.get("rows", []) + history.get("outcome_rows", []):
        if row.get("event_id") and row.get("content_sha256"):
            seen[row["event_id"]] = row["content_sha256"]
        if row.get("source_sha256"):
            seen["raw:" + row["source_sha256"]] = row["source_sha256"]
    for label, digest in history.get("source_artifact_hashes", {}).items():
        path = Path(label)
        path = path if path.is_absolute() else ROOT / path
        if path.is_relative_to(ROOT / "results/raw") and path.suffix == ".json":
            digest = digest.get("sha256") if isinstance(digest, dict) else digest
            seen["raw:" + digest] = digest
    adapted = dict(prior, cutoff_receipt_hashes=seen)
    delta = previous.audit(adapted)
    delta.update(
        cutoff_receipt_hashes=accepted,
        seen_receipt_hashes=seen,
        null_fast_path=delta["new_outcome_count"] == 0,
    )
    return delta


def commands(private: Path) -> list[dict[str, Any]]:
    """Adapt the existing command builder while retaining its consumer and E2E closure."""
    rows = previous.commands(private)
    for row in rows:
        row["argv"] = [
            CLI
            if arg == previous.CLI
            else "--include=" + INCLUDE
            if arg.startswith("--include=")
            else arg
            for arg in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "unit_coverage"}:
            row["argv"].append(TEST)
        if row["name"] in {"ruff_check", "ruff_format", "mypy"}:
            row["argv"].append(MODULE)
        if row["name"] in {"ruff_check", "ruff_format", "scoped_spec"}:
            row["argv"].append(TEST)
        if row["name"].startswith("e2e_016"):
            row["argv"][row["argv"].index("--date") + 1] = "20260929"
    combine = next(row for row in rows if row["name"] == "coverage_combine")
    combine["argv"].insert(2, "--keep")
    combine["argv"].append(str(private / "coverage.terminal"))
    replay = next(row for row in rows if row["name"] == "cli_replay_coverage")
    terminal = dict(
        replay,
        name="cli_terminal_coverage",
        argv=[
            arg.replace("coverage.replay", "coverage.terminal").replace(
                "--cold-replay", "--terminal-recheck"
            )
            for arg in replay["argv"]
        ],
    )
    rows.insert(rows.index(combine), terminal)
    report = next(row for row in rows if row["name"] == "coverage_report")
    rows.insert(
        rows.index(report) + 1,
        dict(
            report,
            name="coverage_json",
            argv=[
                str(ROOT / ".venv/bin/coverage"),
                "json",
                f"--data-file={private / 'coverage.combined'}",
                "--include=" + INCLUDE,
                "-o",
                str(private / "coverage.json"),
            ],
        ),
    )
    fixture = next(row for row in rows if row["name"] == "e2e_016_fixture-e2e")
    wrong = dict(
        fixture,
        name="e2e_016_wrong_date",
        argv=list(fixture["argv"]),
        expected_exit=1,
        expected_text="run_date_mismatch",
    )
    wrong["argv"][wrong["argv"].index("--date") + 1] = "20260930"
    rows.append(wrong)
    return rows


def replay(value: dict[str, Any]) -> list[str]:
    """Recount primitive firing dispositions so a counterfeit headline fails cold replay."""
    errors = previous.replay_delta(value)
    live = [row for row in value["rows"] if row["status"] in {"completed", "censored"}]
    if value.get("new_live_outcome_count", len(live)) != len(live):
        errors.append("new_live_outcome_count")
    if value.get("new_level_solves") != 0:
        errors.append("new_level_solves")
    for status in ("completed", "failed", "censored", "excluded"):
        if value["sample_size_budget"][status] != sum(
            row["status"] == status for row in value["rows"]
        ):
            errors.append("sample_size_budget." + status)
    budget = dict(
        intended=len(value["rows"]),
        eligible=len(live),
        started=len(live),
        independent_n=len({row["event_id"] for row in live}),
    )
    errors.extend(
        "sample_size_budget." + key
        for key, count in budget.items()
        if value["sample_size_budget"][key] != count
    )
    games = {str(row["game"]) for row in live}
    if set(value["per_game_results"]) != games:
        errors.append("per_game_results")
    for game in games:
        rows = [row for row in live if str(row["game"]) == game]
        cell = value["per_game_results"].get(game, {})
        if cell.get("eligible") != len(rows):
            errors.append("per_game_results.eligible")
        arms = {row["arm"] for row in rows}
        if set(cell.get("arms", {})) != arms:
            errors.append("per_game_results.arms")
        for arm in arms:
            selected = [row for row in rows if row["arm"] == arm]
            expected = dict(
                firings=len(selected),
                resolved_by_levelup=sum(row["resolved_by_levelup"] for row in selected),
                actions_to_levelup=[
                    row["actions_to_levelup"]
                    for row in selected
                    if isinstance(row["actions_to_levelup"], int)
                ],
                stagnations_unredirected=[
                    row["stagnations_unredirected"]
                    for row in selected
                    if isinstance(row["stagnations_unredirected"], int)
                ],
                source_run_hashes=list(dict.fromkeys(row["source_sha256"] for row in selected)),
            )
            if cell.get("arms", {}).get(arm) != expected:
                errors.append("per_game_results.arm_observations")
    eligible_arms = {
        row["arm"]
        for row in live
        if sum(other["arm"] == row["arm"] for other in live) >= 20
        and len({other["game"] for other in live if other["arm"] == row["arm"]}) >= 3
    }
    if {row.get("arm") for row in value["recommendation_rows"]} != eligible_arms:
        errors.append("recommendation_rows")
    return errors


def candidate(
    checked: dict[str, Any],
    delta: dict[str, Any],
    receipts: list[dict[str, Any]],
    manifest: Path,
    sources: dict[str, str],
    started: float,
    spans: list[dict[str, Any]],
    counts: dict[str, Any],
) -> dict[str, Any]:
    """Keep execution readiness separate from scientific benefit and historical failures."""
    failed = [row for row in receipts if row["classification"] == "required" and not row["passed"]]
    gates = list(checked["failures"])
    gates.extend(
        dict(
            upstream_id="exp7924",
            path=row["log_path"],
            sha256=row["log_sha256"],
            artifact_field="exit_code",
            op="==",
            expected=row["expected_exit"],
            observed=row["exit_code"],
            check=row["name"],
        )
        for row in failed
    )
    state = (
        "disqualified"
        if failed
        else "blocked"
        if gates
        else "null"
        if not delta["new_outcome_count"]
        else "positive"
    )
    verdict = {
        "disqualified": "complete_disqualified_required_validation",
        "blocked": "complete_blocked_source_precondition",
        "null": "complete_null_no_new_supervisor_outcomes",
        "positive": "complete_positive_observational_supervisor_delta",
    }[state]
    history = checked["history"]
    value = dict(delta)
    value.update(
        experiment_id=7924,
        task_id="exp7924-arc-supervisor-delta",
        milestone="2026.09.687",
        run_date="20260930",
        honest_verdict=verdict,
        verdict_class=state,
        flagged_adversarial=False,
        gate_check_summary=gates,
        outcome_rows=delta["rows"],
        acceptance_gate_results=dict(
            validity=not gates,
            readiness=int(not gates),
            scientific_benefit=None,
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        source_artifact_hashes=sources,
        preconditions_checked=checked["checks"],
        resolved_imports={
            name: str(Path(module.__file__).resolve())
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and (name.startswith("carnot.") or name.startswith("scripts."))
        },
        validation_receipts=receipts,
        validation_command_manifest_path=str(manifest),
        observed_child_commands=[row["command_argv"] for row in receipts],
        coverage_statement_counts=counts,
        historical_required_failures=[
            *checked["prior"].get("historical_required_failures", []),
            dict(
                experiment_id=7911,
                path=str(HISTORY),
                sha256=sha256_file(HISTORY) if HISTORY.is_file() else None,
                honest_verdict=history.get("honest_verdict"),
                required_failures=[
                    row
                    for row in history.get("validation_receipts", [])
                    if row.get("classification", "required") == "required" and not row["passed"]
                ],
                terminal_validation_sidecar_path=history.get("terminal_validation_sidecar_path"),
                resolved=False,
            ),
        ],
        repository_health=dict(
            affects_required_checks=False,
            current_repository_checks=[
                row for row in receipts if row["classification"] == "repository_health"
            ],
            repository_wide_check_repeated=True,
        ),
        verifier_is_oracle=False,
        claim_scope="exposed_development; observational live-receipt delta; scientific benefit unmeasured",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model="none",
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        arc_delta_ready_score=int(not gates),
        new_live_outcome_count=delta["new_outcome_count"],
        new_level_solves=0,
        solve_provenance=[
            row["solve_provenance"]
            for row in delta["rows"]
            if row["status"] in {"completed", "censored"}
        ],
        random_seed=0,
        reproducibility_checksum=canonical_hash(
            dict(sources=sources, seen=delta.get("seen_receipt_hashes", {}), seed=0)
        ),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        execution_date="20260930",
        historical_fixture_date="20260929",
        receipt_inventory_path=str(manifest.parent / "receipt_inventory.json"),
        terminal_validation_sidecar_path=str(manifest.parent / "terminal_reports.json"),
        retire_if_same_verdict=dict(
            prior_id=7899,
            prior_verdict=checked["prior"].get("honest_verdict"),
            same=verdict == checked["prior"].get("honest_verdict"),
            action="retire_unchanged_null_as_new_evidence"
            if state == "null"
            else "preserve_historical_failure",
        ),
    )
    value["sample_size_budget"].update(
        unit="supervisor_firing", independent=0, independence_scope="exposed_development"
    )
    value["field_principles"] = {
        key: "Bind current producer evidence and preserve historical custody; this field does not establish independent benefit."
        for key in value
    }
    value["field_principles"].update(
        cutoff_receipt_hashes="Only qualified Exp7899 defines the accepted cutoff.",
        seen_receipt_hashes="Disqualified input exposure prevents repeat evidence without accepting a headline.",
        acceptance_gate_results="Validity and readiness are separate from unmeasured scientific benefit.",
        new_level_solves="Receipt aggregation cannot solve a game.",
        coverage_statement_counts="Nonempty measured files must cover every added statement.",
    )
    return value


def run(spec: dict[str, Any], private: Path, durable: Path) -> dict[str, Any]:
    """Keep the shared child supervisor and check declared negative failure reasons."""
    row = validation.run_check(ROOT, spec, private, durable)
    row["passed"] = row["passed"] and (
        spec.get("expected_text") is None or spec["expected_text"] in row["output_tail"]
    )
    return row


def publish(value: dict[str, Any], output: Path, private: Path, durable: Path) -> None:
    """Validate each final byte sequence before atomic publication, including verdict changes."""
    path = private / "terminal_candidate.json"
    for attempt in range(3):
        atomic_json(path, value)
        digest = sha256_file(path)
        reports = [
            run(
                dict(
                    name=name,
                    argv=[str(ROOT / ".venv/bin/python"), script, option, str(path)],
                    deadline_s=90,
                    expected_exit=0,
                    classification="required",
                    expected_text=None,
                ),
                private,
                durable / "sealed",
            )
            for name, script, option in (
                ("terminal_adversarial", "scripts/adversarial_verify.py", "--json"),
                ("terminal_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ]
        atomic_json(
            durable / (f"terminal_reports_{attempt}.json"),
            dict(candidate_sha256=digest, reports=reports),
        )
        actual_flag = not reports[0]["passed"]
        if all(row["passed"] for row in reports) and value["flagged_adversarial"] == actual_flag:
            atomic_json(
                durable / "terminal_reports.json", dict(candidate_sha256=digest, reports=reports)
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name("." + output.name + ".checked")
            shutil.copyfile(path, temporary)
            assert sha256_file(temporary) == digest
            temporary.replace(output)
            return
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            flagged_adversarial=actual_flag,
            arc_delta_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].extend(
            dict(
                upstream_id="exp7924-terminal",
                path=row["log_path"],
                sha256=row["log_sha256"],
                artifact_field="passed",
                op="==",
                expected=True,
                observed=False,
            )
            for row in reports
            if not row["passed"]
        )
    raise ValueError("terminal_recheck_failed")


def resume_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse only diagnostic log bytes; every owned qualification check runs again."""
    prior = json.loads(path.read_text())
    assert prior["experiment_id"] == 7924 and prior["verdict_class"] == "disqualified"
    receipt = next(
        row for row in prior["validation_receipts"] if row["name"] == "full_python_suite"
    )
    if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
        raise ValueError("repository_health_log_hash_mismatch")
    return dict(receipt, reused=True, evidence_role="prior_attempt_diagnostic"), dict(
        experiment_id=7924,
        path=str(path),
        sha256=sha256_file(path),
        required_failures=prior["gate_check_summary"],
        resolved=False,
    )


def execute(output: Path, private: Path, resume_from: Path | None = None) -> int:
    """Freeze current work, validate it once, and publish its terminal evidence."""
    started = time.monotonic()
    progress(started, "start")
    checked = inputs()
    durable = ROOT / "results/raw/experiment_7924_v687_arc_supervisor_delta"
    specs = commands(private)
    reused, historical = resume_receipt(resume_from) if resume_from else ({}, {})
    closure = [
        MODULE,
        CLI,
        TEST,
        *previous.TESTS,
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
        "python/carnot/reporting/arc_supervisor_receipt_delta.py",
        "python/carnot/agentic/arc_solver_kit.py",
    ]
    sources = validation.dependency_hashes(ROOT, paths=closure)
    sources.update({row["path"]: row["sha256"] for row in checked["checks"] if row.get("sha256")})
    if historical:
        sources[str(resume_from)] = historical["sha256"]
    for name in ("pyproject.toml", "ops/exclusion_manifest.yaml"):
        sources[name] = sha256_file(ROOT / name)
    manifest = durable / "validation_command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=specs,
            dependency_hashes=sources,
            coverage_includes=[MODULE, CLI],
            coverage_include=INCLUDE,
            applicable_e2e=["E2E-016", "E2E-017"],
            environment=dict(PYTHONPATH="python:.", JAX_PLATFORMS="cpu"),
            execution_date="20260930",
            historical_fixture_date="20260929",
            preconditions=checked["checks"],
            repository_health_reuse=historical,
        ),
    )
    sources[str(manifest)] = sha256_file(manifest)
    progress(started, "scope_frozen", len(sources))
    delta = (
        audit(checked) if not checked["failures"] else previous.reduce_receipts(ROOT, [], {}, {})
    )
    inventory = durable / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    atomic_json(
        durable / ("checkpoint-" + canonical_hash(sources)[7:] + ".json"),
        dict(status="reduced", inventory_sha256=sha256_file(inventory)),
    )
    reduced = time.monotonic() - started
    progress(
        started,
        "complete_null_no_new_supervisor_outcomes"
        if not delta["new_outcome_count"]
        else "delta_reduced",
        delta["new_outcome_count"],
    )
    receipts = [
        reused
        if reused and spec["name"] == "full_python_suite"
        else run(spec, private, durable / "sealed")
        for spec in specs
    ]
    report = private / "coverage.json"
    counts = json.loads(report.read_text())["files"] if report.is_file() else {}
    finished = time.monotonic() - started
    spans = [
        dict(phase="authenticate_freeze_reduce", start_s=0, end_s=reduced, duration_s=reduced),
        dict(
            phase="owned_validation", start_s=reduced, end_s=finished, duration_s=finished - reduced
        ),
    ]
    value = candidate(
        checked,
        delta,
        receipts,
        manifest,
        sources,
        started,
        spans,
        {name: row["summary"] for name, row in counts.items()},
    )
    if historical:
        value["historical_required_failures"].append(historical)
        value["repository_health"]["repository_wide_check_repeated"] = False
        value["observed_child_commands"] = [
            row["command_argv"] for row in receipts if not row.get("reused")
        ]
    if replay(value) or not validation.coverage_complete(report, includes=[MODULE, CLI]):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_replay_or_coverage",
            arc_delta_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
    publish(value, output, private, durable)
    progress(started, "deliverable_written", delta["new_outcome_count"])
    return int(value["arc_delta_ready_score"] == 0)
