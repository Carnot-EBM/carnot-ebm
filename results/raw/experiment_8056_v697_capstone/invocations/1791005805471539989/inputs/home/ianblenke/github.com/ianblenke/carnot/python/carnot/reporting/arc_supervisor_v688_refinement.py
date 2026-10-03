"""Seal a current supervisor inspection without model calls. REQ-REPORT-7936-TERMINAL."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import arc_supervisor_v687_delta as previous
from carnot.reporting import arc_supervisor_v688_receipts as reader
from carnot.reporting import v686_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts import conductor_gates, in_process_doc_reconcile

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "results/experiment_7924_v687_arc_supervisor_delta.json"
INVENTORY = ROOT / "results/raw/experiment_7924_v687_arc_supervisor_delta/receipt_inventory.json"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
OUTPUT = ROOT / "results/experiment_7936_v688_arc_supervisor_refinement.json"
PINNED = {
    str(PRIOR): "sha256:764c42c087183050ba5160b2617f5459822a64f4a54c49d27c5ebc7e362efc28",
    str(INVENTORY): "sha256:e1741199e078912e42807c891565de165d777069bd9c660bfb15f5854d6ee88e",
    str(REGISTRY): "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}
MODULES = [
    "python/carnot/reporting/arc_supervisor_v688_receipts.py",
    "python/carnot/reporting/arc_supervisor_v688_refinement.py",
]
CLI = "scripts/experiments/experiment_7936_v688_arc_supervisor_refinement.py"
TEST = "tests/python/test_arc_supervisor_refinement_7936.py"
INCLUDE = ",".join("*/" + path for path in [*MODULES, CLI])
CONSUMERS = [
    "tests/python/test_arc_supervisor_refinement.py",
    "tests/python/test_arc_trajectory_supervisor.py",
    "tests/python/test_arc_supervisor_delta_7924.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
]


def inputs() -> dict[str, Any]:
    """Pin accepted and seen authorities before enumerating current producers."""
    checks = []
    for path in (PRIOR, INVENTORY, REGISTRY):
        observed = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            dict(
                upstream_id=path.stem,
                path=str(path),
                sha256=None if observed == "missing" else observed,
                artifact_field="sha256",
                op="==",
                expected=PINNED[str(path)],
                observed=observed,
                role="solve_registry_precheck" if path == REGISTRY else "frozen_exp7924_inventory",
                exposure_status="exposed_development",
            )
        )
    failures = [r for r in checks if r["expected"] != r["observed"]]
    prior = json.loads(PRIOR.read_text()) if not failures else {}
    inventory = json.loads(INVENTORY.read_text()) if not failures else {}
    registry = yaml.safe_load(REGISTRY.read_text()) if not failures else {}
    for field, expected in (
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("arc_delta_ready_score", 1),
    ):
        if prior:
            check = dict(
                upstream_id="exp7924",
                path=str(PRIOR),
                sha256=sha256_file(PRIOR),
                artifact_field=field,
                op="==",
                expected=expected,
                observed=prior.get(field),
                role="qualified_baseline",
            )
            checks.append(check)
            if check["expected"] != check["observed"]:
                failures.append(check)
    return dict(
        prior=prior, inventory=inventory, registry=registry, checks=checks, failures=failures
    )


def commands(private: Path) -> list[dict[str, Any]]:
    """Reuse bounded historical checks with current includes and private fixtures."""
    private.mkdir(parents=True, exist_ok=True)
    rows = previous.commands(private)
    fixture = private / "results/raw/fixture/rows.json"
    atomic_json(
        fixture,
        {
            "rows": [
                dict(
                    game="fixture",
                    seed=0,
                    invocation_id="fixture-call",
                    receipt_id="fixture-receipt",
                    solve_provenance="live_agent_self_discovery",
                    event_timestamp="2026-09-30T12:00:00Z",
                    trajectory_supervisor=dict(
                        enabled=True,
                        mode="applied",
                        redirects=[
                            dict(
                                id="r",
                                arm="drop_goal_bias",
                                fired=True,
                                resolved_by_levelup=True,
                                actions_to_levelup=3,
                            )
                        ],
                    ),
                )
            ]
        },
    )
    atomic_json(
        private / "producer.json",
        dict(
            run_date="20260930",
            verdict_class="null",
            source_artifact_hashes={fixture.relative_to(private).as_posix(): sha256_file(fixture)},
        ),
    )
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
            row["argv"].extend(CONSUMERS)
        if row["name"] in {"ruff_check", "ruff_format", "mypy"}:
            row["argv"] = [
                row["argv"][0],
                *(
                    ["--strict"]
                    if row["name"] == "mypy"
                    else ["check"]
                    if row["name"] == "ruff_check"
                    else ["format", "--check"]
                ),
                *MODULES,
                CLI,
            ]
            if row["name"] != "mypy":
                row["argv"].append(TEST)
        if row["name"] == "scoped_spec":
            row["argv"].append(TEST)
        if row["name"] == "full_python_suite":
            row["deadline_s"] = 600
    return rows


def run(spec: dict[str, Any], private: Path, durable: Path) -> dict[str, Any]:
    """The shared runner prints boundaries, enforces deadlines and seals exited logs."""
    result = validation.run_check(ROOT, spec, private, durable / "sealed")
    result["passed"] = result["passed"] and (
        not spec.get("expected_text") or spec["expected_text"] in result["output_tail"]
    )
    return result


def resolution(output: Path, durable: Path) -> dict[str, Any]:
    """Actual readers must return the primary after raw sidecars become newer."""
    atomic_json(durable / "newer_sidecar.json", dict(role="validator_sidecar"))
    result = conductor_gates.evaluate_gates(
        dict(
            gated_on=[
                dict(
                    upstream="exp7936-arc-supervisor-refinement",
                    artifact_field="experiment_id",
                    op="==",
                    value=7936,
                )
            ]
        ),
        results_dir=output.parent,
    )
    selected = result.gates_evaluated[0]
    reconciled = in_process_doc_reconcile.find_artifact(
        "exp7936-arc-supervisor-refinement", output.parent
    )
    digest = sha256_file(output)
    assert result.passed and Path(selected.artifact_path or "") == output
    assert selected.artifact_sha256 == digest and reconciled == output
    receipt = dict(
        path=str(output),
        sha256=digest,
        conductor_path=selected.artifact_path,
        conductor_sha256=selected.artifact_sha256,
        reconciliation_path=str(reconciled),
        reconciliation_sha256=sha256_file(reconciled),
        sidecar_sha256=sha256_file(durable / "newer_sidecar.json"),
    )
    atomic_json(durable / "primary_resolution_receipt.json", receipt)
    return receipt


def publish(value: dict[str, Any], output: Path, private: Path, durable: Path) -> None:
    """Changed verdicts require another validation of their exact final bytes."""
    candidate = private / "terminal_candidate.json"
    for attempt in range(3):
        atomic_json(candidate, value)
        digest = sha256_file(candidate)
        reports = [
            run(
                dict(
                    name=name,
                    argv=[str(ROOT / ".venv/bin/python"), script, option, str(candidate)],
                    deadline_s=90,
                    expected_exit=0,
                    classification="required",
                    expected_text=None,
                ),
                private,
                durable,
            )
            for name, script, option in (
                ("terminal_adversarial", "scripts/adversarial_verify.py", "--json"),
                ("terminal_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ]
        atomic_json(
            durable / f"terminal_reports_{attempt}.json",
            dict(candidate_sha256=digest, reports=reports),
        )
        flagged = not reports[0]["passed"]
        if all(r["passed"] for r in reports) and value["flagged_adversarial"] == flagged:
            atomic_json(
                durable / "terminal_reports.json", dict(candidate_sha256=digest, reports=reports)
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name("." + output.name + ".checked")
            shutil.copyfile(candidate, temporary)
            assert sha256_file(temporary) == digest
            temporary.replace(output)
            return
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            flagged_adversarial=flagged,
            arc_evidence_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].extend(
            dict(
                upstream_id="exp7936-terminal",
                path=r["log_path"],
                sha256=r["log_sha256"],
                artifact_field="passed",
                op="==",
                expected=True,
                observed=False,
            )
            for r in reports
            if not r["passed"]
        )
    raise ValueError("terminal_recheck_failed")


def execute(output: Path, private: Path) -> int:
    """Freeze inputs, run owned checks once and publish a terminal observational result."""
    started = time.monotonic()
    print("[exp7936] phase=authenticate completed_units=0", flush=True)
    checked = inputs()
    durable = ROOT / "results/raw/experiment_7936_v688_arc_supervisor_refinement"
    specs = commands(private)
    producers = [
        p
        for p in sorted((ROOT / "results").glob("experiment_*_*.json"))
        if p.name.split("_")[1].isdigit() and 7831 < int(p.name.split("_")[1]) < 7936
    ]
    closure = [
        *MODULES,
        CLI,
        TEST,
        *CONSUMERS,
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
    ]
    sources = validation.dependency_hashes(ROOT, paths=closure)
    sources.update({r["path"]: r["sha256"] for r in checked["checks"] if r["sha256"]})
    sources.update({str(p): sha256_file(p) for p in producers})
    for producer in producers:
        try:
            doc = json.loads(producer.read_text())
            hashes = doc.get("source_artifact_hashes", {}) if isinstance(doc, dict) else {}
            for label in hashes if isinstance(hashes, dict) else []:
                path = Path(label)
                path = (path if path.is_absolute() else ROOT / path).resolve()
                if (
                    path.is_relative_to(ROOT / "results/raw")
                    and path.is_file()
                    and path.suffix == ".json"
                ):
                    sources[str(path)] = sha256_file(path)
        except ValueError:
            continue
    for label in (
        "pyproject.toml",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/arc-agi/spec.md",
    ):
        sources[label] = sha256_file(ROOT / label)
    manifest = durable / "validation_command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=specs,
            dependency_hashes=sources,
            producers=[str(p) for p in producers],
            coverage_includes=[*MODULES, CLI],
            coverage_include=INCLUDE,
            preconditions=checked["checks"],
            environment=dict(PYTHONPATH="python:.", JAX_PLATFORMS="cpu", CARNOT_FORCE_LIVE="1"),
            applicable_e2e=["E2E-016", "E2E-017"],
            execution_date="20260930",
            historical_fixture_date="20260929",
        ),
    )
    sources[str(manifest)] = sha256_file(manifest)
    print(
        f"[exp7936] phase=scope_frozen completed_units={len(sources)} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    prior = checked["prior"]
    frozen = checked["inventory"]
    seen = frozen.get("seen_receipt_hashes", {})
    delta = reader.inspect(
        ROOT, producers if not checked["failures"] else [], seen, "20260930", "20260930", {}
    )
    delta.update(
        cutoff_receipt_hashes=frozen.get("cutoff_receipt_hashes", {}), seen_receipt_hashes=seen
    )
    gates = list(checked["failures"])
    gates.extend(
        dict(
            upstream_id=Path(r["producer_path"]).stem,
            path=r["source_path"],
            sha256=r["source_sha256"],
            artifact_field="sha256",
            op="==",
            expected=r["expected_sha256"],
            observed=r["source_sha256"] or "missing",
        )
        for r in delta["rows"]
        if r["reason"] in {"missing_raw", "raw_hash_mismatch"}
    )
    inventory = durable / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    checkpoint = canonical_hash(sources)
    atomic_json(
        durable / ("checkpoint-" + checkpoint[7:] + ".json"),
        dict(code_config_input_sha256=checkpoint, inventory_sha256=sha256_file(inventory)),
    )
    reduced = time.monotonic() - started
    print(
        f"[exp7936] phase=live_reduced completed_units={delta['identity_filter_count']} elapsed_s={reduced:.3f}",
        flush=True,
    )
    receipts = [run(spec, private, durable) for spec in specs]
    coverage_path = private / "coverage.json"
    counts = json.loads(coverage_path.read_text())["files"] if coverage_path.is_file() else {}
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    complete = validation.coverage_complete(coverage_path, includes=[*MODULES, CLI])
    gates.extend(
        dict(
            upstream_id="exp7936",
            path=r["log_path"],
            sha256=r["log_sha256"],
            artifact_field="exit_code",
            op="==",
            expected=r["expected_exit"],
            observed=r["exit_code"],
            check=r["name"],
        )
        for r in failed
    )
    state = (
        "disqualified"
        if failed or not complete or reader.replay(delta)
        else "blocked"
        if gates
        else "null"
    )
    finished = time.monotonic() - started
    verdict = (
        "complete_"
        + state
        + (
            "_no_recovered_live_supervisor_events"
            if state == "null" and not delta["identity_filter_count"]
            else "_observational_receipt_refinement"
        )
    )
    value = dict(
        delta,
        experiment_id=7936,
        task_id="exp7936-arc-supervisor-refinement",
        milestone="2026.09.688",
        run_date="20260930",
        honest_verdict=verdict,
        verdict_class=state,
        flagged_adversarial=False,
        gate_check_summary=gates,
        arc_evidence_ready_score=int(state == "null"),
        acceptance_gate_results=dict(
            validity=state == "null",
            readiness=int(state == "null"),
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
            scientific_benefit=None,
        ),
        duration_s=finished,
        phase_spans=[
            dict(phase="authenticate_freeze_reduce", start_s=0, end_s=reduced, duration_s=reduced),
            dict(
                phase="owned_validation",
                start_s=reduced,
                end_s=finished,
                duration_s=finished - reduced,
            ),
        ],
        random_seed=0,
        reproducibility_checksum=checkpoint,
        source_artifact_hashes=sources,
        preconditions_checked=checked["checks"],
        resolved_imports={
            name: str(Path(mod.__file__).resolve())
            for name, mod in tuple(sys.modules.items())
            if getattr(mod, "__file__", None)
            and (name.startswith("carnot.") or name.startswith("scripts."))
        },
        validation_receipts=receipts,
        validation_command_manifest_path=str(manifest),
        observed_child_commands=[r["command_argv"] for r in receipts],
        coverage_statement_counts={name: row["summary"] for name, row in counts.items()},
        historical_required_failures=prior.get("historical_required_failures", []),
        baseline_receipt=dict(
            path=str(PRIOR), sha256=PINNED[str(PRIOR)], accepted_cutoff_unchanged=True
        ),
        repository_health=dict(
            affects_required_checks=False,
            repository_wide_check_repeated=True,
            current_repository_checks=[
                r for r in receipts if r["classification"] == "repository_health"
            ],
        ),
        primary_resolution_receipt=dict(
            path=str(output),
            receipt_path=str(durable / "primary_resolution_receipt.json"),
            hash_binding="external sidecar avoids a self-referential artifact hash",
        ),
        terminal_validation_sidecar_path=str(durable / "terminal_reports.json"),
        verifier_is_oracle=False,
        claim_scope="exposed_development ARC-AGI-3 generalization research through supervisor refinement; no independent or causal benefit",
        fixture_claim_scope="circular_positive mechanics only; never independent scientific benefit",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model="none",
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        solve_provenance="live_agent_self_discovery",
        registry_precheck=checked["registry"].get("games", {}),
        current_date="20260930",
        historical_fixture_date="20260929",
        retire_if_same_verdict=dict(
            prior_id=7924,
            prior_verdict=prior.get("honest_verdict"),
            same_scientific_verdict=state == prior.get("verdict_class"),
            action="retire_unchanged_delta_only_scan" if state == "null" else "preserve_failures",
        ),
    )
    value["field_principles"] = {
        key: "Bind executing producer custody; recovered observations do not establish performance gain."
        for key in value
    }
    value["field_principles"].update(
        receipt_inventory="Immutable invocation and receipt IDs bind content hashes, separately from accepted cutoff bytes.",
        chronology_unknown_count="Missing authenticated order cannot become a prospective gain claim.",
        calendar_filter_count="Calendar and identity counts use the same authenticated corpus.",
        arm_outcomes="Censoring, unknown credit, Wilson bounds and held-out summaries limit association claims.",
        refinement_decisions="Curated sample and confidence thresholds permit proposals only; defaults remain unchanged.",
        acceptance_gate_results="Validity and readiness do not establish calibration or scientific benefit.",
        primary_resolution_receipt="Actual readers return the final primary path and hash in a bound external sidecar.",
    )
    publish(value, output, private, durable)
    resolution(output, durable)
    print(
        f"[exp7936] phase=published completed_units={delta['identity_filter_count']} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )
    return int(value["arc_evidence_ready_score"] == 0)
