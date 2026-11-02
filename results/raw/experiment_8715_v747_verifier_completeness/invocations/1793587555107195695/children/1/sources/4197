"""Advance a qualified receipt frontier without repeating solves. REQ-REPORT-7962.

Empty evidence is useful when its custody is checked. Reused validation keeps
successful execution separate from scientific benefit and historical health.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import arc_supervisor_v689_delta as previous
from carnot.reporting import v686_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "results/experiment_7949_v689_arc_supervisor_delta.json"
INVENTORY = ROOT / "results/raw/experiment_7949_v689_arc_supervisor_delta/receipt_inventory.json"
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
OUTPUT = ROOT / "results/experiment_7962_v690_arc_supervisor_delta.json"
PINNED = {
    str(PRIOR): "sha256:a1267a22308e88aecd37ae27d3eff057b0f1e720bbe4a7ddfe147e2ff314e65c",
    str(INVENTORY): "sha256:bad0a6ef36f43e8b4d06a1192ed00b1ed8ebc0097a1b1c1f27827c46b1d1e4ad",
    str(REGISTRY): "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}
MODULE = "python/carnot/reporting/arc_supervisor_v690_delta.py"
CLI = "scripts/experiments/experiment_7962_v690_arc_supervisor_delta.py"
TEST = "tests/python/test_arc_supervisor_delta_7962.py"
ADDED = [MODULE, CLI, previous.MODULE]
INCLUDE = ",".join("*/" + p for p in ADDED)
CONSUMERS = [previous.TEST, *previous.CONSUMERS]
EXPERIMENT_ID = 7962
PRIOR_ID = 7949
MILESTONE = "2026.09.690"
RUN_DATE = "20261001"
COVERAGE_TESTS = [previous.TEST]


def inputs(scope: Any = None) -> dict[str, Any]:
    """Check exact bytes and scalar gates before the inventory can supply evidence."""
    scope = scope or sys.modules[__name__]
    checks = [
        previous.operand(
            p,
            "sha256",
            scope.PINNED[str(p)],
            sha256_file(p) if p.is_file() else "missing",
            sha256_file(p) if p.is_file() else None,
        )
        for p in (scope.PRIOR, scope.INVENTORY, scope.REGISTRY)
    ]
    documents = [
        json.loads(p.read_text()) if checks[i]["expected"] == checks[i]["observed"] else {}
        for i, p in enumerate((scope.PRIOR, scope.INVENTORY))
    ]
    prior, inventory = documents
    for key, expected in (
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("arc_evidence_ready_score", 1),
        ("receipt_inventory", inventory.get("receipt_inventory")),
        ("seen_receipt_hashes", inventory.get("seen_receipt_hashes")),
    ):
        checks.append(
            previous.operand(
                scope.PRIOR, key, expected, prior.get(key), scope.PINNED[str(scope.PRIOR)]
            )
        )
    registry = (
        yaml.safe_load(scope.REGISTRY.read_text())
        if checks[2]["expected"] == checks[2]["observed"]
        else {}
    )
    games = registry.get("games", {})
    if isinstance(games, list):
        games = {row["game"]: row for row in games}
    return dict(
        prior=prior,
        inventory=inventory,
        checks=checks,
        failures=[r for r in checks if r["expected"] != r["observed"]],
        registry_precheck={g: r.get("levels_reproduced", 0) for g, r in games.items()},
    )


def commands(private: Path, scope: Any = None) -> list[dict[str, Any]]:
    """Parameterize existing CLI checks, retaining historical dates and failures."""
    scope = scope or sys.modules[__name__]
    specs = previous.commands(private)
    replacements = {
        previous.CLI: scope.CLI,
        previous.TEST: scope.TEST,
        "--include=" + previous.INCLUDE: "--include=" + scope.INCLUDE,
        previous.OUTPUT.name: scope.OUTPUT.name,
    }
    for spec in specs:
        argv = []
        for arg in spec["argv"]:
            if arg.startswith("--include="):
                arg = "--include=" + scope.INCLUDE
            else:
                for old, new in replacements.items():
                    arg = arg.replace(old, new)
            argv.append(scope.RUN_DATE if arg == "20260930" else arg)
        spec["argv"] = argv
        if spec["name"] in {"affected_pytest", "unit_coverage", "spec_coverage"}:
            spec["argv"] = [a for a in spec["argv"] if not a.startswith("tests/python/")]
            spec["argv"].extend(
                [
                    scope.TEST,
                    *(scope.COVERAGE_TESTS if spec["name"] == "unit_coverage" else scope.CONSUMERS),
                ]
            )
        if spec["name"] in {"affected_pytest", "unit_coverage", "full_python_suite"}:
            spec["argv"].append(f"--basetemp={private / spec['name'] / 'pytest'}")
            (private / spec["name"]).mkdir(parents=True, exist_ok=True)
        if spec["name"] in {"ruff_check", "ruff_format", "mypy"}:
            spec["argv"] = [
                a for a in spec["argv"] if a not in {previous.MODULE, scope.CLI, scope.TEST}
            ]
            spec["argv"].extend(scope.ADDED + ([] if spec["name"] == "mypy" else [scope.TEST]))
        if spec["name"] == "cli_negative":
            spec["argv"].extend(["--output", str(private / "negative/report.json")])
    return specs


def replay(value: dict[str, Any]) -> list[str]:
    """Recount new counters so a terminal summary cannot invent firings or help."""
    errors = previous.replay(value)
    expected = {
        "new_firing_count": len(value["new_event_rows"]),
        "new_helped_count": sum(r["resolved_by_levelup"] is True for r in value["new_event_rows"]),
        "live_path_receipts": value["new_event_rows"],
    }
    return errors + [
        key for key, observed in expected.items() if key in value and value[key] != observed
    ]


def execute(output: Path, private: Path, scope: Any = None) -> int:
    """End science at the delta, then seal only fully validated terminal bytes."""
    scope = scope or sys.modules[__name__]
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    previous.progress(started, f"exp{scope.EXPERIMENT_ID}_authenticate")
    checked = scope.inputs()
    durable = output.parent / "raw" / output.stem
    specs = scope.commands(private)
    sources = validation.dependency_hashes(
        scope.ROOT,
        paths=[
            *scope.ADDED,
            scope.TEST,
            *scope.CONSUMERS,
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
        ],
    )
    sources.update(
        {
            str(p): sha256_file(p)
            for p in (
                scope.PRIOR,
                scope.INVENTORY,
                scope.REGISTRY,
                scope.ROOT / "ops/exclusion_manifest.yaml",
                scope.ROOT / "openspec/capabilities/research-reporting/spec.md",
            )
            if p.is_file()
        }
    )
    sources.update(checked.get("additional_source_hashes", {}))
    if getattr(scope, "FREEZE_SOURCES", False):
        for label in scope.ADDED:
            source = scope.ROOT / label
            snapshot = durable / "source_snapshots" / label
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            sources[str(snapshot)] = sha256_file(snapshot)
    producers = [
        p
        for p in sorted((scope.ROOT / "results").glob("experiment_*arc*.json"))
        if p not in {scope.PRIOR, output}
        and p.name.split("_")[1].isdigit()
        and 7831 < int(p.name.split("_")[1]) < scope.EXPERIMENT_ID
    ]
    manifest = durable / "validation_command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=specs,
            dependency_hashes=sources,
            affected_files=[*scope.ADDED, scope.TEST],
            transitive_consumers=scope.CONSUMERS,
            coverage_includes=scope.ADDED,
            coverage_include=scope.INCLUDE,
            candidate_producers=[str(p) for p in producers],
            preconditions=checked["checks"],
            scan_cap_s=120,
            heartbeat_s=30,
            execution_date=scope.RUN_DATE,
            historical_fixture_date="20260929",
            applicable_e2e=getattr(scope, "APPLICABLE_E2E", ["E2E-016", "E2E-017"]),
            terminal_checks=[
                dict(script=s, expected_exit=0, deadline_s=60)
                for s in (
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                )
            ]
            + getattr(scope, "TERMINAL_CHECKS", []),
        ),
    )
    sources[str(manifest)] = sha256_file(manifest)
    previous.progress(started, f"exp{scope.EXPERIMENT_ID}_scope_frozen", len(sources))
    qualification = None
    qualification_ready = True
    qualification_end = 0.0
    if getattr(scope, "QUALIFY_BEFORE_SCAN", False):
        qualification = [previous.run(s, private, durable) for s in specs]
        qualification_ready = all(
            r["passed"] for r in qualification if r["classification"] == "required"
        ) and validation.coverage_complete(private / "coverage.json", includes=scope.ADDED)
        previous.progress(
            started, f"exp{scope.EXPERIMENT_ID}_qualification_terminal", len(qualification)
        )
        qualification_end = time.monotonic() - started
    delta = getattr(scope, "scan", previous.scan)(
        scope.ROOT,
        producers if not checked["failures"] and qualification_ready else [],
        checked,
        private,
        current_date=scope.RUN_DATE,
    )
    delta.update(
        new_firing_count=len(delta["new_event_rows"]),
        new_helped_count=sum(r["resolved_by_levelup"] is True for r in delta["new_event_rows"]),
        live_path_receipts=delta["new_event_rows"],
        seen_receipt_hashes=delta["receipt_inventory"],
    )
    gates = checked["failures"] + delta["scan_failures"]
    science_end = time.monotonic() - started
    previous.progress(
        started, f"exp{scope.EXPERIMENT_ID}_science_terminal", delta["identity_filter_count"]
    )
    primitive_errors = replay(delta)
    if primitive_errors:
        delta.update(previous.reduce(delta["rows"]))
    sources.update(delta["scan_source_hashes"])
    inventory = durable / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    receipts = (
        qualification
        if qualification is not None
        else [
            dict(
                s["reuse_receipt"],
                reused=True,
                classification=s["classification"],
                reuse_source_path=s["reuse_source_path"],
                reuse_source_sha256=s["reuse_source_sha256"],
            )
            if "reuse_receipt" in s
            else previous.run(s, private, durable)
            for s in specs
        ]
    )
    complete = validation.coverage_complete(private / "coverage.json", includes=scope.ADDED)
    counts = (
        json.loads((private / "coverage.json").read_text())["files"]
        if (private / "coverage.json").is_file()
        else {}
    )
    coverage_receipts = []
    for path in sorted(private.glob("coverage.*")):
        archived = durable / "coverage" / (path.name + "-" + sha256_file(path)[7:])
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archived)
        coverage_receipts.append(dict(path=str(archived), sha256=sha256_file(archived)))
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    gates.extend(
        previous.operand(
            Path(r["log_path"]), "exit_code", r["expected_exit"], r["exit_code"], r["log_sha256"]
        )
        for r in failed
    )
    if not complete or primitive_errors:
        gates.append(
            previous.operand(
                output,
                "coverage_and_cold_reduction",
                [True, []],
                [complete, primitive_errors],
                None,
            )
        )
    state = (
        "disqualified"
        if failed or not complete or primitive_errors
        else "blocked"
        if gates
        else "null"
    )
    verdict = (
        "complete_null_no_new_supervisor_outcomes"
        if not delta["identity_filter_count"]
        else "complete_null_no_transferable_refinement"
    )
    if state != "null":
        verdict = "complete_" + state + "_supervisor_delta_prerequisites"
    finished = time.monotonic() - started
    value = dict(
        delta,
        experiment_id=scope.EXPERIMENT_ID,
        experiment=scope.EXPERIMENT_ID,
        task_id=f"exp{scope.EXPERIMENT_ID}-arc-supervisor-delta",
        milestone=scope.MILESTONE,
        run_date=scope.RUN_DATE,
        execution_date=scope.RUN_DATE,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        status="complete",
        schema="arc-supervisor-delta-v" + scope.MILESTONE.rsplit(".", 1)[1],
        title="New live ARC supervisor outcome inventory",
        honest_verdict=verdict,
        verdict_class=state,
        flagged_adversarial=False,
        gate_check_summary=gates,
        arc_evidence_ready_score=int(state == "null"),
        acceptance_gate_results=dict(
            validity=state == "null",
            readiness=int(state == "null"),
            probability_quality=None,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=finished,
        phase_spans=[
            dict(
                phase="supervisor_qualification",
                start_s=0,
                end_s=qualification_end,
                duration_s=qualification_end,
            ),
            dict(
                phase="authenticated_frontier",
                start_s=qualification_end,
                end_s=science_end,
                duration_s=science_end - qualification_end,
            ),
            dict(
                phase="candidate_freeze",
                start_s=science_end,
                end_s=finished,
                duration_s=finished - science_end,
            ),
        ]
        if qualification is not None
        else [
            dict(phase="authenticated_delta", start_s=0, end_s=science_end, duration_s=science_end),
            dict(
                phase="owned_validation",
                start_s=science_end,
                end_s=finished,
                duration_s=finished - science_end,
            ),
        ],
        random_seed=0,
        reproducibility_checksum=canonical_hash(sources),
        source_artifact_hashes=sources,
        preconditions_checked=checked["checks"],
        resolved_imports={
            n: dict(path=str(Path(m.__file__).resolve()), role="exposed_development_dependency")
            for n, m in tuple(sys.modules.items())
            if getattr(m, "__file__", None)
            and (n.startswith("carnot.") or n.startswith("scripts."))
        },
        validation_receipts=receipts,
        validation_command_manifest_path=str(manifest),
        observed_child_commands=[r["command_argv"] for r in receipts if not r.get("reused")],
        coverage_statement_counts={k: v["summary"] for k, v in counts.items()},
        coverage_receipts=coverage_receipts,
        scratch_root_receipt=dict(
            path=str(private),
            outside_checkout=not private.resolve().is_relative_to(scope.ROOT.resolve()),
            allocation="TemporaryDirectory",
            removed_after_exit=True,
        ),
        historical_required_failures=checked["prior"].get("historical_required_failures", []),
        repository_health=dict(
            current_pass_claimed=False,
            affects_required_checks=False,
            cited_upstream_id=scope.PRIOR_ID,
            historical_receipts=checked["prior"].get("repository_health", {}),
            checks=[
                *checked.get("owned_repository_health", []),
                *[
                    r
                    for r in receipts
                    if r["classification"] == "repository_health" and not r.get("reused")
                ],
            ],
        ),
        primary_resolution_receipt=dict(
            path=str(output), receipt_path=str(durable / "primary_resolution_receipt.json")
        ),
        terminal_validation_sidecar_path=str(durable / "terminal_reports.json"),
        verifier_is_oracle=False,
        claim_scope="exposed_development; observational inventory without independent or causal benefit",
        fixture_claim_scope="circular_positive mechanics only",
        methodology="Authenticate the seen frontier and cold-reduce only new live supervisor outcomes.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model="none",
        model_invocation_counts=0,
        trained_head_specs=[],
        current_device_execution_count=0,
        solve_provenance=sorted({r["solve_provenance"] for r in delta["new_event_rows"]}),
        new_level_solves_claimed=0,
        registry_precheck=checked["registry_precheck"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=scope.PRIOR_ID,
                path=str(p),
                sha256=scope.PINNED[str(p)],
                fields_imported=fields,
            )
            for p, fields in (
                (
                    scope.PRIOR,
                    [
                        "receipt_inventory",
                        "seen_receipt_hashes",
                        "source_artifact_hashes",
                        "historical_required_failures",
                        "repository_health",
                    ],
                ),
                (scope.INVENTORY, ["receipt_inventory", "seen_receipt_hashes"]),
            )
        ],
        retire_if_same_verdict=dict(
            prior_id=scope.PRIOR_ID,
            same_verdict=state == "null",
            action="retain_no_change_terminal_inventory",
        ),
        oracle_distinct_corrigendum=dict(
            date="20260928",
            preserved=True,
            gap_oracle_distinct="open",
            diffusion_gemma_gate="STILL-PENDING",
            reference="ops/known-issues.md CORRIGENDUM 2026-09-28",
        ),
    )
    value.update(scope.artifact_fields(value, checked) if hasattr(scope, "artifact_fields") else {})
    value["field_principles"] = {
        k: "Bind current producer custody; inherited observations do not establish benefit."
        for k in value
    }
    value["field_principles"].update(
        receipt_inventory="Content identity prevents a retry from inventing work.",
        sample_size_budget="Redirects are units; games are independent groups.",
        arc_evidence_ready_score="An authenticated empty inventory is valid evidence.",
        acceptance_gate_results="Validity and readiness are distinct from scientific benefit.",
        solve_provenance="Only authenticated input events have live discovery provenance; aggregation claims no solves.",
        new_firing_count="Count distinct qualified redirects; a changed timestamp creates no event.",
        new_helped_count="Count observed level-ups after redirects without assigning causal credit.",
        live_path_receipts="Require authenticated E3AgentPolicy and make_carnot_agent outcome rows.",
        scratch_root_receipt="Private mutable scratch cannot overwrite historical research results.",
        coverage_receipts="Keep immutable measured coverage after owned children exit.",
        started_at="UTC records producer start independently of the declared run date.",
        finished_at="UTC records validation completion independently of monotonic work duration.",
    )
    last: dict[str, Any] = {}

    def validator(candidate: Path) -> dict[str, Any]:
        last.clear()
        last.update(getattr(scope, "terminal", previous.terminal)(candidate, private, durable))
        last["replay_errors"] = replay(json.loads(candidate.read_text()))
        last["passed"] = last["passed"] and not last["replay_errors"]
        return last

    try:
        published = publish_primary(output, value, validator)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            arc_evidence_ready_score=0,
            flagged_adversarial=any(
                row["name"] == "terminal_adversarial" and not row["passed"]
                for row in last["reports"]
            ),
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].append(dict(check="terminal_validation", report=last.copy()))
        published = publish_primary(output, value, validator)
    atomic_json(durable / "terminal_reports.json", dict(published, report=last))
    atomic_json(durable / "newer_sidecar.json", dict(role="validator_sidecar"))
    selected = reader_receipt(
        value["task_id"], output.parent, field="experiment_id", expected=scope.EXPERIMENT_ID
    )
    assert (
        selected["passed"]
        and selected["gate_path"] == str(output)
        and selected["gate_sha256"] == sha256_file(output)
    )
    atomic_json(durable / "primary_resolution_receipt.json", selected)
    previous.progress(
        started, f"exp{scope.EXPERIMENT_ID}_published", delta["identity_filter_count"]
    )
    return int(value["verdict_class"] != "null")
