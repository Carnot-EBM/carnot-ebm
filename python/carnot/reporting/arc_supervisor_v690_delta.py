"""Advance a qualified receipt frontier without repeating solves. REQ-REPORT-7962.

Empty evidence is useful when its custody is checked. Reused validation keeps
successful execution separate from scientific benefit and historical health.
"""

from __future__ import annotations

import json
from pathlib import Path
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


def inputs() -> dict[str, Any]:
    """Check exact bytes and scalar gates before the inventory can supply evidence."""
    checks = [
        previous.operand(
            p,
            "sha256",
            PINNED[str(p)],
            sha256_file(p) if p.is_file() else "missing",
            sha256_file(p) if p.is_file() else None,
        )
        for p in (PRIOR, INVENTORY, REGISTRY)
    ]
    documents = [
        json.loads(p.read_text()) if checks[i]["expected"] == checks[i]["observed"] else {}
        for i, p in enumerate((PRIOR, INVENTORY))
    ]
    prior, inventory = documents
    for key, expected in (
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("arc_evidence_ready_score", 1),
        ("receipt_inventory", inventory.get("receipt_inventory")),
        ("seen_receipt_hashes", inventory.get("seen_receipt_hashes")),
    ):
        checks.append(previous.operand(PRIOR, key, expected, prior.get(key), PINNED[str(PRIOR)]))
    registry = (
        yaml.safe_load(REGISTRY.read_text())
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


def commands(private: Path) -> list[dict[str, Any]]:
    """Parameterize existing CLI checks, retaining historical dates and failures."""
    specs = previous.commands(private)
    replacements = {
        previous.CLI: CLI,
        previous.TEST: TEST,
        "--include=" + previous.INCLUDE: "--include=" + INCLUDE,
        previous.OUTPUT.name: OUTPUT.name,
    }
    for spec in specs:
        argv = []
        for arg in spec["argv"]:
            if arg.startswith("--include="):
                arg = "--include=" + INCLUDE
            else:
                for old, new in replacements.items():
                    arg = arg.replace(old, new)
            argv.append("20261001" if arg == "20260930" else arg)
        spec["argv"] = argv
        if spec["name"] == "affected_pytest":
            spec["argv"].append(previous.TEST)
        if spec["name"] == "unit_coverage":
            spec["argv"].append(previous.TEST)
        if spec["name"] in {"ruff_check", "ruff_format", "mypy"}:
            spec["argv"] = [a for a in spec["argv"] if a not in {previous.MODULE, CLI, TEST}]
            spec["argv"].extend(ADDED + ([] if spec["name"] == "mypy" else [TEST]))
        if spec["name"] == "cli_negative":
            spec["argv"].extend(["--output", str(private / "negative/report.json")])
    return specs


def execute(output: Path, private: Path) -> int:
    """End science at the delta, then seal only fully validated terminal bytes."""
    started = time.monotonic()
    previous.progress(started, "exp7962_authenticate")
    checked = inputs()
    durable = output.parent / "raw" / output.stem
    specs = commands(private)
    sources = validation.dependency_hashes(
        ROOT,
        paths=[
            *ADDED,
            TEST,
            *CONSUMERS,
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
        ],
    )
    sources.update(
        {
            str(p): sha256_file(p)
            for p in (
                PRIOR,
                INVENTORY,
                REGISTRY,
                ROOT / "ops/exclusion_manifest.yaml",
                ROOT / "openspec/capabilities/research-reporting/spec.md",
            )
            if p.is_file()
        }
    )
    producers = [
        p
        for p in sorted((ROOT / "results").glob("experiment_*arc*.json"))
        if p not in {PRIOR, output}
        and p.name.split("_")[1].isdigit()
        and 7831 < int(p.name.split("_")[1]) < 7962
    ]
    manifest = durable / "validation_command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=specs,
            dependency_hashes=sources,
            affected_files=[*ADDED, TEST],
            transitive_consumers=CONSUMERS,
            coverage_includes=ADDED,
            coverage_include=INCLUDE,
            candidate_producers=[str(p) for p in producers],
            preconditions=checked["checks"],
            scan_cap_s=120,
            heartbeat_s=30,
            execution_date="20261001",
            historical_fixture_date="20260929",
            applicable_e2e=["E2E-016", "E2E-017"],
            terminal_checks=[
                dict(script=s, expected_exit=0, deadline_s=60)
                for s in (
                    "scripts/adversarial_verify.py",
                    "scripts/verdict_row_consistency_lint.py",
                )
            ],
        ),
    )
    sources[str(manifest)] = sha256_file(manifest)
    previous.progress(started, "exp7962_scope_frozen", len(sources))
    delta = previous.scan(
        ROOT,
        producers if not checked["failures"] else [],
        checked,
        private,
        current_date="20261001",
    )
    gates = checked["failures"] + delta["scan_failures"]
    science_end = time.monotonic() - started
    previous.progress(started, "exp7962_science_terminal", delta["identity_filter_count"])
    primitive_errors = previous.replay(delta)
    sources.update(delta["scan_source_hashes"])
    inventory = durable / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    receipts = [previous.run(s, private, durable) for s in specs]
    complete = validation.coverage_complete(private / "coverage.json", includes=ADDED)
    counts = (
        json.loads((private / "coverage.json").read_text())["files"]
        if (private / "coverage.json").is_file()
        else {}
    )
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
        experiment_id=7962,
        experiment=7962,
        task_id="exp7962-arc-supervisor-delta",
        milestone="2026.09.690",
        run_date="20261001",
        status="complete",
        schema="arc-supervisor-delta-v690",
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
        observed_child_commands=[r["command_argv"] for r in receipts],
        coverage_statement_counts={k: v["summary"] for k, v in counts.items()},
        historical_required_failures=checked["prior"].get("historical_required_failures", []),
        repository_health=dict(
            current_pass_claimed=False,
            affects_required_checks=False,
            cited_upstream_id=7949,
            historical_receipts=checked["prior"].get("repository_health", {}),
            checks=[r for r in receipts if r["classification"] == "repository_health"],
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
            dict(experiment_id=7949, path=str(p), sha256=PINNED[str(p)], fields_imported=fields)
            for p, fields in (
                (
                    PRIOR,
                    [
                        "receipt_inventory",
                        "seen_receipt_hashes",
                        "source_artifact_hashes",
                        "historical_required_failures",
                        "repository_health",
                    ],
                ),
                (INVENTORY, ["receipt_inventory", "seen_receipt_hashes"]),
            )
        ],
        retire_if_same_verdict=dict(
            prior_id=7949,
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
    )
    last: dict[str, Any] = {}

    def validator(candidate: Path) -> dict[str, Any]:
        last.clear()
        last.update(previous.terminal(candidate, private, durable))
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
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].append(dict(check="terminal_validation", report=last.copy()))
        published = publish_primary(output, value, validator)
    atomic_json(durable / "terminal_reports.json", dict(published, report=last))
    atomic_json(durable / "newer_sidecar.json", dict(role="validator_sidecar"))
    selected = reader_receipt(value["task_id"], output.parent, field="experiment_id", expected=7962)
    assert (
        selected["passed"]
        and selected["gate_path"] == str(output)
        and selected["gate_sha256"] == sha256_file(output)
    )
    atomic_json(durable / "primary_resolution_receipt.json", selected)
    previous.progress(started, "exp7962_published", delta["identity_filter_count"])
    return int(value["verdict_class"] != "null")
