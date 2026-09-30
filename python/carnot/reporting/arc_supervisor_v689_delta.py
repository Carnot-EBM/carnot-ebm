"""Keep a new-outcome inventory terminal. Spec: REQ-REPORT-7949.

The earlier experiment qualified recovery. This wrapper only checks new bytes,
so an empty ledger ends science without another recovery run or model call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from carnot.reporting import arc_supervisor_v688_receipts as reader
from carnot.reporting import v686_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "results/experiment_7936_v688_arc_supervisor_refinement.json"
INVENTORY = (
    ROOT / "results/raw/experiment_7936_v688_arc_supervisor_refinement/receipt_inventory.json"
)
REGISTRY = ROOT / "ops/arc_solve_registry.yaml"
OUTPUT = ROOT / "results/experiment_7949_v689_arc_supervisor_delta.json"
PINNED = {
    str(PRIOR): "sha256:563b9b896430acc06cee8675786f98ff71eb151b76f5786d1f49fa5c39e04756",
    str(INVENTORY): "sha256:a05746144eee9c24fcfa356e7130d555975131f64305dff0899f7b6bd540a643",
    str(REGISTRY): "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}
MODULE = "python/carnot/reporting/arc_supervisor_v689_delta.py"
CLI = "scripts/experiments/experiment_7949_v689_arc_supervisor_delta.py"
TEST = "tests/python/test_arc_supervisor_delta_7949.py"
ADDED = [MODULE, CLI]
INCLUDE = ",".join("*/" + p for p in ADDED)
CONSUMERS = [
    "tests/python/test_arc_supervisor_refinement_7936.py",
    "tests/python/test_arc_supervisor_delta_7924.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_arc_supervisor_delta_7874.py",
]
SCAN_CAP_S = 120


def progress(started: float, phase: str, units: int = 0) -> None:
    """Flushed elapsed time distinguishes work from a silent stalled process."""
    print(
        f"[exp7949] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def operand(
    path: Path, field: str, expected: Any, observed: Any, digest: str | None
) -> dict[str, Any]:
    """Keep failed thresholds distinct from missing external bytes."""
    return dict(
        upstream_id=path.stem,
        path=str(path),
        sha256=digest,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
    )


def inputs() -> dict[str, Any]:
    """Authenticate the qualified inventory before exposing any event to reduction."""
    checks = [
        operand(
            p,
            "sha256",
            PINNED[str(p)],
            sha256_file(p) if p.is_file() else "missing",
            sha256_file(p) if p.is_file() else None,
        )
        for p in (PRIOR, INVENTORY, REGISTRY)
    ]
    good = all(r["expected"] == r["observed"] for r in checks)
    prior = json.loads(PRIOR.read_text()) if checks[0]["expected"] == checks[0]["observed"] else {}
    inventory = (
        json.loads(INVENTORY.read_text()) if checks[1]["expected"] == checks[1]["observed"] else {}
    )
    registry = (
        yaml.safe_load(REGISTRY.read_text())
        if checks[2]["expected"] == checks[2]["observed"]
        else {}
    )
    if good:
        checks.extend(
            operand(PRIOR, k, v, prior.get(k), PINNED[str(PRIOR)])
            for k, v in (
                ("verdict_class", "null"),
                ("flagged_adversarial", False),
                ("arc_evidence_ready_score", 1),
                ("receipt_inventory", inventory.get("receipt_inventory")),
                ("seen_receipt_hashes", inventory.get("seen_receipt_hashes")),
            )
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


def reduce(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse the counts but retain priorities until separate follow-up evidence exists."""
    value = reader.reduce_rows(rows)
    for decision in value["refinement_decisions"]:
        decision["decision"] = "unchanged"
    value["new_event_rows"] = value["recovered_rows"]
    return value


def replay(value: dict[str, Any]) -> list[str]:
    """Cold reduction checks all task-owned derived claims from primitive rows."""
    return [key for key, observed in reduce(value["rows"]).items() if value.get(key) != observed]


def scan(
    root: Path, producers: list[Path], checked: dict[str, Any], private: Path
) -> dict[str, Any]:
    """Read candidate originals once, then use the existing reader on frozen copies."""
    started = time.monotonic()
    snapshot = private / "scan"
    sources: dict[str, str] = {}
    failures = []
    snapshots = []
    seen = checked["inventory"].get("receipt_inventory", {})
    frozen = checked["prior"].get("source_artifact_hashes", {})
    copied: dict[str, str] = {}
    episodes_by_copy: dict[str, list[dict[str, Any]]] = {}
    producer_by_copy: dict[str, tuple[str, str]] = {}
    for index, producer in enumerate(producers):
        if time.monotonic() - started >= SCAN_CAP_S:
            failures.append(operand(producer, "scan_deadline_s", SCAN_CAP_S, "expired", None))
            break
        try:
            data = producer.read_bytes()
            document = json.loads(data)
            if not isinstance(document, dict) or not isinstance(
                document.get("source_artifact_hashes"), dict
            ):
                raise ValueError("unsupported_producer")
        except (OSError, ValueError) as error:
            failures.append(operand(producer, "readable_json_producer", True, str(error), None))
            continue
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        sources[str(producer)] = digest
        if frozen.get(str(producer), frozen.get(str(producer.relative_to(root)))) == digest:
            continue
        hashes = {}
        for label, declared in document.get("source_artifact_hashes", {}).items():
            raw = Path(label)
            raw = (raw if raw.is_absolute() else root / raw).resolve()
            if raw.suffix != ".json" or not raw.is_relative_to((root / "results/raw").resolve()):
                continue
            expected = declared.get("sha256") if isinstance(declared, dict) else declared
            if "raw:" + str(expected) in seen or str(raw) in copied:
                continue
            content = raw.read_bytes() if raw.is_file() else b""
            observed = (
                "sha256:" + hashlib.sha256(content).hexdigest() if raw.is_file() else "missing"
            )
            sources[str(raw)] = observed
            if expected != observed:
                failures.append(operand(raw, "sha256", expected, observed, observed))
                continue
            copy = snapshot / "results/raw" / f"{len(copied)}.json"
            try:
                raw_document = json.loads(content)
                episodes = reader.extract_rows(raw_document)
            except ValueError as error:
                failures.append(operand(raw, "readable_json_receipt", True, str(error), observed))
                continue
            atomic_json(copy, {"rows": episodes})
            copied[str(raw)] = str(copy)
            episodes_by_copy[str(copy)] = episodes
            hashes[str(copy)] = sha256_file(copy)
        copy_producer = snapshot / f"producer-{index}.json"
        atomic_json(copy_producer, dict(document, source_artifact_hashes=hashes))
        snapshots.append(copy_producer)
        producer_by_copy[str(copy_producer)] = (str(producer), digest)
        progress(started, "scan", index + 1)
    inspected = reader.inspect(snapshot, snapshots, seen, "20260930", "20260930", {})
    original_by_copy = {v: k for k, v in copied.items()}
    for row in inspected["rows"]:
        row["producer_path"], row["producer_sha256"] = producer_by_copy[row["producer_path"]]
        copy = row.get("source_path")
        if copy in original_by_copy:
            row["source_path"] = original_by_copy[copy]
            row["source_sha256"] = sources[row["source_path"]]
            row["expected_sha256"] = row["source_sha256"]
        if row["status"] in {"completed", "censored", "unknown"}:
            episode = next(
                e for e in episodes_by_copy[copy] if canonical_hash(e) == row["content_sha256"]
            )
            provenance = episode.get("live_agent_provenance", {})
            if provenance != dict(
                policy_class="E3AgentPolicy",
                agent_factory="make_carnot_agent",
                execution_mode="live",
            ) or any(
                key not in redirect
                for redirect in episode["trajectory_supervisor"]["redirects"]
                for key in ("resolved_by_levelup", "actions_to_levelup")
            ):
                row.update(status="excluded", reason="unqualified_live_agent_provenance")
    value = reduce(inspected["rows"])
    value.update(
        rows=inspected["rows"],
        receipt_inventory=inspected["receipt_inventory"],
        seen_receipt_hashes=seen,
        scan_duration_s=time.monotonic() - started,
        scan_failures=failures,
        scan_source_hashes=sources,
    )
    return value


def commands(private: Path) -> list[dict[str, Any]]:
    """Freeze bounded checks and separate each private publication scenario."""
    py, cov, pytest, ruff, mypy = (
        str(ROOT / ".venv/bin" / n) for n in ("python", "coverage", "pytest", "ruff", "mypy")
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    include = "--include=" + INCLUDE
    fixture = private / "success"
    producer = fixture / "producer.json"
    atomic_json(
        producer, dict(run_date="20260930", verdict_class="null", source_artifact_hashes={})
    )
    missing = private / "blocked/producer.json"
    atomic_json(
        missing,
        dict(
            run_date="20260930",
            verdict_class="null",
            source_artifact_hashes={
                str(private / "blocked/results/raw/missing.json"): "sha256:" + "0" * 64
            },
        ),
    )
    negative = private / "negative/candidate.json"
    atomic_json(negative, dict(reduce([]), rows=[], identity_filter_count=1))
    success = [
        CLI,
        "--date",
        "20260930",
        "--reduce-ledger",
        str(fixture),
        "--producer",
        str(producer),
        "--output",
        str(fixture / OUTPUT.name),
    ]
    routes = {
        "success": (success, 0, None),
        "negative": ([CLI, "--cold-replay", str(negative)], 1, "identity_filter_count"),
        "blocked": (
            [
                CLI,
                "--reduce-ledger",
                str(private / "blocked"),
                "--producer",
                str(missing),
                "--output",
                str(private / "blocked" / OUTPUT.name),
            ],
            1,
            "complete_blocked",
        ),
        "replay": (
            [
                CLI,
                "--cold-replay",
                str(fixture / OUTPUT.name),
                "--output",
                str(private / "replay" / OUTPUT.name),
            ],
            0,
            None,
        ),
        "terminal": (
            [
                CLI,
                "--terminal-recheck",
                str(fixture / OUTPUT.name),
                "--output",
                str(private / "terminal" / OUTPUT.name),
            ],
            0,
            None,
        ),
        "arguments": ([CLI, "--reduce-ledger", str(private / "arguments")], 2, "requires"),
    }
    specs = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        deadline: int = 120,
        classification: str = "required",
    ) -> None:
        specs.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                expected_text=reason,
                deadline_s=deadline,
                classification=classification,
            )
        )

    add("affected_pytest", [pytest, *common, TEST, *CONSUMERS])
    add(
        "unit_coverage",
        [
            cov,
            "run",
            f"--data-file={private / 'coverage.unit'}",
            include,
            "-m",
            "pytest",
            *common,
            TEST,
        ],
    )
    for name, (argv, expected, reason) in routes.items():
        add(
            "cli_" + name,
            [cov, "run", f"--data-file={private / ('coverage.' + name)}", include, *argv],
            expected,
            reason,
        )
    historical = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    historical_fixture = private / "e2e016" / "fixture.json"
    historical_fixture.parent.mkdir(parents=True, exist_ok=True)
    add(
        "e2e_016_fixture",
        [py, historical, "--date", "20260929", "--fixture-e2e", str(historical_fixture)],
    )
    add(
        "e2e_016_replay",
        [py, historical, "--date", "20260929", "--cold-replay", str(historical_fixture)],
    )
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            "--keep",
            f"--data-file={private / 'coverage.combined'}",
            *[str(private / ("coverage." + n)) for n in ["unit", *routes]],
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={private / 'coverage.combined'}",
            include,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            f"--data-file={private / 'coverage.combined'}",
            include,
            "-o",
            str(private / "coverage.json"),
        ],
    )
    add("ruff_check", [ruff, "check", *ADDED, TEST])
    add("ruff_format", [ruff, "format", "--check", *ADDED, TEST])
    add("mypy", [mypy, "--strict", *ADDED])
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST, *CONSUMERS])
    add(
        "full_python_suite",
        [pytest, "tests/python", "-q"],
        deadline=120,
        classification="repository_health",
    )
    return specs


def run(spec: dict[str, Any], private: Path, durable: Path) -> dict[str, Any]:
    """Expected failures must match their reason as well as the real exit."""
    row = validation.run_check(ROOT, spec, private, durable / "logs")
    row["passed"] = row["passed"] and (
        not spec.get("expected_text") or spec["expected_text"] in row["output_tail"]
    )
    return row


def terminal(candidate: Path, private: Path, durable: Path) -> dict[str, Any]:
    """Run both validators on the same bytes and cold-reduce all derived claims."""
    reports = [
        run(
            dict(
                name=name,
                argv=[str(ROOT / ".venv/bin/python"), script, flag, str(candidate)],
                deadline_s=60,
                expected_exit=0,
                expected_text=None,
                classification="required",
            ),
            private,
            durable,
        )
        for name, script, flag in (
            ("terminal_adversarial", "scripts/adversarial_verify.py", "--json"),
            ("terminal_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        )
    ]
    errors = replay(json.loads(candidate.read_text()))
    return dict(
        passed=all(r["passed"] for r in reports) and not errors,
        reports=reports,
        replay_errors=errors,
    )


def execute(output: Path, private: Path) -> int:
    """Stop science at an empty inventory, validate once, and seal checked bytes."""
    started = time.monotonic()
    progress(started, "authenticate")
    checked = inputs()
    durable = output.parent / "raw" / output.stem
    specs = commands(private)
    producers = [
        p
        for p in sorted((ROOT / "results").glob("experiment_*arc*.json"))
        if p != PRIOR and p.name.split("_")[1].isdigit() and 7831 < int(p.name.split("_")[1]) < 7949
    ]
    sources = validation.dependency_hashes(
        ROOT,
        paths=[
            *ADDED,
            TEST,
            *CONSUMERS,
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
            "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
        ],
    )
    for p in (
        PRIOR,
        INVENTORY,
        REGISTRY,
        ROOT / "ops/exclusion_manifest.yaml",
        ROOT / "pyproject.toml",
        ROOT / "openspec/capabilities/research-reporting/spec.md",
    ):
        if p.is_file():
            sources[str(p)] = sha256_file(p)
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
            preconditions=checked["checks"],
            candidate_producers=[str(p) for p in producers],
            scan_cap_s=SCAN_CAP_S,
            heartbeat_s=30,
            applicable_e2e=["E2E-016", "E2E-017"],
            execution_date="20260930",
            historical_fixture_date="20260929",
            environment=dict(PYTHONPATH="python:.", JAX_PLATFORMS="cpu", CARNOT_FORCE_LIVE="1"),
            terminal_checks=[
                dict(
                    script="scripts/adversarial_verify.py",
                    flag="--json",
                    expected_exit=0,
                    deadline_s=60,
                ),
                dict(
                    script="scripts/verdict_row_consistency_lint.py",
                    flag="--strict",
                    expected_exit=0,
                    deadline_s=60,
                ),
            ],
        ),
    )
    sources[str(manifest)] = sha256_file(manifest)
    progress(started, "scope_frozen", len(sources))
    delta = scan(ROOT, producers if not checked["failures"] else [], checked, private)
    gates = [*checked["failures"], *delta["scan_failures"]]
    primitive_errors = replay(delta)
    if primitive_errors:
        gates.append(operand(output, "cold_reduction", [], primitive_errors, None))
        delta.update(reduce(delta["rows"]))
    sources.update(delta["scan_source_hashes"])
    inventory = durable / "receipt_inventory.json"
    atomic_json(inventory, delta)
    sources[str(inventory)] = sha256_file(inventory)
    reduced = time.monotonic() - started
    progress(started, "science_terminal_no_change", delta["identity_filter_count"])
    receipts = [run(spec, private, durable) for spec in specs]
    coverage = private / "coverage.json"
    complete = validation.coverage_complete(coverage, includes=ADDED)
    counts = json.loads(coverage.read_text())["files"] if coverage.is_file() else {}
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    gates.extend(
        operand(
            Path(r["log_path"]), "exit_code", r["expected_exit"], r["exit_code"], r["log_sha256"]
        )
        for r in failed
    )
    if not complete:
        gates.append(
            operand(
                coverage,
                "statement_coverage",
                "nonempty_100_percent",
                counts,
                sha256_file(coverage) if coverage.is_file() else None,
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
        experiment_id=7949,
        experiment=7949,
        task_id="exp7949-arc-supervisor-delta",
        milestone="2026.09.689",
        run_date="20260930",
        schema="arc-supervisor-delta-v689",
        status="complete",
        title="New live supervisor outcome inventory",
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
            dict(phase="authenticate_inventory", start_s=0, end_s=reduced, duration_s=reduced),
            dict(
                phase="owned_validation",
                start_s=reduced,
                end_s=finished,
                duration_s=finished - reduced,
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
            affects_required_checks=False,
            checks=[r for r in receipts if r["classification"] == "repository_health"],
        ),
        primary_resolution_receipt=dict(
            path=str(output), receipt_path=str(durable / "primary_resolution_receipt.json")
        ),
        terminal_validation_sidecar_path=str(durable / "terminal_reports.json"),
        verifier_is_oracle=False,
        claim_scope="exposed_development; observational inventory without independent or causal benefit",
        fixture_claim_scope="circular_positive mechanics only",
        methodology="Authenticate qualified content identities and cold-reduce only new live supervisor outcome rows.",
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
        retire_if_same_verdict=dict(
            prior_id=7936,
            same_verdict=state == "null",
            action="retain_no_change_terminal_inventory",
        ),
    )
    value["field_principles"] = {
        k: "Bind executing producer custody and keep observational inventory distinct from benefit."
        for k in value
    }
    value["field_principles"].update(
        receipt_inventory="Content hashes prevent retries from inventing new events.",
        sample_size_budget="Redirects are units; games are the independent grouping.",
        arc_evidence_ready_score="A qualified empty inventory is valid evidence with readiness one.",
        solve_provenance="Only authenticated input events carry live discovery provenance; aggregation claims zero solves.",
        acceptance_gate_results="Validity and readiness are separate from probability, calibration, decision, retention and efficiency.",
    )
    last: dict[str, Any] = {}

    def validator(candidate: Path) -> dict[str, Any]:
        last.update(terminal(candidate, private, durable))
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
            flagged_adversarial=False,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        value["gate_check_summary"].append(dict(check="terminal_validation", report=last.copy()))
        published = publish_primary(output, value, validator)
    atomic_json(durable / "terminal_reports.json", dict(published, report=last))
    atomic_json(durable / "newer_sidecar.json", dict(role="validator_sidecar"))
    selected = reader_receipt(value["task_id"], output.parent, field="experiment_id", expected=7949)
    assert (
        selected["passed"]
        and selected["gate_path"] == str(output)
        and selected["gate_sha256"] == sha256_file(output)
    )
    atomic_json(durable / "primary_resolution_receipt.json", selected)
    progress(started, "published", delta["identity_filter_count"])
    return int(value["verdict_class"] != "null")
