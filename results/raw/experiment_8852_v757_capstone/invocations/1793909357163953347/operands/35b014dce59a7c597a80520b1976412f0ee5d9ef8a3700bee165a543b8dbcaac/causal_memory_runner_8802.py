"""REQ-REPORT-8802: qualify causal execution without claiming downstream benefit."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from coverage import Coverage

from carnot.memory import causal_memory_8802 as m
from carnot.reporting import v754_repaired_memory as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v753_qualification_runner import coverage_summary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8802_v754_causal_memory_prototype"
CLI = f"scripts/experiments/{NAME}.py"
UPSTREAM = "results/experiment_8801_v754_repaired_memory_qualification.json"
UPSTREAM_PIN = "sha256:c55ed8c40986921597e6f9a1eb760f0844a32585d40ca3ef95b8b7316aa3c1e6"
OWNED = [
    "python/carnot/memory/causal_memory_8802.py",
    "python/carnot/reporting/causal_memory_runner_8802.py",
    CLI,
]
TESTS = [
    "tests/python/test_causal_memory_8802.py",
    "tests/python/test_causal_memory_runner_8802.py",
]


def reference(path: Path) -> Json:
    """Bind replay to exact source and receipt bytes rather than their filenames."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def upstream_gate(path: Path) -> Json:
    """Name absent fields, rejected producers and observed zero separately."""
    value = json.loads(path.read_text()) if path.is_file() else {}
    observed = (
        "missing_file"
        if not path.is_file()
        else value.get("memory_protocol_ready_score", "missing_field")
    )
    if value.get("verdict_class") in {"blocked", "disqualified"}:
        observed = "rejected_producer"
    return dict(
        check="qualified_memory_protocol",
        upstream="exp8801-repaired-memory-qualification",
        artifact_field="memory_protocol_ready_score",
        expected=1,
        observed=observed,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        passed=observed == 1 and value.get("experiment_id") == 8801,
    )


def preflight(raw: Path) -> Json:
    """Charge custody and imports once before projecting a measured cheap work unit."""
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "scratch.json", {"private": True})
    gate = upstream_gate(ROOT / UPSTREAM)
    sources = []
    protocol: Json = {}
    if gate["passed"]:
        upstream = json.loads((ROOT / UPSTREAM).read_text())
        sources = [
            reference(ROOT / UPSTREAM),
            *upstream["code_config_hashes"],
            upstream["branch_protocol_refs"]["memory"],
        ]
        protocol_path = Path(sources[-1]["path"])
        if protocol_path.is_file():
            protocol = json.loads(protocol_path.read_text())
    current = authority.authority(ROOT)
    atomic_json(raw / "current_authority.json", current)
    setup = time.monotonic() - start
    probe_start = time.monotonic()
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    source_hashes = all(
        Path(s["path"]).is_file() and sha256_file(Path(s["path"])) == s["sha256"] for s in sources
    )
    probe = 0.0
    if gate["passed"] and source_hashes and current["ready"] and memory > 1024**3:
        m.progress("numerical_probe_before")
        params = m.initial_parameters(11)
        payload = m.world("resource-probe", "support")
        observation = m.observe(payload, [])
        m.update(params, observation["counts"], "support")
        probe = time.monotonic() - probe_start
        m.progress("numerical_probe_after", 1, 0)
    checks = dict(
        upstream=gate["passed"],
        pinned_upstream=gate["sha256"] == UPSTREAM_PIN,
        source_hashes=source_hashes,
        dependency_imports=True,
        current_authority=current["ready"],
        private_scratch=json.loads((raw / "scratch.json").read_text())["private"],
        available_memory=memory > 1024**3,
        disk=shutil.disk_usage(raw).free > 128 * 1024**2,
        projected_time=setup + probe * 250 + 900 < 4800,
    )
    return dict(
        checked=checks,
        passed=all(checks.values()),
        gate=gate,
        sources=sources,
        protocol=protocol,
        import_and_custody_s=setup,
        measured_cheap_unit_s=probe,
        projected_work_s=probe * 250,
        validation_reserve_s=900,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(raw).free,
    )


def run(raw: Path, date: str) -> Json:
    """Retain a frozen manifest and raw measured work before any readiness publication."""
    started = time.monotonic()
    m.progress("preconditions_before")
    pre = preflight(raw)
    m.progress("preconditions_after")
    manifest = m.freeze_manifest()
    atomic_json(raw / "frozen_manifest.json", manifest)
    if pre["passed"]:
        m.progress("prototype_before", 0, 180)
        work = m.prototype(raw / "checkpoints")
        m.progress("prototype_after", len(work["rows"]), 0)
    else:
        work = dict(
            rows=[],
            template_addition_rows=[],
            gradient_update_rows=[],
            restart_rows=[],
            leakage_controls=[],
            calibrated_head_parameters=[],
            numerical_training_calls=0,
        )
    atomic_json(raw / "measurement.json", work)
    rows = work["rows"]
    gates = [pre["gate"]] + [
        dict(
            check=k,
            upstream="exp8802-owned-preconditions",
            artifact_field=k,
            expected=True,
            observed=v,
            passed=v,
            path=str(raw / "current_authority.json"),
            sha256=sha256_file(raw / "current_authority.json"),
        )
        for k, v in pre["checked"].items()
    ]
    value = dict(
        experiment_id=8802,
        task_id="exp8802-causal-memory-prototype",
        milestone="2026.11.754",
        run_date=date,
        honest_verdict="complete_disqualified_pending_owned_validation"
        if pre["passed"]
        else "complete_blocked_preconditions",
        verdict_class="disqualified" if pre["passed"] else "blocked",
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        verifier_is_oracle=True,
        duration_s=time.monotonic() - started,
        random_seed=8802,
        reproducibility_checksum=canonical_hash(rows),
        source_artifact_hashes=pre["sources"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        raw_shard_hashes=[
            reference(raw / "measurement.json"),
            reference(raw / "frozen_manifest.json"),
            reference(raw / "current_authority.json"),
        ]
        + [reference(p) for p in sorted((raw / "checkpoints").glob("*.json"))],
        preconditions_checked=pre,
        phase_spans=[dict(phase="prototype_and_custody", duration_s=time.monotonic() - started)],
        intended_count=180,
        completed_count=len(rows),
        censored_count=180 - len(rows),
        failed_count=0,
        excluded_count=0,
        independent_count=0,
        n_samples=len(rows),
        gate_check_summary=gates,
        acceptance_gates=dict(
            preconditions=pre["passed"],
            controls=bool(rows) and all(c["passed"] for c in work["leakage_controls"]),
            restarts=bool(rows) and all(c["passed"] for c in work["restart_rows"]),
            numerical_gradients=bool(rows)
            and all(c["gradient_norm"] > 0 for c in work["gradient_update_rows"]),
            owned_validation=False,
        ),
        required_checks_passed=False,
        owned_statement_coverage=dict(complete=False),
        validation_receipts=[],
        exposure_scope="constructed_exposed_development_prototype_only",
        independent_generalization_score=0,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8801,
                fields_imported=[
                    "memory_protocol_ready_score",
                    "branch_protocol_refs.memory",
                    "code_config_hashes",
                    "full_task_objects",
                ],
                sha256=pre["gate"]["sha256"],
            )
        ],
        memory_execution_ready_score=0,
        frozen_memory_protocol=dict(qualified=pre["protocol"], manifest=manifest),
        feature_schema=list(m.COUNTS) + list(m.LABELS),
        complete_operation_boundary="five frozen serving checks; seal durable prediction; close query; verify feedback after eight queries; atomic templates/coefficients/pending/acknowledgements commit",
        methodology_note="Constructed exact-verifier prototype; Brier plus NCE numerical training is declared separately from inference. Thirty downstream streams are frozen, unexecuted. No benefit claim.",
        numerical_training_calls_scope="Measured precondition probe and prototype, including restarted branch updates. Tests and cold replay recompute independently under validation receipts.",
        global_reconciliation_debt="Ops, traceability and full-repository health belong to the conductor reconciliation; owned validation is separate.",
        field_principles={},
        **work,
    )
    value["numerical_training_calls"] += int(pre["measured_cheap_unit_s"] > 0)
    health_path = ROOT / "results/raw" / NAME / "global-health/full_python_once.receipt.json"
    if health_path.is_file():
        value["repository_health_reference"] = reference(health_path)
    value["field_principles"] = {
        k: "Exact owned prototype operand; constructed readiness does not imply independent benefit."
        for k in value
    }
    return value


def replay(path: Path) -> Json:
    """Authenticate byte operands and reconstruct aggregates from the retained raw work."""
    try:
        value = json.loads(path.read_text())
        refs = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        hashes = all(sha256_file(Path(ref["path"])) == ref["sha256"] for ref in refs)
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        rows = work["rows"]
        expected_ready = int(
            all(value["acceptance_gates"].values())
            and value["owned_statement_coverage"]["complete"]
        )
        passed = (
            hashes
            and all(value[k] == work[k] for k in work if k != "numerical_training_calls")
            and value["completed_count"] == len(rows)
            and value["intended_count"] == 180
            and value["censored_count"] == 180 - len(rows)
            and value["reproducibility_checksum"] == canonical_hash(rows)
            and value["memory_execution_ready_score"] == expected_ready
            and value["numerical_training_calls"]
            == work["numerical_training_calls"]
            + int(value["preconditions_checked"]["measured_cheap_unit_s"] > 0)
        )
        if passed and rows:
            manifest = json.loads(Path(value["raw_shard_hashes"][1]["path"]).read_text())
            passed = manifest == m.freeze_manifest() == value["frozen_memory_protocol"]["manifest"]
            with tempfile.TemporaryDirectory(prefix="carnot8802-cold-replay-") as directory:
                reconstructed = m.prototype(Path(directory))
            passed = passed and reconstructed == work
        return dict(passed=bool(passed), reconstructed_rows=len(rows))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="replay_operand_failure")


def validate(raw: Path) -> Json:
    """Combine genuine unit and direct-CLI SQLite data before checking owned statements."""
    covdir = raw / "coverage"
    covdir.mkdir(parents=True, exist_ok=True)
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(covdir / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    py = str(ROOT / ".venv/bin/python")
    paths = OWNED + TESTS
    plan = [
        (
            "owned_pytest",
            [
                "env",
                "CARNOT_8802_COVERAGE_DIRECTORY=" + str(covdir),
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *TESTS,
            ],
            240,
        ),
        (
            "private_E2E_018_021",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
                "tests/python/test_primary_publication_7928.py",
            ],
            300,
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *paths], 60),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *paths], 60),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            120,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS], 60),
    ]
    receipts = [
        child(name, argv, raw / "validation", deadline=deadline, heartbeat=60)
        for name, argv, deadline in plan
    ]
    cov = Coverage(data_file=str(covdir / ".coverage"), config_file=False)
    cov.combine(data_paths=[str(covdir)], keep=True)
    cov.save()
    summary = coverage_summary(covdir / ".coverage", OWNED)
    return dict(
        receipts=receipts,
        coverage=summary,
        passed=all(r["passed"] for r in receipts) and summary["complete"],
    )


def terminal_checks(candidate: Path, raw: Path) -> Json:
    """Keep shared adversarial and row validators unchanged and retain exact argv."""
    py = str(ROOT / ".venv/bin/python")
    receipts = [
        child(
            "adversarial",
            [py, "scripts/adversarial_verify.py", str(candidate), "--json"],
            raw / "terminal",
            deadline=60,
            heartbeat=60,
        ),
        child(
            "row_consistency",
            [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            raw / "terminal",
            deadline=60,
            heartbeat=60,
        ),
        child(
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/tmp",
                py,
                "-u",
                str(ROOT / CLI),
                "--replay",
                str(candidate),
            ],
            raw / "terminal",
            deadline=60,
            heartbeat=60,
        ),
    ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def qualify_and_publish(value: Json, raw: Path, output: Path) -> None:
    """Readiness requires passing owned validation as well as causal controls."""
    started = time.monotonic()
    if value["acceptance_gates"]["preconditions"]:
        checked = validate(raw)
        value["validation_receipts"] = checked["receipts"]
        value["owned_statement_coverage"] = checked["coverage"]
        value["acceptance_gates"]["owned_validation"] = checked["passed"]
    ready = all(value["acceptance_gates"].values())
    value["memory_execution_ready_score"] = int(ready)
    value["required_checks_passed"] = ready
    if value["verdict_class"] != "blocked":
        value["verdict_class"] = "null" if ready else "disqualified"
        value["honest_verdict"] = (
            "complete_null_causal_execution_qualified_no_benefit_claim"
            if ready
            else "complete_disqualified_owned_validation"
        )
    value["phase_spans"].append(
        dict(phase="owned_validation", duration_s=time.monotonic() - started)
    )
    value["duration_s"] += time.monotonic() - started
    atomic_json(raw / "prepublication.json", value)
    m.progress("terminal_validation_before")
    try:
        publication = publish_primary(output, value, lambda path: terminal_checks(path, raw))
    except ValueError:
        atomic_json(raw / "rejected_candidate.json", value)
        raise
    atomic_json(raw / "publication.json", publication)
    m.progress("terminal_publication_complete", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Expose one direct no-model CLI and an outside-checkout replay entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        result = replay(args.replay)
        print(json.dumps(dict(replay=result)), flush=True)
        return int(not result["passed"])
    raw = args.raw or ROOT / "results/raw" / NAME / "invocations" / str(time.time_ns())
    value = run(raw, args.date)
    qualify_and_publish(value, raw, args.output)
    return 0
