"""REQ-REPORT-9000-SCOPES: qualify invocation custody without loading a model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from coverage import Coverage

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from scripts import adversarial_verify as verifier

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9000_v769_invocation_provenance", "exp9000-invocation-provenance"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/invocation_provenance_9000.py", CLI]
PRODUCERS = [
    "python/carnot/reporting/graph_corpus_a_8961.py",
    "python/carnot/reporting/graph_corpus_a_8990.py",
]
TEST = "tests/python/test_invocation_provenance_9000.py"
CONTROLS = (
    "live_fixture",
    "cached_current",
    "load_only",
    "missing_ledger",
    "contradictory_current",
)
CHECKS = {
    "owned_unit_child",
    "coverage_combine",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
    "full_python_suite",
}
INPUTS = [
    (
        "results/experiment_8990_v768_graph_corpus_a.json",
        "sha256:70882818afd83e69a4ea72a8530f99ce28d71ee8c87ffd3aa5b8f8a69c05c2bd",
    ),
    (
        "results/experiment_8987_v768_spline_development_admission.json",
        "sha256:32fbf7d2e347f56cf3baf5d92a73b7d62ab8bd68373518d0e5346a2ac485e920",
    ),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real phase boundaries and outstanding units without extending deadlines."""
    print(f"[exp9000] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind an existing operand to its exact bytes, naming absence explicitly."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def retain(path: Path, raw: Path) -> Json:
    """Keep complete original bytes under their content identity, refusing corruption."""
    digest = sha256_file(path)
    target = raw / "objects" / digest.split(":")[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) != digest:
        raise ValueError("authority_bytes")
    if not target.exists():
        shutil.copyfile(path, target)
    return dict(reference(target), source_path=str(path), scope="upstream")


def serialize_authentication(evidence: Json, raw: Path) -> Json:
    """Keep checks inline and replace imported protocol data with complete byte custody."""
    result = deepcopy(evidence)
    protocol = result.get("protocol", {})
    if protocol:
        refs = evidence["references"]
        if not refs or not Path(refs[0]["path"]).is_file():
            raise ValueError("authority_bytes")
        path = Path(refs[0]["path"])
        original = json.loads(path.read_bytes())
        if sha256_file(path) != refs[0]["sha256"] or any(
            original.get(k) != v for k, v in protocol.items()
        ):
            raise ValueError("authority_bytes")
        result["protocol"] = dict(
            retain(path, raw),
            experiment_id=protocol.get("experiment_id"),
            fields_imported=sorted(protocol),
        )
    return result


def ledger_counts(ledger: list[Json]) -> Json:
    """Independently reduce completed and rejected operations without headline counters."""
    counts = dict(ZERO_INVOCATION_COUNTS)
    seen: set[str] = set()
    for row in ledger:
        operation = {"model_load": "model_loads", "generation": "generation_calls"}[
            row["operation"]
        ]
        if (
            row["call_id"] in seen
            or row["scope"] != "current"
            or type(row["output_tokens"]) is not int
            or row["output_tokens"] < 0
        ):
            raise ValueError("ledger_identity")
        seen.add(row["call_id"])
        counts[operation + "_attempted"] += 1
        counts[operation + "_" + row["status"]] += 1
    return counts


def serialize_invocation(counts: Json, ledger: list[Json], raw: Path) -> Json:
    """Expose active operations; retain zero-generation load receipts without ambiguity."""
    if any(counts.values()) and (not ledger or ledger_counts(ledger) != counts):
        raise ValueError("missing_or_inconsistent_ledger")
    generated = sum(r["output_tokens"] for r in ledger if r["operation"] == "generation")
    cls = (
        "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "no_model_load"
    )
    inline = dict(counts)
    if cls == "model_load_no_generation":
        path = raw / "current_invocation_counts.json"
        atomic_json(path, dict(counts=counts, ledger=ledger, output_tokens=generated))
        inline = {k: v for k, v in counts.items() if k.startswith("model_loads_")}
        inline["complete_counts_reference"] = reference(path)
    return dict(inference_substrate_class=cls, model_invocation_counts=inline)


def checksum(value: Json) -> str:
    """Hash every field except the checksum itself for fresh-process replay."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def gate(path: Path, field: str, expected: Any, observed: Any, check: str) -> Json:
    """Retain the exact comparison and observed missing operand for every refusal."""
    row = dict(
        reference(path),
        upstream=path.name,
        check=check,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )
    if check == "historical_counts":
        row.update(
            expected=dict(expected, scope="upstream"), observed=dict(observed, scope="upstream")
        )
    return row


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate rejected evidence and every retained input before reducing history."""
    from carnot.reporting import graph_corpus_a_8990 as shard

    raw.mkdir(parents=True, exist_ok=True)
    checks: list[Json] = []
    refs: list[Json] = []
    evidence: Json = dict(
        checks=checks,
        references=refs,
        failures=[],
        historical_counts={},
        historical_ledger=[],
        original_flag_rows=[],
        invocation_path_rows=[],
    )
    for name, pin in INPUTS:
        path = root / name
        checks.append(
            gate(path, "sha256", pin, reference(path)["sha256"] or "missing_file", "input_bytes")
        )
        if checks[-1]["passed"]:
            refs.append(retain(path, raw))
    if any(not r["passed"] for r in checks):
        evidence["failures"] = [r for r in checks if not r["passed"]]
        return evidence
    original = json.loads((root / INPUTS[0][0]).read_bytes())
    for field, expected in [
        ("experiment_id", 8990),
        ("task_id", "exp8990-graph-corpus-a"),
        ("completed_count", 512),
    ]:
        checks.append(
            gate(
                root / INPUTS[0][0],
                field,
                expected,
                original.get(field, "missing_field"),
                "original_identity",
            )
        )
    operands = (
        original["raw_shard_hashes"]
        + original["source_artifact_hashes"]
        + original["code_config_hashes"]
    )
    for index, ref in enumerate(operands):
        path = Path(ref["path"])
        checks.append(
            gate(
                path,
                "sha256",
                ref["sha256"],
                reference(path)["sha256"] or "missing_file",
                "retained_bytes",
            )
        )
        if checks[-1]["passed"]:
            refs.append(retain(path, raw))
        if index % 250 == 0:
            progress("authenticate_retained", index + 1, len(operands) - index - 1)
    if not any(not r["passed"] for r in checks):
        work = json.loads(Path(original["raw_shard_hashes"][0]["path"]).read_bytes())
        ledger = json.loads((Path(work["raw"]) / "runtime/invocation_ledger.json").read_bytes())[
            "rows"
        ]
        counts = ledger_counts(ledger)
        runtime_path = Path(work["raw"]) / "runtime"
        loaded = {
            k: v
            for k, v in work["runtime"]["model_receipt"].items()
            if k not in {"preflight_gguf_sha256", "server_binary", "resident_gpu_state"}
        }
        checks.extend(
            [
                gate(
                    runtime_path / "invocation_ledger.json",
                    "rows[0].response_sha256",
                    canonical_hash(loaded),
                    ledger[0]["response_sha256"],
                    "historical_model_load",
                ),
                gate(
                    runtime_path / "invocation_ledger.json",
                    "rows[0].request_sha256",
                    canonical_hash(
                        json.loads((runtime_path / "preflight.json").read_bytes())["model"]
                    ),
                    ledger[0]["request_sha256"],
                    "historical_load_request",
                ),
                gate(
                    runtime_path / "invocation_ledger.json",
                    "generation.output_tokens",
                    {
                        r["unit_id"]: r["actual_token_usage"]["completion_tokens"]
                        for r in work["runtime"]["rows"]
                    },
                    {
                        r["call_id"]: r["output_tokens"]
                        for r in ledger
                        if r["operation"] == "generation"
                    },
                    "historical_token_usage",
                ),
                gate(
                    runtime_path / "invocation_ledger.json",
                    "owner_and_monotonic_clocks",
                    True,
                    all(
                        r["owner_pid"] == ledger[0]["owner_pid"]
                        and r["ended_monotonic_ns"] >= r["started_monotonic_ns"] > 0
                        for r in ledger
                    ),
                    "historical_owner_clocks",
                ),
            ]
        )
        checks.extend(
            [
                gate(
                    root / INPUTS[0][0],
                    "model_invocation_counts",
                    counts,
                    original["model_invocation_counts"],
                    "historical_counts",
                ),
                gate(
                    Path(work["raw"]),
                    "raw_complete",
                    True,
                    shard.raw_complete(work),
                    "raw_requests_responses",
                ),
                gate(
                    root / INPUTS[1][0],
                    "collection_ready_score",
                    1,
                    work["authentication"]["protocol"].get(
                        "collection_ready_score", "missing_field"
                    ),
                    "upstream_collection_gate",
                ),
                gate(
                    root / INPUTS[1][0],
                    "required_checks_passed",
                    True,
                    work["authentication"]["protocol"].get(
                        "required_checks_passed", "missing_field"
                    ),
                    "upstream_owned_gate",
                ),
            ]
        )
        private = raw / "original" / Path(INPUTS[0][0]).name
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes((root / INPUTS[0][0]).read_bytes())
        report = verifier.verify_artifact(private)
        atomic_json(raw / "original_report.json", report)
        expected = {"INFERENCE_PROVENANCE_CONTRADICTION", "SUBSTRATE_CLASS_MISMATCH"}
        flags = report["flags"]
        checks.append(
            gate(
                private,
                "critical_kinds",
                sorted(expected),
                sorted(r["kind"] for r in flags if r["severity"] == "critical"),
                "original_flags",
            )
        )
        paths = [
            dict(r, evidence_kind=kind)
            for kind, rows in zip(
                ("positive", "negative", "cited"),
                verifier._typed_invocation_evidence(original),
                strict=True,
            )
            for r in rows
        ]
        corrected = deepcopy(original)
        corrected["preconditions_checked"]["authentication"] = serialize_authentication(
            original["preconditions_checked"]["authentication"], raw
        )
        corrected.update(serialize_invocation(counts, ledger, raw))
        corrected["reproducibility_checksum"] = checksum(corrected)
        corrected_path = raw / "normalized_historical_candidate.json"
        atomic_json(corrected_path, corrected)
        corrected_report = verifier.verify_artifact(corrected_path)
        atomic_json(raw / "normalized_historical_report.json", corrected_report)
        checks.append(
            gate(
                corrected_path,
                "flag_count",
                0,
                corrected_report["flag_count"],
                "producer_serialization",
            )
        )
        paths.extend(
            dict(r, evidence_kind="normalized_" + kind)
            for kind, rows in zip(
                ("positive", "negative", "cited"),
                verifier._typed_invocation_evidence(corrected),
                strict=True,
            )
            for r in rows
        )
        evidence.update(
            historical_counts=counts,
            historical_ledger=ledger,
            original_flag_rows=flags,
            invocation_path_rows=paths,
        )
    evidence["failures"] = [r for r in checks if not r["passed"]]
    return evidence


def control_value(kind: str, raw: Path, input_root: Path | None = None) -> Json:
    """Build explicitly simulated control shapes; historical clocks are fixture data."""
    original = json.loads((ROOT / INPUTS[0][0]).read_bytes())
    counts = dict(ZERO_INVOCATION_COUNTS)
    ledger: list[Json] = []
    if kind in {"live_fixture", "load_only", "missing_ledger"}:
        ledger = [
            dict(
                call_id="fixture-load",
                operation="model_load",
                status="completed",
                scope="current",
                output_tokens=0,
            )
        ]
        if kind != "load_only":
            ledger.append(
                dict(
                    call_id="fixture-generation",
                    operation="generation",
                    status="completed",
                    scope="current",
                    output_tokens=2,
                )
            )
        counts = ledger_counts(ledger)
    invocation = serialize_invocation(counts, ledger, raw)
    value: Json = dict(
        experiment_id=9000,
        task_id=TASK,
        run_date="20261111",
        title="Private invocation contract fixture",
        honest_verdict="complete_null_invocation_fixture",
        verdict_class="null",
        fixture_only=True,
        control_kind=kind,
        raw_directory=str(raw),
        inference_substrate="aggregation_from_upstream_artifacts"
        if kind in {"cached_current", "contradictory_current"}
        else "live_llm_inference",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        target_model="private_schema_fixture_from_exp8990",
        random_seed=769000,
        model_invoked=bool(counts["model_loads_attempted"]),
        duration_s=original["duration_s"],
        duration_scope="fixture_clock_from_authenticated_exp8990_not_current_runtime",
        invocation_ledger=ledger if kind != "missing_ledger" else [],
        **invocation,
        preconditions_checked=dict(
            authentication=serialize_authentication(
                original["preconditions_checked"]["authentication"], raw
            )
        ),
        provenance_ready_score=0,
        independent_generalization_score=0,
        verifier_is_oracle=False,
        claim_scope="Private schema fixture; no live model or scientific claim",
        exposure_scope="fixture_only",
        intended_count=1,
        independent_count=1,
        completed_count=1,
        censored_count=0,
        excluded_count=0,
        failed_count=0,
        methodology_note="Fixture invocation shape replayed through unchanged validation; no current model was loaded.",
        field_principles={},
        raw_shard_hashes=[],
    )
    if kind == "contradictory_current":
        value["runtime"] = dict(model_invoked=True)
    if kind == "load_only":
        value["raw_shard_hashes"] = [
            invocation["model_invocation_counts"]["complete_counts_reference"]
        ]
    if input_root is not None:
        evidence = authenticate(input_root, raw / "fixture_authentication")
        value.update(fixture_input_root=str(input_root), gate_check_summary=evidence["checks"])
        if evidence["failures"]:
            value.update(
                honest_verdict="complete_blocked_" + evidence["failures"][0]["check"],
                verdict_class="blocked",
            )
    value["reproducibility_checksum"] = checksum(value)
    return value


def check_control(value: Json, path: Path) -> Json:
    """Use the real validator and independently require the active operation ledger."""
    atomic_json(path, value)
    report: Json = verifier.verify_artifact(path)
    counts = ledger_counts(value["invocation_ledger"])
    expected = {
        k: v for k, v in value["model_invocation_counts"].items() if k in ZERO_INVOCATION_COUNTS
    }
    passed = all(counts[k] == v for k, v in expected.items()) and report["flag_count"] == 0
    return dict(passed=passed, report=report, ledger_counts=counts)


def replay(path: Path) -> Json:
    """Rebuild fixtures or terminal summaries and authenticate every retained byte."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            raise ValueError("checksum_drift")
        for ref in (
            value["raw_shard_hashes"]
            + value.get("source_artifact_hashes", [])
            + value.get("code_config_hashes", [])
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        if value["fixture_only"]:
            rebuilt = control_value(
                value["control_kind"],
                Path(value["raw_directory"]),
                Path(value["fixture_input_root"]) if value.get("fixture_input_root") else None,
            )
        else:
            work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
            rebuilt = build(work)
        passed = value == rebuilt
        return dict(passed=passed, replay_passed=passed)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def build(work: Json) -> Json:
    """Reduce primitive control outcomes, preserving blocked and disqualified terminals."""
    evidence, validation = work["evidence"], work["validation"]
    rows = [
        dict(
            unit_id=r["name"],
            group_id=r["name"],
            arm="invocation_contract",
            condition=r["name"],
            metric="expected_exit_observed",
            value=int(r["passed"]),
            numerator=int(r["passed"]),
            denominator=1,
            status="completed" if r["passed"] else "failed",
            exclusion_reason=None if r["passed"] else "control_failed",
        )
        for r in work["controls"]
    ]
    owned = (
        {r["name"] for r in validation["receipts"]} == CHECKS
        and len(validation["receipts"]) == len(CHECKS)
        and all(
            r["passed"] and r["exit_code"] == 0 and not r["timed_out"]
            for r in validation["receipts"]
        )
        and verify_coverage(validation["coverage"])
        and not work.get("terminal_rejected", False)
    )
    controls = {r["name"] for r in work["controls"]} == set(CONTROLS) and all(
        r["passed"] for r in work["controls"]
    )
    ready = int(owned and controls and not evidence["failures"])
    cls = (
        "disqualified"
        if not owned
        else "blocked"
        if evidence["failures"]
        else "positive"
        if ready
        else "disqualified"
    )
    verdict = (
        "complete_blocked_" + evidence["failures"][0]["check"]
        if cls == "blocked"
        else "complete_" + cls + "_invocation_provenance"
    )
    value: Json = dict(
        experiment_id=9000,
        task_id=TASK,
        title="Current and historical invocation provenance qualification",
        milestone="2026.11.769",
        run_date="20261111",
        honest_verdict=verdict,
        verdict_class=cls,
        status="success" if ready else "blocked" if cls == "blocked" else "error",
        verifier_is_oracle=False,
        claim_scope="Producer invocation custody only; no verifier benefit or new inference",
        exposure_scope="authenticated_historical_artifacts_and_private_fixtures",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        model_invoked=False,
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(name).name.split("_")[1]),
                sha256=pin,
                fields_imported=[
                    "model_invocation_counts",
                    "invocation_ledger",
                    "preconditions_checked",
                    "raw_shard_hashes",
                    "code_config_hashes",
                ],
            )
            for name, pin in INPUTS
        ],
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("terminal_rejected")),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        preconditions_checked=dict(
            environment=work["environment"], checks=evidence["checks"], required_network_requests=0
        ),
        gate_check_summary=evidence["checks"],
        rows=rows,
        intended_count=len(CONTROLS),
        independent_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=len(CONTROLS) - len(rows),
        excluded_count=0,
        duration_s=work["duration_s"],
        random_seed=769000,
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        raw_shard_hashes=[
            reference(Path(work["raw"]) / "work.json"),
            *work["raw_references"],
            *work.get("control_references", []),
        ],
        code_config_hashes=work["code_references"],
        producer_code_hashes=[r for r in work["code_references"] if r["source_path"] in PRODUCERS],
        provenance_ready_score=ready,
        original_flag_rows=evidence["original_flag_rows"],
        invocation_path_rows=evidence["invocation_path_rows"],
        mutation_rows=work["controls"],
        fixture_only=False,
        methodology_note="Authenticate historical bytes and independently reduce invocation ledgers. Private typed controls cross the actual publication CLI; fixtures establish serialization only. No current model was loaded.",
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Only authentic operands and passing normal-exit owned checks qualify custody.",
            )
            for k, v in dict(
                authenticated_inputs=not evidence["failures"], controls=controls, owned_checks=owned
            ).items()
        },
        repository_health=work.get("repository_health", {}),
        field_principles={},
    )
    value["field_principles"] = {
        k: "Current work is separate from cited history; primitive bytes and normal exits determine readiness."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def environment(raw: Path) -> Json:
    """Authenticate virtualenv, installed dependencies, free memory and writable disk."""
    raw.mkdir(parents=True, exist_ok=True)
    probe = raw / ".write_probe"
    probe.write_bytes(b"9000")
    probe.unlink()
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    free = shutil.disk_usage(raw).free
    deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pytest", "coverage", "ruff", "mypy")
    }
    checks = [
        gate(ROOT / ".venv", "sys.prefix", str(ROOT / ".venv"), sys.prefix, "virtualenv"),
        gate(
            raw,
            "resources_available",
            True,
            memory >= 512 * 1024**2 and free >= 512 * 1024**2,
            "resources",
        ),
        gate(raw, "dependencies", True, all(deps.values()), "dependencies"),
    ]
    return dict(
        checks=checks,
        dependencies=deps,
        free_memory_bytes=memory,
        free_disk_bytes=free,
        writable_disk=True,
        network_requests=0,
        python=sys.executable,
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze bounded explicit checks; selectors are passed only to pytest."""
    private.mkdir(parents=True, exist_ok=True)
    rc = private / "coverage.ini"
    rc.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED + PRODUCERS)
        + "[report]\nexclude_lines=\n"
    )
    py, pytest = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/pytest")
    tests = [
        TEST,
        "tests/python/test_graph_corpus_a_8990.py",
        "tests/python/test_graph_corpus_a_8961.py",
        "tests/python/test_graph_corpus_b_8962.py",
    ]
    return [
        CommandSpec(
            "owned_unit_child",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(rc),
                "-m",
                "pytest",
                "--noconftest",
                "-s",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "tests"),
                *tests,
                "-q",
            ),
            "owned",
            300,
        ),
        CommandSpec(
            "coverage_combine",
            (py, "-m", "coverage", "combine", "--rcfile=" + str(rc)),
            "owned",
            60,
        ),
        CommandSpec(
            "ruff_check",
            (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *PRODUCERS, TEST),
            "owned",
            60,
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *PRODUCERS, TEST),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *OWNED,
            ),
            "owned",
            120,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", *tests), "owned", 60),
        *[
            CommandSpec(
                name,
                (
                    pytest,
                    "--noconftest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "--basetemp=" + str(private / name),
                    path,
                    "-q",
                ),
                "owned",
                120,
            )
            for name, path in [
                ("e2e018", "tests/python/test_experiment_7891_v685_authority_lifecycle.py"),
                ("e2e021", "tests/python/test_restricted_decision_audit_8210.py"),
            ]
        ],
    ]


def coverage_files(tracer: Coverage) -> list[Json]:
    """Select new statements against frozen producer bytes rather than legacy coverage."""
    original = json.loads((ROOT / INPUTS[0][0]).read_bytes())
    files: list[Json] = []
    for name in OWNED + PRODUCERS:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        selected = set(statements)
        if name in PRODUCERS:
            old = next(
                r for r in original["code_config_hashes"] if r.get("source_path", "").endswith(name)
            )
            before = Path(old["path"]).read_text().splitlines()
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
                statements=len(selected),
                statement_lines=sorted(selected),
                missing=sorted(selected & set(missing)),
                excluded=sorted(selected & set(excluded)),
            )
        )
    return files


def measure(private: Path, raw: Path, evidence: Json) -> Json:
    """Check all new statements and changed producer statements in real SQLite data."""
    tracer = Coverage(data_file=str(private / ".coverage"), config_file=False)
    tracer.set_option("report:exclude_lines", [])
    tracer.load()
    files = coverage_files(tracer)
    shutil.copyfile(private / ".coverage", raw / "owned.coverage")
    return dict(
        complete=all(not r["missing"] and not r["excluded"] for r in files)
        and all(r["statements"] for r in files[:2]),
        percent=100
        * sum(r["statements"] - len(r["missing"]) for r in files)
        / sum(r["statements"] for r in files),
        files=files,
        data_reference=reference(raw / "owned.coverage"),
    )


def verify_coverage(report: Json) -> bool:
    """Reopen hashed SQLite data so a repaired JSON summary cannot invent coverage."""
    try:
        ref = report["data_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        tracer = Coverage(data_file=ref["path"], config_file=False)
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        files = coverage_files(tracer)
        return (
            files == report["files"]
            and report["complete"] is True
            and all(not r["missing"] and not r["excluded"] for r in files)
            and all(r["statements"] for r in files[:2])
        )
    except (OSError, ValueError, KeyError):
        return False


def foreign_argv(*args: str) -> tuple[str, ...]:
    """Execute the actual CLI outside the checkout after removing PYTHONPATH."""
    return (
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-c",
        "import os,sys,tempfile; os.environ.pop('PYTHONPATH',None); os.chdir(tempfile.gettempdir()); os.execv(sys.executable,[sys.executable,'-u',*sys.argv[1:]])",
        str(ROOT / CLI),
        *args,
    )


def execute(raw: Path, input_root: Path) -> Json:
    """Run bounded no-model qualification and retain all primitive validation receipts."""
    started = time.monotonic()
    progress("before_preconditions")
    resources = environment(raw)
    evidence = authenticate(input_root, raw)
    evidence["checks"].extend(resources["checks"])
    evidence["failures"].extend(r for r in resources["checks"] if not r["passed"])
    progress("after_preconditions")
    controls: list[Json] = []
    control_root = Path(tempfile.mkdtemp(prefix="carnot-9000-controls-"))
    if not evidence["failures"]:
        for i, kind in enumerate(CONTROLS):
            output = control_root / kind / (NAME + ".json")
            receipt = run_commands(
                ROOT,
                [
                    CommandSpec(
                        kind,
                        foreign_argv(
                            "--date", "20261111", "--fixture-e2e", str(output), "--control", kind
                        ),
                        "private_control",
                        60,
                    )
                ],
                log_dir=raw / "control_logs" / kind,
                heartbeat_s=15,
            )[0]
            expected = 1 if kind in {"missing_ledger", "contradictory_current"} else 0
            controls.append(
                dict(
                    name=kind,
                    expected_exit=expected,
                    observed_exit=receipt["exit_code"],
                    passed=receipt["exit_code"] == expected and not receipt["timed_out"],
                    receipt=receipt,
                    fixture_only=True,
                )
            )
            progress("control_complete", i + 1, len(CONTROLS) - i - 1)
    control_files = [
        retain(path, raw) for path in sorted(control_root.rglob("*")) if path.is_file()
    ]
    measured_at = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="carnot-9000-validation-"))
    plan = validation_plan(private)
    atomic_json(
        raw / "validation_plan.json",
        dict(commands=[dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in plan]),
    )
    progress("before_owned_validation")
    receipts = run_commands(
        ROOT,
        plan,
        log_dir=raw / "validation",
        heartbeat_s=15,
        extra_env=dict(
            COVERAGE_RCFILE=str(private / "coverage.ini"),
            PYTHONUNBUFFERED="1",
            PYTHONPATH=os.pathsep.join([str(ROOT / "python"), str(ROOT)]),
        ),
    )
    coverage = (
        measure(private, raw, evidence)
        if (private / ".coverage").is_file()
        else dict(complete=False, files=[], percent=0)
    )
    progress("after_owned_validation")
    health = run_commands(
        ROOT,
        [
            CommandSpec(
                "full_python_suite",
                (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                "repository_health",
                300,
            )
        ],
        log_dir=raw / "repository_health",
        heartbeat_s=15,
    )
    refs = [
        dict(retain(ROOT / name, raw), source_path=name)
        for name in OWNED
        + PRODUCERS
        + [
            TEST,
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "python/carnot/reporting/primary_publication.py",
            "ops/exclusion_manifest.yaml",
            "openspec/change-proposals/research-roadmap-v768-preserved-20261111.md",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
        ]
    ]
    work: Json = dict(
        raw=str(raw),
        evidence=evidence,
        environment=resources,
        validation=dict(receipts=receipts + health, coverage=coverage),
        controls=controls,
        control_references=control_files,
        duration_s=time.monotonic() - started,
        phase_spans=[
            dict(phase="authentication_and_controls", duration_s=measured_at - started),
            dict(phase="validation", duration_s=time.monotonic() - measured_at),
        ],
        code_references=refs,
        raw_references=[],
        repository_health=dict(receipts=health, affects_owned_readiness=True),
    )
    work["raw_references"] = [
        reference(p) for p in sorted(raw.rglob("*")) if p.is_file() and "objects" not in p.parts
    ]
    atomic_json(raw / "work.json", work)
    return work


def main(argv: list[str] | None = None) -> int:
    """Publish private controls or the checked no-model qualification with normal exits."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--control", choices=CONTROLS, default="cached_current")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--input-root", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261111":
        parser.error("this producer requires --date 20261111")
    output = args.fixture_e2e or ROOT / "results" / (NAME + ".json")
    if args.fixture_e2e and output.resolve().is_relative_to(ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.parent / "raw" / NAME
    progress("before_private_control" if args.fixture_e2e else "before_qualification")
    if args.fixture_e2e:
        value = control_value(args.control, raw, args.input_root)
        report = check_control(value, raw / "control_candidate.json")
        atomic_json(raw / "control_report.json", report)
        if not report["passed"]:
            progress("private_control_rejected", 1, 0)
            return 1
        published = publish_primary(
            output,
            value,
            lambda candidate: dict(
                passed=replay(candidate)["passed"]
                and check_control(value, raw / "checked_control.json")["passed"]
            ),
        )
    else:
        raw = raw / "invocations" / str(time.time_ns())
        work = execute(raw, Path(os.environ.get("CARNOT_9000_INPUT_ROOT", str(ROOT))))
        value = build(work)

        def validate(candidate: Path) -> Json:
            py = str(ROOT / ".venv/bin/python")
            receipts = run_commands(
                ROOT,
                [
                    CommandSpec(
                        "cold_replay_cli",
                        foreign_argv("--cold-replay", str(candidate)),
                        "terminal",
                        60,
                    ),
                    CommandSpec(
                        "adversarial",
                        (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
                        "terminal",
                        60,
                    ),
                    CommandSpec(
                        "strict_rows",
                        (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
                        "terminal",
                        60,
                    ),
                ],
                log_dir=raw / "terminal_validation",
                heartbeat_s=15,
            )
            return dict(
                passed=replay(candidate)["passed"] and all(r["passed"] for r in receipts),
                receipts=receipts,
            )

        try:
            published = publish_primary(output, value, validate)
        except ValueError as error:
            if str(error) != "candidate_rejected":
                raise
            work["terminal_rejected"] = True
            atomic_json(raw / "work.json", work)
            value = build(work)
            published = publish_primary(output, value, validate)
    progress("publication_complete", 1, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                provenance_ready_score=value["provenance_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
