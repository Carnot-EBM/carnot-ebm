"""REQ-REPORT-8656: qualify execution while preserving every historical scientific null."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage.exceptions import DataError

from carnot.reporting import streaming_adapter_8656 as a
from carnot.reporting import streaming_source_8629 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import streaming_evidence as streaming

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8656_v743_streaming_execution_custody"
TASK = "exp8656-streaming-execution-custody"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_streaming_execution_8656.py"
OWNED = [
    "python/carnot/reporting/streaming_adapter_8656.py",
    "python/carnot/reporting/streaming_execution_8656.py",
    "python/carnot/reporting/streaming_execution_runner_8656.py",
    CLI,
]
PROTECTED = [
    "scripts/research_conductor.py",
    "research-roadmap.yaml",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "python/carnot/reporting/primary_publication.py",
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so the supervisor can identify work and bounded waiting."""
    print(f"[exp8656] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(root: Path, raw: Path, date: str) -> Json:
    """Measure unchanged lexical arithmetic and seal source authority before any validation."""
    started = time.monotonic_ns()
    progress("preconditions_no_model_load_current_calls_0")
    fixture = old.measure(root, raw / "fixtures", date)
    scratch = Path(fixture["preconditions_checked"]["scratch"])
    shutil.rmtree(scratch)
    refs = []
    for row in fixture["chunk_rows"]:
        row["primitive"] = a.c.freeze(Path(row["primitive"]["path"]), raw)
        refs.append(row["primitive"])
    history = a.authority(root, raw)
    refs += [r["reference"] for view in history.get("views", {}).values() for r in view]
    refs += history.get("failure_logs", [])
    refs += [ref for row in history.get("consumer_failures", []) for ref in row["logs"]]
    gates = list(history["gates"])
    methods = root / "results/experiment_8653_v743_contract_methods.json"
    branch = [
        a.gate(methods, field) for field in ("methods_ready_score", "extraction_inputs_ready_score")
    ]
    gates += [r for r in branch if not r["passed"]]
    semantic = []
    for row in history.get("views", {}).get("8629", []):
        path = Path(row["origin_path"])
        semantic.append(
            dict(
                path=str(path),
                original_reference=row["reference"],
                original_sha256=row["reference"]["sha256"],
                current_sha256=sha256_file(path),
                role="semantic_extractor"
                if path.name in {"streaming_evidence.py", "streaming_source_8629.py"}
                else "historical_test"
                if path.name.startswith("test_")
                else "historical_driver",
            )
        )
    authority = [
        dict(origin_path=str(ROOT / p), reference=a.c.freeze(ROOT / p, raw))
        for p in [*OWNED, TEST, *old.INPUTS, *PROTECTED, *old.OWNED]
    ]
    refs += [r["reference"] for r in authority]
    if methods.is_file():
        refs.append(a.c.freeze(methods, raw))
    progress("measurement_complete", len(fixture["chunk_rows"]), 33 - len(fixture["chunk_rows"]))
    return dict(
        raw=str(raw.absolute()),
        fixture=fixture,
        history=history,
        gates=gates,
        branch_checks=branch,
        semantic_source_identity_rows=semantic,
        current_authority=authority,
        immutable_refs=refs,
        started_monotonic_ns=started,
        scratch_cleanup=not scratch.exists(),
    )


def seal(work: Json) -> Json:
    """Freeze measured operands once; validation never edits their invocation anchor."""
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [
        dict(
            phase="measurement",
            started_monotonic_ns=work["started_monotonic_ns"],
            ended_monotonic_ns=ended,
        )
    ]
    path = Path(work["raw"]) / "measurement.json"
    atomic_json(path, work)
    ref = a.c.freeze(path, Path(work["raw"]))
    path.chmod(0o444)
    progress("immutable_measurement_sealed")
    return ref


def seal_validation(validation: Json, raw: Path) -> Json:
    """Bind later execution evidence separately, without rewriting measured fixture operands."""
    path = raw / "validation.json"
    atomic_json(path, validation)
    reference = a.c.freeze(path, raw)
    path.chmod(0o444)
    return dict(validation, seal_reference=reference)


def build(ref: Json, validation: Json, receipts: list[Json], findings: Json, output: Path) -> Json:
    """Readiness requires owned execution; unavailable independent inputs remain blocked."""
    work = json.loads(Path(ref["path"]).read_bytes())
    all_receipts = validation["receipts"] + receipts
    value = old.build(
        work["fixture"], dict(receipts=all_receipts, adversarial_findings=findings), output
    )
    acceptance = dict(
        value["acceptance_gates"],
        sealed_validation=bool(validation.get("seal_reference"))
        and a.c.authenticate(validation["seal_reference"]),
        original_node_union=validation["original_node_union"]["passed"],
        current_node_union=validation["current_node_union"]["passed"],
        role_coverage=validation["coverage"]["passed"],
        child_capture=all(m["passed"] for m in validation["child_shard_manifest"]),
        legacy_failures_reproduced=validation["legacy_failures_reproduced"],
        immutable_operands=all(a.c.authenticate(r) for r in work["immutable_refs"]),
        semantic_sources=all(
            r["original_sha256"] == r["current_sha256"]
            for r in work["semantic_source_identity_rows"]
            if r["role"] == "semantic_extractor"
        ),
        disk_cleanup=work["scratch_cleanup"],
        adversarial_clear=not findings.get("flagged_count", 0),
    )
    owned = all(
        v
        for k, v in acceptance.items()
        if k != "original_inputs"
        and (
            acceptance["original_inputs"]
            or k not in {"exact_parity", "bounded_memory", "resource_rejections"}
        )
    )
    gates = value["gate_check_summary"] + work["gates"]
    if not owned:
        gates.append(
            dict(
                upstream=TASK,
                check="owned_execution",
                path=ref["path"],
                sha256=ref["sha256"],
                field="acceptance_gates",
                operator="all_true",
                expected=True,
                observed=acceptance,
            )
        )
    verdict = "disqualified" if not owned else "blocked" if gates else "null"
    ready = int(verdict == "null")
    value.update(
        experiment_id=8656,
        task_id=TASK,
        milestone="2026.10.743",
        honest_verdict=f"complete_{verdict}_v743_streaming_execution_custody",
        verdict_class=verdict,
        streaming_extractor_ready_score=ready,
        branch_gate_values=dict(streaming_extractor_ready_score=ready),
        required_checks_passed=owned,
        acceptance_gates=acceptance,
        gate_check_summary=gates,
        immutable_measurement_reference=ref,
        measurement_reference=ref,
        primitive_reference=ref,
        historical_driver_reference=work["history"].get("historical_driver_reference"),
        semantic_source_identity_rows=work["semantic_source_identity_rows"],
        historical_quarantines=work["history"],
        child_source_roles=[m["role"] for m in validation["child_shard_manifest"]],
        child_shard_manifest=validation["child_shard_manifest"],
        original_node_union=validation["original_node_union"],
        current_node_union=validation["current_node_union"],
        owned_statement_coverage=validation["coverage"],
        validation_evidence=validation,
        validation_reference=validation.get("seal_reference"),
        extra_validation_receipts=receipts,
        source_artifact_hashes=work["history"]["historical"],
        cited_upstream_artifacts=work["history"]["historical"],
        code_config_hashes=work["current_authority"],
        raw_shard_hashes=work["immutable_refs"] + validation.get("immutable_refs", []),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=8656,
        methodology_note="Constructed lexical execution qualification only. Original/current drivers have separate authority. No natural responses, entailment, head inference, learned benefit or generalization measurement.",
    )
    if len(value["rows"]) < 33:
        value["rows"] = [
            dict(
                family_id=r["family_id"],
                chunk_size=chunk,
                status="censored",
                metric=None,
                missing_reason="missing_external_operand",
            )
            for r in old.fixtures()
            for chunk in (1, 32, 128)
        ]
    value["field_principles"] = {
        k: "Invocation-bound execution qualification; never scientific benefit." for k in value
    }
    value["field_principles"].update(
        historical_driver_reference="Original sealed driver executes only in its private source view.",
        child_source_roles="Consumer data cannot supply producer or adapter statement coverage.",
        immutable_measurement_reference="Frozen before validation and bound to the invocation anchor.",
        duration_s="Real measurement elapsed time; each later validation receipt retains its separate clocks.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute original windows and exact features, even when a tamperer repairs every hash."""
    progress("cold_replay_start")
    try:
        value = json.loads(path.read_bytes())
        ref = value["immutable_measurement_reference"]
        if not a.c.authenticate(ref):
            raise ValueError("measurement_address")
        work = json.loads(Path(ref["path"]).read_bytes())
        if sha256_file(Path(work["raw"]) / "measurement.json") != ref["sha256"]:
            raise ValueError("invocation_anchor")
        if not all(a.c.authenticate(r) for r in work["immutable_refs"]):
            raise ValueError("immutable_operand")
        for row in work["current_authority"]:
            if sha256_file(Path(row["origin_path"])) != row["reference"]["sha256"]:
                raise ValueError("current_source")
        for row in work["history"]["historical"]:
            origin = str(Path(row["origin_path"]).relative_to(ROOT))
            if row["reference"]["sha256"] != a.HISTORICAL[origin] or not a.c.authenticate(
                row["reference"]
            ):
                raise ValueError("historical_primary")
        for row in work["semantic_source_identity_rows"]:
            if sha256_file(Path(row["path"])) != row["current_sha256"] or not a.c.authenticate(
                row["original_reference"]
            ):
                raise ValueError("mixed_source")
        validation = value["validation_evidence"]
        vref = value["validation_reference"]
        if (
            not a.c.authenticate(vref)
            or not all(a.c.authenticate(r) for r in validation.get("immutable_refs", []))
            or sha256_file(Path(work["raw"]) / "validation.json") != vref["sha256"]
            or json.loads(Path(vref["path"]).read_bytes())
            != {k: v for k, v in validation.items() if k != "seal_reference"}
        ):
            raise ValueError("validation_anchor")
        for manifest in validation["child_shard_manifest"]:
            role = manifest["role"]
            sources = {r["source"]: r for r in role["sources"]}
            expected_role = (
                "historical_driver"
                if manifest["name"].startswith("original")
                else "historical_consumer"
                if manifest["name"].startswith("consumer")
                else "streaming_producer"
                if manifest["name"].startswith("current8629")
                else "current_adapter"
            )
            if role["role"] != expected_role or not all(
                a.c.authenticate(r["reference"]) for r in sources.values()
            ):
                raise ValueError("source_role")
            for shard in manifest["shards"]:
                if not a.c.authenticate(shard["reference"]):
                    raise ValueError("child_shard")
                data = a.CoverageData(basename=shard["reference"]["path"])
                data.read()
                actual = [
                    dict(
                        source=f,
                        source_sha256=sources[f]["source_sha256"],
                        lines=sorted(data.lines(f) or []),
                        contexts={str(n): sorted(v) for n, v in data.contexts_by_lineno(f).items()},
                    )
                    for f in sorted(data.measured_files())
                ]
                if actual != shard["inventory"]:
                    raise ValueError("source_inventory")
        if not work["fixture"]["gate_check_summary"]:
            rows = work["fixture"]["chunk_rows"]
            if [(r["family_id"], r["chunk_size"]) for r in rows] != [
                (r["family_id"], c) for r in old.fixtures() for c in (1, 32, 128)
            ]:
                raise ValueError("original_chunk_union")
            with TemporaryDirectory(prefix="exp8656-replay-", dir="/var/tmp") as directory:
                for i, fixture in enumerate(old.fixtures()):
                    expected, records = old.eager(fixture)
                    for row in rows[i * 3 : i * 3 + 3]:
                        actual_result = streaming.extract(
                            fixture, Path(directory) / "windows.jsonl", chunk_size=row["chunk_size"]
                        )
                        if (
                            Path(row["primitive"]["path"]).read_bytes() != records
                            or row["features"] != expected
                            or any(
                                row[k] != actual_result[k]
                                for k in (
                                    "features",
                                    "complete",
                                    "complete_window_digest",
                                    "total_window_count",
                                    "reserved_working_bytes",
                                )
                            )
                        ):
                            raise ValueError("original_semantic_parity")
                    progress("cold_fixture", i + 1, 10 - i)
        expected_value = build(
            ref,
            validation,
            value["extra_validation_receipts"],
            value["adversarial_findings"],
            Path(value["publication_output"]),
        )
        if expected_value != value:
            raise ValueError("reduction_drift")
    except (OSError, ValueError, KeyError, TypeError, DataError) as error:
        progress("rejected_" + str(error)[:100])
        return False
    progress("replay_passed")
    return True
