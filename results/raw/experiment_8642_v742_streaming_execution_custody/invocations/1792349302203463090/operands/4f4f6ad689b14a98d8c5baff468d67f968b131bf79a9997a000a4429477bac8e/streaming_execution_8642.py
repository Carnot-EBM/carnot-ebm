"""REQ-REPORT-8642: qualify execution without claiming natural detection benefit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import streaming_custody_8642 as custody
from carnot.reporting import streaming_source_8629 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import streaming_evidence as streaming

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8642_v742_streaming_execution_custody"
TASK = "exp8642-streaming-execution-custody"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_streaming_custody_8642.py"
OWNED = [
    "python/carnot/reporting/streaming_custody_8642.py",
    "python/carnot/reporting/streaming_execution_8642.py",
    "python/carnot/reporting/streaming_execution_runner_8642.py",
    CLI,
]
SOURCES = [ROOT / p for p in [*OWNED, *old.OWNED]]
SUITES = [
    old.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    TEST,
]
HISTORY = [
    "results/experiment_8629_v741_streaming_source_prototype.json",
    "results/experiment_8639_v741_capstone.json",
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so the supervisor can distinguish work from a stall."""
    print(f"[exp8642] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze new fixture operands; old disqualification and changed bytes stay historical."""
    started = time.monotonic_ns()
    progress("preconditions_no_model_load_current_calls_0")
    dedup_before = set(Path("/var/tmp").glob("exp8629-dedup-*"))
    fixture = old.measure(root, raw / "fixtures", date)
    dedup_clean = set(Path("/var/tmp").glob("exp8629-dedup-*")) == dedup_before
    scratch = Path(fixture["preconditions_checked"]["scratch"])
    shutil.rmtree(scratch)
    refs = [
        custody.freeze(Path(r["path"]), raw) for r in fixture["source_refs"] + fixture["code_refs"]
    ]
    for row in fixture["chunk_rows"]:
        row["primitive"] = custody.freeze(Path(row["primitive"]["path"]), raw)
        refs.append(row["primitive"])
    historical = []
    drift, source_checks = [], []
    for name in HISTORY:
        path = root / name
        if path.is_file():
            historical.append(dict(origin_path=str(path), reference=custody.freeze(path, raw)))
            if name == HISTORY[0]:
                prior = json.loads(path.read_bytes())
                operand = prior["primitive_reference"]
                current = Path(operand["path"])
                observed = sha256_file(current)
                retained = custody.freeze(current, raw)
                refs.append(retained)
                drift.append(
                    dict(
                        path=str(current),
                        expected=operand["sha256"],
                        observed=observed,
                        immutable_observed_copy=retained,
                        historical_verdict=prior["honest_verdict"],
                        historical_quarantine_preserved=True,
                    )
                )
                source_checks = [
                    dict(
                        path=r["path"],
                        expected=r["sha256"],
                        observed=sha256_file(Path(r["path"])),
                        passed=sha256_file(Path(r["path"])) == r["sha256"],
                    )
                    for r in prior["code_config_hashes"]
                ]
        else:
            fixture["gate_check_summary"].append(
                dict(
                    check="required_historical_input",
                    upstream=Path(name).stem,
                    path=str(path),
                    sha256=None,
                    field="exists",
                    operator="==",
                    expected=True,
                    observed=None,
                )
            )
    regression = custody.retention_regression(raw)
    refs += [r["reference"] for r in historical]
    refs += [r["reference"] for r in regression["direct_capture"]]
    refs.append(regression["source_reference"])
    authority = [
        dict(origin_path=str(p), reference=custody.freeze(p, raw))
        for p in [*SOURCES, *(ROOT / s for s in SUITES)]
    ]
    refs += [r["reference"] for r in authority]
    return dict(
        raw=str(raw.absolute()),
        fixture=fixture,
        immutable_refs=refs,
        historical=historical,
        historical_operand_drift=drift,
        original_source_hash_checks=source_checks,
        original_authority=authority,
        retention_regression=regression,
        scratch_cleanup=not scratch.exists() and dedup_clean,
        started_monotonic_ns=started,
    )


def seal(work: Json) -> Json:
    """Seal once before candidate construction; validators only read this measurement."""
    raw = Path(work["raw"])
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [
        dict(
            phase="measurement_and_child_qualification",
            started_monotonic_ns=work["started_monotonic_ns"],
            ended_monotonic_ns=ended,
        )
    ]
    path = raw / "measurement.json"
    atomic_json(path, work)
    ref = custody.freeze(path, raw)
    path.chmod(0o444)
    progress("immutable_measurement_sealed")
    return ref


def build(ref: Json, receipts: list[Json], findings: Json, output: Path) -> Json:
    """Readiness follows all owned checks; exposed fixture parity remains a scientific null."""
    work = json.loads(Path(ref["path"]).read_bytes())
    validation = work["validation"]
    all_receipts = validation["receipts"] + receipts
    value = old.build(
        work["fixture"], dict(receipts=all_receipts, adversarial_findings=findings), output
    )
    acceptance = dict(
        value["acceptance_gates"],
        original_node_union=validation["original_node_union"]["passed"],
        actual_owned_statements=validation["coverage"]["passed"],
        direct_child_capture=all(r["passed"] for r in validation["child_shard_manifest"]),
        old_retention_failure_reproduced=work["retention_regression"]["passed"],
        actual_disk_cleanup=work["scratch_cleanup"],
        immutable_operands=all(custody.authenticate(r) for r in work["immutable_refs"]),
        original_source_hashes=all(r["passed"] for r in work["original_source_hash_checks"]),
        adversarial_clear=not findings.get("flagged_count", 0),
    )
    verdict = value["verdict_class"]
    owned = all(
        v
        for k, v in acceptance.items()
        if k != "original_inputs"
        and (
            acceptance["original_inputs"]
            or k not in {"exact_parity", "bounded_memory", "resource_rejections"}
        )
    )
    if not owned or findings.get("flagged_count", 0):
        verdict = "disqualified"
    ready = int(verdict == "null")
    value.update(
        experiment_id=8642,
        task_id=TASK,
        milestone="2026.10.742",
        honest_verdict=f"complete_{verdict}_v742_streaming_execution_custody",
        verdict_class=verdict,
        streaming_extractor_ready_score=ready,
        branch_gate_values=dict(streaming_extractor_ready_score=ready),
        required_checks_passed=owned,
        acceptance_gates=acceptance,
        measurement_reference=ref,
        primitive_reference=ref,
        child_shard_manifest=validation["child_shard_manifest"],
        original_node_union=validation["original_node_union"],
        owned_statement_coverage=validation["coverage"],
        immutable_revalidation_rows=validation.get("immutable_revalidation_rows", []),
        historical_quarantines=work["historical"],
        historical_operand_drift=work["historical_operand_drift"],
        original_coverage_retention=False,
        retention_regression=work["retention_regression"],
        source_artifact_hashes=work["historical"],
        cited_upstream_artifacts=work["historical"],
        code_config_hashes=work["original_authority"],
        raw_shard_hashes=work["immutable_refs"],
        extra_validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=8642,
        methodology_note="Constructed fixture execution qualification only. Natural coverage belongs exclusively to Exp8645. No semantic detection, generalization or learning claim.",
    )
    if not owned and not any(r["check"] == "owned_validation" for r in value["gate_check_summary"]):
        value["gate_check_summary"].append(
            dict(
                check="owned_validation",
                upstream=TASK,
                path=ref["path"],
                sha256=ref["sha256"],
                field="acceptance_gates",
                operator="all_true",
                expected=True,
                observed=acceptance,
            )
        )
    value["field_principles"] = {
        k: "Invocation-bound execution qualification; never scientific benefit." for k in value
    }
    value["field_principles"].update(
        measurement_reference="Hash-addressed frozen operands precede candidates.",
        child_shard_manifest="Actual SQLite statements, source identities, commands, exits and logs are captured per child.",
        original_node_union="Collected original assertions execute exactly once; killed children fail.",
        target_model="Only fixed lexical numerical arithmetic executes; no energy head or checkpoint is loaded.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute original eager semantics and custody so repaired summaries cannot pass."""
    progress("cold_replay_start")
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        if not custody.authenticate(ref):
            raise ValueError("measurement_address")
        work = json.loads(Path(ref["path"]).read_bytes())
        anchor = Path(work["raw"]) / "measurement.json"
        if sha256_file(anchor) != ref["sha256"] or Path(ref["path"]) != (
            anchor.parent / "operands" / ref["sha256"].split(":")[1] / anchor.name
        ):
            raise ValueError("invocation_measurement_anchor")
        if not all(custody.authenticate(r) for r in work["immutable_refs"]):
            raise ValueError("immutable_operand")
        for row in work["original_authority"]:
            if sha256_file(Path(row["origin_path"])) != row["reference"]["sha256"]:
                raise ValueError("original_authority")
        for manifest in work["validation"]["child_shard_manifest"]:
            for shard in manifest["shards"]:
                if (
                    custody.inventory(Path(shard["reference"]["path"]), SOURCES)
                    != shard["inventory"]
                ):
                    raise ValueError("source_inventory")
        fixture = work["fixture"]
        if not fixture["gate_check_summary"]:
            rows = fixture["chunk_rows"]
            roster = [(r["family_id"], c) for r in old.fixtures() for c in (1, 32, 128)]
            if [(r["family_id"], r["chunk_size"]) for r in rows] != roster:
                raise ValueError("original_chunk_union")
            with TemporaryDirectory(prefix="exp8642-replay-", dir="/var/tmp") as directory:
                for i, row in enumerate(old.fixtures()):
                    expected, records = old.eager(row)
                    for result in rows[i * 3 : i * 3 + 3]:
                        actual = streaming.extract(
                            row, Path(directory) / "windows.jsonl", chunk_size=result["chunk_size"]
                        )
                        if (
                            Path(result["primitive"]["path"]).read_bytes() != records
                            or result["features"] != expected
                            or any(
                                result[k] != actual[k]
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
                    progress("cold_fixture", i + 1, len(old.fixtures()) - i - 1)
        expected_value = build(
            ref,
            value["extra_validation_receipts"],
            value["adversarial_findings"],
            Path(value["publication_output"]),
        )
        if value != expected_value:
            raise ValueError("reduction_drift")
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error)[:100])
        return False
    progress("replay_passed")
    return True
