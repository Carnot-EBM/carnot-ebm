"""Deterministic evidence for the forward ARC provenance contract.

Spec: REQ-ARC-7010. This experiment exercises serialization and rejection
logic only. It invokes no model, claims no solve, and never edits historical
artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from carnot.agentic.arc_eval_provenance import (
    ARC_EVAL_PROVENANCE_REQUIRED_KEYS,
    ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
    CONSUMER_REQUIRED_KEYS,
    NO_LLM_INFERENCE_SUBSTRATE,
    NOT_APPLICABLE,
    PRODUCER_REQUIRED_KEYS,
    SOLVE_PROVENANCE_VALUES,
    ArcEvalProvenanceInput,
    build_arc_eval_provenance,
    canonical_arc_eval_provenance_bytes,
    compute_arc_eval_provenance_hash,
    validate_arc_eval_provenance,
    validate_arc_evaluation_row,
)

ARTIFACT_SCHEMA_VERSION = "carnot.experiment.arc_eval_provenance_contract.v1"
INFERENCE_SUBSTRATE = "deterministic_arc_provenance_contract_no_llm"
RANDOM_SEED = 7010
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "schema_version",
    "required_key_rows",
    "producer_wiring_rows",
    "consumer_wiring_rows",
    "rows",
    "accepted_fixture_rows",
    "rejected_fixture_rows",
    "gpu_identity_rows",
    "model_provenance_rows",
    "context_rows",
    "server_rows",
    "lease_rows",
    "completion_counter_rows",
    "policy_factory_rows",
    "solve_provenance_rows",
    "missing_field_rows",
    "alias_rejection_rows",
    "historical_artifacts_modified",
    "command_receipt_rows",
    "arc_eval_provenance_contract_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_SOURCE_PATHS = (
    "python/carnot/agentic/arc_eval_provenance.py",
    "python/carnot/agentic/arc_run_envelope.py",
    "python/carnot/agentic/arc_executable_world_model.py",
    "scripts/arc_leaderboard_eval.py",
    "scripts/outer_loop_dashboard.py",
    "python/carnot/agentic/arc_live_runner_capability_lease.py",
    "python/carnot/agentic/arc_live_runner_execution_binding.py",
    "openspec/capabilities/arc-agi/spec.md",
    "openspec/capabilities/research-reporting/spec.md",
    "results/experiment_6993_arc_producer_evidence_contract.json",
    "results/experiment_7005_arc_live_envelope_audit.json",
)
_HISTORICAL_PATHS = (
    "results/experiment_6993_arc_producer_evidence_contract.json",
    "results/experiment_7005_arc_live_envelope_audit.json",
)


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _live_input(**changes: Any) -> ArcEvalProvenanceInput:
    source = ArcEvalProvenanceInput(
        inference_substrate="local_gguf_cuda",
        gpu_uuid="GPU-00000000-0000-0000-0000-000000007010",
        gpu_model="NVIDIA RTX 3090",
        cuda_device=1,
        model_repository="unsloth/Qwen3.8-27B-GGUF",
        model_filename="Qwen3.8-27B-Q4_K_M.gguf",
        model_hash=_digest("1"),
        n_ctx=98304,
        server_binary="/opt/llama.cpp/llama-server",
        server_binary_hash=_digest("2"),
        server_command_hash=_digest("3"),
        endpoint="http://127.0.0.1:8919",
        port=8919,
        lease_id="lease-exp7010",
        lease_hash=_digest("4"),
        lease_issued_at="2026-09-05T00:00:00+00:00",
        lease_expires_at="2026-09-05T02:00:00+00:00",
        lease_checked_at="2026-09-05T01:00:00+00:00",
        request_count=3,
        completion_count=2,
        error_count=1,
        policy_hash=_digest("5"),
        factory_hash=_digest("6"),
        git_commit="7" * 40,
        solve_provenance="live_agent_self_discovery",
    )
    return replace(source, **changes)


def _no_llm_input(solve_provenance: str = "development_proxy") -> ArcEvalProvenanceInput:
    na = NOT_APPLICABLE
    return ArcEvalProvenanceInput(
        inference_substrate=NO_LLM_INFERENCE_SUBSTRATE,
        gpu_uuid=na,
        gpu_model=na,
        cuda_device=na,
        model_repository=na,
        model_filename=na,
        model_hash=na,
        n_ctx=na,
        server_binary=na,
        server_binary_hash=na,
        server_command_hash=na,
        endpoint=na,
        port=na,
        lease_id=na,
        lease_hash=na,
        lease_issued_at=na,
        lease_expires_at=na,
        lease_checked_at=na,
        request_count=0,
        completion_count=0,
        error_count=0,
        policy_hash=_digest("8"),
        factory_hash=_digest("9"),
        git_commit="a" * 40,
        solve_provenance=solve_provenance,
    )


def _rehash(record: dict[str, Any]) -> None:
    record["provenance_hash"] = compute_arc_eval_provenance_hash(record)


def _rejection_row(
    fixture_id: str, field: str, mode: str, record: dict[str, Any]
) -> dict[str, Any]:
    decision = validate_arc_eval_provenance(record)
    return {
        "fixture_id": fixture_id,
        "field": field,
        "mode": mode,
        "record": record,
        "accepted": decision.valid,
        "errors": list(decision.errors),
    }


def _missing_rows(accepted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field in ARC_EVAL_PROVENANCE_REQUIRED_KEYS:
        record = copy.deepcopy(accepted)
        del record[field]
        if field != "provenance_hash":
            _rehash(record)
        rows.append(_rejection_row(f"missing_{field}", field, "absent", record))
    return rows


def _null_rows(accepted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field in ARC_EVAL_PROVENANCE_REQUIRED_KEYS:
        record = copy.deepcopy(accepted)
        record[field] = None
        if field != "provenance_hash":
            _rehash(record)
        rows.append(_rejection_row(f"null_{field}", field, "null", record))
    return rows


def _malformed_rows(accepted: dict[str, Any]) -> list[dict[str, Any]]:
    malformed: dict[str, Any] = {
        "schema_version": "legacy",
        "inference_substrate": "cuda_maybe",
        "gpu_uuid": "cuda:1",
        "gpu_model": "",
        "cuda_device": -1,
        "model_repository": "Qwen3.8",
        "model_filename": "/tmp/model.gguf",
        "model_hash": "md5:bad",
        "n_ctx": 0,
        "server_binary": "llama-server",
        "server_binary_hash": "bad",
        "server_command_hash": "bad",
        "endpoint": "http://127.0.0.1",
        "port": 70000,
        "lease_id": "",
        "lease_hash": "bad",
        "lease_issued_at": "yesterday",
        "lease_expires_at": "tomorrow",
        "lease_checked_at": "now",
        "request_count": -1,
        "completion_count": -1,
        "error_count": -1,
        "policy_hash": "bad",
        "factory_hash": "bad",
        "git_commit": "short",
        "solve_provenance": "self_discovery",
        "provenance_hash": "bad",
    }
    rows = []
    for field in ARC_EVAL_PROVENANCE_REQUIRED_KEYS:
        record = copy.deepcopy(accepted)
        record[field] = malformed[field]
        if field != "provenance_hash":
            _rehash(record)
        rows.append(_rejection_row(f"malformed_{field}", field, "malformed", record))
    return rows


def _alias_rows(accepted: dict[str, Any]) -> list[dict[str, Any]]:
    aliases = {
        "gpu_uuid": "gpu_id",
        "gpu_model": "gpu_name",
        "cuda_device": "cuda_index",
        "model_repository": "model_repo",
        "model_filename": "model_path",
        "model_hash": "model_sha256",
        "n_ctx": "context_length",
        "server_binary": "server_path",
        "lease_id": "lease",
        "completion_count": "completions",
        "policy_hash": "policy_sha256",
        "factory_hash": "factory_sha256",
        "git_commit": "commit",
        "solve_provenance": "solve_origin",
    }
    rows = []
    for field, alias in aliases.items():
        record = copy.deepcopy(accepted)
        record[alias] = record.pop(field)
        _rehash(record)
        row = _rejection_row(f"alias_{field}_as_{alias}", field, "aliased", record)
        row["alias"] = alias
        rows.append(row)
    return rows


def _contradiction_rows(accepted: dict[str, Any]) -> list[dict[str, Any]]:
    cases = (
        ("partial_request", "completion_count", 1),
        ("port_reuse", "endpoint", "http://127.0.0.1:8920"),
        ("stale_lease", "lease_checked_at", "2026-09-05T03:00:00+00:00"),
    )
    rows = []
    for fixture_id, field, value in cases:
        record = copy.deepcopy(accepted)
        record[field] = value
        _rehash(record)
        rows.append(_rejection_row(fixture_id, field, "contradictory", record))
    no_llm = build_arc_eval_provenance(_no_llm_input())
    no_llm["model_filename"] = "false-claim.gguf"
    _rehash(no_llm)
    rows.append(_rejection_row("no_llm_gguf_claim", "model_filename", "contradictory", no_llm))
    return rows


def _fresh_process_receipt(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    code = (
        "import json,sys; from carnot.agentic.arc_eval_provenance import "
        "compute_arc_eval_provenance_hash as h; print(h(json.loads(sys.stdin.read())))"
    )
    command = [sys.executable, "-c", code]
    completed = subprocess.run(
        command,
        cwd=root,
        input=json.dumps(record),
        text=True,
        capture_output=True,
        check=False,
    )
    observed = completed.stdout.strip()
    expected = record["provenance_hash"]
    return {
        "check": "fresh_process_stable_hash",
        "command": command,
        "returncode": completed.returncode,
        "expected": expected,
        "observed": observed,
        "passed": completed.returncode == 0 and observed == expected,
        "stderr": completed.stderr.strip(),
    }


def _principles() -> dict[str, str]:
    general = {
        field: f"{field} is explicit so absence cannot be mistaken for measured evidence."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    general.update(
        {
            "historical_artifacts_modified": "Forward schemas preserve historical evidence bytes.",
            "verifier_is_oracle": "Schema validation checks evidence shape, not ARC truth.",
            "verdict_class": "Verdict class distinguishes positive evidence from blocked evidence.",
            "honest_verdict": "The terminal prefix makes the reported class machine-checkable.",
            "arc_eval_provenance_contract_ready_score": (
                "Readiness is one only when producer, consumer, rejection, round-trip, and row gates pass."
            ),
        }
    )
    return general


def _preconditions(root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in _SOURCE_PATHS:
        path = root / relative
        rows.append(
            {
                "check": f"readable:{relative}",
                "expected": True,
                "observed": path.is_file() and os.access(path, os.R_OK),
                "passed": path.is_file() and os.access(path, os.R_OK),
            }
        )
    writable_paths = (
        root / "python/carnot/agentic/arc_eval_provenance.py",
        root / "tests/python",
        root / "scripts/experiments",
        root / "results",
    )
    for path in writable_paths:
        target = path if path.is_dir() else path.parent
        passed = target.is_dir() and os.access(target, os.W_OK)
        rows.append(
            {
                "check": f"writable:{path.relative_to(root)}",
                "expected": True,
                "observed": passed,
                "passed": passed,
            }
        )
    return rows


def _checksum(artifact: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_artifact(root: Path, *, execution_date: str) -> dict[str, Any]:
    """Build the complete positive or blocked Exp7010 artifact in memory."""

    started = time.monotonic()
    preconditions = _preconditions(root)
    historical_before = {
        relative: _hash_file(root / relative)
        for relative in _HISTORICAL_PATHS
        if (root / relative).is_file()
    }
    accepted = build_arc_eval_provenance(_live_input())
    accepted_decision = validate_arc_eval_provenance(json.loads(json.dumps(accepted)))
    no_llm = build_arc_eval_provenance(_no_llm_input())
    solve_rows = []
    evaluation_rows = []
    for solve_provenance in sorted(SOLVE_PROVENANCE_VALUES):
        record = build_arc_eval_provenance(_live_input(solve_provenance=solve_provenance))
        row = {
            "row_id": f"solve_{solve_provenance}",
            "solve_provenance": solve_provenance,
            "arc_eval_provenance": record,
        }
        decision = validate_arc_evaluation_row(row)
        evaluation_rows.append(row)
        solve_rows.append(
            {
                "solve_provenance": solve_provenance,
                "accepted": decision.valid,
                "headline_eligible": decision.headline_eligible,
            }
        )
    evaluation_rows.append(
        {
            "row_id": "explicit_no_llm",
            "solve_provenance": no_llm["solve_provenance"],
            "arc_eval_provenance": no_llm,
        }
    )

    missing = _missing_rows(accepted)
    nulls = _null_rows(accepted)
    malformed = _malformed_rows(accepted)
    aliases = _alias_rows(accepted)
    contradictions = _contradiction_rows(accepted)
    rejected = missing + nulls + malformed + aliases + contradictions
    fresh_receipt = _fresh_process_receipt(root, accepted)
    historical_after = {
        relative: _hash_file(root / relative)
        for relative in _HISTORICAL_PATHS
        if (root / relative).is_file()
    }
    historical_modified = historical_before != historical_after

    producer_source = (root / "scripts/arc_leaderboard_eval.py").read_text(errors="replace")
    consumer_source = (root / "scripts/outer_loop_dashboard.py").read_text(errors="replace")
    producer_wired = (
        "build_arc_eval_provenance_for_policy" in producer_source
        and '"arc_eval_provenance"' in producer_source
        and '"solve_provenance": _solve_provenance' in producer_source
    )
    consumer_wired = (
        "validate_arc_evaluation_row" in consumer_source
        and '"arc_eval_provenance"' in consumer_source
        and "headline_eligible" in consumer_source
    )
    gate_rows = [
        {
            "check": "preconditions",
            "expected": True,
            "observed": all(row["passed"] for row in preconditions),
            "passed": all(row["passed"] for row in preconditions),
        },
        {
            "check": "producer_consumer_same_schema",
            "expected": list(ARC_EVAL_PROVENANCE_REQUIRED_KEYS),
            "observed": list(PRODUCER_REQUIRED_KEYS)
            if PRODUCER_REQUIRED_KEYS == CONSUMER_REQUIRED_KEYS
            else {
                "producer": list(PRODUCER_REQUIRED_KEYS),
                "consumer": list(CONSUMER_REQUIRED_KEYS),
            },
            "passed": PRODUCER_REQUIRED_KEYS == CONSUMER_REQUIRED_KEYS,
        },
        {
            "check": "producer_wired",
            "expected": True,
            "observed": producer_wired,
            "passed": producer_wired,
        },
        {
            "check": "consumer_wired",
            "expected": True,
            "observed": consumer_wired,
            "passed": consumer_wired,
        },
        {
            "check": "accepted_fixture_round_trip",
            "expected": True,
            "observed": accepted_decision.valid,
            "passed": accepted_decision.valid,
        },
        {
            "check": "all_rejection_fixtures_fail_closed",
            "expected": 0,
            "observed": sum(row["accepted"] for row in rejected),
            "passed": all(not row["accepted"] for row in rejected),
        },
        {
            "check": "solve_provenance_on_every_evaluation_row",
            "expected": len(evaluation_rows),
            "observed": sum("solve_provenance" in row for row in evaluation_rows),
            "passed": all("solve_provenance" in row for row in evaluation_rows),
        },
        {
            "check": "fresh_process_stable_hash",
            "expected": accepted["provenance_hash"],
            "observed": fresh_receipt["observed"],
            "passed": fresh_receipt["passed"],
        },
        {
            "check": "historical_artifacts_unchanged",
            "expected": False,
            "observed": historical_modified,
            "passed": not historical_modified,
        },
    ]
    ready = int(all(row["passed"] for row in gate_rows))
    verdict_class = "positive" if ready else "blocked"
    artifact: dict[str, Any] = {
        "field_principles": _principles(),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": {
            relative: _hash_file(root / relative)
            for relative in _SOURCE_PATHS
            if (root / relative).is_file()
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "required_key_rows": [
            {
                "index": index,
                "field": field,
                "producer_required": field in PRODUCER_REQUIRED_KEYS,
                "consumer_required": field in CONSUMER_REQUIRED_KEYS,
            }
            for index, field in enumerate(ARC_EVAL_PROVENANCE_REQUIRED_KEYS)
        ],
        "producer_wiring_rows": [
            {
                "producer": "scripts/arc_leaderboard_eval.py:run_game",
                "builder": "build_arc_eval_provenance_for_policy",
                "persisted_key": "arc_eval_provenance",
                "wired": producer_wired,
            }
        ],
        "consumer_wiring_rows": [
            {
                "consumer": "scripts/outer_loop_dashboard.py:generalization_levels",
                "validator": "validate_arc_evaluation_row",
                "eligibility_key": "headline_eligible",
                "wired": consumer_wired,
            }
        ],
        "rows": evaluation_rows,
        "accepted_fixture_rows": [
            {
                "fixture_id": "deterministic_live_cuda",
                "record": accepted,
                "accepted": accepted_decision.valid,
                "round_trip": json.loads(json.dumps(accepted)) == accepted,
            },
            {"fixture_id": "explicit_no_llm", "record": no_llm, "accepted": True},
        ],
        "rejected_fixture_rows": rejected,
        "gpu_identity_rows": [
            {
                "fixture_id": "deterministic_live_cuda",
                "gpu_uuid": accepted["gpu_uuid"],
                "gpu_model": accepted["gpu_model"],
                "cuda_device": accepted["cuda_device"],
            },
            {
                "fixture_id": "explicit_no_llm",
                "gpu_uuid": no_llm["gpu_uuid"],
                "gpu_model": no_llm["gpu_model"],
                "cuda_device": no_llm["cuda_device"],
            },
        ],
        "model_provenance_rows": [
            {key: accepted[key] for key in ("model_repository", "model_filename", "model_hash")}
        ],
        "context_rows": [{"n_ctx": accepted["n_ctx"]}, {"n_ctx": no_llm["n_ctx"]}],
        "server_rows": [
            {
                key: accepted[key]
                for key in (
                    "server_binary",
                    "server_binary_hash",
                    "server_command_hash",
                    "endpoint",
                    "port",
                )
            }
        ],
        "lease_rows": [
            {
                key: accepted[key]
                for key in (
                    "lease_id",
                    "lease_hash",
                    "lease_issued_at",
                    "lease_expires_at",
                    "lease_checked_at",
                )
            }
        ],
        "completion_counter_rows": [
            {key: accepted[key] for key in ("request_count", "completion_count", "error_count")},
            {key: no_llm[key] for key in ("request_count", "completion_count", "error_count")},
        ],
        "policy_factory_rows": [
            {key: accepted[key] for key in ("policy_hash", "factory_hash", "git_commit")}
        ],
        "solve_provenance_rows": solve_rows,
        "missing_field_rows": [
            {
                "fixture_id": row["fixture_id"],
                "field": row["field"],
                "accepted": row["accepted"],
                "errors": row["errors"],
            }
            for row in missing
        ],
        "alias_rejection_rows": aliases,
        "historical_artifacts_modified": historical_modified,
        "command_receipt_rows": [fresh_receipt],
        "arc_eval_provenance_contract_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_rows,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": f"{verdict_class}_arc_eval_provenance_contract_"
        + ("ready" if ready else "blocked"),
        "execution_date": execution_date,
        "contract_schema_version": ARC_EVAL_PROVENANCE_SCHEMA_VERSION,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Validate the emitted evidence independently from its stored score."""

    if not isinstance(artifact, dict):
        return ["artifact must be an object"]
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if errors:
        return [f"missing artifact field: {field}" for field in errors]
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"])
    if missing_principles:
        errors.append("missing field principles: " + ",".join(sorted(missing_principles)))
    for row in artifact["accepted_fixture_rows"]:
        if not validate_arc_eval_provenance(row["record"]).valid:
            errors.append(f"accepted fixture rejected: {row['fixture_id']}")
    for row in artifact["rejected_fixture_rows"]:
        if validate_arc_eval_provenance(row["record"]).valid or row["accepted"]:
            errors.append(f"rejection fixture accepted: {row['fixture_id']}")
    if any(not validate_arc_evaluation_row(row).valid for row in artifact["rows"]):
        errors.append("an evaluation row lacks valid solve provenance")
    expected_ready = int(all(row["passed"] for row in artifact["gate_check_summary"]))
    if artifact["arc_eval_provenance_contract_ready_score"] != expected_ready:
        errors.append("ready score disagrees with gates")
    expected_class = "positive" if expected_ready else "blocked"
    if artifact["verdict_class"] != expected_class:
        errors.append("verdict_class disagrees with gates")
    if not artifact["honest_verdict"].startswith(expected_class + "_"):
        errors.append("honest_verdict prefix disagrees with verdict_class")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact["historical_artifacts_modified"] is not False:
        errors.append("historical artifacts were modified")
    if artifact["reproducibility_checksum"] != _checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(root: Path, *, execution_date: str) -> Path:
    """Write one atomically published artifact after self-validation."""

    artifact = build_artifact(root, execution_date=execution_date)
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise ValueError("invalid Exp7010 artifact: " + "; ".join(validation_errors))
    out = root / "results/experiment_7010_arc_eval_provenance_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    out = write_artifact(root, execution_date=args.date)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
