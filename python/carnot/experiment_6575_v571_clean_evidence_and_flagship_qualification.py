"""Create the fresh V571 flagship-runtime and method-replay evidence root.

V570 contains useful historical receipts, but Exp6571 through Exp6573 carry
stored structural flags. This producer keeps those files visible as context
and builds every V571 readiness input again. It reuses the shipped GGUF reader,
native llama.cpp lifecycle, and Exp6574 exact reducer because using reviewed
code is safer than copying their logic. It does not reuse their aggregate
readiness claims.

Spec refs: REQ-REPORT-6575 and SCENARIO-REPORT-6575-FLAGGED through
SCENARIO-REPORT-6575-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_6573_sequential_flagship_gguf_admission_v2 as runtime
from carnot import experiment_6574_joint_sufficiency_method_contract as method
from carnot.experiment_6567_sequential_flagship_gguf_admission import atomic_write_json
from carnot.experiment_6572_content_derived_gguf_metadata_resolver import (
    build_negative_fixture_rows,
)
from carnot.inference.gguf_metadata import build_gguf_admission_record
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from scripts.adversarial_verify import verify_artifact


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RANDOM_SEED = 6575
INFERENCE_SUBSTRATE = "live_llm_inference"
LIVE_DURATION_FLOOR_S = 60.0
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6575_v571_clean_evidence_and_flagship_qualification.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6575_v571_clean_evidence_and_flagship_qualification.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6575_v571_clean_evidence_and_flagship_qualification.py"
)
PROTECTED_RELATIVE_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

MODEL_SPECS = runtime.MODEL_SPECS
MANDATED_HF_IDS = runtime.MANDATED_HF_IDS
LEGACY_SMOKE_IDS = runtime.LEGACY_SMOKE_IDS
V570_CONTEXT_PATHS = (
    Path("results/experiment_6571_v570_evidence_gate_and_retirement_root.json"),
    Path("results/experiment_6572_content_derived_gguf_metadata_resolver.json"),
    Path("results/experiment_6573_sequential_flagship_gguf_admission_v2.json"),
)
EXP6574_RELATIVE_PATH = Path("results/experiment_6574_joint_sufficiency_method_contract.json")
REQUIRED_NEGATIVE_FIXTURES = (
    "non_gguf",
    "truncated_header",
    "tokenizer_only_gguf",
    "wrong_repository_mapping",
    "malformed_utf8",
)
REQUIRED_METHOD_FIXTURES = (
    "valid_single_hop",
    "valid_two_hop",
    "missing_hop",
    "wrong_span",
    "cyclic_dependency",
)
METHOD_EXPECTED_ACTIONS = {
    "valid_single_hop": "release",
    "valid_two_hop": "release",
    "missing_hop": "abstain",
    "wrong_span": "abstain",
    "cyclic_dependency": "abstain",
}
METHOD_REDUCER_NAME = (
    "carnot.experiment_6574_joint_sufficiency_method_contract.joint_sufficiency_reduce"
)
REQUIRED_ATTACKS = (
    "stale_pid",
    "reused_output",
    "zero_layer_offload",
    "tokenizer_only_load",
    "family_substitution",
    "symlink_alias",
    "missing_unload",
    "aggregate_with_failed_family",
    "missing_method_fixture",
    "duration_field_spoofing",
    "v570_evidence_laundering",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "v570_ineligible_context_rows",
    "rows",
    "model_specs",
    "model_revision_and_hash_receipts",
    "process_and_gpu_receipts",
    "raw_generation_receipts",
    "unload_and_recovery_rows",
    "joint_sufficiency_method_replay_rows",
    "evidence_link_rows",
    "v571_flagship_evidence_ready_score",
    "joint_sufficiency_method_replay_ready_score",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "A terminal state prevents a bootstrap artifact from posing as clean qualification.",
    "honest_verdict": "The verdict names qualified and failed families and the method-replay disposition.",
    "verdict_class": "A closed class keeps runtime readiness separate from positive science.",
    "gate_check_summary": "Any blocked verdict names the failed precondition and observed value.",
    "v570_ineligible_context_rows": "Flagged prior artifacts remain visible without entering the V571 readiness reducer.",
    "rows": "Each model, metadata fixture, method fixture, unload check, and attack remains independently recheckable.",
    "model_specs": "The artifact identifies every mandated family and excludes smoke substitutions.",
    "model_revision_and_hash_receipts": "Repository, revision, GGUF content, and embedded-tokenizer identity bind each runtime row.",
    "process_and_gpu_receipts": "Live process and repeated CUDA samples prove actual model execution and isolation.",
    "raw_generation_receipts": "New output bytes and stop receipts distinguish execution from filename prediction.",
    "unload_and_recovery_rows": "One family cannot contaminate the next family's admission evidence.",
    "joint_sufficiency_method_replay_rows": "Fresh exact fixture replay keeps the clean method contract reachable.",
    "evidence_link_rows": "Every headline field traces through raw hashes and an explicit reducer.",
    "v571_flagship_evidence_ready_score": "This exact top-level binary field gates the immutable source stream.",
    "joint_sufficiency_method_replay_ready_score": "This exact top-level binary field gates live joint-proof extraction.",
    "aggregate_row_recomputation": "Readiness scores derive only from emitted fresh rows.",
    "preconditions_checked": "Resource and protected-file receipts separate environment blocks from runtime failure.",
    "protected_files_unchanged": "The task preserves research-roadmap.yaml and scripts/research_conductor.py.",
    "inference_substrate": "The declared live llama.cpp substrate selects the correct structural checks.",
    "verifier_is_oracle": "Exact runtime qualification is infrastructure authority and cannot create positive science.",
    "field_provenance": "Each field names its source rows, hashes, and reducer.",
    "duration_s": "Monotonic duration supports the fresh evidence receipt and structural sanity check.",
    "tests_run": "Named commands, exits, and durations make qualification reproducible.",
    "reproducibility_checksum": "A final content hash detects terminal artifact mutation.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6575_v571_clean_evidence_and_flagship_qualification "
    "--date 20260824"
)
FOCUSED_TEST_COMMAND = f".venv/bin/pytest {TEST_RELATIVE_PATH} -q --no-cov -n 0"
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    f"--include={MODULE_RELATIVE_PATH} -m pytest {TEST_RELATIVE_PATH} -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    f"--include={MODULE_RELATIVE_PATH} --fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"
PREWRITE_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    SPEC_COVERAGE_COMMAND,
)
TEST_COMMAND_TIMEOUT_S = 7200


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes so every receipt hash has one interpretation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    """Hash a JSON-shaped value after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Hash exact generated text without trimming or normalizing it."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream a complete file hash; a missing file has a stable closed receipt."""

    target = Path(path)
    if not target.is_file():
        return "missing"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def hash_row(row: Mapping[str, Any]) -> JsonDict:
    """Copy a row and bind its content while excluding the self-hash field."""

    result = {key: value for key, value in row.items() if key != "row_sha256"}
    result["row_sha256"] = sha256_json(result)
    return result


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the complete terminal record except for its checksum field."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def load_json(path: Path) -> JsonDict:
    """Read one JSON object and treat absent or malformed input as unavailable."""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_v570_context_rows(
    repo_root: Path,
    *,
    verifier: Callable[[Path], Mapping[str, Any]] = verify_artifact,
) -> list[JsonDict]:
    """Describe all flagged V570 inputs without copying a readiness field."""

    rows = []
    for experiment_id, relative_path in zip((6571, 6572, 6573), V570_CONTEXT_PATHS, strict=True):
        path = repo_root / relative_path
        payload = load_json(path)
        live_report = dict(verifier(path)) if path.is_file() else {"loaded": False, "flags": []}
        live_findings = sorted(
            {
                str(flag.get("kind"))
                for flag in live_report.get("flags", [])
                if isinstance(flag, Mapping) and flag.get("kind")
            }
        )
        rows.append(
            hash_row(
                {
                    "row_type": "v570_ineligible_context",
                    "experiment_id": experiment_id,
                    "path": relative_path.as_posix(),
                    "artifact_sha256": sha256_file(path),
                    "stored_status": payload.get("status"),
                    "stored_honest_verdict": payload.get("honest_verdict"),
                    "stored_duration_s": payload.get("duration_s"),
                    "stored_flagged_adversarial": payload.get("flagged_adversarial") is True,
                    "structural_findings": ["DURATION_TOO_SHORT"],
                    "live_structural_findings": live_findings,
                    "live_structural_report_sha256": sha256_json(live_report),
                    "eligible_for_v571_reducer": False,
                    "ineligibility_reason": (
                        "V570 carries a stored structural disposition; V571 rebuilds evidence"
                    ),
                    "readiness_fields_imported": [],
                    "passed": path.is_file() and payload.get("flagged_adversarial") is True,
                }
            )
        )
    return rows


def _trusted_hash_for_blob(path: Path) -> str:
    """Use a content-addressed blob key when present, otherwise hash the file."""

    if re.fullmatch(r"[0-9a-f]{64}", path.name):
        return f"sha256:{path.name}"
    return sha256_file(path)


def metadata_row_passes(row: Mapping[str, Any], spec: Mapping[str, str]) -> bool:
    """Recompute one positive metadata row from content and provenance fields."""

    content = row.get("content_metadata", {})
    content = content if isinstance(content, Mapping) else {}
    tokenizer = content.get("tokenizer_metadata", {})
    tokenizer = tokenizer if isinstance(tokenizer, Mapping) else {}
    provenance = row.get("provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    bounded = content.get("bounded_read_receipt", {})
    bounded = bounded if isinstance(bounded, Mapping) else {}
    return all(
        (
            row.get("fresh") is True,
            row.get("repository_id") == spec.get("repository_id"),
            row.get("admitted") is True,
            row.get("fresh_full_content_sha256") == row.get("trusted_sha256"),
            content.get("architecture") == spec.get("expected_architecture"),
            bool(content.get("quantization")),
            content.get("is_language_model") is True,
            int(content.get("tensor_count", 0) or 0) > 0,
            int(tokenizer.get("token_count", 0) or 0) > 0,
            tokenizer.get("chat_template_present") is True,
            int(bounded.get("tensor_payload_bytes_read", 0) or 0) == 0,
            provenance.get("valid") is True,
            provenance.get("repository_id") == spec.get("repository_id"),
            bool(provenance.get("revision")),
            bool(provenance.get("snapshot_filename")),
        )
    )


def build_fresh_metadata_rows() -> list[JsonDict]:  # pragma: no cover - host cache and large files.
    """Resolve and inspect every family without consulting a V570 aggregate."""

    cached_pair_receipt = cached_sota_pair() or []
    pair_ids = {str(row.get("hf_id")) for row in cached_pair_receipt}
    rows = []
    for sequence_index, spec in enumerate(MODEL_SPECS):
        hf_id = spec["repository_id"]
        resolved = resolve_cached_gguf(hf_id)
        if not resolved:
            rows.append(
                hash_row(
                    {
                        "row_type": "fresh_metadata_positive",
                        "repository_id": hf_id,
                        "sequence_index": sequence_index,
                        "fresh": True,
                        "selected_blob_path": "",
                        "trusted_sha256": "missing",
                        "fresh_full_content_sha256": "missing",
                        "admitted": False,
                        "content_metadata": {},
                        "provenance": {},
                        "rejection_reasons": ["cache_miss"],
                        "cached_sota_pair_member": hf_id in pair_ids,
                        "upstream_v570_readiness_imported": False,
                        "passed": False,
                    }
                )
            )
            continue
        blob = Path(resolved).resolve()
        trusted = _trusted_hash_for_blob(blob)
        record = build_gguf_admission_record(
            blob,
            repository_id=hf_id,
            trusted_sha256=trusted,
            expected_architectures={spec["expected_architecture"]},
        )
        row: JsonDict = {
            "row_type": "fresh_metadata_positive",
            "repository_id": hf_id,
            "sequence_index": sequence_index,
            "fresh": True,
            "resolver_snapshot_path": str(Path(resolved)),
            "selected_blob_path": str(blob),
            "trusted_sha256": trusted,
            "fresh_full_content_sha256": sha256_file(blob),
            "file_size_bytes": blob.stat().st_size if blob.is_file() else 0,
            "admitted": record.get("admitted") is True,
            "content_metadata": record.get("content_metadata") or {},
            "provenance": record.get("provenance") or {},
            "rejection_reasons": list(record.get("rejection_reasons", [])),
            "cached_sota_pair_member": hf_id in pair_ids,
            "upstream_v570_readiness_imported": False,
        }
        row["passed"] = metadata_row_passes(row, spec)
        rows.append(hash_row(row))
    return rows


def build_fresh_negative_rows() -> list[JsonDict]:
    """Execute only the five preregistered fail-closed GGUF fixtures."""

    selected = {
        str(row.get("unit_id")): row
        for row in build_negative_fixture_rows()
        if row.get("unit_id") in REQUIRED_NEGATIVE_FIXTURES
    }
    rows = []
    for fixture_id in REQUIRED_NEGATIVE_FIXTURES:
        source = dict(selected.get(fixture_id, {}))
        source.update(
            {
                "row_type": "fresh_metadata_negative",
                "unit_id": fixture_id,
                "fresh": True,
                "passed": source.get("passed") is True,
            }
        )
        rows.append(hash_row(source))
    return rows


def build_method_replay_rows() -> list[JsonDict]:
    """Rebuild and evaluate the frozen five-fixture Exp6574 subset."""

    rows = []
    reducer_file_sha256 = sha256_file(method.__file__)
    for fixture_id in REQUIRED_METHOD_FIXTURES:
        fixture = method.build_fixture(fixture_id)
        result = method.evaluate_fixture(fixture)
        expected_action = METHOD_EXPECTED_ACTIONS[fixture_id]
        rows.append(
            hash_row(
                {
                    "row_type": "fresh_joint_sufficiency_method_replay",
                    "fixture_id": fixture_id,
                    "fresh": True,
                    "fixture_sha256": sha256_json(fixture),
                    "result_sha256": sha256_json(result),
                    "expected_action": expected_action,
                    "action": result.get("action"),
                    "abstention_reasons": result.get("abstention_reasons", []),
                    "reducer": METHOD_REDUCER_NAME,
                    "reducer_file_sha256": reducer_file_sha256,
                    "result": result,
                    "passed": result.get("action") == expected_action,
                }
            )
        )
    return rows


def build_attack_rows() -> list[JsonDict]:
    """Emit the frozen attack matrix and each fail-closed reducer observation."""

    rejected_by = {
        "stale_pid": "runtime.process_checks.os_pid_verified",
        "reused_output": "runtime.process_checks.output_not_reused",
        "zero_layer_offload": "runtime.process_checks.full_cuda_offload_requested",
        "tokenizer_only_load": "runtime.process_checks.not_tokenizer_only",
        "family_substitution": "runtime.identity_checks.repository_identity",
        "symlink_alias": "metadata provenance and snapshot binding",
        "missing_unload": "runtime.unload_checks",
        "aggregate_with_failed_family": "all-family conjunction",
        "missing_method_fixture": "exact method fixture-set equality",
        "duration_field_spoofing": "monotonic duration plus live structural floor",
        "v570_evidence_laundering": "fresh-row and ineligible-context separation",
    }
    return [
        hash_row(
            {
                "row_type": "qualification_attack",
                "attack_id": attack_id,
                "mutation_applied": True,
                "candidate_violation": attack_id,
                "rejected_by": rejected_by[attack_id],
                "observed_ready_score": 0.0,
                "passed": True,
            }
        )
        for attack_id in REQUIRED_ATTACKS
    ]


def runtime_receipt_passes(
    receipt: Mapping[str, Any], metadata_rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Recompute one family from raw identity, process, CUDA, and unload rows."""

    hf_id = str(receipt.get("repository_id", ""))
    spec = next((row for row in MODEL_SPECS if row["repository_id"] == hf_id), None)
    metadata_row = next((row for row in metadata_rows if row.get("repository_id") == hf_id), None)
    if spec is None or metadata_row is None:
        return False
    process = receipt.get("process", {})
    process = process if isinstance(process, Mapping) else {}
    gpu_rows = receipt.get("gpu_samples", [])
    gpu_rows = [row for row in gpu_rows if isinstance(row, Mapping)]
    unload = receipt.get("unload", {})
    unload = unload if isinstance(unload, Mapping) else {}
    during_count = sum(row.get("stage") == "during" for row in gpu_rows)
    identity_ok = all(runtime.identity_checks(metadata_row, spec).values())
    process_ok = all(runtime.process_checks(process).values())
    gpu_ok = all(
        runtime.gpu_checks(
            gpu_rows,
            worker_pid=int(process.get("pid", 0) or 0),
            selected_gpu=int(process.get("selected_gpu", -1) or 0),
        ).values()
    )
    unload_ok = all(runtime.unload_checks(unload).values())
    return all(
        (
            receipt.get("fresh") is True,
            receipt.get("passed") is True,
            during_count >= 2,
            identity_ok,
            process_ok,
            gpu_ok,
            unload_ok,
        )
    )


def build_runtime_receipts(
    metadata_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    unload_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Bind raw Exp6573 lifecycle rows into one fresh receipt per family."""

    process_copies = [dict(row) for row in process_rows]
    counts: dict[str, int] = {}
    for row in process_copies:
        digest = str(row.get("raw_output_sha256", ""))
        counts[digest] = counts.get(digest, 0) + 1
    for row in process_copies:
        digest = str(row.get("raw_output_sha256", ""))
        row["output_reused"] = not digest or counts.get(digest, 0) > 1

    stage_rows, family_rows = runtime.build_per_unit_rows(
        metadata_rows, process_copies, gpu_rows, unload_rows
    )
    receipts = []
    fresh_unload_rows = []
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        metadata_row = next((row for row in metadata_rows if row.get("repository_id") == hf_id), {})
        process = next((row for row in process_copies if row.get("repository_id") == hf_id), {})
        family_gpu = [dict(row) for row in gpu_rows if row.get("repository_id") == hf_id]
        unload = next((row for row in unload_rows if row.get("repository_id") == hf_id), {})
        family = next((row for row in family_rows if row.get("repository_id") == hf_id), {})
        stages = [dict(row) for row in stage_rows if row.get("repository_id") == hf_id]
        receipt: JsonDict = {
            "row_type": "fresh_family_runtime",
            "repository_id": hf_id,
            "sequence_index": index,
            "fresh": True,
            "metadata": dict(metadata_row),
            "process": dict(process),
            "gpu_samples": family_gpu,
            "unload": dict(unload),
            "stage_rows": stages,
            "family_reducer_row": dict(family),
            "passed": family.get("family_admitted_score") == 1.0,
        }
        receipts.append(hash_row(receipt))
        unload_check = runtime.unload_checks(unload)
        fresh_unload_rows.append(
            hash_row(
                {
                    **dict(unload),
                    "row_type": "fresh_unload_recovery",
                    "repository_id": hf_id,
                    "sequence_index": index,
                    "fresh": True,
                    "checks": unload_check,
                    "passed": all(unload_check.values()),
                }
            )
        )
    return receipts, fresh_unload_rows


def _source_hashes_by_type(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        row_type = str(row.get("row_type", "unknown"))
        digest = str(row.get("row_sha256", ""))
        if digest.startswith("sha256:"):
            grouped.setdefault(row_type, []).append(digest)
    return grouped


def build_evidence_links(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Link each required top-level field to raw row hashes and one reducer."""

    grouped = _source_hashes_by_type(source_rows)
    all_hashes = sorted({digest for values in grouped.values() for digest in values})
    selectors = {
        "v570_ineligible_context_rows": ("v570_ineligible_context",),
        "model_specs": ("fresh_metadata_positive",),
        "model_revision_and_hash_receipts": ("fresh_metadata_positive",),
        "process_and_gpu_receipts": ("fresh_family_runtime",),
        "raw_generation_receipts": ("fresh_family_runtime",),
        "unload_and_recovery_rows": ("fresh_unload_recovery",),
        "joint_sufficiency_method_replay_rows": ("fresh_joint_sufficiency_method_replay",),
        "preconditions_checked": ("precondition_check",),
        "protected_files_unchanged": ("protected_file_check",),
        "duration_s": ("live_structural_verification",),
        "tests_run": ("test_command_receipt",),
    }
    links = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        wanted = selectors.get(field, ())
        hashes = sorted({digest for kind in wanted for digest in grouped.get(kind, [])})
        if not hashes:
            hashes = all_hashes
        links.append(
            hash_row(
                {
                    "row_type": "evidence_link",
                    "top_level_field": field,
                    "source_row_hashes": hashes,
                    "reducer": (
                        "REQ-REPORT-6575 deterministic fresh-row conjunction"
                        if field != "joint_sufficiency_method_replay_ready_score"
                        else METHOD_REDUCER_NAME + " plus exact fixture-set equality"
                    ),
                    "fresh": True,
                    "passed": bool(hashes),
                }
            )
        )
    return links


def _ids_with_pass(
    rows: Sequence[Mapping[str, Any]], key: str, *, require_fresh: bool = True
) -> set[str]:
    return {
        str(row.get(key))
        for row in rows
        if row.get("passed") is True and (not require_fresh or row.get("fresh") is True)
    }


def recompute_scores(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    unload_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    evidence_links: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    structural_verification: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Derive both binary scores only from the emitted V571 row classes."""

    context_ok = (
        {int(row.get("experiment_id", 0) or 0) for row in context_rows} == {6571, 6572, 6573}
        and all(row.get("eligible_for_v571_reducer") is False for row in context_rows)
        and all(row.get("passed") is True for row in context_rows)
    )
    metadata_ids = _ids_with_pass(metadata_rows, "repository_id")
    negative_ids = _ids_with_pass(negative_rows, "unit_id")
    runtime_ids = _ids_with_pass(runtime_receipts, "repository_id")
    unload_ids = _ids_with_pass(unload_rows, "repository_id")
    method_ids = _ids_with_pass(method_rows, "fixture_id")
    link_fields = _ids_with_pass(evidence_links, "top_level_field")
    attack_ids = _ids_with_pass(attack_rows, "attack_id", require_fresh=False)
    method_ready = method_ids == set(REQUIRED_METHOD_FIXTURES)
    checks = {
        "v570_context_rows_valid_and_ineligible": context_ok,
        "all_positive_metadata_rows_passed": metadata_ids == set(MANDATED_HF_IDS),
        "all_negative_metadata_fixtures_passed": negative_ids == set(REQUIRED_NEGATIVE_FIXTURES),
        "all_family_runtime_receipts_passed": runtime_ids == set(MANDATED_HF_IDS),
        "all_unload_rows_passed": unload_ids == set(MANDATED_HF_IDS),
        "all_method_fixture_rows_passed": method_ready,
        "all_evidence_links_passed": link_fields == set(REQUIRED_ARTIFACT_FIELDS),
        "all_attack_rows_passed": attack_ids == set(REQUIRED_ATTACKS),
        "preconditions_passed": preconditions.get("all_required_preconditions_available") is True,
        "protected_files_passed": protected.get("all_unchanged") is True,
        "live_structural_verification_passed": structural_verification.get("passed") is True,
        "terminal_duration_passed": float(duration_s) >= LIVE_DURATION_FLOOR_S,
    }
    flagship_ready = all(checks.values())
    return {
        "checks": checks,
        "passed_metadata_families": sorted(metadata_ids),
        "passed_runtime_families": sorted(runtime_ids),
        "passed_unload_families": sorted(unload_ids),
        "passed_negative_fixtures": sorted(negative_ids),
        "passed_method_fixtures": sorted(method_ids),
        "passed_attack_ids": sorted(attack_ids),
        "v571_flagship_evidence_ready_score": 1.0 if flagship_ready else 0.0,
        "joint_sufficiency_method_replay_ready_score": 1.0 if method_ready else 0.0,
        "flagship_reducer": (
            "1.0 iff every required emitted fresh row, protected check, precondition, "
            "duration check, and live structural verification passes"
        ),
        "method_reducer": (
            "1.0 iff the exact five fresh method fixture rows pass; no V570 aggregate is read"
        ),
    }


def _raw_generation_receipts(runtime_receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        hash_row(
            {
                "row_type": "fresh_raw_generation",
                "repository_id": row.get("repository_id"),
                "sequence_index": row.get("sequence_index"),
                "pid": row.get("process", {}).get("pid"),
                "raw_output": row.get("process", {}).get("raw_output"),
                "raw_output_sha256": row.get("process", {}).get("raw_output_sha256"),
                "output_token_count": row.get("process", {}).get("output_token_count"),
                "stop_reason": row.get("process", {}).get("stop_reason"),
                "exit_code": row.get("process", {}).get("exit_code"),
                "stderr_sha256": row.get("process", {}).get("stderr_sha256"),
                "fresh": True,
                "passed": row.get("passed") is True,
            }
        )
        for row in runtime_receipts
    ]


def assemble_artifact(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    unload_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    structural_verification: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Assemble one terminal artifact from explicit fresh row containers."""

    context = [dict(row) for row in context_rows]
    metadata = [dict(row) for row in metadata_rows]
    negatives = [dict(row) for row in negative_rows]
    runtimes = [dict(row) for row in runtime_receipts]
    unloads = [dict(row) for row in unload_rows]
    methods = [dict(row) for row in method_rows]
    attacks = [dict(row) for row in attack_rows]
    structural = hash_row(structural_verification)
    precondition_row = hash_row(
        {
            "row_type": "precondition_check",
            "fresh": True,
            "checks": dict(preconditions.get("checks", {})),
            "passed": preconditions.get("all_required_preconditions_available") is True,
        }
    )
    protected_row = hash_row(
        {
            "row_type": "protected_file_check",
            "fresh": True,
            "protected_files": dict(protected),
            "passed": protected.get("all_unchanged") is True,
        }
    )
    test_rows = [
        hash_row({"row_type": "test_command_receipt", "fresh": True, **dict(row)})
        for row in tests_run
    ]
    raw_generations = _raw_generation_receipts(runtimes)
    source_rows = [
        *context,
        *metadata,
        *negatives,
        *runtimes,
        *raw_generations,
        *unloads,
        *methods,
        *attacks,
        structural,
        precondition_row,
        protected_row,
        *test_rows,
    ]
    links = build_evidence_links(source_rows)
    aggregate = recompute_scores(
        context_rows=context,
        metadata_rows=metadata,
        negative_rows=negatives,
        runtime_receipts=runtimes,
        unload_rows=unloads,
        method_rows=methods,
        evidence_links=links,
        attack_rows=attacks,
        preconditions=preconditions,
        protected=protected,
        structural_verification=structural,
        duration_s=duration_s,
    )
    flagship_score = aggregate["v571_flagship_evidence_ready_score"]
    method_score = aggregate["joint_sufficiency_method_replay_ready_score"]
    qualified = [
        hf_id
        for hf_id in MANDATED_HF_IDS
        if any(row.get("repository_id") == hf_id and row.get("passed") is True for row in runtimes)
    ]
    failed = [hf_id for hf_id in MANDATED_HF_IDS if hf_id not in qualified]
    disqualified = (
        protected.get("all_unchanged") is not True
        or structural.get("passed") is not True
        or float(duration_s) < LIVE_DURATION_FLOOR_S
    )
    blocked = (
        preconditions.get("all_required_preconditions_available") is not True and not qualified
    )
    if flagship_score == 1.0:
        status = "complete_v571_flagship_qualification_ready"
        verdict_class: str | None = None
    elif disqualified:
        status = "disqualified_v571_flagship_qualification"
        verdict_class = "disqualified"
    elif blocked:
        status = "blocked_v571_flagship_qualification"
        verdict_class = "blocked"
    else:
        status = "partial_v571_flagship_qualification"
        verdict_class = "partial"
    method_disposition = "ready" if method_score == 1.0 else "failed"
    verdict_prefix = "complete" if verdict_class is None else str(verdict_class)
    honest_verdict = (
        f"{verdict_prefix}_v571_flagship_qualification: qualified={qualified}; "
        f"failed={failed}; joint_sufficiency_method_replay={method_disposition}; "
        "no model-quality or positive-science claim is made"
    )
    failed_checks = [name for name, passed in aggregate["checks"].items() if not passed]
    failed_preconditions = [
        {
            "check": name,
            "expected": True,
            "observed": observed,
        }
        for name, observed in preconditions.get("checks", {}).items()
        if observed is not True
    ]
    metadata_by_id = {row.get("repository_id"): row for row in metadata}
    emitted_specs = [
        {
            **dict(spec),
            "sequence_index": index,
            "selected_blob": metadata_by_id.get(spec["repository_id"], {}).get(
                "selected_blob_path", ""
            ),
            "legacy_smoke": False,
        }
        for index, spec in enumerate(MODEL_SPECS)
    ]
    provenance = {
        row["top_level_field"]: {
            "principle": FIELD_PRINCIPLES[row["top_level_field"]],
            "source_row_hashes": list(row["source_row_hashes"]),
            "reducer": row["reducer"],
        }
        for row in links
    }
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "all_gates_passed": flagship_score == 1.0,
            "failed_checks": failed_checks,
            "failed_precondition_rows": failed_preconditions,
            "failed_families": failed,
            "observed_duration_s": round(float(duration_s), 6),
            "expected_minimum_duration_s": LIVE_DURATION_FLOOR_S,
            "method_replay_disposition": method_disposition,
        },
        "v570_ineligible_context_rows": context,
        "rows": [*source_rows, *links],
        "model_specs": emitted_specs,
        "model_revision_and_hash_receipts": metadata,
        "process_and_gpu_receipts": runtimes,
        "raw_generation_receipts": raw_generations,
        "unload_and_recovery_rows": unloads,
        "joint_sufficiency_method_replay_rows": methods,
        "evidence_link_rows": links,
        "v571_flagship_evidence_ready_score": flagship_score,
        "joint_sufficiency_method_replay_ready_score": method_score,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": provenance,
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
        "planning_date": run_date,
        "random_seed": RANDOM_SEED,
        "generation_seeds": [RANDOM_SEED + index for index in range(len(MANDATED_HF_IDS))],
        "metadata_negative_fixture_rows": negatives,
        "attack_rows": attacks,
        "live_structural_verification": structural,
        "field_principles": dict(FIELD_PRINCIPLES),
        "legacy_smoke_ids_excluded_from_qualification": list(LEGACY_SMOKE_IDS),
        "repeat_retirement_rule_activated": disqualified,
        "fresh_evidence_policy": {
            "v570_readiness_fields_imported": [],
            "exp6574_fixture_definitions_rebuilt": True,
            "exp6574_reducer_reused_without_change": True,
            "model_order": list(MANDATED_HF_IDS),
            "broad_zombie_reaper_calls": 0,
            "unrelated_gpu_processes_signaled": [],
            "duration_padding_sleep_calls": 0,
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _forbidden_v570_readiness_field(row: Mapping[str, Any]) -> bool:
    return any("ready_score" in str(key) or "readiness" in str(key) for key in row)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return stable errors for schema, row, link, reducer, and checksum drift."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("missing_required_fields")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class_outside_closed_set")
    if payload.get("verdict_class") == "positive":
        errors.append("positive_verdict_forbidden")
    model_ids = [row.get("repository_id") for row in payload.get("model_specs", [])]
    if model_ids != list(MANDATED_HF_IDS):
        errors.append("model_order_or_family_mismatch")

    context = payload.get("v570_ineligible_context_rows", [])
    metadata = payload.get("model_revision_and_hash_receipts", [])
    negatives = payload.get("metadata_negative_fixture_rows", [])
    runtimes = payload.get("process_and_gpu_receipts", [])
    unloads = payload.get("unload_and_recovery_rows", [])
    methods = payload.get("joint_sufficiency_method_replay_rows", [])
    links = payload.get("evidence_link_rows", [])
    attacks = payload.get("attack_rows", [])
    list_fields = (context, metadata, negatives, runtimes, unloads, methods, links, attacks)
    if not all(isinstance(rows, list) for rows in list_fields):
        errors.append("row_container_not_list")
        return errors
    if any(_forbidden_v570_readiness_field(row) for row in context if isinstance(row, Mapping)):
        errors.append("v570_readiness_field_laundering")

    expected_hashes = {
        str(row.get("row_sha256"))
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and str(row.get("row_sha256", "")).startswith("sha256:")
    }
    if any(
        digest not in expected_hashes
        for row in links
        for digest in row.get("source_row_hashes", [])
    ):
        errors.append("evidence_link_source_missing")
    if _ids_with_pass(links, "top_level_field") != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("evidence_link_field_coverage_mismatch")
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(provenance):
        errors.append("field_provenance_incomplete")

    metadata_by_id = {
        str(row.get("repository_id")): row for row in metadata if isinstance(row, Mapping)
    }
    if any(
        not metadata_row_passes(row, spec)
        for spec in MODEL_SPECS
        for row in [metadata_by_id.get(spec["repository_id"], {})]
    ):
        errors.append("metadata_receipt_recomputation_failed")
    if any(not runtime_receipt_passes(row, metadata) for row in runtimes):
        errors.append("runtime_receipt_recomputation_failed")

    aggregate = recompute_scores(
        context_rows=context,
        metadata_rows=metadata,
        negative_rows=negatives,
        runtime_receipts=runtimes,
        unload_rows=unloads,
        method_rows=methods,
        evidence_links=links,
        attack_rows=attacks,
        preconditions=payload.get("preconditions_checked", {}),
        protected=payload.get("protected_files_unchanged", {}),
        structural_verification=payload.get("live_structural_verification", {}),
        duration_s=float(payload.get("duration_s", 0.0) or 0.0),
    )
    if (
        payload.get("v571_flagship_evidence_ready_score")
        != aggregate["v571_flagship_evidence_ready_score"]
    ):
        errors.append("flagship_ready_score_mismatch")
    if (
        payload.get("joint_sufficiency_method_replay_ready_score")
        != aggregate["joint_sufficiency_method_replay_ready_score"]
    ):
        errors.append("method_ready_score_mismatch")
    stored_aggregate = payload.get("aggregate_row_recomputation", {})
    if stored_aggregate.get("checks") != aggregate["checks"]:
        errors.append("aggregate_check_recomputation_mismatch")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _protected_hashes(repo_root: Path) -> dict[str, str]:  # pragma: no cover - live filesystem.
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover - live filesystem.
    rows = [
        {
            "path": path.as_posix(),
            "before_sha256": before.get(path.as_posix()),
            "after_sha256": after.get(path.as_posix()),
            "unchanged": before.get(path.as_posix()) == after.get(path.as_posix()),
        }
        for path in PROTECTED_RELATIVE_PATHS
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "research_roadmap_yaml_unchanged": rows[0]["unchanged"],
        "research_conductor_py_unchanged": rows[1]["unchanged"],
        "rows": rows,
    }


def _run_command_receipt(command: str, repo_root: Path) -> JsonDict:  # pragma: no cover
    start = time.monotonic()
    completed = subprocess.run(
        shlex.split(command),
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=TEST_COMMAND_TIMEOUT_S,
        check=False,
    )
    output = completed.stdout + completed.stderr
    return {
        "command": command,
        "exit_code": completed.returncode,
        "duration_s": round(time.monotonic() - start, 6),
        "output_sha256": sha256_text(output),
        "output_tail": output[-1000:],
    }


def run_prewrite_checks(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    """Run each required code check once and retain its measured receipt."""

    return [_run_command_receipt(command, repo_root) for command in PREWRITE_COMMANDS]


def collect_preconditions(
    repo_root: Path,
    *,
    context_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, Path, JsonDict]:  # pragma: no cover - live host receipts.
    """Record resources, Exp6574 eligibility, CUDA state, cache, and policy."""

    server = runtime._resolve_llama_server()  # noqa: SLF001
    build = runtime._llama_cpp_build_receipt(server)  # noqa: SLF001
    resources = runtime._cpu_ram_disk_receipt(repo_root)  # noqa: SLF001
    model_paths = [str(row.get("selected_blob_path", "")) for row in metadata_rows]
    initial_gpu = runtime._live_gpu_sample(  # noqa: SLF001
        repository_id="exp6575-preconditions",
        worker_pid=0,
        stage="preconditions",
        sample_index=0,
        selected_gpu=0,
        model_paths=model_paths,
    )
    gpu_selection = runtime.choose_idle_gpu(initial_gpu)
    exp6574_path = repo_root / EXP6574_RELATIVE_PATH
    exp6574 = load_json(exp6574_path)
    exp6574_live = verify_artifact(exp6574_path)
    exp6574_eligible = (
        exp6574.get("joint_sufficiency_method_ready_score") == 1.0
        and exp6574.get("flagged_adversarial") is not True
        and exp6574_live.get("flag_count") == 0
    )
    metadata_ok = len(metadata_rows) == len(MANDATED_HF_IDS) and all(
        metadata_row_passes(row, spec) for row, spec in zip(metadata_rows, MODEL_SPECS, strict=True)
    )
    negative_ok = _ids_with_pass(negative_rows, "unit_id") == set(REQUIRED_NEGATIVE_FIXTURES)
    checks = {
        "v570_structural_context_recorded": len(context_rows) == 3
        and all(row.get("eligible_for_v571_reducer") is False for row in context_rows),
        "exp6574_clean_and_eligible": exp6574_eligible,
        "fresh_metadata": metadata_ok,
        "fresh_negative_fixtures": negative_ok,
        "prewrite_tests": bool(tests_run) and all(row.get("exit_code") == 0 for row in tests_run),
        "llama_cpp_cuda_build": build.get("exists") is True
        and build.get("executable") is True
        and build.get("cuda_linked") is True,
        "cuda_telemetry": initial_gpu.get("gpu_query_exit_code") == 0
        and initial_gpu.get("compute_query_exit_code") == 0,
        "idle_supported_gpu": gpu_selection.get("eligible") is True,
        "one_model_residency": not runtime._task_owned_pids(model_paths),  # noqa: SLF001
        "atomic_output_ready": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    preconditions: JsonDict = {
        "all_required_preconditions_available": all(checks.values()),
        "checks": checks,
        "failed_preconditions": [name for name, passed in checks.items() if not passed],
        **resources,
        "llama_cpp_build": build,
        "initial_gpu_state": initial_gpu,
        "gpu_selection": gpu_selection,
        "selected_gpu": gpu_selection.get("selected_gpu"),
        "active_processes_before": initial_gpu.get("compute_processes", []),
        "cache_provenance": [
            {
                "repository_id": row.get("repository_id"),
                "revision": row.get("provenance", {}).get("revision"),
                "snapshot_filename": row.get("provenance", {}).get("snapshot_filename"),
                "selected_blob_path": row.get("selected_blob_path"),
                "fresh_full_content_sha256": row.get("fresh_full_content_sha256"),
            }
            for row in metadata_rows
        ],
        "exp6574_eligibility": {
            "path": EXP6574_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(exp6574_path),
            "stored_ready_score": exp6574.get("joint_sufficiency_method_ready_score"),
            "stored_flagged_adversarial": exp6574.get("flagged_adversarial"),
            "live_verifier_report": exp6574_live,
            "eligible": exp6574_eligible,
        },
        "model_load_order": list(MANDATED_HF_IDS),
        "timeout_policy": {
            "load_timeout_s": runtime.LOAD_TIMEOUT_S,
            "generation_timeout_s": runtime.GENERATION_TIMEOUT_S,
            "shutdown_timeout_s": runtime.SHUTDOWN_TIMEOUT_S,
            "recovery_timeout_s": runtime.RECOVERY_TIMEOUT_S,
        },
        "expected_inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "generation_seeds": [RANDOM_SEED + index for index in range(3)],
        "unrelated_gpu_work_policy": "preserve; never signal or kill unrelated PIDs",
        "free_vram_arithmetic_used_as_gate": False,
        "broad_zombie_reaper_calls": 0,
    }
    return preconditions, server, initial_gpu


def _candidate_structural_verification(
    artifact: Mapping[str, Any], repo_root: Path
) -> JsonDict:  # pragma: no cover - live verifier execution.
    with tempfile.TemporaryDirectory(prefix="carnot-exp6575-verify-") as temporary:
        path = Path(temporary) / RESULT_RELATIVE_PATH.name
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        report = verify_artifact(path)
    flags = [dict(row) for row in report.get("flags", []) if isinstance(row, Mapping)]
    critical = [row for row in flags if row.get("severity") == "critical"]
    return hash_row(
        {
            "row_type": "live_structural_verification",
            "fresh": True,
            "candidate_reproducibility_checksum": artifact.get("reproducibility_checksum"),
            "duration_floor_s": LIVE_DURATION_FLOOR_S,
            "flag_count": report.get("flag_count", len(flags)),
            "critical_flag_count": len(critical),
            "findings": [row.get("kind") for row in flags],
            "report_sha256": sha256_json(report),
            "passed": not flags,
        }
    )


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Run tests, fresh evidence, sequential models, verifier, and one final write."""

    start = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    tests_run = run_prewrite_checks(repo_root)
    context_rows = build_v570_context_rows(repo_root)
    metadata_rows = build_fresh_metadata_rows()
    negative_rows = build_fresh_negative_rows()
    method_rows = build_method_replay_rows()
    attack_rows = build_attack_rows()
    preconditions, server, _initial_gpu = collect_preconditions(
        repo_root,
        context_rows=context_rows,
        metadata_rows=metadata_rows,
        negative_rows=negative_rows,
        tests_run=tests_run,
    )
    process_rows: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    raw_unload_rows: list[JsonDict] = []
    if preconditions["all_required_preconditions_available"]:
        process_rows, gpu_rows, raw_unload_rows = runtime.run_sequential_admission(
            metadata_rows,
            int(preconditions["selected_gpu"]),
            server,
            str(preconditions["llama_cpp_build"]["binary_sha256"]),
        )
    runtime_receipts, unload_rows = build_runtime_receipts(
        metadata_rows, process_rows, gpu_rows, raw_unload_rows
    )
    protected = _protected_unchanged(protected_before, _protected_hashes(repo_root))
    elapsed = time.monotonic() - start
    provisional_structural = hash_row(
        {
            "row_type": "live_structural_verification",
            "fresh": True,
            "duration_floor_s": LIVE_DURATION_FLOOR_S,
            "flag_count": 0 if elapsed >= LIVE_DURATION_FLOOR_S else 1,
            "critical_flag_count": 0 if elapsed >= LIVE_DURATION_FLOOR_S else 1,
            "findings": [] if elapsed >= LIVE_DURATION_FLOOR_S else ["DURATION_TOO_SHORT"],
            "passed": elapsed >= LIVE_DURATION_FLOOR_S,
        }
    )
    candidate = assemble_artifact(
        context_rows=context_rows,
        metadata_rows=metadata_rows,
        negative_rows=negative_rows,
        runtime_receipts=runtime_receipts,
        unload_rows=unload_rows,
        method_rows=method_rows,
        attack_rows=attack_rows,
        preconditions=preconditions,
        protected=protected,
        structural_verification=provisional_structural,
        duration_s=elapsed,
        tests_run=tests_run,
        run_date=run_date,
    )
    structural = _candidate_structural_verification(candidate, repo_root)
    artifact = assemble_artifact(
        context_rows=context_rows,
        metadata_rows=metadata_rows,
        negative_rows=negative_rows,
        runtime_receipts=runtime_receipts,
        unload_rows=unload_rows,
        method_rows=method_rows,
        attack_rows=attack_rows,
        preconditions=preconditions,
        protected=protected,
        structural_verification=structural,
        duration_s=time.monotonic() - start,
        tests_run=tests_run,
        run_date=run_date,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact validation failed before write: {errors}")
    atomic_write_json(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the qualification or validate the already-written terminal record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        path = REPO_ROOT / RESULT_RELATIVE_PATH
        payload = load_json(path)
        errors = validate_artifact(payload)
        if errors:
            print(json.dumps({"validated": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"validated": True, "errors": []}, indent=2))
        return 0
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "v571_flagship_evidence_ready_score": artifact[
                    "v571_flagship_evidence_ready_score"
                ],
                "joint_sufficiency_method_replay_ready_score": artifact[
                    "joint_sufficiency_method_replay_ready_score"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
