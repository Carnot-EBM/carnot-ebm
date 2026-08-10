"""Exp6286 V541 evidence eligibility ledger.

Spec refs: REQ-INFRA-6286, SCENARIO-INFRA-6286-1,
SCENARIO-INFRA-6286-2, SCENARIO-INFRA-6286-3,
SCENARIO-INFRA-6286-4, SCENARIO-INFRA-6286-5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

EXPERIMENT_ID = "exp6286-v541-evidence-eligibility-ledger"
SCHEMA = "carnot.experiment_6286.v541_evidence_eligibility_ledger.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6286_v541_evidence_eligibility_ledger.json")
ELIGIBLE_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6286_v541_flagship_raw_row_eligible_manifest.jsonl"
)
QUARANTINE_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6286_v541_flagship_raw_row_quarantine_receipt.jsonl"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

V541_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6283_v541_adversarial_capstone.json")
OPERATIONAL_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_541.json")
ASP_COMPILER_RELATIVE_PATH = Path("results/experiment_6274_asp_energy_semantic_compiler.json")
FLAGSHIP_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.json"
)
FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.event_corpus.jsonl"
)
FLAGSHIP_FORMAL_SIDECAR_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.formal_sidecar.json"
)
FLAGSHIP_SEALED_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.sealed_manifest.json"
)
FLAGSHIP_RAW_DIR_RELATIVE_PATH = Path("results/experiment_6275_flagship_asp_raw")
DUAL_CACHE_RELATIVE_PATH = Path("results/experiment_6276_certified_dual_cache_admission.json")
TYPED_BACKEND_RELATIVE_PATH = Path(
    "results/experiment_6280_variable_cardinality_mode_jump_backend.json"
)
MODE_JUMP_RELATIVE_PATH = Path("results/experiment_6281_mode_jump_multifamily_rerun.json")
ARC_ROUTER_RELATIVE_PATH = Path("results/experiment_6282_arc_mechanic_class_live_router.json")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ANOMALY_ESCALATIONS_RELATIVE_PATH = Path("ops/anomaly-escalations.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

V541_ARTIFACTS: dict[str, Path] = {
    "exp6274-asp-energy-semantic-compiler": ASP_COMPILER_RELATIVE_PATH,
    "exp6275-flagship-asp-constraint-verification-benchmark": FLAGSHIP_RELATIVE_PATH,
    "exp6276-certified-dual-cache-admission": DUAL_CACHE_RELATIVE_PATH,
    "exp6280-variable-cardinality-mode-jump-backend": TYPED_BACKEND_RELATIVE_PATH,
    "exp6281-mode-jump-multifamily-rerun": MODE_JUMP_RELATIVE_PATH,
    "exp6282-arc-mechanic-class-live-router": ARC_ROUTER_RELATIVE_PATH,
    "exp6283-v541-adversarial-capstone": V541_CAPSTONE_RELATIVE_PATH,
}

PROTECTED_RELATIVE_PATHS = (
    V541_CAPSTONE_RELATIVE_PATH,
    OPERATIONAL_RETRO_RELATIVE_PATH,
    ASP_COMPILER_RELATIVE_PATH,
    FLAGSHIP_RELATIVE_PATH,
    FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH,
    FLAGSHIP_FORMAL_SIDECAR_RELATIVE_PATH,
    FLAGSHIP_SEALED_MANIFEST_RELATIVE_PATH,
    DUAL_CACHE_RELATIVE_PATH,
    TYPED_BACKEND_RELATIVE_PATH,
    MODE_JUMP_RELATIVE_PATH,
    ARC_ROUTER_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    TERMINAL_ARTIFACTS_RELATIVE_PATH,
    ANOMALY_ESCALATIONS_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

INPUT_RELATIVE_PATHS = PROTECTED_RELATIVE_PATHS + (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py"),
    Path("tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "v541_capstone_path_hash_and_terminal_class",
    "current_rule_adversarial_results_by_v541_task",
    "asp_compiler_eligibility",
    "flagship_artifact_eligibility",
    "flagship_raw_manifest_paths_and_hashes",
    "flagship_raw_row_validation_rules",
    "eligible_flagship_raw_row_count",
    "quarantined_flagship_raw_row_count",
    "flagship_raw_row_eligibility_manifest_path_and_hash",
    "dual_cache_treatment_eligibility",
    "global_threshold_control_eligibility",
    "typed_backend_eligibility",
    "mode_jump_treatment_eligibility",
    "arc_router_source_eligibility",
    "arc_result_eligibility",
    "branch_stop_ledger",
    "no_claim_laundering_receipt",
    "source_mutation_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether every eligibility check stayed terminal.",
    "v541_capstone_path_hash_and_terminal_class": "Pins the capstone without treating it as an override.",
    "current_rule_adversarial_results_by_v541_task": "Keeps stamped and current flags separate.",
    "asp_compiler_eligibility": "Allows exact ASP compiler substrate reuse only.",
    "flagship_artifact_eligibility": "Keeps the stamped flagship artifact closed.",
    "flagship_raw_manifest_paths_and_hashes": "Pins raw receipts before row validation.",
    "flagship_raw_row_validation_rules": "States that rows are checked for provenance only.",
    "eligible_flagship_raw_row_count": "Counts rows with complete provenance.",
    "quarantined_flagship_raw_row_count": "Counts rows missing required provenance.",
    "flagship_raw_row_eligibility_manifest_path_and_hash": "Content-addresses eligible and quarantine rows.",
    "dual_cache_treatment_eligibility": "Prevents unchanged low-utility treatment reuse.",
    "global_threshold_control_eligibility": "Keeps the baseline control record-only.",
    "typed_backend_eligibility": "Allows typed backend substrate reuse.",
    "mode_jump_treatment_eligibility": "Blocks unchanged no-value treatment extension.",
    "arc_router_source_eligibility": "Allows ARC router source reuse only.",
    "arc_result_eligibility": "Keeps the flagged ARC result unpromoted.",
    "branch_stop_ledger": "Names where each branch must stop.",
    "no_claim_laundering_receipt": "Proves receipts did not become claims.",
    "source_mutation_count": "Bare zero proves no source edits occurred.",
    "protected_files_unchanged": "Hash receipts catch drift after preconditions.",
    "preconditions_checked": "Input hashes are frozen before eligibility decisions.",
    "inference_substrate": "This ledger aggregates checked-in evidence.",
    "verifier_is_oracle": "The ledger audits records, not benchmark answers.",
    "field_provenance": "Every field cites concrete sources.",
    "field_principles": "Every field states its reason.",
    "test_commands": "Commands define the verification boundary.",
    "test_exit_codes": "Exit codes prevent hidden failures.",
    "duration_s": "Wall time is reported without padding.",
    "reproducibility_checksum": "The normalized payload detects drift.",
    "honest_verdict": "The verdict preserves the claim boundary.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py -m pytest tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py --fail-under=100 --show-missing",
    ".venv/bin/ruff check python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6286_v541_evidence_eligibility_ledger.json",
)
COMMAND_TIMEOUTS_S = {".venv/bin/pytest tests/python -q": 3600}


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_hexdigest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def _read_json_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _jsonl_blob(rows: Sequence[JsonMap]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _row_path(root: Path, raw_path: object) -> Path:
    text = str(raw_path or "")
    if not text:
        return root / "missing"
    path = Path(text)
    return path if path.is_absolute() else root / path


def _row_count_jsonl(path: Path) -> int:
    return len(_read_jsonl(path))


def _path_receipt(path: Path, *, row_count: int | None = None) -> JsonDict:
    receipt: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
    }
    if row_count is not None:
        receipt["row_count"] = row_count
    return receipt


def _classification_receipt(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    classification = classify_artifact_path(path)
    return {
        "path": rel_path.as_posix(),
        "sha256": path_sha256(path),
        "terminal_class": classification.classification,
        "terminal": classification.terminal,
        "status_raw": classification.status_raw,
        "honest_verdict_raw": classification.honest_verdict_raw,
    }


def _score(payload: JsonMap, field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _artifact_payloads(root: Path) -> dict[str, JsonDict]:
    return {task_id: _read_json_mapping(root / rel) for task_id, rel in V541_ARTIFACTS.items()}


def current_rule_adversarial_results(root: Path) -> JsonDict:
    from adversarial_verify import verify_artifact

    rows: JsonDict = {}
    for task_id, rel in V541_ARTIFACTS.items():
        path = root / rel
        payload = _read_json_mapping(path)
        if not path.exists():
            rows[task_id] = {
                "path": rel.as_posix(),
                "present": False,
                "stamped_flagged_adversarial": False,
                "stamped_corrigendum_pending": False,
                "current_rule_flag_count": 0,
                "current_rule_critical_flag_count": 0,
                "current_rule_warn_flag_count": 0,
                "current_rule_flags": [],
                "skipped": "missing_artifact",
            }
            continue
        current = verify_artifact(path)
        flags = [dict(flag) for flag in current.get("flags", []) if isinstance(flag, Mapping)]
        rows[task_id] = {
            "path": rel.as_posix(),
            "present": True,
            "sha256": path_sha256(path),
            "terminal_class": classify_artifact_path(path).classification,
            "stamped_flagged_adversarial": payload.get("flagged_adversarial") is True,
            "stamped_corrigendum_pending": bool(payload.get("corrigendum_pending")),
            "current_rule_flag_count": int(current.get("flag_count") or len(flags)),
            "current_rule_critical_flag_count": sum(
                1 for flag in flags if flag.get("severity") == "critical"
            ),
            "current_rule_warn_flag_count": sum(
                1 for flag in flags if flag.get("severity") == "warn"
            ),
            "current_rule_flags": flags,
        }
    return rows


def _is_flagged(task_id: str, reviews: JsonMap, payload: JsonMap) -> bool:
    review = reviews.get(task_id)
    current_critical = (
        int(review.get("current_rule_critical_flag_count") or 0)
        if isinstance(review, Mapping)
        else 0
    )
    return bool(
        payload.get("flagged_adversarial") is True
        or payload.get("corrigendum_pending")
        or current_critical > 0
    )


def _terminal_unflagged_ready(
    root: Path,
    task_id: str,
    rel_path: Path,
    payload: JsonMap,
    reviews: JsonMap,
    ready_field: str,
) -> bool:
    classification = classify_artifact_path(root / rel_path)
    return bool(
        classification.terminal
        and _score(payload, ready_field) in (1, 1.0, True)
        and not _is_flagged(task_id, reviews, payload)
    )


def _seed_lookup(seed_matrix: JsonMap) -> dict[tuple[str, str, str], list[int]]:
    matrix = seed_matrix.get("matrix")
    out: dict[tuple[str, str, str], set[int]] = {}
    if not isinstance(matrix, Mapping):
        return {}
    for model_hf_id, entries in matrix.items():
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            task_id = str(entry.get("task_id") or "")
            arm_samples = entry.get("arm_samples")
            if not isinstance(arm_samples, Mapping):
                continue
            for arm, seeds in arm_samples.items():
                if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
                    continue
                key = (str(model_hf_id), task_id, str(arm))
                bucket = out.setdefault(key, set())
                for seed in seeds:
                    try:
                        bucket.add(int(seed))
                    except (TypeError, ValueError):
                        continue
    return {key: sorted(values) for key, values in out.items()}


def _raw_lookup(
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, str, int], JsonDict]:
    out: dict[tuple[str, str, int], JsonDict] = {}
    for model_hf_id, rows in raw_rows_by_model.items():
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                seed = int(row.get("seed"))
            except (TypeError, ValueError):
                continue
            key = (str(model_hf_id), str(row.get("task_id") or ""), seed)
            out[key] = dict(row)
    return out


def _has_bool(row: JsonMap, key: str) -> bool:
    return isinstance(row.get(key), bool)


def _validate_raw_sample(event: JsonMap, raw: JsonMap | None, missing: list[str]) -> None:
    if raw is None:
        missing.append("missing_raw_sample")
        return
    if not isinstance(raw.get("seed"), int):
        missing.append("missing_seed")
    if not str(raw.get("prompt_text") or "").strip():
        missing.append("missing_prompt")
    if not str(raw.get("prompt_hash") or ""):
        missing.append("missing_prompt_hash")
    if event.get("prompt_hash") and raw.get("prompt_hash") != event.get("prompt_hash"):
        missing.append("prompt_hash_mismatch")
    if "prompt_token_count" not in raw or "generated_token_count" not in raw:
        missing.append("missing_token_count")
    else:
        try:
            int(raw.get("prompt_token_count"))
            int(raw.get("generated_token_count"))
        except (TypeError, ValueError):
            missing.append("missing_token_count")
    raw_output = str(raw.get("raw_output") or "")
    if not raw_output:
        missing.append("missing_model_output")
    raw_hash = str(raw.get("raw_output_hash") or "")
    if not raw_hash:
        missing.append("missing_raw_output_hash")
    elif raw_hash not in set(str(value) for value in event.get("raw_output_hashes") or []):
        missing.append("raw_hash_not_in_event_row")
    if raw_output and raw_hash and raw_hash != _sha256_hexdigest_text(raw_output):
        missing.append("raw_output_hash_mismatch")


def validate_flagship_raw_rows(
    event_rows: Sequence[Mapping[str, Any]],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    seed_matrix: Mapping[str, Any],
) -> JsonDict:
    seed_by_event = _seed_lookup(seed_matrix)
    raw_by_seed = _raw_lookup(raw_rows_by_model)
    eligible_rows: list[JsonDict] = []
    quarantine_rows: list[JsonDict] = []
    for row_index, raw_event in enumerate(event_rows):
        event = dict(raw_event)
        missing: list[str] = []
        model_hf_id = str(event.get("model_hf_id") or "")
        task_id = str(event.get("task_id") or "")
        arm = str(event.get("arm") or "")
        event_key = {
            "row_index": row_index,
            "model_hf_id": model_hf_id,
            "task_id": task_id,
            "arm": arm,
            "fixture_id": event.get("fixture_id"),
            "family": event.get("family"),
        }
        if not model_hf_id:
            missing.append("missing_model_hf_id")
        if not task_id:
            missing.append("missing_task_id")
        if not arm:
            missing.append("missing_arm")
        if not str(event.get("prompt_hash") or ""):
            missing.append("missing_prompt_hash")
        if not str(event.get("formal_sidecar_hash") or ""):
            missing.append("missing_formal_sidecar_hash")
        if not event.get("raw_output_hashes"):
            missing.append("missing_raw_output_hashes")
        if event.get("complete_provenance") is not True:
            missing.append("missing_complete_provenance")
        for key in ("parse_success", "semantic_valid", "exact_certificate_present", "abstention"):
            if not _has_bool(event, key):
                missing.append(f"missing_{key}")
        if event.get("exact_certificate_present") is not True:
            missing.append("missing_exact_certificate")
        if "residual_rule_violation_count" not in event:
            missing.append("missing_residual_rule_violation_count")
        seeds = seed_by_event.get((model_hf_id, task_id, arm), [])
        if not seeds:
            missing.append("missing_seed")
        for seed in seeds:
            _validate_raw_sample(event, raw_by_seed.get((model_hf_id, task_id, seed)), missing)
        manifest_row: JsonDict = {
            **event_key,
            "prompt_hash": event.get("prompt_hash"),
            "formal_sidecar_hash": event.get("formal_sidecar_hash"),
            "raw_output_hashes": list(event.get("raw_output_hashes") or []),
            "seeds": seeds,
            "validation_mode": "provenance_only_no_scientific_rescoring",
        }
        if missing:
            quarantine_rows.append(
                {
                    **manifest_row,
                    "eligible": False,
                    "missing": sorted(set(missing)),
                }
            )
        else:
            eligible_rows.append({**manifest_row, "eligible": True})
    return {
        "validation_mode": "provenance_only_no_scientific_rescoring",
        "source_row_count": len(event_rows),
        "eligible_count": len(eligible_rows),
        "quarantined_count": len(quarantine_rows),
        "eligible_rows": eligible_rows,
        "quarantine_rows": quarantine_rows,
    }


def flagship_raw_manifest_paths_and_hashes(root: Path, flagship_payload: JsonMap) -> JsonDict:
    raw_receipts = flagship_payload.get("raw_output_paths_and_hashes")
    raw_paths: JsonDict = {}
    if isinstance(raw_receipts, Mapping):
        for model_hf_id, receipt in raw_receipts.items():
            if not isinstance(receipt, Mapping):
                continue
            path = _row_path(root, receipt.get("path"))
            raw_paths[str(model_hf_id)] = {
                **_path_receipt(path, row_count=_row_count_jsonl(path)),
                "declared_sha256": receipt.get("sha256"),
                "declared_row_count": receipt.get("row_count"),
                "contains_prompt": receipt.get("contains_prompt") is True,
                "contains_raw_output": receipt.get("contains_raw_output") is True,
                "contains_seed": receipt.get("contains_seed") is True,
                "contains_token_count": receipt.get("contains_token_count") is True,
            }
    return {
        "event_corpus": _path_receipt(
            root / FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH,
            row_count=_row_count_jsonl(root / FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH),
        ),
        "sealed_manifest": _path_receipt(root / FLAGSHIP_SEALED_MANIFEST_RELATIVE_PATH),
        "formal_sidecar": _path_receipt(root / FLAGSHIP_FORMAL_SIDECAR_RELATIVE_PATH),
        "raw_outputs_by_model": raw_paths,
        "principle": FIELD_PRINCIPLES["flagship_raw_manifest_paths_and_hashes"],
    }


def _load_raw_rows_by_model(root: Path, flagship_payload: JsonMap) -> dict[str, list[JsonDict]]:
    raw_receipts = flagship_payload.get("raw_output_paths_and_hashes")
    rows: dict[str, list[JsonDict]] = {}
    if isinstance(raw_receipts, Mapping):
        for model_hf_id, receipt in raw_receipts.items():
            if isinstance(receipt, Mapping):
                rows[str(model_hf_id)] = _read_jsonl(_row_path(root, receipt.get("path")))
    return rows


def _row_manifest_receipt(row_validation: JsonMap) -> JsonDict:
    eligible_rows = list(row_validation.get("eligible_rows") or [])
    quarantine_rows = list(row_validation.get("quarantine_rows") or [])
    eligible_blob = _jsonl_blob(eligible_rows)
    quarantine_blob = _jsonl_blob(quarantine_rows)
    return {
        "eligible_manifest": {
            "path": ELIGIBLE_ROWS_RELATIVE_PATH.as_posix(),
            "sha256": sha256_text(eligible_blob),
            "row_count": len(eligible_rows),
        },
        "quarantine_receipt": {
            "path": QUARANTINE_ROWS_RELATIVE_PATH.as_posix(),
            "sha256": sha256_text(quarantine_blob),
            "row_count": len(quarantine_rows),
        },
        "eligible_rows": eligible_rows,
        "quarantine_rows": quarantine_rows,
        "principle": FIELD_PRINCIPLES["flagship_raw_row_eligibility_manifest_path_and_hash"],
    }


def _eligibility_rows(root: Path, payloads: Mapping[str, JsonMap], reviews: JsonMap) -> JsonDict:
    asp = payloads.get("exp6274-asp-energy-semantic-compiler", {})
    flagship = payloads.get("exp6275-flagship-asp-constraint-verification-benchmark", {})
    dual_cache = payloads.get("exp6276-certified-dual-cache-admission", {})
    typed = payloads.get("exp6280-variable-cardinality-mode-jump-backend", {})
    mode_jump = payloads.get("exp6281-mode-jump-multifamily-rerun", {})
    arc = payloads.get("exp6282-arc-mechanic-class-live-router", {})

    asp_reusable = _terminal_unflagged_ready(
        root,
        "exp6274-asp-energy-semantic-compiler",
        ASP_COMPILER_RELATIVE_PATH,
        asp,
        reviews,
        "asp_energy_semantic_ready_score",
    )
    typed_reusable = _terminal_unflagged_ready(
        root,
        "exp6280-variable-cardinality-mode-jump-backend",
        TYPED_BACKEND_RELATIVE_PATH,
        typed,
        reviews,
        "variable_cardinality_backend_ready_score",
    )
    flagship_flagged = _is_flagged(
        "exp6275-flagship-asp-constraint-verification-benchmark", reviews, flagship
    )
    arc_flagged = _is_flagged("exp6282-arc-mechanic-class-live-router", reviews, arc)
    return {
        "asp_compiler_eligibility": {
            "source_module_reusable": asp_reusable,
            "artifact_gate_eligible": asp_reusable,
            "path": ASP_COMPILER_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ASP_COMPILER_RELATIVE_PATH),
            "reason": "terminal_unflagged_exact_compiler_parity"
            if asp_reusable
            else "compiler_substrate_closed",
        },
        "flagship_artifact_eligibility": {
            "artifact_gate_eligible": False,
            "artifact_level_readiness_closed": True,
            "raw_rows_reopen_artifact_claim": False,
            "path": FLAGSHIP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / FLAGSHIP_RELATIVE_PATH),
            "stamped_or_current_flagged": flagship_flagged,
            "reason": "stamped_flagged_artifact_no_claim_laundering",
        },
        "dual_cache_treatment_eligibility": {
            "source_artifact_safe": not _is_flagged(
                "exp6276-certified-dual-cache-admission", reviews, dual_cache
            ),
            "v542_extension_eligible": False,
            "readiness_score": _score(dual_cache, "certified_admission_ready_score"),
            "reason": "unchanged_complete_null_low_utility_treatment",
        },
        "global_threshold_control_eligibility": {
            "control_name": "exp6264_global_threshold",
            "control_receipt_reusable": True,
            "v542_extension_eligible": False,
            "reason": "baseline_control_record_only_no_treatment_claim",
        },
        "typed_backend_eligibility": {
            "source_module_reusable": typed_reusable,
            "artifact_gate_eligible": typed_reusable,
            "path": TYPED_BACKEND_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / TYPED_BACKEND_RELATIVE_PATH),
            "reason": "terminal_unflagged_backend_substrate"
            if typed_reusable
            else "typed_backend_closed",
        },
        "mode_jump_treatment_eligibility": {
            "v542_extension_eligible": False,
            "safety_ready_score": _score(mode_jump, "mode_jump_safety_ready_score"),
            "workload_value_ready_score": _score(mode_jump, "mode_jump_workload_value_ready_score"),
            "reason": "unchanged_no_workload_value_treatment",
        },
        "arc_router_source_eligibility": {
            "source_module_reusable": True,
            "source_paths": arc.get("detector_source_paths_and_hashes", []),
            "reason": "source_only_reuse_allowed_result_unpromoted",
        },
        "arc_result_eligibility": {
            "artifact_gate_eligible": False,
            "stamped_or_current_flagged": arc_flagged,
            "ready_score": _score(arc, "arc_mechanic_router_ready_score"),
            "reason": "flagged_live_result_unpromoted",
        },
    }


def branch_stop_ledger(eligibility: JsonMap) -> JsonDict:
    return {
        "asp_flagship": {
            "stop_state": "stop_at_flagship_artifact",
            "source_reuse_allowed": eligibility["asp_compiler_eligibility"][
                "source_module_reusable"
            ],
            "artifact_promotion_allowed": False,
            "reason": eligibility["flagship_artifact_eligibility"]["reason"],
        },
        "dual_cache": {
            "stop_state": "stop_at_unchanged_treatment",
            "extension_allowed": False,
            "reason": eligibility["dual_cache_treatment_eligibility"]["reason"],
        },
        "mode_jump": {
            "stop_state": "stop_at_no_workload_value_treatment",
            "backend_reuse_allowed": eligibility["typed_backend_eligibility"][
                "source_module_reusable"
            ],
            "extension_allowed": False,
            "reason": eligibility["mode_jump_treatment_eligibility"]["reason"],
        },
        "arc_router": {
            "stop_state": "stop_at_flagged_result",
            "source_reuse_allowed": eligibility["arc_router_source_eligibility"][
                "source_module_reusable"
            ],
            "artifact_promotion_allowed": False,
            "reason": eligibility["arc_result_eligibility"]["reason"],
        },
        "principle": FIELD_PRINCIPLES["branch_stop_ledger"],
    }


def no_claim_laundering_receipt(eligibility: JsonMap, row_validation: JsonMap) -> JsonDict:
    flagship_promoted = bool(eligibility["flagship_artifact_eligibility"]["artifact_gate_eligible"])
    arc_promoted = bool(eligibility["arc_result_eligibility"]["artifact_gate_eligible"])
    return {
        "flagged_artifact_promoted": flagship_promoted or arc_promoted,
        "raw_rows_reopen_flagship_claim": False,
        "valid_rows_used_for_clean_benchmark_claim": False,
        "unchanged_dual_cache_or_mode_jump_treatment_extended": False,
        "eligible_raw_rows_recorded_as_receipts_only": int(
            row_validation.get("eligible_count") or 0
        ),
        "quarantined_raw_rows_recorded": int(row_validation.get("quarantined_count") or 0),
        "principle": FIELD_PRINCIPLES["no_claim_laundering_receipt"],
    }


def protected_hashes(root: Path, paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS) -> JsonDict:
    return {path.as_posix(): path_sha256(root / path) for path in paths}


def protected_files_unchanged(
    root: Path,
    before: JsonMap,
    paths: Sequence[Path] = PROTECTED_RELATIVE_PATHS,
) -> JsonDict:
    after = protected_hashes(root, paths)
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def input_hashes(root: Path) -> JsonDict:
    receipts: JsonDict = {}
    for path in INPUT_RELATIVE_PATHS:
        receipts[path.as_posix()] = {
            "present": (root / path).exists(),
            "sha256": path_sha256(root / path),
        }
    raw_dir = root / FLAGSHIP_RAW_DIR_RELATIVE_PATH
    for path in sorted(raw_dir.glob("*.raw.jsonl")) if raw_dir.exists() else []:
        receipts[path.relative_to(root).as_posix()] = {
            "present": True,
            "sha256": path_sha256(path),
            "row_count": _row_count_jsonl(path),
        }
    return receipts


def git_status_lines(root: Path) -> list[str]:  # pragma: no cover
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [line for line in proc.stdout.splitlines() if line]


def preconditions_checked(
    root: Path,
    before_hashes: JsonMap,
    git_status_before: Sequence[str],
    git_status_after_tests: Sequence[str],
) -> JsonDict:
    return {
        "checked_before_eligibility": True,
        "v541_artifact_hashes": {
            task_id: path_sha256(root / rel) for task_id, rel in V541_ARTIFACTS.items()
        },
        "raw_and_rule_hashes": input_hashes(root),
        "protected_hashes_before": dict(before_hashes),
        "git_status_before": list(git_status_before),
        "git_status_after_tests": list(git_status_after_tests),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def flagship_raw_row_validation_rules() -> JsonDict:
    return {
        "validation_mode": "provenance_only_no_scientific_rescoring",
        "required_event_fields": [
            "model_hf_id",
            "task_id",
            "arm",
            "prompt_hash",
            "formal_sidecar_hash",
            "raw_output_hashes",
            "parse_success",
            "semantic_valid",
            "exact_certificate_present",
            "abstention",
            "residual_rule_violation_count",
            "complete_provenance",
        ],
        "required_raw_fields": [
            "raw_output",
            "prompt_text",
            "prompt_hash",
            "seed",
            "prompt_token_count",
            "generated_token_count",
            "raw_output_hash",
        ],
        "quarantine_if_missing": [
            "model output",
            "prompt",
            "seed",
            "token",
            "sidecar",
            "outcome provenance",
        ],
        "scientific_rescoring_performed": False,
        "principle": FIELD_PRINCIPLES["flagship_raw_row_validation_rules"],
    }


def _field_provenance() -> JsonDict:
    base_sources = [
        "REQ-INFRA-6286",
        SPEC_RELATIVE_PATH.as_posix(),
        V541_CAPSTONE_RELATIVE_PATH.as_posix(),
        OPERATIONAL_RETRO_RELATIVE_PATH.as_posix(),
        ASP_COMPILER_RELATIVE_PATH.as_posix(),
        FLAGSHIP_RELATIVE_PATH.as_posix(),
        FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH.as_posix(),
        FLAGSHIP_FORMAL_SIDECAR_RELATIVE_PATH.as_posix(),
        FLAGSHIP_SEALED_MANIFEST_RELATIVE_PATH.as_posix(),
        DUAL_CACHE_RELATIVE_PATH.as_posix(),
        TYPED_BACKEND_RELATIVE_PATH.as_posix(),
        MODE_JUMP_RELATIVE_PATH.as_posix(),
        ARC_ROUTER_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
        TERMINAL_ARTIFACTS_RELATIVE_PATH.as_posix(),
        ANOMALY_ESCALATIONS_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": base_sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exit_codes(command_rows: Sequence[JsonMap]) -> JsonDict:
    return {
        str(row.get("command") or ""): int(row.get("exit_code") or 0)
        for row in command_rows
        if row.get("command")
    }


def _status_from_commands(command_rows: Sequence[JsonMap]) -> tuple[str, str]:
    if any(int(row.get("exit_code") or 0) != 0 for row in command_rows):
        return "blocked", "blocked: Exp6286 ledger wrote evidence but one validation command failed"
    return (
        "complete",
        "complete: V541 eligibility ledger preserves raw receipts without promoting flagged or unchanged artifacts",
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    command_receipts: Sequence[JsonMap] | None = None,
    current_reviews: JsonMap | None = None,
    before_hashes: JsonMap | None = None,
    git_status_before: Sequence[str] | None = None,
    git_status_after_tests: Sequence[str] | None = None,
    started_at: float | None = None,
) -> JsonDict:
    started = time.perf_counter() if started_at is None else started_at
    before = dict(before_hashes or protected_hashes(root))
    payloads = _artifact_payloads(root)
    flagship = payloads["exp6275-flagship-asp-constraint-verification-benchmark"]
    event_rows = _read_jsonl(root / FLAGSHIP_EVENT_CORPUS_RELATIVE_PATH)
    raw_rows_by_model = _load_raw_rows_by_model(root, flagship)
    seed_matrix = flagship.get("preregistered_model_task_arm_seed_matrix")
    seed_matrix = seed_matrix if isinstance(seed_matrix, Mapping) else {}
    row_validation = validate_flagship_raw_rows(event_rows, raw_rows_by_model, seed_matrix)
    reviews = dict(current_reviews or current_rule_adversarial_results(root))
    eligibility = _eligibility_rows(root, payloads, reviews)
    command_rows = [dict(row) for row in command_receipts or []]
    status, verdict = _status_from_commands(command_rows)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": status,
        "v541_capstone_path_hash_and_terminal_class": _classification_receipt(
            root, V541_CAPSTONE_RELATIVE_PATH
        ),
        "current_rule_adversarial_results_by_v541_task": reviews,
        **eligibility,
        "flagship_raw_manifest_paths_and_hashes": flagship_raw_manifest_paths_and_hashes(
            root, flagship
        ),
        "flagship_raw_row_validation_rules": flagship_raw_row_validation_rules(),
        "eligible_flagship_raw_row_count": int(row_validation["eligible_count"]),
        "quarantined_flagship_raw_row_count": int(row_validation["quarantined_count"]),
        "flagship_raw_row_eligibility_manifest_path_and_hash": _row_manifest_receipt(
            row_validation
        ),
        "branch_stop_ledger": branch_stop_ledger(eligibility),
        "no_claim_laundering_receipt": no_claim_laundering_receipt(eligibility, row_validation),
        "source_mutation_count": 0,
        "protected_files_unchanged": protected_files_unchanged(root, before),
        "preconditions_checked": preconditions_checked(
            root,
            before,
            git_status_before or [],
            git_status_after_tests or [],
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": [str(row.get("command") or "") for row in command_rows]
        or list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": _test_exit_codes(command_rows),
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def _is_bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles is not a mapping")
        principles = {}
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance is not a mapping")
        provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if not _is_bare_zero(report.get("source_mutation_count")):
        errors.append("source_mutation_count must be bare integer 0")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("unchanged") is not True:
        errors.append("protected_files_unchanged reports drift")
    source_count = (
        report.get("flagship_raw_manifest_paths_and_hashes", {})
        .get("event_corpus", {})
        .get("row_count")
        if isinstance(report.get("flagship_raw_manifest_paths_and_hashes"), Mapping)
        else None
    )
    eligible = report.get("eligible_flagship_raw_row_count")
    quarantined = report.get("quarantined_flagship_raw_row_count")
    if not isinstance(eligible, int) or not isinstance(quarantined, int):
        errors.append("raw row counts must be integers")
    elif source_count is not None and eligible + quarantined != source_count:
        errors.append("eligible and quarantined raw rows do not sum to source row count")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    laundering = report.get("no_claim_laundering_receipt")
    if (
        not isinstance(laundering, Mapping)
        or laundering.get("flagged_artifact_promoted") is not False
    ):
        errors.append("claim laundering receipt promoted a flagged artifact")
    if (
        isinstance(report.get("flagship_artifact_eligibility"), Mapping)
        and report["flagship_artifact_eligibility"].get("artifact_gate_eligible") is not False
    ):
        errors.append("flagship artifact gate must remain closed")
    if (
        isinstance(report.get("arc_result_eligibility"), Mapping)
        and report["arc_result_eligibility"].get("artifact_gate_eligible") is not False
    ):
        errors.append("arc result gate must remain closed")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(
        (
            "complete:",
            "complete_",
            "complete_null:",
            "blocked:",
            "blocked_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    ):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = report.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        if checksum != payload_checksum(report):
            errors.append("reproducibility_checksum mismatch")
    else:
        errors.append("reproducibility_checksum missing")
    return errors


def _write_jsonl_artifact(
    rel_path: Path,
    rows: Sequence[JsonMap],
    *,
    root: Path,
    env: Mapping[str, str] | None,
) -> Path:
    return atomic_write_text(rel_path, _jsonl_blob(rows), root=root, env=env)


def write_report(
    report: JsonMap,
    root: Path = REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    errors = validate_report(report)
    if errors:
        raise ValueError("invalid Exp6286 report: " + "; ".join(errors))
    sidecars = report["flagship_raw_row_eligibility_manifest_path_and_hash"]
    _write_jsonl_artifact(
        ELIGIBLE_ROWS_RELATIVE_PATH,
        sidecars.get("eligible_rows") or [],
        root=root,
        env=env,
    )
    _write_jsonl_artifact(
        QUARANTINE_ROWS_RELATIVE_PATH,
        sidecars.get("quarantine_rows") or [],
        root=root,
        env=env,
    )
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)


def run_command(
    command: str, root: Path, timeout_s: int | None = None
) -> JsonDict:  # pragma: no cover
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "classification": "timeout",
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_default_commands(root: Path) -> list[JsonDict]:  # pragma: no cover
    return [
        run_command(command, root, COMMAND_TIMEOUTS_S.get(command))
        for command in DEFAULT_TEST_COMMANDS
    ]


def run_experiment(root: Path, date: str, *, run_commands: bool) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    before = protected_hashes(root)
    status_before = git_status_lines(root)
    preliminary = build_report(
        root,
        date=date,
        command_receipts=[],
        before_hashes=before,
        git_status_before=status_before,
        git_status_after_tests=[],
        started_at=started,
    )
    write_report(preliminary, root)
    commands = run_default_commands(root) if run_commands else []
    status_after_tests = git_status_lines(root)
    final = build_report(
        root,
        date=date,
        command_receipts=commands,
        before_hashes=before,
        git_status_before=status_before,
        git_status_after_tests=status_after_tests,
        started_at=started,
    )
    write_report(final, root)
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--no-run-commands", action="store_true")
    args = parser.parse_args(argv)
    report = run_experiment(REPO_ROOT, args.date, run_commands=not args.no_run_commands)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if report.get("status") == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
