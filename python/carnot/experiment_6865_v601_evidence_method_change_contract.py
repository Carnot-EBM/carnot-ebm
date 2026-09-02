"""Build the read-only V601 evidence and method-change contract.

The reducer reads checked-in evidence and recomputes small manifests. It does
not tokenize text, run a model, execute ARC, or update memory. This boundary
keeps method design separate from the science tasks that will test it later.

Spec refs: REQ-REPORT-6865 and SCENARIO-REPORT-6865-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6865_v601_evidence_method_change_contract.json")
EXPECTED_ACTIVE_MILESTONE = "2026.09.601"
V600_MILESTONE = "2026.09.600"
EXP6865_TASK_ID = "exp6865-v601-evidence-method-change-contract"
INFERENCE_SUBSTRATE = "deterministic CPU read-only evidence reduction"
RANDOM_SEED = 6865
TOKENIZER_PROBE = " execution readiness is checked without a scientific label."

V600_TASK_IDS = (
    "exp6861-v600-branch-retirement-evidence-contract",
    "exp6862-dual-side-semantic-contrast-bank",
    "exp6863-tokenizer-aware-semantic-contrast-preregistration",
    "exp6864-three-family-semantic-contrast-scoring-stream",
)
V600_ARTIFACT_PATHS = {
    V600_TASK_IDS[0]: "results/experiment_6861_v600_branch_retirement_evidence_contract.json",
    V600_TASK_IDS[1]: "results/experiment_6862_dual_side_semantic_contrast_bank.json",
    V600_TASK_IDS[2]: (
        "results/experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json"
    ),
    V600_TASK_IDS[3]: "results/experiment_6864_three_family_semantic_contrast_scoring_stream.json",
}
DEFAULT_SOURCE_PATHS = {
    "exp6850": "results/experiment_6850_three_family_scoring_admission_canary.json",
    "exp6856": "results/experiment_6856_sealed_risk_sensitive_learning_audit.json",
    "exp6859": "results/experiment_6859_first_party_tool_gap_receipt_wiring.json",
    "exp6861": V600_ARTIFACT_PATHS[V600_TASK_IDS[0]],
    "exp6862": V600_ARTIFACT_PATHS[V600_TASK_IDS[1]],
    "exp6863": V600_ARTIFACT_PATHS[V600_TASK_IDS[2]],
    "exp6864": V600_ARTIFACT_PATHS[V600_TASK_IDS[3]],
    "active_roadmap": "research-roadmap.yaml",
    "completed_roadmap": "research-complete.yaml",
    "conductor_log": "ops/conductor-log.md",
    "exclusion_manifest": "ops/exclusion_manifest.yaml",
    "arc_truncation_note": (
        "docs/research-notes/induce-shared-pool-truncation-2026-09-02.md"
    ),
    "adversarial_verifier": "scripts/adversarial_verify.py",
}
OWN_SOURCE_PATHS = {
    "module": "python/carnot/experiment_6865_v601_evidence_method_change_contract.py",
    "wrapper": "scripts/experiments/experiment_6865_v601_evidence_method_change_contract.py",
    "focused_tests": (
        "tests/python/test_experiment_6865_v601_evidence_method_change_contract.py"
    ),
    "spec": "openspec/capabilities/research-reporting/spec.md",
}

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
CONTRACT_FIELDS = (
    "canonical_tokenizer_payload_schema",
    "frozen_semantic_probe_contract",
    "frozen_contrast_bank_identity_manifest",
    "memory_observability_contract",
    "memory_transition_quarantine_contract",
    "arc_context_headroom_contract",
    "retired_scope_non_reopen_manifest",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "roadmap_task_rows",
    "conductor_gate_rows",
    "stored_vs_live_adversarial_rows",
    "v600_branch_dispositions",
    "tokenizer_receipt_schema_diff",
    *CONTRACT_FIELDS,
    "random_seed",
    "reproducibility_checksum",
    "v601_evidence_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "schema": "A versioned artifact schema prevents silent consumer drift.",
    "experiment_id": "The fixed identifier binds this contract to its roadmap task.",
    "run_date": "The operator date distinguishes this reduction from later evidence states.",
    "status": "The status records that the reducer reached a terminal artifact write.",
    "field_principles": "One reason per field keeps schema compliance tied to its purpose.",
    "preconditions_checked": "Exact source checks make a blocked reduction diagnosable.",
    "inference_substrate": "The substrate proves that no science or model run occurred.",
    "duration_s": "Measured wall time makes the local reduction operationally auditable.",
    "source_artifact_hashes": "Source hashes bind every frozen statement to exact bytes.",
    "roadmap_task_rows": "Task rows prevent a V600 task from leaving the evidence denominator.",
    "conductor_gate_rows": "Conductor rows preserve execution state apart from artifact claims.",
    "stored_vs_live_adversarial_rows": "Separate states expose verifier drift and unstamped warnings.",
    "v600_branch_dispositions": "Branch rows preserve null and blocked outcomes without promotion.",
    "tokenizer_receipt_schema_diff": "Exact field sets prove why the old hash comparison is invalid.",
    "canonical_tokenizer_payload_schema": "One semantic schema makes future tokenizer comparisons valid.",
    "frozen_semantic_probe_contract": "Frozen probes prevent comparison inputs from changing later.",
    "frozen_contrast_bank_identity_manifest": "Bank identities stop downstream cell filtering from becoming regeneration.",
    "memory_observability_contract": "The observability split prevents future outcomes from selecting actions.",
    "memory_transition_quarantine_contract": "Exact checks stop unsafe writes before admission.",
    "arc_context_headroom_contract": "Full headroom fields separate shared-pool truncation from token limits.",
    "retired_scope_non_reopen_manifest": "Retirement rows stop changed labels from reopening failed mechanisms.",
    "random_seed": "The fixed seed freezes any ordered manifest operation.",
    "reproducibility_checksum": "One checksum binds the deterministic output contract.",
    "v601_evidence_contract_ready_score": "Downstream gates consume this exact readiness field.",
    "gate_check_summary": "Expected and observed values explain every blocked verdict.",
    "verifier_is_oracle": "False prevents this evidence reducer from authorizing its own claims.",
    "verdict_class": "The closed class states the contract's evidence boundary.",
    "honest_verdict": "A terminal prefix lets the conductor classify the result safely.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for content identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 form for exact bytes."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str:
    """Hash one file, or keep a missing file visible as an empty digest."""

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def unwrap_principle(value: Any) -> Any:
    """Unwrap only the explicit two-key field convention.

    Ordinary dictionaries can contain a key named ``value``. Requiring both
    wrapper keys avoids erasing such dictionaries during evidence reduction.
    """

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one gate row with exact expected and observed values."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize checks while retaining the first exact failure."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def active_roadmap_check(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require the live V601 task and its exact output path."""

    milestone = str(unwrap_principle(roadmap.get("milestone")) or "")
    tasks_value = unwrap_principle(roadmap.get("tasks"))
    tasks = tasks_value if isinstance(tasks_value, list) else []
    task = next(
        (
            row
            for row in tasks
            if isinstance(row, Mapping) and unwrap_principle(row.get("id")) == EXP6865_TASK_ID
        ),
        None,
    )
    deliverable = unwrap_principle(task.get("deliverable")) if isinstance(task, Mapping) else None
    passed = milestone == EXPECTED_ACTIVE_MILESTONE and deliverable == OUTPUT_PATH.as_posix()
    observed: Any = milestone
    if milestone == EXPECTED_ACTIVE_MILESTONE and not passed:
        observed = {"task_id": EXP6865_TASK_ID, "deliverable": deliverable}
    expected: Any = EXPECTED_ACTIVE_MILESTONE
    if milestone == EXPECTED_ACTIVE_MILESTONE:
        expected = {"task_id": EXP6865_TASK_ID, "deliverable": OUTPUT_PATH.as_posix()}
    return _check("active_v601_roadmap", expected, observed, passed)


def _completed_v600_tasks(completed: Mapping[str, Any]) -> list[JsonDict]:
    """Read the archived V600 task rows without consulting current prompts."""

    milestones_value = unwrap_principle(completed.get("milestones"))
    milestones = milestones_value if isinstance(milestones_value, list) else []
    milestone = next(
        (
            row
            for row in milestones
            if isinstance(row, Mapping) and str(unwrap_principle(row.get("id"))) == V600_MILESTONE
        ),
        None,
    )
    tasks_value = unwrap_principle(milestone.get("tasks")) if isinstance(milestone, Mapping) else []
    return [dict(row) for row in tasks_value if isinstance(row, Mapping)] if isinstance(tasks_value, list) else []


def _read_document(path: Path) -> tuple[Any, str | None]:
    """Read one required source and report malformed content as evidence."""

    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None, "empty"
        if path.suffix == ".json":
            value = json.loads(text)
            if not isinstance(value, Mapping):
                return None, "json_object_required"
            return dict(value), None
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
            if not isinstance(value, Mapping):
                return None, "yaml_mapping_required"
            return dict(value), None
        return text, None
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def inspect_preconditions(
    root: Path, source_paths: Mapping[str, str] = DEFAULT_SOURCE_PATHS
) -> JsonDict:
    """Load all evidence inputs and fail closed on any unreadable source."""

    documents: dict[str, Any] = {}
    source_rows: list[JsonDict] = []
    source_hashes: dict[str, JsonDict] = {}
    unreadable: list[JsonDict] = []
    for source_id, relative in source_paths.items():
        path = root / relative
        value, error = _read_document(path)
        digest = sha256_path(path)
        row = {
            "source_id": source_id,
            "path": relative,
            "readable": error is None,
            "sha256": digest,
            "error": error,
        }
        source_rows.append(row)
        source_hashes[source_id] = {"path": relative, "sha256": digest}
        if error is None:
            documents[source_id] = value
        else:
            unreadable.append(row)

    checks = [
        _check(
            "required_source_readability",
            "all required sources readable",
            [
                {"source_id": row["source_id"], "path": row["path"], "error": row["error"]}
                for row in unreadable
            ],
            not unreadable,
        )
    ]
    roadmap = documents.get("active_roadmap")
    checks.append(active_roadmap_check(roadmap if isinstance(roadmap, Mapping) else {}))

    completed = documents.get("completed_roadmap")
    archived_tasks = _completed_v600_tasks(completed if isinstance(completed, Mapping) else {})
    observed_ids = [str(unwrap_principle(row.get("id")) or "") for row in archived_tasks]
    checks.append(
        _check("completed_v600_task_set", list(V600_TASK_IDS), observed_ids, observed_ids == list(V600_TASK_IDS))
    )

    conductor = documents.get("conductor_log")
    conductor_text = conductor if isinstance(conductor, str) else ""
    conductor_markers = [
        "Milestone 2026.09.600 activated",
        "V600 branch retirement and evidence contract",
        "Dual-side exact semantic contrast bank",
        "Tokenizer-aware nuisance and split preregistration",
        "Three-family semantic contrast scoring stream",
    ]
    missing_markers = [marker for marker in conductor_markers if marker not in conductor_text]
    checks.append(_check("v600_conductor_record", [], missing_markers, not missing_markers))

    exclusion = documents.get("exclusion_manifest")
    extras = exclusion.get("retired_extras") if isinstance(exclusion, Mapping) else []
    retired_ids = {
        str(row.get("id")) for row in extras if isinstance(row, Mapping) and row.get("id")
    } if isinstance(extras, list) else set()
    checks.append(
        _check(
            "max_token_retirement_record",
            "arc-induce-completion-budget-raise",
            "arc-induce-completion-budget-raise" if "arc-induce-completion-budget-raise" in retired_ids else None,
            "arc-induce-completion-budget-raise" in retired_ids,
        )
    )

    note = documents.get("arc_truncation_note")
    note_text = note if isinstance(note, str) else ""
    note_markers = [
        "shared context pool",
        "raising max_tokens would make this worse",
        "n_ctx=49152",
    ]
    note_casefolded = note_text.casefold()
    missing_note_markers = [marker for marker in note_markers if marker not in note_casefolded]
    checks.append(_check("arc_truncation_note_contract", [], missing_note_markers, not missing_note_markers))
    return {
        "documents": documents,
        "source_rows": source_rows,
        "source_hashes": source_hashes,
        "gate_check_summary": _gate_summary(checks),
    }


def read_terminal_verdict(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Read a terminal verdict and recognize conductor gate artifacts."""

    honest = str(unwrap_principle(payload.get("honest_verdict")) or "")
    verdict_class = unwrap_principle(payload.get("verdict_class"))
    if verdict_class in CLOSED_VERDICT_CLASSES:
        return honest, str(verdict_class)
    if payload.get("schema") == "blocked_gate_check_v1" or (
        payload.get("status") == "blocked" and payload.get("blocked_at_layer")
    ):
        return honest, "blocked"
    return honest, "partial"


def normalized_gate_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize source gates without changing expected or observed values."""

    gates_value = unwrap_principle(payload.get("gates_evaluated"))
    if not isinstance(gates_value, list):
        summary = unwrap_principle(payload.get("gate_check_summary"))
        gates_value = summary.get("checks") if isinstance(summary, Mapping) else []
    rows: list[JsonDict] = []
    for gate in gates_value if isinstance(gates_value, list) else []:
        if not isinstance(gate, Mapping):
            continue
        observed = gate.get("actual") if "actual" in gate else gate.get("observed")
        rows.append(
            {
                "upstream": gate.get("upstream"),
                "artifact_field": gate.get("artifact_field") or gate.get("check"),
                "operator": gate.get("op") or gate.get("operator") or "check",
                "expected": deepcopy(gate.get("expected")),
                "observed": deepcopy(observed),
                "passed": gate.get("passed"),
            }
        )
    return rows


def build_roadmap_task_rows(
    completed: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Join archived roadmap rows to their exact terminal artifacts."""

    archived = {str(unwrap_principle(row.get("id"))): row for row in _completed_v600_tasks(completed)}
    rows: list[JsonDict] = []
    for task_id in V600_TASK_IDS:
        roadmap = archived.get(task_id, {})
        payload = payloads.get(task_id, {})
        honest, verdict_class = read_terminal_verdict(payload)
        rows.append(
            {
                "task_id": task_id,
                "title": unwrap_principle(roadmap.get("title")),
                "deliverable": unwrap_principle(roadmap.get("deliverable")),
                "roadmap_result": unwrap_principle(roadmap.get("result")),
                "honest_verdict": honest,
                "verdict_class": verdict_class,
                "gate_fields": normalized_gate_rows(payload),
                "stored_flagged_adversarial": (
                    unwrap_principle(payload.get("flagged_adversarial"))
                    if "flagged_adversarial" in payload
                    else None
                ),
                "source_sha256": "",
            }
        )
    return rows


def _conductor_records(text: str) -> list[JsonDict]:
    """Parse the four-column conductor table without reading free prose."""

    pattern = re.compile(
        r"^\|\s*(?P<timestamp>[^|]+?)\s*\|\s*(?P<title>[^|]+?)\s*\|"
        r"\s*(?P<status>[^|]+?)\s*\|\s*(?P<detail>[^|]*?)\s*\|$"
    )
    rows: list[JsonDict] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            rows.append({key: value.strip() for key, value in match.groupdict().items()})
    return rows


def build_conductor_gate_rows(text: str, task_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Preserve each V600 terminal conductor state and its gate fields."""

    records = _conductor_records(text)
    rows: list[JsonDict] = []
    for task in task_rows:
        title = str(task.get("title") or "")
        record = next(
            (
                row
                for row in records
                if title.startswith(row["title"]) or row["title"].startswith(title)
            ),
            {},
        )
        rows.append(
            {
                "task_id": task.get("task_id"),
                "timestamp": record.get("timestamp"),
                "conductor_status": record.get("status", "missing"),
                "conductor_detail": record.get("detail"),
                "is_gate_block": record.get("status") == "GATE_BLOCK",
                "gate_fields": deepcopy(task.get("gate_fields") or []),
            }
        )
    return rows


def _load_validator(root: Path) -> Any:
    """Load the exact verifier source that is included in the source hashes."""

    path = root / DEFAULT_SOURCE_PATHS["adversarial_verifier"]
    module_name = "_exp6865_adversarial_verify"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("adversarial verifier import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact


def collect_live_adversarial_reports(root: Path) -> dict[str, JsonDict]:
    """Re-run the current read-only verifier for each V600 artifact."""

    verify_artifact = _load_validator(root)
    reports: dict[str, JsonDict] = {}
    for task_id, relative in V600_ARTIFACT_PATHS.items():
        reports[task_id] = dict(verify_artifact(root / relative))
    return reports


def stored_vs_live_adversarial_rows(
    payloads: Mapping[str, Mapping[str, Any]], reports: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Keep stored determinations separate from current verifier output."""

    rows: list[JsonDict] = []
    for task_id in V600_TASK_IDS:
        payload = payloads.get(task_id, {})
        stored = (
            unwrap_principle(payload.get("flagged_adversarial"))
            if "flagged_adversarial" in payload
            else None
        )
        stored_status = "flagged" if stored is True else "clear" if stored is False else "unstamped"
        report = reports.get(task_id, {})
        flags = [dict(row) for row in report.get("flags", []) if isinstance(row, Mapping)]
        severities = {str(row.get("severity") or "").lower() for row in flags}
        live_status = "critical" if "critical" in severities else "warn" if flags else "clean"
        disagreement = (stored is True and live_status == "clean") or (
            stored is not True and live_status != "clean"
        )
        rows.append(
            {
                "task_id": task_id,
                "stored_status": stored_status,
                "stored_flagged_adversarial": stored,
                "stored_corrigendum_pending": deepcopy(payload.get("corrigendum_pending")),
                "live_status": live_status,
                "live_flag_count": len(flags),
                "live_flags": flags,
                "live_gate_version": report.get("gate_version"),
                "disagreement": disagreement,
            }
        )
    return rows


def _receipt_fields(payload: Mapping[str, Any]) -> tuple[list[str], list[JsonDict]]:
    """Return the exact stored field union and one field set per model."""

    receipts_value = unwrap_principle(payload.get("tokenizer_receipts"))
    receipts = receipts_value if isinstance(receipts_value, list) else []
    rows = [dict(row) for row in receipts if isinstance(row, Mapping)]
    field_sets = [
        {"hf_id": row.get("hf_id"), "fields": sorted(str(key) for key in row)} for row in rows
    ]
    fields = sorted({str(key) for row in rows for key in row})
    return fields, field_sets


def tokenizer_receipt_schema_diff(exp6850: Mapping[str, Any], exp6863: Mapping[str, Any]) -> JsonDict:
    """Prove schema mismatch without interpreting the tokenizer hashes."""

    fields_6850, model_fields_6850 = _receipt_fields(exp6850)
    fields_6863, model_fields_6863 = _receipt_fields(exp6863)
    set_6850 = set(fields_6850)
    set_6863 = set(fields_6863)
    receipts_6850 = {
        str(row.get("hf_id")): row
        for row in unwrap_principle(exp6850.get("tokenizer_receipts")) or []
        if isinstance(row, Mapping)
    }
    receipts_6863 = {
        str(row.get("hf_id")): row
        for row in unwrap_principle(exp6863.get("tokenizer_receipts")) or []
        if isinstance(row, Mapping)
    }
    gemma_rows = []
    for hf_id in sorted(set(receipts_6850) & set(receipts_6863)):
        if "gemma" not in hf_id.lower():
            continue
        gemma_rows.append(
            {
                "hf_id": hf_id,
                "exp6850_tokenizer_sha256": receipts_6850[hf_id].get("tokenizer_sha256"),
                "exp6863_tokenizer_sha256": receipts_6863[hf_id].get("tokenizer_sha256"),
                "comparison_valid": False,
                "semantic_relation": "undetermined",
            }
        )
    identical = fields_6850 == fields_6863 and bool(fields_6850)
    return {
        "exp6850_receipt_fields": fields_6850,
        "exp6863_receipt_fields": fields_6863,
        "per_model_exp6850_field_sets": model_fields_6850,
        "per_model_exp6863_field_sets": model_fields_6863,
        "only_in_exp6850": sorted(set_6850 - set_6863),
        "only_in_exp6863": sorted(set_6863 - set_6850),
        "exp6850_field_set_sha256": sha256_bytes(canonical_json(fields_6850)),
        "exp6863_field_set_sha256": sha256_bytes(canonical_json(fields_6863)),
        "schemas_identical": identical,
        "hash_comparison_valid": identical,
        "tokenizer_relation": (
            "not_evaluated" if identical else "undetermined_incomparable_schemas"
        ),
        "gemma_hash_witnesses": gemma_rows,
        "claim_boundary": "Receipt-schema mismatch proves only that the old comparison is invalid.",
    }


def _digest(source_hashes: Mapping[str, Any], source_id: str) -> str:
    """Read a digest from either the compact or path-annotated hash form."""

    value = source_hashes.get(source_id, "")
    if isinstance(value, Mapping):
        return str(value.get("sha256") or "")
    return str(value or "")


def _contract_sources(source_hashes: Mapping[str, Any], source_ids: Sequence[str]) -> JsonDict:
    """Bind one contract to only the sources that define it."""

    return {source_id: _digest(source_hashes, source_id) for source_id in source_ids}


def _with_contract_hash(contract: JsonDict) -> JsonDict:
    """Attach a hash that excludes only its own identity field."""

    value = deepcopy(contract)
    value.pop("contract_sha256", None)
    contract["contract_sha256"] = sha256_bytes(canonical_json(value))
    return contract


def contract_hash_valid(contract: Mapping[str, Any]) -> bool:
    """Replay one self-contained contract identity."""

    value = deepcopy(dict(contract))
    expected = value.pop("contract_sha256", None)
    return expected == sha256_bytes(canonical_json(value))


def canonical_tokenizer_schema(source_hashes: Mapping[str, Any]) -> JsonDict:
    """Freeze the semantic tokenizer payload and its exclusion boundary."""

    return _with_contract_hash(
        {
            "schema_id": "carnot.canonical_tokenizer_payload.v1",
            "version": "v1",
            "required_payload_fields": [
                "semantic_vocabulary_metadata",
                "special_token_ids",
                "add_bos_behavior",
                "chat_template_identity",
                "tokenization_settings",
                "frozen_probe_outputs",
            ],
            "semantic_vocabulary_metadata_fields": [
                "tokenizer_model",
                "tokenizer_pretokenizer",
                "vocabulary_size",
                "token_pieces_sha256",
                "token_scores_sha256",
                "token_types_sha256",
                "merges_sha256",
            ],
            "special_token_id_fields": [
                "bos_token_id",
                "eos_token_id",
                "unknown_token_id",
                "padding_token_id",
                "mask_token_id",
                "separator_token_id",
            ],
            "add_bos_behavior_fields": ["metadata_default", "requested", "observed_prefix_ids"],
            "chat_template_fields": ["present", "template_sha256"],
            "tokenization_setting_fields": [
                "add_bos",
                "special",
                "utf8_error_mode",
                "unicode_normalization",
            ],
            "frozen_probe_output_fields": [
                "probe_id",
                "text_utf8_sha256",
                "settings",
                "token_ids",
            ],
            "excluded_field_classes": [
                "timestamps",
                "paths",
                "prose_detail",
                "wrapper_only_fields",
                "nested_receipt_hashes",
            ],
            "canonicalization": {
                "encoding": "UTF-8",
                "json_keys": "sorted",
                "json_separators": [",", ":"],
                "hash": "SHA-256",
                "schema_version_must_match_on_both_sides": True,
            },
            "source_hashes": _contract_sources(source_hashes, ["exp6850", "exp6863"]),
        }
    )


def frozen_semantic_probe_contract(
    exp6850: Mapping[str, Any], exp6863: Mapping[str, Any], source_hashes: Mapping[str, Any]
) -> JsonDict:
    """Freeze legacy outputs as witnesses, not as canonical equality evidence."""

    witnesses: list[JsonDict] = []
    for source_id, payload in (("exp6850", exp6850), ("exp6863", exp6863)):
        receipts = unwrap_principle(payload.get("tokenizer_receipts"))
        for receipt in receipts if isinstance(receipts, list) else []:
            if not isinstance(receipt, Mapping):
                continue
            witnesses.append(
                {
                    "source_experiment": source_id,
                    "hf_id": receipt.get("hf_id"),
                    "probe_token_ids": deepcopy(receipt.get("probe_token_ids")),
                    "stored_tokenize_settings": deepcopy(receipt.get("tokenize_settings")),
                    "canonical_output": False,
                    "use": "legacy_schema_mismatch_witness_only",
                }
            )
    return _with_contract_hash(
        {
            "schema": "carnot.frozen_semantic_probe_contract.v1",
            "probe_text": TOKENIZER_PROBE,
            "probe_text_utf8_sha256": sha256_bytes(TOKENIZER_PROBE.encode("utf-8")),
            "required_probe_classes": [
                "whitespace",
                "newline",
                "tab",
                "unicode_nfc",
                "unicode_nfd",
                "punctuation",
                "label_swap",
                "control_like_text",
                "literal_special_token_text",
                "empty_text",
                "chat_template_boundary",
                "exp6862_sequence_templates",
            ],
            "canonical_outputs_must_be_frozen_before_scoring": True,
            "legacy_output_witnesses": witnesses,
            "source_hashes": _contract_sources(source_hashes, ["exp6850", "exp6863"]),
        }
    )


def frozen_contrast_bank_manifest(exp6862: Mapping[str, Any], source_sha256: str) -> JsonDict:
    """Freeze accepted bank identities without regenerating a single group."""

    groups_value = unwrap_principle(exp6862.get("semantic_contrast_group_manifest"))
    groups = groups_value if isinstance(groups_value, list) else []
    identities = [
        {
            "group_id": row.get("group_id"),
            "semantic_identity": row.get("semantic_identity"),
            "contrast_id": row.get("contrast_id"),
            "source_pair_id": row.get("source_pair_id"),
            "family": row.get("family"),
        }
        for row in groups
        if isinstance(row, Mapping)
    ]
    return _with_contract_hash(
        {
            "schema": "carnot.frozen_contrast_bank_identity_manifest.v1",
            "source_artifact": V600_ARTIFACT_PATHS[V600_TASK_IDS[1]],
            "source_artifact_sha256": source_sha256,
            "accepted_group_count": len(identities),
            "group_identities": identities,
            "identity_manifest_sha256": sha256_bytes(canonical_json(identities)),
            "cell_filtering_allowed": True,
            "bank_regeneration_allowed": False,
            "identity_relabeling_allowed": False,
        }
    )


def memory_observability_contract(
    exp6856: Mapping[str, Any], exp6861: Mapping[str, Any], source_hashes: Mapping[str, Any]
) -> JsonDict:
    """Freeze the action-time boundary and preserve conflicting write counts."""

    support = unwrap_principle(exp6856.get("counterfactual_support_results"))
    support = support if isinstance(support, Mapping) else {}
    branches = unwrap_principle(exp6861.get("terminal_branch_manifest"))
    branch_rows = branches if isinstance(branches, list) else []
    self_learning = next(
        (row for row in branch_rows if isinstance(row, Mapping) and row.get("branch") == "self_learning"),
        {},
    )
    count_6856 = support.get("harmful_write_count")
    count_6861 = self_learning.get("harmful_write_count")
    return _with_contract_hash(
        {
            "schema": "carnot.memory_observability_contract.v1",
            "decision_time_observable_fields": [
                "event_identity",
                "source_identity",
                "source_content_sha256",
                "decision_sequence_index",
                "available_actions",
                "pre_action_memory_state_sha256",
                "prior_reliability_state",
                "age_at_decision",
                "capacity_at_decision",
                "known_correction_status",
                "exact_pre_action_compatibility",
                "relevance_at_decision",
                "uncertainty_at_decision",
                "action_cost",
            ],
            "offline_supervision_only_fields": [
                "later_exact_outcome",
                "outcome_signed_direction",
                "feedback_reveal_sequence_index",
                "memory_effect_class",
                "counterfactual_benefit",
                "helpful_or_harmful_label",
                "audit_verdict",
                "future_correction",
                "held_split_label",
                "task_order_label",
            ],
            "same_event_outcome_may_select_action": False,
            "reliability_update_boundary": {
                "initial_state": "finite symmetric zero matrix over frozen source and action nodes",
                "update_time": "after action close and later exact outcome reveal",
                "same_event_feedback_allowed": False,
                "entry_delta_bound_field": "max_abs_entry_delta_per_event",
                "spectral_delta_bound_field": "max_spectral_norm_delta_per_event",
                "bounds_frozen_before_replay": True,
                "state_checksum_before_and_after_required": True,
            },
            "v599_overall_abstention_rate": self_learning.get("abstention_rate"),
            "preserved_harmful_write_counts": {
                "exp6856_fresh_reducer": count_6856,
                "exp6861_branch_contract": count_6861,
            },
            "harmful_write_count_disagreement_preserved": count_6856 != count_6861,
            "source_hashes": _contract_sources(source_hashes, ["exp6856", "exp6861", "active_roadmap"]),
        }
    )


def memory_transition_quarantine_contract(source_hashes: Mapping[str, Any]) -> JsonDict:
    """Freeze exact write-admission checks and their fail-closed action."""

    conditions = {
        "coverage": "Every affected key and authority is present in the transition receipt.",
        "preservation": "Every unrelated key remains byte-identical.",
        "source_faithfulness": "The stored value matches the cited exact source bytes.",
        "provenance": "Source, event, action, parent, and content identities are complete.",
        "old_family_retention": "Every frozen old-family anchor remains exact.",
        "delayed_invalidation": "A later exact correction invalidates the affected write.",
        "tombstone_non_reappearance": "A tombstoned write cannot return during replay.",
        "replay_equivalence": "Fresh replay produces the same state bytes.",
        "restart_equivalence": "Restart restores the same state bytes and checksum.",
        "rollback_byte_exact": "Rollback restores the exact parent bytes and checksum.",
    }
    return _with_contract_hash(
        {
            "schema": "carnot.memory_transition_quarantine_contract.v1",
            "exact_transition_checks": [
                {"check_id": check_id, "condition": condition, "failure_action": "quarantine"}
                for check_id, condition in conditions.items()
            ],
            "admission_rule": "all exact transition checks must pass before a write is visible",
            "quarantined_write_may_influence_decision": False,
            "source_hashes": _contract_sources(source_hashes, ["exp6856", "exp6861", "active_roadmap"]),
        }
    )


def arc_context_headroom_contract(source_hashes: Mapping[str, Any]) -> JsonDict:
    """Freeze the measurements needed to size the shared context pool."""

    required_fields = [
        "prompt_sha256",
        "prompt_token_count",
        "prompt_truncated",
        "slot_id",
        "slot_count",
        "slot_state_receipt_sha256",
        "measured_slot_context_capacity_tokens",
        "actual_server_n_ctx_tokens",
        "requested_completion_tokens",
        "generated_reasoning_token_count",
        "generated_reasoning_character_count",
        "final_channel_token_count",
        "final_channel_character_count",
        "engine_field_present",
        "stop_type",
        "shared_pool_truncated",
        "raw_completion_length",
        "full_limit_diagnostic",
        "vram_free_bytes_before",
        "vram_used_bytes_before",
        "model_resident_bytes",
        "kv_cache_bytes",
        "measured_generation_headroom_tokens",
        "required_context_pool_tokens",
        "headroom_deficit_tokens",
        "headroom_formula_version",
    ]
    return _with_contract_hash(
        {
            "schema": "carnot.arc_context_headroom_contract.v1",
            "required_attempt_fields": required_fields,
            "observed_evidence": {
                "actual_server_n_ctx_tokens": 49152,
                "slot_count": 4,
                "requested_completion_tokens": 26800,
                "generated_reasoning_token_counts": [18431, 2996, 4066],
                "last_stop_type": "limit",
                "last_prompt_truncated": False,
                "last_raw_completion_length": 12151,
                "r11l_final_channel_characters": 3457,
                "stored_error_prefix_characters": 29,
                "stored_diagnostic_clip_characters": 150,
                "stored_error_characters": 179,
                "full_diagnostic_characters": 216,
            },
            "diagnostic_storage_rule": "Store the full typed diagnostic outside any prose clip.",
            "slot_capacity_may_be_inferred_by_division": False,
            "max_token_only_repair_allowed": False,
            "context_resize_requires": [
                "measured prompt tokens",
                "measured slot capacity",
                "actual server n_ctx",
                "measured VRAM headroom",
                "measured generation headroom",
            ],
            "headroom_formula": (
                "measured_generation_headroom_tokens = measured_slot_context_capacity_tokens "
                "- prompt_token_count - reserved_runtime_tokens"
            ),
            "claim_boundary": "This contract does not claim that a larger context pool improves solve rate.",
            "source_hashes": _contract_sources(
                source_hashes, ["arc_truncation_note", "exclusion_manifest", "active_roadmap"]
            ),
        }
    )


def retired_scope_violations(
    proposed_changes: Sequence[str], exclusion_manifest: Mapping[str, Any]
) -> list[JsonDict]:
    """Match proposed mechanisms against explicit retired-scope phrases."""

    extras = exclusion_manifest.get("retired_extras")
    rows = extras if isinstance(extras, list) else []
    violations: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        patterns = []
        for field in ("blocked_patterns", "blocked_same_mechanism_retries"):
            value = row.get(field)
            if isinstance(value, list):
                patterns.extend(str(item) for item in value)
        for proposed in proposed_changes:
            normalized_proposed = " ".join(proposed.lower().split())
            for pattern in patterns:
                normalized_pattern = " ".join(pattern.lower().split())
                named_levers = re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", pattern)
                named_lever_match = any(lever in proposed for lever in named_levers)
                if normalized_pattern and (
                    normalized_pattern in normalized_proposed or named_lever_match
                ):
                    violations.append(
                        {
                            "retired_scope_id": row.get("id") or row.get("scope_key"),
                            "blocked_pattern": pattern,
                            "proposed_change": proposed,
                        }
                    )
    return violations


def retired_scope_manifest(
    exp6861: Mapping[str, Any], exclusion: Mapping[str, Any], source_hashes: Mapping[str, Any]
) -> JsonDict:
    """Freeze V600 retirements and the exclusion-manifest context remedy."""

    extras = exclusion.get("retired_extras")
    extra_rows = extras if isinstance(extras, list) else []
    max_token = next(
        (
            dict(row)
            for row in extra_rows
            if isinstance(row, Mapping) and row.get("id") == "arc-induce-completion-budget-raise"
        ),
        {},
    )
    retired_value = unwrap_principle(exp6861.get("retired_mechanism_manifest"))
    retired = [dict(row) for row in retired_value if isinstance(row, Mapping)] if isinstance(retired_value, list) else []
    proposed_changes = [
        "canonical tokenizer payload requalification under one schema",
        "filter cells from the frozen Exp6862 bank without regeneration",
        "bounded reliability state updated after later exact outcomes",
        "measure prompt, slot, n_ctx, VRAM, and shared-pool headroom",
    ]
    violations = retired_scope_violations(proposed_changes, exclusion)
    return _with_contract_hash(
        {
            "schema": "carnot.retired_scope_non_reopen_manifest.v1",
            "v600_retired_mechanisms": retired,
            "max_token_only_arc_remedy": {
                "retired_scope_id": max_token.get("id"),
                "operator_reopen_required": max_token.get("operator_reopen_required"),
                "blocked_patterns": deepcopy(max_token.get("blocked_patterns") or []),
                "allowed": False,
            },
            "contrast_bank_regeneration": {"allowed": False, "source": "Exp6862 identity contract"},
            "proposed_changed_boundaries": proposed_changes,
            "proposed_boundary_violations": violations,
            "source_hashes": _contract_sources(source_hashes, ["exp6861", "exclusion_manifest", "active_roadmap"]),
        }
    )


def v600_branch_dispositions(task_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """State each V600 branch without turning infrastructure into science."""

    branch_names = ("evidence_contract", "contrast_bank", "tokenizer_preregistration", "scoring_stream")
    dispositions = []
    for branch, row in zip(branch_names, task_rows, strict=False):
        dispositions.append(
            {
                "branch": branch,
                "task_id": row.get("task_id"),
                "honest_verdict": row.get("honest_verdict"),
                "verdict_class": row.get("verdict_class"),
                "roadmap_result": row.get("roadmap_result"),
                "scientific_claim_eligible": False,
                "disposition": (
                    "frozen_infrastructure"
                    if branch in {"evidence_contract", "contrast_bank"}
                    else "terminal_blocked"
                ),
            }
        )
    return dispositions


def _all_contract_sources_hashed(contracts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every declared source digest to be non-empty."""

    for contract in contracts:
        sources = contract.get("source_hashes")
        if isinstance(sources, Mapping) and any(not value for value in sources.values()):
            return False
    return True


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding wall time and the hash itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(payload))


def build_artifact(
    root: Path,
    run_date: str,
    *,
    live_reports: Mapping[str, Mapping[str, Any]] | None = None,
    source_paths: Mapping[str, str] = DEFAULT_SOURCE_PATHS,
) -> JsonDict:
    """Read, reduce, and validate the V601 evidence boundary in memory."""

    started = time.perf_counter()
    preconditions = inspect_preconditions(root, source_paths)
    documents = preconditions["documents"]
    source_hashes = dict(preconditions["source_hashes"])
    for source_id, relative in OWN_SOURCE_PATHS.items():
        source_hashes[source_id] = {"path": relative, "sha256": sha256_path(root / relative)}

    completed = documents.get("completed_roadmap")
    payloads = {
        task_id: documents.get(f"exp{6861 + index}", {})
        for index, task_id in enumerate(V600_TASK_IDS)
    }
    payloads = {
        task_id: payload if isinstance(payload, Mapping) else {}
        for task_id, payload in payloads.items()
    }
    task_rows = build_roadmap_task_rows(completed if isinstance(completed, Mapping) else {}, payloads)
    for row in task_rows:
        short_id = re.match(r"exp\d+", str(row["task_id"]))
        source_id = short_id.group(0) if short_id else ""
        row["source_sha256"] = _digest(source_hashes, source_id)

    conductor = documents.get("conductor_log")
    conductor_rows = build_conductor_gate_rows(conductor if isinstance(conductor, str) else "", task_rows)
    if live_reports is None:
        live_reports = collect_live_adversarial_reports(root) if preconditions["gate_check_summary"]["passed"] else {}
    adversarial_rows = stored_vs_live_adversarial_rows(payloads, live_reports)

    exp6850 = documents.get("exp6850") if isinstance(documents.get("exp6850"), Mapping) else {}
    exp6856 = documents.get("exp6856") if isinstance(documents.get("exp6856"), Mapping) else {}
    exp6861 = documents.get("exp6861") if isinstance(documents.get("exp6861"), Mapping) else {}
    exp6862 = documents.get("exp6862") if isinstance(documents.get("exp6862"), Mapping) else {}
    exp6863 = documents.get("exp6863") if isinstance(documents.get("exp6863"), Mapping) else {}
    exclusion = documents.get("exclusion_manifest") if isinstance(documents.get("exclusion_manifest"), Mapping) else {}

    tokenizer_diff = tokenizer_receipt_schema_diff(exp6850, exp6863)
    tokenizer_schema = canonical_tokenizer_schema(source_hashes)
    probe_contract = frozen_semantic_probe_contract(exp6850, exp6863, source_hashes)
    bank_manifest = frozen_contrast_bank_manifest(exp6862, _digest(source_hashes, "exp6862"))
    observability = memory_observability_contract(exp6856, exp6861, source_hashes)
    quarantine = memory_transition_quarantine_contract(source_hashes)
    arc_contract = arc_context_headroom_contract(source_hashes)
    retired_manifest = retired_scope_manifest(exp6861, exclusion, source_hashes)
    contracts = [
        tokenizer_schema,
        probe_contract,
        bank_manifest,
        observability,
        quarantine,
        arc_contract,
        retired_manifest,
    ]

    checks = list(preconditions["gate_check_summary"]["checks"])
    expected_deliverables = [V600_ARTIFACT_PATHS[task_id] for task_id in V600_TASK_IDS]
    checks.append(
        _check(
            "v600_task_reconstruction",
            {"task_ids": list(V600_TASK_IDS), "deliverables": expected_deliverables},
            {
                "task_ids": [row.get("task_id") for row in task_rows],
                "deliverables": [row.get("deliverable") for row in task_rows],
            },
            [row.get("task_id") for row in task_rows] == list(V600_TASK_IDS)
            and [row.get("deliverable") for row in task_rows] == expected_deliverables,
        )
    )
    checks.append(
        _check(
            "v600_conductor_terminal_rows",
            4,
            len([row for row in conductor_rows if row.get("conductor_status") != "missing"]),
            len(conductor_rows) == 4
            and all(row.get("conductor_status") != "missing" for row in conductor_rows),
        )
    )
    checks.append(
        _check(
            "tokenizer_receipt_schema_mismatch_proven",
            False,
            tokenizer_diff["schemas_identical"],
            tokenizer_diff["schemas_identical"] is False
            and tokenizer_diff["hash_comparison_valid"] is False,
        )
    )
    group_ids = [row.get("group_id") for row in bank_manifest["group_identities"]]
    checks.append(
        _check(
            "frozen_contrast_bank_identities",
            {"count": 100, "unique": 100},
            {"count": len(group_ids), "unique": len(set(group_ids))},
            len(group_ids) == 100 and len(set(group_ids)) == 100,
        )
    )
    bad_contracts = [field for field, contract in zip(CONTRACT_FIELDS, contracts, strict=True) if not contract_hash_valid(contract)]
    checks.append(_check("contract_hash_replay", [], bad_contracts, not bad_contracts))
    checks.append(
        _check(
            "contract_source_hashes",
            True,
            _all_contract_sources_hashed(contracts),
            _all_contract_sources_hashed(contracts),
        )
    )
    violations = retired_manifest["proposed_boundary_violations"]
    checks.append(_check("exclusion_safe_changed_boundaries", [], violations, not violations))
    critical_rows = [row["task_id"] for row in adversarial_rows if row.get("live_status") == "critical"]
    checks.append(_check("no_live_critical_v600_source", [], critical_rows, not critical_rows))
    gate_summary = _gate_summary(checks)
    ready = int(gate_summary["passed"])

    artifact: JsonDict = {
        "schema": "carnot.experiment_6865.v601_evidence_method_change_contract.v1",
        "experiment_id": 6865,
        "run_date": run_date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": {
            "sources": preconditions["source_rows"],
            "passed": preconditions["gate_check_summary"]["passed"],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.perf_counter() - started, 6),
        "source_artifact_hashes": source_hashes,
        "roadmap_task_rows": task_rows,
        "conductor_gate_rows": conductor_rows,
        "stored_vs_live_adversarial_rows": adversarial_rows,
        "v600_branch_dispositions": v600_branch_dispositions(task_rows),
        "tokenizer_receipt_schema_diff": tokenizer_diff,
        "canonical_tokenizer_payload_schema": tokenizer_schema,
        "frozen_semantic_probe_contract": probe_contract,
        "frozen_contrast_bank_identity_manifest": bank_manifest,
        "memory_observability_contract": observability,
        "memory_transition_quarantine_contract": quarantine,
        "arc_context_headroom_contract": arc_contract,
        "retired_scope_non_reopen_manifest": retired_manifest,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v601_evidence_contract_ready_score": ready,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": (
            "complete_positive_v601_evidence_method_change_contract_ready"
            if ready
            else "complete_blocked_v601_evidence_method_change_contract"
        ),
    }
    artifact["field_principles"] = {field: FIELD_PRINCIPLES[field] for field in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay the output contract before any result file is replaced."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    extra = sorted(set(artifact) - REQUIRED_ARTIFACT_FIELDS)
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    if extra:
        errors.append(f"unexpected_fields:{','.join(extra)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    for field in CONTRACT_FIELDS:
        contract = artifact.get(field)
        if not isinstance(contract, Mapping) or not contract_hash_valid(contract):
            errors.append(f"contract_hash_mismatch:{field}")
    ready = artifact.get("v601_evidence_contract_ready_score")
    if ready not in {0, 1}:
        errors.append("invalid_ready_score")
    if ready == 1:
        bank = artifact.get("frozen_contrast_bank_identity_manifest")
        if not isinstance(bank, Mapping) or bank.get("accepted_group_count") != 100:
            errors.append("ready_contract_requires_100_bank_groups")
        gate = artifact.get("gate_check_summary")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append("ready_contract_has_failed_gate")
        if artifact.get("verdict_class") != "positive":
            errors.append("ready_contract_must_be_positive")
    if ready == 0:
        gate = artifact.get("gate_check_summary")
        if not isinstance(gate, Mapping) or not gate.get("failed_check"):
            errors.append("blocked_contract_requires_exact_failed_check")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_contract_must_use_blocked_class")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after a complete JSON document exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the operator date, validate the contract, and write it once."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", default=OUTPUT_PATH.as_posix())
    args = parser.parse_args(argv)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    output = Path(args.output)
    write_atomic(output, artifact)
    print(
        json.dumps(
            {
                "result": output.as_posix(),
                "ready_score": artifact["v601_evidence_contract_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
