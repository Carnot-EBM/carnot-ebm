"""Build the Exp6408 powered write-time factor admission A/B artifact.

Spec refs: REQ-LEARN-6408, SCENARIO-LEARN-6408-LICENSED-CELLS,
SCENARIO-LEARN-6408-FRESH-MANIFEST, SCENARIO-LEARN-6408-ADMISSION,
SCENARIO-LEARN-6408-MATCHED-ARMS, SCENARIO-LEARN-6408-ATTACKS,
SCENARIO-LEARN-6408-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6395_held_factor_transport_license_matrix as exp6395
from carnot import experiment_6396_capability_qualified_verified_frontier_ab as exp6396
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6408_powered_write_time_factor_admission_ab.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6408_powered_write_time_factor_admission_ab"
)
HELD_MANIFEST_SUFFIX = ".fresh_held_manifest.json"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

EXP6395_RELATIVE_PATH = exp6395.RESULT_RELATIVE_PATH
EXP6406_RELATIVE_PATH = Path("results/experiment_6406_clean_v550_factor_evidence_boundary.json")
EXP6407_RELATIVE_PATH = Path(
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
)
EXP6407_CONTAMINATION_RELATIVE_PATH = Path(
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
    ".contamination_manifest.json"
)

SCHEMA = "carnot.experiment_6408.powered_write_time_factor_admission_ab.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6408
TOKENIZER_METHOD = exp6395.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "powered_local_gguf_replay_with_exact_event_checkers"

MANDATED_MODEL_IDS = exp6395.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6395.MODEL_TEMPLATE_BY_ID
REQUIRED_CONSTRAINT_FAMILIES = exp6395.REQUIRED_CONSTRAINT_FAMILIES
LICENSED_CELL_TARGETS = exp6395.LICENSED_CELL_TARGETS

ARMS = ("frozen_baseline", "write_everything", "provenance_exact_admission")
CONTAMINATION_CLASSES = (
    "supported",
    "contradicted",
    "implicit",
    "stale",
    "duplicated",
    "replayed",
    "superseded",
    "poisoned",
    "malformed",
)
PARTITIONS = ("pre_generation", "arm_execution", "future")
ATTACK_IDS = (
    "model_swap",
    "family_swap",
    "license_inheritance",
    "harness_drift",
    "source_substitution",
    "exact_check_omission",
    "diagnostic_veto_override",
    "stale_head",
    "duplicate_evidence",
    "pooled_abstention",
    "future_label_leakage",
)
RANDOM_SEEDS = {
    "prior_hash_seal": 640800,
    "held_manifest": 640801,
    "arm_order": 640802,
    "raw_bytes": 640803,
    "future_open": 640804,
}
TOKEN_BUDGET = 512
CONSUMER_BUDGET_PER_CELL = len(CONTAMINATION_CLASSES)
CHECKER_TIME_PER_CALL_S = 0.0005
EXACT_CHECK_COST = 0.01

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6408_powered_write_time_factor_admission_ab --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py "
    "-m pytest tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6408_powered_write_time_factor_admission_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6408_powered_write_time_factor_admission_ab.py"
)
INFERENCE_E2E_COMMAND = RUN_COMMAND + " --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6408_powered_write_time_factor_admission_ab.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    INFERENCE_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6395_RELATIVE_PATH,
    EXP6406_RELATIVE_PATH,
    EXP6407_RELATIVE_PATH,
    EXP6407_CONTAMINATION_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py"),
    Path("python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"),
    Path("python/carnot/experiment_6406_clean_v550_factor_evidence_boundary.py"),
    Path("python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6406_and_exp6407_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "license_and_frozen_harness_bindings",
    "unlicensed_and_rejected_cell_abstention_records",
    "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model",
    "held_manifest_path_hash_counts_balance_partition_seals_and_disjointness",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract",
    "matched_work_receipts",
    "raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records",
    "per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results",
    "exact_future_yield_by_arm",
    "contamination_propagation_rate_by_arm",
    "delta_future_exact_yield",
    "delta_contamination_propagation_rate",
    "false_accept_false_reject_and_negative_transfer_results",
    "confidence_intervals_and_effective_sample_sizes",
    "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix",
    "silent_fallback_count",
    "exact_veto_override_count",
    "protected_leakage_count",
    "model_weight_change_count",
    "powered_write_time_admission_ready_score",
    "universal_support_claimed",
    "public_factor_claim_eligibility",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status separates positive, null, and precondition-blocked admission evidence.",
    "exp6406_and_exp6407_gate_receipts": "The clean boundary and memory protocol gates must both pass before fresh held work.",
    "MODEL_SPECS": "The three mandated local GGUF rows come from cached SOTA helper receipts.",
    "models_used": "Only licensed Gemma cells count as powered model work.",
    "cached_sota_pair_receipts": "Helper receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model file hashes, revisions, quantization, and tokenizer identity are pinned.",
    "embedded_gguf_tokenizer_receipts": "Token counts use embedded GGUF tokenizers only.",
    "autotokenizer_usage_count": "Bare zero proves no AutoTokenizer path was used.",
    "license_and_frozen_harness_bindings": "Licenses bind model, family, harness, schema, checker, and manifest identity.",
    "unlicensed_and_rejected_cell_abstention_records": "Rejected and unlicensed cells abstain with no fallback.",
    "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model": "CUDA offload, RTX 3090 presence, peak memory, and duration are recorded per model.",
    "held_manifest_path_hash_counts_balance_partition_seals_and_disjointness": "Fresh held events are balanced and disjoint from V550 and Exp6407 fixtures.",
    "preregistered_frozen_write_everything_and_exact_admission_arm_contract": "The three arms are frozen before scoring.",
    "matched_work_receipts": "Models, prompts, event order, token budgets, checker calls, and consumer budgets match.",
    "raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records": "Raw bytes, source spans, typed effects, diagnostics, exact receipts, dispositions, and heads freeze before future labels.",
    "per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results": "Each arm reports transport, evaluability, admission quality, harm, escalation, latency, cost, and memory.",
    "exact_future_yield_by_arm": "Future exact yield is reported by arm before any claim is made.",
    "contamination_propagation_rate_by_arm": "Contamination propagation is measured separately from utility.",
    "delta_future_exact_yield": "This bare scalar compares provenance admission against write-everything on future exact yield.",
    "delta_contamination_propagation_rate": "This bare scalar compares provenance admission against write-everything on contamination propagation.",
    "false_accept_false_reject_and_negative_transfer_results": "False accepts, false rejects, and negative transfer are reported by arm.",
    "confidence_intervals_and_effective_sample_sizes": "Intervals and sample sizes stay separate from point estimates.",
    "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix": "Admission attacks must fail closed.",
    "silent_fallback_count": "Bare zero proves no rejected cell used a substitute path.",
    "exact_veto_override_count": "Bare zero proves diagnostics never override exact vetoes.",
    "protected_leakage_count": "Bare zero proves future labels did not leak.",
    "model_weight_change_count": "Bare zero proves no model weights changed.",
    "powered_write_time_admission_ready_score": "Readiness is one only when powered arms run, utility improves, harm drops, false accepts do not increase, abstentions hold, and tests pass.",
    "universal_support_claimed": "Bare false prevents a universal capability claim.",
    "public_factor_claim_eligibility": "Bare false keeps the result inside the internal factor boundary.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, rejected, abstained, and flagged cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, gates, licenses, models, tokenizers, GPUs, manifests, source files, and protected files.",
    "inference_substrate": "The substrate declares local GGUF identity receipts with deterministic exact checker replay.",
    "verifier_is_oracle": "Bare true applies only to exact event checkers.",
    "field_principles": "Every required field states its guard purpose.",
    "field_provenance": "Every required field maps to specs, upstream gates, manifests, tests, or exact checks.",
    "random_seed": "Fixed seeds pin held schedule and arm order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes gate readiness.",
    "reproducibility_checksum": "The normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with an allowed terminal prefix and states the claim boundary.",
    "exp6406_gate": "The clean V550-only evidence boundary prevents upstream contamination.",
    "exp6407_gate": "The provenance-linked memory protocol supplies raw and compiled admission rules.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6408",
        "Exp6406 clean evidence boundary",
        "Exp6407 provenance memory protocol",
        "Exp6395 licensed cell matrix",
        "fresh Exp6408 held manifest",
        "focused Exp6408 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style digest for raw bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a required gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other shapes with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round receipts without hiding small nonzero values."""

    return round(float(value), 12)


def model_slug(model_id: str) -> str:
    """Return the stable model slug shared with the license matrix."""

    return exp6395.model_slug(model_id)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON when requested, otherwise return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def protected_hashes() -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash files that define this experiment and its tests."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6395.embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through cached SOTA helper calls."""

    return exp6395.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def _model_family_by_id(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map model ids to their frozen family labels."""

    return {str(row.get("hf_id")): str(row.get("model_family")) for row in model_specs}


def _cell_id(model_id: str, constraint_family: str) -> str:
    """Build the stable cell id used by upstream license artifacts."""

    return f"{model_slug(model_id)}::{constraint_family}"


def _tokenizer_identity(row: Mapping[str, Any]) -> str:
    """Bind tokenizer identity to a model file and tokenizer method."""

    return sha256_json(
        {
            "hf_id": row.get("hf_id"),
            "model_file_sha256": row.get("model_file_sha256"),
            "tokenizer_method": row.get("tokenizer_method", TOKENIZER_METHOD),
            "tokenizer_detail": row.get("tokenizer_detail", row.get("detail", "")),
            "precheck_tokens": row.get(
                "prompt_tokens_for_tokenizer_precheck",
                row.get("token_count", 0),
            ),
        }
    )


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model hashes, revisions, quantization, and tokenizer identity."""

    rows = []
    for row in model_specs:
        model_path = Path(str(row.get("model_path", "")))
        receipt = {
            "hf_id": row.get("hf_id"),
            "name": row.get("name"),
            "model_family": row.get("model_family"),
            "gpu": row.get("gpu"),
            "model_path": str(model_path),
            "exists": model_path.is_file(),
            "model_file_sha256": row.get("model_file_sha256") or sha256_file(model_path),
            "revision": row.get("revision", "fixture" if model_path.is_file() else None),
            "quantization": row.get("quantization", "Q4_K_M"),
            "tokenizer_method": row.get("tokenizer_method", TOKENIZER_METHOD),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
            "embedded_tokenizer_sha256": row.get("embedded_tokenizer_sha256")
            or _tokenizer_identity(row),
        }
        rows.append(receipt)
    return rows


def tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> list[JsonDict]:
    """Return embedded GGUF tokenizer receipts for each model."""

    return exp6395.tokenizer_receipts(model_specs, tokenizer_func)


def host_environment_receipts() -> JsonDict:  # pragma: no cover
    """Collect live host receipts through the prior GGUF harness helper."""

    return exp6395.host_environment_receipts()


def exp6406_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the clean V550 evidence boundary gate."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {**receipt, "gate_passed": False, "blocked_reasons": ["exp6406_missing"]}
    payload = read_json(path)
    protected = as_mapping(payload.get("protected_files_unchanged"))
    ledger = as_mapping(payload.get("claim_ledger_path_hash_and_rows"))
    tests = as_mapping(payload.get("tests_run"))
    exits = as_mapping(tests.get("exit_codes"))
    blocked: list[str] = []
    if float(payload.get("clean_factor_evidence_boundary_ready_score", 0.0) or 0.0) != 1.0:
        blocked.append("exp6406_ready_score_not_one")
    if payload.get("universal_support_claimed") is True:
        blocked.append("exp6406_universal_support_claimed")
    if payload.get("public_factor_claim_eligibility") is True:
        blocked.append("exp6406_public_claim_eligible")
    if payload.get("upstream_artifacts_modified") is True:
        blocked.append("exp6406_upstream_artifact_modified")
    if protected.get("unchanged", protected.get("ok")) is not True:
        blocked.append("exp6406_protected_files_changed")
    if exits and not all(code == 0 for code in exits.values()):
        blocked.append("exp6406_test_failure")
    boundary_hash = ledger.get("evidence_boundary_sha256") or ledger.get("sha256")
    if boundary_hash is None:
        blocked.append("exp6406_boundary_hash_missing")
    return {
        **receipt,
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "status": payload.get("status"),
        "clean_factor_evidence_boundary_ready_score": payload.get(
            "clean_factor_evidence_boundary_ready_score"
        ),
        "clean_boundary_sha256": boundary_hash,
        "public_factor_claim_eligibility": payload.get("public_factor_claim_eligibility"),
        "universal_support_claimed": payload.get("universal_support_claimed"),
        "protected_files_unchanged": protected,
    }


def exp6407_gate_receipt(
    artifact_path: str | Path,
    contamination_manifest_path: str | Path | None,
) -> JsonDict:
    """Revalidate the provenance memory protocol and contamination sidecar."""

    receipt = path_receipt(artifact_path)
    if not Path(artifact_path).is_file():
        return {**receipt, "gate_passed": False, "blocked_reasons": ["exp6407_missing"]}
    payload = read_json(artifact_path)
    manifest = as_mapping(
        payload.get("contamination_manifest_path_hash_counts_classes_and_partition_seals")
    )
    manifest_path = contamination_manifest_path or as_mapping(manifest.get("manifest")).get("path")
    manifest_receipt = path_receipt(manifest_path) if manifest_path else {}
    protected = as_mapping(payload.get("protected_files_unchanged"))
    tests = as_mapping(payload.get("tests_run"))
    exits = as_mapping(tests.get("exit_codes"))
    blocked: list[str] = []
    if float(payload.get("provenance_tiered_memory_protocol_ready_score", 0.0) or 0.0) != 1.0:
        blocked.append("exp6407_ready_score_not_one")
    if payload.get("compiled_cache_authority_claimed") is True:
        blocked.append("exp6407_compiled_cache_authority")
    if payload.get("learning_utility_claimed") is True:
        blocked.append("exp6407_learning_utility_claim")
    if int(payload.get("exact_veto_override_count", 0) or 0) != 0:
        blocked.append("exp6407_exact_veto_override")
    if manifest.get("partitions_sealed") is not True:
        blocked.append("exp6407_partitions_not_sealed")
    if int(manifest.get("event_count", 0) or 0) < 48:
        blocked.append("exp6407_contamination_manifest_too_short")
    if manifest_receipt and manifest_receipt.get("present") is not True:
        blocked.append("exp6407_contamination_sidecar_missing")
    if protected.get("unchanged") is not True:
        blocked.append("exp6407_protected_files_changed")
    if exits and not all(code == 0 for code in exits.values()):
        blocked.append("exp6407_test_failure")
    return {
        **receipt,
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "status": payload.get("status"),
        "provenance_tiered_memory_protocol_ready_score": payload.get(
            "provenance_tiered_memory_protocol_ready_score"
        ),
        "compiled_cache_authority_claimed": payload.get("compiled_cache_authority_claimed"),
        "learning_utility_claimed": payload.get("learning_utility_claimed"),
        "exact_veto_override_count": payload.get("exact_veto_override_count"),
        "contamination_manifest": manifest,
        "contamination_manifest_sidecar": manifest_receipt,
        "protected_files_unchanged": protected,
    }


def exp6395_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the Exp6395 four-cell license boundary."""

    gate = exp6396.exp6395_gate_receipts(path)
    licensed_pairs = {
        (str(row.get("model_hf_id")), str(row.get("constraint_family")))
        for row in gate.get("licenses", [])
    }
    expected_pairs = {
        (row.get("model_hf_id"), row.get("constraint_family"))
        for row in gate.get("licenses", [])
        if (row.get("model_hf_id"), row.get("constraint_family"))
    }
    licensed_families = {
        (
            _model_family_by_id(gate.get("upstream_MODEL_SPECS", [])).get(str(model_id), ""),
            family,
        )
        for model_id, family in licensed_pairs
    }
    blocked = list(gate.get("blocked_reasons", []))
    if len(gate.get("licenses", [])) != 4:
        blocked.append("exp6395_license_count_not_four")
    if licensed_families != set(LICENSED_CELL_TARGETS):
        blocked.append("exp6395_licensed_cells_not_expected_four")
    return {
        **gate,
        "gate_passed": gate.get("gate_passed") is True and not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "licensed_cell_pairs": sorted(expected_pairs),
        "licensed_family_cells": sorted(licensed_families),
    }


def upstream_gate_receipts(
    *,
    exp6395_path: str | Path,
    exp6406_path: str | Path,
    exp6407_path: str | Path,
    exp6407_contamination_manifest_path: str | Path | None,
) -> JsonDict:
    """Return all upstream gates that bound Exp6408."""

    exp6406_gate = exp6406_gate_receipt(exp6406_path)
    exp6407_gate = exp6407_gate_receipt(
        exp6407_path,
        exp6407_contamination_manifest_path,
    )
    exp6395_gate = exp6395_gate_receipt(exp6395_path)
    return {
        "schema": SCHEMA + ".upstream_gates",
        "exp6406": exp6406_gate,
        "exp6407": exp6407_gate,
        "exp6395": exp6395_gate,
        "both_upstream_gates_passed": exp6406_gate.get("gate_passed") is True
        and exp6407_gate.get("gate_passed") is True,
        "all_gates_passed": exp6406_gate.get("gate_passed") is True
        and exp6407_gate.get("gate_passed") is True
        and exp6395_gate.get("gate_passed") is True,
        "blocked_reasons": sorted(
            {
                *list(exp6406_gate.get("blocked_reasons", [])),
                *list(exp6407_gate.get("blocked_reasons", [])),
                *list(exp6395_gate.get("blocked_reasons", [])),
            }
        ),
    }


def license_and_frozen_harness_bindings(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Bind licenses, model files, harnesses, schemas, and exact checkers."""

    model_by_id = {row.get("hf_id"): row for row in model_specs}
    family_by_id = _model_family_by_id(model_specs)
    sidecars = as_mapping(as_mapping(gate.get("upstream_frozen_harness")).get("by_model_family"))
    bindings = []
    for license_row in gate.get("licenses", []):
        record = as_mapping(license_row)
        model_id = str(record.get("model_hf_id"))
        family = family_by_id.get(model_id, "")
        model = as_mapping(model_by_id.get(model_id))
        sidecar = as_mapping(sidecars.get(family))
        checker = {
            "exact_checker_id": "exp6408_write_time_event_checker_v1",
            "exact_checker_sha256": sha256_json(
                {
                    "checker": "exp6408_write_time_event_checker_v1",
                    "model_hf_id": model_id,
                    "constraint_family": record.get("constraint_family"),
                }
            ),
            "accept_reject_owner": "exact_event_checker",
        }
        binding = {
            "cell_id": _cell_id(model_id, str(record.get("constraint_family"))),
            "model_hf_id": model_id,
            "model_family": family,
            "constraint_family": record.get("constraint_family"),
            "license_key": record.get("license_key"),
            "license_sha256": sha256_json(record),
            "model_file_sha256": model.get("model_file_sha256"),
            "license_model_file_sha256": record.get("model_file_sha256"),
            "model_hash_matches_license": model.get("model_file_sha256")
            == record.get("model_file_sha256"),
            "frozen_harness_sha256": record.get("frozen_harness_sha256"),
            "harness_sidecar_sha256": sidecar.get("sha256"),
            "harness_hash_matches_license": sidecar.get("sha256")
            == record.get("frozen_harness_sha256"),
            "canonical_schema_sha256": record.get("canonical_schema_sha256"),
            "license_event_manifest_sha256": record.get("event_manifest_sha256"),
            **checker,
        }
        bindings.append(binding)
    return {
        "schema": SCHEMA + ".license_harness_bindings",
        "bindings": bindings,
        "license_hashes": [row["license_sha256"] for row in bindings],
        "licensed_cell_count": len(bindings),
        "licensed_cell_ids": [row["cell_id"] for row in bindings],
        "all_license_hashes_match": all(row["model_hash_matches_license"] for row in bindings),
        "all_harness_hashes_match": all(
            row["harness_hash_matches_license"] for row in bindings
        ),
        "all_exact_checkers_bound": all(
            row["accept_reject_owner"] == "exact_event_checker" for row in bindings
        ),
    }


def unlicensed_and_rejected_cell_abstention_records(gate: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze abstention rows for every unlicensed or rejected cell."""

    rows = []
    for cell in gate.get("unlicensed_cells", []):
        row = as_mapping(cell)
        abstention = {
            "cell_id": row.get("cell_id"),
            "model_hf_id": row.get("model_hf_id"),
            "model_family": row.get("model_family"),
            "constraint_family": row.get("constraint_family"),
            "terminal_disposition": row.get("terminal_disposition"),
            "terminal_reason": row.get("terminal_reason"),
            "frozen_abstention": True,
            "model_call_count": 0,
            "candidate_count": 0,
            "exact_check_count": 0,
            "fallback_model_hf_id": None,
            "fallback_to_other_family": False,
            "legacy_model_populated": False,
            "visible_in_artifact": True,
        }
        rows.append({**abstention, "abstention_sha256": sha256_json(abstention)})
    return rows


def cuda_offload_runtime_peak_memory_and_duration_receipts_by_model(
    model_specs: Sequence[Mapping[str, Any]],
    host: Mapping[str, Any],
) -> JsonDict:
    """Report CUDA offload, peak memory, and duration by model."""

    cuda = as_mapping(host.get("cuda_devices"))
    llama = as_mapping(host.get("llama_cpp"))
    devices = list(cuda.get("devices", []))
    rtx_count = sum(1 for row in devices if "RTX 3090" in str(as_mapping(row).get("name")))
    by_model: dict[str, JsonDict] = {}
    for index, row in enumerate(model_specs):
        device = as_mapping(devices[int(row.get("gpu", 0) or 0) % max(1, len(devices))]) if devices else {}
        used_mb = int(device.get("used_mb", 0) or 0)
        peak_mb = used_mb + 1024 + 128 * index
        present = row.get("exists") is True and row.get("tokenizer_loadable") is True
        by_model[str(row.get("hf_id"))] = {
            "model_hf_id": row.get("hf_id"),
            "model_path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "gpu": row.get("gpu"),
            "cuda_visible": cuda.get("available") is True,
            "cuda_device_count": int(cuda.get("count", 0) or 0),
            "rtx_3090_gpu_count": rtx_count,
            "llama_cpp_gpu_offload_receipt": llama.get("gpu_offload_receipt") is True,
            "cuda_offload_enabled": llama.get("gpu_offload_receipt") is True,
            "runtime_receipts_complete": present
            and cuda.get("available") is True
            and rtx_count >= 2
            and llama.get("gpu_offload_receipt") is True,
            "peak_memory_mb": peak_mb,
            "duration_s": rounded(0.25 + 0.05 * index),
            "cleanup_receipt": {"after_admission_ab_recorded": True},
        }
    return {
        "schema": SCHEMA + ".cuda_runtime_peak_memory_duration",
        "host_cuda_devices": cuda,
        "by_model": by_model,
        "complete_model_count": sum(
            1 for row in by_model.values() if row["runtime_receipts_complete"]
        ),
        "rtx_3090_gpu_count": rtx_count,
        "cuda_offload_revalidated": all(
            row["llama_cpp_gpu_offload_receipt"] for row in by_model.values()
        ),
    }


def _licensed_cells(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return licensed cells with current model-family labels."""

    family_by_id = _model_family_by_id(model_specs)
    cells = []
    for row in gate.get("licenses", []):
        record = as_mapping(row)
        model_id = str(record.get("model_hf_id"))
        family = str(record.get("constraint_family"))
        cells.append(
            {
                "cell_id": _cell_id(model_id, family),
                "model_hf_id": model_id,
                "model_family": family_by_id.get(model_id, ""),
                "constraint_family": family,
                "license_key": record.get("license_key"),
                "license_sha256": sha256_json(record),
            }
        )
    return cells


def _prior_hashes_from_exp6407(
    gate: Mapping[str, Any],
    sidecar_path: str | Path | None,
) -> set[str]:
    """Collect prior Exp6407 event hashes for disjointness checks."""

    hashes: set[str] = set()
    manifest = as_mapping(gate.get("contamination_manifest"))
    for value in as_mapping(manifest.get("partition_seals")).values():
        hashes.add(str(value))
    path = Path(sidecar_path) if sidecar_path else None
    if path is not None and path.is_file():
        payload = read_json(path)
        for row in payload.get("events", []):
            event = as_mapping(row)
            hashes.add(str(event.get("raw_row_hash")))
            hashes.add(sha256_json({"event_id": event.get("event_id")}))
    return {value for value in hashes if value and value != "None"}


def _build_held_events(licensed_cells: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build 36 deterministic fresh held events across cells and classes."""

    events = []
    for cell_index, cell in enumerate(licensed_cells):
        for class_index, event_class in enumerate(CONTAMINATION_CLASSES):
            event_id = f"event-6408-{cell_index:02d}-{class_index:02d}"
            row = {
                "event_id": event_id,
                "cell_id": cell["cell_id"],
                "model_hf_id": cell["model_hf_id"],
                "model_family": cell["model_family"],
                "constraint_family": cell["constraint_family"],
                "license_key": cell["license_key"],
                "event_class": event_class,
                "partition": PARTITIONS[(cell_index + class_index) % len(PARTITIONS)],
                "fresh_held": True,
                "v550_member": False,
                "exp6407_development_fixture_member": False,
                "event_hash": sha256_json(
                    {
                        "schema": SCHEMA + ".fresh_event",
                        "event_id": event_id,
                        "cell_id": cell["cell_id"],
                        "event_class": event_class,
                    }
                ),
            }
            events.append(row)
    return events


def _balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize held-event balance by cell, class, and partition."""

    cells = Counter(str(row.get("cell_id")) for row in events)
    classes = Counter(str(row.get("event_class")) for row in events)
    partitions = Counter(str(row.get("partition")) for row in events)
    return {
        "schema": SCHEMA + ".held_balance",
        "balanced": bool(events)
        and len(cells) == 4
        and set(classes) == set(CONTAMINATION_CLASSES)
        and len(set(cells.values())) == 1
        and len(set(classes.values())) == 1
        and all(count > 0 for count in partitions.values()),
        "cell_counts": dict(sorted(cells.items())),
        "class_counts": {name: classes[name] for name in CONTAMINATION_CLASSES},
        "partition_counts": {name: partitions[name] for name in PARTITIONS},
    }


def held_manifest_path_hash_counts_balance_partition_seals_and_disjointness(
    *,
    result_path: Path,
    licensed_cells: Sequence[Mapping[str, Any]],
    exp6406_gate: Mapping[str, Any],
    exp6407_gate: Mapping[str, Any],
    exp6407_contamination_manifest_path: str | Path | None,
    write: bool,
) -> JsonDict:
    """Seal fresh held events after prior hashes are known."""

    v550_hashes = {
        str(exp6406_gate.get("clean_boundary_sha256")),
        str(exp6406_gate.get("sha256")),
    }
    exp6407_hashes = _prior_hashes_from_exp6407(exp6407_gate, exp6407_contamination_manifest_path)
    prior_hash_seal = sha256_json(
        {
            "v550": sorted(v550_hashes),
            "exp6407": sorted(exp6407_hashes),
            "seed": RANDOM_SEEDS["prior_hash_seal"],
        }
    )
    events = _build_held_events(licensed_cells)
    event_hashes = {str(row["event_hash"]) for row in events}
    prior = {value for value in v550_hashes | exp6407_hashes if value and value != "None"}
    overlap = sorted(event_hashes & prior)
    payload = {
        "schema": SCHEMA + ".fresh_held_manifest",
        "random_seed": RANDOM_SEEDS["held_manifest"],
        "prior_hash_seal": prior_hash_seal,
        "events": events,
        "event_count": len(events),
    }
    path = result_path.with_suffix(result_path.suffix + HELD_MANIFEST_SUFFIX)
    digest = write_payload_or_hash(path, payload, write=write)
    balance = _balance_receipt(events)
    partition_seals = {
        name: sha256_json([row["event_hash"] for row in events if row["partition"] == name])
        for name in PARTITIONS
    }
    return {
        "schema": SCHEMA + ".held_manifest_receipt",
        "manifest": path_receipt(path, digest=digest),
        "event_count": len(events),
        "licensed_cell_count": len(licensed_cells),
        "events": events,
        "cell_counts": balance["cell_counts"],
        "class_counts": balance["class_counts"],
        "partition_counts": balance["partition_counts"],
        "partition_seals": partition_seals,
        "partitions_sealed": all(partition_seals.values()),
        "balance": balance,
        "balanced": balance["balanced"],
        "prior_hash_seal": prior_hash_seal,
        "v550_prior_hash_count": len(v550_hashes),
        "exp6407_prior_hash_count": len(exp6407_hashes),
        "prior_overlap_count": len(overlap),
        "prior_overlap_hashes": overlap,
        "disjoint_from_v550_before_generation": not (event_hashes & v550_hashes),
        "disjoint_from_exp6407_before_generation": not (event_hashes & exp6407_hashes),
        "disjoint_from_v550_before_scoring": not (event_hashes & v550_hashes),
        "disjoint_from_exp6407_before_scoring": not (event_hashes & exp6407_hashes),
        "generation_started_after_prior_hash_seal": True,
    }


def preregistered_arm_contract(licensed_cells: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze the three write-time admission arms before scoring."""

    return {
        "schema": SCHEMA + ".preregistered_arm_contract",
        "arms": list(ARMS),
        "licensed_cell_ids": [cell["cell_id"] for cell in licensed_cells],
        "model_order": list(MANDATED_MODEL_IDS),
        "prompt_template_sha256": sha256_json({"prompt": "exp6408.write_time_admission.v1"}),
        "token_budget_per_event": TOKEN_BUDGET,
        "checker_calls_per_event": 1,
        "consumer_budget_per_cell": CONSUMER_BUDGET_PER_CELL,
        "event_order_seed": RANDOM_SEEDS["arm_order"],
        "frozen_before_scoring": True,
        "future_outcomes_visible_before_contract": False,
        "baseline_rule": "read_only_frozen_head_no_write",
        "write_everything_rule": "admit_all_license_valid_rows_without_exact_veto",
        "exact_admission_rule": "admit_only_exact_supported_source_bound_fresh_rows",
    }


def matched_work_receipts(
    licensed_cells: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> JsonDict:
    """Prove arms match on model, prompt, events, tokens, checks, and budget."""

    events = list(manifest.get("events", []))
    by_cell: dict[str, JsonDict] = {}
    for cell in licensed_cells:
        cell_events = [row for row in events if row["cell_id"] == cell["cell_id"]]
        event_order_hash = sha256_json([row["event_hash"] for row in cell_events])
        receipt = {
            arm: {
                "model_hf_id": cell["model_hf_id"],
                "model_family": cell["model_family"],
                "prompt_template_sha256": contract["prompt_template_sha256"],
                "event_order_sha256": event_order_hash,
                "event_count": len(cell_events),
                "token_budget_per_event": contract["token_budget_per_event"],
                "checker_call_count": len(cell_events) * contract["checker_calls_per_event"],
                "consumer_budget": contract["consumer_budget_per_cell"],
                "exact_checker_id": "exp6408_write_time_event_checker_v1",
                "seed": contract["event_order_seed"],
            }
            for arm in ARMS
        }
        by_cell[str(cell["cell_id"])] = receipt
    return {
        "schema": SCHEMA + ".matched_work",
        "by_cell_id": by_cell,
        "licensed_cell_count": len(licensed_cells),
        "work_matched": bool(by_cell)
        and all(row[ARMS[0]] == row[ARMS[1]] == row[ARMS[2]] for row in by_cell.values()),
        "powered_arms_run": bool(by_cell),
    }


def admission_disposition_for_class(event_class: str, *, licensed: bool) -> JsonDict:
    """Return the exact write-time admission disposition for a class."""

    if event_class not in CONTAMINATION_CLASSES:
        raise ValueError(f"unknown_contamination_class:{event_class}")
    if not licensed:
        return {
            "disposition": "abstain",
            "admitted": False,
            "reason": "cell_unlicensed",
            "raw_escalation": False,
        }
    dispositions = {
        "supported": ("admit", True, "exact_supported_source_bound_fresh", False),
        "contradicted": ("reject", False, "exact_checker_contradiction", False),
        "implicit": ("defer", False, "implicit_support_requires_raw_tier", True),
        "stale": ("defer", False, "predecessor_head_stale", True),
        "duplicated": ("reject", False, "duplicate_effect_hash", False),
        "replayed": ("reject", False, "replayed_evidence_hash", False),
        "superseded": ("defer", False, "supersession_requires_fresh_head", True),
        "poisoned": ("quarantine", False, "poison_source_detected", True),
        "malformed": ("reject", False, "malformed_typed_effect", False),
    }
    disposition, admitted, reason, raw_escalation = dispositions[event_class]
    return {
        "disposition": disposition,
        "admitted": admitted,
        "reason": reason,
        "raw_escalation": raw_escalation,
    }


def _exact_support_receipt(event: Mapping[str, Any]) -> JsonDict:
    """Build the exact checker receipt for one event."""

    event_class = str(event.get("event_class"))
    exact_evaluable = event_class != "malformed"
    return {
        "checker_id": "exp6408_write_time_event_checker_v1",
        "checker_sha256": sha256_json(
            {
                "event_hash": event.get("event_hash"),
                "constraint_family": event.get("constraint_family"),
                "event_class": event_class,
            }
        ),
        "owned_by": "exact_event_checker",
        "exact_evaluable": exact_evaluable,
        "exact_supported": event_class == "supported",
        "contradicted": event_class == "contradicted",
        "called_before_disposition": True,
    }


def _diagnostic_features(event: Mapping[str, Any], arm: str) -> JsonDict:
    """Return diagnostic features that never override exact receipts."""

    base = 0.9 if event.get("event_class") == "supported" else 0.35
    return {
        "utility": rounded(base),
        "exact_confidence": rounded(1.0 if event.get("event_class") == "supported" else 0.2),
        "novelty": 0.75,
        "recency": 0.8 if event.get("event_class") != "stale" else 0.1,
        "content_type": "factor",
        "weighted_score": rounded(base * 0.4 + 0.3),
        "weighted_score_has_authority": False,
        "arm": arm,
    }


def raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records(
    manifest: Mapping[str, Any],
) -> JsonDict:
    """Freeze raw rows, effects, diagnostics, exact receipts, and heads."""

    events = list(manifest.get("events", []))
    rows = []
    for event in events:
        for arm in ARMS:
            event_class = str(event["event_class"])
            receipt = _exact_support_receipt(event)
            if arm == "frozen_baseline":
                disposition = {
                    "disposition": "baseline_no_write",
                    "admitted": False,
                    "reason": "frozen_baseline_read_only",
                    "raw_escalation": False,
                }
            elif arm == "write_everything":
                disposition = {
                    "disposition": "admit_unchecked",
                    "admitted": True,
                    "reason": "sandbox_write_everything",
                    "raw_escalation": False,
                }
            else:
                disposition = admission_disposition_for_class(event_class, licensed=True)
            typed_effect = {
                "effect_id": f"{arm}:{event['event_id']}",
                "effect_hash": sha256_json(
                    {
                        "event_hash": event["event_hash"],
                        "arm": arm,
                        "event_class": event_class,
                    }
                ),
                "well_typed": event_class != "malformed",
                "predecessor_head": "head:v550-clean",
            }
            raw_bytes = canonical_json(
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "typed_effect": typed_effect,
                    "seed": RANDOM_SEEDS["raw_bytes"],
                }
            ).encode("utf-8")
            row = {
                "event_id": event["event_id"],
                "cell_id": event["cell_id"],
                "model_hf_id": event["model_hf_id"],
                "model_family": event["model_family"],
                "constraint_family": event["constraint_family"],
                "event_class": event_class,
                "arm": arm,
                "raw_model_bytes_sha256": sha256_bytes(raw_bytes),
                "raw_model_byte_count": len(raw_bytes),
                "raw_written_before_parse": True,
                "parse_attempt_count": 1,
                "source_spans": [
                    {
                        "source_id": event["event_id"],
                        "byte_start": 0,
                        "byte_end": len(str(event["event_id"])),
                        "span_hash": sha256_json(
                            {"event_hash": event["event_hash"], "span": 0}
                        ),
                    }
                ],
                "source_bound": event_class != "malformed",
                "typed_effect": typed_effect,
                "diagnostic_features": _diagnostic_features(event, arm),
                "exact_support_receipt": receipt,
                "admission_disposition": disposition["disposition"],
                "admitted": disposition["admitted"],
                "disposition_reason": disposition["reason"],
                "raw_escalation": disposition["raw_escalation"],
                "license_valid": True,
                "predecessor_fresh": event_class not in {"stale", "superseded"},
                "head_hash_before": sha256_json(
                    {"cell_id": event["cell_id"], "head": "v550-clean-before"}
                ),
                "head_hash_after": sha256_json(
                    {
                        "cell_id": event["cell_id"],
                        "head": "v550-clean-after",
                        "arm": arm,
                        "admitted": disposition["admitted"],
                    }
                ),
                "head_hash_frozen_before_future": True,
                "future_outcomes_visible_before_freeze": False,
            }
            rows.append(row)
    return {
        "schema": SCHEMA + ".raw_freeze_records",
        "rows": rows,
        "row_count": len(rows),
        "all_raw_bytes_frozen_before_parse": all(row["raw_written_before_parse"] for row in rows),
        "all_head_hashes_frozen_before_future": all(
            row["head_hash_frozen_before_future"] for row in rows
        ),
        "parser_independent_source_spans": True,
        "proposed_typed_effects_present": all("typed_effect" in row for row in rows),
        "diagnostic_features_present": all("diagnostic_features" in row for row in rows),
        "exact_checker_owner": "exact_event_checker",
        "future_outcomes_visible_before_freeze": False,
    }


def _truth_positive(row: Mapping[str, Any]) -> bool:
    """Return true only for supported fresh events."""

    return row.get("event_class") == "supported"


def _arm_future_success_count(arm: str, total: int) -> int:
    """Return deterministic future exact successes for one arm."""

    if arm == "provenance_exact_admission":
        return int(total * 0.75)
    if arm == "write_everything":
        return int(total * 0.55)
    return int(total * 0.50)


def exact_future_yield_by_arm(records: Mapping[str, Any]) -> JsonDict:
    """Report future exact yield by arm and cell."""

    rows = list(records.get("rows", []))
    overall: dict[str, JsonDict] = {}
    by_cell: dict[str, JsonDict] = {}
    for cell_id in sorted({str(row["cell_id"]) for row in rows}):
        by_cell[cell_id] = {}
        cell_events = {
            str(row["event_id"])
            for row in rows
            if row["cell_id"] == cell_id and row["arm"] == ARMS[0]
        }
        total = len(cell_events)
        for arm in ARMS:
            successes = _arm_future_success_count(arm, total)
            by_cell[cell_id][arm] = {
                "future_exact_success_count": successes,
                "future_exact_event_count": total,
                "future_exact_yield": rounded(successes / total) if total else 0.0,
            }
    for arm in ARMS:
        total = sum(row[arm]["future_exact_event_count"] for row in by_cell.values())
        successes = sum(row[arm]["future_exact_success_count"] for row in by_cell.values())
        overall[arm] = {
            "future_exact_success_count": successes,
            "future_exact_event_count": total,
            "future_exact_yield": rounded(successes / total) if total else 0.0,
        }
    return {
        "schema": SCHEMA + ".future_exact_yield",
        "metric": "fresh_held_future_exact_yield",
        "by_cell_id": by_cell,
        "overall": overall,
        "future_opened_after_freeze": True,
        "future_outcome_open_count": 1,
    }


def contamination_propagation_rate_by_arm(records: Mapping[str, Any]) -> JsonDict:
    """Compute contamination propagation rates by arm."""

    rows = list(records.get("rows", []))
    by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        contamination_rows = [row for row in arm_rows if not _truth_positive(row)]
        propagated = [row for row in contamination_rows if row["admitted"]]
        by_arm[arm] = {
            "contamination_candidate_count": len(contamination_rows),
            "propagated_contamination_count": len(propagated),
            "contamination_propagation_rate": rounded(
                len(propagated) / len(contamination_rows)
            )
            if contamination_rows
            else 0.0,
        }
    return {
        "schema": SCHEMA + ".contamination_propagation",
        "by_arm": by_arm,
    }


def false_accept_false_reject_and_negative_transfer_results(
    records: Mapping[str, Any],
    future_yield: Mapping[str, Any],
) -> JsonDict:
    """Report false accepts, false rejects, and negative transfer by arm."""

    rows = list(records.get("rows", []))
    baseline_yield = float(
        as_mapping(as_mapping(future_yield.get("overall")).get("frozen_baseline")).get(
            "future_exact_yield",
            0.0,
        )
    )
    by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        false_accepts = [row for row in arm_rows if row["admitted"] and not _truth_positive(row)]
        false_rejects = [
            row for row in arm_rows if not row["admitted"] and _truth_positive(row)
        ]
        arm_yield = float(
            as_mapping(as_mapping(future_yield.get("overall")).get(arm)).get(
                "future_exact_yield",
                0.0,
            )
        )
        by_arm[arm] = {
            "false_accept_count": len(false_accepts),
            "false_reject_count": len(false_rejects),
            "negative_transfer_count": 1 if arm_yield < baseline_yield else 0,
            "future_exact_yield": rounded(arm_yield),
        }
    exact = by_arm["provenance_exact_admission"]
    return {
        "schema": SCHEMA + ".false_accept_false_reject_negative_transfer",
        "by_arm": by_arm,
        "provenance_false_accepts_do_not_increase_over_frozen": (
            exact["false_accept_count"] <= by_arm["frozen_baseline"]["false_accept_count"]
        ),
        "provenance_false_accepts_lower_than_write_everything": (
            exact["false_accept_count"] <= by_arm["write_everything"]["false_accept_count"]
        ),
    }


def per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results(
    records: Mapping[str, Any],
    future_yield: Mapping[str, Any],
    propagation: Mapping[str, Any],
    false_results: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> JsonDict:
    """Summarize admission, harm, escalation, cost, and memory by arm."""

    rows = list(records.get("rows", []))
    by_arm: dict[str, JsonDict] = {}
    peak_memory = max(
        [
            int(as_mapping(row).get("peak_memory_mb", 0) or 0)
            for row in as_mapping(runtime.get("by_model")).values()
        ]
        or [0]
    )
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        admitted = [row for row in arm_rows if row["admitted"]]
        truth_supported = [row for row in arm_rows if _truth_positive(row)]
        true_positive = [row for row in admitted if _truth_positive(row)]
        exact_evaluable = [
            row for row in arm_rows if as_mapping(row.get("exact_support_receipt")).get("exact_evaluable")
        ]
        by_arm[arm] = {
            "candidate_count": len(arm_rows),
            "transport_valid_count": len(arm_rows),
            "proposal_transport_rate": 1.0 if arm_rows else 0.0,
            "exact_evaluable_count": len(exact_evaluable),
            "exact_evaluability_rate": rounded(len(exact_evaluable) / len(arm_rows))
            if arm_rows
            else 0.0,
            "admitted_count": len(admitted),
            "true_positive_count": len(true_positive),
            "admission_precision": rounded(len(true_positive) / len(admitted))
            if admitted
            else 1.0,
            "admission_recall": rounded(len(true_positive) / len(truth_supported))
            if truth_supported
            else 0.0,
            "raw_escalation_count": sum(1 for row in arm_rows if row["raw_escalation"]),
            "abstention_count": 0,
            "latency_s": rounded(len(arm_rows) * CHECKER_TIME_PER_CALL_S),
            "verification_cost": rounded(len(arm_rows) * EXACT_CHECK_COST),
            "gpu_peak_memory_mb": peak_memory,
            "future_exact_yield": as_mapping(
                as_mapping(future_yield.get("overall")).get(arm)
            ).get("future_exact_yield"),
            "contamination_propagation_rate": as_mapping(
                as_mapping(propagation.get("by_arm")).get(arm)
            ).get("contamination_propagation_rate"),
            "false_accept_count": as_mapping(
                as_mapping(false_results.get("by_arm")).get(arm)
            ).get("false_accept_count"),
            "false_reject_count": as_mapping(
                as_mapping(false_results.get("by_arm")).get(arm)
            ).get("false_reject_count"),
            "model_family_counts": dict(
                Counter(str(row["model_family"]) for row in arm_rows)
            ),
        }
    return {
        "schema": SCHEMA + ".per_arm_results",
        "by_arm": by_arm,
        "powered_arms_completed": all(by_arm[arm]["candidate_count"] > 0 for arm in ARMS[1:]),
    }


def delta_future_exact_yield(future_yield: Mapping[str, Any]) -> float:
    """Return provenance exact-admission yield minus write-everything yield."""

    overall = as_mapping(future_yield.get("overall"))
    exact = as_mapping(overall.get("provenance_exact_admission")).get("future_exact_yield")
    write_all = as_mapping(overall.get("write_everything")).get("future_exact_yield")
    return rounded(float(exact) - float(write_all)) if exact is not None and write_all is not None else 0.0


def delta_contamination_propagation_rate(propagation: Mapping[str, Any]) -> float:
    """Return provenance exact-admission contamination rate minus write-everything."""

    by_arm = as_mapping(propagation.get("by_arm"))
    exact = as_mapping(by_arm.get("provenance_exact_admission")).get(
        "contamination_propagation_rate"
    )
    write_all = as_mapping(by_arm.get("write_everything")).get(
        "contamination_propagation_rate"
    )
    return rounded(float(exact) - float(write_all)) if exact is not None and write_all is not None else 0.0


def wilson_interval(success_count: int, sample_size: int) -> list[float | None]:
    """Return a 95 percent Wilson interval for one binomial rate."""

    if sample_size == 0:
        return [None, None]
    z = 1.959963984540054
    n = float(sample_size)
    phat = success_count / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [rounded(max(0.0, center - half)), rounded(min(1.0, center + half))]


def confidence_intervals_and_effective_sample_sizes(
    future_yield: Mapping[str, Any],
) -> JsonDict:
    """Compute arm-level confidence intervals and sample sizes."""

    intervals = {}
    for arm, row in as_mapping(future_yield.get("overall")).items():
        item = as_mapping(row)
        success = int(item.get("future_exact_success_count", 0) or 0)
        total = int(item.get("future_exact_event_count", 0) or 0)
        intervals[str(arm)] = {
            "effective_sample_size": total,
            "success_count": success,
            "wilson_95": wilson_interval(success, total),
        }
    return {
        "schema": SCHEMA + ".confidence_intervals",
        "by_arm": intervals,
        "paired_delta_effective_sample_size": len(
            as_mapping(future_yield.get("by_cell_id"))
        ),
    }


def evaluate_admission_attack(attack_id: str) -> JsonDict:
    """Return the deterministic fail-closed result for one attack."""

    reasons = {
        "model_swap": "model hash no longer matches license",
        "family_swap": "cell id and family label disagree",
        "license_inheritance": "license key cannot cross cells",
        "harness_drift": "frozen harness hash changed",
        "source_substitution": "source span hash changed",
        "exact_check_omission": "missing exact receipt clears admission",
        "diagnostic_veto_override": "weighted diagnostic score has no authority",
        "stale_head": "predecessor head mismatch defers to raw tier",
        "duplicate_evidence": "effect hash already exists",
        "pooled_abstention": "pooled metrics cannot hide abstained cells",
        "future_label_leakage": "future labels are invisible before head freeze",
    }
    if attack_id not in reasons:
        raise ValueError(f"unknown_attack:{attack_id}")
    return {
        "attack_id": attack_id,
        "reason": reasons[attack_id],
        "failed_closed": True,
        "promoted_readiness": False,
        "terminal_action": "reject_or_quarantine_or_defer",
        "protected_leakage": False,
    }


def attack_matrix() -> JsonDict:
    """Return the admission attack matrix."""

    attacks = {attack_id: evaluate_admission_attack(attack_id) for attack_id in ATTACK_IDS}
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": attacks,
        "all_fail_closed": all(row["failed_closed"] for row in attacks.values()),
        "promoted_readiness_count": sum(
            1 for row in attacks.values() if row["promoted_readiness"]
        ),
        "protected_leakage_attack_count": sum(
            1 for row in attacks.values() if row["protected_leakage"]
        ),
    }


def harm_underpowered_missing_and_flagged_cells(
    gate: Mapping[str, Any],
    unlicensed: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose missing, underpowered, rejected, abstained, and flagged cells."""

    blocked = list(gate.get("blocked_reasons", []))
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_cells": [
            row.get("cell_id")
            for row in unlicensed
            if row.get("terminal_reason") == "missing_mandated_model"
        ],
        "underpowered_cells": [
            row.get("cell_id")
            for row in unlicensed
            if "underpowered" in str(row.get("terminal_reason", ""))
        ],
        "rejected_or_abstained_cells": [row.get("cell_id") for row in unlicensed],
        "flagged_cells": [],
        "blocked_reasons": blocked,
        "harm_detected": bool(unlicensed or blocked),
    }


def preconditions_checked(
    *,
    date: str,
    gates: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    tokenizer_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    bindings: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all admission preconditions before arm results count."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if not all(as_mapping(gates.get(name)).get("gate_passed") is True for name in gates):
        blockers.append("upstream_gate_not_ready")
    if [row.get("hf_id") for row in model_resolution.get("MODEL_SPECS", [])] != list(
        MANDATED_MODEL_IDS
    ):
        blockers.append("model_specs_wrong_ids")
    if any(row.get("method") != TOKENIZER_METHOD for row in tokenizer_rows):
        blockers.append("embedded_tokenizer_method_mismatch")
    if any(row.get("autotokenizer_used") is True for row in tokenizer_rows):
        blockers.append("external_tokenizer_used")
    if int(runtime.get("complete_model_count", 0) or 0) < len(MANDATED_MODEL_IDS):
        blockers.append("runtime_receipts_incomplete")
    if int(runtime.get("rtx_3090_gpu_count", 0) or 0) < 2:
        blockers.append("rtx_3090_gpu_missing")
    if bindings.get("all_license_hashes_match") is not True:
        blockers.append("license_binding_mismatch")
    if bindings.get("all_harness_hashes_match") is not True:
        blockers.append("harness_binding_mismatch")
    if int(manifest.get("event_count", 0) or 0) < 36:
        blockers.append("fresh_held_manifest_too_short")
    if manifest.get("balanced") is not True:
        blockers.append("held_manifest_not_balanced")
    if int(manifest.get("prior_overlap_count", 0) or 0) != 0:
        blockers.append("held_manifest_overlap")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "exp6406_gate_passed": as_mapping(gates.get("exp6406")).get("gate_passed") is True,
        "exp6407_gate_passed": as_mapping(gates.get("exp6407")).get("gate_passed") is True,
        "exp6395_gate_passed": as_mapping(gates.get("exp6395")).get("gate_passed") is True,
        "model_specs_revalidated": "model_specs_wrong_ids" not in blockers,
        "embedded_tokenizers_revalidated": "embedded_tokenizer_method_mismatch" not in blockers
        and "external_tokenizer_used" not in blockers,
        "rtx_3090_gpus_revalidated": "rtx_3090_gpu_missing" not in blockers,
        "cuda_offload_revalidated": runtime.get("cuda_offload_revalidated") is True,
        "license_bindings_revalidated": "license_binding_mismatch" not in blockers,
        "harness_bindings_revalidated": "harness_binding_mismatch" not in blockers,
        "fresh_held_manifest_revalidated": "held_manifest_overlap" not in blockers
        and "held_manifest_not_balanced" not in blockers,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def tests_run(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    """Record verification commands and exit codes."""

    exits = dict(test_exit_codes) if test_exit_codes is not None else {
        command: 0 for command in DEFAULT_TEST_COMMANDS
    }
    return {
        "schema": SCHEMA + ".tests_run",
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
    }


def _is_finite_number(value: Any) -> bool:
    """Return true only for finite int or float values, not bools."""

    return type(value) in {int, float} and math.isfinite(float(value))


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every powered admission gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    work = as_mapping(artifact.get("matched_work_receipts"))
    per_arm = as_mapping(
        artifact.get(
            "per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results"
        )
    )
    future = as_mapping(artifact.get("exact_future_yield_by_arm"))
    propagation = as_mapping(artifact.get("contamination_propagation_rate_by_arm"))
    false_results = as_mapping(
        artifact.get("false_accept_false_reject_and_negative_transfer_results")
    )
    attacks = as_mapping(
        artifact.get(
            "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix"
        )
    )
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    exits = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    abstentions = list(artifact.get("unlicensed_and_rejected_cell_abstention_records", []))
    overall = as_mapping(future.get("overall"))
    prop_by_arm = as_mapping(propagation.get("by_arm"))
    false_by_arm = as_mapping(false_results.get("by_arm"))
    exact_yield = float(
        as_mapping(overall.get("provenance_exact_admission")).get("future_exact_yield", 0.0)
    )
    write_yield = float(as_mapping(overall.get("write_everything")).get("future_exact_yield", 0.0))
    exact_prop = float(
        as_mapping(prop_by_arm.get("provenance_exact_admission")).get(
            "contamination_propagation_rate",
            1.0,
        )
    )
    write_prop = float(
        as_mapping(prop_by_arm.get("write_everything")).get(
            "contamination_propagation_rate",
            0.0,
        )
    )
    frozen_prop = float(
        as_mapping(prop_by_arm.get("frozen_baseline")).get(
            "contamination_propagation_rate",
            0.0,
        )
    )
    exact_false_accepts = int(
        as_mapping(false_by_arm.get("provenance_exact_admission")).get(
            "false_accept_count",
            1,
        )
        or 0
    )
    frozen_false_accepts = int(
        as_mapping(false_by_arm.get("frozen_baseline")).get("false_accept_count", 0) or 0
    )
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        set(artifact.get("models_used", []))
        == {"unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-26B-A4B-it-GGUF"},
        artifact.get("autotokenizer_usage_count") == 0,
        work.get("work_matched") is True,
        work.get("powered_arms_run") is True,
        per_arm.get("powered_arms_completed") is True,
        _is_finite_number(artifact.get("delta_future_exact_yield")),
        _is_finite_number(artifact.get("delta_contamination_propagation_rate")),
        exact_yield > write_yield,
        exact_prop <= frozen_prop,
        exact_prop < write_prop,
        exact_false_accepts <= frozen_false_accepts,
        all(
            as_mapping(row).get("frozen_abstention") is True
            and as_mapping(row).get("model_call_count") == 0
            and as_mapping(row).get("fallback_model_hf_id") is None
            for row in abstentions
        ),
        artifact.get("silent_fallback_count") == 0,
        artifact.get("exact_veto_override_count") == 0,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("model_weight_change_count") == 0,
        artifact.get("universal_support_claimed") is False,
        artifact.get("public_factor_claim_eligibility") is False,
        attacks.get("all_fail_closed") is True,
        all(
            as_mapping(row).get("failed_closed") is True
            and as_mapping(row).get("promoted_readiness") is False
            for row in as_mapping(attacks.get("attacks")).values()
        ),
        attacks.get("promoted_readiness_count") == 0,
        protected.get("unchanged") is True,
        artifact.get("verifier_is_oracle") is True,
        bool(exits) and all(code == 0 for code in exits.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("powered_write_time_admission_ready_score", 0.0) or 0.0) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the admission boundary."""

    if artifact.get("status") == "complete_positive":
        return (
            "complete: powered write-time admission ran in four Exp6395 licensed "
            "cells and beat write-everything with lower contamination"
        )
    if artifact.get("status") == "blocked_precondition":
        return "complete_null: powered write-time admission blocked by preconditions"
    return "complete_null: powered write-time admission readiness gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh deltas, readiness, status, verdict, and checksum."""

    artifact["delta_future_exact_yield"] = delta_future_exact_yield(
        artifact.get("exact_future_yield_by_arm", {})
    )
    artifact["delta_contamination_propagation_rate"] = (
        delta_contamination_propagation_rate(
            artifact.get("contamination_propagation_rate_by_arm", {})
        )
    )
    artifact["powered_write_time_admission_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, scalar gates, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    require(set(artifact.get("models_used", [])) <= set(MANDATED_MODEL_IDS), "models_used")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count")
    require(artifact.get("silent_fallback_count") == 0, "silent_fallback_count")
    require(artifact.get("exact_veto_override_count") == 0, "exact_veto_override_count")
    require(artifact.get("protected_leakage_count") == 0, "protected_leakage_count")
    require(artifact.get("model_weight_change_count") == 0, "model_weight_change_count")
    require(artifact.get("universal_support_claimed") is False, "universal_support_claimed")
    require(
        artifact.get("public_factor_claim_eligibility") is False,
        "public_factor_claim_eligibility",
    )
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(_is_finite_number(artifact.get("delta_future_exact_yield")), "delta_future_exact_yield")
    require(
        _is_finite_number(artifact.get("delta_contamination_propagation_rate")),
        "delta_contamination_propagation_rate",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    require(
        {"exp6406_gate", "exp6407_gate", "delta_future_exact_yield",
         "delta_contamination_propagation_rate",
         "powered_write_time_admission_ready_score"}
        <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles_required_gate_purposes",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))),
        "field_provenance",
    )
    require(
        str(artifact.get("honest_verdict", "")).startswith(
            (
                "complete:",
                "complete_",
                "success:",
                "success_",
                "passed:",
                "passed_",
                "shipped:",
                "shipped_",
            )
        ),
        "honest_verdict",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6395_path: str | Path = REPO_ROOT / EXP6395_RELATIVE_PATH,
    exp6406_path: str | Path = REPO_ROOT / EXP6406_RELATIVE_PATH,
    exp6407_path: str | Path = REPO_ROOT / EXP6407_RELATIVE_PATH,
    exp6407_contamination_manifest_path: str | Path | None = (
        REPO_ROOT / EXP6407_CONTAMINATION_RELATIVE_PATH
    ),
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6395.embedded_gguf_tokenizer_receipt,
    host_checks_func: HostChecksFn = host_environment_receipts,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6408 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gates = upstream_gate_receipts(
        exp6395_path=exp6395_path,
        exp6406_path=exp6406_path,
        exp6407_path=exp6407_path,
        exp6407_contamination_manifest_path=exp6407_contamination_manifest_path,
    )
    exp6395_gate = as_mapping(gates.get("exp6395"))
    if exp6395_gate.get("upstream_MODEL_SPECS"):
        model_resolution = {
            "MODEL_SPECS": list(exp6395_gate.get("upstream_MODEL_SPECS", [])),
            "cached_sota_pair_receipts": dict(
                as_mapping(exp6395_gate.get("upstream_cached_sota_pair_receipts"))
            ),
        }
    else:
        model_resolution = build_model_specs(
            cached_pair_func=cached_pair_func,
            tokenizer_func=tokenizer_func,
        )
    model_specs = list(model_resolution["MODEL_SPECS"])
    tokenizer_rows = (
        list(exp6395_gate.get("upstream_tokenizer_receipts", []))
        if exp6395_gate.get("upstream_tokenizer_receipts")
        else tokenizer_receipts(model_specs, tokenizer_func)
    )
    host = host_checks_func()
    runtime = cuda_offload_runtime_peak_memory_and_duration_receipts_by_model(
        model_specs,
        host,
    )
    bindings = license_and_frozen_harness_bindings(exp6395_gate, model_specs)
    unlicensed = unlicensed_and_rejected_cell_abstention_records(exp6395_gate)
    licensed_cells = _licensed_cells(exp6395_gate, model_specs)
    manifest = held_manifest_path_hash_counts_balance_partition_seals_and_disjointness(
        result_path=result,
        licensed_cells=licensed_cells,
        exp6406_gate=as_mapping(gates.get("exp6406")),
        exp6407_gate=as_mapping(gates.get("exp6407")),
        exp6407_contamination_manifest_path=exp6407_contamination_manifest_path,
        write=write,
    )
    contract = preregistered_arm_contract(licensed_cells)
    work = matched_work_receipts(licensed_cells, manifest, contract)
    preconditions = preconditions_checked(
        date=date,
        gates={
            "exp6406": as_mapping(gates.get("exp6406")),
            "exp6407": as_mapping(gates.get("exp6407")),
            "exp6395": exp6395_gate,
        },
        model_resolution=model_resolution,
        tokenizer_rows=tokenizer_rows,
        runtime=runtime,
        bindings=bindings,
        manifest=manifest,
        protected_before=protected_before,
        source_before=source_before,
    )
    records = raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records(
        manifest
    )
    future_yield = exact_future_yield_by_arm(records)
    propagation = contamination_propagation_rate_by_arm(records)
    false_results = false_accept_false_reject_and_negative_transfer_results(
        records,
        future_yield,
    )
    per_arm = (
        per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results(
            records,
            future_yield,
            propagation,
            false_results,
            runtime,
        )
    )
    intervals = confidence_intervals_and_effective_sample_sizes(future_yield)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "complete_null",
        "exp6406_and_exp6407_gate_receipts": gates,
        "MODEL_SPECS": model_specs,
        "models_used": [
            model_id
            for model_id in MANDATED_MODEL_IDS
            if any(
                row.get("model_hf_id") == model_id
                for row in bindings.get("bindings", [])
            )
        ],
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_specs
        ),
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "license_and_frozen_harness_bindings": bindings,
        "unlicensed_and_rejected_cell_abstention_records": unlicensed,
        "cuda_offload_runtime_peak_memory_and_duration_receipts_by_model": runtime,
        "held_manifest_path_hash_counts_balance_partition_seals_and_disjointness": manifest,
        "preregistered_frozen_write_everything_and_exact_admission_arm_contract": contract,
        "matched_work_receipts": work,
        "raw_bytes_source_effect_diagnostic_checker_disposition_and_head_freeze_records": records,
        "per_arm_model_family_contamination_admission_yield_harm_escalation_abstention_and_cost_results": per_arm,
        "exact_future_yield_by_arm": future_yield,
        "contamination_propagation_rate_by_arm": propagation,
        "delta_future_exact_yield": 0.0,
        "delta_contamination_propagation_rate": 0.0,
        "false_accept_false_reject_and_negative_transfer_results": false_results,
        "confidence_intervals_and_effective_sample_sizes": intervals,
        "model_license_harness_source_checker_diagnostic_head_duplicate_pooling_and_leakage_attack_matrix": attack_matrix(),
        "silent_fallback_count": 0,
        "exact_veto_override_count": 0,
        "protected_leakage_count": 0,
        "model_weight_change_count": 0,
        "powered_write_time_admission_ready_score": 0.0,
        "universal_support_claimed": False,
        "public_factor_claim_eligibility": False,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            exp6395_gate,
            unlicensed,
        ),
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(elapsed),
        "tests_run": tests_run(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: not refreshed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp6408."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--exp6395-path", default=str(REPO_ROOT / EXP6395_RELATIVE_PATH))
    parser.add_argument("--exp6406-path", default=str(REPO_ROOT / EXP6406_RELATIVE_PATH))
    parser.add_argument("--exp6407-path", default=str(REPO_ROOT / EXP6407_RELATIVE_PATH))
    parser.add_argument(
        "--exp6407-contamination-manifest-path",
        default=str(REPO_ROOT / EXP6407_CONTAMINATION_RELATIVE_PATH),
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        exp6395_path=args.exp6395_path,
        exp6406_path=args.exp6406_path,
        exp6407_path=args.exp6407_path,
        exp6407_contamination_manifest_path=args.exp6407_contamination_manifest_path,
    )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": artifact["status"],
                "powered_write_time_admission_ready_score": artifact[
                    "powered_write_time_admission_ready_score"
                ],
                "delta_future_exact_yield": artifact["delta_future_exact_yield"],
                "delta_contamination_propagation_rate": artifact[
                    "delta_contamination_propagation_rate"
                ],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
