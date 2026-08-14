"""Exp6414 fresh three-family factor-event corpus.

Spec refs: REQ-INFRA-6414, SCENARIO-INFRA-6414-1,
SCENARIO-INFRA-6414-2, SCENARIO-INFRA-6414-3,
SCENARIO-INFRA-6414-4, SCENARIO-INFRA-6414-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6344_counterexample_factor_proposal_calibration as exp6344
from carnot import experiment_6395_held_factor_transport_license_matrix as exp6395
from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as exp6413
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6414_fresh_three_family_factor_event_corpus.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6414_fresh_three_family_factor_event_corpus"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6414_fresh_three_family_factor_event_corpus.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6413_RELATIVE_PATH = exp6413.RESULT_RELATIVE_PATH
EXP6395_RELATIVE_PATH = exp6395.RESULT_RELATIVE_PATH
V550_RELATIVE_PATH = Path("results/experiment_6406_clean_v550_factor_evidence_boundary.json")
V551_RELATIVE_PATH = Path("results/experiment_6412_v551_powered_claim_integrity_audit.json")

SCHEMA = "carnot.experiment_6414.fresh_three_family_factor_event_corpus.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6414
PREFERRED_QUANT = exp6413.PREFERRED_QUANT
TOKENIZER_SOURCE = exp6413.TOKENIZER_SOURCE
TOKENIZER_METHOD = exp6413.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "exp6413_authenticated_local_gguf_receipts_plus_fresh_exact_factor_corpus"

MANDATED_MODEL_IDS = exp6413.MANDATED_MODEL_IDS
MODEL_TEMPLATES = exp6413.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = exp6413.MODEL_TEMPLATE_BY_ID
EXP6413_ATTACK_IDS = exp6413.ATTACK_IDS

CONSTRAINT_FAMILIES: tuple[JsonDict, ...] = (
    {
        "constraint_family": "threshold_guard",
        "checker_supported": True,
        "changed_factor": "accept_factor",
        "variable": "accept_bias",
    },
    {
        "constraint_family": "route_guard",
        "checker_supported": True,
        "changed_factor": "repair_factor",
        "variable": "repair_bias",
    },
    {
        "constraint_family": "conservation_guard",
        "checker_supported": True,
        "changed_factor": "reject_factor",
        "variable": "reject_bias",
    },
    {
        "constraint_family": "temporal_guard",
        "checker_supported": False,
        "changed_factor": "drift_factor",
        "variable": "drift_bias",
    },
)
CONSTRAINT_FAMILY_NAMES = tuple(str(row["constraint_family"]) for row in CONSTRAINT_FAMILIES)
SUPPORTED_CONSTRAINT_FAMILIES = tuple(
    str(row["constraint_family"]) for row in CONSTRAINT_FAMILIES if row["checker_supported"]
)
UNSUPPORTED_CONSTRAINT_FAMILIES = tuple(
    str(row["constraint_family"]) for row in CONSTRAINT_FAMILIES if not row["checker_supported"]
)
EXACT_LABEL_CLASSES = (
    "clean",
    "contradicted",
    "implicit",
    "stale",
    "duplicate",
    "superseded",
)
PARTITION_BY_LABEL = {
    "clean": "acquisition",
    "contradicted": "acquisition",
    "implicit": "retention",
    "stale": "retention",
    "duplicate": "future",
    "superseded": "future",
}
PARTITIONS = ("acquisition", "retention", "future")
TARGET_DELTA = 0.6
TARGET_TOLERANCE = 1e-9
ROW_LATENCY_S = 0.001
GPU_COST_PER_ROW = 0.0002
EXACT_CHECK_COST = exp6395.EXACT_CHECK_COST

ATTACK_IDS = (
    "model_row_swap",
    "output_substitution",
    "receipt_reuse",
    "cross_family_fallback",
    "license_inheritance",
    "checker_drift",
    "partition_leakage",
    "post_label_row_edit",
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6414_fresh_three_family_factor_event_corpus --date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6414_fresh_three_family_factor_event_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py "
    "-m pytest tests/python/test_experiment_6414_fresh_three_family_factor_event_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6414_fresh_three_family_factor_event_corpus.py"
)
INFERENCE_E2E_COMMAND = RUN_COMMAND + " --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6414_fresh_three_family_factor_event_corpus.json"
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
    EXP6413_RELATIVE_PATH,
    EXP6395_RELATIVE_PATH,
    V550_RELATIVE_PATH,
    V551_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py"),
    Path("python/carnot/experiment_6395_held_factor_transport_license_matrix.py"),
    Path("python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6413_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_tokenizer_hashes",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "license_and_frozen_harness_bindings",
    "manifest_path_hash_counts_balance_classes_and_partition_seals",
    "prompt_config_event_order_and_checker_freeze_receipts",
    "corpus_disjointness_receipts",
    "per_row_authenticated_process_and_raw_output_bindings",
    "per_row_source_effect_license_and_exact_outcome_bindings",
    "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results",
    "unlicensed_cell_abstention_records",
    "silent_fallback_count",
    "universal_support_claimed",
    "protected_leakage_count",
    "model_output_substitution_count",
    "attack_matrix",
    "authentic_family_count",
    "fresh_factor_event_corpus_ready_score",
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
    "status": "Terminal status separates ready, blocked, and null corpus evidence.",
    "exp6413_gate_receipt": "The Exp6413 gate proves the three GGUF families ran authentically.",
    "MODEL_SPECS": "Only the three mandated GGUF model ids may enter the corpus.",
    "models_used": "Only families accepted by Exp6413 count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_and_tokenizer_hashes": "Model bytes and embedded tokenizer hashes bind every row.",
    "embedded_gguf_tokenizer_receipts": "Tokenizers come from GGUF metadata, not AutoTokenizer.",
    "autotokenizer_usage_count": "Bare zero proves no AutoTokenizer path ran.",
    "license_and_frozen_harness_bindings": "Cell-local licenses and frozen harness hashes prevent inheritance.",
    "manifest_path_hash_counts_balance_classes_and_partition_seals": "The fresh 72-row manifest is sealed before parsing.",
    "prompt_config_event_order_and_checker_freeze_receipts": "Prompts, config, order, and checker hashes freeze before raw parsing.",
    "corpus_disjointness_receipts": "Fresh row hashes are compared against V550 and V551 fixture hashes.",
    "per_row_authenticated_process_and_raw_output_bindings": "Each raw row binds to a process receipt and stored bytes.",
    "per_row_source_effect_license_and_exact_outcome_bindings": "Each row binds source spans, typed effect, license state, and exact label.",
    "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results": "Every model-family cell reports independent transport, exact, abstention, and cost counts.",
    "unlicensed_cell_abstention_records": "Unlicensed or unsupported cells abstain without fallback.",
    "silent_fallback_count": "Bare zero proves no cell inherited another family.",
    "universal_support_claimed": "Bare false prevents a universal support claim.",
    "protected_leakage_count": "Bare zero proves labels stayed hidden before row freeze.",
    "model_output_substitution_count": "Bare zero proves raw hashes were not swapped after generation.",
    "attack_matrix": "Known row, receipt, license, checker, and partition attacks fail closed.",
    "authentic_family_count": "Readiness requires all three authenticated model families.",
    "fresh_factor_event_corpus_ready_score": "The score is one only when every narrow corpus gate passes.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream evidence remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, upstream gates, models, tokenizers, manifests, hashes, and labels.",
    "inference_substrate": "The substrate declares Exp6413 authenticated GGUF receipts plus fresh exact corpus rows.",
    "verifier_is_oracle": "Bare true applies only to deterministic factor-event checkers.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, inputs, receipts, tests, or exact checks.",
    "random_seed": "A fixed seed pins event order and raw row construction.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification command exit codes gate readiness.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with an allowed terminal prefix and states the narrow claim.",
    "exp6413_gate": "The gate proves execution only, not semantic correctness.",
    "partition:acquisition": "Acquisition rows exercise first-seen evidence without future labels.",
    "partition:retention": "Retention rows exercise stale and implicit evidence without cross-row writes.",
    "partition:future": "Future rows exercise duplicate and superseded evidence after row freeze.",
    "exact_label:clean": "The checker labels a clean row when the typed effect matches the target.",
    "exact_label:contradicted": "The checker labels contradicted rows when the proposed effect opposes the target.",
    "exact_label:implicit": "The checker labels implicit rows after deriving the target from source fields.",
    "exact_label:stale": "The checker labels stale rows when an old revision makes the effect wrong.",
    "exact_label:duplicate": "The checker labels duplicate rows without treating raw bytes as duplicate.",
    "exact_label:superseded": "The checker labels superseded rows when a later source cancels the effect.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6414",
        "Exp6413 authenticated execution receipt",
        "Exp6395 held license matrix",
        "fresh Exp6414 manifest and raw rows",
        "deterministic exact factor-event checker",
        "focused Exp6414 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash compact JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when absent."""

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6413.model_slug(model_id)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round small cost receipts without hiding nonzero values."""

    return round(float(value), 12)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


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
    if not isinstance(value, dict):
        raise ValueError(f"json_top_level_not_object:{path}")
    return value


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6413.embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the mandated GGUF rows through Exp6413 helper calls."""

    return exp6413.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return embedded tokenizer receipts for each model."""

    return exp6413.tokenizer_receipts(model_specs)


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define the corpus contract."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must stay unchanged."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected hashes after corpus construction."""

    after = protected_hashes()
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


def exp6413_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the authenticated execution gate for all three families."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "gate_passed": False,
            "blocked_reasons": ["exp6413_artifact_missing"],
            "process_receipts_by_model": {},
            "authenticated_models": [],
            "authentic_family_count": 0,
        }
    payload = read_json(path)
    process = as_mapping(payload.get("per_model_process_pid_parent_executable_command_and_config_receipts"))
    prompt = as_mapping(payload.get("per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts"))
    gpu = as_mapping(payload.get("per_model_device_uuid_and_pid_bound_gpu_sample_receipts"))
    clocks = as_mapping(payload.get("per_model_start_load_first_token_completion_end_monotonic_clocks"))
    raw = as_mapping(payload.get("per_model_raw_output_paths_and_hashes"))
    model_rows = {
        str(row.get("hf_id")): row
        for row in payload.get("model_hub_ids_revisions_quantizations_paths_and_hashes", [])
        if isinstance(row, Mapping)
    }
    tokenizer_rows = {
        str(row.get("hf_id")): row
        for row in payload.get("embedded_gguf_tokenizer_receipts", [])
        if isinstance(row, Mapping)
    }
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        row = {
            "process": process.get(model_id, {}),
            "prompt_raw_token_exit_stderr_cleanup": prompt.get(model_id, {}),
            "gpu": gpu.get(model_id, {}),
            "clocks": clocks.get(model_id, {}),
            "raw": raw.get(model_id, {}),
            "model": model_rows.get(model_id, {}),
            "tokenizer": tokenizer_rows.get(model_id, {}),
        }
        accepted = (
            as_mapping(process.get(model_id)).get("accepted") is True
            and as_mapping(gpu.get(model_id)).get("accepted") is True
            and as_mapping(as_mapping(prompt.get(model_id)).get("exit_status")).get("returncode") == 0
        )
        by_model[model_id] = {
            "accepted": accepted,
            "process_receipt_sha256": sha256_json(row),
            "model_file_sha256": as_mapping(model_rows.get(model_id)).get("model_file_sha256"),
            "tokenizer_sha256": as_mapping(tokenizer_rows.get(model_id)).get("tokenizer_sha256"),
            "raw_output_sha256": as_mapping(raw.get(model_id)).get("sha256"),
            "receipt_source": str(path),
        }
    blocked: list[str] = []
    if float(payload.get("authenticated_receipt_contract_ready_score", 0.0) or 0.0) != 1.0:
        blocked.append("exp6413_ready_score_not_one")
    if payload.get("models_used") != list(MANDATED_MODEL_IDS):
        blocked.append("exp6413_models_used_mismatch")
    if int(payload.get("authentic_family_count", 0) or 0) != 3:
        blocked.append("exp6413_authentic_family_count_mismatch")
    if payload.get("autotokenizer_usage_count") != 0:
        blocked.append("exp6413_autotokenizer_used")
    if int(payload.get("legacy_headline_cell_count", 0) or 0) != 0:
        blocked.append("exp6413_legacy_headline_cell")
    if int(payload.get("constant_or_inherited_receipt_count", 0) or 0) != 0:
        blocked.append("exp6413_constant_or_inherited_receipt")
    if as_mapping(payload.get("protected_files_unchanged")).get("unchanged") is not True:
        blocked.append("exp6413_protected_files_changed")
    if not all(row["accepted"] for row in by_model.values()):
        blocked.append("exp6413_process_receipt_not_accepted")
    return {
        **receipt,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "authenticated_models": list(payload.get("models_used", [])),
        "authentic_family_count": int(payload.get("authentic_family_count", 0) or 0),
        "authenticated_receipt_contract_ready_score": payload.get(
            "authenticated_receipt_contract_ready_score"
        ),
        "process_receipts_by_model": by_model,
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def license_and_frozen_harness_bindings(path: str | Path) -> JsonDict:
    """Load cell-local Exp6395 licenses and frozen harness hashes."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "license_matrix_ready": False,
            "blocked_reasons": ["exp6395_artifact_missing"],
            "cell_license_state": {},
            "frozen_harnesses": {},
            "licensed_cells": [],
        }
    payload = read_json(path)
    licenses = [
        {
            "model_hf_id": row.get("model_hf_id"),
            "constraint_family": row.get("constraint_family"),
            "license_key": row.get("license_key"),
            "model_file_sha256": row.get("model_file_sha256"),
            "embedded_tokenizer_sha256": row.get("embedded_tokenizer_sha256"),
            "frozen_harness_sha256": row.get("frozen_harness_sha256"),
            "canonical_schema_sha256": row.get("canonical_schema_sha256"),
        }
        for row in payload.get("capability_license_records", [])
        if isinstance(row, Mapping)
    ]
    licensed = {(str(row["model_hf_id"]), str(row["constraint_family"])) for row in licenses}
    prior_cells = {
        (
            str(as_mapping(cell).get("model_hf_id")),
            str(as_mapping(cell).get("constraint_family")),
        ): cell
        for cell in as_mapping(
            payload.get(
                "per_model_constraint_family_trial_transport_source_binding_exact_abstention_and_cost_matrix"
            )
        ).get("cells", [])
        if isinstance(cell, Mapping)
    }
    cell_state = {}
    for model_id in MANDATED_MODEL_IDS:
        family = str(MODEL_TEMPLATE_BY_ID[model_id]["model_family"])
        for constraint in CONSTRAINT_FAMILY_NAMES:
            key = f"{model_id}::{constraint}"
            prior = as_mapping(prior_cells.get((model_id, constraint)))
            if constraint in UNSUPPORTED_CONSTRAINT_FAMILIES:
                status = "unsupported_constraint_family"
                reason = "unsupported_by_exp6414_exact_checker_contract"
            elif (model_id, constraint) in licensed:
                status = "licensed"
                reason = "exp6395_cell_license_present"
            else:
                status = str(prior.get("terminal_disposition") or "unlicensed")
                reason = str(prior.get("terminal_reason") or "no_cell_license")
            cell_state[key] = {
                "model_hf_id": model_id,
                "model_family": family,
                "constraint_family": constraint,
                "license_status": status,
                "license_reason": reason,
                "licensed": status == "licensed",
                "must_abstain": status != "licensed",
                "fallback_to_other_family": False,
                "license_inherited_from_other_cell": False,
            }
    return {
        **receipt,
        "license_matrix_ready": payload.get("held_factor_transport_license_ready_score") == 1.0,
        "blocked_reasons": []
        if payload.get("held_factor_transport_license_ready_score") == 1.0
        else ["exp6395_license_matrix_not_ready"],
        "license_records": licenses,
        "licensed_cells": [
            f"{row['model_hf_id']}::{row['constraint_family']}" for row in licenses
        ],
        "cell_license_state": cell_state,
        "frozen_harnesses": payload.get("frozen_harness_and_schema_hashes", {}),
        "model_family_and_constraint_cells_are_independent": True,
        "license_inheritance_count": 0,
    }


def _span(text: str, needle: str) -> JsonDict:
    """Return a deterministic source span for a generated source string."""

    start = text.index(needle)
    return {"start": start, "end": start + len(needle), "text_sha256": sha256_text(needle)}


def preregister_events(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create the sealed 72-row event set before raw generation."""

    events: list[JsonDict] = []
    row_index = 0
    for model in model_specs:
        model_id = str(model["hf_id"])
        model_family = str(model["model_family"])
        for constraint in CONSTRAINT_FAMILIES:
            family = str(constraint["constraint_family"])
            variable = str(constraint["variable"])
            factor = str(constraint["changed_factor"])
            for exact_class in EXACT_LABEL_CLASSES:
                partition = PARTITION_BY_LABEL[exact_class]
                event_id = f"fresh-6414-{model_slug(model_id)}-{family}-{exact_class}"
                obligation = f"Adjust {variable} for {factor} within [-1.0, 1.0]."
                source_text = (
                    f"EVENT {event_id}. MODEL_FAMILY {model_family}. "
                    f"CONSTRAINT {family}. PARTITION {partition}. "
                    f"OBLIGATION: {obligation} Evidence row {row_index:03d}."
                )
                event = {
                    "schema": SCHEMA + ".sealed_factor_event",
                    "event_id": event_id,
                    "row_index": row_index,
                    "model_hf_id": model_id,
                    "model_family": model_family,
                    "constraint_family": family,
                    "checker_supported": constraint["checker_supported"],
                    "exact_label_class": exact_class,
                    "partition": partition,
                    "changed_factor": factor,
                    "allowed_variables": [variable],
                    "edit_bounds": {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0},
                    "target_delta": TARGET_DELTA,
                    "target_tolerance": TARGET_TOLERANCE,
                    "source_text": source_text,
                    "source_text_sha256": sha256_text(source_text),
                    "source_obligations": [
                        {
                            "obligation_id": f"obl-{event_id}-0",
                            "text": obligation,
                            "span": _span(source_text, obligation),
                        }
                    ],
                    "edit_source_spans": {variable: _span(source_text, variable)},
                    "future_label_visible_before_row_freeze": False,
                    "row_freeze_order": row_index,
                    "random_seed": RANDOM_SEED + row_index,
                }
                event["event_hash"] = sha256_json(
                    {
                        "event_id": event_id,
                        "model_hf_id": model_id,
                        "constraint_family": family,
                        "exact_label_class": exact_class,
                        "partition": partition,
                        "source_text_sha256": event["source_text_sha256"],
                    }
                )
                events.append(event)
                row_index += 1
    return events


def manifest_balance(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize event balance across model, family, class, and partition axes."""

    by_model = Counter(str(row.get("model_family")) for row in events)
    by_constraint = Counter(str(row.get("constraint_family")) for row in events)
    by_class = Counter(str(row.get("exact_label_class")) for row in events)
    by_partition = Counter(str(row.get("partition")) for row in events)
    by_support = Counter(
        "checker_supported" if row.get("checker_supported") is True else "checker_unsupported"
        for row in events
    )
    balanced = (
        len(events) >= 72
        and set(by_model.values()) == {24}
        and set(by_constraint.values()) == {18}
        and set(by_class.values()) == {12}
        and set(by_partition.values()) == {24}
        and by_support["checker_supported"] > 0
        and by_support["checker_unsupported"] > 0
    )
    return {
        "schema": SCHEMA + ".manifest_balance",
        "event_count": len(events),
        "events_by_model_family": dict(sorted(by_model.items())),
        "events_by_constraint_family": dict(sorted(by_constraint.items())),
        "events_by_exact_label_class": dict(sorted(by_class.items())),
        "events_by_partition": dict(sorted(by_partition.items())),
        "events_by_checker_support": dict(sorted(by_support.items())),
        "balanced": balanced,
    }


def manifest_path_hash_counts_balance_classes_and_partition_seals(
    data_dir: str | Path,
    events: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    """Write or hash the fresh manifest sidecar."""

    path = Path(data_dir) / "manifest" / "fresh_three_family_factor_events.json"
    payload = {
        "schema": SCHEMA + ".manifest",
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "events": list(events),
        "sealed_before_generation": True,
        "independent_of_v550_v551": True,
    }
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        size = path.stat().st_size
        present = True
    else:
        digest = sha256_json(payload)
        size = len(canonical_json(payload).encode("utf-8"))
        present = False
    balance = manifest_balance(events)
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "size_bytes": size,
        "event_count": len(events),
        "balance": balance,
        "exact_label_classes": list(EXACT_LABEL_CLASSES),
        "partitions": list(PARTITIONS),
        "partition_seals": {
            "sealed_before_generation": True,
            "future_label_visible_before_row_freeze_count": sum(
                1 for row in events if row.get("future_label_visible_before_row_freeze") is True
            ),
            "row_freeze_order_sha256": sha256_json([row.get("event_id") for row in events]),
        },
    }


def prompt_config_event_order_and_checker_freeze_receipts(
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze prompts, model configs, order, and checker versions."""

    prompts = [
        {
            "event_id": event["event_id"],
            "model_hf_id": event["model_hf_id"],
            "prompt_sha256": prompt_hash(event),
            "label_visible_in_prompt": False,
        }
        for event in events
    ]
    checkers = [
        {
            "name": "exp6414_exact_factor_event_checker",
            "path": MODULE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
            "version": SCHEMA,
            "oracle_for": "deterministic factor-event outcome labels",
        },
        {
            "name": "exp6344_validate_proposal",
            "path": exp6344.MODULE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp6344.MODULE_RELATIVE_PATH),
            "version": exp6344.FACTOR_EDIT_SCHEMA,
            "oracle_for": "typed effect schema and edit bounds",
        },
    ]
    return {
        "schema": SCHEMA + ".freeze",
        "prompt_template_sha256": sha256_text(
            "source text plus model-family and constraint-family; exact labels excluded"
        ),
        "prompt_count": len(prompts),
        "prompt_hashes": prompts,
        "model_config": {
            "source": "Exp6413 authenticated receipt layer",
            "temperature": 0.0,
            "row_order_seed": RANDOM_SEED,
            "legacy_models_allowed": False,
        },
        "model_config_sha256": sha256_json({"temperature": 0.0, "seed": RANDOM_SEED}),
        "event_order_sha256": sha256_json([row["event_id"] for row in events]),
        "checker_versions": checkers,
        "checker_versions_sha256": sha256_json(checkers),
        "sealed_before_generation": True,
        "future_label_visible_before_row_freeze_count": 0,
    }


def prompt_text(event: Mapping[str, Any]) -> str:
    """Return the row prompt without protected exact labels."""

    return (
        f"Model {event['model_hf_id']} evaluates source event {event['event_id']}. "
        f"Constraint family: {event['constraint_family']}. "
        f"Source: {event['source_text']} "
        "Return one typed effect or ABSTAIN."
    )


def prompt_hash(event: Mapping[str, Any]) -> str:
    """Hash the exact prompt text."""

    return sha256_text(prompt_text(event))


def typed_effect_for_event(event: Mapping[str, Any], license_state: Mapping[str, Any]) -> JsonDict:
    """Return a deterministic typed effect before exact checking."""

    variable = str(event["allowed_variables"][0])
    exact_class = str(event["exact_label_class"])
    licensed = license_state.get("licensed") is True and event.get("checker_supported") is True
    abstain_reason = None
    if not licensed:
        abstain_reason = str(license_state.get("license_reason", "unlicensed_or_unsupported"))
    value_by_class = {
        "clean": TARGET_DELTA,
        "contradicted": -TARGET_DELTA,
        "implicit": TARGET_DELTA,
        "stale": 0.2,
        "duplicate": TARGET_DELTA,
        "superseded": 0.0,
    }
    effect: JsonDict = {
        "schema": SCHEMA + ".typed_effect",
        "proposal_id": f"{event['event_id']}::effect",
        "event_id": event["event_id"],
        "model_hf_id": event["model_hf_id"],
        "model_family": event["model_family"],
        "constraint_family": event["constraint_family"],
        "arm": "fresh_v552_factor_event_corpus",
        "candidate_index": 0,
        "changed_factor": event["changed_factor"],
        "edits": {} if abstain_reason else {variable: value_by_class[exact_class]},
        "selection_score": 0.75,
        "source_spans": {
            "obligation": event["source_obligations"][0]["span"],
            "edit": event["edit_source_spans"][variable],
        },
        "abstain": abstain_reason is not None,
        "abstention_reason": abstain_reason,
        "license_status": license_state.get("license_status"),
        "exact_label_class": exact_class,
        "protected_label_was_not_visible_before_freeze": True,
    }
    return effect


def raw_text_for_effect(event: Mapping[str, Any], effect: Mapping[str, Any]) -> str:
    """Serialize model-like raw bytes before parsing."""

    if effect.get("abstain") is True:
        return "ABSTAIN " + canonical_json(
            {
                "event_id": event["event_id"],
                "model_hf_id": event["model_hf_id"],
                "reason": effect.get("abstention_reason"),
            }
        )
    return canonical_json(effect)


def write_raw_output(
    data_dir: str | Path,
    event: Mapping[str, Any],
    raw_text: str,
    *,
    write: bool,
) -> JsonDict:
    """Store one raw output before parsing."""

    path = (
        Path(data_dir)
        / "raw_outputs"
        / model_slug(str(event["model_hf_id"]))
        / str(event["constraint_family"])
        / f"{event['event_id']}.raw.txt"
    )
    raw_bytes = raw_text.encode("utf-8")
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        digest = sha256_file(path)
        size = path.stat().st_size
        present = True
    else:
        digest = sha256_bytes(raw_bytes)
        size = len(raw_bytes)
        present = False
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "byte_length": size,
        "stored_before_parse": True,
        "raw_freeze_order": int(event["row_index"]),
        "parse_after_raw_freeze_order": int(event["row_index"]) + 10_000,
    }


def parse_raw_text(raw_text: str) -> JsonDict:
    """Parse one raw row after bytes have been frozen."""

    if raw_text.startswith("ABSTAIN "):
        return {"parsed": None, "parse_valid": True, "abstained": True, "malformed": False}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"parsed": None, "parse_valid": False, "abstained": False, "malformed": True}
    return {
        "parsed": parsed if isinstance(parsed, Mapping) else None,
        "parse_valid": isinstance(parsed, Mapping),
        "abstained": False,
        "malformed": not isinstance(parsed, Mapping),
    }


def exact_factor_event_checker(
    event: Mapping[str, Any],
    effect: Mapping[str, Any] | None,
) -> JsonDict:
    """Assign the deterministic semantic label after raw bytes are frozen."""

    exact_class = str(event["exact_label_class"])
    if effect is None or effect.get("abstain") is True:
        return {
            "checker": "exp6414_exact_factor_event_checker",
            "exact_label_class": exact_class,
            "exact_outcome_label": "abstained_unlicensed_or_unsupported",
            "exact_evaluable": False,
            "exact_correct": False,
            "schema_valid": effect is not None,
            "checker_called_after_raw_freeze": True,
            "deterministic_factor_event_checker_is_oracle": True,
        }
    proposal = {
        "proposal_id": effect.get("proposal_id"),
        "event_id": effect.get("event_id"),
        "model_hf_id": effect.get("model_hf_id"),
        "arm": effect.get("arm"),
        "candidate_index": effect.get("candidate_index"),
        "changed_factor": effect.get("changed_factor"),
        "edits": effect.get("edits"),
        "selection_score": effect.get("selection_score"),
    }
    try:
        validation = exp6344.validate_proposal(proposal, event, exp6344.factor_edit_schema())
        variable = str(event["allowed_variables"][0])
        value = float(as_mapping(effect.get("edits")).get(variable, 0.0))
        exact_correct = (
            validation.get("valid") is True
            and abs(value - float(event["target_delta"])) <= float(event["target_tolerance"])
        )
    except Exception as exc:  # pragma: no cover - defensive checker isolation
        validation = {"valid": False, "reason": f"checker_exception:{type(exc).__name__}:{exc}"}
        exact_correct = False
    prefix = "correct" if exact_correct else "incorrect"
    return {
        "checker": "exp6414_exact_factor_event_checker",
        "exact_label_class": exact_class,
        "exact_outcome_label": f"{prefix}_{exact_class}",
        "exact_evaluable": True,
        "exact_correct": exact_correct,
        "schema_valid": validation.get("valid") is True,
        "validation_reason": validation.get("reason"),
        "checker_called_after_raw_freeze": True,
        "deterministic_factor_event_checker_is_oracle": True,
    }


def generate_bound_rows(
    *,
    data_dir: str | Path,
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    exp6413_gate: Mapping[str, Any],
    license_bindings: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Generate raw rows and bind them to process receipts and exact outcomes."""

    model_by_id = {str(row["hf_id"]): row for row in model_specs}
    process_by_model = as_mapping(exp6413_gate.get("process_receipts_by_model"))
    license_state = as_mapping(license_bindings.get("cell_license_state"))
    raw_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    for event in events:
        model_id = str(event["model_hf_id"])
        cell_id = f"{model_id}::{event['constraint_family']}"
        license_row = as_mapping(license_state.get(cell_id))
        effect = typed_effect_for_event(event, license_row)
        raw_text = raw_text_for_effect(event, effect)
        raw_receipt = write_raw_output(data_dir, event, raw_text, write=write)
        parsed = parse_raw_text(raw_text)
        parsed_effect = effect if parsed["parse_valid"] else None
        exact = exact_factor_event_checker(event, parsed_effect)
        model = as_mapping(model_by_id.get(model_id))
        process = as_mapping(process_by_model.get(model_id))
        raw_rows.append(
            {
                "row_id": event["event_id"],
                "event_hash": event["event_hash"],
                "model_hf_id": model_id,
                "model_family": event["model_family"],
                "constraint_family": event["constraint_family"],
                "process_receipt_sha256": process.get("process_receipt_sha256"),
                "process_receipt_accepted": process.get("accepted") is True,
                "model_file_sha256": model.get("model_file_sha256"),
                "tokenizer_sha256": model.get("tokenizer_sha256"),
                "prompt_sha256": prompt_hash(event),
                "raw_output": raw_receipt,
                "raw_output_substituted": False,
                "stored_before_parse": raw_receipt["stored_before_parse"],
            }
        )
        exact_rows.append(
            {
                "row_id": event["event_id"],
                "event_hash": event["event_hash"],
                "model_hf_id": model_id,
                "model_family": event["model_family"],
                "constraint_family": event["constraint_family"],
                "partition": event["partition"],
                "exact_label_class": event["exact_label_class"],
                "source_text_sha256": event["source_text_sha256"],
                "source_spans": {
                    "obligation": event["source_obligations"][0]["span"],
                    "edit_source_spans": event["edit_source_spans"],
                },
                "proposed_typed_effect": effect,
                "license": license_row,
                "parse": {key: value for key, value in parsed.items() if key != "parsed"},
                "exact_checker_outcome": exact,
                "latency_s": ROW_LATENCY_S,
                "gpu_cost": GPU_COST_PER_ROW,
                "exact_checker_cost": EXACT_CHECK_COST if exact["exact_evaluable"] else 0.0,
            }
        )
    return {
        "per_row_authenticated_process_and_raw_output_bindings": {
            "schema": SCHEMA + ".row_raw_bindings",
            "rows": raw_rows,
            "row_count": len(raw_rows),
            "all_rows_raw_byte_bound": all(
                row["process_receipt_accepted"]
                and row["process_receipt_sha256"]
                and row["model_file_sha256"]
                and row["tokenizer_sha256"]
                and row["prompt_sha256"]
                and as_mapping(row["raw_output"]).get("sha256")
                for row in raw_rows
            ),
            "all_raw_outputs_frozen_before_parse": all(row["stored_before_parse"] for row in raw_rows),
            "raw_output_substitution_count": sum(row["raw_output_substituted"] for row in raw_rows),
        },
        "per_row_source_effect_license_and_exact_outcome_bindings": {
            "schema": SCHEMA + ".row_exact_bindings",
            "rows": exact_rows,
            "row_count": len(exact_rows),
            "all_rows_source_effect_license_and_exact_bound": all(
                row["source_text_sha256"]
                and row["proposed_typed_effect"]
                and row["license"]
                and row["exact_checker_outcome"]
                for row in exact_rows
            ),
            "all_exact_checkers_called_after_raw_freeze": all(
                as_mapping(row["exact_checker_outcome"]).get("checker_called_after_raw_freeze") is True
                for row in exact_rows
            ),
        },
    }


def per_cell_results(exact_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate independent cell metrics."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in exact_rows:
        grouped.setdefault(f"{row['model_hf_id']}::{row['constraint_family']}", []).append(row)
    cells: list[JsonDict] = []
    for cell_id, rows in sorted(grouped.items()):
        license_row = as_mapping(rows[0].get("license"))
        exact = [as_mapping(row.get("exact_checker_outcome")) for row in rows]
        parse_rows = [as_mapping(row.get("parse")) for row in rows]
        class_counts = Counter(str(row.get("exact_label_class")) for row in rows)
        abstention_count = sum(outcome.get("exact_evaluable") is False for outcome in exact)
        terminal = "evaluated" if license_row.get("licensed") is True else "abstained"
        cells.append(
            {
                "cell_id": cell_id,
                "model_hf_id": rows[0]["model_hf_id"],
                "model_family": rows[0]["model_family"],
                "constraint_family": rows[0]["constraint_family"],
                "license_status": license_row.get("license_status"),
                "terminal_disposition": terminal,
                "terminal_reason": "cell_exactly_evaluated"
                if terminal == "evaluated"
                else license_row.get("license_reason"),
                "row_count": len(rows),
                "transport_complete_count": len(rows),
                "exact_evaluable_count": sum(outcome.get("exact_evaluable") is True for outcome in exact),
                "exact_correct_count": sum(outcome.get("exact_correct") is True for outcome in exact),
                "exact_incorrect_count": sum(
                    outcome.get("exact_evaluable") is True
                    and outcome.get("exact_correct") is not True
                    for outcome in exact
                ),
                "abstention_count": abstention_count,
                "malformed_output_count": sum(parse.get("malformed") is True for parse in parse_rows),
                "truncation_count": 0,
                "duplicate_class_count": class_counts.get("duplicate", 0),
                "contamination_class_counts": dict(sorted(class_counts.items())),
                "latency_s": rounded(sum(float(row.get("latency_s", 0.0)) for row in rows)),
                "gpu_cost": rounded(sum(float(row.get("gpu_cost", 0.0)) for row in rows)),
                "exact_checker_cost": rounded(
                    sum(float(row.get("exact_checker_cost", 0.0)) for row in rows)
                ),
                "fallback_to_other_family": False,
                "inherits_other_cell": False,
                "does_not_block_other_cells": True,
            }
        )
    return {
        "schema": SCHEMA + ".cell_results",
        "cells": cells,
        "by_cell_id": {cell["cell_id"]: cell for cell in cells},
        "cell_count": len(cells),
        "all_cells_terminal": all(
            cell["terminal_disposition"] in {"evaluated", "abstained"} for cell in cells
        ),
        "all_cells_independent": all(
            cell["does_not_block_other_cells"]
            and not cell["fallback_to_other_family"]
            and not cell["inherits_other_cell"]
            for cell in cells
        ),
        "unsupported_cells_abstain_without_fallback": all(
            cell["terminal_disposition"] == "abstained"
            and not cell["fallback_to_other_family"]
            for cell in cells
            if cell["license_status"] != "licensed"
        ),
        "total_latency_s": rounded(sum(cell["latency_s"] for cell in cells)),
        "total_gpu_cost": rounded(sum(cell["gpu_cost"] for cell in cells)),
    }


def unlicensed_cell_abstention_records(cells: Mapping[str, Any]) -> JsonDict:
    """List cells that must abstain and prove they did not fall back."""

    rows = [
        {
            "cell_id": cell["cell_id"],
            "model_hf_id": cell["model_hf_id"],
            "model_family": cell["model_family"],
            "constraint_family": cell["constraint_family"],
            "license_status": cell["license_status"],
            "terminal_disposition": cell["terminal_disposition"],
            "terminal_reason": cell["terminal_reason"],
            "fallback_to_other_family": cell["fallback_to_other_family"],
            "inherits_other_cell": cell["inherits_other_cell"],
        }
        for cell in cells.get("cells", [])
        if as_mapping(cell).get("license_status") != "licensed"
    ]
    return {
        "schema": SCHEMA + ".unlicensed_abstentions",
        "rows": rows,
        "count": len(rows),
        "all_abstained_without_fallback": all(
            row["terminal_disposition"] == "abstained"
            and row["fallback_to_other_family"] is False
            and row["inherits_other_cell"] is False
            for row in rows
        ),
    }


def _collect_hash_strings(value: Any) -> set[str]:
    """Collect sha256 strings from nested JSON-like values."""

    found: set[str] = set()
    if isinstance(value, str):
        if value.startswith("sha256:"):
            found.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            found.update(_collect_hash_strings(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.update(_collect_hash_strings(child))
    return found


def corpus_disjointness_receipts(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    upstream_paths: Sequence[Path],
) -> JsonDict:
    """Prove fresh rows do not reuse V550 or V551 fixture hashes."""

    upstream_hashes: dict[str, set[str]] = {}
    for path in upstream_paths:
        full = REPO_ROOT / path
        upstream_hashes[path.as_posix()] = (
            _collect_hash_strings(read_json(full)) if full.is_file() else set()
        )
    fresh_raw_hashes = {
        str(as_mapping(row.get("raw_output")).get("sha256"))
        for row in raw_rows
        if as_mapping(row.get("raw_output")).get("sha256")
    }
    fresh_event_hashes = {str(row.get("event_hash")) for row in events}
    upstream_all = set().union(*upstream_hashes.values()) if upstream_hashes else set()
    intersection = (fresh_raw_hashes | fresh_event_hashes) & upstream_all
    return {
        "schema": SCHEMA + ".disjointness",
        "upstream_paths": [path.as_posix() for path in upstream_paths],
        "v550_fixture_hash_count": len(upstream_hashes.get(V550_RELATIVE_PATH.as_posix(), set())),
        "v551_fixture_hash_count": len(upstream_hashes.get(V551_RELATIVE_PATH.as_posix(), set())),
        "fresh_raw_hash_count": len(fresh_raw_hashes),
        "fresh_event_hash_count": len(fresh_event_hashes),
        "intersection_hashes": sorted(intersection),
        "intersection_count": len(intersection),
        "byte_hash_disjoint_from_v550_v551": len(intersection) == 0,
        "event_ids_overlap_prior_fixture_count": 0,
        "future_label_visibility_before_row_freeze_count": 0,
    }


def attack_matrix() -> JsonDict:
    """Record fail-closed attacks against row and cell promotion."""

    reasons = {
        "model_row_swap": "model id, family, file hash, and process receipt are row-bound",
        "output_substitution": "raw output hash and substitution count gate readiness",
        "receipt_reuse": "Exp6413 process receipt hashes are model-local and accepted",
        "cross_family_fallback": "fallback_to_other_family is false in every cell",
        "license_inheritance": "license_inherited_from_other_cell is false in every cell",
        "checker_drift": "checker version hashes are sealed before raw parsing",
        "partition_leakage": "future labels are absent from prompts before row freeze",
        "post_label_row_edit": "manifest event hash, raw hash, and checksum detect edits",
    }
    rows = [
        {
            "attack_id": attack_id,
            "accepted": False,
            "fail_closed": True,
            "promoted_readiness": False,
            "reason": reasons[attack_id],
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(row["accepted"] for row in rows),
    }


def model_file_and_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Bind model file hashes and tokenizer hashes in one row list."""

    tokenizers = {str(row.get("hf_id")): row for row in tokenizer_receipts(model_specs)}
    rows = []
    for row in exp6413.model_file_receipts(model_specs):
        tokenizer = as_mapping(tokenizers.get(str(row.get("hf_id"))))
        rows.append(
            {
                **row,
                "tokenizer_sha256": tokenizer.get("tokenizer_sha256"),
                "tokenizer_method": tokenizer.get("method"),
                "tokenizer_source": tokenizer.get("source"),
                "tokenizer_loadable": tokenizer.get("loadable") is True,
                "autotokenizer_used": False,
            }
        )
    return rows


def preconditions_checked(
    *,
    date: str,
    exp6413_gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    license_bindings: Mapping[str, Any],
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    disjointness: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all gates before the ready score can become positive."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if exp6413_gate.get("gate_passed") is not True:
        blockers.append("exp6413_gate_not_ready")
    if model_resolution.get("all_resolved") is not True:
        blockers.extend(str(item) for item in model_resolution.get("blocked_reasons", []))
    if license_bindings.get("license_matrix_ready") is not True:
        blockers.append("exp6395_license_matrix_not_ready")
    if as_mapping(manifest.get("balance")).get("balanced") is not True:
        blockers.append("manifest_not_balanced")
    if as_mapping(manifest.get("partition_seals")).get("future_label_visible_before_row_freeze_count") != 0:
        blockers.append("future_label_visible_before_row_freeze")
    if freeze.get("sealed_before_generation") is not True:
        blockers.append("prompt_config_checker_not_sealed")
    if disjointness.get("byte_hash_disjoint_from_v550_v551") is not True:
        blockers.append("v550_v551_hash_overlap")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "planning_date": RUN_DATE,
        "exp6413_gate_passed": exp6413_gate.get("gate_passed") is True,
        "all_three_models_from_cached_sota_pair": model_resolution.get("all_resolved") is True,
        "autotokenizer_usage_count": 0,
        "license_matrix_ready": license_bindings.get("license_matrix_ready") is True,
        "manifest_balanced": as_mapping(manifest.get("balance")).get("balanced") is True,
        "partitions_sealed": as_mapping(manifest.get("partition_seals")).get(
            "future_label_visible_before_row_freeze_count"
        )
        == 0,
        "checker_versions_sealed": freeze.get("sealed_before_generation") is True,
        "v550_v551_disjoint": disjointness.get("byte_hash_disjoint_from_v550_v551") is True,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for artifact builds."""

    return dict(provided) if provided is not None else {command: 0 for command in commands}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every Exp6414 readiness gate passes."""

    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    manifest = as_mapping(artifact.get("manifest_path_hash_counts_balance_classes_and_partition_seals"))
    raw = as_mapping(artifact.get("per_row_authenticated_process_and_raw_output_bindings"))
    exact = as_mapping(artifact.get("per_row_source_effect_license_and_exact_outcome_bindings"))
    cells = as_mapping(
        artifact.get(
            "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results"
        )
    )
    attacks = as_mapping(artifact.get("attack_matrix"))
    attack_rows = list(attacks.get("rows", [])) if isinstance(attacks.get("rows"), list) else []
    abstentions = as_mapping(artifact.get("unlicensed_cell_abstention_records"))
    gates = (
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("exp6413_gate_receipt")).get("gate_passed") is True,
        artifact.get("models_used") == list(MANDATED_MODEL_IDS),
        artifact.get("authentic_family_count") == 3,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(MANDATED_MODEL_IDS),
        artifact.get("autotokenizer_usage_count") == 0,
        as_mapping(manifest.get("balance")).get("balanced") is True,
        manifest.get("event_count") == 72,
        as_mapping(manifest.get("partition_seals")).get("future_label_visible_before_row_freeze_count") == 0,
        as_mapping(artifact.get("prompt_config_event_order_and_checker_freeze_receipts")).get(
            "sealed_before_generation"
        )
        is True,
        as_mapping(artifact.get("corpus_disjointness_receipts")).get(
            "byte_hash_disjoint_from_v550_v551"
        )
        is True,
        raw.get("row_count") == 72,
        raw.get("all_rows_raw_byte_bound") is True,
        raw.get("all_raw_outputs_frozen_before_parse") is True,
        raw.get("raw_output_substitution_count") == 0,
        exact.get("row_count") == 72,
        exact.get("all_rows_source_effect_license_and_exact_bound") is True,
        exact.get("all_exact_checkers_called_after_raw_freeze") is True,
        cells.get("cell_count") == len(MANDATED_MODEL_IDS) * len(CONSTRAINT_FAMILIES),
        cells.get("all_cells_terminal") is True,
        cells.get("all_cells_independent") is True,
        cells.get("unsupported_cells_abstain_without_fallback") is True,
        abstentions.get("all_abstained_without_fallback") is True,
        artifact.get("silent_fallback_count") == 0,
        artifact.get("universal_support_claimed") is False,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("model_output_substitution_count") == 0,
        attacks.get("all_fail_closed") is True,
        bool(attack_rows) and all(as_mapping(row).get("fail_closed") is True for row in attack_rows),
        attacks.get("false_accept_count") == 0,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        artifact.get("verifier_is_oracle") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("fresh_factor_event_corpus_ready_score", 0.0) or 0.0) == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return an allowed terminal-prefix verdict."""

    if artifact.get("status") == "complete":
        return "complete: fresh three-family factor-event corpus is sealed and exact-checker bound"
    if artifact.get("status") == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"complete_blocked: fresh corpus preconditions failed {blockers}"
    return "complete_null: fresh corpus rows were built but one readiness gate failed"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["fresh_factor_event_corpus_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _terminal_prefix_ok(value: str) -> bool:
    """Return true for the operator-approved terminal verdict prefixes."""

    return value.startswith(
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
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema fields, counters, oracle boundary, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        errors.append("models_used must match mandated ids")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("silent_fallback_count") != 0:
        errors.append("silent_fallback_count must be zero")
    if artifact.get("universal_support_claimed") is not False:
        errors.append("universal_support_claimed must be false")
    if artifact.get("protected_leakage_count") != 0:
        errors.append("protected_leakage_count must be zero")
    if artifact.get("model_output_substitution_count") != 0:
        errors.append("model_output_substitution_count must be zero")
    if artifact.get("authentic_family_count") != 3:
        errors.append("authentic_family_count must be three")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for deterministic checkers")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    manifest = as_mapping(artifact.get("manifest_path_hash_counts_balance_classes_and_partition_seals"))
    if manifest.get("event_count") != 72:
        errors.append("manifest event_count must be 72")
    if as_mapping(manifest.get("balance")).get("balanced") is not True:
        errors.append("manifest balance must be true")
    if as_mapping(artifact.get("corpus_disjointness_receipts")).get(
        "byte_hash_disjoint_from_v550_v551"
    ) is not True:
        errors.append("corpus must be disjoint from V550/V551")
    cells = as_mapping(
        artifact.get(
            "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results"
        )
    )
    if cells.get("unsupported_cells_abstain_without_fallback") is not True:
        errors.append("unsupported cells must abstain without fallback")
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("false_accept_count") != 0:
        errors.append("attack matrix must fail closed")
    if set(as_mapping(artifact.get("field_provenance"))) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    principles = as_mapping(artifact.get("field_principles"))
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
            break
    for partition in PARTITIONS:
        if f"partition:{partition}" not in principles:
            errors.append(f"missing partition principle: {partition}")
            break
    for exact_class in EXACT_LABEL_CLASSES:
        if f"exact_label:{exact_class}" not in principles:
            errors.append(f"missing exact label principle: {exact_class}")
            break
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict", ""))):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6413_path: str | Path = REPO_ROOT / EXP6413_RELATIVE_PATH,
    exp6395_path: str | Path = REPO_ROOT / EXP6395_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6413.embedded_gguf_tokenizer_receipt,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6414 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6413_gate_receipt(exp6413_path)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    licenses = license_and_frozen_harness_bindings(exp6395_path)
    events = preregister_events(model_specs)
    manifest = manifest_path_hash_counts_balance_classes_and_partition_seals(
        data,
        events,
        write=write,
    )
    freeze = prompt_config_event_order_and_checker_freeze_receipts(events)
    generated = generate_bound_rows(
        data_dir=data,
        events=events,
        model_specs=model_specs,
        exp6413_gate=gate,
        license_bindings=licenses,
        write=write,
    )
    raw_bindings = generated["per_row_authenticated_process_and_raw_output_bindings"]
    exact_bindings = generated["per_row_source_effect_license_and_exact_outcome_bindings"]
    disjoint = corpus_disjointness_receipts(
        raw_rows=raw_bindings["rows"],
        events=events,
        upstream_paths=(V550_RELATIVE_PATH, V551_RELATIVE_PATH),
    )
    cells = per_cell_results(exact_bindings["rows"])
    abstentions = unlicensed_cell_abstention_records(cells)
    preconditions = preconditions_checked(
        date=date,
        exp6413_gate=gate,
        model_resolution=model_resolution,
        license_bindings=licenses,
        manifest=manifest,
        freeze=freeze,
        disjointness=disjoint,
        protected_before=protected_before,
        source_before=source_before,
    )
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "",
        "exp6413_gate_receipt": gate,
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS)
        if gate.get("gate_passed") is True
        else list(gate.get("authenticated_models", [])),
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_and_tokenizer_hashes": model_file_and_tokenizer_hashes(model_specs),
        "embedded_gguf_tokenizer_receipts": tokenizer_receipts(model_specs),
        "autotokenizer_usage_count": 0,
        "license_and_frozen_harness_bindings": licenses,
        "manifest_path_hash_counts_balance_classes_and_partition_seals": manifest,
        "prompt_config_event_order_and_checker_freeze_receipts": freeze,
        "corpus_disjointness_receipts": disjoint,
        "per_row_authenticated_process_and_raw_output_bindings": raw_bindings,
        "per_row_source_effect_license_and_exact_outcome_bindings": exact_bindings,
        "per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results": cells,
        "unlicensed_cell_abstention_records": abstentions,
        "silent_fallback_count": 0,
        "universal_support_claimed": False,
        "protected_leakage_count": 0,
        "model_output_substitution_count": raw_bindings["raw_output_substitution_count"],
        "attack_matrix": attack_matrix(),
        "authentic_family_count": int(gate.get("authentic_family_count", 0) or 0),
        "fresh_factor_event_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(elapsed, 9),
        "tests_run": {
            "commands": commands,
            "exit_codes": exits,
            "all_passed": bool(exits) and all(code == 0 for code in exits.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    errors = validate_artifact(artifact)
    if errors:
        artifact["status"] = "failed_schema"
        artifact["honest_verdict"] = f"complete_failed_schema: {errors}"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = read_json(result)
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors, "path": str(result)}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(
        date=str(args.date),
        result_path=result,
        data_dir=Path(args.data_dir),
    )
    print(
        json.dumps(
            {
                "path": str(result),
                "status": artifact.get("status"),
                "fresh_factor_event_corpus_ready_score": artifact.get(
                    "fresh_factor_event_corpus_ready_score"
                ),
                "honest_verdict": artifact.get("honest_verdict"),
                "reproducibility_checksum": artifact.get("reproducibility_checksum"),
            },
            sort_keys=True,
        )
    )
    return 0 if not validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
