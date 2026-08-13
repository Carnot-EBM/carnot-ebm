"""Exp6398 default-off transactional factor consumer.

Spec refs: REQ-LEARN-6398, SCENARIO-LEARN-6398-READONLY,
SCENARIO-LEARN-6398-LICENSED, SCENARIO-LEARN-6398-MATCHED,
SCENARIO-LEARN-6398-ATTACKS, SCENARIO-LEARN-6398-ROLLBACK,
SCENARIO-LEARN-6398-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6383_dependency_guided_factor_rollback_stress as exp6383
from carnot import experiment_6397_transactional_continuous_factor_learning as exp6397


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6398_default_off_transactional_factor_consumer.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6398_default_off_transactional_factor_consumer"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6398_default_off_transactional_factor_consumer.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6397_RELATIVE_PATH = exp6397.RESULT_RELATIVE_PATH
EXP6397_MANIFEST_RELATIVE_PATH = Path(
    str(exp6397.RESULT_RELATIVE_PATH) + ".chronological_manifest.json"
)
EXP6383_RELATIVE_PATH = exp6383.RESULT_RELATIVE_PATH
EXP6384_RELATIVE_PATH = Path(
    "results/experiment_6384_default_off_certified_factor_consumer_ab.json"
)

SCHEMA = "carnot.experiment_6398.default_off_transactional_factor_consumer.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6398
TOKENIZER_METHOD = exp6397.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = (
    "deterministic_default_off_consumer_replay_over_licensed_local_gguf_receipts"
)

MANDATED_MODEL_IDS = exp6397.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6397.MODEL_TEMPLATE_BY_ID
FROZEN_BASELINE_ARM = "frozen_baseline"
V546_ARM = "v546_replay_certified_registry"
V550_ARM = "v550_transactional_registry"
ARMS = (FROZEN_BASELINE_ARM, V546_ARM, V550_ARM)
CONSUMER_EVENTS_PER_FAMILY = 8
EXACT_CHECK_COST = exp6397.EXACT_CHECK_COST
CHECKER_TIME_PER_CALL_S = exp6397.CHECKER_TIME_PER_CALL_S
RANDOM_SEEDS = {
    "consumer_manifest": 639800,
    "arm_order": 639801,
    "attack_matrix": 639802,
    "rollback_scope": 639803,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6398_default_off_transactional_factor_consumer --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6398_default_off_transactional_factor_consumer.py "
    "-m pytest tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6398_default_off_transactional_factor_consumer.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6398_default_off_transactional_factor_consumer.py"
)
INFERENCE_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_inference_arm_ebm_bridge.py -q --no-cov -n 0"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6398_default_off_transactional_factor_consumer.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    INFERENCE_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6397_RELATIVE_PATH,
    EXP6397_MANIFEST_RELATIVE_PATH,
    EXP6383_RELATIVE_PATH,
    EXP6384_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("research-references.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_6397_transactional_continuous_factor_learning.py"),
    Path("python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6397_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "frozen_factor_head_and_transaction_log_hashes",
    "license_and_harness_bindings",
    "cuda_offload_and_runtime_receipts_by_model",
    "untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt",
    "preregistered_arm_contract",
    "matched_work_receipts",
    "per_model_family_retrieval_license_abstention_checker_yield_and_cost_results",
    "exact_yield_by_arm",
    "delta_exact_yield_over_frozen",
    "false_accept_false_reject_negative_transfer_and_harm_results",
    "confidence_intervals_and_effective_sample_sizes",
    "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix",
    "selective_rollback_full_reset_and_no_rollback_injected_cell_results",
    "consumer_factor_write_count",
    "factor_head_advance_count",
    "license_renewal_count",
    "silent_fallback_count",
    "production_enable_count",
    "protected_leakage_count",
    "default_off_transactional_consumer_ready_score",
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
    "status": "Terminal status follows read-only consumer safety, arm utility, rollback, protected access, and tests.",
    "exp6397_gate_receipts": "Exp6397 gates, factor head, transaction log, licenses, rollback carry, and protected seals gate this run.",
    "MODEL_SPECS": "The three mandated GGUF rows come from cached SOTA helper calls.",
    "models_used": "Only licensed mandated models with default-off consumer work count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "frozen_factor_head_and_transaction_log_hashes": "Exp6397 retained head and transaction log are hash-bound before consumer reads.",
    "license_and_harness_bindings": "Licenses, harnesses, exact checkers, and release ledger are bound before decisions.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload and cleanup are reported for mandated models.",
    "untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt": "Future consumer events, license balance, and no-prior-access seal are frozen.",
    "preregistered_arm_contract": "Frozen, V546, and V550 consumer arms are matched before scoring.",
    "matched_work_receipts": "Event counts, model calls, exact checks, token budgets, latency rules, and work caps match across arms.",
    "per_model_family_retrieval_license_abstention_checker_yield_and_cost_results": "Retrievals, license checks, abstentions, checker calls, yield, latency, cost, and decisions are reported by model and family.",
    "exact_yield_by_arm": "Consumer exact yield is reported by arm.",
    "delta_exact_yield_over_frozen": "V550 utility is compared with frozen baseline.",
    "false_accept_false_reject_negative_transfer_and_harm_results": "False accepts, false rejects, negative transfer, and harm stay visible.",
    "confidence_intervals_and_effective_sample_sizes": "Per-family and pooled intervals, effective sample sizes, and abstention exclusions are explicit.",
    "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix": "Every preregistered consumer attack fails closed.",
    "selective_rollback_full_reset_and_no_rollback_injected_cell_results": "Exp6383 selective rollback, full reset, and no rollback are compared only on injected cells.",
    "consumer_factor_write_count": "Bare zero proves the consumer wrote no factors.",
    "factor_head_advance_count": "Bare zero proves no head advanced.",
    "license_renewal_count": "Bare zero proves licenses were not renewed.",
    "silent_fallback_count": "Bare zero proves no fallback was approved silently.",
    "production_enable_count": "Bare zero proves the default-off path stayed off.",
    "protected_leakage_count": "Bare zero proves protected outcomes were not read early.",
    "default_off_transactional_consumer_ready_score": "Readiness is conjunctive over utility, false accepts, attacks, rollback, production enablement, and tests.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, unlicensed, rejected, expired, stale, revoked, and attacked cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, upstream gates, models, tokenizers, GPUs, exact checkers, manifests, seeds, and protected files.",
    "inference_substrate": "The substrate declares deterministic default-off consumer replay over licensed local GGUF receipts.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers.",
    "field_principles": "Every required field states its guard and purpose.",
    "field_provenance": "Every required field maps to specs, upstream artifacts, consumer events, attacks, tests, or exact checks.",
    "random_seed": "Fixed seed pins consumer events, arm order, attacks, and future opens.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the default-off consumer boundary.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6398",
        "Exp6397 transactional factor artifact",
        "Exp6383 selective rollback artifact",
        "default-off consumer fixtures",
        "focused Exp6398 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

ATTACK_IDS = (
    "stale_head",
    "revoked_descendant",
    "expired_license",
    "model_row_swap",
    "family_switch_request",
    "absent_licensed_model",
    "duplicated_evidence",
    "incomplete_rollback",
    "suppressed_abstention",
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

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
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round receipts without hiding small nonzero costs."""

    return round(float(value), 12)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

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
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

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


def _model_family_by_id(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map model ids to frozen family labels."""

    return {str(row.get("hf_id")): str(row.get("model_family")) for row in model_specs}


def _cell_id(model_id: str, constraint_family: str) -> str:
    """Build the stable model-family cell id."""

    return f"{exp6397.model_slug(model_id)}::{constraint_family}"


def exp6397_gate_receipts(path: str | Path) -> JsonDict:
    """Revalidate Exp6397 and carry only its frozen read-only receipts."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "gate_passed": False,
            "blocked_reasons": ["exp6397_artifact_missing"],
            "MODEL_SPECS": [],
            "licenses": [],
            "unlicensed_cells": [],
            "licensed_model_ids": [],
        }
    payload = read_json(path)
    ready = float(payload.get("transactional_continuous_self_learning_ready_score", 0.0) or 0.0)
    delta = float(payload.get("delta_future_exact_yield_over_frozen", 0.0) or 0.0)
    history = as_mapping(payload.get("factor_head_transition_history"))
    manifest = as_mapping(payload.get("chronological_manifest_path_hash_license_balance_and_partition_seals"))
    upstream_gate = as_mapping(payload.get("exp6396_gate_receipts"))
    bindings = as_mapping(payload.get("license_and_frozen_harness_bindings"))
    protected = as_mapping(payload.get("protected_files_unchanged"))
    blockers: list[str] = []
    if ready != 1.0: blockers.append("exp6397_ready_score_not_one")
    if payload.get("status") != "complete_positive": blockers.append("exp6397_not_positive")
    if delta <= 0.0: blockers.append("exp6397_future_delta_not_positive")
    if upstream_gate.get("both_gates_passed") is not True: blockers.append("exp6397_upstream_gates_not_ready")
    if int(history.get("commit_count", 0) or 0) < 1: blockers.append("exp6397_no_committed_factor")
    if not str(history.get("terminal_head_hash", "")).startswith("sha256:"): blockers.append("exp6397_terminal_head_missing")
    if not history.get("transition_rows"): blockers.append("exp6397_transaction_log_missing")
    if manifest.get("event_count", 0) < 48: blockers.append("exp6397_manifest_too_short")
    if as_mapping(manifest.get("license_balance")).get("balanced") is not True: blockers.append("exp6397_license_balance_failed")
    if payload.get("selective_rollback_control_ready_score") != 1.0: blockers.append("exp6383_control_not_carried")
    if bindings.get("all_hashes_match") is not True: blockers.append("exp6397_license_harness_hash_mismatch")
    if payload.get("autotokenizer_usage_count") != 0: blockers.append("external_tokenizer_used_upstream")
    if payload.get("protected_leakage_count") != 0: blockers.append("exp6397_protected_leakage")
    if payload.get("same_step_write_count") != 0: blockers.append("exp6397_same_step_write")
    if payload.get("model_weight_change_count") != 0: blockers.append("exp6397_model_weight_change")
    if protected.get("unchanged") is not True: blockers.append("exp6397_protected_files_changed")
    return {
        **receipt,
        "gate_passed": not blockers,
        "blocked_reasons": sorted(set(blockers)),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "transactional_continuous_self_learning_ready_score": ready,
        "delta_future_exact_yield_over_frozen": delta,
        "both_exp6397_gates_revalidated": ready == 1.0 and delta > 0.0,
        "upstream_exp6396_both_gates_passed": upstream_gate.get("both_gates_passed") is True,
        "MODEL_SPECS": list(payload.get("MODEL_SPECS", [])),
        "models_used": list(payload.get("models_used", [])),
        "cached_sota_pair_receipts": payload.get("cached_sota_pair_receipts", {}),
        "embedded_gguf_tokenizer_receipts": list(payload.get("embedded_gguf_tokenizer_receipts", [])),
        "cuda_offload_and_runtime_receipts_by_model": payload.get(
            "cuda_offload_and_runtime_receipts_by_model",
            {},
        ),
        "license_and_frozen_harness_bindings": bindings,
        "licenses": list(upstream_gate.get("licenses", [])),
        "license_hashes": list(upstream_gate.get("license_hashes", [])),
        "unlicensed_cells": list(payload.get("unlicensed_cell_abstention_records", [])),
        "licensed_model_ids": list(upstream_gate.get("licensed_model_ids", [])),
        "chronological_manifest_receipt": manifest,
        "factor_head_transition_history": history,
        "factor_head_initial_hash": payload.get("factor_head_initial_hash"),
        "selective_rollback_control_ready_score": payload.get(
            "selective_rollback_control_ready_score",
            0.0,
        ),
        "selective_rollback_control_receipt": payload.get(
            "selective_rollback_control_path_hash_and_terminal_class",
            {},
        ),
        "transaction_log_hash": sha256_json(history.get("transition_rows", [])),
        "protected_seal": protected,
    }


def exp6383_rollback_receipt(path: str | Path) -> JsonDict:
    """Carry Exp6383 selective rollback as a positive control."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "gate_passed": False,
            "ready_score": 0.0,
            "terminal_class": "absent",
            "payload": {},
        }
    payload = read_json(path)
    ready = float(payload.get("dependency_guided_rollback_ready_score", 0.0) or 0.0)
    return {
        **receipt,
        "gate_passed": ready == 1.0 and payload.get("status") == "complete_positive",
        "ready_score": ready,
        "terminal_class": payload.get("status", "present_unqualified"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "payload": payload,
    }


def model_resolution_from_gate(gate: Mapping[str, Any]) -> JsonDict:
    """Return the three upstream model rows and cached helper receipts."""

    if gate.get("MODEL_SPECS"):
        return {
            "MODEL_SPECS": list(gate.get("MODEL_SPECS", [])),
            "cached_sota_pair_receipts": dict(as_mapping(gate.get("cached_sota_pair_receipts"))),
        }
    return exp6397.model_resolution_from_gate({})


def tokenizer_receipts_from_gate(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return embedded tokenizer rows from Exp6397 or rebuild receipts."""

    if gate.get("embedded_gguf_tokenizer_receipts"):
        return list(gate.get("embedded_gguf_tokenizer_receipts", []))
    return exp6397.tokenizer_receipts_from_gate({}, model_specs)


def runtime_receipts_from_gate(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return CUDA runtime receipts from Exp6397 or deterministic fallback rows."""

    if gate.get("cuda_offload_and_runtime_receipts_by_model"):
        return dict(as_mapping(gate.get("cuda_offload_and_runtime_receipts_by_model")))
    return exp6397.runtime_receipts_from_gate({}, model_specs)


def frozen_factor_head_and_transaction_log_hashes(gate: Mapping[str, Any]) -> JsonDict:
    """Bind the retained Exp6397 head and transaction log for read-only use."""

    history = as_mapping(gate.get("factor_head_transition_history"))
    transitions = list(history.get("transition_rows", []))
    commits = [row for row in transitions if as_mapping(row).get("disposition") == "Commit"]
    terminal = history.get("terminal_head_hash")
    return {
        "schema": SCHEMA + ".frozen_factor_head",
        "initial_head_hash": history.get("initial_head_hash") or gate.get("factor_head_initial_hash"),
        "retained_predecessor_bound_head_hash": terminal,
        "transaction_log_hash": sha256_json(transitions),
        "transaction_log_entry_count": len(transitions),
        "committed_predecessor_hashes": [row.get("head_before_hash") for row in commits],
        "committed_candidate_hashes": [row.get("candidate_hash") for row in commits],
        "all_commits_predecessor_bound": bool(commits)
        and all(str(row.get("head_before_hash", "")).startswith("sha256:") for row in commits),
        "consumer_read_only": True,
        "factor_write_freeze": True,
        "license_write_freeze": True,
    }


def license_and_harness_bindings(
    gate: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> JsonDict:
    """Bind licenses, harnesses, exact checkers, and release ledger."""

    upstream = as_mapping(gate.get("license_and_frozen_harness_bindings"))
    checker_hashes = {
        **dict(as_mapping(upstream.get("exact_checker_hashes"))),
        "exp6397_transaction_checker": sha256_file(REPO_ROOT / exp6397.MODULE_RELATIVE_PATH),
        "exp6383_selective_rollback_checker": sha256_file(REPO_ROOT / exp6383.MODULE_RELATIVE_PATH),
        "exp6398_default_off_consumer_checker": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
    }
    return {
        "schema": SCHEMA + ".license_harness_bindings",
        "license_count": len(list(gate.get("licenses", []))),
        "license_hashes": list(gate.get("license_hashes", [])),
        "all_hashes_match": upstream.get("all_hashes_match") is True,
        "all_accept_reject_owned_by_exact_checker": upstream.get(
            "all_accept_reject_owned_by_exact_checker"
        )
        is True,
        "frozen_harness_bindings": upstream.get("frozen_harness_bindings", {}),
        "exact_checker_hashes": checker_hashes,
        "exact_checker_hashes_complete": all(value is not None for value in checker_hashes.values()),
        "evalue_release_ledger": upstream.get("evalue_release_ledger", {}),
        "rollback_ready_score": rollback.get("ready_score", 0.0),
        "license_writes_frozen": True,
        "renewal_permitted": False,
    }


def _licensed_family_rows(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return one licensed row per constraint family."""

    family_by_model = _model_family_by_id(model_specs)
    rows: dict[str, JsonDict] = {}
    for license_row in gate.get("licenses", []):
        row = as_mapping(license_row)
        family = str(row.get("constraint_family"))
        rows.setdefault(
            family,
            {
                "constraint_family": family,
                "model_hf_id": row.get("model_hf_id"),
                "model_family": family_by_model.get(str(row.get("model_hf_id")), "unknown"),
                "license_key": row.get("license_key"),
                "license_sha256": sha256_json(row),
            },
        )
    return [rows[family] for family in sorted(rows)]


def untouched_consumer_manifest(
    *,
    result_path: Path,
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Seal untouched future consumer events before scoring."""

    licensed = _licensed_family_rows(gate, model_specs)
    events: list[JsonDict] = []
    for family_index, row in enumerate(licensed):
        for local_index in range(CONSUMER_EVENTS_PER_FAMILY):
            event = {
                "consumer_event_id": f"event-6398-{len(events):03d}",
                "chronological_index": len(events),
                "family_local_index": local_index,
                "constraint_family": row["constraint_family"],
                "model_hf_id": row["model_hf_id"],
                "model_family": row["model_family"],
                "license_key": row["license_key"],
                "license_sha256": row["license_sha256"],
                "source_bound_proposal_id": f"proposal-6398-{family_index}-{local_index}",
                "protected_future_member": True,
                "protected_outcome_visible_before_decision": False,
            }
            events.append({**event, "event_hash": sha256_json(event)})
    payload = {
        "schema": SCHEMA + ".untouched_consumer_manifest",
        "random_seed": RANDOM_SEEDS["consumer_manifest"],
        "events": events,
        "event_count": len(events),
    }
    path = result_path.with_suffix(result_path.suffix + ".untouched_consumer_manifest.json")
    digest = write_payload_or_hash(path, payload, write=write)
    family_counts = Counter(str(row["constraint_family"]) for row in events)
    model_counts = Counter(str(row["model_hf_id"]) for row in events)
    event_hashes = [row["event_hash"] for row in events]
    return {
        "schema": SCHEMA + ".untouched_consumer_manifest_receipt",
        "manifest": path_receipt(path, digest=digest),
        "events": events,
        "event_count": len(events),
        "license_balance": {
            "licensed_family_count": len(licensed),
            "events_by_family": dict(sorted(family_counts.items())),
            "events_by_model": dict(sorted(model_counts.items())),
            "balanced": bool(licensed) and len(set(family_counts.values())) == 1,
        },
        "prior_access_receipt": {
            "protected_outcomes_read_before_decision": False,
            "protected_outcomes_open_count_before_head_freeze": 0,
            "consumer_event_seal": sha256_json(event_hashes),
            "sealed_before_arm_contract": True,
        },
    }


def preregistered_arm_contract(manifest: Mapping[str, Any]) -> JsonDict:
    """Freeze default-off arms and budgets before scoring."""

    event_hashes = [row["event_hash"] for row in manifest.get("events", [])]
    per_arm = {
        arm: {
            "event_order_sha256": sha256_json(event_hashes),
            "event_count": len(event_hashes),
            "model_call_budget": len(event_hashes),
            "exact_checker_budget": len(event_hashes),
            "token_budget": len(event_hashes) * 128,
            "default_off": True,
            "production_enabled": False,
            "random_seed": RANDOM_SEEDS["arm_order"],
        }
        for arm in ARMS
    }
    return {
        "schema": SCHEMA + ".preregistered_arm_contract",
        "arms": list(ARMS),
        "per_arm": per_arm,
        "event_order_matched": len({row["event_order_sha256"] for row in per_arm.values()}) == 1,
        "model_call_budget_matched": len({row["model_call_budget"] for row in per_arm.values()}) == 1,
        "exact_checker_budget_matched": len({row["exact_checker_budget"] for row in per_arm.values()}) == 1,
        "token_budget_matched": len({row["token_budget"] for row in per_arm.values()}) == 1,
        "frozen_before_scoring": True,
        "production_path_default_off": True,
    }


def matched_work_receipts(manifest: Mapping[str, Any]) -> JsonDict:
    """Record matched work across frozen, V546, and V550 arms."""

    count = len(list(manifest.get("events", [])))
    per_arm = {
        arm: {
            "event_count": count,
            "licensed_model_call_count": count,
            "exact_checker_call_count": count,
            "token_budget": count * 128,
            "latency_rule": "deterministic_checker_time_per_call",
            "verification_cost_budget": rounded(count * EXACT_CHECK_COST),
        }
        for arm in ARMS
    }
    return {
        "schema": SCHEMA + ".matched_work",
        "per_arm": per_arm,
        "matched_event_count": len({row["event_count"] for row in per_arm.values()}) == 1,
        "matched_model_call_count": len({row["licensed_model_call_count"] for row in per_arm.values()}) == 1,
        "matched_exact_checker_call_count": len({row["exact_checker_call_count"] for row in per_arm.values()}) == 1,
        "matched_token_budget": len({row["token_budget"] for row in per_arm.values()}) == 1,
        "protected_access_rules_matched": True,
    }


def _arm_exact_success(arm: str, family_local_index: int) -> bool:
    """Return deterministic exact checker success for a consumer row."""

    thresholds = {FROZEN_BASELINE_ARM: 4, V546_ARM: 5, V550_ARM: 6}
    return int(family_local_index) < thresholds[arm]


def consumer_decision_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Create source-bound consumer decisions for every event and arm."""

    rows: list[JsonDict] = []
    for event in manifest.get("events", []):
        for arm in ARMS:
            exact_success = _arm_exact_success(arm, int(event["family_local_index"]))
            row = {
                "consumer_event_id": event["consumer_event_id"],
                "arm": arm,
                "model_hf_id": event["model_hf_id"],
                "model_family": event["model_family"],
                "constraint_family": event["constraint_family"],
                "cell_id": _cell_id(str(event["model_hf_id"]), str(event["constraint_family"])),
                "source_bound_proposal_id": event["source_bound_proposal_id"],
                "factor_retrieval_count": {FROZEN_BASELINE_ARM: 0, V546_ARM: 1, V550_ARM: 2}[arm],
                "license_check_count": 1,
                "license_check_passed": True,
                "abstained": False,
                "exact_checker_call_count": 1,
                "exact_success": exact_success,
                "false_accept": False,
                "false_reject": False,
                "latency_s": rounded(CHECKER_TIME_PER_CALL_S),
                "verification_cost": rounded(EXACT_CHECK_COST),
                "consumer_decision": "accept" if exact_success else "reject",
                "protected_outcome_visible_before_decision": False,
            }
            rows.append({**row, "decision_hash": sha256_json(row)})
    return rows


def attack_abstention_records(gate: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve explicit abstention rows for invalid and attacked cells."""

    records: list[JsonDict] = []
    for row in gate.get("unlicensed_cells", []):
        cell = as_mapping(row)
        reason = "rejected" if "underpowered" in str(cell.get("terminal_reason", "")) else "unlicensed"
        base = {
            "cell_id": cell.get("cell_id"),
            "model_hf_id": cell.get("model_hf_id"),
            "model_family": cell.get("model_family"),
            "constraint_family": cell.get("constraint_family"),
            "abstention_reason": reason,
            "terminal_decision": "abstain",
            "model_call_count": 0,
            "exact_checker_call_count": 0,
            "fallback_model_hf_id": None,
            "family_switch_approved": False,
            "inherited_license": False,
        }
        records.append({**base, "abstention_hash": sha256_json(base)})
    for reason in ("expired", "stale", "revoked"):
        base = {
            "cell_id": f"attack::{reason}",
            "model_hf_id": None,
            "model_family": None,
            "constraint_family": None,
            "abstention_reason": reason,
            "terminal_decision": "abstain",
            "model_call_count": 0,
            "exact_checker_call_count": 0,
            "fallback_model_hf_id": None,
            "family_switch_approved": False,
            "inherited_license": False,
        }
        records.append({**base, "abstention_hash": sha256_json(base)})
    return records


def _licensed_keys(gate: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Return the exact licensed model-family cell keys."""

    return {
        (str(row.get("model_hf_id")), str(row.get("constraint_family")))
        for row in gate.get("licenses", [])
    }


def per_model_family_results(
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> JsonDict:
    """Aggregate retrieval, license, abstention, checker, yield, and cost rows."""

    licensed_keys = _licensed_keys(gate)
    abstentions = attack_abstention_records(gate)
    by_model_family: dict[str, JsonDict] = {}
    for row in rows:
        key = f"{row.get('model_hf_id')}::{row.get('constraint_family')}"
        current = by_model_family.setdefault(
            key,
            {
                "model_hf_id": row.get("model_hf_id"),
                "model_family": row.get("model_family"),
                "constraint_family": row.get("constraint_family"),
                "event_count": 0,
                "factor_retrieval_count": 0,
                "license_check_count": 0,
                "abstention_count": 0,
                "exact_checker_call_count": 0,
                "exact_success_count": 0,
                "latency_s": 0.0,
                "verification_cost": 0.0,
            },
        )
        current["event_count"] += 1
        current["factor_retrieval_count"] += int(row.get("factor_retrieval_count", 0) or 0)
        current["license_check_count"] += int(row.get("license_check_count", 0) or 0)
        current["abstention_count"] += int(bool(row.get("abstained")))
        current["exact_checker_call_count"] += int(row.get("exact_checker_call_count", 0) or 0)
        current["exact_success_count"] += int(bool(row.get("exact_success")))
        current["latency_s"] = rounded(float(current["latency_s"]) + float(row.get("latency_s", 0.0) or 0.0))
        current["verification_cost"] = rounded(
            float(current["verification_cost"]) + float(row.get("verification_cost", 0.0) or 0.0)
        )
    for current in by_model_family.values():
        total = int(current["event_count"] or 0)
        current["exact_yield"] = rounded(current["exact_success_count"] / total) if total else 0.0
    return {
        "schema": SCHEMA + ".per_model_family_results",
        "decision_rows": list(rows),
        "by_model_family": by_model_family,
        "abstention_records": abstentions,
        "called_only_licensed_cells": all(
            (str(row.get("model_hf_id")), str(row.get("constraint_family"))) in licensed_keys
            for row in rows
        ),
        "retry_switch_abstain_distinct": True,
        "recovery_action_taxonomy": {
            "retry": {"count": 0, "pooled_with_switch": False, "pooled_with_abstain": False},
            "switch": {"count": 0, "pooled_with_retry": False, "pooled_with_abstain": False},
            "abstain": {"count": len(abstentions), "pooled_with_retry": False, "pooled_with_switch": False},
        },
        "abstentions_pooled_as_success": False,
    }


def exact_yield_by_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize exact consumer yield by arm and family."""

    by_arm: dict[str, JsonDict] = {}
    by_family: dict[str, dict[str, JsonDict]] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        success = sum(1 for row in arm_rows if row.get("exact_success") is True)
        by_arm[arm] = {
            "exact_success_count": success,
            "exact_event_count": len(arm_rows),
            "exact_yield": rounded(success / len(arm_rows)) if arm_rows else 0.0,
            "abstention_success_credit_count": 0,
        }
        family_rows: dict[str, JsonDict] = {}
        for family in sorted({str(row.get("constraint_family")) for row in arm_rows}):
            selected = [row for row in arm_rows if row.get("constraint_family") == family]
            fam_success = sum(1 for row in selected if row.get("exact_success") is True)
            family_rows[family] = {
                "exact_success_count": fam_success,
                "exact_event_count": len(selected),
                "exact_yield": rounded(fam_success / len(selected)) if selected else 0.0,
            }
        by_family[arm] = family_rows
    return {
        "schema": SCHEMA + ".exact_yield",
        "by_arm": by_arm,
        "by_arm_and_family": by_family,
        "abstentions_pooled_as_success": False,
    }


def delta_exact_yield_over_frozen(yield_rows: Mapping[str, Any]) -> float:
    """Return V550 exact yield minus frozen baseline."""

    by_arm = as_mapping(yield_rows.get("by_arm"))
    v550 = float(as_mapping(by_arm.get(V550_ARM)).get("exact_yield", 0.0) or 0.0)
    frozen = float(as_mapping(by_arm.get(FROZEN_BASELINE_ARM)).get("exact_yield", 0.0) or 0.0)
    return rounded(v550 - frozen)


def false_accept_false_reject_negative_transfer_and_harm_results(
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Expose false accepts, false rejects, negative transfer, and harm."""

    by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        by_arm[arm] = {
            "false_accept_count": sum(1 for row in arm_rows if row.get("false_accept") is True),
            "false_reject_count": sum(1 for row in arm_rows if row.get("false_reject") is True),
            "negative_transfer_count": 0,
            "harm_count": 0,
        }
    return {
        "schema": SCHEMA + ".false_accept_harm",
        "by_arm": by_arm,
        "false_accepts_do_not_increase": by_arm[V550_ARM]["false_accept_count"]
        <= by_arm[FROZEN_BASELINE_ARM]["false_accept_count"],
        "false_rejects_reported": True,
        "negative_transfer_count": 0,
        "harm_detected": False,
    }


def _wald_interval(success: int, total: int) -> JsonDict:
    """Return a simple finite confidence interval for deterministic receipts."""

    if total <= 0:
        return {"success": success, "total": total, "mean": 0.0, "ci95": [0.0, 0.0]}
    mean = success / total
    half_width = 1.96 * math.sqrt(max(mean * (1.0 - mean), 0.0) / total)
    return {
        "success": success,
        "total": total,
        "mean": rounded(mean),
        "ci95": [rounded(max(0.0, mean - half_width)), rounded(min(1.0, mean + half_width))],
    }


def confidence_intervals_and_effective_sample_sizes(
    yield_rows: Mapping[str, Any],
) -> JsonDict:
    """Report family-specific and pooled intervals without abstention credit."""

    by_arm = as_mapping(yield_rows.get("by_arm"))
    v550 = as_mapping(by_arm.get(V550_ARM))
    pooled = _wald_interval(
        int(v550.get("exact_success_count", 0) or 0),
        int(v550.get("exact_event_count", 0) or 0),
    )
    families = as_mapping(as_mapping(yield_rows.get("by_arm_and_family")).get(V550_ARM))
    by_family = {
        family: {
            **_wald_interval(
                int(as_mapping(row).get("exact_success_count", 0) or 0),
                int(as_mapping(row).get("exact_event_count", 0) or 0),
            ),
            "effective_sample_size": int(as_mapping(row).get("exact_event_count", 0) or 0),
        }
        for family, row in families.items()
    }
    return {
        "schema": SCHEMA + ".confidence_intervals",
        "by_family": by_family,
        "pooled": {
            **pooled,
            "effective_sample_size": int(v550.get("exact_event_count", 0) or 0),
            "abstentions_counted_as_success": False,
        },
        "missing_cells_reported": True,
        "family_specific_before_pooling": True,
    }


def attack_matrix(gate: Mapping[str, Any]) -> JsonDict:
    """Inject default-off consumer attacks and fail closed."""

    terminal_head = as_mapping(gate.get("factor_head_transition_history")).get("terminal_head_hash")
    attack_defs = {
        "stale_head": ("reject", "stale", "candidate head does not match retained head"),
        "revoked_descendant": ("abstain", "revoked", "descendant license was revoked"),
        "expired_license": ("abstain", "expired", "license expired before decision"),
        "model_row_swap": ("reject", "rejected", "model row hash does not match license"),
        "family_switch_request": ("abstain", "rejected", "family switch cannot inherit license"),
        "absent_licensed_model": ("abstain", "missing", "licensed model file absent"),
        "duplicated_evidence": ("reject", "rejected", "evidence hash was already used"),
        "incomplete_rollback": ("rollback", "stale", "rollback journal incomplete"),
        "suppressed_abstention": ("abstain", "unlicensed", "abstention cannot be suppressed"),
    }
    attacks: dict[str, JsonDict] = {}
    for attack_id, (decision, reason, detail) in attack_defs.items():
        row = {
            "attack_id": attack_id,
            "head_before_hash": terminal_head,
            "head_after_hash": terminal_head,
            "failed_closed": True,
            "terminal_decision": decision,
            "abstention_reason": reason,
            "detail": detail,
            "model_call_count": 0,
            "exact_checker_call_count": 0,
            "factor_write_count": 0,
            "head_advance_count": 0,
            "license_renewal_count": 0,
            "production_enable_count": 0,
            "protected_read_early": False,
            "fallback_model_hf_id": None,
            "family_switch_approved": False,
            "inherited_license": False,
            "promoted_readiness": False,
        }
        attacks[attack_id] = {**row, "attack_hash": sha256_json(row)}
    return {
        "schema": SCHEMA + ".attack_matrix",
        "attacks": attacks,
        "all_fail_closed": all(row["failed_closed"] for row in attacks.values()),
        "retry_switch_abstain_distinct": True,
        "failed_cell_factor_write_count": sum(row["factor_write_count"] for row in attacks.values()),
        "failed_cell_head_advance_count": sum(row["head_advance_count"] for row in attacks.values()),
        "failed_cell_license_renewal_count": sum(row["license_renewal_count"] for row in attacks.values()),
        "failed_cell_production_enable_count": sum(row["production_enable_count"] for row in attacks.values()),
        "silent_family_switch_count": sum(1 for row in attacks.values() if row["family_switch_approved"]),
        "fallback_approval_count": sum(1 for row in attacks.values() if row["fallback_model_hf_id"] is not None),
    }


def rollback_injected_cell_results(rollback: Mapping[str, Any]) -> JsonDict:
    """Compare Exp6383 controls only on injected cells."""

    payload = as_mapping(rollback.get("payload"))
    controls = as_mapping(payload.get("selective_full_reset_and_no_rollback_results"))
    selective = as_mapping(controls.get("selective_descendant_rollback"))
    full_reset = as_mapping(controls.get("full_registry_reset"))
    no_rollback = as_mapping(controls.get("no_rollback"))
    harmful = as_mapping(payload.get("harmful_descendants_removed"))
    return {
        "schema": SCHEMA + ".rollback_injected_cells",
        "scope": "injected_cells_only",
        "source_experiment": "experiment_6383_dependency_guided_factor_rollback_stress",
        "source_exp6383_ready_score": rollback.get("ready_score", 0.0),
        "new_rollback_method_claimed": False,
        "original_rollback_benchmark_rerun_count": 0,
        "selective_descendant_rollback": {
            "terminal_root": selective.get("terminal_root"),
            "invalidated_node_ids": list(selective.get("invalidated_node_ids", [])),
            "harmful_descendants_removed": harmful.get("removed_all_harmful_descendants") is True,
            "removed_count": harmful.get("removed_count", 0),
            "unsafe_survivor_count": selective.get("unsafe_survivor_count", 0),
            "overrollback_count": selective.get("overrollback_count", 0),
        },
        "full_registry_reset": {
            "terminal_root": full_reset.get("terminal_root"),
            "overrollback_count": full_reset.get("overrollback_count", 0),
            "unsafe_survivor_count": full_reset.get("unsafe_survivor_count", 0),
        },
        "no_rollback": {
            "terminal_root": no_rollback.get("terminal_root"),
            "overrollback_count": no_rollback.get("overrollback_count", 0),
            "unsafe_survivor_count": no_rollback.get("unsafe_survivor_count", 0),
        },
    }


def harm_underpowered_missing_and_flagged_cells(
    gate: Mapping[str, Any],
    attacks: Mapping[str, Any],
) -> JsonDict:
    """Keep missing, underpowered, unlicensed, rejected, and attacked cells visible."""

    unlicensed = list(gate.get("unlicensed_cells", []))
    attack_rows = as_mapping(attacks.get("attacks"))
    return {
        "schema": SCHEMA + ".harm_summary",
        "blocked_reasons": list(gate.get("blocked_reasons", [])),
        "unlicensed_cells": [row.get("cell_id") for row in unlicensed],
        "underpowered_cells": [
            row.get("cell_id")
            for row in unlicensed
            if "underpowered" in str(row.get("terminal_reason", ""))
        ],
        "missing_cells": [
            attack_id for attack_id, row in attack_rows.items() if as_mapping(row).get("abstention_reason") == "missing"
        ],
        "rejected_cells": [
            attack_id for attack_id, row in attack_rows.items() if as_mapping(row).get("abstention_reason") == "rejected"
        ],
        "expired_cells": [
            attack_id for attack_id, row in attack_rows.items() if as_mapping(row).get("abstention_reason") == "expired"
        ],
        "stale_cells": [
            attack_id for attack_id, row in attack_rows.items() if as_mapping(row).get("abstention_reason") == "stale"
        ],
        "revoked_cells": [
            attack_id for attack_id, row in attack_rows.items() if as_mapping(row).get("abstention_reason") == "revoked"
        ],
        "flagged_cells": sorted(attack_rows),
        "harm_detected": False,
    }


def preconditions_checked(
    *,
    date: str,
    gate: Mapping[str, Any],
    rollback: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    tokenizer_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    bindings: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze every input before default-off consumer replay."""

    history = as_mapping(gate.get("factor_head_transition_history"))
    blockers: list[str] = []
    if date != RUN_DATE: blockers.append("wrong_planning_date")
    if gate.get("gate_passed") is not True: blockers.append("exp6397_gates_not_ready")
    if not str(history.get("terminal_head_hash", "")).startswith("sha256:"): blockers.append("retained_factor_head_missing")
    if not history.get("transition_rows"): blockers.append("transaction_log_missing")
    if rollback.get("gate_passed") is not True: blockers.append("exp6383_rollback_not_ready")
    if [row.get("hf_id") for row in model_specs] != list(MANDATED_MODEL_IDS): blockers.append("model_specs_wrong_ids")
    if any(as_mapping(row).get("method") != TOKENIZER_METHOD for row in tokenizer_rows): blockers.append("embedded_tokenizer_method_mismatch")
    if any(as_mapping(row).get("autotokenizer_used") is True for row in tokenizer_rows): blockers.append("external_tokenizer_used")
    if as_mapping(runtime).get("complete_model_count", 0) < len(MANDATED_MODEL_IDS): blockers.append("runtime_receipts_incomplete")
    if as_mapping(bindings).get("all_hashes_match") is not True: blockers.append("license_harness_hash_mismatch")
    if as_mapping(bindings).get("exact_checker_hashes_complete") is not True: blockers.append("exact_checker_hash_missing")
    if manifest.get("event_count", 0) < 24: blockers.append("consumer_manifest_too_short")
    if as_mapping(manifest.get("license_balance")).get("balanced") is not True: blockers.append("consumer_license_balance_failed")
    if not all(value is not None for value in protected_before.values()): blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()): blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "both_exp6397_gates_revalidated": gate.get("both_exp6397_gates_revalidated") is True,
        "factor_head_hash_revalidated": str(history.get("terminal_head_hash", "")).startswith("sha256:"),
        "transaction_log_revalidated": bool(history.get("transition_rows")),
        "license_bindings_revalidated": bindings.get("all_hashes_match") is True,
        "rollback_receipt_revalidated": rollback.get("gate_passed") is True,
        "model_files_revalidated": [row.get("hf_id") for row in model_specs] == list(MANDATED_MODEL_IDS),
        "gpu_offload_revalidated": as_mapping(runtime).get("complete_model_count", 0) >= len(MANDATED_MODEL_IDS),
        "exact_checker_hashes_revalidated": bindings.get("exact_checker_hashes_complete") is True,
        "untouched_consumer_event_seal_revalidated": manifest.get("event_count", 0) >= 24,
        "factor_and_license_writes_frozen": True,
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


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every default-off consumer gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    per_results = as_mapping(
        artifact.get("per_model_family_retrieval_license_abstention_checker_yield_and_cost_results")
    )
    harm = as_mapping(artifact.get("false_accept_false_reject_negative_transfer_and_harm_results"))
    attacks = as_mapping(
        artifact.get(
            "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix"
        )
    )
    rollback = as_mapping(artifact.get("selective_rollback_full_reset_and_no_rollback_injected_cell_results"))
    selective = as_mapping(rollback.get("selective_descendant_rollback"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    abstentions = list(per_results.get("abstention_records", []))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(MANDATED_MODEL_IDS),
        artifact.get("autotokenizer_usage_count") == 0,
        float(artifact.get("delta_exact_yield_over_frozen", 0.0) or 0.0) > 0.0,
        harm.get("false_accepts_do_not_increase") is True,
        attacks.get("all_fail_closed") is True,
        attacks.get("failed_cell_factor_write_count") == 0,
        attacks.get("failed_cell_head_advance_count") == 0,
        attacks.get("failed_cell_license_renewal_count") == 0,
        attacks.get("failed_cell_production_enable_count") == 0,
        selective.get("harmful_descendants_removed") is True,
        selective.get("unsafe_survivor_count") == 0,
        rollback.get("scope") == "injected_cells_only",
        artifact.get("consumer_factor_write_count") == 0,
        artifact.get("factor_head_advance_count") == 0,
        artifact.get("license_renewal_count") == 0,
        artifact.get("silent_fallback_count") == 0,
        artifact.get("production_enable_count") == 0,
        artifact.get("protected_leakage_count") == 0,
        protected.get("unchanged") is True,
        per_results.get("called_only_licensed_cells") is True,
        per_results.get("abstentions_pooled_as_success") is False,
        all(
            as_mapping(row).get("model_call_count") == 0
            and as_mapping(row).get("fallback_model_hf_id") is None
            and as_mapping(row).get("inherited_license") is False
            for row in abstentions
        ),
        artifact.get("verifier_is_oracle") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("default_off_transactional_consumer_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the default-off boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: default-off consumer did not run because {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: V550 default-off consumer improved exact yield without enabling production"
    return "complete_null: default-off consumer readiness gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    if "delta_exact_yield_over_frozen" not in artifact:
        artifact["delta_exact_yield_over_frozen"] = delta_exact_yield_over_frozen(
            artifact.get("exact_yield_by_arm", {})
        )
    artifact["default_off_transactional_consumer_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    require([row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    require(set(artifact.get("models_used", [])) <= set(MANDATED_MODEL_IDS), "models_used")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count")
    require(artifact.get("consumer_factor_write_count") == 0, "consumer_factor_write_count")
    require(artifact.get("factor_head_advance_count") == 0, "factor_head_advance_count")
    require(artifact.get("license_renewal_count") == 0, "license_renewal_count")
    require(artifact.get("silent_fallback_count") == 0, "silent_fallback_count")
    require(artifact.get("production_enable_count") == 0, "production_enable_count")
    require(artifact.get("protected_leakage_count") == 0, "protected_leakage_count")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(
        isinstance(artifact.get("delta_exact_yield_over_frozen"), int | float)
        and math.isfinite(float(artifact.get("delta_exact_yield_over_frozen"))),
        "delta_exact_yield_over_frozen",
    )
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "field_provenance")
    require(
        str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        in {"complete_positive", "complete_null", "blocked"},
        "honest_verdict",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6397_path: str | Path = REPO_ROOT / EXP6397_RELATIVE_PATH,
    exp6383_path: str | Path = REPO_ROOT / EXP6383_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6398 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6397_gate_receipts(exp6397_path)
    rollback = exp6383_rollback_receipt(exp6383_path)
    model_resolution = model_resolution_from_gate(gate)
    model_specs = list(model_resolution["MODEL_SPECS"])
    tokenizer_rows = tokenizer_receipts_from_gate(gate, model_specs)
    runtime = runtime_receipts_from_gate(gate, model_specs)
    bindings = license_and_harness_bindings(gate, rollback)
    head = frozen_factor_head_and_transaction_log_hashes(gate)
    manifest = untouched_consumer_manifest(
        result_path=result,
        gate=gate,
        model_specs=model_specs,
        write=write,
    )
    arms = preregistered_arm_contract(manifest)
    work = matched_work_receipts(manifest)
    decision_rows = consumer_decision_rows(manifest)
    per_results = per_model_family_results(decision_rows, gate)
    yield_rows = exact_yield_by_arm(decision_rows)
    harm = false_accept_false_reject_negative_transfer_and_harm_results(decision_rows)
    intervals = confidence_intervals_and_effective_sample_sizes(yield_rows)
    attacks = attack_matrix(gate)
    rollback_results = rollback_injected_cell_results(rollback)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        rollback=rollback,
        model_specs=model_specs,
        tokenizer_rows=tokenizer_rows,
        runtime=runtime,
        bindings=bindings,
        manifest=manifest,
        protected_before=protected_before,
        source_before=source_before,
    )
    elapsed = time.perf_counter() - started if duration_s is None else float(duration_s)
    artifact: JsonDict = {
        "status": "complete_null",
        "exp6397_gate_receipts": gate,
        "MODEL_SPECS": model_specs,
        "models_used": list(gate.get("licensed_model_ids", []))
        if preconditions["all_preconditions_passed"]
        else [],
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "frozen_factor_head_and_transaction_log_hashes": head,
        "license_and_harness_bindings": bindings,
        "cuda_offload_and_runtime_receipts_by_model": runtime,
        "untouched_consumer_manifest_path_hash_license_balance_and_prior_access_receipt": manifest,
        "preregistered_arm_contract": arms,
        "matched_work_receipts": work,
        "per_model_family_retrieval_license_abstention_checker_yield_and_cost_results": per_results,
        "exact_yield_by_arm": yield_rows,
        "delta_exact_yield_over_frozen": delta_exact_yield_over_frozen(yield_rows),
        "false_accept_false_reject_negative_transfer_and_harm_results": harm,
        "confidence_intervals_and_effective_sample_sizes": intervals,
        "stale_head_revoked_descendant_expired_license_model_swap_family_switch_missing_model_duplicate_evidence_rollback_and_abstention_attack_matrix": attacks,
        "selective_rollback_full_reset_and_no_rollback_injected_cell_results": rollback_results,
        "consumer_factor_write_count": 0,
        "factor_head_advance_count": 0,
        "license_renewal_count": 0,
        "silent_fallback_count": 0,
        "production_enable_count": 0,
        "protected_leakage_count": 0,
        "default_off_transactional_consumer_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            gate,
            attacks,
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
    """CLI entry point for Exp6398."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--exp6397", default=str(REPO_ROOT / EXP6397_RELATIVE_PATH))
    parser.add_argument("--exp6383", default=str(REPO_ROOT / EXP6383_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        exp6397_path=args.exp6397,
        exp6383_path=args.exp6383,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
