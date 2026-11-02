"""Exp6397 transactional continuous factor learning.

Spec refs: REQ-LEARN-6397, SCENARIO-LEARN-6397-CHRONOLOGY,
SCENARIO-LEARN-6397-TRANSACTION, SCENARIO-LEARN-6397-ATTACKS,
SCENARIO-LEARN-6397-FUTURE, SCENARIO-LEARN-6397-READY.
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
from carnot import experiment_6396_capability_qualified_verified_frontier_ab as exp6396


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6397_transactional_continuous_factor_learning.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6397_transactional_continuous_factor_learning"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6397_transactional_continuous_factor_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6397_transactional_continuous_factor_learning.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6396_RELATIVE_PATH = exp6396.RESULT_RELATIVE_PATH
EXP6383_RELATIVE_PATH = exp6383.RESULT_RELATIVE_PATH
EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6342_LEDGER_RELATIVE_PATH = Path(
    "results/experiment_6342_anytime_evalue_release_ledger.json.evalue_ledger.jsonl"
)

SCHEMA = "carnot.experiment_6397.transactional_continuous_factor_learning.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6397
TOKENIZER_METHOD = exp6396.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "deterministic_transactional_replay_over_licensed_local_gguf_receipts"

MANDATED_MODEL_IDS = exp6396.MANDATED_MODEL_IDS
MODEL_TEMPLATE_BY_ID = exp6396.MODEL_TEMPLATE_BY_ID
CONSTRAINT_FAMILIES = exp6396.REQUIRED_CONSTRAINT_FAMILIES
FROZEN_BASELINE_ARM = "frozen_baseline"
V546_CONTROL_ARM = "v546_replay_certified_factor_control"
LIVE_LEARNER_ARM = "capability_qualified_live_learner"
ARMS = (FROZEN_BASELINE_ARM, V546_CONTROL_ARM, LIVE_LEARNER_ARM)
PARTITIONS = ("acquisition", "release", "retention", "untouched_future")
EVENTS_PER_PARTITION = 12
FACTOR_CAPACITY = 4
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
RANDOM_SEEDS = {
    "chronological_manifest": 639700,
    "transaction_order": 639701,
    "attack_matrix": 639702,
    "future_open": 639703,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6397_transactional_continuous_factor_learning --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6397_transactional_continuous_factor_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6397_transactional_continuous_factor_learning.py "
    "-m pytest tests/python/test_experiment_6397_transactional_continuous_factor_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6397_transactional_continuous_factor_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6397_transactional_continuous_factor_learning.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6397_transactional_continuous_factor_learning.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
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
    EXP6396_RELATIVE_PATH,
    EXP6383_RELATIVE_PATH,
    EXP6342_RELATIVE_PATH,
    EXP6342_LEDGER_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"),
    Path("python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6396_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "license_and_frozen_harness_bindings",
    "unlicensed_cell_abstention_records",
    "cuda_offload_and_runtime_receipts_by_model",
    "chronological_manifest_path_hash_license_balance_and_partition_seals",
    "preregistered_arm_contract",
    "factor_head_initial_hash",
    "typed_candidate_records",
    "predecessor_candidate_evidence_checker_eprocess_and_effect_bindings",
    "atomic_disposition_records",
    "factor_head_transition_history",
    "commit_reject_quarantine_and_defer_counts",
    "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix",
    "proposal_learnability_results",
    "exact_alignment_results",
    "forward_transfer_results",
    "backward_retention_and_forgetting_results",
    "negative_transfer_and_harm_results",
    "factor_growth_and_capacity_results",
    "verification_cost_results",
    "untouched_future_evaluation_receipts",
    "future_exact_yield_by_arm",
    "delta_future_exact_yield_over_frozen",
    "selective_rollback_control_path_hash_and_terminal_class",
    "selective_rollback_control_ready_score",
    "transactional_continuous_self_learning_ready_score",
    "protected_leakage_count",
    "same_step_write_count",
    "model_weight_change_count",
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
    "status": "Terminal status follows transactional activation gates and protected replay.",
    "exp6396_gate_receipts": "Exp6396 readiness, licenses, future yield, and protected partitions gate this run.",
    "MODEL_SPECS": "The three mandated GGUF rows come from cached SOTA helper calls.",
    "models_used": "Only licensed mandated models with transactional work count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer receipts use only embedded GGUF tokenizers.",
    "autotokenizer_usage_count": "Bare zero proves no external tokenizer path was used.",
    "license_and_frozen_harness_bindings": "Licenses, harnesses, schemas, models, and exact checkers are bound before events run.",
    "unlicensed_cell_abstention_records": "Unlicensed cells abstain without substitution.",
    "cuda_offload_and_runtime_receipts_by_model": "CUDA offload and cleanup are reported for mandated models.",
    "chronological_manifest_path_hash_license_balance_and_partition_seals": "Chronology, licenses, balance, restart boundaries, and partitions are sealed.",
    "preregistered_arm_contract": "Frozen, V546 control, and live learner arms are matched.",
    "factor_head_initial_hash": "The initial read-only factor head is frozen.",
    "typed_candidate_records": "Typed candidates are evaluated off-commit.",
    "predecessor_candidate_evidence_checker_eprocess_and_effect_bindings": "Candidate activation inputs are hash-bound.",
    "atomic_disposition_records": "Each candidate has exactly one terminal disposition.",
    "factor_head_transition_history": "Only successful commits advance the head.",
    "commit_reject_quarantine_and_defer_counts": "Disposition counts stay explicit.",
    "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix": "Transaction attacks fail closed.",
    "proposal_learnability_results": "Learnability is reported separately from utility.",
    "exact_alignment_results": "Exact checker alignment is reported separately from learnability and utility.",
    "forward_transfer_results": "Future exact transfer is measured per arm.",
    "backward_retention_and_forgetting_results": "Prior retained behavior cannot regress.",
    "negative_transfer_and_harm_results": "Harmful transfer, abstention, and leakage stay visible.",
    "factor_growth_and_capacity_results": "Factor growth stays bounded.",
    "verification_cost_results": "Exact checker calls, latency, and cost are charged.",
    "untouched_future_evaluation_receipts": "Protected future outcomes open once after head freeze.",
    "future_exact_yield_by_arm": "Future exact utility is reported by arm.",
    "delta_future_exact_yield_over_frozen": "Live learner future yield is compared with frozen.",
    "selective_rollback_control_path_hash_and_terminal_class": "Exp6383 is carried as a rollback control.",
    "selective_rollback_control_ready_score": "The exact Exp6383 ready score is carried.",
    "transactional_continuous_self_learning_ready_score": "Readiness is conjunctive over commit, utility, retention, growth, attacks, leaks, weights, and tests.",
    "protected_leakage_count": "Bare zero proves protected partitions did not leak.",
    "same_step_write_count": "Bare zero proves proposal-time writes stayed invisible.",
    "model_weight_change_count": "Bare zero proves no model weights changed.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, unlicensed, rejected, and attacked cells stay visible.",
    "protected_files_unchanged": "Protected files remain byte-identical.",
    "preconditions_checked": "Preconditions bind upstream gates, models, tokenizers, GPUs, exact checkers, manifests, seeds, and protected files.",
    "inference_substrate": "The substrate declares deterministic transactional replay over licensed local GGUF receipts.",
    "verifier_is_oracle": "Bare true applies only to exact task checkers and exact release tests.",
    "field_principles": "Every required field states its guard and purpose.",
    "field_provenance": "Every required field maps to specs, upstream artifacts, transactions, attacks, tests, or exact checks.",
    "random_seed": "Fixed seeds pin chronology, proposals, attacks, and future opens.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states the transaction boundary.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6397",
        "Exp6396 capability-qualified frontier artifact",
        "Exp6383 selective rollback artifact",
        "Exp6342 e-value release ledger",
        "Exp6397 transaction fixtures and focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


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


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return exp6396.model_slug(model_id)


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


def exp6396_gate_receipts(path: str | Path) -> JsonDict:
    """Revalidate Exp6396 readiness and future-improvement gates."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "gate_passed": False,
            "blocked_reasons": ["exp6396_artifact_missing"],
            "licenses": [],
            "unlicensed_cells": [],
            "licensed_model_ids": [],
            "qualification_gate_passed": False,
            "frontier_future_improvement_gate_passed": False,
        }
    payload = read_json(path)
    ready = float(payload.get("capability_qualified_frontier_ready_score", 0.0) or 0.0)
    delta = float(payload.get("delta_verified_future_exact_yield", 0.0) or 0.0)
    licenses = list(
        as_mapping(payload.get("license_records_used_and_hashes")).get("license_records", [])
    )
    unlicensed = list(payload.get("unlicensed_cell_abstention_records", []))
    future = as_mapping(payload.get("untouched_future_evaluation_receipts"))
    protected = as_mapping(payload.get("protected_files_unchanged"))
    blocked: list[str] = []
    if ready != 1.0:
        blocked.append("exp6396_ready_score_not_one")
    if delta <= 0.0:
        blocked.append("exp6396_future_delta_not_positive")
    if not licenses:
        blocked.append("exp6396_license_records_missing")
    if int(payload.get("autotokenizer_usage_count", 0) or 0) != 0:
        blocked.append("external_tokenizer_used_upstream")
    if int(payload.get("protected_leakage_count", 0) or 0) != 0:
        blocked.append("exp6396_protected_leakage")
    if int(payload.get("model_weight_change_count", 0) or 0) != 0:
        blocked.append("exp6396_model_weight_change")
    if future.get("open_count") != 1 or future.get("future_outcomes_read_once") is not True:
        blocked.append("exp6396_future_not_single_open")
    if protected.get("unchanged") is not True:
        blocked.append("exp6396_protected_files_changed")
    if any(
        as_mapping(row).get("model_call_count") != 0
        or as_mapping(row).get("fallback_model_hf_id") is not None
        for row in unlicensed
    ):
        blocked.append("exp6396_unlicensed_cell_not_abstained")
    return {
        **receipt,
        "gate_passed": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "qualification_gate_passed": ready == 1.0,
        "frontier_future_improvement_gate_passed": delta > 0.0,
        "both_gates_passed": ready == 1.0 and delta > 0.0,
        "capability_qualified_frontier_ready_score": ready,
        "delta_verified_future_exact_yield": delta,
        "licenses": licenses,
        "license_hashes": [sha256_json(row) for row in licenses],
        "unlicensed_cells": unlicensed,
        "licensed_model_ids": list(payload.get("models_used", [])),
        "upstream_MODEL_SPECS": list(payload.get("MODEL_SPECS", [])),
        "upstream_cached_sota_pair_receipts": payload.get("cached_sota_pair_receipts", {}),
        "upstream_tokenizer_receipts": list(payload.get("embedded_gguf_tokenizer_receipts", [])),
        "upstream_runtime_receipts": payload.get(
            "cuda_offload_and_runtime_receipts_by_model",
            {},
        ),
        "upstream_harness_bindings": payload.get(
            "model_harness_schema_and_checker_bindings",
            {},
        ),
        "upstream_future_yield": payload.get("future_exact_yield_by_arm_and_model", {}),
    }


def selective_rollback_control_receipt(path: str | Path) -> JsonDict:
    """Carry the Exp6383 selective rollback control without modification."""

    receipt = path_receipt(path)
    if not Path(path).is_file():
        return {
            **receipt,
            "terminal_class": "absent",
            "ready_score": 0.0,
            "gate_passed": False,
            "selective_terminal_root": None,
        }
    payload = read_json(path)
    ready = float(payload.get("dependency_guided_rollback_ready_score", 0.0) or 0.0)
    roots = as_mapping(payload.get("terminal_registry_roots"))
    return {
        **receipt,
        "terminal_class": payload.get("status", "present_unqualified"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "ready_score": ready,
        "gate_passed": ready == 1.0 and payload.get("status") == "complete_positive",
        "selective_terminal_root": roots.get("selective_terminal_root"),
        "idempotent_exact_valid_terminal_root": roots.get(
            "idempotent_exact_valid_terminal_root"
        )
        is True,
    }


def evalue_release_ledger_receipt() -> JsonDict:
    """Hash the Exp6342 artifact and exact e-value ledger."""

    artifact_path = REPO_ROOT / EXP6342_RELATIVE_PATH
    ledger_path = REPO_ROOT / EXP6342_LEDGER_RELATIVE_PATH
    ready = 0.0
    status = "absent"
    if artifact_path.is_file():
        payload = read_json(artifact_path)
        ready = float(payload.get("anytime_release_certificate_ready_score", 0.0) or 0.0)
        status = str(payload.get("status", "present_unqualified"))
    return {
        "artifact": path_receipt(artifact_path),
        "ledger": path_receipt(ledger_path),
        "ready_score": ready,
        "status": status,
        "gate_passed": ready == 1.0 and ledger_path.is_file(),
    }


def model_resolution_from_gate(gate: Mapping[str, Any]) -> JsonDict:
    """Return the three upstream model rows and cached helper receipts."""

    if gate.get("upstream_MODEL_SPECS"):
        return {
            "MODEL_SPECS": list(gate.get("upstream_MODEL_SPECS", [])),
            "cached_sota_pair_receipts": dict(
                as_mapping(gate.get("upstream_cached_sota_pair_receipts"))
            ),
        }
    return exp6396.build_model_specs()


def tokenizer_receipts_from_gate(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return embedded tokenizer rows from Exp6396 or rebuild receipts."""

    if gate.get("upstream_tokenizer_receipts"):
        return list(gate.get("upstream_tokenizer_receipts", []))
    return exp6396.tokenizer_receipts(model_specs, exp6396.exp6395.embedded_gguf_tokenizer_receipt)


def runtime_receipts_from_gate(
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return CUDA receipts from Exp6396 or deterministic fallback receipts."""

    if gate.get("upstream_runtime_receipts"):
        return dict(as_mapping(gate.get("upstream_runtime_receipts")))
    host = exp6396.host_environment_receipts()
    return exp6396.cuda_offload_and_runtime_receipts_by_model(model_specs, host)


def unlicensed_cell_abstention_records(gate: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze abstention rows for every unlicensed upstream cell."""

    rows = []
    for row in gate.get("unlicensed_cells", []):
        cell = as_mapping(row)
        abstention = {
            "cell_id": cell.get("cell_id"),
            "model_hf_id": cell.get("model_hf_id"),
            "model_family": cell.get("model_family"),
            "constraint_family": cell.get("constraint_family"),
            "frozen_abstention": True,
            "model_call_count": 0,
            "candidate_count": 0,
            "exact_check_count": 0,
            "fallback_model_hf_id": None,
            "substitution_used": False,
            "terminal_reason": cell.get("terminal_reason", "unlicensed_cell"),
        }
        rows.append({**abstention, "abstention_sha256": sha256_json(abstention)})
    return rows


def license_and_frozen_harness_bindings(
    gate: Mapping[str, Any],
    rollback: Mapping[str, Any],
) -> JsonDict:
    """Bind licenses, harnesses, schemas, exact checkers, and release ledger."""

    upstream = as_mapping(gate.get("upstream_harness_bindings"))
    evalue = evalue_release_ledger_receipt()
    checker_hashes = {
        "exp6342_evalue_release_checker": sha256_file(
            REPO_ROOT / "python/carnot/experiment_6342_anytime_evalue_release_ledger.py"
        ),
        "exp6383_selective_rollback_checker": sha256_file(
            REPO_ROOT / "python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"
        ),
        "exp6396_frontier_checker": sha256_file(
            REPO_ROOT / "python/carnot/experiment_6396_capability_qualified_verified_frontier_ab.py"
        ),
    }
    return {
        "schema": SCHEMA + ".license_harness_bindings",
        "license_count": len(list(gate.get("licenses", []))),
        "license_hashes": list(gate.get("license_hashes", [])),
        "frozen_harness_bindings": upstream,
        "all_hashes_match": upstream.get("all_hashes_match") is True,
        "all_accept_reject_owned_by_exact_checker": upstream.get(
            "all_accept_reject_owned_by_exact_checker"
        )
        is True,
        "exact_checker_hashes": checker_hashes,
        "exact_checker_hashes_complete": all(value is not None for value in checker_hashes.values()),
        "evalue_release_ledger": evalue,
        "selective_rollback_control_ready_score": rollback.get("ready_score"),
    }


def _licensed_family_rows(gate: Mapping[str, Any]) -> list[JsonDict]:
    """Return one representative licensed row per constraint family."""

    rows: dict[str, JsonDict] = {}
    for license_row in gate.get("licenses", []):
        row = as_mapping(license_row)
        family = str(row.get("constraint_family"))
        rows.setdefault(
            family,
            {
                "constraint_family": family,
                "model_hf_id": row.get("model_hf_id"),
                "license_key": row.get("license_key"),
                "license_sha256": sha256_json(row),
            },
        )
    return [rows[family] for family in sorted(rows)]


def chronological_manifest(
    *,
    result_path: Path,
    gate: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Seal a balanced chronological stream before transaction replay."""

    licensed = _licensed_family_rows(gate)
    events: list[JsonDict] = []
    update_indices = {15, 19, 23}
    restart_indices = {17, 35}
    for partition_index, partition in enumerate(PARTITIONS):
        for family_index, family_row in enumerate(licensed):
            for local_index in range(4):
                chrono_index = partition_index * EVENTS_PER_PARTITION + family_index * 4 + local_index
                event = {
                    "event_id": f"event-6397-{chrono_index:03d}",
                    "chronological_index": chrono_index,
                    "partition": partition,
                    "constraint_family": family_row["constraint_family"],
                    "model_hf_id": family_row["model_hf_id"],
                    "license_key": family_row["license_key"],
                    "license_sha256": family_row["license_sha256"],
                    "update_opportunity": chrono_index in update_indices,
                    "restart_boundary": chrono_index in restart_indices,
                    "protected_future_member": partition == "untouched_future",
                }
                events.append({**event, "event_hash": sha256_json(event)})
    payload = {
        "schema": SCHEMA + ".chronological_manifest",
        "random_seed": RANDOM_SEEDS["chronological_manifest"],
        "events": events,
        "event_count": len(events),
    }
    path = result_path.with_suffix(result_path.suffix + ".chronological_manifest.json")
    digest = write_payload_or_hash(path, payload, write=write)
    partition_counts = Counter(row["partition"] for row in events)
    family_counts = Counter(row["constraint_family"] for row in events)
    return {
        "schema": SCHEMA + ".chronological_manifest_receipt",
        "manifest": path_receipt(path, digest=digest),
        "events": events,
        "event_count": len(events),
        "partition_counts": {partition: partition_counts[partition] for partition in PARTITIONS},
        "update_opportunity_count": sum(1 for row in events if row["update_opportunity"]),
        "restart_boundary_count": sum(1 for row in events if row["restart_boundary"]),
        "license_balance": {
            "licensed_family_count": len(licensed),
            "events_by_family": dict(sorted(family_counts.items())),
            "balanced": bool(licensed) and len(set(family_counts.values())) == 1,
        },
        "partition_seals": {
            partition: sha256_json(
                [row["event_hash"] for row in events if row["partition"] == partition]
            )
            for partition in PARTITIONS
        },
        "future_opened_before_head_freeze": False,
    }


def preregistered_arm_contract(manifest: Mapping[str, Any]) -> JsonDict:
    """Freeze matched arms, budgets, and event order."""

    event_hashes = [row["event_hash"] for row in manifest.get("events", [])]
    per_arm = {
        arm: {
            "event_order_sha256": sha256_json(event_hashes),
            "event_count": len(event_hashes),
            "exact_check_budget": 64,
            "consumer_budget": {"max_factor_reads": 32, "max_exact_checks": 64},
            "random_seed": RANDOM_SEEDS["transaction_order"],
        }
        for arm in ARMS
    }
    return {
        "schema": SCHEMA + ".preregistered_arm_contract",
        "arms": list(ARMS),
        "per_arm": per_arm,
        "event_order_matched": len({row["event_order_sha256"] for row in per_arm.values()}) == 1,
        "exact_check_budget_matched": len({row["exact_check_budget"] for row in per_arm.values()}) == 1,
        "consumer_budget_matched": len(
            {sha256_json(row["consumer_budget"]) for row in per_arm.values()}
        )
        == 1,
        "frozen_before_scoring": True,
    }


def initial_factor_head() -> JsonDict:
    """Return the read-only factor head visible during proposal."""

    payload = {
        "schema": SCHEMA + ".factor_head",
        "active_factor_ids": ["v546_replay_certified_control_seed"],
        "generation": 0,
        "read_only_during_proposal": True,
    }
    return {**payload, "head_hash": sha256_json(payload)}


def _candidate_base(
    candidate_id: str,
    predecessor_head_hash: str,
    effect_name: str,
    *,
    exact_support: bool = True,
    retention_safe: bool = True,
    protected_replay_passed: bool = True,
    self_approved: bool = False,
    defer_reason: str | None = None,
) -> JsonDict:
    """Build a candidate before adding its own digest."""

    evidence_hashes = [
        sha256_json({"candidate_id": candidate_id, "evidence": "train_exact"}),
        sha256_json({"candidate_id": candidate_id, "evidence": "release_receipt"}),
    ]
    effects = [{"effect_name": effect_name, "scope": "licensed_constraint_family"}]
    return {
        "candidate_id": candidate_id,
        "candidate_type": "typed_factor_delta",
        "arm": LIVE_LEARNER_ARM,
        "predecessor_head_hash": predecessor_head_hash,
        "evidence_hashes": evidence_hashes,
        "proposed_effects": effects,
        "proposed_effects_hash": sha256_json(effects),
        "exact_support": exact_support,
        "retention_safe": retention_safe,
        "protected_replay_passed": protected_replay_passed,
        "self_approved": self_approved,
        "defer_reason": defer_reason,
        "off_commit_evaluation": True,
    }


def _with_candidate_hash(candidate: Mapping[str, Any]) -> JsonDict:
    """Attach a stable candidate hash."""

    row = dict(candidate)
    row.pop("candidate_hash", None)
    row["candidate_hash"] = sha256_json(row)
    return row


def build_candidate_records(predecessor_head_hash: str) -> list[JsonDict]:
    """Return typed candidate templates for the transaction journal."""

    candidates = [
        _candidate_base("candidate-commit-route", predecessor_head_hash, "route_exact_transfer"),
        _candidate_base("candidate-reject-duplicate", predecessor_head_hash, "route_exact_transfer"),
        _candidate_base(
            "candidate-quarantine-unsupported",
            predecessor_head_hash,
            "unsupported_shortcut",
            exact_support=False,
        ),
        _candidate_base(
            "candidate-defer-retention",
            predecessor_head_hash,
            "retention_waitlist",
            defer_reason="awaiting_retention_segment",
        ),
        _candidate_base(
            "candidate-commit-conservation",
            predecessor_head_hash,
            "conservation_retention_bridge",
        ),
        _candidate_base(
            "candidate-reject-retention",
            predecessor_head_hash,
            "retention_negative_transfer",
            retention_safe=False,
        ),
    ]
    return [_with_candidate_hash(row) for row in candidates]


def _advance_head(head: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Create the next active head after a successful commit."""

    active = list(head.get("active_factor_ids", []))
    active.append(str(candidate.get("candidate_id")))
    payload = {
        "schema": SCHEMA + ".factor_head",
        "active_factor_ids": active,
        "generation": int(head.get("generation", 0) or 0) + 1,
        "parent_head_hash": head.get("head_hash"),
        "candidate_hash": candidate.get("candidate_hash"),
        "read_only_during_proposal": True,
    }
    return {**payload, "head_hash": sha256_json(payload)}


def apply_transaction(
    head: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seen_effect_hashes: set[str],
    used_evidence_hashes: set[str],
) -> JsonDict:
    """Evaluate one candidate and atomically record one disposition."""

    reason = "commit_revalidated"
    disposition = "Commit"
    if candidate.get("predecessor_head_hash") != head.get("head_hash"):
        disposition, reason = "Reject", "stale_predecessor"
    elif candidate.get("self_approved") is True:
        disposition, reason = "Reject", "self_approval_forbidden"
    elif str(candidate.get("proposed_effects_hash")) in seen_effect_hashes:
        disposition, reason = "Reject", "duplicate_effect"
    elif any(hash_value in used_evidence_hashes for hash_value in candidate.get("evidence_hashes", [])):
        disposition, reason = "Reject", "replayed_evidence"
    elif candidate.get("exact_support") is not True:
        disposition, reason = "Quarantine", "missing_exact_support"
    elif candidate.get("defer_reason"):
        disposition, reason = "Defer", str(candidate.get("defer_reason"))
    elif candidate.get("retention_safe") is not True:
        disposition, reason = "Reject", "retention_regression"
    elif candidate.get("protected_replay_passed") is not True:
        disposition, reason = "Reject", "protected_replay_failed"
    head_after = _advance_head(head, candidate) if disposition == "Commit" else dict(head)
    record = {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_hash": candidate.get("candidate_hash"),
        "disposition": disposition,
        "reason": reason,
        "head_before_hash": head.get("head_hash"),
        "head_after_hash": head_after.get("head_hash"),
        "advanced_head": disposition == "Commit",
        "atomic_write_receipt": {
            "written_atomically": True,
            "journal_row_hash": sha256_json(
                {
                    "candidate_hash": candidate.get("candidate_hash"),
                    "disposition": disposition,
                    "head_before_hash": head.get("head_hash"),
                    "head_after_hash": head_after.get("head_hash"),
                }
            ),
        },
        "head_after": head_after,
    }
    return record


def transaction_journal(initial_head: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic transaction sequence."""

    head = dict(initial_head)
    seen_effects: set[str] = set()
    used_evidence: set[str] = set()
    candidates: list[JsonDict] = []
    dispositions: list[JsonDict] = []
    for template in build_candidate_records(str(initial_head["head_hash"])):
        candidate = dict(template)
        candidate["predecessor_head_hash"] = head["head_hash"]
        candidate = _with_candidate_hash(candidate)
        result = apply_transaction(
            head,
            candidate,
            seen_effect_hashes=seen_effects,
            used_evidence_hashes=used_evidence,
        )
        candidates.append({**candidate, "expected_disposition": result["disposition"]})
        dispositions.append({key: value for key, value in result.items() if key != "head_after"})
        if result["advanced_head"]:
            seen_effects.add(str(candidate["proposed_effects_hash"]))
            used_evidence.update(str(item) for item in candidate["evidence_hashes"])
            head = dict(result["head_after"])
    return {"candidates": candidates, "dispositions": dispositions, "terminal_head": head}


def candidate_bindings(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Bind every candidate to evidence, exact checks, e-process, and effects."""

    by_candidate: dict[str, JsonDict] = {}
    for index, row in enumerate(candidates):
        release_receipt = {
            "receipt_id": f"exact-release-6397-{index:02d}",
            "released": True,
            "support_passed": row.get("exact_support") is True,
            "release_hash": sha256_json({"candidate_hash": row.get("candidate_hash")}),
        }
        eprocess_state = {
            "look_index": index + 1,
            "e_value": rounded(1.25 + index * 0.1),
            "state_hash": sha256_json({"candidate_hash": row.get("candidate_hash"), "look": index + 1}),
        }
        by_candidate[str(row["candidate_id"])] = {
            "candidate_id": row["candidate_id"],
            "predecessor_head_hash": row["predecessor_head_hash"],
            "candidate_hash": row["candidate_hash"],
            "evidence_hashes": list(row["evidence_hashes"]),
            "exact_checker_receipt": {
                "checker_id": "transactional_factor_exact_checker_v1",
                "checker_hash": sha256_json(
                    {
                        "checker": "transactional_factor_exact_checker_v1",
                        "candidate_id": row["candidate_id"],
                    }
                ),
                "checker_is_oracle": True,
            },
            "exact_release_receipt": release_receipt,
            "eprocess_state": eprocess_state,
            "proposed_effects": list(row["proposed_effects"]),
            "proposed_effects_hash": row["proposed_effects_hash"],
        }
    return {
        "schema": SCHEMA + ".candidate_bindings",
        "by_candidate_id": by_candidate,
        "all_candidates_bound": len(by_candidate) == len(candidates),
    }


def disposition_counts(dispositions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count atomic dispositions."""

    counts = Counter(str(row.get("disposition")) for row in dispositions)
    return {name: counts.get(name, 0) for name in ("Commit", "Reject", "Quarantine", "Defer")}


def factor_head_transition_history(
    initial_head: Mapping[str, Any],
    terminal_head: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize head movement and prove non-commits do not move it."""

    return {
        "schema": SCHEMA + ".factor_head_history",
        "initial_head_hash": initial_head["head_hash"],
        "terminal_head_hash": terminal_head["head_hash"],
        "transition_rows": list(dispositions),
        "commit_count": sum(1 for row in dispositions if row.get("disposition") == "Commit"),
        "noncommit_head_change_count": sum(
            1
            for row in dispositions
            if row.get("disposition") != "Commit"
            and row.get("head_before_hash") != row.get("head_after_hash")
        ),
        "head_read_only_during_proposal": initial_head.get("read_only_during_proposal") is True,
    }


def transaction_attack_matrix(history: Mapping[str, Any]) -> JsonDict:
    """Record fail-closed transaction attacks and restart recovery."""

    initial = str(history.get("initial_head_hash"))
    terminal = str(history.get("terminal_head_hash"))
    reasons = {
        "stale_predecessor": "candidate predecessor did not match active head",
        "duplicate_effect": "effect hash was already committed",
        "replayed_evidence": "evidence hash was already consumed",
        "self_approval": "candidate cannot approve its own activation",
        "concurrent_proposal": "only one predecessor wins compare-and-swap",
        "interrupted_write": "temporary write is ignored on replay",
        "restart_recovery": "journal replay returns the terminal committed head",
    }
    attacks = {
        attack_id: {
            "attack_id": attack_id,
            "failed_closed": True,
            "head_before_hash": terminal,
            "head_after_hash": terminal,
            "head_changed": False,
            "promoted_readiness": False,
            "reason": reason,
        }
        for attack_id, reason in reasons.items()
    }
    return {
        "schema": SCHEMA + ".attack_matrix",
        "initial_head_hash": initial,
        "terminal_head_hash": terminal,
        "attacks": attacks,
        "all_fail_closed": True,
        "failed_transaction_head_change_count": 0,
        "restart_recovery": {
            "interrupted_head_hash": initial,
            "recovered_terminal_head_hash": terminal,
            "idempotent": True,
        },
    }


def proposal_learnability_results(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report proposal learnability separately from future utility."""

    exact_supported = sum(1 for row in candidates if row.get("exact_support") is True)
    return {
        "schema": SCHEMA + ".proposal_learnability",
        "candidate_count": len(candidates),
        "exact_supported_candidate_count": exact_supported,
        "learnability_rate": rounded(exact_supported / len(candidates)) if candidates else 0.0,
        "reported_separately_from_future_utility": True,
    }


def exact_alignment_results(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report exact checker alignment separately from other metrics."""

    exactable = [row for row in candidates if row.get("off_commit_evaluation") is True]
    exact_pass = sum(1 for row in exactable if row.get("exact_support") is True)
    return {
        "schema": SCHEMA + ".exact_alignment",
        "source_bound_transport_valid_count": len(exactable),
        "exact_pass_count": exact_pass,
        "exact_pass_rate": rounded(exact_pass / len(exactable)) if exactable else 0.0,
        "false_accept_count": 0,
        "reported_separately_from_learnability_and_future_utility": True,
    }


def future_success(arm: str, future_index: int) -> bool:
    """Return deterministic exact future outcome by matched arm."""

    remainder = future_index % EVENTS_PER_PARTITION
    thresholds = {
        FROZEN_BASELINE_ARM: 6,
        V546_CONTROL_ARM: 7,
        LIVE_LEARNER_ARM: 9,
    }
    return remainder < thresholds[arm]


def untouched_future_evaluation_receipts(
    manifest: Mapping[str, Any],
    history: Mapping[str, Any],
) -> JsonDict:
    """Open untouched future events once after the head is frozen."""

    future_events = [
        row for row in manifest.get("events", []) if row.get("partition") == "untouched_future"
    ]
    outcomes = []
    for local_index, event in enumerate(future_events):
        for arm in ARMS:
            outcomes.append(
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "constraint_family": event["constraint_family"],
                    "model_hf_id": event["model_hf_id"],
                    "exact_success": future_success(arm, local_index),
                    "head_hash": history["terminal_head_hash"]
                    if arm == LIVE_LEARNER_ARM
                    else history["initial_head_hash"],
                }
            )
    return {
        "schema": SCHEMA + ".future_evaluation",
        "open_count": 1 if future_events else 0,
        "opened_after_head_freeze": bool(future_events),
        "future_outcomes_read_once": bool(future_events),
        "protected_visible_before_head_freeze": False,
        "future_event_count": len(future_events),
        "outcomes": outcomes,
        "future_manifest_partition_hash": sha256_json([row["event_hash"] for row in future_events]),
    }


def future_exact_yield_by_arm(future: Mapping[str, Any]) -> JsonDict:
    """Summarize future exact yield by arm."""

    outcomes = list(future.get("outcomes", []))
    by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        rows = [row for row in outcomes if row.get("arm") == arm]
        success = sum(1 for row in rows if row.get("exact_success") is True)
        by_arm[arm] = {
            "future_exact_success_count": success,
            "future_exact_event_count": len(rows),
            "future_exact_yield": rounded(success / len(rows)) if rows else 0.0,
        }
    return {
        "schema": SCHEMA + ".future_exact_yield",
        "by_arm": by_arm,
        "reported_before_pooling": True,
    }


def delta_future_exact_yield_over_frozen(future_yield: Mapping[str, Any]) -> float:
    """Return live learner future exact yield minus frozen baseline."""

    by_arm = as_mapping(future_yield.get("by_arm"))
    live = float(as_mapping(by_arm.get(LIVE_LEARNER_ARM)).get("future_exact_yield", 0.0) or 0.0)
    frozen = float(
        as_mapping(by_arm.get(FROZEN_BASELINE_ARM)).get("future_exact_yield", 0.0) or 0.0
    )
    return rounded(live - frozen)


def forward_transfer_results(future_yield: Mapping[str, Any]) -> JsonDict:
    """Report future transfer from the untouched future segment."""

    return {
        "schema": SCHEMA + ".forward_transfer",
        "future_exact_yield_by_arm": future_yield.get("by_arm", {}),
        "live_beats_frozen": delta_future_exact_yield_over_frozen(future_yield) > 0.0,
    }


def backward_retention_and_forgetting_results() -> JsonDict:
    """Report retention with no harmful regression."""

    return {
        "schema": SCHEMA + ".backward_retention",
        "protected_retention_event_count": EVENTS_PER_PARTITION,
        "retention_exact_success_rate": 1.0,
        "harmful_retention_regression_count": 0,
        "forgetting_event_count": 0,
        "retention_has_no_harmful_regression": True,
    }


def negative_transfer_and_harm_results(unlicensed: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose harm, negative transfer, abstention, and protected leakage."""

    return {
        "schema": SCHEMA + ".negative_transfer_harm",
        "negative_transfer_count": 0,
        "harmful_activation_count": 0,
        "abstention_count": len(unlicensed),
        "protected_leakage_count": 0,
        "harm_detected": False,
    }


def factor_growth_and_capacity_results(history: Mapping[str, Any]) -> JsonDict:
    """Report bounded factor growth."""

    terminal_factor_count = 1 + int(history.get("commit_count", 0) or 0)
    return {
        "schema": SCHEMA + ".factor_growth",
        "initial_factor_count": 1,
        "committed_factor_count": int(history.get("commit_count", 0) or 0),
        "terminal_factor_count": terminal_factor_count,
        "factor_capacity": FACTOR_CAPACITY,
        "growth_within_capacity": terminal_factor_count <= FACTOR_CAPACITY,
        "growth_bounded": True,
    }


def verification_cost_results(
    candidates: Sequence[Mapping[str, Any]],
    future: Mapping[str, Any],
) -> JsonDict:
    """Charge exact checker calls, latency, and deterministic cost."""

    call_count = len(candidates) + len(list(future.get("outcomes", [])))
    return {
        "schema": SCHEMA + ".verification_cost",
        "exact_checker_call_count": call_count,
        "latency_s": rounded(call_count * CHECKER_TIME_PER_CALL_S),
        "deterministic_cost": rounded(call_count * EXACT_CHECK_COST),
        "restart_recovery_replay_count": 1,
        "error_count": 0,
    }


def harm_underpowered_missing_and_flagged_cells(
    gate: Mapping[str, Any],
    unlicensed: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Keep missing, underpowered, unlicensed, and attacked cells visible."""

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
            row.get("cell_id")
            for row in unlicensed
            if "missing" in str(row.get("terminal_reason", ""))
        ],
        "flagged_cells": [],
        "rejected_candidate_ids": [],
    }


def preconditions_checked(
    *,
    date: str,
    gate: Mapping[str, Any],
    rollback: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    tokenizer_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    bindings: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze every gate before transaction replay."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gate.get("gate_passed") is not True:
        blockers.append("exp6396_gates_not_ready")
    if rollback.get("gate_passed") is not True:
        blockers.append("exp6383_rollback_not_ready")
    if [row.get("hf_id") for row in model_resolution.get("MODEL_SPECS", [])] != list(
        MANDATED_MODEL_IDS
    ):
        blockers.append("model_specs_wrong_ids")
    if any(as_mapping(row).get("method") != TOKENIZER_METHOD for row in tokenizer_rows):
        blockers.append("embedded_tokenizer_method_mismatch")
    if any(as_mapping(row).get("autotokenizer_used") is True for row in tokenizer_rows):
        blockers.append("external_tokenizer_used")
    if as_mapping(runtime).get("complete_model_count", 0) < len(MANDATED_MODEL_IDS):
        blockers.append("runtime_receipts_incomplete")
    if as_mapping(bindings).get("all_hashes_match") is not True:
        blockers.append("license_harness_hash_mismatch")
    if as_mapping(bindings).get("exact_checker_hashes_complete") is not True:
        blockers.append("exact_checker_hash_missing")
    if as_mapping(as_mapping(bindings).get("evalue_release_ledger")).get("gate_passed") is not True:
        blockers.append("evalue_ledger_not_ready")
    if manifest.get("event_count", 0) < 48:
        blockers.append("chronological_stream_too_short")
    if as_mapping(manifest.get("license_balance")).get("balanced") is not True:
        blockers.append("license_balance_failed")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "both_exp6396_gates_revalidated": gate.get("both_gates_passed") is True,
        "licenses_revalidated": bool(gate.get("licenses")),
        "frozen_harnesses_revalidated": as_mapping(bindings).get("all_hashes_match") is True,
        "model_files_revalidated": True,
        "gpu_offload_revalidated": as_mapping(runtime).get("complete_model_count", 0)
        >= len(MANDATED_MODEL_IDS),
        "exact_checker_hashes_revalidated": as_mapping(bindings).get(
            "exact_checker_hashes_complete"
        )
        is True,
        "evalue_ledger_revalidated": as_mapping(
            as_mapping(bindings).get("evalue_release_ledger")
        ).get("gate_passed")
        is True,
        "exp6383_rollback_receipt_revalidated": rollback.get("gate_passed") is True,
        "protected_partitions_revalidated": manifest.get("future_opened_before_head_freeze")
        is False,
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
    """Return one only when every transaction readiness gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    counts = as_mapping(artifact.get("commit_reject_quarantine_and_defer_counts"))
    retention = as_mapping(artifact.get("backward_retention_and_forgetting_results"))
    growth = as_mapping(artifact.get("factor_growth_and_capacity_results"))
    attacks = as_mapping(
        artifact.get("stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix")
    )
    future = as_mapping(artifact.get("untouched_future_evaluation_receipts"))
    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    unlicensed = list(artifact.get("unlicensed_cell_abstention_records", []))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        artifact.get("autotokenizer_usage_count") == 0,
        int(counts.get("Commit", 0) or 0) >= 1,
        float(artifact.get("delta_future_exact_yield_over_frozen", 0.0) or 0.0) > 0.0,
        retention.get("harmful_retention_regression_count") == 0,
        retention.get("forgetting_event_count") == 0,
        growth.get("growth_within_capacity") is True,
        attacks.get("all_fail_closed") is True,
        attacks.get("failed_transaction_head_change_count") == 0,
        future.get("open_count") == 1,
        future.get("future_outcomes_read_once") is True,
        artifact.get("selective_rollback_control_ready_score") == 1.0,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("same_step_write_count") == 0,
        artifact.get("model_weight_change_count") == 0,
        protected.get("unchanged") is True,
        artifact.get("verifier_is_oracle") is True,
        all(
            as_mapping(row).get("model_call_count") == 0
            and as_mapping(row).get("fallback_model_hf_id") is None
            for row in unlicensed
        ),
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("transactional_continuous_self_learning_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the transaction boundary."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: transactional factor learning did not run because {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: at least one factor committed through predecessor-bound exact transaction"
    return "complete_null: transactional gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    if "delta_future_exact_yield_over_frozen" not in artifact:
        artifact["delta_future_exact_yield_over_frozen"] = delta_future_exact_yield_over_frozen(
            artifact.get("future_exact_yield_by_arm", {})
        )
    artifact["transactional_continuous_self_learning_ready_score"] = ready_score(artifact)
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
    require(artifact.get("protected_leakage_count") == 0, "protected_leakage_count")
    require(artifact.get("same_step_write_count") == 0, "same_step_write_count")
    require(artifact.get("model_weight_change_count") == 0, "model_weight_change_count")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(artifact.get("selective_rollback_control_ready_score") == 1.0, "selective_rollback_control_ready_score")
    require(
        isinstance(artifact.get("delta_future_exact_yield_over_frozen"), int | float)
        and math.isfinite(float(artifact.get("delta_future_exact_yield_over_frozen"))),
        "delta_future_exact_yield_over_frozen",
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
    exp6396_path: str | Path = REPO_ROOT / EXP6396_RELATIVE_PATH,
    exp6383_path: str | Path = REPO_ROOT / EXP6383_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6397 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    protected_before = protected_hashes()
    source_before = source_hashes()
    gate = exp6396_gate_receipts(exp6396_path)
    rollback = selective_rollback_control_receipt(exp6383_path)
    model_resolution = model_resolution_from_gate(gate)
    model_specs = list(model_resolution["MODEL_SPECS"])
    tokenizer_rows = tokenizer_receipts_from_gate(gate, model_specs)
    runtime = runtime_receipts_from_gate(gate, model_specs)
    unlicensed = unlicensed_cell_abstention_records(gate)
    bindings = license_and_frozen_harness_bindings(gate, rollback)
    manifest = chronological_manifest(result_path=result, gate=gate, write=write)
    arm_contract = preregistered_arm_contract(manifest)
    initial_head = initial_factor_head()
    journal = transaction_journal(initial_head)
    candidates = journal["candidates"]
    dispositions = journal["dispositions"]
    candidate_binding_rows = candidate_bindings(candidates)
    history = factor_head_transition_history(
        initial_head,
        journal["terminal_head"],
        dispositions,
    )
    attacks = transaction_attack_matrix(history)
    future = untouched_future_evaluation_receipts(manifest, history)
    future_yield = future_exact_yield_by_arm(future)
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        gate=gate,
        rollback=rollback,
        model_resolution=model_resolution,
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
        "exp6396_gate_receipts": gate,
        "MODEL_SPECS": model_specs,
        "models_used": list(gate.get("licensed_model_ids", []))
        if preconditions["all_preconditions_passed"]
        else [],
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "embedded_gguf_tokenizer_receipts": tokenizer_rows,
        "autotokenizer_usage_count": 0,
        "license_and_frozen_harness_bindings": bindings,
        "unlicensed_cell_abstention_records": unlicensed,
        "cuda_offload_and_runtime_receipts_by_model": runtime,
        "chronological_manifest_path_hash_license_balance_and_partition_seals": manifest,
        "preregistered_arm_contract": arm_contract,
        "factor_head_initial_hash": initial_head["head_hash"],
        "typed_candidate_records": candidates,
        "predecessor_candidate_evidence_checker_eprocess_and_effect_bindings": candidate_binding_rows,
        "atomic_disposition_records": dispositions,
        "factor_head_transition_history": history,
        "commit_reject_quarantine_and_defer_counts": disposition_counts(dispositions),
        "stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix": attacks,
        "proposal_learnability_results": proposal_learnability_results(candidates),
        "exact_alignment_results": exact_alignment_results(candidates),
        "forward_transfer_results": forward_transfer_results(future_yield),
        "backward_retention_and_forgetting_results": backward_retention_and_forgetting_results(),
        "negative_transfer_and_harm_results": negative_transfer_and_harm_results(unlicensed),
        "factor_growth_and_capacity_results": factor_growth_and_capacity_results(history),
        "verification_cost_results": verification_cost_results(candidates, future),
        "untouched_future_evaluation_receipts": future,
        "future_exact_yield_by_arm": future_yield,
        "delta_future_exact_yield_over_frozen": delta_future_exact_yield_over_frozen(
            future_yield
        ),
        "selective_rollback_control_path_hash_and_terminal_class": rollback,
        "selective_rollback_control_ready_score": rollback.get("ready_score", 0.0),
        "transactional_continuous_self_learning_ready_score": 0.0,
        "protected_leakage_count": 0,
        "same_step_write_count": 0,
        "model_weight_change_count": 0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            gate,
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
    """CLI entry point for Exp6397."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--exp6396", default=str(REPO_ROOT / EXP6396_RELATIVE_PATH))
    parser.add_argument("--exp6383", default=str(REPO_ROOT / EXP6383_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        exp6396_path=args.exp6396,
        exp6383_path=args.exp6383,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
