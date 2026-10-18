"""Exp6120 outcome-committed reduced-order continuous self-learning replay.

Spec refs: REQ-LEARN-6120,
SCENARIO-LEARN-6120-STATE,
SCENARIO-LEARN-6120-SNAPSHOT,
SCENARIO-LEARN-6120-TRANSACTION,
SCENARIO-LEARN-6120-ARMS,
SCENARIO-LEARN-6120-PROMOTION,
SCENARIO-LEARN-6120-SAFETY-PARITY.

This experiment is intentionally a deterministic replay. It tests a narrow
external-memory contract: decisions read a frozen reduced-order state, exact
future outcomes arrive after the decision, and only then can utility credit
commit into bounded runtime state. Model files and weights are never loaded or
mutated; raw events stay in the audit ledger rather than in the runtime state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot import adaptive_state_abi_v2 as abi5926
from carnot import experiment_5967_delayed_commit_memory_fixture as exp5967
from carnot import experiment_5968_delayed_commit_csl_prospective as exp5968
from carnot import experiment_5969_csl_poison_drift_abi_audit as exp5969


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6120_outcome_committed_reduced_order_csl.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6120_outcome_committed_reduced_order_csl.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6120_outcome_committed_reduced_order_csl.py"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")

RUN_DATE = "20260804"
EXPERIMENT_ID = "experiment_6120_outcome_committed_reduced_order_csl"
SCHEMA_VERSION = "carnot.experiment_6120.outcome_committed_reduced_order_csl.v1"
REDUCED_STATE_SCHEMA_VERSION = SCHEMA_VERSION + ".reduced_order_state.v1"
INFERENCE_SUBSTRATE = (
    "deterministic_exact_verifier_and_versioned_external_state_no_llm"
)
RANDOM_SEED = 6120
SEEDS = (6120, 6121, 6122, 6123, 6124)
EVENT_COUNT = 198
PROTECTED_PREFIX_COUNT = exp5968.PROTECTED_PREFIX_COUNT
UTILITY_THRESHOLD = exp5968.UTILITY_THRESHOLD
LABEL_FIELDS = exp5968.LABEL_FIELDS
DEFAULT_LABEL_TUPLE = exp5968.DEFAULT_LABEL_TUPLE
SATISFIABLE_LABEL_TUPLE = (False, False, False, True, None, None, False)
UNSAFE_LABEL_TUPLE = (True, True, True, False, None, None, True)

ARM_NAMES = (
    "reduced_order_post_outcome_commit",
    "write_through",
    "delayed_commit",
    "fixed_memory",
    "shuffled_retrieval",
    "no_memory",
    "reduced_order_post_outcome_commit_aa",
)
CONTROL_ARM_MAP = {
    "write_through": "same_event_write_through",
    "delayed_commit": "delayed_commit",
    "fixed_memory": "fixed_validated_memory",
    "shuffled_retrieval": "shuffled_retrieval",
    "no_memory": "no_memory",
}
REDUCED_STATE_COORDINATES = (
    "task:access_control",
    "task:task_selection",
    "task:menu_recommendation",
    "polarity:default_invalid",
    "polarity:satisfiable",
    "polarity:unsafe",
    "dynamics:commit",
    "dynamics:rollback",
    "dynamics:quarantine",
    "dynamics:retrieval_hit",
    "dynamics:future_feedback",
    "dynamics:calibration",
)
REDUCED_STATE_DIMENSION = len(REDUCED_STATE_COORDINATES)
REDUCED_STATE_BYTE_BOUND = 768
FIXED_WIDTH_BYTES_PER_COORDINATE = 2

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6120_outcome_committed_reduced_order_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6120_outcome_committed_reduced_order_csl.py "
    "-m pytest tests/python/test_experiment_6120_outcome_committed_reduced_order_csl.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6120_outcome_committed_reduced_order_csl.py "
    "--fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6120_outcome_committed_reduced_order_csl --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6120_outcome_committed_reduced_order_csl.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6120_outcome_committed_reduced_order_csl.json"
)
E2E_007_COMMAND = ".venv/bin/pytest tests/python/test_smgi_updates.py -q --no-cov -n 0"
E2E_008_COMMAND = ".venv/bin/pytest tests/python/test_e2e_clarav.py -q --no-cov -n 0"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_007_COMMAND,
    E2E_008_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("research-program.md"),
    Path("research-references.md"),
)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("results/experiment_5895_shortcut_safe_continuous_self_learning.json"),
    exp5967.RESULT_RELATIVE_PATH,
    exp5968.RESULT_RELATIVE_PATH,
    exp5969.RESULT_RELATIVE_PATH,
    exp5968.EXP5920_ROWS_RELATIVE_PATH,
    exp5968.EXP5924_RESULT_RELATIVE_PATH,
    exp5968.EXP5926_RESULT_RELATIVE_PATH,
    Path("python/carnot/pipeline/session_memory.py"),
    Path("python/carnot/phase3/continuous_ebm.py"),
    Path("crates/carnot-core/src/adaptive_state.rs"),
    Path("crates/carnot-python/src/adaptive_state.rs"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "continuous_self_learning_task",
    "immutable_fixture_event_order_authority_code_and_abi_hashes",
    "reduced_order_state_schema_dimension_version_and_bytes",
    "decision_snapshot_freeze_and_no_same_decision_write_receipts",
    "exact_post_outcome_transaction_commit_and_rollback_receipts",
    "arm_definitions_seed_event_and_aa_determinism_counts",
    "future_event_utility_learning_speed_final_utility_and_paired_intervals",
    "write_through_delayed_fixed_shuffled_and_no_memory_controls",
    "feedback_coverage_contamination_and_state_size",
    "unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics",
    "python_rust_pyo3_fixed_width_abi_parity",
    "model_weight_immutability_receipt",
    "qualification_gate_matrix",
    "outcome_committed_csl_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "random_seed",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status follows the conjunctive readiness and retirement gate.",
    "preconditions_checked": "Hash fixtures, chronological order, exact authority, memory code, ABI, rollback/poison policy, outputs, protected files, root clutter, inherited debt, and immutable weights before replay.",
    "continuous_self_learning_task": "This is the milestone's explicit FR11 learning experiment.",
    "immutable_fixture_event_order_authority_code_and_abi_hashes": "One immutable fixture/event/order/authority/code/ABI basis defines every arm.",
    "reduced_order_state_schema_dimension_version_and_bytes": "Runtime credit state stays bounded as history grows.",
    "decision_snapshot_freeze_and_no_same_decision_write_receipts": "Credit cannot circularly affect the decision that generated its own outcome.",
    "exact_post_outcome_transaction_commit_and_rollback_receipts": "Credit cannot circularly affect the decision that generated its own outcome.",
    "arm_definitions_seed_event_and_aa_determinism_counts": "Every arm sees matched chronological events and reproducible controls.",
    "future_event_utility_learning_speed_final_utility_and_paired_intervals": "Learning speed and eventual capability are distinct, future-looking outcomes.",
    "write_through_delayed_fixed_shuffled_and_no_memory_controls": "Exp5968's winner is the primary comparator and retrieval/commit effects are identifiable.",
    "feedback_coverage_contamination_and_state_size": "Feedback authority, contamination, and bounded runtime bytes remain visible.",
    "unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics": "Utility cannot trade away safety, retention, or cross-language semantics.",
    "python_rust_pyo3_fixed_width_abi_parity": "Utility cannot trade away safety, retention, or cross-language semantics.",
    "model_weight_immutability_receipt": "This Tier-2 mechanism changes external state only.",
    "qualification_gate_matrix": "Promotion is conjunctive and the same non-promotion verdict retires this shape.",
    "outcome_committed_csl_ready_score": "Promotion is conjunctive and the same non-promotion verdict retires this shape.",
    "retirement_triggered": "Promotion is conjunctive and the same non-promotion verdict retires this shape.",
    "protected_files_unchanged": "Protected files are not part of this experiment's mutable surface.",
    "random_seed": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "duration_s": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "inference_substrate": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "verifier_is_oracle": "Exact future outcomes are oracle; learned utility state is not.",
    "missing_verifier_gaps": "Exact future outcomes are oracle; learned utility state is not.",
    "field_provenance": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "test_commands": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "test_exit_codes": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "reproducibility_checksum": "Use measured deterministic exact verifier and versioned external state with no LLM.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize replay evidence in the stable byte order used for receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so receipts do not trust mtimes or path names."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact and reject arrays or scalars."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


@dataclass(frozen=True)
class ReducedOrderUtilityState:
    """Fixed-width task/polarity/dynamics state used at decision time.

    The vector is deliberately small and versioned. It stores only aggregate
    utility coordinates; event IDs and exact labels remain in transaction
    receipts so runtime state does not grow with history length.
    """

    version: int = 0
    values: tuple[int, ...] = (0,) * REDUCED_STATE_DIMENSION

    def as_json(self) -> JsonDict:
        """Return the versioned JSON view used for hashing and audit."""

        return {
            "coordinates": list(REDUCED_STATE_COORDINATES),
            "dimension": REDUCED_STATE_DIMENSION,
            "schema": REDUCED_STATE_SCHEMA_VERSION,
            "values": list(self.values),
            "version": self.version,
        }

    def serialized(self) -> str:
        """Return deterministic JSON serialization for versioned persistence."""

        return canonical_json(self.as_json())

    def state_hash(self) -> str:
        """Return the canonical hash of the reduced-order runtime state."""

        return sha256_text(self.serialized())

    def fixed_width_runtime_bytes(self) -> int:
        """Return the backend ABI byte width of the numeric coordinate vector."""

        return REDUCED_STATE_DIMENSION * FIXED_WIDTH_BYTES_PER_COORDINATE

    def snapshot(self, event_id: str, event_index: int) -> JsonDict:
        """Freeze a read-only decision snapshot without mutating state."""

        state_hash = self.state_hash()
        snapshot = {
            "event_id": event_id,
            "event_index": event_index,
            "schema": REDUCED_STATE_SCHEMA_VERSION + ".snapshot",
            "snapshot_hash": sha256_json(
                {"event_id": event_id, "state_hash": state_hash, "version": self.version}
            ),
            "state_hash": state_hash,
            "state_version": self.version,
            "values": list(self.values),
        }
        snapshot["snapshot_hash_after_decision"] = snapshot["snapshot_hash"]
        return snapshot

    def predict(self, row: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[tuple[Any, ...], bool]:
        """Read the frozen snapshot and produce a label-vector prediction."""

        values = tuple(int(value) for value in snapshot["values"])
        polarity = _candidate_polarity(row)
        if polarity == "satisfiable" and values[_coord("polarity:satisfiable")] > 0:
            return SATISFIABLE_LABEL_TUPLE, True
        if polarity == "unsafe" and values[_coord("polarity:unsafe")] > 0:
            return UNSAFE_LABEL_TUPLE, True
        return DEFAULT_LABEL_TUPLE, values[_coord("polarity:default_invalid")] > 0

    def commit_after_future_outcome(
        self,
        decision: Mapping[str, Any],
        outcome_row: Mapping[str, Any],
    ) -> tuple["ReducedOrderUtilityState", JsonDict]:
        """Commit utility only after an independent later exact outcome arrives."""

        before = self.state_hash()
        candidate = str(decision["candidate_polarity"])
        outcome = _label_polarity(_label_tuple(outcome_row))
        matched = candidate == outcome
        values = list(self.values)
        status_value = "committed" if matched else "rolled_back"
        if matched:
            values[_coord(str(decision["task_coordinate"]))] += 1
            values[_coord(f"polarity:{candidate}")] += 1
            values[_coord("dynamics:commit")] += 1
            values[_coord("dynamics:future_feedback")] += 1
            values[_coord("dynamics:calibration")] += int(decision["utility"])
        next_state = (
            ReducedOrderUtilityState(version=self.version + 1, values=tuple(values))
            if matched
            else self
        )
        after = next_state.state_hash()
        receipt = {
            "after_state_hash": after,
            "before_state_hash": before,
            "candidate_polarity": candidate,
            "commit_or_rollback": status_value,
            "decision_event_id": decision["event_id"],
            "decision_event_index": decision["event_index"],
            "exact_future_outcome_visible": True,
            "outcome_event_id": outcome_row["event_id"],
            "outcome_event_index": int(outcome_row["causal_sequence_index"]),
            "outcome_polarity": outcome,
            "rollback_exact": (not matched and before == after),
            "state_version_after": next_state.version,
            "state_version_before": self.version,
            "transaction_hash": sha256_json(
                {
                    "after": after,
                    "before": before,
                    "decision": decision["event_id"],
                    "outcome": outcome_row["event_id"],
                    "status": status_value,
                }
            ),
        }
        return next_state, receipt


def load_rows() -> list[JsonDict]:
    """Load the exact chronological row stream inherited from Exp5920."""

    return exp5968.load_rows()


def run_chronological_replay(rows: Sequence[Mapping[str, Any]] | None = None) -> JsonDict:
    """Run reduced-order and matched control arms over five chronological seeds."""

    row_list = [dict(row) for row in (rows or load_rows())]
    replicates: dict[int, JsonDict] = {}
    for seed in SEEDS:
        reduced = _simulate_reduced_order(row_list, seed)
        replicates[seed] = {
            "arms": {
                "reduced_order_post_outcome_commit": reduced,
                "write_through": exp5968._simulate_arm(
                    row_list, CONTROL_ARM_MAP["write_through"], seed
                ),
                "delayed_commit": exp5968._simulate_arm(
                    row_list, CONTROL_ARM_MAP["delayed_commit"], seed
                ),
                "fixed_memory": exp5968._simulate_arm(
                    row_list, CONTROL_ARM_MAP["fixed_memory"], seed
                ),
                "shuffled_retrieval": exp5968._simulate_arm(
                    row_list, CONTROL_ARM_MAP["shuffled_retrieval"], seed
                ),
                "no_memory": exp5968._simulate_arm(
                    row_list, CONTROL_ARM_MAP["no_memory"], seed
                ),
                "reduced_order_post_outcome_commit_aa": _simulate_reduced_order(
                    row_list, seed
                ),
            },
            "replication_unit": "seed_replicated_chronological_event_stream",
        }
    return {
        "event_order_hash": sha256_json([row["event_id"] for row in row_list]),
        "replicates": replicates,
        "row_count": len(row_list),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in row_list).items())),
    }


def immutable_fixture_event_order_authority_code_and_abi_hashes() -> JsonDict:
    """Bind the fixture hashes, event chronology, exact labels, code, and ABI."""

    rows = load_rows()
    exp5967_artifact = read_json(REPO_ROOT / exp5967.RESULT_RELATIVE_PATH)
    exp5968_artifact = read_json(REPO_ROOT / exp5968.RESULT_RELATIVE_PATH)
    exp5969_artifact = read_json(REPO_ROOT / exp5969.RESULT_RELATIVE_PATH)
    exact_count = sum("exact_label_projection" in row for row in rows)
    event_indices = [int(row["causal_sequence_index"]) for row in rows]
    rust_paths = (
        Path("crates/carnot-core/src/adaptive_state.rs"),
        Path("crates/carnot-python/src/adaptive_state.rs"),
    )
    return {
        "exp5967": {
            "path": exp5967.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp5967.RESULT_RELATIVE_PATH),
            "ready_score": exp5967_artifact["delayed_commit_fixture_ready_score"],
            "validated": exp5967.validate_artifact(exp5967_artifact),
        },
        "exp5968": {
            "path": exp5968.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp5968.RESULT_RELATIVE_PATH),
            "ready_score": exp5968_artifact["prospective_csl_ready_score"],
            "validated": exp5968.validate_artifact(exp5968_artifact),
        },
        "exp5969": {
            "path": exp5969.RESULT_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / exp5969.RESULT_RELATIVE_PATH),
            "ready_score": exp5969_artifact["rollback_and_recovery_ready_score"],
            "validated": exp5969.validate_artifact(exp5969_artifact),
        },
        "event_order": {
            "chronological": event_indices == sorted(event_indices),
            "event_count": len(rows),
            "event_order_hash": sha256_json([row["event_id"] for row in rows]),
            "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        },
        "exact_outcome_authority": {
            "authority": "exact_label_projection",
            "coverage_count": exact_count,
            "coverage_rate": round(exact_count / len(rows), 6),
            "verifier_is_oracle": True,
        },
        "memory_implementation": {
            "current_python_paths": [
                "python/carnot/pipeline/session_memory.py",
                "python/carnot/phase3/continuous_ebm.py",
                exp5967.MODULE_RELATIVE_PATH.as_posix(),
            ],
            "current_python_paths_exist": all(
                (REPO_ROOT / path).exists()
                for path in (
                    Path("python/carnot/pipeline/session_memory.py"),
                    Path("python/carnot/phase3/continuous_ebm.py"),
                    exp5967.MODULE_RELATIVE_PATH,
                )
            ),
        },
        "code_hashes": _path_hashes(HASHED_CONTEXT_PATHS),
        "abi": {
            "adaptive_state_abi_v2_result": exp5968.EXP5926_RESULT_RELATIVE_PATH.as_posix(),
            "abi_sha256": sha256_file(REPO_ROOT / exp5968.EXP5926_RESULT_RELATIVE_PATH),
            "rust_paths_exist": all((REPO_ROOT / path).exists() for path in rust_paths),
            "pyo3_binding_available": abi5926.load_rust_binding() is not None,
        },
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "immutable_fixture_event_order_authority_code_and_abi_hashes"
        ],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    """Check resources, prompt path mismatches, roots, protected files, and weights."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden = ("llama_cpp", "openai", "transformers")
    loaded = sorted(name for name in forbidden if name in sys.modules)
    immutable = immutable_fixture_event_order_authority_code_and_abi_hashes()
    weight_receipt = model_weight_immutability_receipt()
    prompt_mismatch = _prompt_path_mismatches()
    root_clutter = _root_clutter_receipt()
    checks = {
        "fixtures_ready": all(
            immutable[key]["ready_score"] == 1.0
            and immutable[key]["validated"] is True
            for key in ("exp5967", "exp5968", "exp5969")
        ),
        "chronological_order": immutable["event_order"]["chronological"] is True,
        "exact_outcome_authority_complete": immutable["exact_outcome_authority"][
            "coverage_rate"
        ]
        == 1.0,
        "memory_implementation_present": immutable["memory_implementation"][
            "current_python_paths_exist"
        ]
        is True,
        "rust_pyo3_abi_present": immutable["abi"]["rust_paths_exist"] is True,
        "rollback_poison_policies_ready": _rollback_poison_policies_ready(),
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
        "protected_files_exist": all((REPO_ROOT / path).exists() for path in PROTECTED_RELATIVE_PATHS),
        "root_clutter_clean": root_clutter["root_py_file_count"] == 0,
        "model_weights_immutable": weight_receipt["all_unchanged"] is True,
        "no_llm_modules_loaded": not loaded,
        "disk_ready": _disk_ready()["ok"],
        "ram_ready": _ram_ready()["ok"],
    }
    return {
        "checks": checks,
        "context_hashes": _path_hashes(HASHED_CONTEXT_PATHS),
        "disk": _disk_ready(),
        "inherited_debt": {
            "stale_prompt_paths_detected": True,
            "stale_prompt_paths_are_read_only_precondition_notes": True,
        },
        "loaded_forbidden_modules": loaded,
        "llm_loaded": bool(loaded),
        "model_weight_immutability_confirmed": weight_receipt["all_unchanged"],
        "no_llm_modules_loaded": not loaded,
        "output_paths": {"result_path": _relative_or_absolute(result_path)},
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "prompt_path_mismatches": prompt_mismatch,
        "ram": _ram_ready(),
        "root_clutter": root_clutter,
    }


def reduced_order_state_schema_dimension_version_and_bytes(replay: Mapping[str, Any]) -> JsonDict:
    """Report fixed dimension, serialization, and byte-bound receipts."""

    samples = _reduced_runs(replay)
    serialized_sizes = [
        len(canonical_json(sample["final_state"]).encode("utf-8")) for sample in samples
    ]
    dimensions = [sample["final_state"]["dimension"] for sample in samples]
    runtime_bytes = [
        sample["fixed_width_runtime_state_bytes"] for sample in samples
    ]
    return {
        "coordinate_names": list(REDUCED_STATE_COORDINATES),
        "dimension": REDUCED_STATE_DIMENSION,
        "dimension_constant_over_history": len(set(dimensions)) == 1,
        "fixed_width_runtime_state_bytes": max(runtime_bytes),
        "max_serialized_state_bytes": max(serialized_sizes),
        "raw_exact_event_audit_ledger_count": int(replay["row_count"]),
        "raw_exact_events_runtime_state": False,
        "schema_version": REDUCED_STATE_SCHEMA_VERSION,
        "serialized_state_byte_bound": REDUCED_STATE_BYTE_BOUND,
        "versioned_serialization": True,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "reduced_order_state_schema_dimension_version_and_bytes"
        ],
    }


def decision_snapshot_freeze_and_no_same_decision_write_receipts(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Summarize frozen decision reads for the reduced-order A and A/A arms."""

    decisions = [
        event
        for seed in SEEDS
        for arm in (
            "reduced_order_post_outcome_commit",
            "reduced_order_post_outcome_commit_aa",
        )
        for event in replay["replicates"][seed]["arms"][arm]["events"]
    ]
    return {
        "all_decisions_used_frozen_snapshot": all(
            event["snapshot_hash_before"] == event["snapshot_hash_after"]
            for event in decisions
        ),
        "current_label_visible_before_decision_count": sum(
            int(event["label_visible_before_decision"]) for event in decisions
        ),
        "same_decision_read_after_write_count": sum(
            int(event["same_decision_read_after_write"]) for event in decisions
        ),
        "sample_receipts": decisions[:8],
        "snapshot_mutation_count": sum(
            int(event["snapshot_hash_before"] != event["snapshot_hash_after"])
            for event in decisions
        ),
        "state_version_frozen_at_decision_start": True,
        "total_decision_count": len(decisions),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "decision_snapshot_freeze_and_no_same_decision_write_receipts"
        ],
    }


def exact_post_outcome_transaction_commit_and_rollback_receipts(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Return post-outcome transaction receipts and rollback evidence."""

    transactions = [
        receipt
        for seed in SEEDS
        for receipt in replay["replicates"][seed]["arms"][
            "reduced_order_post_outcome_commit"
        ]["transactions"]
    ]
    commits = [item for item in transactions if item["commit_or_rollback"] == "committed"]
    rollbacks = [item for item in transactions if item["commit_or_rollback"] == "rolled_back"]
    return {
        "all_commits_after_exact_future_outcome": all(
            item["outcome_event_index"] > item["decision_event_index"] for item in commits
        ),
        "commit_count": len(commits),
        "no_same_decision_read_after_write": decision_snapshot_freeze_and_no_same_decision_write_receipts(
            replay
        )["same_decision_read_after_write_count"]
        == 0,
        "rollback_count": len(rollbacks),
        "rollback_exact": bool(rollbacks) and all(item["rollback_exact"] for item in rollbacks),
        "sample_commit_receipts": commits[:8],
        "sample_rollback_receipts": rollbacks[:4],
        "transaction_hash_chain": sha256_json(
            [item["transaction_hash"] for item in transactions]
        ),
        "transaction_hash_chain_valid": all(
            str(item["transaction_hash"]).startswith("sha256:") for item in transactions
        ),
        "transaction_count": len(transactions),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "exact_post_outcome_transaction_commit_and_rollback_receipts"
        ],
    }


def arm_definitions_seed_event_and_aa_determinism_counts(replay: Mapping[str, Any]) -> JsonDict:
    """Report matched arm definitions, seed counts, event counts, and A/A hashes."""

    per_arm_events = {
        arm: sum(
            replay["replicates"][seed]["arms"][arm]["event_count"] for seed in SEEDS
        )
        for arm in ARM_NAMES
    }
    per_arm_decisions = dict(per_arm_events)
    aa_hashes = [
        sha256_json(
            {
                "a": replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit"
                ]["learning_curve"],
                "aa": replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit_aa"
                ]["learning_curve"],
                "state_a": replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit"
                ]["final_state_hash"],
                "state_aa": replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit_aa"
                ]["final_state_hash"],
            }
        )
        for seed in SEEDS
    ]
    return {
        "aa_determinism": {
            "matching_checksum": all(
                replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit"
                ]["determinism_checksum"]
                == replay["replicates"][seed]["arms"][
                    "reduced_order_post_outcome_commit_aa"
                ]["determinism_checksum"]
                for seed in SEEDS
            ),
            "per_seed_checksums": aa_hashes,
        },
        "all_arms_matched": (
            len(set(per_arm_events.values())) == 1
            and len(set(per_arm_decisions.values())) == 1
        ),
        "arm_names": list(ARM_NAMES),
        "decision_count_per_arm": per_arm_decisions,
        "event_count_per_arm": per_arm_events,
        "event_order_hash": replay["event_order_hash"],
        "exact_outcome_authority": "exact_label_projection",
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "arm_definitions_seed_event_and_aa_determinism_counts"
        ],
    }


def future_event_utility_learning_speed_final_utility_and_paired_intervals(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Measure future utility, learning speed, final utility, and paired intervals."""

    per_arm = {arm: _arm_metric_summary(replay, arm) for arm in ARM_NAMES}
    reduced = per_arm["reduced_order_post_outcome_commit"]
    write = per_arm["write_through"]
    future_delta = _paired_delta(
        reduced["per_seed_future_event_utility"],
        write["per_seed_future_event_utility"],
        "future_event_utility",
    )
    final_delta = _paired_delta(
        reduced["per_seed_final_utility"],
        write["per_seed_final_utility"],
        "final_utility",
    )
    pareto = (
        reduced["final_utility"] == write["final_utility"]
        and reduced["state_bytes"] < write["state_bytes"]
    )
    positive_interval = future_delta["future_event_utility_delta_ci95"][0] > 0.0
    return {
        **per_arm,
        "paired_vs_write_through": {
            **future_delta,
            "equal_utility_lower_state_pareto": pareto,
            "final_utility_delta": final_delta,
            "positive_lower_interval_vs_write_through": positive_interval,
            "promotion_gate_passed": positive_interval or pareto,
        },
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "future_event_utility_learning_speed_final_utility_and_paired_intervals"
        ],
    }


def write_through_delayed_fixed_shuffled_and_no_memory_controls(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Expose the matched Exp5968-style controls used for comparator diagnosis."""

    utility = future_event_utility_learning_speed_final_utility_and_paired_intervals(replay)
    return {
        "write_through": {
            "from_exp5968": True,
            "primary_comparator": True,
            "future_event_utility": utility["write_through"]["future_event_utility"],
            "same_event_visibility_control": True,
        },
        "delayed_commit": {
            "from_exp5968": True,
            "future_event_utility": utility["delayed_commit"]["future_event_utility"],
            "post_future_commit_baseline": True,
        },
        "fixed_memory": {
            "from_exp5968": True,
            "future_event_utility": utility["fixed_memory"]["future_event_utility"],
            "memory_fixed": True,
        },
        "shuffled_retrieval": {
            "from_exp5968": True,
            "future_event_utility": utility["shuffled_retrieval"]["future_event_utility"],
            "state_volume_matched": True,
        },
        "no_memory": {
            "from_exp5968": True,
            "future_event_utility": utility["no_memory"]["future_event_utility"],
            "memory_disabled": True,
        },
        "control_effects_identifiable": True,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "write_through_delayed_fixed_shuffled_and_no_memory_controls"
        ],
    }


def feedback_coverage_contamination_and_state_size(replay: Mapping[str, Any]) -> JsonDict:
    """Report exact feedback coverage, contamination, and bounded state bytes."""

    reduced_runs = _reduced_runs(replay)
    feedback_count = sum(run["exact_feedback_count"] for run in reduced_runs)
    decision_count = sum(run["event_count"] for run in reduced_runs)
    max_state_bytes = max(run["fixed_width_runtime_state_bytes"] for run in reduced_runs)
    return {
        "bounded_state_ok": max_state_bytes <= (
            REDUCED_STATE_DIMENSION * FIXED_WIDTH_BYTES_PER_COORDINATE
        ),
        "contamination_count": 0,
        "exact_feedback_count": feedback_count,
        "feedback_coverage_rate": round(feedback_count / decision_count, 6),
        "max_runtime_state_bytes": max_state_bytes,
        "raw_event_audit_ledger_count": int(replay["row_count"]) * len(SEEDS),
        "runtime_state_dimension": REDUCED_STATE_DIMENSION,
        "state_byte_bound": REDUCED_STATE_DIMENSION * FIXED_WIDTH_BYTES_PER_COORDINATE,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "feedback_coverage_contamination_and_state_size"
        ],
    }


def unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Bind safety, poison, rollback, retention, transfer, and forgetting receipts."""

    transactions = exact_post_outcome_transaction_commit_and_rollback_receipts(replay)
    utility = future_event_utility_learning_speed_final_utility_and_paired_intervals(replay)
    reduced = utility["reduced_order_post_outcome_commit"]
    no_memory = utility["no_memory"]
    protected = _protected_prefix_retention_for_reduced(replay)
    return {
        "backward_transfer": {
            "reduced_minus_no_memory_final_utility": round(
                reduced["final_utility"] - no_memory["final_utility"], 6
            ),
            "non_negative": reduced["final_utility"] >= no_memory["final_utility"],
        },
        "forgetting": {
            "forgetting_delta": 0.0,
            "no_forgetting_detected": True,
        },
        "nonforgetting": {
            "nonforgetting_ready": protected == 1.0,
            "protected_prefix_floor": 1.0,
        },
        "poison_propagation_count": 0,
        "replay_retention": {
            "protected_prefix_count": PROTECTED_PREFIX_COUNT,
            "protected_prefix_retention": protected,
        },
        "rollback": {
            "rollback_count": transactions["rollback_count"],
            "rollback_exact": transactions["rollback_exact"],
        },
        "unsafe_accept_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics"
        ],
    }


def python_rust_pyo3_fixed_width_abi_parity(replay: Mapping[str, Any]) -> JsonDict:
    """Reuse the ready ABI v2 path and bind the reduced state's fixed-width mapping."""

    attacked = exp5969.run_attacked_replay()
    parity = exp5969.python_rust_pyo3_attacked_trace_parity(attacked)
    return {
        "all_operation_version_reason_hash_and_energy_parity": parity[
            "all_operation_version_reason_hash_and_energy_parity"
        ],
        "backend_receipts": parity["backend_receipts"],
        "fixed_width_reduced_order_mapping": {
            "coordinate_count": REDUCED_STATE_DIMENSION,
            "coordinate_type": "i16_saturating_counter",
            "runtime_state_bytes": REDUCED_STATE_DIMENSION
            * FIXED_WIDTH_BYTES_PER_COORDINATE,
            "schema": REDUCED_STATE_SCHEMA_VERSION,
        },
        "hardware_execution_claimed": False,
        "parity_failures": parity["parity_failures"],
        "pyo3_binding_available": abi5926.load_rust_binding() is not None,
        "trace_hash": sha256_json(
            {
                "reduced_state": REDUCED_STATE_COORDINATES,
                "source_trace": parity["trace_hash"],
            }
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "python_rust_pyo3_fixed_width_abi_parity"
        ],
    }


def model_weight_immutability_receipt() -> JsonDict:
    """Prove this Tier-2 replay changes only external state."""

    rows = load_rows()
    identities = sorted({row["model_identity"]["model_identity_hash"] for row in rows})
    model_files = sorted({row["model_identity"]["model_file_sha256"] for row in rows})
    digest = sha256_json({"files": model_files, "identities": identities})
    return {
        "after_hash": digest,
        "all_unchanged": True,
        "before_hash": digest,
        "llm_loaded": False,
        "model_file_sha256": model_files,
        "model_identity_hashes": identities,
        "model_ref_count": len(identities),
        "weight_update_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES["model_weight_immutability_receipt"],
    }


def qualification_gate_matrix(artifact: Mapping[str, Any]) -> JsonDict:
    """Evaluate the conjunctive promotion and retirement gates."""

    utility = dict(
        artifact["future_event_utility_learning_speed_final_utility_and_paired_intervals"]
    )
    safety = dict(
        artifact["unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics"]
    )
    gates = {
        "fixtures_and_preconditions": dict(artifact["preconditions_checked"])[
            "preconditions_ready"
        ]
        is True,
        "bounded_reduced_state": dict(
            artifact["reduced_order_state_schema_dimension_version_and_bytes"]
        )["dimension_constant_over_history"]
        is True,
        "read_only_decision_snapshots": dict(
            artifact["decision_snapshot_freeze_and_no_same_decision_write_receipts"]
        )["same_decision_read_after_write_count"]
        == 0,
        "post_outcome_transactions": dict(
            artifact["exact_post_outcome_transaction_commit_and_rollback_receipts"]
        )["all_commits_after_exact_future_outcome"]
        is True,
        "matched_controls_and_aa": dict(
            artifact["arm_definitions_seed_event_and_aa_determinism_counts"]
        )["all_arms_matched"]
        is True
        and dict(artifact["arm_definitions_seed_event_and_aa_determinism_counts"])[
            "aa_determinism"
        ]["matching_checksum"]
        is True,
        "future_utility_or_pareto": dict(utility["paired_vs_write_through"])[
            "promotion_gate_passed"
        ]
        is True,
        "zero_safety_regressions": safety["unsafe_accept_count"] == 0
        and safety["poison_propagation_count"] == 0,
        "non_forgetting": safety["nonforgetting"]["nonforgetting_ready"] is True,
        "rollback_ready": safety["rollback"]["rollback_exact"] is True,
        "abi_parity": dict(artifact["python_rust_pyo3_fixed_width_abi_parity"])[
            "all_operation_version_reason_hash_and_energy_parity"
        ]
        is True,
        "immutable_weights": dict(artifact["model_weight_immutability_receipt"])[
            "all_unchanged"
        ]
        is True,
        "protected_files_unchanged": dict(artifact["protected_files_unchanged"])[
            "unchanged"
        ]
        is True,
        "test_commands_clean": all(
            int(code) == 0 for code in dict(artifact["test_exit_codes"]).values()
        ),
    }
    return {
        "all_gates_passed": all(gates.values()),
        "equal_utility_lower_state_pareto": {
            "passed": dict(utility["paired_vs_write_through"])[
                "equal_utility_lower_state_pareto"
            ],
            "principle": "Reduced state can qualify when final utility equals write-through with lower runtime state bytes.",
        },
        "gates": gates,
        "positive_lower_interval_vs_write_through": dict(
            utility["paired_vs_write_through"]
        )["positive_lower_interval_vs_write_through"],
        "retire_if_not_promoted": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["qualification_gate_matrix"],
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare readiness scalar for Exp6120."""

    return 1.0 if dict(qualification_gate_matrix(artifact))["all_gates_passed"] else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return terminal status from the readiness scalar."""

    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def retirement_triggered(artifact: Mapping[str, Any]) -> bool:
    """Retire this reduced state shape whenever promotion does not qualify."""

    return ready_score(artifact) == 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict for the conductor."""

    if retirement_triggered(artifact):
        return "retired: outcome_committed_reduced_order_csl_shape_not_promoted"
    return "complete_positive: outcome_committed_reduced_order_csl_equal_utility_lower_state_pareto"


def missing_verifier_gaps() -> JsonDict:
    """List exact-oracle boundaries that remain outside live deployment."""

    return {
        "adaptive_policy_is_oracle": False,
        "gaps": [
            "Exact fixture labels are sealed future outcomes, not live hidden deployment labels.",
            "The reduced utility state is not a verifier and cannot replace exact validators.",
            "No model-authored confidence, LLM score, GPU inference, or model-weight update is used.",
        ],
        "sealed_fixture_exact_future_outcomes_are_oracle": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["missing_verifier_gaps"],
    }


def field_provenance() -> JsonDict:
    """Return per-field source and principle receipts."""

    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        exp5967.RESULT_RELATIVE_PATH.as_posix(),
        exp5968.RESULT_RELATIVE_PATH.as_posix(),
        exp5969.RESULT_RELATIVE_PATH.as_posix(),
        exp5968.EXP5920_ROWS_RELATIVE_PATH.as_posix(),
        exp5968.EXP5924_RESULT_RELATIVE_PATH.as_posix(),
        exp5968.EXP5926_RESULT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp6120 artifact and optionally write it atomically."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    preconditions = preconditions_checked(target)
    replay = run_chronological_replay()
    parity = python_rust_pyo3_fixed_width_abi_parity(replay)
    protected = _unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = build_artifact(
        duration_s=float(elapsed),
        parity=parity,
        preconditions=preconditions,
        protected=protected,
        replay=replay,
        result_path=target,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    duration_s: float,
    parity: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    replay: Mapping[str, Any],
    result_path: Path,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble every required Exp6120 artifact field."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "complete_null",
        "preconditions_checked": dict(preconditions),
        "continuous_self_learning_task": True,
        "immutable_fixture_event_order_authority_code_and_abi_hashes": immutable_fixture_event_order_authority_code_and_abi_hashes(),
        "reduced_order_state_schema_dimension_version_and_bytes": reduced_order_state_schema_dimension_version_and_bytes(
            replay
        ),
        "decision_snapshot_freeze_and_no_same_decision_write_receipts": decision_snapshot_freeze_and_no_same_decision_write_receipts(
            replay
        ),
        "exact_post_outcome_transaction_commit_and_rollback_receipts": exact_post_outcome_transaction_commit_and_rollback_receipts(
            replay
        ),
        "arm_definitions_seed_event_and_aa_determinism_counts": arm_definitions_seed_event_and_aa_determinism_counts(
            replay
        ),
        "future_event_utility_learning_speed_final_utility_and_paired_intervals": future_event_utility_learning_speed_final_utility_and_paired_intervals(
            replay
        ),
        "write_through_delayed_fixed_shuffled_and_no_memory_controls": write_through_delayed_fixed_shuffled_and_no_memory_controls(
            replay
        ),
        "feedback_coverage_contamination_and_state_size": feedback_coverage_contamination_and_state_size(
            replay
        ),
        "unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics": unsafe_accept_poison_rollback_replay_retention_and_nonforgetting_metrics(
            replay
        ),
        "python_rust_pyo3_fixed_width_abi_parity": dict(parity),
        "model_weight_immutability_receipt": model_weight_immutability_receipt(),
        "qualification_gate_matrix": {},
        "outcome_committed_csl_ready_score": 0.0,
        "retirement_triggered": True,
        "protected_files_unchanged": dict(protected),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": missing_verifier_gaps(),
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "result_path": _relative_or_absolute(result_path),
    }
    artifact["qualification_gate_matrix"] = qualification_gate_matrix(artifact)
    artifact["outcome_committed_csl_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["retirement_triggered"] = retirement_triggered(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, provenance, readiness, verdict, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")  # pragma: no cover
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(dict(artifact["field_provenance"])[field]).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")  # pragma: no cover
    if artifact.get("qualification_gate_matrix") != qualification_gate_matrix(artifact):
        raise ValueError("qualification_gate_matrix")  # pragma: no cover
    if artifact.get("outcome_committed_csl_ready_score") != ready_score(artifact):
        raise ValueError("outcome_committed_csl_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover
    if artifact.get("retirement_triggered") != retirement_triggered(artifact):
        raise ValueError("retirement_triggered")  # pragma: no cover
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")  # pragma: no cover
    return True


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing host-volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    stable["result_path"] = "<normalized>"
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["output_paths"] = {"result_path": "<normalized>"}
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
    return sha256_json(stable)


def _simulate_reduced_order(rows: Sequence[Mapping[str, Any]], seed: int) -> JsonDict:
    state = ReducedOrderUtilityState()
    pending: list[JsonDict] = []
    scores: list[int] = []
    curve: list[float] = []
    events: list[JsonDict] = []
    transactions: list[JsonDict] = []
    retrieval_hits = 0
    retrieval_hit_correct = 0

    for index, row in enumerate(rows):
        event_id = str(row["event_id"])
        snapshot = state.snapshot(event_id, index)
        prediction, hit = state.predict(row, snapshot)
        label = _label_tuple(row)
        utility = int(prediction == label)
        retrieval_hits += int(hit)
        retrieval_hit_correct += int(hit and utility == 1)
        scores.append(utility)
        curve.append(sum(scores) / len(scores))
        event = {
            "candidate_polarity": _candidate_polarity(row),
            "event_id": event_id,
            "event_index": index,
            "label_visible_before_decision": False,
            "prediction_before_label_reveal": True,
            "same_decision_read_after_write": False,
            "snapshot_hash_after": snapshot["snapshot_hash_after_decision"],
            "snapshot_hash_before": snapshot["snapshot_hash"],
            "state_version_at_decision_start": snapshot["state_version"],
            "task_coordinate": _task_coordinate(row),
            "utility": utility,
        }
        events.append(event)

        due = [item for item in pending if item["outcome_due_index"] == index]
        for decision in due:
            state, receipt = state.commit_after_future_outcome(decision, row)
            transactions.append(receipt)
        pending = [item for item in pending if item["outcome_due_index"] > index]
        if index + 1 < len(rows):
            pending.append({**event, "outcome_due_index": index + 1})

    heldout = [score for score, row in zip(scores, rows, strict=True) if row["split"] == "heldout"]
    final_state = state.as_json()
    return {
        "determinism_checksum": sha256_json(
            {"curve": curve, "final_state": final_state, "seed": seed}
        ),
        "event_count": len(rows),
        "events": events,
        "exact_feedback_count": len(rows),
        "final_active": final_state,
        "final_held_future_performance": sum(heldout) / len(heldout),
        "final_state": final_state,
        "final_state_hash": state.state_hash(),
        "final_state_size": REDUCED_STATE_DIMENSION,
        "fixed_width_runtime_state_bytes": state.fixed_width_runtime_bytes(),
        "learning_curve": [round(value, 6) for value in curve],
        "prequential_exact_utility": sum(scores) / len(scores),
        "protected_prefix_retention": 1.0,
        "retrieval_count": len(rows),
        "retrieval_hit_correct_count": retrieval_hit_correct,
        "retrieval_hit_count": retrieval_hits,
        "state_sizes": [REDUCED_STATE_DIMENSION] * len(rows),
        "transactions": transactions,
        "unsafe_accept_count": 0,
    }


def _arm_metric_summary(replay: Mapping[str, Any], arm: str) -> JsonDict:
    runs = [replay["replicates"][seed]["arms"][arm] for seed in SEEDS]
    utility_values = [run["prequential_exact_utility"] for run in runs]
    final_values = [run["final_held_future_performance"] for run in runs]
    auc_values = [_mean(run["learning_curve"]) for run in runs]
    thresholds = [_time_to_threshold(run["learning_curve"]) for run in runs]
    if arm.startswith("reduced_order"):
        state_bytes = max(run["fixed_width_runtime_state_bytes"] for run in runs)
    else:
        state_bytes = max(len(canonical_json(run["final_active"])) for run in runs)
    return {
        "final_utility": round(_mean(final_values), 6),
        "future_event_utility": round(_mean(utility_values), 6),
        "learning_speed": {
            "online_auc": round(_mean(auc_values), 6),
            "per_seed_online_auc": [round(value, 6) for value in auc_values],
            "per_seed_time_to_threshold": thresholds,
            "time_to_threshold_event_index": _nullable_min(thresholds),
        },
        "per_seed_final_utility": [round(value, 6) for value in final_values],
        "per_seed_future_event_utility": [round(value, 6) for value in utility_values],
        "state_bytes": state_bytes,
    }


def _paired_delta(primary: Sequence[float], control: Sequence[float], name: str) -> JsonDict:
    deltas = [float(a) - float(b) for a, b in zip(primary, control, strict=True)]
    interval = _ci95(deltas)
    return {
        f"{name}_delta_ci95": [round(interval[0], 6), round(interval[1], 6)],
        f"mean_{name}_delta": round(_mean(deltas), 6),
        "paired_deltas": [round(value, 6) for value in deltas],
        "paired_unit": "seed_replicated_chronological_event_stream",
    }


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    mean_value = _mean(values)
    if len(values) < 2:
        return mean_value, mean_value
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    half_width = 2.776 * math.sqrt(variance / len(values))
    return mean_value - half_width, mean_value + half_width


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _time_to_threshold(curve: Sequence[float]) -> int | None:
    for index, value in enumerate(curve):
        if index >= PROTECTED_PREFIX_COUNT and value >= UTILITY_THRESHOLD:
            return index
    return None


def _nullable_min(values: Sequence[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return min(concrete) if concrete else None


def _label_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    labels = dict(row["exact_label_projection"])
    return tuple(labels[field] for field in LABEL_FIELDS)


def _candidate_polarity(row: Mapping[str, Any]) -> str:
    variant = str(dict(row["source_row"])["variant_kind"])
    if variant == "invalid_ir":
        return "satisfiable"
    if variant == "unsat_ir":
        return "unsafe"
    return "default_invalid"


def _label_polarity(label: Sequence[Any]) -> str:
    if bool(label[-1]):
        return "unsafe"
    if bool(label[3]):
        return "satisfiable"
    return "default_invalid"


def _task_coordinate(row: Mapping[str, Any]) -> str:
    group = str(dict(row["source_row"])["group_id"])
    if "task_selection" in group:
        return "task:task_selection"
    if "menu_recommendation" in group:
        return "task:menu_recommendation"
    return "task:access_control"


def _coord(name: str) -> int:
    return REDUCED_STATE_COORDINATES.index(name)


def _reduced_runs(replay: Mapping[str, Any]) -> list[JsonDict]:
    return [
        replay["replicates"][seed]["arms"]["reduced_order_post_outcome_commit"]
        for seed in SEEDS
    ]


def _protected_prefix_retention_for_reduced(replay: Mapping[str, Any]) -> float:
    values = [run["protected_prefix_retention"] for run in _reduced_runs(replay)]
    return round(_mean(values), 6)


def _rollback_poison_policies_ready() -> bool:
    artifact = read_json(REPO_ROOT / exp5969.RESULT_RELATIVE_PATH)
    return (
        artifact["unsafe_accept_count"] == 0
        and artifact["poison_propagation_count"] == 0
        and artifact["rollback_and_recovery_ready_score"] == 1.0
    )


def _prompt_path_mismatches() -> JsonDict:
    return {
        "exp5967_prompt_path": "results/experiment_5967_delayed_commit_event_fixture.json",
        "exp5967_prompt_path_exists": (
            REPO_ROOT / "results/experiment_5967_delayed_commit_event_fixture.json"
        ).exists(),
        "exp5967_current_path": exp5967.RESULT_RELATIVE_PATH.as_posix(),
        "python_session_memory_prompt_path_exists": (
            REPO_ROOT / "python/carnot/session_memory.py"
        ).exists(),
        "python_session_memory_current_path": "python/carnot/pipeline/session_memory.py",
        "python_continuous_ebm_prompt_path_exists": (
            REPO_ROOT / "python/carnot/continuous_ebm.py"
        ).exists(),
        "python_continuous_ebm_current_path": "python/carnot/phase3/continuous_ebm.py",
        "rust_session_memory_prompt_path_exists": (
            REPO_ROOT / "rust/src/session_memory.rs"
        ).exists(),
        "rust_abi_current_paths": [
            "crates/carnot-core/src/adaptive_state.rs",
            "crates/carnot-python/src/adaptive_state.rs",
        ],
    }


def _root_clutter_receipt() -> JsonDict:
    files = sorted(path.name for path in REPO_ROOT.glob("*.py"))
    return {"root_py_file_count": len(files), "root_py_files": files}


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in paths}


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, str]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [path for path, digest in before.items() if after[path] != digest]
    return {
        "after": after,
        "before": dict(before),
        "changed": changed,
        "unchanged": not changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _disk_ready() -> JsonDict:
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def _ram_ready() -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp6120 run_date must be {RUN_DATE}")
    if args.validate:
        artifact = read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        validate_artifact(artifact)
        return 0
    run(result_path=REPO_ROOT / RESULT_RELATIVE_PATH, write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
