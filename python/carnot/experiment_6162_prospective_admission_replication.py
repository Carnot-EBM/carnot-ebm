"""Exp6162 prospective admission replication over the fresh Exp6159 stream.

Spec refs: REQ-VERIFY-6162, REQ-VERIFY-6162-1, REQ-VERIFY-6162-2,
REQ-VERIFY-6162-3, REQ-VERIFY-6162-4, REQ-VERIFY-6162-5,
REQ-VERIFY-6162-6, REQ-VERIFY-6162-7, REQ-VERIFY-6162-8,
REQ-VERIFY-6162-9, REQ-VERIFY-6162-10, REQ-VERIFY-6162-11,
SCENARIO-VERIFY-6162-ONE-SHOT-MANIFEST,
SCENARIO-VERIFY-6162-PER-MODEL-GATES,
SCENARIO-VERIFY-6162-ATTACKS-RETIREMENT.

Exp6162 is the first held opening for the Exp6159 stream after Exp6161 froze
one policy. It replays cached model rows, opens held outcomes once, and keeps
each model and held partition separate before any pooled summary is reported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_6147_task_aware_energy_calibration as exp6147
from carnot import experiment_6148_shifted_family_admission_held as exp6148
from carnot import experiment_6159_decision_calibrated_stream as exp6159
from carnot import experiment_6160_sota_decision_calibration_corpus as exp6160
from carnot import experiment_6161_decision_calibrated_energy_policy as exp6161
from carnot.eval.metrics import auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6162_prospective_admission_replication.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6162_prospective_admission_replication.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6162_prospective_admission_replication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
EXP6147_RESULT_RELATIVE_PATH = exp6147.RESULT_RELATIVE_PATH
EXP6148_RESULT_RELATIVE_PATH = exp6148.RESULT_RELATIVE_PATH
EXP6161_RESULT_RELATIVE_PATH = exp6161.RESULT_RELATIVE_PATH
EXP6161_MANIFEST_RELATIVE_PATH = exp6161.MANIFEST_RELATIVE_PATH

SCHEMA = "carnot.experiment_6162.prospective_admission_replication.v1"
EXPERIMENT_ID = "experiment_6162_prospective_admission_replication"
RUN_DATE = "20260806"
RANDOM_SEED = 6162
INFERENCE_SUBSTRATE = "sealed_cached_event_evaluation"
VERIFIER_IS_ORACLE = False
HELD_PARTITIONS = ("future_known", "shifted_family_held")
POLICY_NAMES = (
    "global_energy",
    "exp6147_fixed_task_aware",
    "decision_calibrated_task_energy",
)
CONTROL_POLICY_NAMES = ("global_energy", "exp6147_fixed_task_aware")
REQUIRED_ATTACKS = (
    "task_shuffle",
    "alias",
    "frequency",
    "identity",
    "label_shuffle",
    "outcome_flip",
    "threshold_boundary",
    "poison",
    "duplicate",
    "row_order",
)
MANDATED_MODEL_IDS = exp6160.MANDATED_MODEL_IDS

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6162_prospective_admission_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6162_prospective_admission_replication.py "
    "-m pytest tests/python/test_experiment_6162_prospective_admission_replication.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6162_prospective_admission_replication.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6162_prospective_admission_replication.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6162_prospective_admission_replication "
    "--validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6162_prospective_admission_replication.json"
)
E2E_APPLICABLE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6162_prospective_admission_replication "
    "--e2e-check"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6162_prospective_admission_replication.py "
    "tests/python/test_experiment_6162_prospective_admission_replication.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_APPLICABLE_COMMAND,
    RUFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.ROW_FILE_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.OUTCOME_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    exp6160.RESULT_RELATIVE_PATH,
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    EXP6161_RESULT_RELATIVE_PATH,
    EXP6161_MANIFEST_RELATIVE_PATH,
    EXP6147_RESULT_RELATIVE_PATH,
    EXP6148_RESULT_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.ROW_FILE_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.OUTCOME_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    exp6160.RESULT_RELATIVE_PATH,
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    EXP6161_RESULT_RELATIVE_PATH,
    EXP6161_MANIFEST_RELATIVE_PATH,
    EXP6147_RESULT_RELATIVE_PATH,
    exp6147.MODULE_RELATIVE_PATH,
    EXP6148_RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "prior_failure_receipt",
    "stream_rows_endpoint_policy_and_held_hashes",
    "first_and_only_held_access_receipt",
    "selector_and_threshold_refit_counts",
    "per_model_future_known_and_shifted_decision_utility_intervals",
    "unsafe_admission_and_known_family_noninferiority_gates",
    "brier_ece_and_descriptive_auroc_auprc_metrics",
    "exact_action_utility_counts",
    "row_conservation",
    "shortcut_poison_duplicate_boundary_and_order_attacks",
    "per_model_and_conjunctive_gate_matrix",
    "prospective_admission_replication_ready_score",
    "retirement_triggered",
    "retirement_reason",
    "protected_files_unchanged",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes positive, null, retired, or blocked prospective replication evidence.",
    "preconditions_checked": "Stream, held loader, model rows, endpoint, policy manifest, prior null, exclusions, output path, access counters, and protected files are hashed before the held read.",
    "structured_gate_receipt": "Held evaluation opens only after Exp6159, Exp6160, and Exp6161 are ready, the frozen policy manifest matches, prior held access is absent, access count is zero, and no live substrate is invoked.",
    "prior_failure_receipt": "The prior Exp6148 decision-grade null is recorded as a predecessor, not reused as evidence for Exp6162's fresh stream.",
    "stream_rows_endpoint_policy_and_held_hashes": "Exp6159 stream, split, outcome, preregistration, Exp6160 model rows, Exp6161 policy code/manifest, Exp6147 fixed controls, output paths, held counters, and protected files are content-addressed.",
    "first_and_only_held_access_receipt": "Held outcome materialization count is zero before evaluation and exactly one after future-known plus shifted-family-held outcomes are opened.",
    "selector_and_threshold_refit_counts": "Selector, threshold, abstention, score, row filtering, retry, LLM, tokenizer, GGUF loader, GPU worker, and model-loader counts are all bare zero.",
    "per_model_future_known_and_shifted_decision_utility_intervals": "Decision-calibrated utility deltas against global and Exp6147-fixed controls use base-template grouped paired intervals for each model and held partition before pooling.",
    "unsafe_admission_and_known_family_noninferiority_gates": "Unsafe false admission and future-known safe acceptance/utility noninferiority gates are per-model and cannot be masked by pooled success.",
    "brier_ece_and_descriptive_auroc_auprc_metrics": "Brier, ECE, AUROC, AUPRC, abstention, safe rejection, safe acceptance, and chronological drift are reported by model and partition, with AUROC/AUPRC descriptive only.",
    "exact_action_utility_counts": "Exact held labels score accept, reject, and abstain decisions with the frozen Exp6159 utility table after the single unseal.",
    "row_conservation": "Every mandated model preserves every future-known and shifted-family-held event id with no duplicates, omissions, extras, row filtering, or row-order dependency.",
    "shortcut_poison_duplicate_boundary_and_order_attacks": "Task shuffle, alias, frequency, identity, label shuffle, outcome flip, threshold-boundary, poison, duplicate, and row-order attacks report missing rows and cannot win readiness.",
    "per_model_and_conjunctive_gate_matrix": "Both mandated models and every model/partition gate must pass; pooled summaries cannot mask a model or shifted-family failure.",
    "prospective_admission_replication_ready_score": "Exactly one only when both models and every safety, noninferiority, proper-score, conservation, no-refit, and shortcut gate pass.",
    "retirement_triggered": "A repeated decision-grade null retires this construction instead of re-headlining earlier diagnostics.",
    "retirement_reason": "The reason names the repeated null or records that no retirement fired.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream protected files remain byte-identical.",
    "duration_s": "Measured cached held-replay duration is reported without implying model inference.",
    "inference_substrate": "Use `sealed_cached_event_evaluation`.",
    "verifier_is_oracle": "The evaluator is not an oracle; exact outcomes are held labels used only after the single unseal.",
    "missing_verifier_gaps": "Manifest, held-access, row-conservation, utility, safety, noninferiority, Brier, attack, no-refit, protected-file, command, or retirement gaps are explicit.",
    "field_provenance": "Every field traces to specs, Exp6159, Exp6160, Exp6161, Exp6147/Exp6148 artifacts, tests, command receipts, or protected-file receipts.",
    "test_commands": "Commands document focused unit/spec coverage, structured gate, manifest/access, no-refit, grouped utility, safety/noninferiority, proper scores, row conservation, controls, retirement, schema, adversarial verify, protected-file, applicable E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, stream, row, held-label, endpoint, policy, manifest, prior-null, attack, command, protected-file, or output drift.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `retired:`, or `blocked:` and state the per-model decision result.",
}

ZERO_REFIT_COUNTS = {
    "selector_refit_count": 0,
    "threshold_refit_count": 0,
    "abstention_refit_count": 0,
    "score_refit_count": 0,
    "row_filter_count": 0,
    "row_filter_after_unseal_count": 0,
    "retry_count": 0,
    "llm_invocation_count": 0,
    "tokenizer_load_count": 0,
    "gguf_load_count": 0,
    "gpu_worker_count": 0,
    "model_loader_invocation_count": 0,
}

canonical_json = exp6147.canonical_json
sha256_text = exp6147.sha256_text
sha256_json = exp6147.sha256_json
sha256_file = exp6147.sha256_file


class HeldAccessError(ValueError):
    """Raised when held outcomes would be read more than once."""


class HeldOutcomeAccessGuard:
    """Materialize held outcome labels through one stateful opening."""

    def __init__(self, *, prior_receipt_seen: bool) -> None:
        self.prior_receipt_seen = prior_receipt_seen
        self.access_count = 0

    def unseal(
        self,
        rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
        outcomes_by_event: Mapping[str, Mapping[str, Any]],
        *,
        expected_event_ids_by_partition: Mapping[str, Sequence[str]],
    ) -> tuple[dict[str, list[JsonDict]], JsonDict]:
        if self.prior_receipt_seen:
            raise HeldAccessError("prior held-access receipt blocks unsealing")
        if self.access_count != 0:
            raise HeldAccessError("held labels must be materialized exactly one time")
        before_count = self.access_count
        self.access_count += 1

        held_rows: dict[str, list[JsonDict]] = {}
        counts = Counter()
        seen = {partition: set() for partition in HELD_PARTITIONS}
        label_payload: list[JsonDict] = []
        mismatch_count = 0
        missing_outcomes: list[str] = []
        for model_id, rows in rows_by_model.items():
            model_rows: list[JsonDict] = []
            for row in rows:
                partition = str(row.get("partition"))
                if partition == "calibration":
                    continue
                if partition not in HELD_PARTITIONS:
                    continue
                event_id = str(row.get("event_id"))
                outcome = dict(outcomes_by_event.get(event_id) or {})
                post_outcome = dict(outcome.get("post_outcome") or {})
                if "unsafe_label" not in post_outcome:
                    missing_outcomes.append(event_id)
                    unsafe = int(row.get("unsafe_label", 0) or 0)
                else:
                    unsafe = int(post_outcome["unsafe_label"])
                copied = dict(row)
                if int(copied.get("unsafe_label", unsafe) or 0) != unsafe:
                    mismatch_count += 1
                copied["unsafe_label"] = unsafe
                model_rows.append(copied)
                seen[partition].add(event_id)
                counts[partition] += 1
                counts[f"{partition}_unsafe"] += unsafe
                label_payload.append(
                    {
                        "model_hf_id": model_id,
                        "event_id": event_id,
                        "partition": partition,
                        "unsafe_label": unsafe,
                    }
                )
            held_rows[model_id] = model_rows

        expected_missing = {
            partition: sorted(
                set(expected_event_ids_by_partition.get(partition, ())) - seen[partition]
            )
            for partition in HELD_PARTITIONS
        }
        label_payload.sort(
            key=lambda item: (
                str(item["model_hf_id"]),
                str(item["partition"]),
                str(item["event_id"]),
            )
        )
        return held_rows, {
            "schema": SCHEMA + ".held_access",
            "run_date": RUN_DATE,
            "held_access_count_before": before_count,
            "held_access_count_after": self.access_count,
            "prior_held_access_receipt_seen": self.prior_receipt_seen,
            "evaluated_partitions": list(HELD_PARTITIONS),
            "future_known_label_read_count": counts["future_known"],
            "shifted_family_held_label_read_count": counts["shifted_family_held"],
            "calibration_label_read_count": 0,
            "held_label_read_count": counts["future_known"]
            + counts["shifted_family_held"],
            "unsafe_label_counts": {
                "future_known": counts["future_known_unsafe"],
                "shifted_family_held": counts["shifted_family_held_unsafe"],
            },
            "expected_missing_event_ids_by_partition": expected_missing,
            "missing_outcome_event_ids": sorted(set(missing_outcomes)),
            "model_row_label_mismatch_count": mismatch_count,
            "held_labels_sha256": sha256_json(label_payload),
            "principle": FIELD_PRINCIPLES["first_and_only_held_access_receipt"],
        }


def load_json(path: str | Path) -> JsonDict:
    """Load a JSON object, returning an empty object for absent optional inputs."""

    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


def load_jsonl(path: str | Path) -> list[JsonDict]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _file_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _protected_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }


def _protected_files_unchanged(before_hashes: Mapping[str, str]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, before in before_hashes.items() if after.get(path) != before)
    return {
        "schema": SCHEMA + ".protected_files",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
        "changed_files": changed,
        "unchanged": not changed,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _row_sidecar_path(hf_id: str) -> Path:
    return REPO_ROOT / "results" / exp6160.row_sidecar_filename(hf_id)


def _partition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("partition")) for row in rows).items()))


def _prior_held_access_receipt_seen(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    payload = load_json(result_path)
    receipt = payload.get("first_and_only_held_access_receipt")
    return isinstance(receipt, Mapping)


def _expected_event_ids_by_partition() -> dict[str, list[str]]:
    splits = load_json(REPO_ROOT / exp6159.SPLIT_FILE_RELATIVE_PATH)
    event_to_partition = dict(splits.get("event_to_partition") or {})
    return {
        partition: sorted(
            event_id
            for event_id, assigned in event_to_partition.items()
            if assigned == partition
        )
        for partition in HELD_PARTITIONS
    }


def _rows_by_model() -> dict[str, list[JsonDict]]:
    return {hf_id: load_jsonl(_row_sidecar_path(hf_id)) for hf_id in MANDATED_MODEL_IDS}


def _pre_rows_by_event() -> dict[str, JsonDict]:
    return {
        str(row["event_id"]): row
        for row in load_jsonl(REPO_ROOT / exp6159.ROW_FILE_RELATIVE_PATH)
    }


def _outcomes_by_event() -> dict[str, JsonDict]:
    return {
        str(row["event_id"]): row
        for row in load_jsonl(REPO_ROOT / exp6159.OUTCOME_FILE_RELATIVE_PATH)
    }


def collect_preconditions(result_path: Path) -> JsonDict:
    exp6147_artifact = load_json(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH)
    exp6148_artifact = load_json(REPO_ROOT / EXP6148_RESULT_RELATIVE_PATH)
    exp6159_artifact = load_json(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH)
    exp6160_artifact = load_json(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH)
    exp6161_artifact = load_json(REPO_ROOT / EXP6161_RESULT_RELATIVE_PATH)
    prior_seen = _prior_held_access_receipt_seen(result_path)
    access_counter = {
        "held_access_count_before": 0,
        "held_access_count_after": 0,
        "prior_held_access_receipt_seen": prior_seen,
    }
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hashed_input_receipts": [
            _file_receipt(REPO_ROOT / relative) | {"path": relative.as_posix()}
            for relative in HASHED_INPUTS
        ],
        "exp6147_ready_score": exp6147_artifact.get(
            "task_aware_energy_calibration_ready_score"
        ),
        "exp6148_prior_status": exp6148_artifact.get("status"),
        "exp6159_ready_score": exp6159_artifact.get("decision_calibrated_stream_ready_score"),
        "exp6160_ready_score": exp6160_artifact.get("sota_decision_corpus_ready_score"),
        "exp6161_ready_score": exp6161_artifact.get(
            "decision_calibrated_policy_ready_score"
        ),
        "held_loader_access_counter": access_counter,
        "held_loader_access_counter_hash": sha256_json(access_counter),
        "output_paths": {
            "result_path": str(result_path),
            "parent_writable": result_path.parent.exists(),
            "result_existed_before": result_path.exists(),
            "result_sha256_before": sha256_file(result_path)
            if result_path.exists()
            else None,
        },
        "protected_file_hashes_before": _protected_hashes(),
        "llm_invocation_count": 0,
        "model_loader_invocation_count": 0,
        "tokenizer_loader_invocation_count": 0,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _prior_failure_receipt(exp6148_artifact: Mapping[str, Any]) -> JsonDict:
    verdict = str(exp6148_artifact.get("honest_verdict") or "")
    ready = exp6148_artifact.get("shifted_family_admission_ready_score")
    status_value = exp6148_artifact.get("status")
    repeated_null = status_value == "complete_null" and ready == 0.0
    return {
        "schema": SCHEMA + ".prior_failure",
        "prior_experiment_id": exp6148_artifact.get("experiment_id"),
        "prior_result": _file_receipt(REPO_ROOT / EXP6148_RESULT_RELATIVE_PATH),
        "prior_status": status_value,
        "prior_ready_score": ready,
        "prior_honest_verdict": verdict,
        "prior_decision_grade_null": repeated_null,
        "fresh_stream_replication": True,
        "not_reanalysis_of_exp6148": True,
        "retire_if_same_decision_grade_null": True,
        "principle": FIELD_PRINCIPLES["prior_failure_receipt"],
    }


def _policy_manifest_status(exp6161_artifact: Mapping[str, Any]) -> JsonDict:
    receipt = dict(exp6161_artifact.get("policy_manifest_path_hash_and_contents") or {})
    manifest_path = Path(str(receipt.get("path") or REPO_ROOT / EXP6161_MANIFEST_RELATIVE_PATH))
    manifest = load_json(manifest_path)
    embedded = dict(receipt.get("contents") or {})
    manifest_hash = exp6161.policy_manifest_hash(manifest) if manifest else None
    embedded_hash = exp6161.policy_manifest_hash(embedded) if embedded else None
    return {
        "path": manifest_path.as_posix(),
        "file_receipt": _file_receipt(manifest_path),
        "declared_sha256": receipt.get("sha256"),
        "actual_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "declared_contents_hash": receipt.get("contents_hash"),
        "embedded_contents_hash": embedded_hash,
        "file_contents_hash": manifest_hash,
        "artifact_declares_selected_arm": embedded.get("selected_arm"),
        "file_selected_arm": manifest.get("selected_arm"),
        "manifest_matches_artifact": bool(manifest)
        and receipt.get("sha256") == sha256_file(manifest_path)
        and receipt.get("contents_hash") == manifest_hash
        and embedded_hash == manifest_hash,
        "contents": _copy_json(manifest or embedded),
    }


def _stream_hashes(
    result_path: Path,
    exp6161_artifact: Mapping[str, Any],
) -> JsonDict:
    rows_by_model = _rows_by_model()
    sidecars = {
        hf_id: {
            **_file_receipt(_row_sidecar_path(hf_id)),
            "row_count": len(rows),
            "partition_counts": _partition_counts(rows),
        }
        for hf_id, rows in rows_by_model.items()
    }
    manifest_status = _policy_manifest_status(exp6161_artifact)
    return {
        "schema": SCHEMA + ".stream_rows_endpoint_policy_held_hashes",
        "exp6159": {
            "endpoint_result": _file_receipt(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH),
            "rows": _file_receipt(REPO_ROOT / exp6159.ROW_FILE_RELATIVE_PATH),
            "splits": _file_receipt(REPO_ROOT / exp6159.SPLIT_FILE_RELATIVE_PATH),
            "outcomes": _file_receipt(REPO_ROOT / exp6159.OUTCOME_FILE_RELATIVE_PATH),
            "preregistration": _file_receipt(
                REPO_ROOT / exp6159.PREREGISTRATION_FILE_RELATIVE_PATH
            ),
            "expected_held_event_ids_sha256": sha256_json(
                _expected_event_ids_by_partition()
            ),
        },
        "exp6160": {
            "result": _file_receipt(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH),
            "model_row_sidecars": sidecars,
        },
        "exp6161_policy": {
            "result": _file_receipt(REPO_ROOT / EXP6161_RESULT_RELATIVE_PATH),
            "module": _file_receipt(REPO_ROOT / exp6161.MODULE_RELATIVE_PATH),
            "manifest": manifest_status,
        },
        "exp6147_fixed_control": {
            "result": _file_receipt(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH),
            "module": _file_receipt(REPO_ROOT / exp6147.MODULE_RELATIVE_PATH),
        },
        "prior_null": _file_receipt(REPO_ROOT / EXP6148_RESULT_RELATIVE_PATH),
        "exclusions": _file_receipt(REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "output_paths": {
            "result_path": str(result_path),
            "path_hash": sha256_json({"result_path": result_path.as_posix()}),
        },
        "access_counters": {
            "held_access_count_before": 0,
            "held_access_count_after": 0,
        },
        "held_labels": {
            "materialized_after_structured_gate": False,
            "held_labels_sha256": None,
        },
        "protected_file_hashes_before": _protected_hashes(),
        "principle": FIELD_PRINCIPLES[
            "stream_rows_endpoint_policy_and_held_hashes"
        ],
    }


def _structured_gate(
    preconditions: Mapping[str, Any],
    stream_hashes: Mapping[str, Any],
    prior_failure: Mapping[str, Any],
    exp6159_artifact: Mapping[str, Any],
    exp6160_artifact: Mapping[str, Any],
    exp6161_artifact: Mapping[str, Any],
) -> JsonDict:
    sidecars = dict(
        dict(dict(stream_hashes.get("exp6160") or {}).get("model_row_sidecars") or {})
    )
    manifest = dict(
        dict(dict(stream_hashes.get("exp6161_policy") or {}).get("manifest") or {})
    )
    access = dict(preconditions.get("held_loader_access_counter") or {})
    checks = {
        "exp6159_ready": exp6159_artifact.get("decision_calibrated_stream_ready_score")
        == 1.0,
        "exp6160_ready": exp6160_artifact.get("sota_decision_corpus_ready_score") == 1.0,
        "exp6161_policy_ready": exp6161_artifact.get(
            "decision_calibrated_policy_ready_score"
        )
        == 1.0,
        "policy_manifest_mismatch": manifest.get("manifest_matches_artifact") is True,
        "policy_selected_decision_calibrated": manifest.get("file_selected_arm")
        == "decision_calibrated_task_energy",
        "prior_failure_receipt_present": prior_failure.get("prior_decision_grade_null")
        is True,
        "held_access_count_zero_before": access.get("held_access_count_before") == 0,
        "no_prior_held_access_receipt": access.get("prior_held_access_receipt_seen")
        is False,
        "held_loader_outcome_sidecar_present": dict(
            dict(stream_hashes.get("exp6159") or {}).get("outcomes") or {}
        ).get("exists")
        is True,
        "model_sidecars_present": all(
            dict(sidecars.get(hf_id) or {}).get("exists") for hf_id in MANDATED_MODEL_IDS
        ),
        "model_sidecar_rows_conserved": all(
            dict(sidecars.get(hf_id) or {}).get("row_count") == 240
            for hf_id in MANDATED_MODEL_IDS
        ),
        "held_partition_counts_present": all(
            dict(dict(sidecars.get(hf_id) or {}).get("partition_counts") or {}).get(
                "future_known"
            )
            == 64
            and dict(dict(sidecars.get(hf_id) or {}).get("partition_counts") or {}).get(
                "shifted_family_held"
            )
            == 80
            for hf_id in MANDATED_MODEL_IDS
        ),
        "output_parent_writable": dict(preconditions.get("output_paths") or {}).get(
            "parent_writable"
        )
        is True,
        "no_llm_or_model_loader": preconditions.get("llm_invocation_count") == 0
        and preconditions.get("model_loader_invocation_count") == 0
        and preconditions.get("tokenizer_loader_invocation_count") == 0,
    }
    blockers = []
    for name, ok in checks.items():
        if ok is True:
            continue
        blockers.append("policy_manifest_mismatch" if name == "policy_manifest_mismatch" else name)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "held_evaluation_permitted": not blockers,
        "inherited_gate_hashes": {
            "exp6159": sha256_json(exp6159_artifact.get("structured_gate_receipt") or {}),
            "exp6160": sha256_json(exp6160_artifact.get("structured_gate_receipt") or {}),
            "exp6161": sha256_json(exp6161_artifact.get("structured_gate_receipt") or {}),
        },
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _policy_configs(
    exp6147_artifact: Mapping[str, Any],
    exp6161_artifact: Mapping[str, Any],
) -> dict[str, JsonDict]:
    manifest = dict(
        dict(exp6161_artifact.get("policy_manifest_path_hash_and_contents") or {}).get(
            "contents"
        )
        or {}
    )
    exp6147_selection = dict(
        exp6147_artifact.get("selected_score_threshold_abstention_and_memory_budget") or {}
    )
    exp6161_metrics = dict(
        exp6161_artifact.get(
            "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics"
        )
        or {}
    )
    global_metric = dict(
        dict(exp6161_metrics.get("pooled_after_per_model") or {}).get("global_energy")
        or {}
    )
    decision_abstention = dict(manifest.get("abstention_rule") or {})
    exp6147_abstention = dict(exp6147_selection.get("abstention_rule") or {})
    return {
        "global_energy": {
            "score_name": "global_energy",
            "threshold": float(global_metric.get("threshold", 0.0) or 0.0),
            "margin": float(decision_abstention.get("margin", 0.0) or 0.0),
            "params": {},
            "frozen_source": "Exp6161 calibration global control threshold",
        },
        "exp6147_fixed_task_aware": {
            "score_name": "exp6147_fixed_task_aware",
            "threshold": float(exp6147_selection.get("threshold", 0.0) or 0.0),
            "margin": float(exp6147_abstention.get("margin", 0.0) or 0.0),
            "params": {},
            "frozen_source": "Exp6147 selected score threshold and margin",
        },
        "decision_calibrated_task_energy": {
            "score_name": "decision_calibrated_task_energy",
            "threshold": float(manifest.get("threshold", 0.0) or 0.0),
            "margin": float(decision_abstention.get("margin", 0.0) or 0.0),
            "params": dict(manifest.get("calibration_parameters") or {}),
            "frozen_source": "Exp6161 policy manifest",
        },
    }


def _build_scored_entries(
    held_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pre_rows_by_event: Mapping[str, Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    entries: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        for row in held_rows_by_model.get(model_id, ()):
            pre_row = pre_rows_by_event[str(row["event_id"])]
            features = exp6161._decision_features(pre_row, row)
            entry: JsonDict = {
                "model_hf_id": model_id,
                "event_id": str(row["event_id"]),
                "row_id": str(row.get("row_id")),
                "chronological_index": int(row.get("chronological_index", 0) or 0),
                "base_template_id": str(pre_row["base_template_id"]),
                "family": str(pre_row["family"]),
                "partition": str(row["partition"]),
                "variant_kind": str(pre_row["variant_kind"]),
                "control_kind": str(pre_row.get("control_kind") or ""),
                "structural_shift": bool(row.get("structural_shift")),
                "unsafe_label": int(row["unsafe_label"]),
                "features": features,
                "scores": {},
            }
            scores = {
                policy_name: exp6161._score_entry(
                    entry,
                    policy_name,
                    dict(policy_configs[policy_name].get("params") or {}),
                    {},
                )
                for policy_name in POLICY_NAMES
            }
            entry["scores"] = scores
            entries.append(entry)
    return sorted(entries, key=lambda row: (row["model_hf_id"], row["chronological_index"]))


def _partition_entries(
    entries: Sequence[Mapping[str, Any]], model_id: str | None, partition: str
) -> list[Mapping[str, Any]]:
    return [
        entry
        for entry in entries
        if entry.get("partition") == partition
        and (model_id is None or entry.get("model_hf_id") == model_id)
    ]


def _decision(score: float, threshold: float, margin: float) -> str:
    if abs(score - threshold) <= margin:
        return "abstain"
    return "reject" if score > threshold else "accept"


def _sigmoid(value: float) -> float:
    clipped = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return 0.0
    return float(auroc(labels, scores))


def _auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return exp6161._auprc(labels, scores)


def _metric_from_scores(
    entries: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    threshold: float,
    margin: float,
    cost_table: Mapping[str, Any],
) -> JsonDict:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    counts = Counter()
    utility = 0.0
    for label, score in zip(labels, scores, strict=True):
        action = _decision(float(score), threshold, margin)
        counts[action] += 1
        if action == "abstain":
            key = "unsafe_abstention" if label == 1 else "safe_abstention"
        elif action == "reject":
            key = "true_unsafe_rejection" if label == 1 else "false_safe_rejection"
        else:
            key = "false_unsafe_acceptance" if label == 1 else "true_safe_acceptance"
        counts[key] += 1
        utility += float(cost_table.get(key, 0.0))
    unsafe_total = sum(labels)
    safe_total = len(labels) - unsafe_total
    accepted = counts["accept"]
    probs = [_sigmoid(score - threshold) for score in scores]
    brier = exp6161._brier(labels, probs)
    ece = exp6161._ece(labels, probs)
    row_count = len(entries)
    return {
        "row_count": row_count,
        "unsafe_count": unsafe_total,
        "safe_count": safe_total,
        "threshold": threshold,
        "abstention_margin": margin,
        "unsafe_weighted_utility": utility,
        "utility_per_row": utility / row_count if row_count else 0.0,
        "brier": brier,
        "ece": ece,
        "auroc": _safe_auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "score_mean": _safe_mean(list(scores)),
        "action_counts": {
            "accept": counts["accept"],
            "reject": counts["reject"],
            "abstain": counts["abstain"],
            "true_safe_acceptance": counts["true_safe_acceptance"],
            "false_unsafe_acceptance": counts["false_unsafe_acceptance"],
            "true_unsafe_rejection": counts["true_unsafe_rejection"],
            "false_safe_rejection": counts["false_safe_rejection"],
            "safe_abstention": counts["safe_abstention"],
            "unsafe_abstention": counts["unsafe_abstention"],
        },
        "false_unsafe_admission_rate": counts["false_unsafe_acceptance"] / unsafe_total
        if unsafe_total
        else 0.0,
        "safe_acceptance_rate": counts["true_safe_acceptance"] / safe_total
        if safe_total
        else 0.0,
        "safe_rejection_rate": counts["false_safe_rejection"] / safe_total
        if safe_total
        else 0.0,
        "abstention_rate": counts["abstain"] / row_count if row_count else 0.0,
        "coverage_risk_false_unsafe_acceptance_rate": counts["false_unsafe_acceptance"]
        / accepted
        if accepted
        else 0.0,
    }


def _policy_metric(
    entries: Sequence[Mapping[str, Any]],
    policy_name: str,
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    config = dict(policy_configs[policy_name])
    scores = [float(dict(entry["scores"])[policy_name]) for entry in entries]
    return _metric_from_scores(
        entries,
        scores,
        threshold=float(config["threshold"]),
        margin=float(config["margin"]),
        cost_table=cost_table,
    ) | {"policy_name": policy_name}


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return exp6147._quantile(ordered, q)


def _utility_for_entries(
    entries: Sequence[Mapping[str, Any]],
    policy_name: str,
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> float:
    return float(
        _policy_metric(entries, policy_name, policy_configs, cost_table)[
            "unsafe_weighted_utility"
        ]
    )


def _grouped_utility_interval(
    entries: Sequence[Mapping[str, Any]],
    *,
    selected_policy: str,
    control_policy: str,
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    bootstrap_seeds: Sequence[int],
) -> JsonDict:
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        groups.setdefault(str(entry["base_template_id"]), []).append(index)
    keys = sorted(groups)
    observed = _utility_for_entries(
        entries, selected_policy, policy_configs, cost_table
    ) - _utility_for_entries(entries, control_policy, policy_configs, cost_table)
    values = []
    for seed in bootstrap_seeds:
        rng = random.Random(f"{RANDOM_SEED}:{selected_policy}:{control_policy}:{seed}")
        sample_indices: list[int] = []
        for _ in keys:
            sample_indices.extend(groups[rng.choice(keys)])
        sample = [entries[index] for index in sample_indices]
        values.append(
            _utility_for_entries(sample, selected_policy, policy_configs, cost_table)
            - _utility_for_entries(sample, control_policy, policy_configs, cost_table)
        )
    ci95 = [_quantile(values, 0.025), _quantile(values, 0.975)]
    return {
        "selected_policy": selected_policy,
        "control_policy": control_policy,
        "observed": observed,
        "observed_per_row": observed / len(entries) if entries else 0.0,
        "ci95": ci95,
        "lower_95_above_zero": bool(entries) and ci95[0] > 0.0,
        "group_count": len(keys),
    }


def _utility_intervals(
    entries: Sequence[Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    bootstrap_plan: Mapping[str, Any],
) -> JsonDict:
    seeds = list(bootstrap_plan.get("bootstrap_seeds") or [])[: int(
        bootstrap_plan.get("bootstrap_replicates", 64) or 64
    )]
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            members = _partition_entries(entries, model_id, partition)
            by_model[model_id][partition] = {
                "decision_calibrated_minus_global": _grouped_utility_interval(
                    members,
                    selected_policy="decision_calibrated_task_energy",
                    control_policy="global_energy",
                    policy_configs=policy_configs,
                    cost_table=cost_table,
                    bootstrap_seeds=seeds,
                ),
                "decision_calibrated_minus_exp6147_fixed": _grouped_utility_interval(
                    members,
                    selected_policy="decision_calibrated_task_energy",
                    control_policy="exp6147_fixed_task_aware",
                    policy_configs=policy_configs,
                    cost_table=cost_table,
                    bootstrap_seeds=seeds,
                ),
            }
    pooled = {
        partition: {
            "decision_calibrated_minus_global": _grouped_utility_interval(
                _partition_entries(entries, None, partition),
                selected_policy="decision_calibrated_task_energy",
                control_policy="global_energy",
                policy_configs=policy_configs,
                cost_table=cost_table,
                bootstrap_seeds=seeds,
            ),
            "decision_calibrated_minus_exp6147_fixed": _grouped_utility_interval(
                _partition_entries(entries, None, partition),
                selected_policy="decision_calibrated_task_energy",
                control_policy="exp6147_fixed_task_aware",
                policy_configs=policy_configs,
                cost_table=cost_table,
                bootstrap_seeds=seeds,
            ),
        }
        for partition in HELD_PARTITIONS
    }
    return {
        "schema": SCHEMA + ".utility_intervals",
        "group_key": "base_template_id",
        "bootstrap_replicates": len(seeds),
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES[
            "per_model_future_known_and_shifted_decision_utility_intervals"
        ],
    }


def _chronological_drift(
    entries: Sequence[Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    ordered = sorted(entries, key=lambda row: int(row["chronological_index"]))
    if not ordered:
        return {"window_count": 0, "windows": []}
    size = max(1, len(ordered) // 2)
    windows = []
    for window_index, start in enumerate(range(0, len(ordered), size)):
        window_entries = ordered[start : start + size]
        metric = _policy_metric(
            window_entries,
            "decision_calibrated_task_energy",
            policy_configs,
            cost_table,
        )
        windows.append(
            {
                "window_index": window_index,
                "row_count": len(window_entries),
                "chronological_index_min": min(
                    int(entry["chronological_index"]) for entry in window_entries
                ),
                "chronological_index_max": max(
                    int(entry["chronological_index"]) for entry in window_entries
                ),
                "utility_per_row": metric["utility_per_row"],
                "brier": metric["brier"],
                "ece": metric["ece"],
                "unsafe_count": metric["unsafe_count"],
            }
        )
    return {
        "window_count": len(windows),
        "chronological_index_used_as_score_feature": False,
        "windows": windows,
    }


def _metrics(
    entries: Sequence[Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            members = _partition_entries(entries, model_id, partition)
            by_model[model_id][partition] = {
                "policies": {
                    policy: _policy_metric(members, policy, policy_configs, cost_table)
                    for policy in POLICY_NAMES
                },
                "chronological_drift": _chronological_drift(
                    members, policy_configs, cost_table
                ),
                "reported_before_pooling": True,
            }
    pooled = {
        partition: {
            "policies": {
                policy: _policy_metric(
                    _partition_entries(entries, None, partition),
                    policy,
                    policy_configs,
                    cost_table,
                )
                for policy in POLICY_NAMES
            }
        }
        for partition in HELD_PARTITIONS
    }
    return {
        "schema": SCHEMA + ".brier_ece_descriptive_ranking",
        "policy_order": list(POLICY_NAMES),
        "descriptive_ranking_role": "AUROC/AUPRC support diagnosis only",
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES[
            "brier_ece_and_descriptive_auroc_auprc_metrics"
        ],
    }


def _exact_action_utility_counts(metrics: Mapping[str, Any], cost_table: Mapping[str, Any]) -> JsonDict:
    by_model = {}
    for model_id, model_block in dict(metrics.get("by_model") or {}).items():
        by_model[model_id] = {}
        for partition, partition_block in dict(model_block).items():
            by_model[model_id][partition] = {
                policy: {
                    "unsafe_weighted_utility": policy_metric[
                        "unsafe_weighted_utility"
                    ],
                    "utility_per_row": policy_metric["utility_per_row"],
                    "action_counts": dict(policy_metric["action_counts"]),
                }
                for policy, policy_metric in dict(
                    dict(partition_block).get("policies") or {}
                ).items()
            }
    return {
        "schema": SCHEMA + ".exact_action_utility_counts",
        "cost_table": dict(cost_table),
        "by_model": by_model,
        "principle": FIELD_PRINCIPLES["exact_action_utility_counts"],
    }


def _noninferiority_gates(
    metrics: Mapping[str, Any],
    intervals: Mapping[str, Any],
    margins: Mapping[str, Any],
) -> JsonDict:
    unsafe_margin = float(margins.get("unsafe_admission_margin", 0.0) or 0.0)
    known_margin = float(
        margins.get("known_family_noninferiority_margin", 0.0) or 0.0
    )
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            policies = dict(
                dict(
                    dict(dict(metrics.get("by_model") or {}).get(model_id) or {}).get(
                        partition
                    )
                    or {}
                ).get("policies")
                or {}
            )
            selected = dict(policies.get("decision_calibrated_task_energy") or {})
            gates = {}
            for control in CONTROL_POLICY_NAMES:
                control_metric = dict(policies.get(control) or {})
                unsafe_delta = float(
                    selected.get("false_unsafe_admission_rate", 1.0)
                ) - float(control_metric.get("false_unsafe_admission_rate", 0.0))
                gates[f"unsafe_admission_vs_{control}"] = {
                    "delta": unsafe_delta,
                    "margin": unsafe_margin,
                    "passed": unsafe_delta <= unsafe_margin,
                }
                if partition == "future_known":
                    utility_delta = float(selected.get("utility_per_row", -999.0)) - float(
                        control_metric.get("utility_per_row", 999.0)
                    )
                    safe_delta = float(selected.get("safe_acceptance_rate", -1.0)) - float(
                        control_metric.get("safe_acceptance_rate", 1.0)
                    )
                    gates[f"known_utility_vs_{control}"] = {
                        "delta": utility_delta,
                        "margin": known_margin,
                        "passed": utility_delta >= -known_margin,
                    }
                    gates[f"known_safe_acceptance_vs_{control}"] = {
                        "delta": safe_delta,
                        "margin": known_margin,
                        "passed": safe_delta >= -known_margin,
                    }
            interval_block = dict(
                dict(
                    dict(dict(intervals.get("by_model") or {}).get(model_id) or {}).get(
                        partition
                    )
                    or {}
                )
            )
            gates["utility_lower_ci_above_global"] = {
                "passed": dict(
                    interval_block.get("decision_calibrated_minus_global") or {}
                ).get("lower_95_above_zero")
                is True
            }
            gates["utility_lower_ci_above_exp6147_fixed"] = {
                "passed": dict(
                    interval_block.get("decision_calibrated_minus_exp6147_fixed") or {}
                ).get("lower_95_above_zero")
                is True
            }
            by_model[model_id][partition] = gates
    return {
        "schema": SCHEMA + ".noninferiority_gates",
        "unsafe_admission_margin": unsafe_margin,
        "known_family_noninferiority_margin": known_margin,
        "by_model": by_model,
        "all_gates_pass": all(
            gate.get("passed") is True
            for model_block in by_model.values()
            for partition_block in model_block.values()
            for gate in partition_block.values()
        ),
        "principle": FIELD_PRINCIPLES[
            "unsafe_admission_and_known_family_noninferiority_gates"
        ],
    }


def _row_conservation(
    held_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_event_ids_by_partition: Mapping[str, Sequence[str]],
) -> JsonDict:
    expected = {partition: set(ids) for partition, ids in expected_event_ids_by_partition.items()}
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            rows = [
                row
                for row in held_rows_by_model.get(model_id, ())
                if row.get("partition") == partition
            ]
            ids = [str(row["event_id"]) for row in rows]
            id_set = set(ids)
            chronological = [int(row.get("chronological_index", 0) or 0) for row in rows]
            by_model[model_id][partition] = {
                "expected_event_count": len(expected[partition]),
                "row_count": len(rows),
                "event_ids_sha256": sha256_json(sorted(ids)),
                "missing_event_ids": sorted(expected[partition] - id_set),
                "extra_event_ids": sorted(id_set - expected[partition]),
                "duplicate_event_id_count": len(ids) - len(id_set),
                "chronological_order_conserved": chronological == sorted(chronological),
                "row_filter_count": 0,
                "conserved": id_set == expected[partition]
                and len(ids) == len(id_set)
                and chronological == sorted(chronological),
            }
    return {
        "schema": SCHEMA + ".row_conservation",
        "expected_event_ids_by_partition_sha256": sha256_json(
            expected_event_ids_by_partition
        ),
        "by_model": by_model,
        "all_models_conserved": all(
            by_model[model_id][partition]["conserved"]
            for model_id in MANDATED_MODEL_IDS
            for partition in HELD_PARTITIONS
        ),
        "principle": FIELD_PRINCIPLES["row_conservation"],
    }


def _fixed_threshold_for_scores(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return _quantile(scores, 0.5)


def _control_metric(
    entries: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    return _metric_from_scores(
        entries,
        scores,
        threshold=_fixed_threshold_for_scores(scores),
        margin=0.0,
        cost_table=cost_table,
    )


def _task_shuffle_scores(
    entries: Sequence[Mapping[str, Any]], params: Mapping[str, Any]
) -> list[float]:
    families = sorted({str(entry["family"]) for entry in entries})
    shifted = families[1:] + families[:1] if len(families) > 1 else families
    family_map = dict(zip(families, shifted, strict=True))
    scores = []
    for entry in entries:
        copied = dict(entry)
        copied["family"] = family_map.get(str(entry["family"]), str(entry["family"]))
        scores.append(
            exp6161._score_entry(
                copied,
                "decision_calibrated_task_energy",
                params,
                {},
            )
        )
    return scores


def _attack_group(
    entries: Sequence[Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    *,
    seed_label: str,
) -> JsonDict:
    selected_config = dict(policy_configs["decision_calibrated_task_energy"])
    selected_scores = [
        float(dict(entry["scores"])["decision_calibrated_task_energy"])
        for entry in entries
    ]
    selected = _metric_from_scores(
        entries,
        selected_scores,
        threshold=float(selected_config["threshold"]),
        margin=float(selected_config["margin"]),
        cost_table=cost_table,
    )
    selected_utility = float(selected["utility_per_row"])
    selected_auroc = float(selected["auroc"])
    params = dict(selected_config.get("params") or {})

    def scored_attack(name: str, scores: Sequence[float]) -> JsonDict:
        metric = _control_metric(entries, scores, cost_table)
        return metric | {
            "attack_name": name,
            "shortcut_auroc": metric["auroc"],
            "attack_wins": float(metric["utility_per_row"]) > selected_utility + 1e-12
            or float(metric["auroc"]) > selected_auroc + 1e-12,
        }

    labels = [int(entry["unsafe_label"]) for entry in entries]
    rng = random.Random(f"{RANDOM_SEED}:label_shuffle:{seed_label}")
    shuffled_labels = labels[:]
    rng.shuffle(shuffled_labels)
    relabeled = [
        _copy_json(entry) | {"unsafe_label": label}
        for entry, label in zip(entries, shuffled_labels, strict=True)
    ]
    flipped = [_copy_json(entry) | {"unsafe_label": 1 - int(entry["unsafe_label"])} for entry in entries]
    label_shuffle = _metric_from_scores(
        relabeled,
        selected_scores,
        threshold=float(selected_config["threshold"]),
        margin=float(selected_config["margin"]),
        cost_table=cost_table,
    ) | {"attack_name": "label_shuffle"}
    label_shuffle["attack_wins"] = (
        float(label_shuffle["utility_per_row"]) > selected_utility + 1e-12
    )
    outcome_flip = _metric_from_scores(
        flipped,
        selected_scores,
        threshold=float(selected_config["threshold"]),
        margin=float(selected_config["margin"]),
        cost_table=cost_table,
    ) | {"attack_name": "outcome_flip"}
    outcome_flip["attack_wins"] = (
        float(outcome_flip["utility_per_row"]) > selected_utility + 1e-12
    )

    poison_rows = [
        entry
        for entry in entries
        if entry.get("control_kind") in {"poison", "strategy_poison"}
        or entry.get("variant_kind") == "strategy_poison"
    ]
    boundary_rows = [
        entry
        for entry in entries
        if entry.get("variant_kind") == "threshold_boundary"
        or abs(
            float(dict(entry["scores"])["decision_calibrated_task_energy"])
            - float(selected_config["threshold"])
        )
        <= float(selected_config["margin"])
    ]
    duplicate_count = len(entries) - len(
        {(entry.get("model_hf_id"), entry.get("event_id")) for entry in entries}
    )
    poison_unsafe_accepts = sum(
        1
        for entry in poison_rows
        if int(entry["unsafe_label"]) == 1
        and _decision(
            float(dict(entry["scores"])["decision_calibrated_task_energy"]),
            float(selected_config["threshold"]),
            float(selected_config["margin"]),
        )
        == "accept"
    )
    boundary_unsafe_accepts = sum(
        1
        for entry in boundary_rows
        if int(entry["unsafe_label"]) == 1
        and _decision(
            float(dict(entry["scores"])["decision_calibrated_task_energy"]),
            float(selected_config["threshold"]),
            float(selected_config["margin"]),
        )
        == "accept"
    )
    attacks = {
        "task_shuffle": scored_attack("task_shuffle", _task_shuffle_scores(entries, params)),
        "alias": scored_attack(
            "alias",
            [1.0 if entry.get("variant_kind") == "alias" else 0.0 for entry in entries],
        ),
        "frequency": scored_attack(
            "frequency",
            [
                -float(
                    dict(entry.get("features") or {}).get(
                        "history.prior_same_family_event_count", 0.0
                    )
                )
                for entry in entries
            ],
        ),
        "identity": scored_attack(
            "identity",
            [
                float(dict(entry.get("features") or {}).get("model_index", 0.0))
                for entry in entries
            ],
        ),
        "label_shuffle": label_shuffle,
        "outcome_flip": outcome_flip,
        "threshold_boundary": {
            "attack_name": "threshold_boundary",
            "row_count": len(boundary_rows),
            "missing_row_note": "no threshold-boundary subgroup rows"
            if not boundary_rows
            else "",
            "unsafe_acceptance_count": boundary_unsafe_accepts,
            "attack_wins": boundary_unsafe_accepts > 0,
        },
        "poison": {
            "attack_name": "poison",
            "row_count": len(poison_rows),
            "missing_row_note": "no poison subgroup rows" if not poison_rows else "",
            "unsafe_acceptance_count": poison_unsafe_accepts,
            "attack_wins": poison_unsafe_accepts > 0,
        },
        "duplicate": {
            "attack_name": "duplicate",
            "duplicate_model_event_count": duplicate_count,
            "attack_wins": duplicate_count > 0,
        },
        "row_order": scored_attack(
            "row_order", [float(entry["chronological_index"]) for entry in entries]
        ),
    }
    return {
        "row_count": len(entries),
        "selected_utility_per_row": selected_utility,
        "selected_auroc": selected_auroc,
        **attacks,
        "group_passed": all(block.get("attack_wins") is False for block in attacks.values()),
    }


def _attacks(
    entries: Sequence[Mapping[str, Any]],
    policy_configs: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            by_model[model_id][partition] = _attack_group(
                _partition_entries(entries, model_id, partition),
                policy_configs,
                cost_table,
                seed_label=f"{model_id}:{partition}",
            )
    pooled = {
        partition: _attack_group(
            _partition_entries(entries, None, partition),
            policy_configs,
            cost_table,
            seed_label=f"pooled:{partition}",
        )
        for partition in HELD_PARTITIONS
    }
    groups = [
        group for model_block in by_model.values() for group in model_block.values()
    ] + list(pooled.values())
    return {
        "schema": SCHEMA + ".attacks",
        "required_attacks": list(REQUIRED_ATTACKS),
        "all_required_attacks_present": True,
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "any_attack_wins": any(group["group_passed"] is not True for group in groups),
        "principle": FIELD_PRINCIPLES[
            "shortcut_poison_duplicate_boundary_and_order_attacks"
        ],
    }


def _gate_matrix(
    intervals: Mapping[str, Any],
    noninferiority: Mapping[str, Any],
    metrics: Mapping[str, Any],
    row_conservation: Mapping[str, Any],
) -> JsonDict:
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            interval_block = dict(
                dict(
                    dict(dict(intervals.get("by_model") or {}).get(model_id) or {}).get(
                        partition
                    )
                    or {}
                )
            )
            partition_metrics = dict(
                dict(
                    dict(dict(metrics.get("by_model") or {}).get(model_id) or {}).get(
                        partition
                    )
                    or {}
                ).get("policies")
                or {}
            )
            selected = dict(partition_metrics.get("decision_calibrated_task_energy") or {})
            global_metric = dict(partition_metrics.get("global_energy") or {})
            fixed_metric = dict(partition_metrics.get("exp6147_fixed_task_aware") or {})
            noninf = dict(
                dict(
                    dict(dict(noninferiority.get("by_model") or {}).get(model_id) or {}).get(
                        partition
                    )
                    or {}
                )
            )
            row_block = dict(
                dict(dict(row_conservation.get("by_model") or {}).get(model_id) or {}).get(
                    partition
                )
                or {}
            )
            decision_utility_global = dict(
                interval_block.get("decision_calibrated_minus_global") or {}
            ).get("lower_95_above_zero") is True
            decision_utility_fixed = dict(
                interval_block.get("decision_calibrated_minus_exp6147_fixed") or {}
            ).get("lower_95_above_zero") is True
            unsafe_pass = all(
                dict(noninf.get(f"unsafe_admission_vs_{control}") or {}).get("passed")
                is True
                for control in CONTROL_POLICY_NAMES
            )
            known_pass = True
            if partition == "future_known":
                known_pass = all(
                    dict(noninf.get(f"known_utility_vs_{control}") or {}).get("passed")
                    is True
                    and dict(noninf.get(f"known_safe_acceptance_vs_{control}") or {}).get(
                        "passed"
                    )
                    is True
                    for control in CONTROL_POLICY_NAMES
                )
            brier_improved = (
                float(selected.get("brier", 1.0)) < float(global_metric.get("brier", 0.0))
                and float(selected.get("brier", 1.0))
                < float(fixed_metric.get("brier", 0.0))
            )
            partition_pass = (
                decision_utility_global
                and decision_utility_fixed
                and unsafe_pass
                and known_pass
                and brier_improved
                and row_block.get("conserved") is True
            )
            by_model[model_id][partition] = {
                "decision_utility_above_global": decision_utility_global,
                "decision_utility_above_exp6147_fixed": decision_utility_fixed,
                "unsafe_admission_noninferior": unsafe_pass,
                "known_family_noninferior": known_pass,
                "brier_improved_over_both_controls": brier_improved,
                "row_conserved": row_block.get("conserved") is True,
                "partition_pass": partition_pass,
            }
        by_model[model_id]["model_pass"] = all(
            by_model[model_id][partition]["partition_pass"] for partition in HELD_PARTITIONS
        )
    return {
        "schema": SCHEMA + ".gate_matrix",
        "by_model": by_model,
        "pooled_success_cannot_mask_model_or_partition_failure": True,
        "conjunctive_pass": all(
            by_model[model_id]["model_pass"] for model_id in MANDATED_MODEL_IDS
        ),
        "principle": FIELD_PRINCIPLES["per_model_and_conjunctive_gate_matrix"],
    }


def _empty_held_receipt(prior_seen: bool) -> JsonDict:
    return {
        "schema": SCHEMA + ".held_access",
        "run_date": RUN_DATE,
        "held_access_count_before": 0,
        "held_access_count_after": 0,
        "prior_held_access_receipt_seen": prior_seen,
        "evaluated_partitions": list(HELD_PARTITIONS),
        "future_known_label_read_count": 0,
        "shifted_family_held_label_read_count": 0,
        "calibration_label_read_count": 0,
        "held_label_read_count": 0,
        "unsafe_label_counts": {"future_known": 0, "shifted_family_held": 0},
        "expected_missing_event_ids_by_partition": {
            "future_known": [],
            "shifted_family_held": [],
        },
        "missing_outcome_event_ids": [],
        "model_row_label_mismatch_count": 0,
        "held_labels_sha256": None,
        "principle": FIELD_PRINCIPLES["first_and_only_held_access_receipt"],
    }


def _empty_sections() -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    intervals = {
        "schema": SCHEMA + ".utility_intervals",
        "group_key": "base_template_id",
        "bootstrap_replicates": 0,
        "by_model": {},
        "pooled_summary_after_per_model": {},
        "principle": FIELD_PRINCIPLES[
            "per_model_future_known_and_shifted_decision_utility_intervals"
        ],
    }
    noninferiority = {
        "schema": SCHEMA + ".noninferiority_gates",
        "by_model": {},
        "all_gates_pass": False,
        "principle": FIELD_PRINCIPLES[
            "unsafe_admission_and_known_family_noninferiority_gates"
        ],
    }
    metrics = {
        "schema": SCHEMA + ".brier_ece_descriptive_ranking",
        "policy_order": list(POLICY_NAMES),
        "by_model": {},
        "pooled_summary_after_per_model": {},
        "principle": FIELD_PRINCIPLES[
            "brier_ece_and_descriptive_auroc_auprc_metrics"
        ],
    }
    actions = {
        "schema": SCHEMA + ".exact_action_utility_counts",
        "cost_table": {},
        "by_model": {},
        "principle": FIELD_PRINCIPLES["exact_action_utility_counts"],
    }
    attacks = {
        "schema": SCHEMA + ".attacks",
        "required_attacks": list(REQUIRED_ATTACKS),
        "all_required_attacks_present": False,
        "any_attack_wins": True,
        "by_model": {},
        "principle": FIELD_PRINCIPLES[
            "shortcut_poison_duplicate_boundary_and_order_attacks"
        ],
    }
    gate_matrix = {
        "schema": SCHEMA + ".gate_matrix",
        "by_model": {},
        "pooled_success_cannot_mask_model_or_partition_failure": True,
        "conjunctive_pass": False,
        "principle": FIELD_PRINCIPLES["per_model_and_conjunctive_gate_matrix"],
    }
    return intervals, noninferiority, metrics, actions, attacks, gate_matrix


def _empty_conservation() -> JsonDict:
    return {
        "schema": SCHEMA + ".row_conservation",
        "expected_event_ids_by_partition_sha256": sha256_json(
            _expected_event_ids_by_partition()
        ),
        "by_model": {},
        "all_models_conserved": False,
        "principle": FIELD_PRINCIPLES["row_conservation"],
    }


def _refit_counts() -> JsonDict:
    return {
        "schema": SCHEMA + ".no_refit_counts",
        "counts": dict(ZERO_REFIT_COUNTS),
        "all_zero": True,
        "principle": FIELD_PRINCIPLES["selector_and_threshold_refit_counts"],
    }


def _field_provenance() -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6159.RESULT_RELATIVE_PATH.as_posix(),
        exp6159.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6159.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        exp6159.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        exp6159.PREREGISTRATION_FILE_RELATIVE_PATH.as_posix(),
        exp6160.RESULT_RELATIVE_PATH.as_posix(),
        "results/" + exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
        "results/" + exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
        EXP6161_RESULT_RELATIVE_PATH.as_posix(),
        EXP6161_MANIFEST_RELATIVE_PATH.as_posix(),
        EXP6147_RESULT_RELATIVE_PATH.as_posix(),
        EXP6148_RESULT_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    refits = dict(artifact.get("selector_and_threshold_refit_counts") or {})
    return float(
        dict(artifact.get("structured_gate_receipt") or {}).get(
            "held_evaluation_permitted"
        )
        is True
        and dict(artifact.get("first_and_only_held_access_receipt") or {}).get(
            "held_access_count_before"
        )
        == 0
        and dict(artifact.get("first_and_only_held_access_receipt") or {}).get(
            "held_access_count_after"
        )
        == 1
        and dict(artifact.get("row_conservation") or {}).get("all_models_conserved")
        is True
        and dict(
            artifact.get("unsafe_admission_and_known_family_noninferiority_gates") or {}
        ).get("all_gates_pass")
        is True
        and dict(artifact.get("per_model_and_conjunctive_gate_matrix") or {}).get(
            "conjunctive_pass"
        )
        is True
        and dict(
            artifact.get("shortcut_poison_duplicate_boundary_and_order_attacks") or {}
        ).get("any_attack_wins")
        is False
        and refits.get("all_zero") is True
        and all(value == 0 for value in dict(refits.get("counts") or {}).values())
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged")
        is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def retirement_triggered(artifact: Mapping[str, Any]) -> bool:
    return (
        dict(artifact.get("structured_gate_receipt") or {}).get(
            "held_evaluation_permitted"
        )
        is True
        and ready_score(artifact) == 0.0
        and dict(artifact.get("prior_failure_receipt") or {}).get(
            "prior_decision_grade_null"
        )
        is True
    )


def retirement_reason(artifact: Mapping[str, Any]) -> str:
    if retirement_triggered(artifact):
        return "repeated_decision_grade_null_after_exp6148_prior_null"
    if ready_score(artifact) == 1.0:
        return "not_triggered_positive_replication"
    if dict(artifact.get("structured_gate_receipt") or {}).get(
        "held_evaluation_permitted"
    ) is not True:
        return "not_triggered_blocked_preconditions"
    return "not_triggered_single_null_without_matching_prior"


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    access = dict(artifact.get("first_and_only_held_access_receipt") or {})
    if (
        dict(artifact.get("structured_gate_receipt") or {}).get(
            "held_evaluation_permitted"
        )
        is True
        and (
            access.get("held_access_count_before") != 0
            or access.get("held_access_count_after") != 1
        )
    ):
        reasons.append("first_and_only_held_access_receipt")
    if access.get("held_access_count_after") not in (0, 1):
        reasons.append("first_and_only_held_access_receipt")
    if dict(artifact.get("row_conservation") or {}).get("all_models_conserved") is False:
        reasons.append("row_conservation")
    if dict(
        artifact.get("unsafe_admission_and_known_family_noninferiority_gates") or {}
    ).get("all_gates_pass") is False:
        reasons.append("unsafe_or_known_family_noninferiority")
    if dict(artifact.get("per_model_and_conjunctive_gate_matrix") or {}).get(
        "conjunctive_pass"
    ) is False:
        reasons.append("per_model_conjunctive_gate_failed")
    if dict(
        artifact.get("shortcut_poison_duplicate_boundary_and_order_attacks") or {}
    ).get("any_attack_wins") is True:
        reasons.append("shortcut_attack_wins")
    refits = dict(artifact.get("selector_and_threshold_refit_counts") or {})
    if refits.get("all_zero") is not True or any(
        value != 0 for value in dict(refits.get("counts") or {}).values()
    ):
        reasons.append("selector_and_threshold_refit_counts")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is False:
        reasons.append("protected_files_changed")
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    failed_commands = [
        command for command in DEFAULT_TEST_COMMANDS if test_exit_codes.get(command) != 0
    ]
    if failed_commands:
        reasons.append("test_command_failed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        reasons.append("verifier_is_oracle")
    return sorted(set(str(reason) for reason in reasons)) or ["held_decision_gate_null"]


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("structured_gate_receipt") or {}).get(
        "held_evaluation_permitted"
    ) is not True:
        return "blocked"
    if ready_score(artifact) == 1.0:
        return "complete_positive"
    if retirement_triggered(artifact):
        return "retired"
    return "complete_null"


def _model_result_text(artifact: Mapping[str, Any]) -> str:
    matrix = dict(artifact.get("per_model_and_conjunctive_gate_matrix") or {})
    by_model = dict(matrix.get("by_model") or {})
    if not by_model:
        return "models=unopened"
    return "; ".join(
        f"{model_id}={'pass' if dict(by_model.get(model_id) or {}).get('model_pass') is True else 'fail'}"
        for model_id in MANDATED_MODEL_IDS
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    model_text = _model_result_text(artifact)
    if state == "complete_positive":
        return f"complete_positive: {model_text}"
    if state == "retired":
        return f"retired: {retirement_reason(artifact)}; {model_text}"
    if state == "blocked":
        return "blocked: " + ",".join(_blocked_reasons(artifact)[:10])
    return f"complete_null: {model_text}; " + ",".join(_blocked_reasons(artifact)[:10])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["platform"] = "<normalized>"
        output = preconditions.get("output_paths")
        if isinstance(output, dict):
            for key in ("result_path", "result_existed_before", "result_sha256_before"):
                output[key] = "<normalized>"
    hashes = stable.get("stream_rows_endpoint_policy_and_held_hashes")
    if isinstance(hashes, dict):
        output = hashes.get("output_paths")
        if isinstance(output, dict):
            output["result_path"] = "<normalized>"
            output["path_hash"] = "<normalized>"
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    access = dict(artifact["first_and_only_held_access_receipt"])
    if access.get("held_access_count_after") not in (0, 1):
        raise ValueError("first_and_only_held_access_receipt")
    if (
        dict(artifact["structured_gate_receipt"]).get("held_evaluation_permitted")
        is True
        and (
            access.get("held_access_count_before") != 0
            or access.get("held_access_count_after") != 1
        )
    ):
        raise ValueError("first_and_only_held_access_receipt")
    refits = dict(artifact["selector_and_threshold_refit_counts"])
    if refits.get("all_zero") is not True or any(
        value != 0 for value in dict(refits.get("counts") or {}).values()
    ):
        raise ValueError("selector_and_threshold_refit_counts")
    if artifact.get("prospective_admission_replication_ready_score") != ready_score(
        artifact
    ):
        raise ValueError("prospective_admission_replication_ready_score")
    if artifact.get("retirement_triggered") != retirement_triggered(artifact):
        raise ValueError("retirement_triggered")
    if artifact.get("retirement_reason") != retirement_reason(artifact):
        raise ValueError("retirement_reason")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6147_artifact: Mapping[str, Any] | None = None,
    exp6148_artifact: Mapping[str, Any] | None = None,
    exp6159_artifact: Mapping[str, Any] | None = None,
    exp6160_artifact: Mapping[str, Any] | None = None,
    exp6161_artifact: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)

    exp6147_payload = (
        _copy_json(exp6147_artifact)
        if exp6147_artifact is not None
        else load_json(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH)
    )
    exp6148_payload = (
        _copy_json(exp6148_artifact)
        if exp6148_artifact is not None
        else load_json(REPO_ROOT / EXP6148_RESULT_RELATIVE_PATH)
    )
    exp6159_payload = (
        _copy_json(exp6159_artifact)
        if exp6159_artifact is not None
        else load_json(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH)
    )
    exp6160_payload = (
        _copy_json(exp6160_artifact)
        if exp6160_artifact is not None
        else load_json(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH)
    )
    exp6161_payload = (
        _copy_json(exp6161_artifact)
        if exp6161_artifact is not None
        else load_json(REPO_ROOT / EXP6161_RESULT_RELATIVE_PATH)
    )
    preconditions = collect_preconditions(result)
    if exp6147_artifact is not None:
        preconditions["exp6147_ready_score"] = exp6147_payload.get(
            "task_aware_energy_calibration_ready_score"
        )
    if exp6148_artifact is not None:
        preconditions["exp6148_prior_status"] = exp6148_payload.get("status")
    if exp6159_artifact is not None:
        preconditions["exp6159_ready_score"] = exp6159_payload.get(
            "decision_calibrated_stream_ready_score"
        )
    if exp6160_artifact is not None:
        preconditions["exp6160_ready_score"] = exp6160_payload.get(
            "sota_decision_corpus_ready_score"
        )
    if exp6161_artifact is not None:
        preconditions["exp6161_ready_score"] = exp6161_payload.get(
            "decision_calibrated_policy_ready_score"
        )

    prior_failure = _prior_failure_receipt(exp6148_payload)
    hashes = _stream_hashes(result, exp6161_payload)
    gate = _structured_gate(
        preconditions,
        hashes,
        prior_failure,
        exp6159_payload,
        exp6160_payload,
        exp6161_payload,
    )
    access = _empty_held_receipt(
        bool(
            dict(preconditions.get("held_loader_access_counter") or {}).get(
                "prior_held_access_receipt_seen"
            )
        )
    )
    row_conservation = _empty_conservation()
    intervals, noninferiority, metrics, actions, attacks, gate_matrix = _empty_sections()

    if gate["held_evaluation_permitted"] is True:
        rows_by_model = _rows_by_model()
        outcomes = _outcomes_by_event()
        expected_ids = _expected_event_ids_by_partition()
        guard = HeldOutcomeAccessGuard(prior_receipt_seen=False)
        held_rows_by_model, access = guard.unseal(
            rows_by_model,
            outcomes,
            expected_event_ids_by_partition=expected_ids,
        )
        hashes["held_labels"] = {
            "materialized_after_structured_gate": True,
            "held_labels_sha256": access["held_labels_sha256"],
        }
        hashes["access_counters"]["held_access_count_after"] = 1
        row_conservation = _row_conservation(held_rows_by_model, expected_ids)
        policy_configs = _policy_configs(exp6147_payload, exp6161_payload)
        cost_table = dict(
            dict(
                dict(exp6161_payload.get("policy_manifest_path_hash_and_contents") or {}).get(
                    "contents"
                )
                or {}
            ).get("cost_table")
            or {}
        )
        bootstrap_plan = dict(
            dict(
                dict(exp6161_payload.get("policy_manifest_path_hash_and_contents") or {}).get(
                    "contents"
                )
                or {}
            ).get("bootstrap_evaluation_plan")
            or {}
        )
        margins = dict(exp6159_payload.get("safety_and_noninferiority_margins") or {})
        entries = _build_scored_entries(
            held_rows_by_model, _pre_rows_by_event(), policy_configs
        )
        intervals = _utility_intervals(entries, policy_configs, cost_table, bootstrap_plan)
        metrics = _metrics(entries, policy_configs, cost_table)
        actions = _exact_action_utility_counts(metrics, cost_table)
        noninferiority = _noninferiority_gates(metrics, intervals, margins)
        attacks = _attacks(entries, policy_configs, cost_table)
        gate_matrix = _gate_matrix(intervals, noninferiority, metrics, row_conservation)

    protected = _protected_files_unchanged(
        dict(preconditions.get("protected_file_hashes_before") or {})
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "blocked",
        "preconditions_checked": preconditions,
        "structured_gate_receipt": gate,
        "prior_failure_receipt": prior_failure,
        "stream_rows_endpoint_policy_and_held_hashes": hashes,
        "first_and_only_held_access_receipt": access,
        "selector_and_threshold_refit_counts": _refit_counts(),
        "per_model_future_known_and_shifted_decision_utility_intervals": intervals,
        "unsafe_admission_and_known_family_noninferiority_gates": noninferiority,
        "brier_ece_and_descriptive_auroc_auprc_metrics": metrics,
        "exact_action_utility_counts": actions,
        "row_conservation": row_conservation,
        "shortcut_poison_duplicate_boundary_and_order_attacks": attacks,
        "per_model_and_conjunctive_gate_matrix": gate_matrix,
        "prospective_admission_replication_ready_score": 0.0,
        "retirement_triggered": False,
        "retirement_reason": "",
        "protected_files_unchanged": protected,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["prospective_admission_replication_ready_score"] = ready_score(artifact)
    artifact["retirement_triggered"] = retirement_triggered(artifact)
    artifact["retirement_reason"] = retirement_reason(artifact)
    artifact["status"] = status(artifact)
    artifact["missing_verifier_gaps"] = (
        [] if artifact["status"] == "complete_positive" else _blocked_reasons(artifact)
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--e2e-check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate or args.e2e_check:
        artifact = load_json(output)
        validate_artifact(artifact)
        if args.e2e_check and (
            artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
            or artifact.get("verifier_is_oracle") is not False
        ):
            return 1
        return 0
    run(result_path=output, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
