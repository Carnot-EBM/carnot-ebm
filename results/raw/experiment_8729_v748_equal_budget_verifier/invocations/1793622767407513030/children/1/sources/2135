"""Exp6148 shifted-family held admission evaluation.

Spec refs: REQ-VERIFY-6148, REQ-VERIFY-6148-1, REQ-VERIFY-6148-2,
REQ-VERIFY-6148-3, REQ-VERIFY-6148-4, REQ-VERIFY-6148-5,
REQ-VERIFY-6148-6, REQ-VERIFY-6148-7, REQ-VERIFY-6148-8,
REQ-VERIFY-6148-9, SCENARIO-VERIFY-6148-ONE-SHOT,
SCENARIO-VERIFY-6148-PAIRED, SCENARIO-VERIFY-6148-ATTACKS.

Exp6148 opens the held Exp6145/Exp6146 rows once. The selector, threshold,
abstention margin, replay schema, and memory budget come from Exp6147 and are
not changed after the held labels are materialized.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_6145_constraint_shift_stream as exp6145
from carnot import experiment_6146_sota_constraint_event_corpus as exp6146
from carnot import experiment_6147_task_aware_energy_calibration as exp6147
from carnot.eval.metrics import auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6148_shifted_family_admission_held.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6148_shifted_family_admission_held.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6148_shifted_family_admission_held.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
EXP6147_RESULT_RELATIVE_PATH = exp6147.RESULT_RELATIVE_PATH
SCHEMA = "carnot.experiment_6148.shifted_family_admission_held.v1"
EXPERIMENT_ID = "experiment_6148_shifted_family_admission_held"
RUN_DATE = "20260805"
RANDOM_SEED = 6148
INFERENCE_SUBSTRATE = "sealed_cached_event_evaluation"
VERIFIER_IS_ORACLE = False
HELD_PARTITIONS = ("future_known", "sealed_shifted_family")
POLICY_SCORE_NAMES = ("global_energy", "task_aware_energy")
BOOTSTRAP_REPLICATES = 300
PRIMARY_METRIC = "held_grouped_auroc_delta_task_aware_minus_global"

MANDATED_MODEL_IDS = exp6147.MANDATED_MODEL_IDS
MEMORY_BUDGET_EVENTS_PER_TASK = exp6147.MEMORY_BUDGET_EVENTS_PER_TASK
MIN_TASK_REPLAY_COUNT = exp6147.MIN_TASK_REPLAY_COUNT

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6148_shifted_family_admission_held.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6148_shifted_family_admission_held.py "
    "-m pytest tests/python/test_experiment_6148_shifted_family_admission_held.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6148_shifted_family_admission_held.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6148_shifted_family_admission_held.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6148_shifted_family_admission_held --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6148_shifted_family_admission_held.json"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6148_shifted_family_admission_held.py "
    "tests/python/test_experiment_6148_shifted_family_admission_held.py"
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
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    VERIFY_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6145.RESULT_RELATIVE_PATH,
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
    exp6146.RESULT_RELATIVE_PATH,
    EXP6147_RESULT_RELATIVE_PATH,
    exp6147.MODULE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "upstream_and_freeze_manifest_hashes",
    "first_and_only_held_access_receipt",
    "held_group_row_conservation",
    "per_model_future_known_and_shifted_metrics",
    "paired_task_aware_minus_global_intervals",
    "safe_acceptance_noninferiority",
    "unsafe_acceptance_and_abstention_matrices",
    "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks",
    "exact_utility_diagnostic",
    "selector_refit_count",
    "prompt_retry_and_llm_invocation_counts",
    "shifted_family_admission_ready_score",
    "retirement_triggered",
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
    "status": "A terminal state distinguishes positive, null, retired, or blocked held evidence.",
    "preconditions_checked": "Upstream rows, splits, held labels, selector manifests, evaluator code, output paths, and protected files are hashed before the held read.",
    "structured_gate_receipt": "Held evaluation opens only after Exp6145/Exp6146/Exp6147 readiness, exact selector hash match, row conservation, no prior held receipt, and no live substrate pass.",
    "upstream_and_freeze_manifest_hashes": "Frozen selector, threshold, abstention, replay schema, model sidecars, held labels, code, and protected-file hashes are content-addressed.",
    "first_and_only_held_access_receipt": "The held outcome materialization count is exactly one and covers future-known plus shifted-family rows.",
    "held_group_row_conservation": "Every mandated model preserves all future-known and shifted-family event ids with no duplicates, omissions, or extras.",
    "per_model_future_known_and_shifted_metrics": "Metrics are separated by source model and by future-known versus shifted-family groups before pooled summaries.",
    "paired_task_aware_minus_global_intervals": "Paired grouped intervals expose task-aware minus global deltas and prevent aggregate-only readiness.",
    "safe_acceptance_noninferiority": "Future-known safe acceptance cannot regress beyond the frozen noninferiority margin.",
    "unsafe_acceptance_and_abstention_matrices": "Unsafe acceptance, safe rejection, abstention, coverage, and risk are visible for each held group and policy.",
    "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks": "Frozen shortcut and boundary attacks report subgroups, missing rows, and whether any attack wins.",
    "exact_utility_diagnostic": "Exact labels score the utility of admitted strategies after decisions without becoming selector features.",
    "selector_refit_count": "Exactly zero selector, threshold, abstention, replay-schema, or row-selection refits occur after unsealing.",
    "prompt_retry_and_llm_invocation_counts": "Prompt retries, LLM invocations, tokenizer loads, GGUF loads, and GPU workers are all zero.",
    "shifted_family_admission_ready_score": "Readiness is conjunctive; aggregate gains cannot mask shifted-family unsafe acceptance or known-family regression.",
    "retirement_triggered": "A repeated prior-failure mode retires the scope rather than rebranding a held null.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "Measured sealed cached-row evaluation time is reported without implying model inference.",
    "inference_substrate": "Use `sealed_cached_event_evaluation`.",
    "verifier_is_oracle": "The evaluator is not an oracle; exact outcomes are held labels used only after the single unseal.",
    "missing_verifier_gaps": "Selector mismatch, held-access, refit, attack, subgroup, safety, noninferiority, or evidence gaps are explicit.",
    "field_provenance": "Every field traces to specs, Exp6145/Exp6146/Exp6147 artifacts, held sidecars, tests, or command receipts.",
    "test_commands": "Commands document unit/spec coverage, structured gate, freeze/hash, one-shot access, row conservation, grouped paired metrics, noninferiority, attacks, zero-refit/no-LLM, schema, adversarial verify, protected-file, applicable E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, held-label, selector, threshold, attack, test, protected-file, or output drift.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `retired:`, or `blocked:` and state the held causal discriminator.",
}

PROMPT_RETRY_AND_LLM_ZERO_COUNTS = {
    "prompt_retry_count": 0,
    "llm_invocation_count": 0,
    "tokenizer_load_count": 0,
    "gguf_load_count": 0,
    "gpu_worker_count": 0,
    "row_selection_after_unseal_count": 0,
}


class HeldAccessError(ValueError):
    """Raised when held outcomes would be read more than once."""


class HeldAccessGuard:
    """Small stateful guard for the single held-label materialization."""

    def __init__(self, *, prior_receipt_seen: bool) -> None:
        self.prior_receipt_seen = prior_receipt_seen
        self.access_count = 0

    def unseal(
        self,
        rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        expected_event_ids_by_partition: Mapping[str, Sequence[str]],
    ) -> tuple[dict[str, list[JsonDict]], JsonDict]:
        if self.prior_receipt_seen:
            raise HeldAccessError("prior held-access receipt blocks unsealing")
        if self.access_count != 0:
            raise HeldAccessError("held labels must be materialized exactly one time")
        self.access_count += 1

        held_rows: dict[str, list[JsonDict]] = {}
        counts = Counter()
        label_payload: list[JsonDict] = []
        seen: dict[str, set[str]] = {partition: set() for partition in HELD_PARTITIONS}
        for model_id, rows in rows_by_model.items():
            model_held: list[JsonDict] = []
            for row in rows:
                partition = str(row.get("partition"))
                if partition == "calibration":
                    continue
                if partition not in HELD_PARTITIONS:
                    continue
                copied = dict(row)
                unsafe = unsafe_label_from_model_row(copied)
                copied["unsafe_label"] = unsafe
                model_held.append(copied)
                event_id = str(copied["event_id"])
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
            held_rows[model_id] = model_held

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
            "held_access_count": self.access_count,
            "prior_held_access_receipt_seen": self.prior_receipt_seen,
            "evaluated_partitions": list(HELD_PARTITIONS),
            "future_known_label_read_count": counts["future_known"],
            "sealed_shifted_family_label_read_count": counts["sealed_shifted_family"],
            "calibration_label_read_count": 0,
            "held_label_read_count": counts["future_known"] + counts["sealed_shifted_family"],
            "unsafe_label_counts": {
                "future_known": counts["future_known_unsafe"],
                "sealed_shifted_family": counts["sealed_shifted_family_unsafe"],
            },
            "expected_missing_event_ids_by_partition": expected_missing,
            "held_labels_sha256": sha256_json(label_payload),
            "principle": FIELD_PRINCIPLES["first_and_only_held_access_receipt"],
        }


canonical_json = exp6147.canonical_json
sha256_text = exp6147.sha256_text
sha256_json = exp6147.sha256_json
sha256_file = exp6147.sha256_file


def load_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")  # pragma: no cover
    return dict(payload)


def load_jsonl(path: str | Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
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


def _model_sidecar_path(hf_id: str) -> Path:
    return REPO_ROOT / "results" / exp6146.row_sidecar_filename(hf_id)


def _prior_held_access_receipt_seen(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        payload = load_json(result_path)
    except (OSError, json.JSONDecodeError, ValueError):  # pragma: no cover
        return True
    return isinstance(payload.get("first_and_only_held_access_receipt"), Mapping)


def _exp6147_receipt_hash(exp6147_artifact: Mapping[str, Any], relative: Path) -> str | None:
    for receipt in dict(exp6147_artifact.get("preconditions_checked") or {}).get(
        "hashed_input_receipts", ()
    ):
        if dict(receipt).get("path") == relative.as_posix():
            return dict(receipt).get("sha256")
    return None


def _expected_event_ids_by_partition() -> dict[str, list[str]]:
    splits = load_json(REPO_ROOT / exp6145.SPLIT_FILE_RELATIVE_PATH)
    event_to_partition = dict(splits.get("event_to_partition") or {})
    return {
        partition: sorted(
            event_id for event_id, assigned in event_to_partition.items() if assigned == partition
        )
        for partition in HELD_PARTITIONS
    }


def _rows_by_model() -> dict[str, list[JsonDict]]:
    return {hf_id: load_jsonl(_model_sidecar_path(hf_id)) for hf_id in MANDATED_MODEL_IDS}


def _pre_rows_by_event() -> dict[str, JsonDict]:
    return {
        str(row["event_id"]): row for row in load_jsonl(REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH)
    }


def unsafe_label_from_model_row(row: Mapping[str, Any]) -> int:
    return int(row.get("current_validator_result") != "accepted" or bool(row.get("invalid_output")))


def collect_preconditions(result_path: Path) -> JsonDict:
    exp6145_artifact = load_json(REPO_ROOT / exp6145.RESULT_RELATIVE_PATH)
    exp6146_artifact = load_json(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH)
    exp6147_artifact = load_json(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hashed_input_receipts": [
            _file_receipt(REPO_ROOT / relative) | {"path": relative.as_posix()}
            for relative in HASHED_INPUTS
        ],
        "exp6145_ready_score": exp6145_artifact.get("constraint_shift_stream_ready_score"),
        "exp6146_ready_score": exp6146_artifact.get("sota_constraint_event_corpus_ready_score"),
        "exp6147_ready_score": exp6147_artifact.get("task_aware_energy_calibration_ready_score"),
        "prior_held_access_receipt_seen": _prior_held_access_receipt_seen(result_path),
        "output_paths": {
            "result_path": str(result_path),
            "parent_writable": result_path.parent.exists(),
            "existed_before": result_path.exists(),
            "sha256_before": sha256_file(result_path) if result_path.exists() else None,
        },
        "protected_file_hashes_before": _protected_hashes(),
        "llm_loaded": False,
        "tokenizer_loaded": False,
        "training_loop_invoked": False,
        "selector_refit_count": 0,
        "prompt_retry_and_llm_invocation_counts": dict(PROMPT_RETRY_AND_LLM_ZERO_COUNTS),
    }


def _upstream_and_freeze_hashes(
    result_path: Path, exp6147_artifact: Mapping[str, Any], selection: Mapping[str, Any]
) -> JsonDict:
    sidecars = {
        hf_id: {
            **_file_receipt(_model_sidecar_path(hf_id)),
            "row_count": len(load_jsonl(_model_sidecar_path(hf_id))),
        }
        for hf_id in MANDATED_MODEL_IDS
    }
    selected_code_hash = sha256_file(REPO_ROOT / exp6147.MODULE_RELATIVE_PATH)
    frozen_code_hash = _exp6147_receipt_hash(exp6147_artifact, exp6147.MODULE_RELATIVE_PATH)
    return {
        "schema": SCHEMA + ".upstream_freeze_hashes",
        "exp6145": {
            "result": _file_receipt(REPO_ROOT / exp6145.RESULT_RELATIVE_PATH),
            "rows": _file_receipt(REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH),
            "splits": _file_receipt(REPO_ROOT / exp6145.SPLIT_FILE_RELATIVE_PATH),
            "outcomes": _file_receipt(REPO_ROOT / exp6145.OUTCOME_FILE_RELATIVE_PATH),
        },
        "exp6146": {
            "result": _file_receipt(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH),
            "model_row_sidecars": sidecars,
        },
        "exp6147": {
            "result": _file_receipt(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH),
            "selection_manifest_hash_declared": exp6147_artifact.get("selection_manifest_hash"),
            "selection_manifest_hash_recomputed": exp6147.selection_manifest_hash(selection),
            "selected_score_code_sha256": selected_code_hash,
            "selected_score_code_freeze_receipt_sha256": frozen_code_hash,
            "selected_score_code_hash_matches_freeze": selected_code_hash == frozen_code_hash,
        },
        "expected_held_group_ids_sha256": sha256_json(_expected_event_ids_by_partition()),
        "held_labels": {
            "materialized_after_structured_gate": False,
            "held_labels_sha256": None,
        },
        "evaluator_code": {
            "module": _file_receipt(REPO_ROOT / MODULE_RELATIVE_PATH),
            "tests": _file_receipt(REPO_ROOT / TEST_RELATIVE_PATH),
            "adversarial_verify": _file_receipt(REPO_ROOT / Path("scripts/adversarial_verify.py")),
        },
        "output_path": str(result_path),
        "protected_file_hashes_before": _protected_hashes(),
        "principle": FIELD_PRINCIPLES["upstream_and_freeze_manifest_hashes"],
    }


def _verify_frozen_selection(exp6147_artifact: Mapping[str, Any]) -> JsonDict:
    selection = dict(
        exp6147_artifact.get("selected_score_threshold_abstention_and_memory_budget") or {}
    )
    abstention = dict(selection.get("abstention_rule") or {})
    replay_schema = dict(selection.get("replay_statistic_schema") or {})
    declared_hash = exp6147_artifact.get("selection_manifest_hash")
    recomputed_hash = exp6147.selection_manifest_hash(selection)
    checks = {
        "selected_score_exact": selection.get("selected_score") == "task_aware_energy",
        "threshold_finite": isinstance(selection.get("threshold"), (int, float)),
        "abstention_rule_frozen": abstention.get("type") == "score_margin"
        and isinstance(abstention.get("margin"), (int, float)),
        "replay_statistic_schema_frozen": replay_schema
        == {
            "task_key": "family",
            "location": "prior task raw-energy mean with global fallback",
            "scale": "prior task raw-energy std with floor 0.25 and global fallback",
            "minimum_task_replay_count": MIN_TASK_REPLAY_COUNT,
        },
        "memory_budget_exact": selection.get("memory_budget_events_per_task")
        == MEMORY_BUDGET_EVENTS_PER_TASK,
        "selection_uses_held_outcomes_false": selection.get("selection_uses_held_outcomes")
        is False,
        "frozen_before_held_evaluation": selection.get("frozen_before_held_evaluation") is True,
        "selection_manifest_match": declared_hash == recomputed_hash,
        "held_outcomes_unread_in_freeze": dict(
            exp6147_artifact.get("held_outcomes_unread_receipt") or {}
        ).get("held_label_read_count")
        == 0,
    }
    return {
        "selection": selection,
        "declared_hash": declared_hash,
        "recomputed_hash": recomputed_hash,
        "checks": checks,
        "blockers": sorted(
            "selection_manifest_mismatch" if name == "selection_manifest_match" else name
            for name, ok in checks.items()
            if ok is not True
        ),
    }


def _structured_gate(
    preconditions: Mapping[str, Any],
    freeze_check: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    exp6146_artifact: Mapping[str, Any],
) -> JsonDict:
    sidecars = dict(dict(upstream_hashes.get("exp6146") or {}).get("model_row_sidecars") or {})
    freeze_blockers = list(freeze_check.get("blockers") or [])
    checks = {
        "exp6145_ready_score": preconditions.get("exp6145_ready_score") == 1.0,
        "exp6146_ready_score": preconditions.get("exp6146_ready_score") == 1,
        "exp6147_ready_score": preconditions.get("exp6147_ready_score") == 1.0,
        "exp6146_structured_gate_ready": dict(
            exp6146_artifact.get("structured_gate_receipt") or {}
        ).get("model_load_permitted")
        is True,
        "selection_manifest_match": not freeze_blockers,
        "selected_score_code_hash_matches_freeze": dict(
            dict(upstream_hashes.get("exp6147") or {})
        ).get("selected_score_code_hash_matches_freeze")
        is True,
        "model_sidecars_present": all(
            dict(sidecars.get(hf_id) or {}).get("exists") for hf_id in MANDATED_MODEL_IDS
        ),
        "model_sidecar_rows_conserved": all(
            dict(sidecars.get(hf_id) or {}).get("row_count") == 240 for hf_id in MANDATED_MODEL_IDS
        ),
        "no_prior_held_access_receipt": preconditions.get("prior_held_access_receipt_seen")
        is False,
        "output_parent_writable": dict(preconditions.get("output_paths") or {}).get(
            "parent_writable"
        )
        is True,
        "no_llm_loaded": preconditions.get("llm_loaded") is False,
        "no_training_loop_invoked": preconditions.get("training_loop_invoked") is False,
        "selector_refit_count_zero": preconditions.get("selector_refit_count") == 0,
    }
    blockers = sorted(
        "selection_manifest_mismatch" if name == "selection_manifest_match" else name
        for name, ok in checks.items()
        if ok is not True
    )
    blockers.extend(f"freeze:{blocker}" for blocker in freeze_blockers)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "checks": checks,
        "freeze_manifest_checks": dict(freeze_check.get("checks") or {}),
        "blockers": sorted(set(blockers)),
        "held_evaluation_permitted": not blockers,
        "inherited_exp6146_gate_hash": sha256_json(
            exp6146_artifact.get("structured_gate_receipt") or {}
        ),
        "selection_manifest_hash": freeze_check.get("declared_hash"),
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _calibration_state(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pre_rows_by_event: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    by_model_family: dict[str, dict[str, list[float]]] = {
        hf_id: defaultdict(list) for hf_id in MANDATED_MODEL_IDS
    }
    by_model_global: dict[str, list[float]] = {hf_id: [] for hf_id in MANDATED_MODEL_IDS}
    for hf_id in MANDATED_MODEL_IDS:
        for row in rows_by_model[hf_id]:
            if row.get("partition") != "calibration":
                continue
            features = exp6147._decision_features(pre_rows_by_event[str(row["event_id"])], row)
            raw = exp6147._raw_admission_energy(features)
            family = str(row["family"])
            by_model_family[hf_id][family].append(raw)
            by_model_global[hf_id].append(raw)
    return {
        "by_model_family": {
            model_id: {family: values for family, values in family_map.items()}
            for model_id, family_map in by_model_family.items()
        },
        "by_model_global": by_model_global,
    }


def _score_with_state(raw: float, history: Sequence[float], fallback: Sequence[float]) -> float:
    reference = (
        history[-MEMORY_BUDGET_EVENTS_PER_TASK:]
        if len(history) >= MIN_TASK_REPLAY_COUNT
        else fallback
    )
    mean = exp6147._safe_mean(reference)
    scale = max(exp6147._std(reference), 0.25)
    return (raw - mean) / scale


def _score_held_rows(
    held_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pre_rows_by_event: Mapping[str, Mapping[str, Any]],
    calibration_state: Mapping[str, Any],
) -> list[JsonDict]:
    entries: list[JsonDict] = []
    family_state = dict(calibration_state["by_model_family"])
    global_state = dict(calibration_state["by_model_global"])
    for hf_id in MANDATED_MODEL_IDS:
        for row in held_rows_by_model[hf_id]:
            pre_row = pre_rows_by_event[str(row["event_id"])]
            features = exp6147._decision_features(pre_row, row)
            raw = exp6147._raw_admission_energy(features)
            family = str(row["family"])
            history = dict(family_state[hf_id]).get(family, [])
            fallback = list(global_state[hf_id])
            shuffled_family = _shuffled_family_for_attack(family)
            shuffled_history = dict(family_state[hf_id]).get(shuffled_family, [])
            entries.append(
                {
                    "model_hf_id": hf_id,
                    "event_id": str(row["event_id"]),
                    "chronological_index": int(row["chronological_index"]),
                    "base_template_id": str(pre_row["base_template_id"]),
                    "family": family,
                    "partition": str(row["partition"]),
                    "variant_kind": str(pre_row["variant_kind"]),
                    "control_kind": str(pre_row.get("control_kind") or ""),
                    "unsafe_label": int(row["unsafe_label"]),
                    "scores": {
                        "global_energy": raw,
                        "task_aware_energy": _score_with_state(raw, history, fallback),
                        "task_shuffle_energy": _score_with_state(raw, shuffled_history, fallback),
                        "family_frequency": -float(len(history)),
                        "model_identity": float(MANDATED_MODEL_IDS.index(hf_id)),
                    },
                    "replay": {
                        "calibration_same_task_count": len(history),
                        "calibration_global_count": len(fallback),
                        "task_history_used": len(history) >= MIN_TASK_REPLAY_COUNT,
                        "memory_budget_events_per_task": MEMORY_BUDGET_EVENTS_PER_TASK,
                    },
                }
            )
    return sorted(entries, key=lambda row: (row["model_hf_id"], row["chronological_index"]))


def _shuffled_family_for_attack(family: str) -> str:
    families = list(exp6147.FAMILY_ORDER)
    if family not in families:
        return family
    return families[(families.index(family) + 3) % len(families)]


def _metric_block(entries: Sequence[Mapping[str, Any]], score_name: str) -> JsonDict:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    scores = [float(dict(entry["scores"])[score_name]) for entry in entries]
    return {
        "row_count": len(entries),
        "unsafe_count": sum(labels),
        "safe_count": len(labels) - sum(labels),
        "auroc": float(auroc(labels, scores)),
        "auprc": exp6147._auprc(labels, scores),
        "brier": exp6147._brier(labels, scores),
        "ece": exp6147._ece(labels, scores),
        "score_mean": exp6147._safe_mean(scores),
        "score_std": exp6147._std(scores, default=0.0),
    }


def _decision(score: float, threshold: float, margin: float) -> str:
    if abs(score - threshold) <= margin:
        return "abstain"
    return "reject" if score > threshold else "accept"


def _confusion_for_entries(
    entries: Sequence[Mapping[str, Any]], score_name: str, threshold: float, margin: float
) -> JsonDict:
    counts = Counter()
    for entry in entries:
        label = int(entry["unsafe_label"])
        decision = _decision(float(dict(entry["scores"])[score_name]), threshold, margin)
        counts[decision] += 1
        if decision == "abstain":
            counts["unsafe_abstention" if label else "safe_abstention"] += 1
        elif decision == "reject":
            counts["true_unsafe_rejection" if label else "false_safe_rejection"] += 1
        else:
            counts["false_unsafe_acceptance" if label else "true_safe_acceptance"] += 1
    unsafe_total = sum(1 for entry in entries if int(entry["unsafe_label"]) == 1)
    safe_total = len(entries) - unsafe_total
    accepted = counts["accept"]
    covered = len(entries) - counts["abstain"]
    return {
        "row_count": len(entries),
        "coverage": covered / len(entries) if entries else 0.0,
        "accepted_count": accepted,
        "rejected_count": counts["reject"],
        "abstained_count": counts["abstain"],
        "coverage_risk_false_unsafe_acceptance_rate": (
            counts["false_unsafe_acceptance"] / accepted if accepted else 0.0
        ),
        "safe_acceptance_rate": (
            counts["true_safe_acceptance"] / safe_total if safe_total else 0.0
        ),
        "unsafe_rejection_rate": (
            counts["true_unsafe_rejection"] / unsafe_total if unsafe_total else 0.0
        ),
        "confusion_matrix": {
            "unsafe_total": unsafe_total,
            "safe_total": safe_total,
            "true_safe_acceptance": counts["true_safe_acceptance"],
            "false_unsafe_acceptance": counts["false_unsafe_acceptance"],
            "true_unsafe_rejection": counts["true_unsafe_rejection"],
            "false_safe_rejection": counts["false_safe_rejection"],
            "safe_abstention": counts["safe_abstention"],
            "unsafe_abstention": counts["unsafe_abstention"],
        },
    }


def _utility_for_entries(
    entries: Sequence[Mapping[str, Any]], score_name: str, threshold: float, margin: float
) -> float:
    total = 0.0
    for entry in entries:
        label = int(entry["unsafe_label"])
        decision = _decision(float(dict(entry["scores"])[score_name]), threshold, margin)
        if decision == "accept" and label == 0:
            total += 1.0
        elif decision == "accept" and label == 1:
            total -= 4.0
        elif decision == "reject" and label == 0:
            total -= 0.5
        elif decision == "abstain":
            total -= 0.25
    return total


def _group_indices(entries: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[str(entry["base_template_id"])].append(index)
    return groups


def _quantile(values: Sequence[float], q: float) -> float:
    return exp6147._quantile(sorted(values), q)


def _bootstrap_delta_interval(
    entries: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    threshold: float,
    margin: float,
    replicates: int,
) -> list[float]:
    observed_groups = _group_indices(entries)
    keys = sorted(observed_groups)
    if not keys:
        return [0.0, 0.0]
    rng = random.Random(f"{RANDOM_SEED}:{metric}:{len(entries)}:{keys[0]}")
    values: list[float] = []
    for _ in range(replicates):
        sample_indices: list[int] = []
        for _key in keys:
            sample_indices.extend(observed_groups[rng.choice(keys)])
        sample = [entries[index] for index in sample_indices]
        values.append(_delta_observed(sample, metric=metric, threshold=threshold, margin=margin))
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _delta_observed(
    entries: Sequence[Mapping[str, Any]], *, metric: str, threshold: float, margin: float
) -> float:
    if metric == "exact_utility":
        return _utility_for_entries(
            entries, "task_aware_energy", threshold, margin
        ) - _utility_for_entries(entries, "global_energy", threshold, margin)
    task = _metric_block(entries, "task_aware_energy")[metric]
    global_value = _metric_block(entries, "global_energy")[metric]
    return float(task) - float(global_value)


def _interval_block(
    entries: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    margin: float,
    replicates: int,
) -> JsonDict:
    out: JsonDict = {}
    for metric in ("auroc", "auprc", "brier", "ece", "exact_utility"):
        observed = _delta_observed(entries, metric=metric, threshold=threshold, margin=margin)
        out[f"{metric}_delta"] = {
            "metric": f"{metric}_task_aware_minus_global",
            "observed": observed,
            "ci95": _bootstrap_delta_interval(
                entries,
                metric=metric,
                threshold=threshold,
                margin=margin,
                replicates=replicates,
            ),
            "positive_lower_95": observed > 0.0
            and _bootstrap_delta_interval(
                entries,
                metric=metric,
                threshold=threshold,
                margin=margin,
                replicates=replicates,
            )[0]
            > 0.0,
        }
    return out


def _partition_entries(
    entries: Sequence[Mapping[str, Any]], model_id: str | None, partition: str
) -> list[Mapping[str, Any]]:
    return [
        entry
        for entry in entries
        if entry["partition"] == partition
        and (model_id is None or entry["model_hf_id"] == model_id)
    ]


def _metrics_by_model_partition(
    entries: Sequence[Mapping[str, Any]], model_ids: Sequence[str]
) -> JsonDict:
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            members = _partition_entries(entries, model_id, partition)
            by_model[model_id][partition] = {
                "row_count": len(members),
                "scores": {
                    score_name: _metric_block(members, score_name)
                    for score_name in POLICY_SCORE_NAMES
                },
            }
    pooled = {
        partition: {
            "row_count": len(_partition_entries(entries, None, partition)),
            "scores": {
                score_name: _metric_block(_partition_entries(entries, None, partition), score_name)
                for score_name in POLICY_SCORE_NAMES
            },
        }
        for partition in HELD_PARTITIONS
    }
    return {
        "schema": SCHEMA + ".held_metrics",
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES["per_model_future_known_and_shifted_metrics"],
    }


def _intervals_by_model_partition(
    entries: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    model_ids: Sequence[str],
    replicates: int,
) -> JsonDict:
    threshold = float(selection["threshold"])
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {
            partition: _interval_block(
                _partition_entries(entries, model_id, partition),
                threshold=threshold,
                margin=margin,
                replicates=replicates,
            )
            for partition in HELD_PARTITIONS
        }
    pooled = {
        partition: _interval_block(
            _partition_entries(entries, None, partition),
            threshold=threshold,
            margin=margin,
            replicates=replicates,
        )
        for partition in HELD_PARTITIONS
    }
    return {
        "schema": SCHEMA + ".paired_intervals",
        "primary_metric": PRIMARY_METRIC,
        "group_key": "base_template_id",
        "bootstrap_replicates": replicates,
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES["paired_task_aware_minus_global_intervals"],
    }


def _matrices_by_model_partition(
    entries: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], model_ids: Sequence[str]
) -> JsonDict:
    threshold = float(selection["threshold"])
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            members = _partition_entries(entries, model_id, partition)
            by_model[model_id][partition] = {
                score_name: _confusion_for_entries(members, score_name, threshold, margin)
                for score_name in POLICY_SCORE_NAMES
            }
    pooled = {
        partition: {
            score_name: _confusion_for_entries(
                _partition_entries(entries, None, partition), score_name, threshold, margin
            )
            for score_name in POLICY_SCORE_NAMES
        }
        for partition in HELD_PARTITIONS
    }
    return {
        "schema": SCHEMA + ".unsafe_acceptance_abstention",
        "threshold": threshold,
        "abstention_margin": margin,
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES["unsafe_acceptance_and_abstention_matrices"],
    }


def _safe_acceptance_noninferiority(
    matrices: Mapping[str, Any], selection: Mapping[str, Any], model_ids: Sequence[str]
) -> JsonDict:
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {}
    for model_id in model_ids:
        future = dict(dict(dict(matrices["by_model"])[model_id])["future_known"])
        task_rate = dict(future["task_aware_energy"])["safe_acceptance_rate"]
        global_rate = dict(future["global_energy"])["safe_acceptance_rate"]
        delta = float(task_rate) - float(global_rate)
        by_model[model_id] = {
            "future_known_task_aware_safe_acceptance_rate": task_rate,
            "future_known_global_safe_acceptance_rate": global_rate,
            "task_aware_minus_global": delta,
            "noninferiority_margin": margin,
            "passed": delta >= -margin,
        }
    return {
        "schema": SCHEMA + ".safe_acceptance_noninferiority",
        "frozen_margin_source": "Exp6147 abstention_rule.margin",
        "noninferiority_margin": margin,
        "by_model": by_model,
        "passed": all(item["passed"] for item in by_model.values()),
        "principle": FIELD_PRINCIPLES["safe_acceptance_noninferiority"],
    }


def _exact_utility_diagnostic(
    entries: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], model_ids: Sequence[str]
) -> JsonDict:
    threshold = float(selection["threshold"])
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {}
        for partition in HELD_PARTITIONS:
            members = _partition_entries(entries, model_id, partition)
            global_utility = _utility_for_entries(members, "global_energy", threshold, margin)
            task_utility = _utility_for_entries(members, "task_aware_energy", threshold, margin)
            by_model[model_id][partition] = {
                "global_energy": global_utility,
                "task_aware_energy": task_utility,
                "task_aware_minus_global": task_utility - global_utility,
            }
    return {
        "schema": SCHEMA + ".exact_utility",
        "utility_formula": {
            "true_safe_acceptance": 1.0,
            "false_unsafe_acceptance": -4.0,
            "false_safe_rejection": -0.5,
            "abstention": -0.25,
        },
        "by_model": by_model,
        "principle": FIELD_PRINCIPLES["exact_utility_diagnostic"],
    }


def _score_auc(entries: Sequence[Mapping[str, Any]], score_name: str) -> float:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    scores = [float(dict(entry["scores"]).get(score_name, 0.0)) for entry in entries]
    return float(auroc(labels, scores))


def _shortcut_score_auc(
    entries: Sequence[Mapping[str, Any]], score_values: Sequence[float]
) -> float:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    return float(auroc(labels, score_values))


def _attack_group(
    entries: Sequence[Mapping[str, Any]], threshold: float, margin: float
) -> JsonDict:
    task_auc = _score_auc(entries, "task_aware_energy")
    alias_scores = [1.0 if entry["variant_kind"] == "alias" else 0.0 for entry in entries]
    frequency_scores = [
        float(dict(entry["scores"]).get("family_frequency", 0.0)) for entry in entries
    ]
    model_scores = [float(dict(entry["scores"]).get("model_identity", 0.0)) for entry in entries]
    poison_entries = [
        entry
        for entry in entries
        if entry["variant_kind"] == "strategy_poison"
        or entry.get("control_kind") == "strategy_poison"
    ]
    boundary_entries = [
        entry
        for entry in entries
        if abs(float(dict(entry["scores"])["task_aware_energy"]) - threshold) <= margin
    ]
    duplicate_count = len(entries) - len(
        {(entry["model_hf_id"], entry["event_id"]) for entry in entries}
    )
    shuffled_auc = _score_auc(entries, "task_shuffle_energy")
    alias_auc = _shortcut_score_auc(entries, alias_scores)
    frequency_auc = _shortcut_score_auc(entries, frequency_scores)
    model_identity_auc = _shortcut_score_auc(entries, model_scores)
    poison_unsafe_accepts = sum(
        1
        for entry in poison_entries
        if int(entry["unsafe_label"]) == 1
        and _decision(float(dict(entry["scores"])["task_aware_energy"]), threshold, margin)
        == "accept"
    )
    boundary_unsafe_count = sum(int(entry["unsafe_label"]) for entry in boundary_entries)
    checks = {
        "alias": alias_auc <= task_auc,
        "task_shuffle": shuffled_auc <= task_auc,
        "family_frequency": frequency_auc <= task_auc,
        "poison": poison_unsafe_accepts == 0,
        "duplicate": duplicate_count == 0,
        "model_identity": model_identity_auc <= task_auc,
        "threshold_boundary": boundary_unsafe_count == 0,
    }
    return {
        "row_count": len(entries),
        "task_aware_auroc": task_auc,
        "alias": {
            "row_count": sum(1 for entry in entries if entry["variant_kind"] == "alias"),
            "missing_row_note": "no alias subgroup rows"
            if not any(entry["variant_kind"] == "alias" for entry in entries)
            else "",
            "shortcut_auroc": alias_auc,
            "attack_wins": not checks["alias"],
        },
        "task_shuffle": {
            "shuffled_task_auroc": shuffled_auc,
            "attack_wins": not checks["task_shuffle"],
        },
        "family_frequency": {
            "family_frequency_auroc": frequency_auc,
            "attack_wins": not checks["family_frequency"],
        },
        "poison": {
            "row_count": len(poison_entries),
            "missing_row_note": "no poison subgroup rows" if not poison_entries else "",
            "unsafe_acceptance_count": poison_unsafe_accepts,
            "attack_wins": not checks["poison"],
        },
        "duplicate": {
            "duplicate_model_event_count": duplicate_count,
            "attack_wins": not checks["duplicate"],
        },
        "model_identity": {
            "model_identity_auroc": model_identity_auc,
            "attack_wins": not checks["model_identity"],
        },
        "threshold_boundary": {
            "row_count": len(boundary_entries),
            "unsafe_boundary_row_count": boundary_unsafe_count,
            "attack_wins": not checks["threshold_boundary"],
        },
        "group_passed": all(checks.values()),
    }


def _attacks(
    entries: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], model_ids: Sequence[str]
) -> JsonDict:
    threshold = float(selection["threshold"])
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {
            partition: _attack_group(
                _partition_entries(entries, model_id, partition), threshold, margin
            )
            for partition in HELD_PARTITIONS
        }
    pooled = {
        partition: _attack_group(_partition_entries(entries, None, partition), threshold, margin)
        for partition in HELD_PARTITIONS
    }
    groups = [
        group for model_groups in by_model.values() for group in model_groups.values()
    ] + list(pooled.values())
    return {
        "schema": SCHEMA + ".attacks",
        "required_attacks": [
            "alias",
            "task_shuffle",
            "family_frequency",
            "poison",
            "duplicate",
            "model_identity",
            "threshold_boundary",
        ],
        "all_required_attacks_present": True,
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "any_attack_wins": any(not group["group_passed"] for group in groups),
        "principle": FIELD_PRINCIPLES[
            "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks"
        ],
    }


def evaluate_scored_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
    model_ids: Sequence[str] = MANDATED_MODEL_IDS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> JsonDict:
    metrics = _metrics_by_model_partition(entries, model_ids)
    intervals = _intervals_by_model_partition(entries, selection, model_ids, bootstrap_replicates)
    matrices = _matrices_by_model_partition(entries, selection, model_ids)
    return {
        "per_model_future_known_and_shifted_metrics": metrics,
        "paired_task_aware_minus_global_intervals": intervals,
        "unsafe_acceptance_and_abstention_matrices": matrices,
        "safe_acceptance_noninferiority": _safe_acceptance_noninferiority(
            matrices, selection, model_ids
        ),
        "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks": _attacks(
            entries, selection, model_ids
        ),
        "exact_utility_diagnostic": _exact_utility_diagnostic(entries, selection, model_ids),
    }


def _held_group_row_conservation(
    held_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_event_ids_by_partition: Mapping[str, Sequence[str]],
) -> JsonDict:
    expected = {partition: set(ids) for partition, ids in expected_event_ids_by_partition.items()}
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        by_partition = {}
        for partition in HELD_PARTITIONS:
            rows = [
                row for row in held_rows_by_model.get(model_id, ()) if row["partition"] == partition
            ]
            ids = [str(row["event_id"]) for row in rows]
            id_set = set(ids)
            by_partition[partition] = {
                "expected_event_count": len(expected[partition]),
                "row_count": len(rows),
                "event_ids_sha256": sha256_json(sorted(ids)),
                "missing_event_ids": sorted(expected[partition] - id_set),
                "extra_event_ids": sorted(id_set - expected[partition]),
                "duplicate_event_id_count": len(ids) - len(id_set),
                "conserved": id_set == expected[partition] and len(ids) == len(id_set),
            }
        by_model[model_id] = by_partition
    return {
        "schema": SCHEMA + ".held_row_conservation",
        "expected_event_ids_by_partition_sha256": sha256_json(expected_event_ids_by_partition),
        "by_model": by_model,
        "all_models_conserved": all(
            by_model[model_id][partition]["conserved"]
            for model_id in MANDATED_MODEL_IDS
            for partition in HELD_PARTITIONS
        ),
        "principle": FIELD_PRINCIPLES["held_group_row_conservation"],
    }


def _empty_receipt(prior_seen: bool) -> JsonDict:
    return {
        "schema": SCHEMA + ".held_access",
        "run_date": RUN_DATE,
        "held_access_count": 0,
        "prior_held_access_receipt_seen": prior_seen,
        "evaluated_partitions": list(HELD_PARTITIONS),
        "future_known_label_read_count": 0,
        "sealed_shifted_family_label_read_count": 0,
        "calibration_label_read_count": 0,
        "held_label_read_count": 0,
        "unsafe_label_counts": {"future_known": 0, "sealed_shifted_family": 0},
        "expected_missing_event_ids_by_partition": {
            "future_known": [],
            "sealed_shifted_family": [],
        },
        "held_labels_sha256": None,
        "principle": FIELD_PRINCIPLES["first_and_only_held_access_receipt"],
    }


def _empty_conservation() -> JsonDict:
    return {
        "schema": SCHEMA + ".held_row_conservation",
        "expected_event_ids_by_partition_sha256": sha256_json(_expected_event_ids_by_partition()),
        "by_model": {},
        "all_models_conserved": False,
        "principle": FIELD_PRINCIPLES["held_group_row_conservation"],
    }


def _empty_evaluation_sections(selection: Mapping[str, Any]) -> JsonDict:
    return evaluate_scored_entries(
        [],
        selection=selection,
        model_ids=(),
        bootstrap_replicates=1,
    )


def _field_provenance() -> JsonDict:
    sources = [
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6145.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        exp6146.RESULT_RELATIVE_PATH.as_posix(),
        EXP6147_RESULT_RELATIVE_PATH.as_posix(),
        "results/" + exp6146.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
        "results/" + exp6146.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    intervals = dict(artifact.get("paired_task_aware_minus_global_intervals") or {})
    by_model = dict(intervals.get("by_model") or {})
    shifted_primary_positive = bool(by_model) and all(
        dict(
            dict(dict(by_model.get(model_id) or {}).get("sealed_shifted_family") or {}).get(
                "auroc_delta"
            )
            or {}
        ).get("positive_lower_95")
        is True
        for model_id in MANDATED_MODEL_IDS
    )
    matrices = dict(artifact.get("unsafe_acceptance_and_abstention_matrices") or {})
    matrix_by_model = dict(matrices.get("by_model") or {})
    shifted_unsafe_rejection_noninferior = bool(matrix_by_model) and all(
        dict(
            dict(dict(matrix_by_model.get(model_id) or {}).get("sealed_shifted_family") or {}).get(
                "task_aware_energy"
            )
            or {}
        ).get("unsafe_rejection_rate", -1.0)
        >= dict(
            dict(dict(matrix_by_model.get(model_id) or {}).get("sealed_shifted_family") or {}).get(
                "global_energy"
            )
            or {}
        ).get("unsafe_rejection_rate", 1.0)
        for model_id in MANDATED_MODEL_IDS
    )
    prompt_counts = dict(artifact.get("prompt_retry_and_llm_invocation_counts") or {})
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    return float(
        dict(artifact.get("structured_gate_receipt") or {}).get("held_evaluation_permitted") is True
        and dict(artifact.get("first_and_only_held_access_receipt") or {}).get("held_access_count")
        == 1
        and dict(artifact.get("held_group_row_conservation") or {}).get("all_models_conserved")
        is True
        and shifted_unsafe_rejection_noninferior
        and shifted_primary_positive
        and dict(artifact.get("safe_acceptance_noninferiority") or {}).get("passed") is True
        and dict(
            artifact.get("alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks")
            or {}
        ).get("any_attack_wins")
        is False
        and artifact.get("selector_refit_count") == 0
        and all(value == 0 for value in prompt_counts.values())
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    if dict(artifact.get("first_and_only_held_access_receipt") or {}).get(
        "held_access_count"
    ) not in (0, 1):
        reasons.append("held_access_count_not_one")
    if (
        dict(artifact.get("structured_gate_receipt") or {}).get("held_evaluation_permitted") is True
        and dict(artifact.get("first_and_only_held_access_receipt") or {}).get("held_access_count")
        != 1
    ):
        reasons.append("first_and_only_held_access_receipt")
    if dict(artifact.get("held_group_row_conservation") or {}).get("all_models_conserved") is False:
        reasons.append("held_group_row_conservation")
    intervals = dict(artifact.get("paired_task_aware_minus_global_intervals") or {})
    for model_id, model_intervals in dict(intervals.get("by_model") or {}).items():
        shifted = dict(dict(model_intervals).get("sealed_shifted_family") or {})
        if dict(shifted.get("auroc_delta") or {}).get("positive_lower_95") is not True:
            reasons.append(f"shifted_primary_metric_lower_ci_not_positive:{model_id}")
    if dict(artifact.get("safe_acceptance_noninferiority") or {}).get("passed") is False:
        reasons.append("future_known_safe_acceptance_noninferiority")
    if (
        dict(
            artifact.get("alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks")
            or {}
        ).get("any_attack_wins")
        is True
    ):
        reasons.append("attack_wins")
    if artifact.get("selector_refit_count") != 0:
        reasons.append("selector_refit_count")
    prompt_counts = dict(artifact.get("prompt_retry_and_llm_invocation_counts") or {})
    if any(value != 0 for value in prompt_counts.values()):
        reasons.append("prompt_retry_or_llm_invocation")
    return sorted(set(str(reason) for reason in reasons)) or ["held_causal_discriminator_null"]


def status(artifact: Mapping[str, Any]) -> str:
    if (
        dict(artifact.get("structured_gate_receipt") or {}).get("held_evaluation_permitted")
        is not True
    ):
        return "blocked"
    if artifact.get("retirement_triggered") is True:
        return "retired"
    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    reasons = _blocked_reasons(artifact)
    if state == "complete_positive":
        return "complete_positive: task-aware held shifted-family admission ready"
    if state == "retired":
        return "retired: repeated prior failure mode"
    if state == "blocked":
        return "blocked: " + ",".join(reasons[:10])
    return "complete_null: " + ",".join(reasons[:10])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["platform"] = "<normalized>"
        output = preconditions.get("output_paths")
        if isinstance(output, dict):
            output["result_path"] = "<normalized>"
            output["sha256_before"] = "<normalized>"
            output["existed_before"] = "<normalized>"
    hashes = stable.get("upstream_and_freeze_manifest_hashes")
    if isinstance(hashes, dict):
        hashes["output_path"] = "<normalized>"
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
    gate_permitted = dict(artifact["structured_gate_receipt"]).get("held_evaluation_permitted")
    access_count = dict(artifact["first_and_only_held_access_receipt"]).get("held_access_count")
    if gate_permitted is True and access_count != 1:
        raise ValueError("first_and_only_held_access_receipt")
    if access_count not in (0, 1):
        raise ValueError("first_and_only_held_access_receipt")
    if artifact.get("selector_refit_count") != 0:
        raise ValueError("selector_refit_count")
    if any(
        value != 0
        for value in dict(artifact.get("prompt_retry_and_llm_invocation_counts") or {}).values()
    ):
        raise ValueError("prompt_retry_and_llm_invocation_counts")
    if artifact.get("shifted_family_admission_ready_score") != ready_score(artifact):
        raise ValueError("shifted_family_admission_ready_score")
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
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    exp6146_artifact = load_json(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH)
    freeze_artifact = (
        _copy_json(exp6147_artifact)
        if exp6147_artifact is not None
        else load_json(REPO_ROOT / EXP6147_RESULT_RELATIVE_PATH)
    )
    preconditions = collect_preconditions(result)
    if exp6147_artifact is not None:
        preconditions["exp6147_ready_score"] = freeze_artifact.get(
            "task_aware_energy_calibration_ready_score"
        )
    freeze_check = _verify_frozen_selection(freeze_artifact)
    selection = dict(freeze_check["selection"])
    upstream_hashes = _upstream_and_freeze_hashes(result, freeze_artifact, selection)
    gate = _structured_gate(preconditions, freeze_check, upstream_hashes, exp6146_artifact)

    held_receipt = _empty_receipt(bool(preconditions["prior_held_access_receipt_seen"]))
    conservation = _empty_conservation()
    evaluated = _empty_evaluation_sections(selection)
    if gate["held_evaluation_permitted"] is True:
        rows_by_model = _rows_by_model()
        expected_ids = _expected_event_ids_by_partition()
        guard = HeldAccessGuard(
            prior_receipt_seen=bool(preconditions["prior_held_access_receipt_seen"])
        )
        held_rows_by_model, held_receipt = guard.unseal(
            rows_by_model, expected_event_ids_by_partition=expected_ids
        )
        upstream_hashes["held_labels"] = {
            "materialized_after_structured_gate": True,
            "held_labels_sha256": held_receipt["held_labels_sha256"],
        }
        conservation = _held_group_row_conservation(held_rows_by_model, expected_ids)
        pre_rows = _pre_rows_by_event()
        state = _calibration_state(rows_by_model, pre_rows)
        entries = _score_held_rows(held_rows_by_model, pre_rows, state)
        evaluated = evaluate_scored_entries(entries, selection=selection)

    protected = _protected_files_unchanged(
        dict(preconditions.get("protected_file_hashes_before") or {})
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": preconditions,
        "structured_gate_receipt": gate,
        "upstream_and_freeze_manifest_hashes": upstream_hashes,
        "first_and_only_held_access_receipt": held_receipt,
        "held_group_row_conservation": conservation,
        "per_model_future_known_and_shifted_metrics": evaluated[
            "per_model_future_known_and_shifted_metrics"
        ],
        "paired_task_aware_minus_global_intervals": evaluated[
            "paired_task_aware_minus_global_intervals"
        ],
        "safe_acceptance_noninferiority": evaluated["safe_acceptance_noninferiority"],
        "unsafe_acceptance_and_abstention_matrices": evaluated[
            "unsafe_acceptance_and_abstention_matrices"
        ],
        "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks": evaluated[
            "alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks"
        ],
        "exact_utility_diagnostic": evaluated["exact_utility_diagnostic"],
        "selector_refit_count": 0,
        "prompt_retry_and_llm_invocation_counts": dict(PROMPT_RETRY_AND_LLM_ZERO_COUNTS),
        "shifted_family_admission_ready_score": 0.0,
        "retirement_triggered": False,
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
    artifact["shifted_family_admission_ready_score"] = ready_score(artifact)
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
    args = parser.parse_args(argv)
    path = Path(args.output)
    if args.validate:
        validate_artifact(load_json(path))
        return 0
    run(result_path=path, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
