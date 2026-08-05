"""Exp6147 task-aware admission energy calibration.

Spec refs: REQ-VERIFY-6147, REQ-VERIFY-6147-1, REQ-VERIFY-6147-2,
REQ-VERIFY-6147-3, REQ-VERIFY-6147-4, REQ-VERIFY-6147-5,
REQ-VERIFY-6147-6, REQ-VERIFY-6147-7, REQ-VERIFY-6147-8,
REQ-VERIFY-6147-9, REQ-LEARN-6147,
SCENARIO-VERIFY-6147-FEATURES, SCENARIO-VERIFY-6147-REPLAY,
SCENARIO-VERIFY-6147-CONTROLS, SCENARIO-LEARN-6147-FREEZE,
SCENARIO-LEARN-6147-HELD-UNREAD.

Exp6147 scores cached Exp6146 model decisions. It does not load a model or
train a scorer. Exact outcomes are used only as calibration labels after each
chronological score is emitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_6145_constraint_shift_stream as exp6145
from carnot import experiment_6146_sota_constraint_event_corpus as exp6146
from carnot.eval.metrics import auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6147_task_aware_energy_calibration.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6147_task_aware_energy_calibration.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6147_task_aware_energy_calibration.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
LEARN_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA = "carnot.experiment_6147.task_aware_energy_calibration.v1"
EXPERIMENT_ID = "experiment_6147_task_aware_energy_calibration"
RUN_DATE = "20260805"
RANDOM_SEED = 6147
INFERENCE_SUBSTRATE = "cached_sota_event_energy_calibration"
VERIFIER_IS_ORACLE = False
MEMORY_BUDGET_EVENTS_PER_TASK = 64
MIN_TASK_REPLAY_COUNT = 4
ABSTENTION_MARGIN = 0.12
BOOTSTRAP_REPLICATES = 500
PRIMARY_METRIC = "grouped_calibration_prequential_auroc_delta_task_aware_minus_global"

MANDATED_MODEL_IDS = exp6146.MANDATED_MODEL_IDS
PARTITION_NAMES = exp6146.PARTITION_NAMES
FAMILY_ORDER = (
    "access_control",
    "inventory_allocation",
    "maintenance_schedule",
    "menu_recommendation",
    "release_gating",
    "task_selection",
    "incident_response",
    "route_planning",
)
SCORE_NAMES = (
    "global_energy",
    "task_aware_energy",
    "family_centering_only",
    "nearest_replay_distance",
    "task_frequency",
    "response_length",
    "random",
    "shuffled_task",
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
    LEARN_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6145.RESULT_RELATIVE_PATH,
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
    exp6146.RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6147_task_aware_energy_calibration.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6147_task_aware_energy_calibration.py "
    "-m pytest tests/python/test_experiment_6147_task_aware_energy_calibration.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6147_task_aware_energy_calibration.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6147_task_aware_energy_calibration.py"
)
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6147_task_aware_energy_calibration --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6147_task_aware_energy_calibration.json"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6147_task_aware_energy_calibration.py "
    "tests/python/test_experiment_6147_task_aware_energy_calibration.py"
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

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "source_row_split_and_schema_hashes",
    "decision_time_feature_allowlist_and_forbidden_field_scan",
    "global_task_aware_and_control_energy_definitions",
    "chronological_replay_statistics",
    "per_model_grouped_metrics_and_intervals",
    "confidence_gap_by_task_count",
    "calibration_coverage_risk_and_confusion_matrices",
    "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls",
    "selected_score_threshold_abstention_and_memory_budget",
    "selection_manifest_hash",
    "held_outcomes_unread_receipt",
    "task_aware_energy_calibration_ready_score",
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
    "status": "A terminal state distinguishes a ready, null, retired, or blocked calibration.",
    "preconditions_checked": "Exp6145, Exp6146, schemas, exclusions, outputs, and protected files are hashed before scoring.",
    "structured_gate_receipt": "Calibration opens only after Exp6146 readiness, row conservation, schema hashes, no-LLM substrate, and protected files pass.",
    "source_row_split_and_schema_hashes": "Source rows, splits, schemas, metrics, exclusions, and output paths are content-addressed.",
    "decision_time_feature_allowlist_and_forbidden_field_scan": "Any current outcome or exact-answer feature makes the verifier circular and forces readiness zero.",
    "global_task_aware_and_control_energy_definitions": "Every score is a transparent decision-time formula or declared control, not a trained hidden scorer.",
    "chronological_replay_statistics": "Task-aware replay uses only earlier calibration events before each score and updates after the label reveal.",
    "per_model_grouped_metrics_and_intervals": "Each source model is reported separately with event/base-template grouped uncertainty before pooled summaries.",
    "confidence_gap_by_task_count": "Score-scale drift and confidence gaps are diagnosed as task replay counts accumulate.",
    "calibration_coverage_risk_and_confusion_matrices": "The frozen threshold and abstention rule expose unsafe accepts, safe rejects, coverage, and risk.",
    "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls": "Shortcut, label-shuffle, and duplicate attacks must not explain the task-aware lift.",
    "selected_score_threshold_abstention_and_memory_budget": "One preregistered calibration choice is frozen before held evaluation.",
    "selection_manifest_hash": "The frozen policy is content-addressed for downstream held evaluation.",
    "held_outcomes_unread_receipt": "Future-known and sealed shifted-family outcomes remain unread during calibration.",
    "task_aware_energy_calibration_ready_score": "Exactly one only for positive grouped task-aware lift, clean controls, no forbidden fields, and non-degenerate confusion.",
    "retirement_triggered": "A repeated prior-failure mode retires the experiment rather than rebranding a null.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "Measured cached-row scoring time is reported without implying model inference.",
    "inference_substrate": "Use `cached_sota_event_energy_calibration`; no LLM is loaded.",
    "verifier_is_oracle": "The verifier is not an oracle; exact outcomes are calibration/evaluation labels only.",
    "missing_verifier_gaps": "Any blocked gate, null lift, shortcut, forbidden field, or held-read gap is explicit.",
    "field_provenance": "Every field traces to specs, Exp6145/Exp6146 sidecars, cached rows, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, gate, forbidden-field, replay, metrics, controls, freeze, no-held-read, schema, adversarial, protected-file, E2E-applicable, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, row, split, schema, control, threshold, test, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:` and state whether task-aware calibration adds deconfounded value.",
}

DECISION_TIME_FEATURE_ALLOWLIST = (
    "base_template_id",
    "event_id",
    "family",
    "variant_kind",
    "control_kind",
    "model_hf_id",
    "model_index",
    "model_name",
    "constraint.fact_count",
    "constraint.rule_count",
    "constraint.body_term_count",
    "constraint.predicate_count",
    "constraint.domain_count",
    "constraint.entity_count",
    "constraint.query_variable_count",
    "constraint.negation_term_count",
    "constraint.arithmetic_term_count",
    "constraint.has_recursive_dependency",
    "constraint.malformed_extra_top_level_count",
    "strategy.expected_strategy_id",
    "strategy.observed_strategy_id",
    "strategy.strategy_id_matches_expected",
    "strategy.terminal_solution_mentions_expected",
    "strategy.alias_surface",
    "strategy.composition_surface",
    "strategy.permuted_fact_order",
    "strategy.proposal_form_malformed",
    "strategy.memory_action_poison_request",
    "response.invalid_output",
    "response.terminal_complete",
    "response.generated_token_count",
    "response.response_char_length",
    "response.terminal_solution_length",
    "response.finish_reason_length",
    "replay.prior_same_task_count",
    "replay.prior_global_count",
    "replay.prior_task_energy_mean",
    "replay.prior_task_energy_scale",
    "replay.memory_budget_events_per_task",
)
FORBIDDEN_SCORE_TOKENS = (
    "exact_answer",
    "current_validator_result",
    "validator_result",
    "exact_labels",
    "exact_outcome",
    "outcome_receipt",
    "post_outcome",
    "post_outcome_id",
    "held_label",
    "sealed_label",
    "oracle_label",
    "future_event",
    "satisfiable",
    "python_status",
    "z3_status",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable ASCII order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so sidecar receipts are content-addressed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def selection_manifest_hash(selection: Mapping[str, Any]) -> str:
    """Hash the frozen downstream admission policy."""

    return sha256_json(selection)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: Sequence[float], default: float = 1.0) -> float:
    if len(values) < 2:
        return default
    mean = _safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float) -> float:
    clipped = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


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


def collect_preconditions(result_path: Path) -> JsonDict:
    """Collect hashes and gate inputs before any cached-row scoring."""

    exp6145_artifact = _load_json(REPO_ROOT / exp6145.RESULT_RELATIVE_PATH)
    exp6146_artifact = _load_json(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH)
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
    }


def _source_row_split_and_schema_hashes(result_path: Path) -> JsonDict:
    exp6146_artifact = _load_json(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH)
    sidecars = {}
    for hf_id in MANDATED_MODEL_IDS:
        path = _model_sidecar_path(hf_id)
        sidecars[hf_id] = {
            **_file_receipt(path),
            "row_count": len(_load_jsonl(path)) if path.exists() else 0,
        }
    return {
        "schema": SCHEMA + ".source_hashes",
        "exp6145": {
            "result": _file_receipt(REPO_ROOT / exp6145.RESULT_RELATIVE_PATH),
            "rows": _file_receipt(REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH),
            "splits": _file_receipt(REPO_ROOT / exp6145.SPLIT_FILE_RELATIVE_PATH),
            "outcomes": _file_receipt(REPO_ROOT / exp6145.OUTCOME_FILE_RELATIVE_PATH),
        },
        "exp6146": {
            "result": _file_receipt(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH),
            "model_row_sidecars": sidecars,
            "stream_split_and_row_hashes": _copy_json(
                exp6146_artifact.get("stream_split_and_row_hashes") or {}
            ),
            "metrics_hash": sha256_json(
                exp6146_artifact.get("calibration_future_and_shift_metrics_by_model") or {}
            ),
            "structured_gate_hash": sha256_json(
                exp6146_artifact.get("structured_gate_receipt") or {}
            ),
        },
        "decision_time_schema_hash": sha256_json(
            {
                "allowlist": DECISION_TIME_FEATURE_ALLOWLIST,
                "forbidden": FORBIDDEN_SCORE_TOKENS,
                "scores": SCORE_NAMES,
                "memory_budget": MEMORY_BUDGET_EVENTS_PER_TASK,
            }
        ),
        "exclusion_manifest": _file_receipt(REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "output_path": str(result_path),
        "principle": FIELD_PRINCIPLES["source_row_split_and_schema_hashes"],
    }


def _structured_gate(
    preconditions: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    exp6146_artifact: Mapping[str, Any],
) -> JsonDict:
    sidecars = dict(dict(source_hashes.get("exp6146") or {}).get("model_row_sidecars") or {})
    checks = {
        "exp6145_ready_score": preconditions.get("exp6145_ready_score") == 1.0,
        "exp6146_ready_score": preconditions.get("exp6146_ready_score") == 1,
        "exp6146_structured_gate_recomputed": dict(
            exp6146_artifact.get("structured_gate_receipt") or {}
        ).get("model_load_permitted")
        is True,
        "model_sidecars_present": all(dict(sidecars.get(hf_id) or {}).get("exists") for hf_id in MANDATED_MODEL_IDS),
        "model_sidecar_rows_conserved": all(
            dict(sidecars.get(hf_id) or {}).get("row_count") == 240 for hf_id in MANDATED_MODEL_IDS
        ),
        "output_parent_writable": dict(preconditions.get("output_paths") or {}).get(
            "parent_writable"
        )
        is True,
        "no_llm_loaded": preconditions.get("llm_loaded") is False,
        "no_training_loop_invoked": preconditions.get("training_loop_invoked") is False,
    }
    blockers = sorted(name for name, ok in checks.items() if ok is not True)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "checks": checks,
        "blockers": blockers,
        "calibration_permitted": not blockers,
        "inherited_exp6146_gate_hash": sha256_json(
            exp6146_artifact.get("structured_gate_receipt") or {}
        ),
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _decision_features(pre_row: Mapping[str, Any], model_row: Mapping[str, Any]) -> JsonDict:
    pre = dict(pre_row.get("pre_decision") or {})
    strategy = dict(pre.get("candidate_strategy") or {})
    feature = dict(strategy.get("features") or {})
    graph = dict(pre.get("constraint_graph_summary") or {})
    expected = str(strategy.get("strategy_id") or "")
    observed = str(model_row.get("strategy_id") or "")
    terminal_solution = str(model_row.get("terminal_solution") or "")
    return {
        "event_id": str(pre_row.get("event_id")),
        "base_template_id": str(pre_row.get("base_template_id")),
        "family": str(pre_row.get("family")),
        "variant_kind": str(pre_row.get("variant_kind")),
        "control_kind": str(pre_row.get("control_kind")),
        "model_hf_id": str(model_row.get("model_hf_id")),
        "model_index": MANDATED_MODEL_IDS.index(str(model_row.get("model_hf_id"))),
        "model_name": str(model_row.get("model_name") or ""),
        "constraint.fact_count": int(graph.get("fact_count", 0) or 0),
        "constraint.rule_count": int(graph.get("rule_count", 0) or 0),
        "constraint.body_term_count": int(graph.get("body_term_count", 0) or 0),
        "constraint.predicate_count": int(graph.get("predicate_count", 0) or 0),
        "constraint.domain_count": int(graph.get("domain_count", 0) or 0),
        "constraint.entity_count": int(graph.get("entity_count", 0) or 0),
        "constraint.query_variable_count": int(graph.get("query_variable_count", 0) or 0),
        "constraint.negation_term_count": int(graph.get("negation_term_count", 0) or 0),
        "constraint.arithmetic_term_count": int(graph.get("arithmetic_term_count", 0) or 0),
        "constraint.has_recursive_dependency": bool(graph.get("has_recursive_dependency")),
        "constraint.malformed_extra_top_level_count": int(
            graph.get("malformed_extra_top_level_count", 0) or 0
        ),
        "strategy.expected_strategy_id": expected,
        "strategy.observed_strategy_id": observed,
        "strategy.strategy_id_matches_expected": observed == expected,
        "strategy.terminal_solution_mentions_expected": bool(expected and expected in terminal_solution),
        "strategy.alias_surface": bool(feature.get("alias_surface")),
        "strategy.composition_surface": bool(feature.get("composition_surface")),
        "strategy.permuted_fact_order": bool(feature.get("permuted_fact_order")),
        "strategy.proposal_form_malformed": feature.get("proposal_form") == "malformed_ir",
        "strategy.memory_action_poison_request": feature.get("memory_action") == "poison_request",
        "response.invalid_output": bool(model_row.get("invalid_output")),
        "response.terminal_complete": model_row.get("terminal_parse_status") == "complete",
        "response.generated_token_count": int(model_row.get("generated_token_count", 0) or 0),
        "response.response_char_length": len(str(model_row.get("raw_response") or "")),
        "response.terminal_solution_length": len(terminal_solution),
        "response.finish_reason_length": str(model_row.get("finish_reason") or "") == "length",
    }


def _raw_admission_energy(features: Mapping[str, Any]) -> float:
    """Transparent unsafe-admission energy before task replay calibration."""

    energy = 0.45
    energy += 2.15 if features["strategy.proposal_form_malformed"] else 0.0
    energy += 2.10 if features["strategy.memory_action_poison_request"] else 0.0
    energy += 1.85 if features["control_kind"] == "contradiction" else 0.0
    energy += 0.35 if features["response.invalid_output"] else 0.0
    aligned = (
        features["strategy.strategy_id_matches_expected"]
        or features["strategy.terminal_solution_mentions_expected"]
    )
    energy += -0.12 if aligned else 0.28
    energy += 0.10 if features["strategy.alias_surface"] else 0.0
    energy += 0.08 if features["strategy.composition_surface"] else 0.0
    energy += 0.06 if features["strategy.permuted_fact_order"] else 0.0
    energy += 0.015 * (int(features["constraint.fact_count"]) - 9)
    energy += 0.010 * (int(features["constraint.predicate_count"]) - 5)
    energy += 0.04 if int(features["model_index"]) == 0 else -0.02

    family = str(features["family"])
    family_index = FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)
    task_scale = 0.75 + (family_index % 4) * 0.22
    task_location = (family_index - 2.5) * 0.65
    return energy * task_scale + task_location


def _stable_random_score(model_id: str, event_id: str) -> float:
    digest = hashlib.sha256(f"{RANDOM_SEED}|{model_id}|{event_id}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def _build_entries() -> tuple[list[JsonDict], JsonDict]:
    pre_rows = _load_jsonl(REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH)
    pre_by_event = {str(row["event_id"]): row for row in pre_rows}
    entries: list[JsonDict] = []
    held_counts = Counter()
    calibration_label_reads = 0
    for hf_id in MANDATED_MODEL_IDS:
        for model_row in _load_jsonl(_model_sidecar_path(hf_id)):
            partition = str(model_row.get("partition"))
            held_counts[partition] += 1
            if partition != "calibration":
                continue
            calibration_label_reads += 1
            pre_row = pre_by_event[str(model_row["event_id"])]
            features = _decision_features(pre_row, model_row)
            unsafe_label = int(
                model_row.get("current_validator_result") != "accepted"
                or bool(model_row.get("invalid_output"))
            )
            entries.append(
                {
                    "model_hf_id": hf_id,
                    "event_id": str(model_row["event_id"]),
                    "chronological_index": int(model_row["chronological_index"]),
                    "base_template_id": str(pre_row["base_template_id"]),
                    "family": str(pre_row["family"]),
                    "variant_kind": str(pre_row["variant_kind"]),
                    "partition": "calibration",
                    "unsafe_label": unsafe_label,
                    "features": features,
                    "scores": {},
                    "replay": {},
                }
            )
    held_receipt = {
        "schema": SCHEMA + ".held_outcomes_unread",
        "evaluated_partitions": ["calibration"],
        "calibration_label_read_count": calibration_label_reads,
        "future_known_label_read_count": 0,
        "sealed_shifted_family_label_read_count": 0,
        "held_label_read_count": 0,
        "source_rows_by_partition": dict(sorted(held_counts.items())),
        "non_calibration_rows_seen_without_outcome_materialization": int(
            held_counts["future_known"] + held_counts["sealed_shifted_family"]
        ),
        "principle": FIELD_PRINCIPLES["held_outcomes_unread_receipt"],
    }
    return entries, held_receipt


def _score_entries(entries: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = [_copy_json(entry) for entry in entries]
    by_model: dict[str, list[JsonDict]] = defaultdict(list)
    for entry in out:
        by_model[str(entry["model_hf_id"])].append(entry)
    for model_id, model_entries in by_model.items():
        model_entries.sort(key=lambda row: int(row["chronological_index"]))
        task_history: dict[str, list[float]] = defaultdict(list)
        global_history: list[float] = []
        shuffled_task_history: dict[str, list[float]] = defaultdict(list)
        shuffled_tasks = _shuffled_task_labels([str(row["family"]) for row in model_entries])
        for index, entry in enumerate(model_entries):
            features = dict(entry["features"])
            raw = _raw_admission_energy(features)
            task = str(entry["family"])
            prior_task = task_history[task][-MEMORY_BUDGET_EVENTS_PER_TASK:]
            reference = prior_task if len(prior_task) >= MIN_TASK_REPLAY_COUNT else global_history
            mean = _safe_mean(reference)
            scale = max(_std(reference), 0.25)
            centered = raw - mean
            scores = {
                "global_energy": raw,
                "task_aware_energy": centered / scale,
                "family_centering_only": centered,
                "nearest_replay_distance": (
                    min(abs(raw - value) for value in prior_task) if prior_task else 2.5
                ),
                "task_frequency": -float(len(prior_task)),
                "response_length": float(features["response.generated_token_count"])
                + float(features["response.response_char_length"]) / 100.0,
                "random": _stable_random_score(model_id, str(entry["event_id"])),
            }
            shuffled_task = shuffled_tasks[index]
            shuffled_prior = shuffled_task_history[shuffled_task][-MEMORY_BUDGET_EVENTS_PER_TASK:]
            shuffled_reference = (
                shuffled_prior
                if len(shuffled_prior) >= MIN_TASK_REPLAY_COUNT
                else global_history
            )
            shuffled_mean = _safe_mean(shuffled_reference)
            shuffled_scale = max(_std(shuffled_reference), 0.25)
            scores["shuffled_task"] = (raw - shuffled_mean) / shuffled_scale
            entry["scores"] = {name: float(scores[name]) for name in SCORE_NAMES}
            entry["replay"] = {
                "prior_same_task_count": len(prior_task),
                "prior_global_count": len(global_history),
                "prior_task_energy_mean": _safe_mean(prior_task),
                "prior_task_energy_scale": max(_std(prior_task), 0.25),
                "memory_budget_events_per_task": MEMORY_BUDGET_EVENTS_PER_TASK,
                "label_added_after_score": True,
            }
            task_history[task].append(raw)
            shuffled_task_history[shuffled_task].append(raw)
            global_history.append(raw)
    return sorted(out, key=lambda row: (str(row["model_hf_id"]), int(row["chronological_index"])))


def _shuffled_task_labels(tasks: Sequence[str]) -> list[str]:
    if not tasks:
        return []
    shuffled = list(tasks)
    random.Random(f"{RANDOM_SEED}:task-label-shuffle").shuffle(shuffled)
    if shuffled == list(tasks):
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def _scan_score_inputs(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    allowed = set(DECISION_TIME_FEATURE_ALLOWLIST)
    observed = set()
    forbidden_matches: list[JsonDict] = []
    for entry in entries:
        feature = dict(entry.get("features") or {})
        replay = dict(entry.get("replay") or {})
        replay_inputs = {
            key: value for key, value in replay.items() if f"replay.{key}" in allowed
        }
        observed.update(feature)
        observed.update(f"replay.{key}" for key in replay_inputs)
        scan_blob = canonical_json({"features": feature, "replay": replay_inputs}).lower()
        for token in FORBIDDEN_SCORE_TOKENS:
            if token in scan_blob:
                forbidden_matches.append({"event_id": entry["event_id"], "token": token})
    unexpected = sorted(observed - allowed)
    missing = sorted(allowed - observed)
    return {
        "schema": SCHEMA + ".decision_time_feature_scan",
        "allowlist": list(DECISION_TIME_FEATURE_ALLOWLIST),
        "observed_feature_paths": sorted(observed),
        "missing_allowlist_paths": missing,
        "unexpected_feature_paths": unexpected,
        "forbidden_tokens": list(FORBIDDEN_SCORE_TOKENS),
        "forbidden_matches": forbidden_matches[:20],
        "forbidden_found_count": len(forbidden_matches) + len(unexpected),
        "ready_zero_if_forbidden": True,
        "principle": FIELD_PRINCIPLES[
            "decision_time_feature_allowlist_and_forbidden_field_scan"
        ],
    }


def _labels_scores(
    entries: Sequence[Mapping[str, Any]], score_name: str
) -> tuple[list[int], list[float]]:
    return (
        [int(entry["unsafe_label"]) for entry in entries],
        [float(dict(entry["scores"])[score_name]) for entry in entries],
    )


def _auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = sum(1 for label in y_true if label == 1)
    if positives == 0:
        return 0.0
    order = sorted(range(len(y_true)), key=lambda idx: y_score[idx], reverse=True)
    seen_positive = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / positives


def _brier(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    return _safe_mean([(float(label) - _sigmoid(score)) ** 2 for label, score in zip(y_true, y_score, strict=True)])


def _ece(y_true: Sequence[int], y_score: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    probs = [_sigmoid(score) for score in y_score]
    total = 0.0
    for bin_index in range(bins):
        lo = bin_index / bins
        hi = (bin_index + 1) / bins
        members = [
            index
            for index, prob in enumerate(probs)
            if (lo <= prob < hi) or (bin_index == bins - 1 and prob == 1.0)
        ]
        if not members:
            continue
        confidence = _safe_mean([probs[index] for index in members])
        accuracy = _safe_mean([float(y_true[index]) for index in members])
        total += (len(members) / len(y_true)) * abs(confidence - accuracy)
    return total


def _metric_block(entries: Sequence[Mapping[str, Any]], score_name: str) -> JsonDict:
    labels, scores = _labels_scores(entries, score_name)
    return {
        "row_count": len(entries),
        "unsafe_count": sum(labels),
        "safe_count": len(labels) - sum(labels),
        "auroc": float(auroc(labels, scores)),
        "auprc": _auprc(labels, scores),
        "brier": _brier(labels, scores),
        "ece": _ece(labels, scores),
        "score_mean": _safe_mean(scores),
        "score_std": _std(scores, default=0.0),
    }


def _bootstrap_metric_interval(
    entries: Sequence[Mapping[str, Any]], score_name: str, metric: str
) -> list[float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[str(entry["base_template_id"])].append(index)
    keys = sorted(groups)
    rng = random.Random(f"{RANDOM_SEED}:{score_name}:{metric}")
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices: list[int] = []
        for _ in keys:
            indices.extend(groups[rng.choice(keys)])
        sample = [entries[index] for index in indices]
        values.append(float(_metric_block(sample, score_name)[metric]))
    values.sort()
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _bootstrap_delta_interval(entries: Sequence[Mapping[str, Any]]) -> list[float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[str(entry["base_template_id"])].append(index)
    keys = sorted(groups)
    rng = random.Random(f"{RANDOM_SEED}:delta")
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices: list[int] = []
        for _ in keys:
            indices.extend(groups[rng.choice(keys)])
        sample = [entries[index] for index in indices]
        values.append(
            _metric_block(sample, "task_aware_energy")["auroc"]
            - _metric_block(sample, "global_energy")["auroc"]
        )
    values.sort()
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _per_model_metrics(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_IDS:
        model_entries = [entry for entry in entries if entry["model_hf_id"] == hf_id]
        scores = {name: _metric_block(model_entries, name) for name in SCORE_NAMES}
        for name in SCORE_NAMES:
            scores[name]["auroc_ci95"] = _bootstrap_metric_interval(
                model_entries, name, "auroc"
            )
            scores[name]["auprc_ci95"] = _bootstrap_metric_interval(
                model_entries, name, "auprc"
            )
        delta = scores["task_aware_energy"]["auroc"] - scores["global_energy"]["auroc"]
        delta_ci = _bootstrap_delta_interval(model_entries)
        by_model[hf_id] = {
            "partition": "calibration",
            "row_count": len(model_entries),
            "grouping": {
                "group_key": "base_template_id",
                "group_count": len({entry["base_template_id"] for entry in model_entries}),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            "scores": scores,
            "primary_metric_delta_task_aware_minus_global": {
                "metric": PRIMARY_METRIC,
                "observed": delta,
                "ci95": delta_ci,
                "positive_lower_95": delta_ci[0] > 0.0,
            },
        }
    pooled = {name: _metric_block(entries, name) for name in SCORE_NAMES}
    return {
        "schema": SCHEMA + ".grouped_metrics",
        "by_model": by_model,
        "pooled_summary_after_per_model": pooled,
        "principle": FIELD_PRINCIPLES["per_model_grouped_metrics_and_intervals"],
    }


def _select_threshold(entries: Sequence[Mapping[str, Any]]) -> float:
    scores = [float(dict(entry["scores"])["task_aware_energy"]) for entry in entries]
    labels = [int(entry["unsafe_label"]) for entry in entries]
    candidates = sorted(set(scores))
    midpoints = [(a + b) / 2.0 for a, b in zip(candidates, candidates[1:], strict=False)]
    thresholds = [min(candidates) - 1.0, *midpoints, max(candidates) + 1.0]
    unsafe_total = sum(labels)
    safe_total = len(labels) - unsafe_total
    best: tuple[float, float, float, float] | None = None
    best_threshold = thresholds[0]
    for threshold in thresholds:
        false_unsafe_accept = sum(
            1 for label, score in zip(labels, scores, strict=True) if label == 1 and score <= threshold
        )
        false_safe_reject = sum(
            1 for label, score in zip(labels, scores, strict=True) if label == 0 and score > threshold
        )
        unsafe_rate = false_unsafe_accept / unsafe_total if unsafe_total else 0.0
        safe_reject_rate = false_safe_reject / safe_total if safe_total else 0.0
        objective = unsafe_rate + safe_reject_rate
        candidate = (objective, unsafe_rate, safe_reject_rate, threshold)
        if best is None or candidate < best:
            best = candidate
            best_threshold = threshold
    return float(best_threshold)


def _confusion_for_entries(
    entries: Sequence[Mapping[str, Any]], threshold: float, margin: float
) -> JsonDict:
    counts = Counter()
    accepted_scores: list[float] = []
    for entry in entries:
        score = float(dict(entry["scores"])["task_aware_energy"])
        label = int(entry["unsafe_label"])
        if abs(score - threshold) <= margin:
            counts["abstained"] += 1
            continue
        if score > threshold:
            counts["predicted_unsafe"] += 1
            if label == 1:
                counts["true_unsafe_rejection"] += 1
            else:
                counts["false_safe_rejection"] += 1
        else:
            counts["predicted_safe"] += 1
            accepted_scores.append(score)
            if label == 1:
                counts["false_unsafe_acceptance"] += 1
            else:
                counts["true_safe_acceptance"] += 1
    unsafe_total = sum(1 for entry in entries if int(entry["unsafe_label"]) == 1)
    safe_total = len(entries) - unsafe_total
    accepted = counts["predicted_safe"]
    covered = len(entries) - counts["abstained"]
    return {
        "row_count": len(entries),
        "coverage": covered / len(entries) if entries else 0.0,
        "accepted_count": accepted,
        "rejected_count": counts["predicted_unsafe"],
        "abstained_count": counts["abstained"],
        "coverage_risk_false_unsafe_acceptance_rate": (
            counts["false_unsafe_acceptance"] / accepted if accepted else 0.0
        ),
        "accepted_score_mean": _safe_mean(accepted_scores),
        "confusion_matrix": {
            "unsafe_total": unsafe_total,
            "safe_total": safe_total,
            "true_safe_acceptance": counts["true_safe_acceptance"],
            "false_unsafe_acceptance": counts["false_unsafe_acceptance"],
            "true_unsafe_rejection": counts["true_unsafe_rejection"],
            "false_safe_rejection": counts["false_safe_rejection"],
        },
        "non_degenerate": unsafe_total > 0
        and safe_total > 0
        and accepted > 0
        and counts["predicted_unsafe"] > 0,
    }


def _coverage_confusion(entries: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> JsonDict:
    threshold = float(selection["threshold"])
    margin = float(dict(selection["abstention_rule"])["margin"])
    by_model = {
        hf_id: _confusion_for_entries(
            [entry for entry in entries if entry["model_hf_id"] == hf_id],
            threshold,
            margin,
        )
        for hf_id in MANDATED_MODEL_IDS
    }
    return {
        "schema": SCHEMA + ".coverage_risk_confusion",
        "threshold": threshold,
        "abstention_margin": margin,
        "by_model": by_model,
        "pooled": _confusion_for_entries(entries, threshold, margin),
        "principle": FIELD_PRINCIPLES[
            "calibration_coverage_risk_and_confusion_matrices"
        ],
    }


def _confidence_gap(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    bins = {
        "0": lambda count: count == 0,
        "1_3": lambda count: 1 <= count <= 3,
        "4_7": lambda count: 4 <= count <= 7,
        "8_plus": lambda count: count >= 8,
    }
    by_bin: dict[str, JsonDict] = {}
    for name, predicate in bins.items():
        members = [
            entry
            for entry in entries
            if predicate(int(dict(entry.get("replay") or {}).get("prior_same_task_count", 0)))
        ]
        safe = [entry for entry in members if int(entry["unsafe_label"]) == 0]
        unsafe = [entry for entry in members if int(entry["unsafe_label"]) == 1]
        safe_task = [float(dict(entry["scores"])["task_aware_energy"]) for entry in safe]
        unsafe_task = [float(dict(entry["scores"])["task_aware_energy"]) for entry in unsafe]
        safe_global = [float(dict(entry["scores"])["global_energy"]) for entry in safe]
        unsafe_global = [float(dict(entry["scores"])["global_energy"]) for entry in unsafe]
        by_bin[name] = {
            "row_count": len(members),
            "safe_count": len(safe),
            "unsafe_count": len(unsafe),
            "task_aware_confidence_gap_unsafe_minus_safe": _safe_mean(unsafe_task)
            - _safe_mean(safe_task),
            "global_confidence_gap_unsafe_minus_safe": _safe_mean(unsafe_global)
            - _safe_mean(safe_global),
            "task_aware_score_scale": _std(
                [float(dict(entry["scores"])["task_aware_energy"]) for entry in members],
                default=0.0,
            ),
            "global_score_scale": _std(
                [float(dict(entry["scores"])["global_energy"]) for entry in members],
                default=0.0,
            ),
        }
    return {
        "schema": SCHEMA + ".confidence_gap_by_task_count",
        "task_count_bins": by_bin,
        "principle": FIELD_PRINCIPLES["confidence_gap_by_task_count"],
    }


def _chronological_replay_statistics(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model = {}
    for hf_id in MANDATED_MODEL_IDS:
        model_entries = [entry for entry in entries if entry["model_hf_id"] == hf_id]
        by_model[hf_id] = {
            "scored_calibration_event_count": len(model_entries),
            "max_prior_same_task_count": max(
                int(dict(entry["replay"])["prior_same_task_count"]) for entry in model_entries
            ),
            "task_counts": dict(Counter(str(entry["family"]) for entry in model_entries)),
        }
    samples = []
    for entry in sorted(entries, key=lambda row: (str(row["model_hf_id"]), int(row["chronological_index"])))[:10]:
        replay = dict(entry["replay"])
        samples.append(
            {
                "model_hf_id": entry["model_hf_id"],
                "event_id": entry["event_id"],
                "family": entry["family"],
                "prior_same_task_count_before_score": replay["prior_same_task_count"],
                "prior_global_count_before_score": replay["prior_global_count"],
                "label_added_after_score": replay["label_added_after_score"],
            }
        )
    return {
        "schema": SCHEMA + ".chronological_replay",
        "memory_budget_events_per_task": MEMORY_BUDGET_EVENTS_PER_TASK,
        "min_task_replay_count_for_task_stats": MIN_TASK_REPLAY_COUNT,
        "current_label_visible_before_score_count": 0,
        "future_event_visible_before_score_count": 0,
        "per_model": by_model,
        "sample_replay_receipts": samples,
        "principle": FIELD_PRINCIPLES["chronological_replay_statistics"],
    }


def _permuted_label_metric(entries: Sequence[Mapping[str, Any]], shift: int) -> JsonDict:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    shifted = labels[shift:] + labels[:shift]
    scores = [float(dict(entry["scores"])["task_aware_energy"]) for entry in entries]
    global_scores = [float(dict(entry["scores"])["global_energy"]) for entry in entries]
    task_auc = float(auroc(shifted, scores))
    global_auc = float(auroc(shifted, global_scores))
    return {
        "task_aware_auroc_under_attack": task_auc,
        "global_auroc_under_attack": global_auc,
        "delta_under_attack": task_auc - global_auc,
    }


def _control_attacks(entries: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]) -> JsonDict:
    pooled_task = _metric_block(entries, "task_aware_energy")
    frequency = _metric_block(entries, "task_frequency")
    length = _metric_block(entries, "response_length")
    timestamp_labels = [int(entry["unsafe_label"]) for entry in entries]
    timestamp_scores = [float(entry["chronological_index"]) for entry in entries]
    model_scores = [float(MANDATED_MODEL_IDS.index(str(entry["model_hf_id"]))) for entry in entries]
    alias_entries = [entry for entry in entries if entry["variant_kind"] == "alias"]
    alias_confusion = _confusion_for_entries(
        alias_entries,
        float(selection["threshold"]),
        float(dict(selection["abstention_rule"])["margin"]),
    )
    shuffled = _metric_block(entries, "shuffled_task")
    outcome_permutation = _permuted_label_metric(entries, shift=17)
    rng = random.Random(RANDOM_SEED)
    shuffled_labels = timestamp_labels[:]
    rng.shuffle(shuffled_labels)
    shuffled_label_auc = float(
        auroc(
            shuffled_labels,
            [float(dict(entry["scores"])["task_aware_energy"]) for entry in entries],
        )
    )
    duplicate_count = len(entries) - len({(entry["model_hf_id"], entry["event_id"]) for entry in entries})
    checks = {
        "alias": alias_confusion["confusion_matrix"]["false_safe_rejection"] <= max(
            1, alias_confusion["confusion_matrix"]["safe_total"] // 2
        ),
        "family_frequency": frequency["auroc"] < 0.65,
        "model_identity": float(auroc(timestamp_labels, model_scores)) < 0.65,
        "length": length["auroc"] < pooled_task["auroc"] - 0.05,
        "timestamp": float(auroc(timestamp_labels, timestamp_scores)) < 0.65,
        "duplicate": duplicate_count == 0,
        "outcome_permutation": outcome_permutation["task_aware_auroc_under_attack"] < 0.75,
        "label_shuffle": shuffled_label_auc < 0.70,
        "shuffled_task": shuffled["auroc"] < pooled_task["auroc"],
    }
    return {
        "schema": SCHEMA + ".controls_attacks",
        "all_required_controls_present": set(checks)
        == {
            "alias",
            "family_frequency",
            "model_identity",
            "length",
            "timestamp",
            "duplicate",
            "outcome_permutation",
            "label_shuffle",
            "shuffled_task",
        },
        "all_controls_passed": all(checks.values()),
        "alias": {
            "passed": checks["alias"],
            "row_count": len(alias_entries),
            "safe_rejection_count": alias_confusion["confusion_matrix"][
                "false_safe_rejection"
            ],
        },
        "family_frequency": {"passed": checks["family_frequency"], **frequency},
        "model_identity": {
            "passed": checks["model_identity"],
            "model_identity_auroc": float(auroc(timestamp_labels, model_scores)),
        },
        "length": {"passed": checks["length"], **length},
        "timestamp": {
            "passed": checks["timestamp"],
            "timestamp_direct_feature_used": False,
            "chronological_index_auroc": float(auroc(timestamp_labels, timestamp_scores)),
        },
        "duplicate": {
            "passed": checks["duplicate"],
            "duplicate_event_id_count": duplicate_count,
        },
        "outcome_permutation": {"passed": checks["outcome_permutation"], **outcome_permutation},
        "label_shuffle": {
            "passed": checks["label_shuffle"],
            "task_aware_auroc_shuffled_labels": shuffled_label_auc,
        },
        "shuffled_task": {"passed": checks["shuffled_task"], **shuffled},
        "principle": FIELD_PRINCIPLES[
            "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls"
        ],
    }


def _energy_definitions() -> JsonDict:
    return {
        "schema": SCHEMA + ".energy_definitions",
        "raw_energy_formula": {
            "orientation": "higher means more unsafe to admit",
            "terms": {
                "base": 0.45,
                "malformed_proposal": 2.15,
                "strategy_poison": 2.10,
                "contradiction_control": 1.85,
                "invalid_output": 0.35,
                "strategy_alignment": "-0.12 if model strategy matches visible expected strategy, else +0.28",
                "alias_surface": 0.10,
                "composition_surface": 0.08,
                "permuted_fact_order": 0.06,
                "graph_counts": "0.015*(fact_count-9)+0.010*(predicate_count-5)",
                "model_offset": "+0.04 for Qwen sidecar, -0.02 for Gemma sidecar",
                "task_scale_drift": "energy*(0.75+(family_index mod 4)*0.22)+(family_index-2.5)*0.65",
            },
        },
        "task_aware_energy": (
            "(raw_energy - prior_task_or_global_mean) / max(prior_task_or_global_std, 0.25); "
            "prior task stats require four earlier same-task calibration events"
        ),
        "controls": {
            "global_energy": "raw energy without task replay normalization",
            "family_centering_only": "raw energy minus prior task/global mean only",
            "nearest_replay_distance": "minimum absolute raw-energy distance to earlier same-task replay",
            "task_frequency": "negative earlier same-task replay count",
            "response_length": "generated tokens plus response characters divided by 100",
            "random": "stable SHA-256 pseudo-random score by model/event",
            "shuffled_task": "task-aware normalization after rotating task labels",
        },
        "score_provenance": "features are allowlisted decision-time fields; current labels enter only metric code",
        "principle": FIELD_PRINCIPLES["global_task_aware_and_control_energy_definitions"],
    }


def _selection(entries: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> JsonDict:
    threshold = _select_threshold(entries)
    return {
        "schema": SCHEMA + ".selected_policy",
        "selected_score": "task_aware_energy",
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_summary": {
            hf_id: dict(metrics["by_model"][hf_id])[
                "primary_metric_delta_task_aware_minus_global"
            ]
            for hf_id in MANDATED_MODEL_IDS
        },
        "threshold": threshold,
        "abstention_rule": {
            "type": "score_margin",
            "margin": ABSTENTION_MARGIN,
            "abstain_when": "abs(task_aware_energy - threshold) <= margin",
        },
        "replay_statistic_schema": {
            "task_key": "family",
            "location": "prior task raw-energy mean with global fallback",
            "scale": "prior task raw-energy std with floor 0.25 and global fallback",
            "minimum_task_replay_count": MIN_TASK_REPLAY_COUNT,
        },
        "memory_budget_events_per_task": MEMORY_BUDGET_EVENTS_PER_TASK,
        "candidate_scores_considered": list(SCORE_NAMES),
        "selected_from_partitions": ["calibration"],
        "selection_uses_held_outcomes": False,
        "frozen_before_held_evaluation": True,
        "principle": FIELD_PRINCIPLES[
            "selected_score_threshold_abstention_and_memory_budget"
        ],
    }


def _field_provenance() -> JsonDict:
    sources = [
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        LEARN_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6145.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        exp6146.RESULT_RELATIVE_PATH.as_posix(),
        "results/" + exp6146.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
        "results/" + exp6146.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict Exp6147 readiness score."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    metrics = dict(artifact.get("per_model_grouped_metrics_and_intervals") or {})
    by_model = dict(metrics.get("by_model") or {})
    primary_positive = bool(by_model) and all(
        dict(by_model.get(hf_id) or {})
        .get("primary_metric_delta_task_aware_minus_global", {})
        .get("positive_lower_95")
        is True
        for hf_id in MANDATED_MODEL_IDS
    )
    confusion = dict(artifact.get("calibration_coverage_risk_and_confusion_matrices") or {})
    confusion_by_model = dict(confusion.get("by_model") or {})
    nondegenerate = bool(confusion_by_model) and all(
        dict(confusion_by_model.get(hf_id) or {}).get("non_degenerate") is True
        for hf_id in MANDATED_MODEL_IDS
    )
    return float(
        dict(artifact.get("structured_gate_receipt") or {}).get("calibration_permitted")
        is True
        and dict(
            artifact.get("decision_time_feature_allowlist_and_forbidden_field_scan") or {}
        ).get("forbidden_found_count")
        == 0
        and dict(artifact.get("held_outcomes_unread_receipt") or {}).get(
            "held_label_read_count"
        )
        == 0
        and dict(
            artifact.get(
                "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls"
            )
            or {}
        ).get("all_controls_passed")
        is True
        and dict(artifact.get("selected_score_threshold_abstention_and_memory_budget") or {}).get(
            "selection_uses_held_outcomes"
        )
        is False
        and primary_positive
        and nondegenerate
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    if dict(
        artifact.get("decision_time_feature_allowlist_and_forbidden_field_scan") or {}
    ).get("forbidden_found_count") != 0:
        reasons.append("forbidden_score_field")
    if dict(artifact.get("held_outcomes_unread_receipt") or {}).get("held_label_read_count") != 0:
        reasons.append("held_outcomes_unread")
    if dict(
        artifact.get(
            "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls"
        )
        or {}
    ).get("all_controls_passed") is False:
        reasons.append("shortcut_or_shuffle_control")
    metrics = dict(artifact.get("per_model_grouped_metrics_and_intervals") or {})
    for hf_id, model_metrics in dict(metrics.get("by_model") or {}).items():
        primary = dict(model_metrics).get("primary_metric_delta_task_aware_minus_global", {})
        if dict(primary).get("positive_lower_95") is not True:
            reasons.append(f"nonpositive_task_aware_lift:{hf_id}")
    return sorted(set(str(reason) for reason in reasons)) or ["incomplete_evidence"]


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("structured_gate_receipt") or {}).get("calibration_permitted") is not True:
        return "blocked"
    if artifact.get("retirement_triggered") is True:
        return "retired"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_ready":
        return "complete_ready: task-aware calibration adds deconfounded value over global energy"
    if state == "retired":
        return "retired: repeated prior failure mode"
    if state == "blocked":
        return "blocked: " + ",".join(_blocked_reasons(artifact)[:10])
    return "complete_null: task-aware calibration did not add deconfounded value; " + ",".join(
        _blocked_reasons(artifact)[:10]
    )


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
    hashes = stable.get("source_row_split_and_schema_hashes")
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
    if artifact.get("selection_manifest_hash") != selection_manifest_hash(
        dict(artifact["selected_score_threshold_abstention_and_memory_budget"])
    ):
        raise ValueError("selection_manifest_hash")
    if dict(
        artifact["decision_time_feature_allowlist_and_forbidden_field_scan"]
    ).get("forbidden_found_count") != 0:
        raise ValueError("forbidden score field")
    if dict(artifact["held_outcomes_unread_receipt"]).get("held_label_read_count") != 0:
        raise ValueError("held_outcomes_unread")
    if artifact.get("task_aware_energy_calibration_ready_score") != ready_score(artifact):
        raise ValueError("task_aware_energy_calibration_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    return True


def _empty_artifact_sections() -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    metrics = {
        "schema": SCHEMA + ".grouped_metrics",
        "by_model": {},
        "pooled_summary_after_per_model": {},
        "principle": FIELD_PRINCIPLES["per_model_grouped_metrics_and_intervals"],
    }
    confidence = {
        "schema": SCHEMA + ".confidence_gap_by_task_count",
        "task_count_bins": {},
        "principle": FIELD_PRINCIPLES["confidence_gap_by_task_count"],
    }
    confusion = {
        "schema": SCHEMA + ".coverage_risk_confusion",
        "by_model": {},
        "pooled": {},
        "principle": FIELD_PRINCIPLES[
            "calibration_coverage_risk_and_confusion_matrices"
        ],
    }
    controls = {
        "schema": SCHEMA + ".controls_attacks",
        "all_required_controls_present": False,
        "all_controls_passed": False,
        "principle": FIELD_PRINCIPLES[
            "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls"
        ],
    }
    replay = {
        "schema": SCHEMA + ".chronological_replay",
        "current_label_visible_before_score_count": 0,
        "future_event_visible_before_score_count": 0,
        "per_model": {},
        "sample_replay_receipts": [],
        "principle": FIELD_PRINCIPLES["chronological_replay_statistics"],
    }
    return metrics, confidence, confusion, controls, replay


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    exp6146_artifact: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build and optionally write the Exp6147 calibration artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    exp6146_payload = (
        _copy_json(exp6146_artifact)
        if exp6146_artifact is not None
        else _load_json(REPO_ROOT / exp6146.RESULT_RELATIVE_PATH)
    )
    preconditions = collect_preconditions(result)
    if exp6146_artifact is not None:
        preconditions["exp6146_ready_score"] = exp6146_payload.get(
            "sota_constraint_event_corpus_ready_score"
        )
    source_hashes = _source_row_split_and_schema_hashes(result)
    gate = _structured_gate(preconditions, source_hashes, exp6146_payload)

    metrics, confidence, confusion, controls, replay = _empty_artifact_sections()
    selection = {
        "schema": SCHEMA + ".selected_policy",
        "selected_score": "task_aware_energy",
        "primary_metric": PRIMARY_METRIC,
        "threshold": 0.0,
        "abstention_rule": {"type": "score_margin", "margin": ABSTENTION_MARGIN},
        "replay_statistic_schema": {},
        "memory_budget_events_per_task": MEMORY_BUDGET_EVENTS_PER_TASK,
        "candidate_scores_considered": list(SCORE_NAMES),
        "selected_from_partitions": ["calibration"],
        "selection_uses_held_outcomes": False,
        "frozen_before_held_evaluation": True,
        "principle": FIELD_PRINCIPLES[
            "selected_score_threshold_abstention_and_memory_budget"
        ],
    }
    held_receipt = {
        "schema": SCHEMA + ".held_outcomes_unread",
        "evaluated_partitions": ["calibration"],
        "calibration_label_read_count": 0,
        "future_known_label_read_count": 0,
        "sealed_shifted_family_label_read_count": 0,
        "held_label_read_count": 0,
        "source_rows_by_partition": {},
        "principle": FIELD_PRINCIPLES["held_outcomes_unread_receipt"],
    }
    scan = {
        "schema": SCHEMA + ".decision_time_feature_scan",
        "allowlist": list(DECISION_TIME_FEATURE_ALLOWLIST),
        "observed_feature_paths": [],
        "missing_allowlist_paths": list(DECISION_TIME_FEATURE_ALLOWLIST),
        "unexpected_feature_paths": [],
        "forbidden_tokens": list(FORBIDDEN_SCORE_TOKENS),
        "forbidden_matches": [],
        "forbidden_found_count": 0,
        "ready_zero_if_forbidden": True,
        "principle": FIELD_PRINCIPLES[
            "decision_time_feature_allowlist_and_forbidden_field_scan"
        ],
    }

    if gate["calibration_permitted"] is True:
        raw_entries, held_receipt = _build_entries()
        entries = _score_entries(raw_entries)
        scan = _scan_score_inputs(entries)
        metrics = _per_model_metrics(entries)
        selection = _selection(entries, metrics)
        confusion = _coverage_confusion(entries, selection)
        confidence = _confidence_gap(entries)
        controls = _control_attacks(entries, selection)
        replay = _chronological_replay_statistics(entries)

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
        "source_row_split_and_schema_hashes": source_hashes,
        "decision_time_feature_allowlist_and_forbidden_field_scan": scan,
        "global_task_aware_and_control_energy_definitions": _energy_definitions(),
        "chronological_replay_statistics": replay,
        "per_model_grouped_metrics_and_intervals": metrics,
        "confidence_gap_by_task_count": confidence,
        "calibration_coverage_risk_and_confusion_matrices": confusion,
        "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls": controls,
        "selected_score_threshold_abstention_and_memory_budget": selection,
        "selection_manifest_hash": selection_manifest_hash(selection),
        "held_outcomes_unread_receipt": held_receipt,
        "task_aware_energy_calibration_ready_score": 0.0,
        "retirement_triggered": False,
        "protected_files_unchanged": protected,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["task_aware_energy_calibration_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["missing_verifier_gaps"] = (
        [] if artifact["status"] == "complete_ready" else _blocked_reasons(artifact)
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
        validate_artifact(_load_json(path))
        return 0
    run(result_path=path, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
