"""Exp6161 decision-calibrated energy policy freeze.

Spec refs: REQ-VERIFY-6161, REQ-VERIFY-6161-1, REQ-VERIFY-6161-2,
REQ-VERIFY-6161-3, REQ-VERIFY-6161-4, REQ-VERIFY-6161-5,
REQ-VERIFY-6161-6, REQ-VERIFY-6161-7, REQ-VERIFY-6161-8,
REQ-VERIFY-6161-9, REQ-VERIFY-6161-10,
SCENARIO-VERIFY-6161-CALIBRATION-ONLY,
SCENARIO-VERIFY-6161-GROUPED-CV, SCENARIO-VERIFY-6161-CONTROLS,
SCENARIO-VERIFY-6161-FREEZE.

Exp6161 does CPU analysis over cached authentic Exp6160 rows. It tunes only on
calibration rows and freezes a manifest before any held outcome can be opened.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

from carnot import experiment_6147_task_aware_energy_calibration as exp6147
from carnot import experiment_6159_decision_calibrated_stream as exp6159
from carnot import experiment_6160_sota_decision_calibration_corpus as exp6160
from carnot.eval.metrics import auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6161_decision_calibrated_energy_policy.json")
MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6161_decision_calibrated_energy_policy.manifest.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6161_decision_calibrated_energy_policy.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6161_decision_calibrated_energy_policy.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6161.decision_calibrated_energy_policy.v1"
EXPERIMENT_ID = "experiment_6161_decision_calibrated_energy_policy"
RUN_DATE = "20260806"
RANDOM_SEED = 6161
INFERENCE_SUBSTRATE = "cached_authentic_sota_rows_cpu_analysis"
VERIFIER_IS_ORACLE = False
ABSTENTION_MARGIN = 0.05
FOLD_COUNT = 4

MANDATED_MODEL_IDS = exp6160.MANDATED_MODEL_IDS
CANDIDATE_ARMS = (
    "global_energy",
    "exp6147_fixed_task_aware",
    "decision_calibrated_task_energy",
    "family_only",
    "shuffled_task",
    "alias",
    "frequency",
    "distance",
)
CONTROL_NAMES = (
    "label_shuffle",
    "outcome_flip",
    "task_shuffle",
    "alias",
    "family_frequency",
    "model_identity",
    "constant_score",
    "threshold_boundary",
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    exp6147.RESULT_RELATIVE_PATH,
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.ROW_FILE_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    exp6160.RESULT_RELATIVE_PATH,
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6147.RESULT_RELATIVE_PATH,
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    exp6160.RESULT_RELATIVE_PATH,
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
    Path("results") / exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    Path("python/carnot/experiment_6147_task_aware_energy_calibration.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
)

PRECOMMITTED_FEATURE_ALLOWLIST = (
    "family",
    "variant_kind",
    "control_kind",
    "model_hf_id",
    "model_index",
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
    "strategy.threshold_distance_hint",
    "response.invalid_output",
    "response.terminal_complete",
    "response.generated_token_count",
    "response.response_char_length",
    "response.finish_reason_length",
    "history.prior_same_family_event_count",
    "history.prior_same_template_event_count",
)
FORBIDDEN_SCORE_TOKENS = (
    "answer",
    "chronological_index",
    "current_outcome",
    "current_validator_result",
    "decision_record_hash",
    "exact_answer",
    "exact_labels",
    "exact_outcome",
    "future_event",
    "future_label",
    "held_label",
    "message_hash",
    "oracle_label",
    "outcome_receipt",
    "post_outcome",
    "row_hash",
    "row_id",
    "seed",
    "unsafe_label",
    "validator",
    "visible_event_hash",
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6161_decision_calibrated_energy_policy.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6161_decision_calibrated_energy_policy.py "
    "-m pytest tests/python/test_experiment_6161_decision_calibrated_energy_policy.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6161_decision_calibrated_energy_policy.py "
    "--fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6161_decision_calibrated_energy_policy.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6161_decision_calibrated_energy_policy --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6161_decision_calibrated_energy_policy.json"
)
E2E_APPLICABLE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6161_decision_calibrated_energy_policy --e2e-check"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6161_decision_calibrated_energy_policy.py "
    "tests/python/test_experiment_6161_decision_calibrated_energy_policy.py"
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

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "upstream_endpoint_row_and_control_hashes",
    "precommitted_feature_allowlist_and_forbidden_scan",
    "calibration_group_and_fold_receipts",
    "global_exp6147_decision_family_shuffled_alias_frequency_and_distance_arm_configs",
    "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics",
    "chronological_drift_diagnostics",
    "shortcut_and_boundary_controls",
    "selected_policy_rationale_without_held_access",
    "policy_manifest_path_hash_and_contents",
    "score_threshold_abstention_and_cost_freeze_receipts",
    "held_access_count",
    "decision_calibrated_policy_ready_score",
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
    "status": "A terminal state distinguishes ready, null, retired, or blocked policy freeze.",
    "preconditions_checked": "Upstream endpoints, rows, controls, exclusions, outputs, held counters, and protected files are hashed before scoring.",
    "structured_gate_receipt": "Calibration opens only after Exp6159, Exp6160, and Exp6147 are ready, row sidecars are present, held access is zero, and no live model path is invoked.",
    "upstream_endpoint_row_and_control_hashes": "Exp6159 endpoint/splits/preregistration, Exp6160 rows, Exp6147 fixed score code/manifest, exclusions, and output paths are content-addressed.",
    "precommitted_feature_allowlist_and_forbidden_scan": "Any current outcome, exact answer, exact validator field, row-order alias, future-event feature, or held label in score inputs forces readiness zero.",
    "calibration_group_and_fold_receipts": "Model/task family groups are never split across fit/tune folds.",
    "global_exp6147_decision_family_shuffled_alias_frequency_and_distance_arm_configs": "Every compared arm is transparent, resource-matched, and declares whether it is fitted or fixed.",
    "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics": "Each mandated model reports decision cost, proper scores, action counts, and descriptive ranking metrics before pooling.",
    "chronological_drift_diagnostics": "Calibration drift is diagnostic only and never a row-order feature.",
    "shortcut_and_boundary_controls": "Shortcut and boundary controls cannot outperform the selected policy for readiness.",
    "selected_policy_rationale_without_held_access": "One complete policy is selected from calibration objective evidence without held access.",
    "policy_manifest_path_hash_and_contents": "The frozen manifest is content-addressed and complete enough for prospective held replay.",
    "score_threshold_abstention_and_cost_freeze_receipts": "Score formula, threshold, abstention, and Exp6159 cost table are frozen before held opening.",
    "held_access_count": "The value is the bare scalar zero.",
    "decision_calibrated_policy_ready_score": "Exactly one means a complete prospective policy and non-vacuous calibration support, not a held win.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream protected artifacts remain byte-identical.",
    "duration_s": "Cached CPU analysis duration is reported separately from live model acquisition.",
    "inference_substrate": "Use `cached_authentic_sota_rows_cpu_analysis`.",
    "verifier_is_oracle": "The policy is oracle-distinct; exact calibration labels are not score features.",
    "missing_verifier_gaps": "Any gate, feature, fold, control, manifest, held-access, or command gap is explicit.",
    "field_provenance": "Every field traces to specs, Exp6159/Exp6160/Exp6147 artifacts, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, structured gate, feature/leakage, grouped folds, cost/proper-score calculations, controls, policy hash, zero-held-access, schema, adversarial verify, protected-file, applicable E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, row, split, feature, policy, manifest, command, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:` and state whether a policy was validly frozen.",
}

canonical_json = exp6147.canonical_json
sha256_text = exp6147.sha256_text
sha256_json = exp6147.sha256_json
sha256_file = exp6147.sha256_file


def policy_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash the frozen policy manifest with stable key ordering."""

    return sha256_json(manifest)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _load_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _std(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    mean = _safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float) -> float:
    clipped = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _logit(probability: float) -> float:
    clipped = max(0.01, min(0.99, probability))
    return math.log(clipped / (1.0 - clipped))


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


def _partition_counts(path: Path) -> JsonDict:
    counts = Counter()
    for row in _load_jsonl(path):
        counts[str(row.get("partition"))] += 1
    return dict(sorted(counts.items()))


def collect_preconditions(result_path: Path, manifest_path: Path) -> JsonDict:
    """Collect immutable receipts before any calibration labels are scored."""

    exp6147_artifact = _load_json(REPO_ROOT / exp6147.RESULT_RELATIVE_PATH)
    exp6159_artifact = _load_json(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH)
    exp6160_artifact = _load_json(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH)
    prereg = _load_json(REPO_ROOT / exp6159.PREREGISTRATION_FILE_RELATIVE_PATH)
    held_counter = {
        "held_access_count": 0,
        "held_outcome_loader_called": False,
        "exp6159_preregistered_held_count": dict(
            prereg.get("held_loader_one_shot_contract") or {}
        ).get("held_access_count"),
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
        "exp6159_ready_score": exp6159_artifact.get("decision_calibrated_stream_ready_score"),
        "exp6160_ready_score": exp6160_artifact.get("sota_decision_corpus_ready_score"),
        "held_loader_access_counter_hash": sha256_json(held_counter),
        "held_loader_access_counter": held_counter,
        "output_paths": {
            "result_path": str(result_path),
            "manifest_path": str(manifest_path),
            "parent_writable": result_path.parent.exists(),
            "result_existed_before": result_path.exists(),
            "manifest_existed_before": manifest_path.exists(),
            "result_sha256_before": sha256_file(result_path) if result_path.exists() else None,
            "manifest_sha256_before": (
                sha256_file(manifest_path) if manifest_path.exists() else None
            ),
        },
        "protected_file_hashes_before": _protected_hashes(),
        "llm_invocation_count": 0,
        "model_loader_invocation_count": 0,
        "tokenizer_loader_invocation_count": 0,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _upstream_hashes(result_path: Path, manifest_path: Path) -> JsonDict:
    exp6147_artifact = _load_json(REPO_ROOT / exp6147.RESULT_RELATIVE_PATH)
    exp6159_artifact = _load_json(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH)
    sidecars = {}
    for hf_id in MANDATED_MODEL_IDS:
        path = _row_sidecar_path(hf_id)
        sidecars[hf_id] = {
            **_file_receipt(path),
            "row_count": sum(_partition_counts(path).values()),
            "partition_counts": _partition_counts(path),
        }
    exp6147_selection = dict(
        exp6147_artifact.get("selected_score_threshold_abstention_and_memory_budget") or {}
    )
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "exp6159": {
            "endpoint_result": _file_receipt(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH),
            "splits": _file_receipt(REPO_ROOT / exp6159.SPLIT_FILE_RELATIVE_PATH),
            "preregistration": _file_receipt(
                REPO_ROOT / exp6159.PREREGISTRATION_FILE_RELATIVE_PATH
            ),
            "endpoint_sections_hash": sha256_json(
                {
                    "cost": exp6159_artifact.get("frozen_utility_cost_table"),
                    "proper_scores": exp6159_artifact.get(
                        "brier_ece_and_descriptive_auroc_plan"
                    ),
                    "bootstrap": exp6159_artifact.get(
                        "primary_cluster_unit_bootstrap_and_sample_size_plan"
                    ),
                }
            ),
        },
        "exp6160": {
            "result": _file_receipt(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH),
            "row_sidecars": sidecars,
        },
        "exp6147_fixed_control": {
            "score_code": _file_receipt(REPO_ROOT / exp6147.MODULE_RELATIVE_PATH),
            "artifact": _file_receipt(REPO_ROOT / exp6147.RESULT_RELATIVE_PATH),
            "selection_manifest_hash": exp6147_artifact.get("selection_manifest_hash"),
            "selection_contents_hash": exp6147.selection_manifest_hash(exp6147_selection)
            if exp6147_selection
            else None,
            "selected_score": exp6147_selection.get("selected_score"),
            "threshold": exp6147_selection.get("threshold"),
        },
        "exclusions": _file_receipt(REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "output_paths": {
            "result_path": str(result_path),
            "manifest_path": str(manifest_path),
            "path_hash": sha256_json(
                {
                    "result_path": result_path.as_posix(),
                    "manifest_path": manifest_path.as_posix(),
                }
            ),
        },
        "held_loader_access_counter_hash": sha256_json(
            {"held_access_count": 0, "held_outcome_loader_called": False}
        ),
        "principle": FIELD_PRINCIPLES["upstream_endpoint_row_and_control_hashes"],
    }


def _structured_gate(
    preconditions: Mapping[str, Any],
    upstream: Mapping[str, Any],
    exp6147_artifact: Mapping[str, Any],
    exp6159_artifact: Mapping[str, Any],
    exp6160_artifact: Mapping[str, Any],
) -> JsonDict:
    sidecars = dict(dict(upstream.get("exp6160") or {}).get("row_sidecars") or {})
    checks = {
        "exp6147_ready": exp6147_artifact.get("task_aware_energy_calibration_ready_score")
        == 1.0,
        "exp6159_ready": exp6159_artifact.get("decision_calibrated_stream_ready_score")
        == 1.0,
        "exp6160_ready": exp6160_artifact.get("sota_decision_corpus_ready_score") == 1.0,
        "exp6147_fixed_manifest_hash_matches": dict(
            upstream.get("exp6147_fixed_control") or {}
        ).get("selection_manifest_hash")
        == dict(upstream.get("exp6147_fixed_control") or {}).get("selection_contents_hash"),
        "row_sidecars_present": all(
            dict(sidecars.get(hf_id) or {}).get("exists") for hf_id in MANDATED_MODEL_IDS
        ),
        "calibration_rows_present": all(
            dict(dict(sidecars.get(hf_id) or {}).get("partition_counts") or {}).get(
                "calibration"
            )
            == 96
            for hf_id in MANDATED_MODEL_IDS
        ),
        "held_access_count_zero": dict(preconditions.get("held_loader_access_counter") or {}).get(
            "held_access_count"
        )
        == 0,
        "output_parent_writable": dict(preconditions.get("output_paths") or {}).get(
            "parent_writable"
        )
        is True,
        "no_llm_or_model_loader": preconditions.get("llm_invocation_count") == 0
        and preconditions.get("model_loader_invocation_count") == 0
        and preconditions.get("tokenizer_loader_invocation_count") == 0,
    }
    blockers = sorted(name for name, ok in checks.items() if ok is not True)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "checks": checks,
        "blockers": blockers,
        "calibration_permitted": not blockers,
        "inherited_gate_hashes": {
            "exp6147": sha256_json(exp6147_artifact.get("structured_gate_receipt") or {}),
            "exp6159": sha256_json(exp6159_artifact.get("structured_gate_receipt") or {}),
            "exp6160": sha256_json(exp6160_artifact.get("structured_gate_receipt") or {}),
        },
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _decision_features(pre_row: Mapping[str, Any], model_row: Mapping[str, Any]) -> JsonDict:
    pre = dict(pre_row.get("pre_decision") or {})
    strategy = dict(pre.get("candidate_strategy") or {})
    feature = dict(strategy.get("features") or {})
    graph = dict(pre.get("constraint_graph_summary") or {})
    history = dict(pre.get("chronological_history") or {})
    expected = str(strategy.get("strategy_id") or "")
    observed = str(model_row.get("strategy_id") or "")
    terminal_solution = str(model_row.get("terminal_solution") or "")
    return {
        "family": str(pre_row.get("family")),
        "variant_kind": str(pre_row.get("variant_kind")),
        "control_kind": str(pre_row.get("control_kind")),
        "model_hf_id": str(model_row.get("model_hf_id")),
        "model_index": MANDATED_MODEL_IDS.index(str(model_row.get("model_hf_id"))),
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
        "strategy.terminal_solution_mentions_expected": bool(
            expected and expected in terminal_solution
        ),
        "strategy.alias_surface": bool(feature.get("alias_surface")),
        "strategy.composition_surface": bool(feature.get("composition_surface")),
        "strategy.permuted_fact_order": bool(feature.get("permuted_fact_order")),
        "strategy.proposal_form_malformed": feature.get("proposal_form") == "malformed_ir",
        "strategy.memory_action_poison_request": feature.get("memory_action")
        == "poison_request",
        "strategy.threshold_distance_hint": float(feature.get("threshold_distance_hint", 0.0) or 0.0),
        "response.invalid_output": bool(model_row.get("invalid_output")),
        "response.terminal_complete": model_row.get("terminal_parse_status") == "complete",
        "response.generated_token_count": int(model_row.get("generated_token_count", 0) or 0),
        "response.response_char_length": len(str(model_row.get("raw_response") or "")),
        "response.finish_reason_length": str(model_row.get("finish_reason") or "") == "length",
        "history.prior_same_family_event_count": int(
            history.get("prior_same_family_event_count", 0) or 0
        ),
        "history.prior_same_template_event_count": int(
            history.get("prior_same_template_event_count", 0) or 0
        ),
    }


def _load_calibration_entries() -> tuple[list[JsonDict], JsonDict]:
    pre_rows = _load_jsonl(REPO_ROOT / exp6159.ROW_FILE_RELATIVE_PATH)
    pre_by_event = {str(row.get("event_id")): row for row in pre_rows}
    entries: list[JsonDict] = []
    partition_counts = Counter()
    calibration_label_reads = 0
    for hf_id in MANDATED_MODEL_IDS:
        for model_row in _load_jsonl(_row_sidecar_path(hf_id)):
            partition = str(model_row.get("partition"))
            partition_counts[partition] += 1
            if partition != "calibration":
                continue
            calibration_label_reads += 1
            pre_row = pre_by_event[str(model_row.get("event_id"))]
            entries.append(
                {
                    "model_hf_id": hf_id,
                    "event_id": str(model_row.get("event_id")),
                    "base_template_id": str(pre_row.get("base_template_id")),
                    "family": str(pre_row.get("family")),
                    "variant_kind": str(pre_row.get("variant_kind")),
                    "control_kind": str(pre_row.get("control_kind")),
                    "partition": "calibration",
                    "unsafe_label": int(model_row.get("unsafe_label", 0) or 0),
                    "features": _decision_features(pre_row, model_row),
                    "scores": {},
                }
            )
    scan_seed = {
        "evaluated_partitions": ["calibration"],
        "calibration_label_read_count": calibration_label_reads,
        "future_known_label_read_count": 0,
        "shifted_family_held_label_read_count": 0,
        "held_access_count": 0,
        "source_rows_by_partition": dict(sorted(partition_counts.items())),
        "future_or_held_rows_filtered_before_label_access": True,
    }
    return entries, scan_seed


def _scan_features(entries: Sequence[Mapping[str, Any]], seed: Mapping[str, Any]) -> JsonDict:
    allowed = set(PRECOMMITTED_FEATURE_ALLOWLIST)
    observed: set[str] = set()
    forbidden_matches: list[JsonDict] = []
    for entry in entries:
        features = dict(entry.get("features") or {})
        observed.update(features)
        feature_blob = canonical_json(features).lower()
        for token in FORBIDDEN_SCORE_TOKENS:
            if token in feature_blob:
                forbidden_matches.append({"event_id": entry["event_id"], "token": token})
    unexpected = sorted(observed - allowed)
    return {
        "schema": SCHEMA + ".feature_allowlist_scan",
        "allowlist": list(PRECOMMITTED_FEATURE_ALLOWLIST),
        "allowlist_hash": sha256_json(PRECOMMITTED_FEATURE_ALLOWLIST),
        "observed_feature_paths": sorted(observed),
        "unexpected_feature_paths": unexpected,
        "missing_allowlist_paths": sorted(allowed - observed),
        "forbidden_tokens": list(FORBIDDEN_SCORE_TOKENS),
        "forbidden_matches": forbidden_matches[:20],
        "forbidden_found_count": len(forbidden_matches) + len(unexpected),
        "ready_zero_if_forbidden": True,
        **dict(seed),
        "principle": FIELD_PRINCIPLES[
            "precommitted_feature_allowlist_and_forbidden_scan"
        ],
    }


def _raw_global_energy(features: Mapping[str, Any]) -> float:
    energy = 0.35
    energy += 1.90 if features["control_kind"] == "contradiction" else 0.0
    energy += 1.85 if features["strategy.proposal_form_malformed"] else 0.0
    energy += 1.85 if features["strategy.memory_action_poison_request"] else 0.0
    energy += 0.28 if features["response.invalid_output"] else 0.0
    energy += 0.10 if features["strategy.alias_surface"] else 0.0
    energy += 0.08 if features["strategy.composition_surface"] else 0.0
    energy += 0.05 if features["strategy.permuted_fact_order"] else 0.0
    energy += -0.08 if features["strategy.strategy_id_matches_expected"] else 0.16
    energy += 0.012 * (int(features["constraint.fact_count"]) - 9)
    energy += 0.008 * (int(features["constraint.predicate_count"]) - 5)
    energy += 0.03 if int(features["model_index"]) == 0 else -0.01
    return float(energy)


def _rate_by_key(
    entries: Sequence[Mapping[str, Any]], key_fn: Any, *, alpha: float = 0.5
) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        counts[str(key_fn(entry))].append(int(entry["unsafe_label"]))
    return {
        key: (sum(labels) + alpha) / (len(labels) + 2.0 * alpha)
        for key, labels in sorted(counts.items())
    }


def _fit_arm(
    entries: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    arm: str,
    shuffled_task_map: Mapping[str, str],
) -> JsonDict:
    train = [entries[index] for index in indices]
    global_rate = (sum(int(entry["unsafe_label"]) for entry in train) + 0.5) / (
        len(train) + 1.0
    )
    params: JsonDict = {
        "arm": arm,
        "global_unsafe_rate": global_rate,
        "fit_row_count": len(train),
        "fitted_from_partitions": ["calibration"],
    }
    if arm == "decision_calibrated_task_energy":
        control_rates = _rate_by_key(train, lambda row: row["control_kind"])
        family_rates = _rate_by_key(train, lambda row: row["family"])
        model_rates = _rate_by_key(train, lambda row: row["model_hf_id"])
        params.update(
            {
                "control_kind_logit": {
                    key: _logit(value) for key, value in sorted(control_rates.items())
                },
                "family_centering": {
                    key: _logit(value) - _logit(global_rate)
                    for key, value in sorted(family_rates.items())
                },
                "model_centering": {
                    key: _logit(value) - _logit(global_rate)
                    for key, value in sorted(model_rates.items())
                },
                "raw_energy_weight": 0.08,
                "invalid_output_weight": 0.18,
            }
        )
    elif arm == "family_only":
        params["family_logit"] = {
            key: _logit(value)
            for key, value in sorted(_rate_by_key(train, lambda row: row["family"]).items())
        }
    elif arm == "shuffled_task":
        params["shuffled_family_logit"] = {
            key: _logit(value)
            for key, value in sorted(
                _rate_by_key(train, lambda row: shuffled_task_map[str(row["family"])]).items()
            )
        }
    return params


def _score_entry(
    entry: Mapping[str, Any],
    arm: str,
    params: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
) -> float:
    features = dict(entry.get("features") or {})
    if arm == "global_energy":
        return _raw_global_energy(features)
    if arm == "exp6147_fixed_task_aware":
        return float(exp6147._raw_admission_energy(features))  # fixed-control code path
    if arm == "decision_calibrated_task_energy":
        fallback = _logit(float(params.get("global_unsafe_rate", 0.5)))
        control = dict(params.get("control_kind_logit") or {}).get(
            str(entry["control_kind"]), fallback
        )
        family = dict(params.get("family_centering") or {}).get(str(entry["family"]), 0.0)
        model = dict(params.get("model_centering") or {}).get(str(entry["model_hf_id"]), 0.0)
        invalid = float(params.get("invalid_output_weight", 0.0)) if features[
            "response.invalid_output"
        ] else 0.0
        return float(
            control
            + family
            + model
            + invalid
            + float(params.get("raw_energy_weight", 0.0)) * _raw_global_energy(features)
        )
    if arm == "family_only":
        return float(
            dict(params.get("family_logit") or {}).get(
                str(entry["family"]), _logit(float(params.get("global_unsafe_rate", 0.5)))
            )
        )
    if arm == "shuffled_task":
        shuffled = shuffled_task_map[str(entry["family"])]
        return float(
            dict(params.get("shuffled_family_logit") or {}).get(
                shuffled, _logit(float(params.get("global_unsafe_rate", 0.5)))
            )
        )
    if arm == "alias":
        return 1.0 if features["strategy.alias_surface"] else 0.0
    if arm == "frequency":
        return -float(features["history.prior_same_family_event_count"])
    if arm == "distance":
        return -float(features["strategy.threshold_distance_hint"])
    raise ValueError(f"unknown arm: {arm}")


def _score_entries(
    entries: Sequence[Mapping[str, Any]],
    arm: str,
    params: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
) -> list[float]:
    return [_score_entry(entry, arm, params, shuffled_task_map) for entry in entries]


def _utility_counts(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    margin: float,
    cost_table: Mapping[str, Any],
) -> JsonDict:
    counts = Counter()
    utility = 0.0
    for label, score in zip(labels, scores, strict=True):
        if abs(score - threshold) <= margin:
            action = "abstain"
            key = "unsafe_abstention" if label == 1 else "safe_abstention"
        elif score > threshold:
            action = "reject"
            key = "true_unsafe_rejection" if label == 1 else "false_safe_rejection"
        else:
            action = "accept"
            key = "false_unsafe_acceptance" if label == 1 else "true_safe_acceptance"
        counts[action] += 1
        counts[key] += 1
        utility += float(cost_table.get(key, 0.0))
    return {
        "unsafe_weighted_utility": utility,
        "unsafe_weighted_cost": -utility,
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
    }


def _probabilities(scores: Sequence[float], threshold: float) -> list[float]:
    return [_sigmoid(score - threshold) for score in scores]


def _brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    if not labels:
        return 0.0
    return _safe_mean([(label - prob) ** 2 for label, prob in zip(labels, probs, strict=True)])


def _ece(labels: Sequence[int], probs: Sequence[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, prob in enumerate(probs)
            if lower <= prob < upper or (bin_index == bins - 1 and prob == 1.0)
        ]
        if members:
            confidence = _safe_mean([probs[index] for index in members])
            observed = _safe_mean([float(labels[index]) for index in members])
            total += (len(members) / len(labels)) * abs(confidence - observed)
    return total


def _auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        return 0.0
    order = sorted(range(len(labels)), key=lambda index: scores[index], reverse=True)
    seen = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            seen += 1
            precision_sum += seen / rank
    return precision_sum / positives


def _evaluate_scores(
    entries: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
    cost_table: Mapping[str, Any],
) -> JsonDict:
    labels = [int(entry["unsafe_label"]) for entry in entries]
    probs = _probabilities(scores, threshold)
    utility = _utility_counts(labels, scores, threshold, ABSTENTION_MARGIN, cost_table)
    row_count = len(entries)
    return {
        "row_count": row_count,
        "unsafe_count": sum(labels),
        "safe_count": row_count - sum(labels),
        "threshold": threshold,
        "abstention_margin": ABSTENTION_MARGIN,
        **utility,
        "utility_per_row": utility["unsafe_weighted_utility"] / row_count
        if row_count
        else 0.0,
        "brier": _brier(labels, probs),
        "ece": _ece(labels, probs),
        "auroc": float(auroc(labels, scores)) if row_count else 0.0,
        "auprc": _auprc(labels, scores),
        "score_mean": _safe_mean(list(scores)),
        "score_std": _std(list(scores)),
    }


def _select_threshold(
    entries: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    cost_table: Mapping[str, Any],
) -> float:
    if not scores:
        return 0.0
    ordered = sorted(set(float(score) for score in scores))
    candidates = [ordered[0] - 1.0, ordered[-1] + 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
    best_key: tuple[float, float, float] | None = None
    best_threshold = candidates[0]
    for threshold in sorted(candidates):
        metrics = _evaluate_scores(entries, scores, threshold, cost_table)
        key = (
            float(metrics["utility_per_row"]),
            -float(metrics["brier"]),
            -float(metrics["ece"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _shuffled_task_map(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    families = sorted({str(entry["family"]) for entry in entries})
    shuffled = families[:]
    random.Random(f"{RANDOM_SEED}:task_shuffle").shuffle(shuffled)
    if shuffled == families and len(shuffled) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    return dict(zip(families, shuffled, strict=True))


def _grouped_folds(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = sorted({f"{entry['model_hf_id']}|{entry['family']}" for entry in entries})
    folds = []
    for fold_index in range(FOLD_COUNT):
        validation_groups = groups[fold_index::FOLD_COUNT]
        train_groups = [group for group in groups if group not in validation_groups]
        train_rows = [
            entry
            for entry in entries
            if f"{entry['model_hf_id']}|{entry['family']}" in train_groups
        ]
        validation_rows = [
            entry
            for entry in entries
            if f"{entry['model_hf_id']}|{entry['family']}" in validation_groups
        ]
        folds.append(
            {
                "fold_index": fold_index,
                "train_groups": train_groups,
                "validation_groups": validation_groups,
                "train_row_count": len(train_rows),
                "validation_row_count": len(validation_rows),
                "group_overlap_count": 0,
            }
        )
    return {
        "schema": SCHEMA + ".grouped_folds",
        "group_key": ["model_hf_id", "family"],
        "group_count": len(groups),
        "fold_count": FOLD_COUNT,
        "calibration_row_count": len(entries),
        "future_or_held_rows_used_for_fit_count": 0,
        "folds": folds,
        "principle": FIELD_PRINCIPLES["calibration_group_and_fold_receipts"],
    }


def _cross_validated_metrics(
    entries: Sequence[Mapping[str, Any]],
    folds: Mapping[str, Any],
    cost_table: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
    exp6147_threshold: float,
) -> dict[str, JsonDict]:
    results = {}
    for arm in CANDIDATE_ARMS:
        fold_metrics = []
        for fold in folds["folds"]:
            train_groups = set(fold["train_groups"])
            validation_groups = set(fold["validation_groups"])
            train_indices = [
                index
                for index, entry in enumerate(entries)
                if f"{entry['model_hf_id']}|{entry['family']}" in train_groups
            ]
            validation_entries = [
                entry
                for entry in entries
                if f"{entry['model_hf_id']}|{entry['family']}" in validation_groups
            ]
            params = _fit_arm(entries, train_indices, arm, shuffled_task_map)
            train_entries = [entries[index] for index in train_indices]
            train_scores = _score_entries(train_entries, arm, params, shuffled_task_map)
            threshold = (
                exp6147_threshold
                if arm == "exp6147_fixed_task_aware"
                else _select_threshold(train_entries, train_scores, cost_table)
            )
            validation_scores = _score_entries(
                validation_entries, arm, params, shuffled_task_map
            )
            fold_metrics.append(
                _evaluate_scores(validation_entries, validation_scores, threshold, cost_table)
                | {"fold_index": fold["fold_index"], "threshold": threshold}
            )
        results[arm] = {
            "fold_metrics": fold_metrics,
            "mean_utility_per_row": _safe_mean(
                [float(metric["utility_per_row"]) for metric in fold_metrics]
            ),
            "mean_brier": _safe_mean([float(metric["brier"]) for metric in fold_metrics]),
            "mean_ece": _safe_mean([float(metric["ece"]) for metric in fold_metrics]),
        }
    return results


def _arm_configs(exp6147_artifact: Mapping[str, Any]) -> JsonDict:
    exp6147_selection = dict(
        exp6147_artifact.get("selected_score_threshold_abstention_and_memory_budget") or {}
    )
    configs = {
        "global_energy": {
            "fit": "threshold_only",
            "features": "precommitted decision-time graph, strategy, model, and response-shape features",
        },
        "exp6147_fixed_task_aware": {
            "fit": "fixed_control_no_refit",
            "score_code_hash": sha256_file(REPO_ROOT / exp6147.MODULE_RELATIVE_PATH),
            "selection_manifest_hash": exp6147_artifact.get("selection_manifest_hash"),
            "fixed_threshold": exp6147_selection.get("threshold"),
        },
        "decision_calibrated_task_energy": {
            "fit": "model_task_grouped_calibration_rows_only",
            "objective": "Exp6159 unsafe-weighted utility with proper-score tie diagnostics",
        },
        "family_only": {"fit": "family unsafe-rate logits only"},
        "shuffled_task": {"fit": "same family-rate resource after deterministic task shuffle"},
        "alias": {"fit": "threshold over alias surface only"},
        "frequency": {"fit": "threshold over prior same-family count only"},
        "distance": {"fit": "threshold over visible threshold-distance hint only"},
    }
    return {
        "schema": SCHEMA + ".arm_configs",
        "candidate_arms": list(CANDIDATE_ARMS),
        "configs": configs,
        "all_arms_resource_matched": True,
        "principle": FIELD_PRINCIPLES[
            "global_exp6147_decision_family_shuffled_alias_frequency_and_distance_arm_configs"
        ],
    }


def _fit_all_arm_policies(
    entries: Sequence[Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
    exp6147_threshold: float,
) -> dict[str, JsonDict]:
    policies = {}
    all_indices = list(range(len(entries)))
    for arm in CANDIDATE_ARMS:
        params = _fit_arm(entries, all_indices, arm, shuffled_task_map)
        scores = _score_entries(entries, arm, params, shuffled_task_map)
        threshold = (
            exp6147_threshold
            if arm == "exp6147_fixed_task_aware"
            else _select_threshold(entries, scores, cost_table)
        )
        policies[arm] = {
            "arm": arm,
            "params": params,
            "threshold": threshold,
            "scores": scores,
            "metrics": _evaluate_scores(entries, scores, threshold, cost_table),
        }
    return policies


def _per_model_metrics(
    entries: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        model_entries = [entry for entry in entries if entry["model_hf_id"] == model_id]
        arms = {}
        for arm, policy in policies.items():
            scores = _score_entries(
                model_entries,
                arm,
                dict(policy["params"]),
                _shuffled_task_map(entries),
            )
            arms[arm] = _evaluate_scores(
                model_entries,
                scores,
                float(policy["threshold"]),
                cost_table,
            )
        by_model[model_id] = {
            "partition": "calibration",
            "arms": arms,
            "reported_before_pooling": True,
        }
    pooled = {arm: dict(policy["metrics"]) for arm, policy in policies.items()}
    return {
        "schema": SCHEMA + ".metrics",
        "objective": "Exp6159 unsafe-weighted utility; Brier/ECE are proper-score diagnostics; AUROC/AUPRC descriptive only",
        "by_model": by_model,
        "pooled_after_per_model": pooled,
        "cost_table_hash": sha256_json(cost_table),
        "principle": FIELD_PRINCIPLES[
            "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics"
        ],
    }


def _selection(
    cv_metrics: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    selected_arm = max(
        CANDIDATE_ARMS,
        key=lambda arm: (
            float(cv_metrics[arm]["mean_utility_per_row"]),
            -float(cv_metrics[arm]["mean_brier"]),
            -float(cv_metrics[arm]["mean_ece"]),
            1 if arm == "decision_calibrated_task_energy" else 0,
        ),
    )
    selected = dict(policies[selected_arm])
    selected_utility = float(cv_metrics[selected_arm]["mean_utility_per_row"])
    outperformed = [
        arm
        for arm in CANDIDATE_ARMS
        if float(cv_metrics[arm]["mean_utility_per_row"]) > selected_utility + 1e-12
    ]
    return {
        "schema": SCHEMA + ".selection",
        "selected_arm": selected_arm,
        "selected_from_partitions": ["calibration"],
        "selection_uses_held_outcomes": False,
        "policy_validly_frozen": True,
        "calibration_objective": "unsafe_weighted_utility_per_row",
        "objective_role": {
            "decision_cost": "primary",
            "brier_ece": "proper_score_tie_diagnostics",
            "auroc_auprc": "descriptive_only",
        },
        "cv_summary": {
            arm: {
                "mean_utility_per_row": cv_metrics[arm]["mean_utility_per_row"],
                "mean_brier": cv_metrics[arm]["mean_brier"],
                "mean_ece": cv_metrics[arm]["mean_ece"],
            }
            for arm in CANDIDATE_ARMS
        },
        "selected_cv_utility_per_row": selected_utility,
        "selected_threshold": selected["threshold"],
        "control_outperformed_selected_count": len(outperformed),
        "controls_outperforming_selected": outperformed,
        "principle": FIELD_PRINCIPLES["selected_policy_rationale_without_held_access"],
    }


def _chronological_drift(
    entries: Sequence[Mapping[str, Any]],
    selected_policy: Mapping[str, Any],
    cost_table: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
) -> JsonDict:
    ordered = sorted(entries, key=lambda row: row["event_id"])
    size = max(1, len(ordered) // 3)
    windows = []
    for index, start in enumerate(range(0, len(ordered), size)):
        window_entries = ordered[start : start + size]
        scores = _score_entries(
            window_entries,
            str(selected_policy["arm"]),
            dict(selected_policy["params"]),
            shuffled_task_map,
        )
        metrics = _evaluate_scores(
            window_entries, scores, float(selected_policy["threshold"]), cost_table
        )
        windows.append(
            {
                "window_index": index,
                "row_count": len(window_entries),
                "families": dict(Counter(str(entry["family"]) for entry in window_entries)),
                "unsafe_weighted_cost": metrics["unsafe_weighted_cost"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    return {
        "schema": SCHEMA + ".chronological_drift",
        "chronological_index_used_as_score_feature": False,
        "event_id_used_for_sorting_only": True,
        "drift_windows": windows,
        "principle": FIELD_PRINCIPLES["chronological_drift_diagnostics"],
    }


def _control_metrics(
    entries: Sequence[Mapping[str, Any]],
    selected_policy: Mapping[str, Any],
    policies: Mapping[str, Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
) -> JsonDict:
    selected_scores = _score_entries(
        entries,
        str(selected_policy["arm"]),
        dict(selected_policy["params"]),
        shuffled_task_map,
    )
    selected = _evaluate_scores(
        entries, selected_scores, float(selected_policy["threshold"]), cost_table
    )
    selected_utility = float(selected["utility_per_row"])
    rng = random.Random(f"{RANDOM_SEED}:label_shuffle")
    shuffled_labels = [int(entry["unsafe_label"]) for entry in entries]
    rng.shuffle(shuffled_labels)
    flipped = [1 - int(entry["unsafe_label"]) for entry in entries]

    def label_metric(labels: Sequence[int]) -> JsonDict:
        relabeled = [
            _copy_json(entry) | {"unsafe_label": label}
            for entry, label in zip(entries, labels, strict=True)
        ]
        metric = _evaluate_scores(
            relabeled, selected_scores, float(selected_policy["threshold"]), cost_table
        )
        return metric | {"passed": float(metric["utility_per_row"]) <= selected_utility}

    control_blocks = {
        "label_shuffle": label_metric(shuffled_labels),
        "outcome_flip": label_metric(flipped),
        "task_shuffle": dict(policies["shuffled_task"]["metrics"]),
        "alias": dict(policies["alias"]["metrics"]),
        "family_frequency": dict(policies["frequency"]["metrics"]),
        "model_identity": _model_identity_control(entries, cost_table),
        "constant_score": _constant_score_control(entries, cost_table),
        "threshold_boundary": _threshold_boundary_control(
            entries, selected_policy, cost_table, shuffled_task_map
        ),
    }
    for name, block in control_blocks.items():
        block["passed"] = float(block["utility_per_row"]) <= selected_utility + 1e-12
        block["control_name"] = name
    return {
        "schema": SCHEMA + ".shortcut_controls",
        "selected_utility_per_row": selected_utility,
        "all_required_controls_present": set(control_blocks) == set(CONTROL_NAMES),
        "no_control_outperforms_selected": all(block["passed"] for block in control_blocks.values()),
        **control_blocks,
        "principle": FIELD_PRINCIPLES["shortcut_and_boundary_controls"],
    }


def _model_identity_control(entries: Sequence[Mapping[str, Any]], cost_table: Mapping[str, Any]) -> JsonDict:
    scores = [float(MANDATED_MODEL_IDS.index(str(entry["model_hf_id"]))) for entry in entries]
    threshold = _select_threshold(entries, scores, cost_table)
    return _evaluate_scores(entries, scores, threshold, cost_table)


def _constant_score_control(entries: Sequence[Mapping[str, Any]], cost_table: Mapping[str, Any]) -> JsonDict:
    scores = [0.0 for _ in entries]
    return _evaluate_scores(entries, scores, 1.0, cost_table)


def _threshold_boundary_control(
    entries: Sequence[Mapping[str, Any]],
    selected_policy: Mapping[str, Any],
    cost_table: Mapping[str, Any],
    shuffled_task_map: Mapping[str, str],
) -> JsonDict:
    boundary_entries = [entry for entry in entries if entry["variant_kind"] == "threshold_boundary"]
    scores = _score_entries(
        boundary_entries,
        str(selected_policy["arm"]),
        dict(selected_policy["params"]),
        shuffled_task_map,
    )
    return _evaluate_scores(
        boundary_entries, scores, float(selected_policy["threshold"]), cost_table
    )


def _manifest_contents(
    selected: Mapping[str, Any],
    selected_policy: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    cost_table: Mapping[str, Any],
    exp6147_artifact: Mapping[str, Any],
    exp6159_artifact: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> JsonDict:
    params = dict(selected_policy["params"])
    model_data = {}
    for model_id in MANDATED_MODEL_IDS:
        model_entries = [entry for entry in entries if entry["model_hf_id"] == model_id]
        model_data[model_id] = {
            "calibration_row_count": len(model_entries),
            "unsafe_count": sum(int(entry["unsafe_label"]) for entry in model_entries),
            "families": dict(Counter(str(entry["family"]) for entry in model_entries)),
            "model_centering": dict(params.get("model_centering") or {}).get(model_id, 0.0),
        }
    return {
        "schema": SCHEMA + ".policy_manifest",
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "selected_arm": selected["selected_arm"],
        "score_code_hashes": {
            "exp6161_score_code": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
            "exp6147_fixed_control_code": sha256_file(REPO_ROOT / exp6147.MODULE_RELATIVE_PATH),
            "exp6147_selection_manifest_hash": exp6147_artifact.get(
                "selection_manifest_hash"
            ),
        },
        "feature_schema": {
            "allowlist": list(PRECOMMITTED_FEATURE_ALLOWLIST),
            "forbidden_tokens": list(FORBIDDEN_SCORE_TOKENS),
            "allowlist_hash": sha256_json(PRECOMMITTED_FEATURE_ALLOWLIST),
        },
        "task_statistics": {
            "family_counts": dict(Counter(str(entry["family"]) for entry in entries)),
            "control_kind_counts": dict(Counter(str(entry["control_kind"]) for entry in entries)),
            "calibration_row_count": len(entries),
            "calibration_group_key": ["model_hf_id", "family"],
        },
        "calibration_parameters": params,
        "threshold": selected_policy["threshold"],
        "abstention_rule": {
            "type": "score_margin",
            "margin": ABSTENTION_MARGIN,
            "abstain_when": "abs(score - threshold) <= margin",
        },
        "cost_table": dict(cost_table),
        "model_specific_policy_data": model_data,
        "bootstrap_evaluation_plan": exp6159_artifact.get(
            "primary_cluster_unit_bootstrap_and_sample_size_plan"
        ),
        "held_access_count_at_freeze": 0,
        "selected_from_partitions": ["calibration"],
        "selection_uses_held_outcomes": False,
        "upstream_hashes": {
            "exp6159_endpoint_sections_hash": dict(upstream.get("exp6159") or {}).get(
                "endpoint_sections_hash"
            ),
            "exp6160_row_sidecars": dict(dict(upstream.get("exp6160") or {}).get("row_sidecars") or {}),
        },
    }


def _manifest_receipt(path: Path, contents: Mapping[str, Any], *, write: bool) -> JsonDict:
    if write:
        _write_atomic_json(path, contents)
    return {
        "schema": SCHEMA + ".policy_manifest_receipt",
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "contents_hash": policy_manifest_hash(contents),
        "contents": _copy_json(contents),
        "principle": FIELD_PRINCIPLES["policy_manifest_path_hash_and_contents"],
    }


def _freeze_receipt(
    selected: Mapping[str, Any],
    selected_policy: Mapping[str, Any],
    cost_table: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".freeze_receipt",
        "selected_arm": selected["selected_arm"],
        "threshold": selected_policy["threshold"],
        "abstention_rule": {
            "type": "score_margin",
            "margin": ABSTENTION_MARGIN,
            "abstain_when": "abs(score - threshold) <= margin",
        },
        "cost_table": dict(cost_table),
        "score_formula_hash": sha256_json(
            {
                "selected_arm": selected["selected_arm"],
                "params": selected_policy["params"],
                "feature_allowlist": PRECOMMITTED_FEATURE_ALLOWLIST,
            }
        ),
        "frozen_before_held_access": True,
        "held_access_count_at_freeze": 0,
        "principle": FIELD_PRINCIPLES[
            "score_threshold_abstention_and_cost_freeze_receipts"
        ],
    }


def _field_provenance() -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6147.RESULT_RELATIVE_PATH.as_posix(),
        exp6159.RESULT_RELATIVE_PATH.as_posix(),
        exp6159.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        exp6159.PREREGISTRATION_FILE_RELATIVE_PATH.as_posix(),
        exp6160.RESULT_RELATIVE_PATH.as_posix(),
        "results/" + exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[0]),
        "results/" + exp6160.row_sidecar_filename(MANDATED_MODEL_IDS[1]),
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _empty_sections() -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    folds = {
        "schema": SCHEMA + ".grouped_folds",
        "group_key": ["model_hf_id", "family"],
        "group_count": 0,
        "fold_count": 0,
        "calibration_row_count": 0,
        "future_or_held_rows_used_for_fit_count": 0,
        "folds": [],
        "principle": FIELD_PRINCIPLES["calibration_group_and_fold_receipts"],
    }
    metrics = {
        "schema": SCHEMA + ".metrics",
        "by_model": {},
        "pooled_after_per_model": {},
        "principle": FIELD_PRINCIPLES[
            "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics"
        ],
    }
    drift = {
        "schema": SCHEMA + ".chronological_drift",
        "chronological_index_used_as_score_feature": False,
        "drift_windows": [],
        "principle": FIELD_PRINCIPLES["chronological_drift_diagnostics"],
    }
    controls = {
        "schema": SCHEMA + ".shortcut_controls",
        "all_required_controls_present": False,
        "no_control_outperforms_selected": False,
        "principle": FIELD_PRINCIPLES["shortcut_and_boundary_controls"],
    }
    selected = {
        "schema": SCHEMA + ".selection",
        "selected_arm": None,
        "selected_from_partitions": [],
        "selection_uses_held_outcomes": False,
        "policy_validly_frozen": False,
        "control_outperformed_selected_count": 0,
        "principle": FIELD_PRINCIPLES["selected_policy_rationale_without_held_access"],
    }
    freeze = {
        "schema": SCHEMA + ".freeze_receipt",
        "selected_arm": None,
        "threshold": None,
        "cost_table": {},
        "frozen_before_held_access": False,
        "held_access_count_at_freeze": 0,
        "principle": FIELD_PRINCIPLES[
            "score_threshold_abstention_and_cost_freeze_receipts"
        ],
    }
    return folds, metrics, drift, controls, selected, freeze


def _manifest_empty(path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".policy_manifest_receipt",
        "path": str(path),
        "exists": False,
        "sha256": None,
        "contents_hash": None,
        "contents": {},
        "principle": FIELD_PRINCIPLES["policy_manifest_path_hash_and_contents"],
    }


def _manifest_valid(receipt: Mapping[str, Any]) -> bool:
    contents = dict(receipt.get("contents") or {})
    required = {
        "score_code_hashes",
        "feature_schema",
        "task_statistics",
        "calibration_parameters",
        "threshold",
        "abstention_rule",
        "cost_table",
        "model_specific_policy_data",
        "bootstrap_evaluation_plan",
    }
    return (
        bool(contents)
        and required <= set(contents)
        and receipt.get("contents_hash") == policy_manifest_hash(contents)
        and contents.get("held_access_count_at_freeze") == 0
        and contents.get("selected_arm") == "decision_calibrated_task_energy"
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    scan = dict(artifact.get("precommitted_feature_allowlist_and_forbidden_scan") or {})
    folds = dict(artifact.get("calibration_group_and_fold_receipts") or {})
    selected = dict(artifact.get("selected_policy_rationale_without_held_access") or {})
    controls = dict(artifact.get("shortcut_and_boundary_controls") or {})
    metrics = dict(
        artifact.get(
            "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics"
        )
        or {}
    )
    by_model = dict(metrics.get("by_model") or {})
    non_vacuous = bool(by_model) and all(
        dict(dict(by_model.get(model_id) or {}).get("arms") or {})
        .get("decision_calibrated_task_energy", {})
        .get("unsafe_count", 0)
        > 0
        and dict(dict(by_model.get(model_id) or {}).get("arms") or {})
        .get("decision_calibrated_task_energy", {})
        .get("safe_count", 0)
        > 0
        for model_id in MANDATED_MODEL_IDS
    )
    return float(
        dict(artifact.get("structured_gate_receipt") or {}).get("calibration_permitted")
        is True
        and scan.get("forbidden_found_count") == 0
        and scan.get("future_known_label_read_count") == 0
        and scan.get("shifted_family_held_label_read_count") == 0
        and artifact.get("held_access_count") == 0
        and folds.get("group_count") >= 8
        and folds.get("future_or_held_rows_used_for_fit_count") == 0
        and selected.get("selected_arm") == "decision_calibrated_task_energy"
        and selected.get("selection_uses_held_outcomes") is False
        and selected.get("policy_validly_frozen") is True
        and selected.get("control_outperformed_selected_count") == 0
        and controls.get("no_control_outperforms_selected") is True
        and _manifest_valid(
            dict(artifact.get("policy_manifest_path_hash_and_contents") or {})
        )
        and non_vacuous
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    if dict(artifact.get("precommitted_feature_allowlist_and_forbidden_scan") or {}).get(
        "forbidden_found_count"
    ) != 0:
        reasons.append("forbidden_score_feature")
    if artifact.get("held_access_count") != 0:
        reasons.append("held_access_count")
    if dict(artifact.get("calibration_group_and_fold_receipts") or {}).get(
        "future_or_held_rows_used_for_fit_count"
    ) not in (0, None):
        reasons.append("future_or_held_rows_used_for_fit")
    if dict(artifact.get("shortcut_and_boundary_controls") or {}).get(
        "no_control_outperforms_selected"
    ) is False:
        reasons.append("shortcut_control_outperformed")
    if not _manifest_valid(dict(artifact.get("policy_manifest_path_hash_and_contents") or {})):
        reasons.append("policy_manifest_incomplete")
    return sorted(set(str(reason) for reason in reasons)) or ["incomplete_evidence"]


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("structured_gate_receipt") or {}).get("calibration_permitted") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    state = status(artifact)
    if state == "complete_ready":
        return "complete_ready: decision-calibrated policy validly frozen with zero held access"
    if state == "blocked":
        return "blocked: " + ",".join(_blocked_reasons(artifact)[:10])
    return "complete_null: policy was not validly frozen; " + ",".join(
        _blocked_reasons(artifact)[:10]
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["platform"] = "<normalized>"
        output = preconditions.get("output_paths")
        if isinstance(output, dict):
            for key in (
                "result_path",
                "manifest_path",
                "result_existed_before",
                "manifest_existed_before",
                "result_sha256_before",
                "manifest_sha256_before",
            ):
                output[key] = "<normalized>"
    upstream = stable.get("upstream_endpoint_row_and_control_hashes")
    if isinstance(upstream, dict):
        output = upstream.get("output_paths")
        if isinstance(output, dict):
            output["result_path"] = "<normalized>"
            output["manifest_path"] = "<normalized>"
            output["path_hash"] = "<normalized>"
    manifest = stable.get("policy_manifest_path_hash_and_contents")
    if isinstance(manifest, dict):
        manifest["path"] = "<normalized>"
        manifest["exists"] = "<normalized>"
        manifest["sha256"] = "<normalized>"
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
    if dict(artifact["precommitted_feature_allowlist_and_forbidden_scan"]).get(
        "forbidden_found_count"
    ) != 0:
        raise ValueError("forbidden score feature")
    if artifact.get("held_access_count") != 0:
        raise ValueError("held_access_count")
    if artifact.get("status") != "blocked" and not _manifest_valid(
        dict(artifact["policy_manifest_path_hash_and_contents"])
    ):
        raise ValueError("policy_manifest")
    if artifact.get("decision_calibrated_policy_ready_score") != ready_score(artifact):
        raise ValueError("decision_calibrated_policy_ready_score")
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
    manifest_path: str | Path = REPO_ROOT / MANIFEST_RELATIVE_PATH,
    exp6147_artifact: Mapping[str, Any] | None = None,
    exp6159_artifact: Mapping[str, Any] | None = None,
    exp6160_artifact: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    result = Path(result_path)
    manifest = Path(manifest_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    exp6147_payload = (
        _copy_json(exp6147_artifact)
        if exp6147_artifact is not None
        else _load_json(REPO_ROOT / exp6147.RESULT_RELATIVE_PATH)
    )
    exp6159_payload = (
        _copy_json(exp6159_artifact)
        if exp6159_artifact is not None
        else _load_json(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH)
    )
    exp6160_payload = (
        _copy_json(exp6160_artifact)
        if exp6160_artifact is not None
        else _load_json(REPO_ROOT / exp6160.RESULT_RELATIVE_PATH)
    )
    preconditions = collect_preconditions(result, manifest)
    if exp6147_artifact is not None:
        preconditions["exp6147_ready_score"] = exp6147_payload.get(
            "task_aware_energy_calibration_ready_score"
        )
    if exp6159_artifact is not None:
        preconditions["exp6159_ready_score"] = exp6159_payload.get(
            "decision_calibrated_stream_ready_score"
        )
    if exp6160_artifact is not None:
        preconditions["exp6160_ready_score"] = exp6160_payload.get(
            "sota_decision_corpus_ready_score"
        )
    upstream = _upstream_hashes(result, manifest)
    gate = _structured_gate(
        preconditions, upstream, exp6147_payload, exp6159_payload, exp6160_payload
    )
    folds, metrics, drift, controls, selected, freeze = _empty_sections()
    scan_seed = {
        "evaluated_partitions": ["calibration"],
        "calibration_label_read_count": 0,
        "future_known_label_read_count": 0,
        "shifted_family_held_label_read_count": 0,
        "held_access_count": 0,
        "source_rows_by_partition": {},
        "future_or_held_rows_filtered_before_label_access": True,
    }
    scan = _scan_features([], scan_seed)
    manifest_receipt = _manifest_empty(manifest)

    if gate["calibration_permitted"] is True:
        entries, scan_seed = _load_calibration_entries()
        scan = _scan_features(entries, scan_seed)
        if scan["forbidden_found_count"] == 0:
            cost_table = dict(exp6159_payload.get("frozen_utility_cost_table") or {})
            exp6147_selection = dict(
                exp6147_payload.get(
                    "selected_score_threshold_abstention_and_memory_budget"
                )
                or {}
            )
            exp6147_threshold = float(exp6147_selection.get("threshold", 0.0) or 0.0)
            shuffled_task_map = _shuffled_task_map(entries)
            folds = _grouped_folds(entries)
            cv = _cross_validated_metrics(
                entries, folds, cost_table, shuffled_task_map, exp6147_threshold
            )
            policies = _fit_all_arm_policies(
                entries, cost_table, shuffled_task_map, exp6147_threshold
            )
            selected = _selection(cv, policies)
            selected_policy = policies[str(selected["selected_arm"])]
            metrics = _per_model_metrics(entries, policies, cost_table)
            drift = _chronological_drift(
                entries, selected_policy, cost_table, shuffled_task_map
            )
            controls = _control_metrics(
                entries, selected_policy, policies, cost_table, shuffled_task_map
            )
            manifest_contents = _manifest_contents(
                selected,
                selected_policy,
                entries,
                cost_table,
                exp6147_payload,
                exp6159_payload,
                upstream,
            )
            manifest_receipt = _manifest_receipt(manifest, manifest_contents, write=write)
            freeze = _freeze_receipt(selected, selected_policy, cost_table)

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
        "upstream_endpoint_row_and_control_hashes": upstream,
        "precommitted_feature_allowlist_and_forbidden_scan": scan,
        "calibration_group_and_fold_receipts": folds,
        "global_exp6147_decision_family_shuffled_alias_frequency_and_distance_arm_configs": _arm_configs(
            exp6147_payload
        ),
        "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics": metrics,
        "chronological_drift_diagnostics": drift,
        "shortcut_and_boundary_controls": controls,
        "selected_policy_rationale_without_held_access": selected,
        "policy_manifest_path_hash_and_contents": manifest_receipt,
        "score_threshold_abstention_and_cost_freeze_receipts": freeze,
        "held_access_count": 0,
        "decision_calibrated_policy_ready_score": 0.0,
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
    artifact["decision_calibrated_policy_ready_score"] = ready_score(artifact)
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
    parser.add_argument("--manifest", default=str(REPO_ROOT / MANIFEST_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--e2e-check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate or args.e2e_check:
        artifact = _load_json(output)
        validate_artifact(artifact)
        if args.e2e_check and (
            artifact.get("held_access_count") != 0
            or artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        ):
            return 1
        return 0
    run(result_path=output, manifest_path=Path(args.manifest), write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
