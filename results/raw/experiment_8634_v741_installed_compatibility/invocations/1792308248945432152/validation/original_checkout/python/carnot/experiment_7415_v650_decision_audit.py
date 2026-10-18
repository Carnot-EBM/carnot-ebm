"""Audit V650 static and selected-feedback evidence as independent branches.

This reducer reads producer data as plain JSON. It does not import either
producer reducer. A failed branch remains visible and cannot suppress the
other branch's diagnosis. Spec refs: REQ-REPORT-7415 and
SCENARIO-REPORT-7415-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7415-decision-audit"
SCHEMA = "carnot.exp7415.v650.decision_audit.v1"
STATIC_SCHEMA = "carnot.exp7413.v650.source_calibration.v1"
ONLINE_SCHEMA = "carnot.exp7414.v650.selected_feedback.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7415_v650_decision_audit.json")
RAW_DIR = Path("results/raw/experiment_7415_v650_decision_audit")
MODULE_PATH = Path("python/carnot/experiment_7415_v650_decision_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7415_v650_decision_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7415_v650_decision_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
STATIC_PATH = Path("results/experiment_7413_v650_source_calibration.json")
ONLINE_PATH = Path("results/experiment_7414_v650_selected_feedback.json")
FEATURE_PATH = Path("results/raw/experiment_7412_v650_source_features/source_feature_rows.json")
CORPUS_MANIFEST_PATH = Path("results/raw/experiment_7410_v650_source_corpus/corpus_manifest.json")
EXPECTED_PRODUCER_HASHES = {
    STATIC_PATH: "sha256:a32ff491648d1459a2948280d5a662b8a30fb9bb3476d10331d597be150ff647",
    ONLINE_PATH: "sha256:12e57814d34d6ed4cd53a7ecafe56693700e4e94a98dada6cd4ac343c40f7091",
}
SOURCE_FEATURE_NAMES = (
    "numeric_novelty_with_context",
    "falsifiability_score",
    "normalized_number_token_overlap",
    "normalized_content_token_overlap",
    "max_answer_source_sentence_overlap",
    "missing_or_empty_source",
)
RESPONSE_FEATURE_NAMES = ("entity_uptake", "falsifiability_score")
THRESHOLD_PAIRS = tuple(
    (accept, reject) for accept in (0.01, 0.025, 0.05) for reject in (0.90, 0.95, 0.99)
)
LEAKAGE_ATTACKS = (
    "oracle_label_feature",
    "article_overlap",
    "test_label_exposure",
    "class_prior_shortcut",
    "unscored_omission",
    "temporal_future_label",
)
STATIC_PLANNED_UNITS = 30
ONLINE_PLANNED_UNITS = 200

V650_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "static_audit_complete_score",
    "online_audit_complete_score",
    "branch_rows",
    "leakage_attack_rows",
)


def binary_log_loss(label: int, probability: float) -> float:
    """Compute finite Bernoulli loss without trusting stored contributions."""

    if label not in {0, 1} or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("binary label and finite probability are required")
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object and keep malformed external bytes unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    """Compare numeric evidence strictly while preserving null values."""

    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and abs(float(left) - float(right)) <= tolerance
        )
    return left == right


def _precondition(
    branch: str,
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
) -> JsonDict:
    """Name an exact branch prerequisite and its observed operand."""

    return {
        "branch": branch,
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _historical_git_hash_exists(root: Path, relative: Path, expected: str) -> bool:
    """Find exact earlier tracked bytes when a mutable source file advanced."""

    command = ["git", "log", "--format=%H", "--", relative.as_posix()]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False
    for commit in result.stdout.splitlines()[:64]:
        shown = subprocess.run(
            ["git", "show", f"{commit}:{relative.as_posix()}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if shown.returncode == 0:
            import hashlib

            observed = "sha256:" + hashlib.sha256(shown.stdout).hexdigest()
            if observed == expected:
                return True
    return False


def _source_hash_checks(root: Path, branch: str, artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate every producer source against current or tracked historical bytes."""

    rows: list[JsonDict] = []
    for value in (artifact.get("source_artifact_hashes") or {}).values():
        if not isinstance(value, Mapping):
            rows.append(
                _precondition(
                    branch,
                    "source_hash_row",
                    branch,
                    "<invalid>",
                    "mapping",
                    True,
                    False,
                )
            )
            continue
        label = str(value.get("path") or "")
        expected = value.get("sha256")
        path = Path(label)
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        historical = False
        if observed != expected and label and not path.is_absolute() and isinstance(expected, str):
            historical = _historical_git_hash_exists(root, path, expected)
        rows.append(
            _precondition(
                branch,
                f"source_hash:{label}",
                branch,
                label,
                "sha256",
                expected,
                observed if not historical else expected,
                passed=observed == expected or historical,
            )
        )
        rows[-1]["authenticated_from"] = (
            "current_bytes"
            if observed == expected
            else ("tracked_historical_bytes" if historical else "unavailable")
        )
    return rows


def collect_preconditions(
    repo_root: Path,
    *,
    expected_hashes: Mapping[Path, str] | None = None,
    verify_sources: bool = True,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Authenticate Exp7413 and Exp7414 without creating a whole-task gate."""

    root = repo_root.resolve()
    expected_map = dict(EXPECTED_PRODUCER_HASHES if expected_hashes is None else expected_hashes)
    definitions = {
        "static": (
            STATIC_PATH,
            STATIC_SCHEMA,
            "exp7413-source-calibration",
            "calibration_capture_complete_score",
        ),
        "online": (
            ONLINE_PATH,
            ONLINE_SCHEMA,
            "exp7414-selected-feedback",
            "online_capture_complete_score",
        ),
    }
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    producers: dict[str, JsonDict] = {}
    for branch, (relative, schema, identity, score_field) in definitions.items():
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                branch,
                "producer_bytes",
                identity,
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
            )
        )
        producer = _load_object(path) if present else {}
        producers[branch] = producer
        if not producer:
            continue
        observed_hash = sha256_file(path)
        hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": observed_hash,
            "original_flagged_adversarial": producer.get("flagged_adversarial"),
        }
        if relative in expected_map:
            checks.append(
                _precondition(
                    branch,
                    "producer_sha256",
                    identity,
                    relative.as_posix(),
                    "sha256",
                    expected_map[relative],
                    observed_hash,
                )
            )
        fields = (
            ("schema", schema),
            ("experiment_id", identity),
            ("milestone", MILESTONE),
            (score_field, 1),
            ("flagged_adversarial", False),
        )
        for field, expected in fields:
            checks.append(
                _precondition(
                    branch,
                    f"producer_{field}",
                    identity,
                    relative.as_posix(),
                    field,
                    expected,
                    producer.get(field),
                )
            )
        verdict = producer.get("verdict_class")
        checks.append(
            _precondition(
                branch,
                "producer_verdict_eligible",
                identity,
                relative.as_posix(),
                "verdict_class",
                ["positive", "circular_positive", "null"],
                verdict,
                passed=verdict in {"positive", "circular_positive", "null"},
            )
        )
        validation = (producer.get("gate_check_summary") or {}).get("required_checks_passed")
        checks.append(
            _precondition(
                branch,
                "producer_required_validation",
                identity,
                relative.as_posix(),
                "gate_check_summary.required_checks_passed",
                True,
                validation,
            )
        )
        if verify_sources:
            checks.extend(_source_hash_checks(root, branch, producer))
    if expected_hashes is None:
        spec = root / SPEC_PATH
        spec_text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
        for branch in definitions:
            checks.append(
                _precondition(
                    branch,
                    "driving_requirement",
                    SPEC_PATH.as_posix(),
                    SPEC_PATH.as_posix(),
                    "REQ-*",
                    "REQ-REPORT-7415",
                    "REQ-REPORT-7415" if "REQ-REPORT-7415" in spec_text else None,
                )
            )
    return checks, hashes, producers


def _typed_action(probability: float, policy: Mapping[str, Any]) -> str:
    """Apply frozen thresholds and disable uncertified actions."""

    if probability <= float(policy.get("accept_threshold", -1.0)):
        return "accept" if policy.get("accept_enabled") is True else "escalate"
    if probability >= float(policy.get("reject_threshold", 2.0)):
        return "reject" if policy.get("reject_enabled") is True else "escalate"
    return "escalate"


def _proper_scores(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute Brier, log loss, action risk, and coverage by arm."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scored") is True and row.get("label") in {0, 1}:
            grouped[str(row.get("arm"))].append(row)
    output: JsonDict = {}
    for arm, selected in sorted(grouped.items()):
        brier = [(float(row["probability"]) - int(row["label"])) ** 2 for row in selected]
        losses = [binary_log_loss(int(row["label"]), float(row["probability"])) for row in selected]
        actions = [row for row in selected if row.get("decision") in {"accept", "reject"}]
        harmful = [
            row
            for row in actions
            if (row.get("decision") == "accept" and row.get("label") == 1)
            or (row.get("decision") == "reject" and row.get("label") == 0)
        ]
        output[arm] = {
            "rows": len(selected),
            "groups": len({str(row.get("group_id")) for row in selected}),
            "brier": float(np.mean(brier)),
            "log_loss": float(np.mean(losses)),
            "coverage": len(actions) / len(selected),
            "observed_action_risk": len(harmful) / len(actions) if actions else None,
        }
    return output


def _checkpoint_policies(
    producer: Mapping[str, Any], checkpoint_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]]
) -> tuple[dict[tuple[str, int, str], Mapping[str, Any]], list[str]]:
    """Read only numeric checkpoint values and bind each unit to one policy."""

    policies: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in producer.get("checkpoint_manifest") or []:
        if not isinstance(row, Mapping):
            errors.append("checkpoint_manifest_row_invalid")
            continue
        value = checkpoint_loader(row)
        policy = value.get("selected_policy") if isinstance(value, Mapping) else None
        if not isinstance(policy, Mapping):
            errors.append(f"checkpoint_policy_missing:{row.get('path')}")
            continue
        key = (str(row.get("arm")), int(row.get("seed") or 0), str(row.get("condition")))
        policies[key] = policy
    return policies, errors


def _recompute_threshold_decisions(
    rows: Sequence[Mapping[str, Any]], policies: Mapping[tuple[str, int, str], Mapping[str, Any]]
) -> JsonDict:
    """Reapply every frozen selected threshold to every official probability."""

    mismatches: list[str] = []
    checked = 0
    for row in rows:
        probability = row.get("probability")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            continue
        key = (str(row.get("arm")), int(row.get("seed") or 0), str(row.get("condition")))
        policy = policies.get(key)
        if not policy:
            continue
        expected = _typed_action(float(probability), policy)
        checked += 1
        if row.get("decision") != expected:
            mismatches.append(str(row.get("unit_id") or row.get("row_key")))
    return {"checked_rows": checked, "mismatches": mismatches[:100]}


def _static_leakage_rows(producer: Mapping[str, Any], proper: Mapping[str, Any]) -> list[JsonDict]:
    """Run label, split, source, temporal, prior, and omission attacks."""

    feature_names = {str(value).lower() for value in producer.get("feature_names") or []}
    prohibited = {"label", "gold", "teacher_score", "expected_verdict", "source_relation"}
    label_clean = not feature_names.intersection(prohibited) and not producer.get(
        "feature_label_fields_present"
    )
    partition_articles = producer.get("partition_articles") or {}
    seen: set[str] = set()
    overlap: set[str] = set()
    for partition in sorted(partition_articles):
        if partition == "excluded_test_overlap":
            continue
        current = {str(value) for value in partition_articles.get(partition) or []}
        overlap.update(seen.intersection(current))
        seen.update(current)
    fit_partitions = {str(value) for value in producer.get("fit_label_partitions") or []}
    final_exposed = "final_test" in fit_partitions
    temporal_exposed = bool({"online_stream", "final_test"}.intersection(fit_partitions))
    rows = producer.get("paired_metric_rows") or []
    primary = (
        "source_aware_6_4_1_gibbs"
        if "source_aware_6_4_1_gibbs" in proper
        else ("source" if "source" in proper else next(iter(proper), ""))
    )
    probabilities = [
        float(row["probability"])
        for row in rows
        if row.get("arm") == primary and row.get("scored") is True
    ]
    prior_shortcut = len({round(value, 14) for value in probabilities}) <= 1
    unit_totals: dict[tuple[str, int, str], int] = defaultdict(int)
    unscored = 0
    for row in rows:
        key = (str(row.get("arm")), int(row.get("seed") or 0), str(row.get("condition")))
        unit_totals[key] += 1
        unscored += int(row.get("scored") is False)
    expected_per_unit = int(producer.get("official_row_count_per_unit") or 0)
    omission = not unit_totals or (
        expected_per_unit > 0 and any(count != expected_per_unit for count in unit_totals.values())
    )
    details = {
        "oracle_label_feature": (label_clean, sorted(feature_names.intersection(prohibited))),
        "article_overlap": (not overlap, sorted(overlap)[:20]),
        "test_label_exposure": (not final_exposed, sorted(fit_partitions)),
        "class_prior_shortcut": (not prior_shortcut, len(set(probabilities))),
        "unscored_omission": (not omission, unscored),
        "temporal_future_label": (not temporal_exposed, sorted(fit_partitions)),
    }
    return [
        {
            "branch": "static",
            "attack": attack,
            "passed": bool(details[attack][0]),
            "observed": details[attack][1],
            "label_authority": "machine_annotation",
        }
        for attack in LEAKAGE_ATTACKS
    ]


def _static_checkpoint_loader(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load a declared checkpoint and reject an inaccessible byte source."""

    if isinstance(row.get("selected_policy"), Mapping):
        return row
    path = Path(str(row.get("path") or ""))
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
        return {}
    return _load_object(resolved)


def _bootstrap_summary(producer: Mapping[str, Any], proper: Mapping[str, Any]) -> JsonDict:
    """Recompute paired group draws and simultaneous intervals from raw rows."""

    stored = (producer.get("paired_group_bootstrap") or {}).get("contrasts") or {}
    if not stored:
        return {"checked": 0, "mismatches": []}
    primary = "source_aware_6_4_1_gibbs"
    rows = [row for row in producer.get("paired_metric_rows") or [] if row.get("scored") is True]
    cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cell[(str(row["arm"]), str(row["group_id"]), str(row["row_key"]))].append(row)
    groups = sorted({key[1] for key in cell})
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for arm in proper:
        metrics: dict[str, list[float]] = {"brier": [], "log_loss": [], "coverage": []}
        for group in groups:
            group_cells = [
                values
                for (cell_arm, cell_group, _key), values in cell.items()
                if cell_arm == arm and cell_group == group
            ]
            if not group_cells:
                continue
            metrics["brier"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["brier_contribution"]) for row in values])
                            for values in group_cells
                        ]
                    )
                )
            )
            metrics["log_loss"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["log_loss_contribution"]) for row in values])
                            for values in group_cells
                        ]
                    )
                )
            )
            metrics["coverage"].append(
                float(
                    np.mean(
                        [
                            np.mean([row.get("decision") != "escalate" for row in values])
                            for values in group_cells
                        ]
                    )
                )
            )
        vectors[arm] = {name: np.asarray(values) for name, values in metrics.items()}
    draws = int((producer.get("paired_group_bootstrap") or {}).get("draws") or 0)
    seed = int((producer.get("paired_group_bootstrap") or {}).get("seed") or 0)
    if draws <= 0 or not groups:
        return {"checked": 0, "mismatches": ["bootstrap_protocol_missing"]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(draws, len(groups)))
    raw: dict[str, dict[str, np.ndarray]] = {}
    for baseline in stored:
        if baseline not in vectors or primary not in vectors:
            continue
        raw[baseline] = {}
        for metric in ("brier", "log_loss", "coverage"):
            delta = vectors[primary][metric] - vectors[baseline][metric]
            raw[baseline][metric] = np.mean(delta[indices], axis=1)
    centered = np.column_stack(
        [
            raw[baseline]["brier"] - np.mean(raw[baseline]["brier"])
            for baseline in stored
            if baseline in raw
        ]
    )
    upper = float(np.quantile(np.max(centered, axis=1), 0.95))
    lower = float(np.quantile(np.min(centered, axis=1), 0.05))
    mismatches: list[str] = []
    checked = 0
    for baseline, evidence in stored.items():
        if baseline not in raw:
            mismatches.append(f"missing_vector:{baseline}")
            continue
        observed_brier = float(np.mean(vectors[primary]["brier"] - vectors[baseline]["brier"]))
        recomputed = {
            "brier_delta": {
                "mean": observed_brier,
                "simultaneous_ci95": [observed_brier + lower, observed_brier + upper],
                "marginal_ci95": [
                    float(np.quantile(raw[baseline]["brier"], 0.025)),
                    float(np.quantile(raw[baseline]["brier"], 0.975)),
                ],
            },
            "log_loss_delta": {
                "mean": float(np.mean(raw[baseline]["log_loss"])),
                "ci95": [
                    float(np.quantile(raw[baseline]["log_loss"], 0.025)),
                    float(np.quantile(raw[baseline]["log_loss"], 0.975)),
                ],
            },
            "coverage_delta": {
                "mean": float(np.mean(raw[baseline]["coverage"])),
                "ci95": [
                    float(np.quantile(raw[baseline]["coverage"], 0.025)),
                    float(np.quantile(raw[baseline]["coverage"], 0.975)),
                ],
            },
        }
        checked += 1
        if canonical_hash(recomputed) != canonical_hash(evidence):
            mismatches.append(f"contrast:{baseline}")
    stored_upper = (producer.get("paired_group_bootstrap") or {}).get("upper_critical_value")
    checked += 1
    if not _close(upper, stored_upper):
        mismatches.append("upper_critical_value")
    return {
        "checked": checked,
        "draws": draws,
        "seed": seed,
        "effective_groups": len(groups),
        "upper_critical_value": upper,
        "mismatches": mismatches,
    }


def audit_static_branch(
    producer: Mapping[str, Any],
    *,
    precondition_rows: Sequence[Mapping[str, Any]] = (),
    checkpoint_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Reduce Exp7413 proper scores, decisions, ablations, and leakage attacks."""

    if not producer:
        return blocked_branch("static", precondition_rows)
    errors: list[str] = []
    identity = (
        producer.get("schema") == STATIC_SCHEMA
        and producer.get("experiment_id") == "exp7413-source-calibration"
        and producer.get("milestone") == MILESTONE
        and producer.get("calibration_capture_complete_score") == 1
        and producer.get("flagged_adversarial") is False
        and producer.get("verdict_class") in {"positive", "circular_positive", "null"}
        and (producer.get("gate_check_summary") or {}).get("required_checks_passed") is True
    )
    if not identity:
        errors.append("static_producer_ineligible")
    if any(row.get("passed") is not True for row in precondition_rows):
        errors.append("static_precondition_failed")
    rows = producer.get("paired_metric_rows") or []
    proper = _proper_scores(rows)
    if not proper:
        errors.append("static_scored_rows_missing")
    for arm, stored in ((producer.get("calibration_metrics") or {}).get("by_arm") or {}).items():
        recomputed = proper.get(arm) or {}
        for metric in ("brier", "log_loss", "coverage"):
            if not _close(recomputed.get(metric), stored.get(metric)):
                errors.append(f"static_metric_mismatch:{arm}:{metric}")
    loader = checkpoint_loader or _static_checkpoint_loader
    policies, policy_errors = _checkpoint_policies(producer, loader)
    errors.extend(policy_errors)
    thresholds = _recompute_threshold_decisions(rows, policies)
    if policies and thresholds["mismatches"]:
        errors.append("static_threshold_decision_mismatch")
    bootstrap = _bootstrap_summary(producer, proper)
    if bootstrap["mismatches"]:
        errors.append("static_bootstrap_contrast_mismatch")
    ablation_scores = _proper_scores(producer.get("source_ablation_rows") or [])
    primary = proper.get("source_aware_6_4_1_gibbs") or proper.get("source") or {}
    ablation_by_condition: JsonDict = {}
    for condition in sorted(
        {str(row.get("condition")) for row in producer.get("source_ablation_rows") or []}
    ):
        selected = [
            row
            for row in producer.get("source_ablation_rows") or []
            if row.get("condition") == condition and row.get("scored") is True
        ]
        if selected:
            score = float(np.mean([float(row["brier_contribution"]) for row in selected]))
            ablation_by_condition[condition] = {
                "brier": score,
                "delta_vs_full": score - float(primary.get("brier", score)),
            }
    leakage = _static_leakage_rows(producer, proper)
    if any(row["passed"] is not True for row in leakage):
        errors.append("static_leakage_attack_failed")
    unit_rows = [{"branch": "static", **deepcopy(dict(row))} for row in producer.get("rows") or []]
    complete = bool(unit_rows) and all(row.get("status") == "completed" for row in unit_rows)
    if not complete:
        errors.append("static_unit_accounting_incomplete")
    valid = not errors
    return {
        "branch": "static",
        "upstream": "exp7413-source-calibration",
        "available": True,
        "valid": valid,
        "complete": complete and valid,
        "value": bool(producer.get("calibration_value_score") == 1 and valid),
        "producer_verdict_class": producer.get("verdict_class"),
        "label_authority": "machine_annotation",
        "errors": list(dict.fromkeys(errors)),
        "proper_scores": proper,
        "threshold_recomputation": thresholds,
        "simultaneous_risk_bounds": [
            {"arm": arm, "seed": seed, "condition": condition, **deepcopy(dict(policy))}
            for (arm, seed, condition), policy in sorted(policies.items())
        ],
        "bootstrap_recomputation": bootstrap,
        "source_ablation_recomputation": ablation_by_condition,
        "source_ablation_arm_scores": ablation_scores,
        "unscored_examples": {
            "expected": sum(
                int(producer.get("official_row_count_per_unit") or 0)
                for _row in producer.get("rows") or []
            )
            - sum(row.get("scored") is True for row in rows),
            "observed": sum(row.get("scored") is False for row in rows),
        },
        "leakage_attack_rows": leakage,
        "unit_rows": unit_rows,
        "gate_check_summary": _branch_gate_summary("static", precondition_rows, errors),
    }


def _unit_key(row: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    """Name one online arm, seed, order, regime, and delay ledger."""

    return (
        str(row.get("arm")),
        int(row.get("seed") or 0),
        str(row.get("ordering")),
        str(row.get("feedback_regime")),
        int(row.get("delay") or 0),
    )


def _online_event_errors(events: Sequence[Mapping[str, Any]]) -> tuple[list[str], JsonDict]:
    """Check shared masks, event order, delayed commits, and final state hashes."""

    errors: list[str] = []
    by_unit: dict[tuple[str, int, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_condition_event: dict[tuple[int, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in events:
        by_unit[_unit_key(row)].append(row)
        condition = (
            int(row.get("seed") or 0),
            str(row.get("ordering")),
            str(row.get("feedback_regime")),
            int(row.get("delay") or 0),
            str(row.get("observation_id")),
        )
        by_condition_event[condition].append(row)
    for key, matched in by_condition_event.items():
        if len({row.get("registered_feedback_mask") for row in matched}) != 1:
            errors.append(f"shared_feedback_mask_mismatch:{key}")
    admitted: set[tuple[tuple[str, int, str, str, int], str]] = set()
    final_hashes: JsonDict = {}
    for key, selected in by_unit.items():
        ordered = sorted(selected, key=lambda row: int(row.get("prediction_index") or 0))
        indices = [int(row.get("prediction_index") or 0) for row in ordered]
        if indices != list(range(len(indices))):
            errors.append(f"event_order_mismatch:{key}")
        for row in ordered:
            index = int(row.get("prediction_index") or 0)
            delay = int(row.get("delay") or 0)
            available = int(row.get("available_at") or 0)
            reveal = int(row.get("reveal_visible_at") or 0)
            if available < index + delay or reveal < available:
                errors.append(f"early_feedback:{key}:{index}")
            if (
                row.get("prediction_before_feedback") is not True
                or row.get("commit_after_prediction") is not True
                or row.get("prediction_persisted") is not True
            ):
                errors.append(f"prediction_commit_order:{key}:{index}")
            if not all(
                isinstance(row.get(field), str) and bool(row.get(field))
                for field in ("state_hash_at_prediction", "prior_state_hash", "new_state_hash")
            ):
                errors.append(f"state_hash_missing:{key}:{index}")
            if delay == 1 and row.get("prior_state_hash") != row.get("state_hash_at_prediction"):
                errors.append(f"delay_one_state_chain:{key}:{index}")
            if row.get("update_admitted") is True:
                identity = (key, str(row.get("observation_id")))
                if identity in admitted:
                    errors.append(f"duplicate_update:{key}:{row.get('observation_id')}")
                admitted.add(identity)
                if (
                    row.get("registered_feedback_mask") is not True
                    or row.get("label") not in {0, 1}
                    or row.get("feedback_status") != "committed"
                    or row.get("new_state_hash") == row.get("prior_state_hash")
                ):
                    errors.append(f"unauthorized_update:{key}:{index}")
            elif row.get("new_state_hash") != row.get("prior_state_hash"):
                errors.append(f"state_changed_without_update:{key}:{index}")
        if ordered:
            final_hashes[":".join(map(str, key))] = ordered[-1].get("new_state_hash")
    return list(dict.fromkeys(errors)), {
        "unit_count": len(by_unit),
        "event_count": len(events),
        "admitted_update_count": len(admitted),
        "final_state_hashes": final_hashes,
        "shared_mask_verified": not any(
            error.startswith("shared_feedback_mask") for error in errors
        ),
    }


def _online_reports(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute online proper scores for each registered condition."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in events:
        if row.get("label") in {0, 1}:
            grouped[
                (
                    str(row.get("ordering")),
                    str(row.get("feedback_regime")),
                    int(row.get("delay") or 0),
                    str(row.get("arm")),
                )
            ].append(row)
    output: list[JsonDict] = []
    for (ordering, regime, delay, arm), selected in sorted(grouped.items()):
        actions = [row for row in selected if row.get("decision") != "escalate"]
        harmful = [
            row
            for row in actions
            if (row.get("decision") == "accept" and row.get("label") == 1)
            or (row.get("decision") == "reject" and row.get("label") == 0)
        ]
        output.append(
            {
                "ordering": ordering,
                "feedback_regime": regime,
                "delay": delay,
                "arm": arm,
                "scored_rows": len(selected),
                "brier": float(
                    np.mean(
                        [(float(row["probability"]) - int(row["label"])) ** 2 for row in selected]
                    )
                ),
                "log_loss": float(
                    np.mean(
                        [
                            binary_log_loss(int(row["label"]), float(row["probability"]))
                            for row in selected
                        ]
                    )
                ),
                "coverage": len(actions) / len(selected),
                "observed_action_risk": len(harmful) / len(actions) if actions else 0.0,
            }
        )
    return output


def _online_interval_estimates(events: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    """Recompute paired point estimates before any moving-block resampling."""

    adaptive = "projected_adaptive_affine_gibbs"
    controls = (
        "frozen_calibrated_gibbs",
        "online_l2_logistic",
        "recent_frequency_beta_1_1_window_128",
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in events:
        if (
            row.get("ordering") == "hash_order"
            and row.get("delay") == 1
            and row.get("arm") in {adaptive, *controls}
            and row.get("brier_contribution") is not None
        ):
            grouped[
                (
                    str(row.get("feedback_regime")),
                    str(row.get("observation_id")),
                    str(row.get("arm")),
                )
            ].append(float(row["brier_contribution"]))
    output: dict[tuple[str, str], float] = {}
    regimes = sorted({key[0] for key in grouped})
    for regime in regimes:
        identities = sorted({key[1] for key in grouped if key[0] == regime})
        for control in controls:
            deltas = [
                float(np.mean(grouped[(regime, identity, adaptive)]))
                - float(np.mean(grouped[(regime, identity, control)]))
                for identity in identities
                if (regime, identity, adaptive) in grouped
                and (regime, identity, control) in grouped
            ]
            if deltas:
                output[(regime, control)] = float(np.mean(deltas))
    return output


def audit_online_branch(
    producer: Mapping[str, Any], *, precondition_rows: Sequence[Mapping[str, Any]] = ()
) -> JsonDict:
    """Reduce Exp7414 masks, order, updates, states, controls, and intervals."""

    if not producer:
        return blocked_branch("online", precondition_rows)
    errors: list[str] = []
    identity = (
        producer.get("schema") == ONLINE_SCHEMA
        and producer.get("experiment_id") == "exp7414-selected-feedback"
        and producer.get("milestone") == MILESTONE
        and producer.get("online_capture_complete_score") == 1
        and producer.get("flagged_adversarial") is False
        and producer.get("verdict_class") in {"positive", "circular_positive", "null"}
        and (producer.get("gate_check_summary") or {}).get("required_checks_passed") is True
    )
    if not identity:
        errors.append("online_producer_ineligible")
    if any(row.get("passed") is not True for row in precondition_rows):
        errors.append("online_precondition_failed")
    events = producer.get("feedback_event_rows") or []
    event_errors, event_summary = _online_event_errors(events)
    errors.extend(event_errors)
    explicit_final = producer.get("final_state_hashes")
    if explicit_final is not None and explicit_final != event_summary["final_state_hashes"]:
        errors.append("online_final_state_mismatch")
    if not event_summary["final_state_hashes"]:
        errors.append("online_final_state_missing")
    controls = producer.get("analytic_controls") or {}
    required_controls = ("no_feedback_equality", "erased_update_replay")
    for name in required_controls:
        if (controls.get(name) or {}).get("passed") is not True:
            errors.append(f"online_control_failed:{name}")
    revocations = producer.get("revocation_rows") or []
    revocation_ok = bool(revocations) and all(
        row.get("trusted_journal_replayed") is True and bool(row.get("state_hash_after"))
        for row in revocations
    )
    if not revocation_ok:
        errors.append("online_revocation_reconstruction_missing")
    reports = _online_reports(events)
    stored_reports = producer.get("condition_reports") or []
    stored_index = {
        (
            str(row.get("ordering")),
            str(row.get("feedback_regime")),
            int(row.get("delay") or 0),
            str(row.get("arm")),
        ): row
        for row in stored_reports
    }
    for row in reports:
        key = (row["ordering"], row["feedback_regime"], row["delay"], row["arm"])
        stored = stored_index.get(key)
        if stored:
            for metric in ("brier", "log_loss", "coverage", "observed_action_risk"):
                if not _close(row[metric], stored.get(metric)):
                    errors.append(f"online_metric_mismatch:{key}:{metric}")
    interval_points = _online_interval_estimates(events)
    interval_mismatches: list[str] = []
    for row in producer.get("paired_moving_block_intervals") or []:
        key = (str(row.get("feedback_regime")), str(row.get("control_arm")))
        observed = interval_points.get(key)
        if observed is None or not _close(observed, row.get("estimate")):
            interval_mismatches.append(":".join(key))
    if interval_mismatches:
        errors.append("online_interval_estimate_mismatch")
    unit_rows = [{"branch": "online", **deepcopy(dict(row))} for row in producer.get("rows") or []]
    complete = bool(unit_rows) and all(row.get("status") == "completed" for row in unit_rows)
    if not complete:
        errors.append("online_unit_accounting_incomplete")
    valid = not errors
    return {
        "branch": "online",
        "upstream": "exp7414-selected-feedback",
        "available": True,
        "valid": valid,
        "complete": complete and valid,
        "value": bool(producer.get("online_value_score") == 1 and valid),
        "producer_verdict_class": producer.get("verdict_class"),
        "label_authority": "machine_annotation",
        "errors": list(dict.fromkeys(errors)),
        "event_count": event_summary["event_count"],
        "admitted_update_count": event_summary["admitted_update_count"],
        "shared_mask_verified": event_summary["shared_mask_verified"],
        "final_state_hashes": event_summary["final_state_hashes"],
        "no_feedback_control_verified": (controls.get("no_feedback_equality") or {}).get("passed")
        is True,
        "erased_update_control_verified": (controls.get("erased_update_replay") or {}).get("passed")
        is True,
        "revocation_reconstruction_verified": revocation_ok,
        "condition_recomputation": reports,
        "interval_recomputation": {
            "point_estimates": {
                f"{key[0]}:{key[1]}": value for key, value in interval_points.items()
            },
            "mismatches": interval_mismatches,
        },
        "unit_rows": unit_rows,
        "gate_check_summary": _branch_gate_summary("online", precondition_rows, errors),
    }


def _branch_gate_summary(
    branch: str, preconditions: Sequence[Mapping[str, Any]], errors: Sequence[str]
) -> JsonDict:
    """Preserve the first exact failed operand and every branch audit error."""

    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    return {
        "branch": branch,
        "blocked_upstream": failed.get("upstream") if failed else None,
        "blocked_path": failed.get("path") if failed else None,
        "blocked_check": failed.get("check") if failed else None,
        "blocked_field": failed.get("field") if failed else None,
        "blocked_expected": failed.get("expected") if failed else None,
        "blocked_observed": failed.get("observed") if failed else None,
        "audit_errors": list(errors),
    }


def blocked_branch(branch: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Represent one unavailable external branch without hiding its peer."""

    planned = STATIC_PLANNED_UNITS if branch == "static" else ONLINE_PLANNED_UNITS
    return {
        "branch": branch,
        "upstream": "exp7413-source-calibration"
        if branch == "static"
        else "exp7414-selected-feedback",
        "available": False,
        "valid": False,
        "complete": False,
        "value": False,
        "producer_verdict_class": None,
        "label_authority": "machine_annotation",
        "errors": ["external_branch_unavailable"],
        "unit_rows": [
            {
                "branch": branch,
                "comparative_unit": f"{branch}:external_producer",
                "status": "unstarted",
                "planned_units": planned,
            }
        ],
        "leakage_attack_rows": [],
        "gate_check_summary": _branch_gate_summary(
            branch, preconditions, ["external_branch_unavailable"]
        ),
    }


def classify_terminal(
    branches: Sequence[Mapping[str, Any]],
    *,
    required_validation_passed: bool,
    own_work_complete: bool,
) -> JsonDict:
    """Keep unavailable, invalid, unfinished, and eligible-null states distinct."""

    by_name = {str(row.get("branch")): row for row in branches}
    static_score = int(bool((by_name.get("static") or {}).get("complete")))
    online_score = int(bool((by_name.get("online") or {}).get("complete")))
    if any(row.get("available") is True and row.get("valid") is not True for row in branches):
        verdict_class = "disqualified"
        honest = "complete_disqualified_invalid_branch_evidence"
    elif not required_validation_passed:
        verdict_class = "disqualified"
        honest = "complete_disqualified_required_validation"
    elif any(row.get("available") is not True for row in branches):
        verdict_class = "blocked"
        honest = "blocked_external_decision_branch_unavailable"
    elif not own_work_complete:
        verdict_class = "partial"
        honest = "complete_partial_owned_audit_work_unfinished"
    elif all(row.get("value") is True for row in branches):
        verdict_class = "positive"
        honest = "complete_positive_both_registered_decision_values_confirmed"
    else:
        verdict_class = "null"
        honest = "complete_null_independent_decision_audits_no_joint_registered_value"
    return {
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "static_audit_complete_score": static_score,
        "online_audit_complete_score": online_score,
    }


def _gate(check: str, category: str, expected: Any, observed: Any, principle: str) -> JsonDict:
    """Publish an explicit operator and keep scientific value separate."""

    return {
        "check": check,
        "category": category,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain ordinary artifact fields without wrapping their values."""

    specific = {
        "schema": "Versioned ordinary top-level fields identify this audit contract.",
        "run_date": "The scheduled date is 20260919; UTC timestamps record actual boundaries.",
        "preconditions_checked": "Exact producer paths, hashes, flags, validation, and sources precede reduction.",
        "MODEL_SPECS": "No current LLM call requires an empty model list.",
        "model_invoked": "This is false because the current audit performs aggregation only.",
        "invocation_counts": "Current owned model loads and generations remain zero.",
        "inference_substrate": "The substrate is a string; CPU and software details stay separate.",
        "inference_substrate_class": "The current work class is aggregation.",
        "execution_venue": "The closed current venue is host.",
        "duration_s": "Measured monotonic audit time excludes producer and validation time descriptions.",
        "phase_spans": "Real phase boundaries keep completed unit counts and heartbeat checkpoints.",
        "random_seed": "The audit preserves producer fit and resampling seeds without fitting new models.",
        "reproducibility_checksum": "The checksum binds sources, audit rows, branches, gates, and classification.",
        "source_artifact_hashes": "Exact producer byte hashes preserve original adversarial flags.",
        "rows": "Every producer comparative unit remains explicit, including unavailable units.",
        "sample_size_budget": "Planned and terminal unit counts remain separate from value.",
        "acceptance_gate_results": "Validity, completion, validation, and benefit use explicit operands.",
        "gate_check_summary": "Each branch keeps exact blocked fields and audit errors.",
        "verifier_is_oracle": "Machine annotations are calibration targets, not semantic truth authority.",
        "honest_verdict": "Completed findings use complete_; unchanged external absence uses blocked_.",
        "verdict_class": "The class is one closed project enum value.",
        "flagged_adversarial": "Critical evidence defects cannot supply readiness.",
        "validation_receipts": "Each scoped command keeps argv, environment, exit, duration, and log hash.",
        "promotion_score": "No automatic rollout, publication, or generator update occurs.",
        "static_audit_complete_score": "Static completion is independent of online availability and value.",
        "online_audit_complete_score": "Online completion requires exact raw events and final states.",
        "branch_rows": "Availability, validity, completion, and value remain separate for both branches.",
        "leakage_attack_rows": "Each label, split, source, temporal, prior, and omission mutation is explicit.",
    }
    return {
        field: specific.get(field, "This ordinary field preserves measured audit evidence.")
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding the checksum slot and wall-clock receipts."""

    keys = (
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "honest_verdict",
        "verdict_class",
        "static_audit_complete_score",
        "online_audit_complete_score",
        "branch_rows",
        "leakage_attack_rows",
        "static_reduction",
        "online_reduction",
    )
    return canonical_hash({key: value.get(key) for key in keys})


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, require_terminal: bool) -> bool:
    """Require the frozen affected set and, for final records, all terminal readers."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True
        and row.get("exit_code") == 0
        and row.get("timed_out") is not True
    }
    required = set(validation_scope.REQUIRED_CHECK_NAMES)
    if require_terminal:
        required.update(
            {
                "cold_artifact_replay",
                "independent_branch_recompute",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            }
        )
    return required.issubset(passed)


def _build_artifact(
    static: Mapping[str, Any],
    online: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    require_terminal: bool,
    own_work_complete: bool = True,
) -> JsonDict:
    """Build one schema-complete candidate from two independent reductions."""

    branches = [static, online]
    validation_passed = _validation_passed(validation_receipts, require_terminal=require_terminal)
    classification = classify_terminal(
        branches,
        required_validation_passed=validation_passed,
        own_work_complete=own_work_complete,
    )
    branch_rows = [
        {
            "branch": row.get("branch"),
            "upstream": row.get("upstream"),
            "available": row.get("available") is True,
            "valid": row.get("valid") is True,
            "complete": row.get("complete") is True,
            "value": row.get("value") is True,
            "producer_verdict_class": row.get("producer_verdict_class"),
            "label_authority": row.get("label_authority"),
            "errors": deepcopy(list(row.get("errors") or [])),
        }
        for row in branches
    ]
    rows = [deepcopy(dict(unit)) for branch in branches for unit in branch.get("unit_rows") or []]
    completed = sum(row.get("status") == "completed" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    censored = sum(row.get("status") == "censored" for row in rows)
    planned = sum(
        len(row.get("unit_rows") or [])
        if row.get("available")
        else (STATIC_PLANNED_UNITS if row.get("branch") == "static" else ONLINE_PLANNED_UNITS)
        for row in branches
    )
    gates = []
    for row in branch_rows:
        branch = str(row["branch"])
        gates.extend(
            (
                _gate(
                    f"{branch}_available",
                    "availability",
                    True,
                    row["available"],
                    "External availability does not gate its peer branch.",
                ),
                _gate(
                    f"{branch}_valid",
                    "validity",
                    True,
                    row["valid"],
                    "Only authenticated raw evidence can supply science.",
                ),
                _gate(
                    f"{branch}_complete",
                    "completion",
                    True,
                    row["complete"],
                    "Complete accounting is independent of benefit.",
                ),
                _gate(
                    f"{branch}_value",
                    "scientific_benefit",
                    True,
                    row["value"],
                    "An eligible null remains a null.",
                ),
            )
        )
    gates.append(
        _gate(
            "affected_and_terminal_validation",
            "validation",
            True,
            validation_passed,
            "Every frozen scoped and requested terminal command must pass.",
        )
    )
    gate_summaries = [deepcopy(dict(row.get("gate_check_summary") or {})) for row in branches]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked"
        if classification["verdict_class"] == "blocked"
        else ("disqualified" if classification["verdict_class"] == "disqualified" else "complete"),
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **deepcopy(dict(current_receipt)),
        "random_seed": {
            "static_training": [65001, 65002, 65003, 65004, 65005],
            "static_bootstrap": 6501307,
            "online_training": [65001, 65002, 65003, 65004, 65005],
            "online_bootstrap": 6501407,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": planned,
            "attempted": completed + failed + censored,
            "completed": completed,
            "failed": failed,
            "censored": censored,
            "unstarted": planned - completed - failed - censored,
            "independent_groups": {
                "static": max(
                    (
                        value.get("groups", 0)
                        for value in (static.get("proper_scores") or {}).values()
                    ),
                    default=0,
                ),
                "online": max(
                    (
                        row.get("scored_rows", 0)
                        for row in online.get("condition_recomputation") or []
                    ),
                    default=0,
                ),
            },
            "stop_rule": "audit every available registered producer unit; never replace an unavailable external branch",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {"branches": gate_summaries},
        "verifier_is_oracle": False,
        **classification,
        "flagged_adversarial": any(
            row.get("available") and not row.get("valid") for row in branch_rows
        ),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "promotion_score": 0,
        "branch_rows": branch_rows,
        "leakage_attack_rows": deepcopy(list(static.get("leakage_attack_rows") or [])),
        "static_reduction": {
            key: deepcopy(value)
            for key, value in static.items()
            if key not in {"unit_rows", "leakage_attack_rows"}
        },
        "online_reduction": {
            key: deepcopy(value) for key, value in online.items() if key != "unit_rows"
        },
        "label_calibration_scope": "machine_annotation_agreement_not_semantic_truth",
        "historical_and_scripted_receipts_are_current_calls": False,
    }
    artifact["field_principles"] = _field_principles(
        (*tuple(artifact), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(static: Mapping[str, Any], online: Mapping[str, Any]) -> JsonDict:
    """Build a compact complete artifact for mutation and cold-reader tests."""

    names = [
        *validation_scope.REQUIRED_CHECK_NAMES,
        "cold_artifact_replay",
        "independent_branch_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False} for name in names
    ]
    receipt = build_current_work_receipt(
        run_id="exp7415-fixture",
        owner_pid=0,
        events=[],
        inference_substrate="host_cpu_json_aggregation",
        inference_substrate_details={"device": "cpu", "software": "python_numpy"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000,
        phase_spans=[
            {
                "phase": "fixture",
                "start_s": 0.0,
                "end_s": 0.001,
                "duration_s": 0.001,
                "completed_units": 7,
            }
        ],
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    return _build_artifact(
        static,
        online,
        preconditions=[],
        source_hashes={},
        validation_receipts=receipts,
        current_receipt=receipt,
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:00.001000+00:00",
        require_terminal=True,
    )


def validate_artifact(
    value: Mapping[str, Any] | Any,
    *,
    root: Path = REPO_ROOT,
    verify_source_bytes: bool = True,
    require_terminal: bool = True,
) -> list[str]:
    """Cold-check identity, declarations, branches, classification, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
        or value.get("run_date") != RUN_DATE
    ):
        errors.append("identity_mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
        or not isinstance(value.get("inference_substrate"), str)
    ):
        errors.append("model_declaration_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    branches = value.get("branch_rows") or []
    classification = classify_terminal(
        branches,
        required_validation_passed=_validation_passed(
            value.get("validation_receipts") or [], require_terminal=require_terminal
        ),
        own_work_complete=value.get("verdict_class") != "partial",
    )
    if (
        value.get("static_audit_complete_score") != classification["static_audit_complete_score"]
        or value.get("online_audit_complete_score") != classification["online_audit_complete_score"]
    ):
        errors.append("branch_score_mismatch")
    if (
        value.get("verdict_class") != classification["verdict_class"]
        or value.get("honest_verdict") != classification["honest_verdict"]
    ):
        errors.append("terminal_classification_mismatch")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_mismatch")
    if value.get("verifier_is_oracle") is not False:
        errors.append("oracle_declaration_mismatch")
    if {row.get("attack") for row in value.get("leakage_attack_rows") or []} != set(
        LEAKAGE_ATTACKS
    ):
        errors.append("leakage_attack_rows_incomplete")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_FIELDS).issubset(principles):
        errors.append("field_principles_incomplete")
    if verify_source_bytes:
        for row in (value.get("source_artifact_hashes") or {}).values():
            if not isinstance(row, Mapping):
                errors.append("source_hash_row_invalid")
                continue
            path = Path(str(row.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path, *, verify_source_bytes: bool = True, require_terminal: bool = True
) -> list[str]:
    """Reload one candidate and run its strict field-level checks."""

    value = _load_object(path)
    return (
        validate_artifact(
            value,
            verify_source_bytes=verify_source_bytes,
            require_terminal=require_terminal,
        )
        if value
        else ["artifact_unreadable_or_not_object"]
    )


def build_validation_commands(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Freeze the exact Exp7358 affected command set for this module."""

    return build_command_plan(repo_root, V650_MANIFEST, private_root)


def _utc_now() -> str:  # pragma: no cover - actual run boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7415] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": _utc_now(),
    }


def _augment_static_context(root: Path, producer: JsonDict) -> JsonDict:  # pragma: no cover
    """Add split-only attack context without changing producer evidence."""

    value = deepcopy(producer)
    features = _load_object(root / FEATURE_PATH)
    records = features.get("records") or []
    value["feature_names"] = list(SOURCE_FEATURE_NAMES)
    value["feature_label_fields_present"] = features.get("label_fields_present") or []
    partitions: dict[str, list[str]] = defaultdict(list)
    for row in records:
        partitions[str(row.get("partition"))].append(str(row.get("group_id")))
    value["partition_articles"] = {key: sorted(set(rows)) for key, rows in partitions.items()}
    value["fit_label_partitions"] = ["train", "probability_calibration", "policy_calibration"]
    budget = producer.get("sample_size_budget") or {}
    value["official_row_count_per_unit"] = int(budget.get("official_rows") or 0)
    manifests = []
    for row in producer.get("checkpoint_manifest") or []:
        changed = deepcopy(dict(row))
        checkpoint = _static_checkpoint_loader(row)
        if checkpoint:
            changed["selected_policy"] = checkpoint.get("selected_policy")
        manifests.append(changed)
    value["checkpoint_manifest"] = manifests
    return value


def _write_source_sidecar(
    root: Path, checks: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    path = root / RAW_DIR / "source_authentication.json"
    atomic_json(path, {"schema": "carnot.exp7415.source_authentication.v1", "rows": list(checks)})
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    python = ".venv/bin/python"
    wrapper = WRAPPER_PATH.as_posix()
    commands = (
        (
            "cold_artifact_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate), "--preterminal"),
            "completion",
        ),
        (
            "independent_branch_recompute",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "completion",
        ),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in commands
    ]


def independent_replay(path: Path, root: Path = REPO_ROOT) -> list[str]:  # pragma: no cover
    """Reload both producer branches and compare fresh reductions to the candidate."""

    candidate = _load_object(path)
    errors = validate_artifact(candidate, root=root, require_terminal=False)
    checks, _hashes, producers = collect_preconditions(root)
    by_branch = {
        name: [row for row in checks if row.get("branch") == name] for name in ("static", "online")
    }
    static = (
        audit_static_branch(
            _augment_static_context(root, producers["static"]),
            precondition_rows=by_branch["static"],
        )
        if producers["static"]
        else blocked_branch("static", by_branch["static"])
    )
    online = audit_online_branch(producers["online"], precondition_rows=by_branch["online"])
    expected = [
        {
            key: row.get(key)
            for key in (
                "branch",
                "upstream",
                "available",
                "valid",
                "complete",
                "value",
                "producer_verdict_class",
                "label_authority",
                "errors",
            )
        }
        for row in (static, online)
    ]
    if canonical_hash(expected) != canonical_hash(candidate.get("branch_rows")):
        errors.append("fresh_branch_reduction_mismatch")
    return list(dict.fromkeys(errors))


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by declared entrypoint E2E.
    """Authenticate, reduce, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    _progress(started, "startup", "start")

    phase_started = time.monotonic()
    _progress(started, "preconditions", "start")
    checks, hashes, producers = collect_preconditions(root)
    source_sidecar = _write_source_sidecar(root, checks)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    _progress(started, "preconditions", "end", checks=len(checks))

    branch_checks = {
        name: [row for row in checks if row.get("branch") == name] for name in ("static", "online")
    }
    phase_started = time.monotonic()
    _progress(started, "static_reduction", "before_benchmark")
    static = (
        audit_static_branch(
            _augment_static_context(root, producers["static"]),
            precondition_rows=branch_checks["static"],
        )
        if producers["static"]
        else blocked_branch("static", branch_checks["static"])
    )
    spans.append(
        _span("static_reduction", phase_started, started, len(static.get("unit_rows") or []))
    )
    _progress(started, "static_reduction", "after_benchmark", valid=static["valid"])

    phase_started = time.monotonic()
    _progress(started, "online_reduction", "before_benchmark")
    online = audit_online_branch(producers["online"], precondition_rows=branch_checks["online"])
    spans.append(
        _span("online_reduction", phase_started, started, len(online.get("unit_rows") or []))
    )
    _progress(started, "online_reduction", "after_benchmark", valid=online["valid"])

    private = Path(tempfile.mkdtemp(prefix="exp7415-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    plan_errors = validate_command_plan(root, V650_MANIFEST, commands)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", plan_errors=len(plan_errors))
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, V650_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started, "affected_validation", "after_subprocesses", passed=affected_reduction["passed"]
    )

    ended_ns = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"exp7415-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_json_numeric_aggregation",
        inference_substrate_details={
            "device": platform.processor() or "cpu",
            "software": "python_numpy",
            "source_authentication_sidecar": source_sidecar,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=ended_ns - started_ns,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    candidate = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        require_terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses")
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
    )

    final_ns = time.monotonic_ns()
    final_receipt = build_current_work_receipt(
        run_id=f"exp7415-{started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_json_numeric_aggregation",
        inference_substrate_details={
            "device": platform.processor() or "cpu",
            "software": "python_numpy",
            "source_authentication_sidecar": source_sidecar,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=final_ns - started_ns,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )
    final = _build_artifact(
        static,
        online,
        preconditions=checks,
        source_hashes=hashes,
        validation_receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        require_terminal=True,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public audit and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--skip-source-bytes", action="store_true")
    parser.add_argument("--preterminal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or one requested fresh-process terminal reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(
            args.cold_replay,
            verify_source_bytes=not args.skip_source_bytes,
            require_terminal=not args.preterminal,
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
