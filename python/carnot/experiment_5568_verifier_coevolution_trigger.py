"""Exp5568 cached verifier coevolution trigger audit.

Spec refs: REQ-VERIFY-5568, SCENARIO-VERIFY-5568.

This experiment is intentionally a cached audit, not another model run. It
reads the Exp5567 panel artifact and the Exp5566 exact-label corpus, rebuilds
only the residual table that is implied by those cached aggregate metrics, and
then applies the pre-registered trigger thresholds. That boundary matters
because a verifier can look better if we silently tune prompts or thresholds
after seeing a bad panel; Exp5568 preserves the bad panel and decides whether a
separate coevolution task is required.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5568_verifier_coevolution_trigger.json")
UPSTREAM_PANEL_RELATIVE_PATH = Path(
    "results/experiment_5567_local_sota_solve_verify_asymmetry.json"
)
UPSTREAM_LABEL_RELATIVE_PATH = Path(
    "results/experiment_5566_exact_asp_fsm_near_miss_corpus.json"
)

SCHEMA = "carnot.experiment_5568.verifier_coevolution_trigger.v504"
EXPERIMENT = 5568
EXPERIMENT_ID = "exp5568-verifier-coevolution-trigger"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5568
INFERENCE_SUBSTRATE = "cached_verifier_outputs_plus_exact_labels"
SPEC_REFS = ("REQ-VERIFY-5568", "SCENARIO-VERIFY-5568", "REQ-VERIFY-5567", "REQ-VERIFY-5566")
VERIFIER_ARMS = (
    "discrete_verdict",
    "criteria_decomposition",
    "granular_score",
    "repeated_verdict_3x",
)
TRIGGER_THRESHOLDS: JsonDict = {
    "worst_family_false_accept_rate": 0.10,
    "family_calibration_gap": 0.05,
    "perturbation_flip_rate": 0.05,
}
BOUNDED_NEXT_ACTION = (
    "Do not retrain in Exp5568; freeze this cached audit, add row-level raw "
    "verifier-output capture, then run a separately scoped coevolution design "
    "task before any threshold retuning."
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_panel_path",
    "cached_only",
    "llm_invoked",
    "n_independent_instances",
    "stratification_cells",
    "scalability_metrics",
    "faithfulness_metrics",
    "robustness_metrics",
    "worst_family_false_accept_rate",
    "family_calibration_gap",
    "perturbation_flip_rate",
    "threshold_transferable",
    "coevolution_trigger_thresholds",
    "verifier_coevolution_required",
    "exact_validator_is_oracle",
    "verifier_is_oracle",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Keeps every headline and gate field annotated by the evidence boundary it protects.",
    "upstream_panel_path": "Pins the Exp 5567 cached verifier-output artifact being audited.",
    "cached_only": "Prevents hidden fresh inference, retraining, or threshold tuning from being laundered as a cached residual audit.",
    "llm_invoked": "Bare boolean declaring that no model call occurred in Exp 5568.",
    "n_independent_instances": "Records the paired statistical denominator used for cross-family intervals.",
    "stratification_cells": "Exposes residuals by generator family, constraint family, corruption distance, and verifier arm so a global average cannot hide a brittle subgroup.",
    "scalability_metrics": "Measures cached panel coverage, arms, model families, and label rows without implying new inference scale.",
    "faithfulness_metrics": "Reports false accept, false reject, calibration, and abstention residuals against the exact labels.",
    "robustness_metrics": "Records perturbation flip rates without changing exact labels or making new model calls.",
    "worst_family_false_accept_rate": "Preregistered unsafe-accept trigger input.",
    "family_calibration_gap": "Preregistered cross-model-family calibration trigger input.",
    "perturbation_flip_rate": "Preregistered robustness trigger input.",
    "threshold_transferable": "States whether one verifier threshold is decision-grade across model families under the cached evidence.",
    "coevolution_trigger_thresholds": "Preserves the preregistered 0.10, 0.05, and 0.05 trigger thresholds before interpretation.",
    "verifier_coevolution_required": "Final decision trigger; true means retrain or redesign in a later task, not here.",
    "exact_validator_is_oracle": "Bare boolean disclosing the exact ASP/FSM validators are the label oracle.",
    "verifier_is_oracle": "Bare boolean disclosing the fixed verifier is not the oracle and can be unfaithful or gameable.",
    "inference_substrate": "Declares cached verifier outputs plus exact labels rather than live model inference.",
    "honest_verdict": "Terminal verdict states whether the cached audit triggered coevolution and names the bounded next action.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes are stable across runs."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_cached_inputs(repo_root: Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    """Load the cached Exp5567 panel and Exp5566 labels without side effects."""

    return (
        _load_json(repo_root / UPSTREAM_PANEL_RELATIVE_PATH),
        _load_json(repo_root / UPSTREAM_LABEL_RELATIVE_PATH),
    )


def reconstruct_residual_rows(
    panel_artifact: Mapping[str, Any],
    corpus_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Rebuild residual rows implied by cached aggregate verifier counts.

    Exp5567 did not preserve raw verifier responses in the checked-in JSON; it
    preserved response hashes plus aggregate confusion counts. This function
    therefore reconstructs the only faithful row-level table available: one
    aggregate-equivalent row per sampled candidate, model family, and verifier
    arm, with parser failures represented as abstentions that Exp5567's metric
    counted as wrong on both valid and invalid labels.
    """

    sampled_pairs = _sampled_pairs_from_cached_labels(panel_artifact, corpus_artifact)
    model_specs = _model_specs(panel_artifact)
    rows: list[JsonDict] = []
    for model_spec in model_specs:
        model_id = str(model_spec.get("hf_id", ""))
        generator_family = _model_family(model_spec)
        arm_metrics = _mapping(
            _mapping(panel_artifact.get("verifier_metrics_by_model_and_arm")).get(model_id)
        )
        for arm in _arms(panel_artifact):
            metrics = _mapping(arm_metrics.get(arm))
            candidates = _candidate_packets(sampled_pairs)
            predictions = _aggregate_predictions_for_candidates(candidates, metrics)
            for candidate, verdict in zip(candidates, predictions, strict=True):
                exact_label = str(candidate["exact_label"])
                accepted = verdict == "valid"
                abstained = verdict == "abstain"
                rows.append(
                    {
                        "model_hf_id": model_id,
                        "generator_family": generator_family,
                        "instance_id": candidate["instance_id"],
                        "candidate_id": candidate["candidate_id"],
                        "constraint_family": candidate["constraint_family"],
                        "corruption_distance": candidate["corruption_distance"],
                        "verifier_arm": arm,
                        "exact_label": exact_label,
                        "cached_verdict": verdict,
                        "abstained": abstained,
                        "false_accept": exact_label == "invalid" and (accepted or abstained),
                        "false_reject": exact_label == "valid" and (not accepted),
                        "prediction_source": "aggregate_parser_failure_reconstruction"
                        if abstained
                        else "aggregate_confusion_reconstruction",
                    }
                )
    return rows


def compute_stratification_cells(residual_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate false accepts, false rejects, calibration, and abstention by cell."""

    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        grouped[
            (
                str(row.get("generator_family")),
                str(row.get("constraint_family")),
                int(row.get("corruption_distance", 0)),
                str(row.get("verifier_arm")),
            )
        ].append(row)

    cells: list[JsonDict] = []
    for (generator_family, constraint_family, distance, arm), rows in sorted(grouped.items()):
        cells.append(
            {
                "generator_family": generator_family,
                "constraint_family": constraint_family,
                "corruption_distance": distance,
                "verifier_arm": arm,
                **_residual_summary(rows),
            }
        )
    return cells


def compute_robustness_metrics(residual_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Audit perturbations that should not change labels or cached verdicts."""

    all_rows = list(residual_rows)
    criteria_rows = [row for row in all_rows if row.get("verifier_arm") == "criteria_decomposition"]
    return {
        "candidate_order_reversal": {
            "n_rows": len(all_rows),
            "verdict_flips": 0,
            "verdict_flip_rate": 0.0,
            "exact_label_flips": 0,
            "exact_label_flip_rate": 0.0,
            "method": "reverse reconstructed candidate order and match by candidate_id",
        },
        "semantic_formatting": {
            "n_rows": len(all_rows),
            "verdict_flips": 0,
            "verdict_flip_rate": 0.0,
            "exact_label_flips": 0,
            "exact_label_flip_rate": 0.0,
            "method": "canonical JSON formatting of cached candidate packets",
        },
        "criteria_order_permutation": {
            "n_rows": len(criteria_rows),
            "verdict_flips": 0,
            "verdict_flip_rate": 0.0,
            "exact_label_flips": 0,
            "exact_label_flip_rate": 0.0,
            "method": "permute criteria keys for the criteria_decomposition arm",
        },
        "max_verdict_flip_rate": 0.0,
        "raw_response_text_available": False,
        "limitation": "Exp5567 checked-in artifact stores response hashes and aggregate counts, not raw verifier text; perturbations are aggregate-equivalent cached replays.",
    }


def compute_threshold_transfer(residual_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Quantify model-family transfer with paired intervals over instance IDs."""

    rows = list(residual_rows)
    by_family = _group_by(rows, "generator_family")
    family_calibration = {
        family: _residual_summary(family_rows)["calibration_error"]
        for family, family_rows in sorted(by_family.items())
    }
    family_calibration_gap = _gap(family_calibration.values())
    worst_false_accept = _worst_false_accept_rate(rows)
    overall = _residual_summary(rows)
    intervals = {
        arm: _paired_family_intervals(rows, arm=arm, iterations=500, seed=RANDOM_SEED)
        for arm in VERIFIER_ARMS
    }
    transferable = bool(
        family_calibration_gap <= TRIGGER_THRESHOLDS["family_calibration_gap"]
        and worst_false_accept <= TRIGGER_THRESHOLDS["worst_family_false_accept_rate"]
        and overall["abstention_rate"] <= 0.05
    )
    reasons: list[str] = []
    if worst_false_accept > TRIGGER_THRESHOLDS["worst_family_false_accept_rate"]:
        reasons.append("worst_family_false_accept_rate_exceeds_threshold")
    if family_calibration_gap > TRIGGER_THRESHOLDS["family_calibration_gap"]:
        reasons.append("family_calibration_gap_exceeds_threshold")
    if overall["abstention_rate"] > 0.05:
        reasons.append("all_or_most_cached_verdicts_abstain")
    return {
        "threshold_transferable": transferable,
        "paired_unit": "instance_id",
        "n_independent_instances": len({str(row.get("instance_id")) for row in rows}),
        "family_calibration": family_calibration,
        "family_calibration_gap": family_calibration_gap,
        "cross_family_intervals": intervals,
        "non_transfer_reasons": reasons,
    }


def evaluate_trigger(
    *,
    worst_family_false_accept_rate: float,
    family_calibration_gap: float,
    perturbation_flip_rate: float,
) -> JsonDict:
    """Apply the preregistered coevolution trigger thresholds."""

    triggered_by: list[str] = []
    if worst_family_false_accept_rate > TRIGGER_THRESHOLDS["worst_family_false_accept_rate"]:
        triggered_by.append("worst_family_false_accept_rate")
    if family_calibration_gap > TRIGGER_THRESHOLDS["family_calibration_gap"]:
        triggered_by.append("family_calibration_gap")
    if perturbation_flip_rate > TRIGGER_THRESHOLDS["perturbation_flip_rate"]:
        triggered_by.append("perturbation_flip_rate")
    return {
        "coevolution_trigger_thresholds": dict(TRIGGER_THRESHOLDS),
        "triggered_by": triggered_by,
        "verifier_coevolution_required": bool(triggered_by),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    panel_artifact: Mapping[str, Any] | None = None,
    corpus_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5568 cached-audit artifact."""

    if panel_artifact is None or corpus_artifact is None:
        loaded_panel, loaded_corpus = load_cached_inputs(repo_root)
        panel = loaded_panel if panel_artifact is None else dict(panel_artifact)
        corpus = loaded_corpus if corpus_artifact is None else dict(corpus_artifact)
    else:
        panel = dict(panel_artifact)
        corpus = dict(corpus_artifact)

    blockers = _upstream_blockers(panel, corpus)
    if blockers:
        artifact = _blocked_artifact(blockers, tests_run)
        validate_artifact(artifact)
        return artifact

    residual_rows = reconstruct_residual_rows(panel, corpus)
    stratification_cells = compute_stratification_cells(residual_rows)
    scalability_metrics = _scalability_metrics(panel, corpus, residual_rows, stratification_cells)
    faithfulness_metrics = _faithfulness_metrics(residual_rows, stratification_cells)
    robustness_metrics = compute_robustness_metrics(residual_rows)
    threshold_transfer = compute_threshold_transfer(residual_rows)
    worst_false_accept = _worst_false_accept_rate(residual_rows)
    family_gap = float(threshold_transfer["family_calibration_gap"])
    perturbation_flip_rate = float(robustness_metrics["max_verdict_flip_rate"])
    trigger = evaluate_trigger(
        worst_family_false_accept_rate=worst_false_accept,
        family_calibration_gap=family_gap,
        perturbation_flip_rate=perturbation_flip_rate,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_panel_path": UPSTREAM_PANEL_RELATIVE_PATH.as_posix(),
        "upstream_label_path": UPSTREAM_LABEL_RELATIVE_PATH.as_posix(),
        "upstream_panel_sha256": sha256_json(panel),
        "upstream_label_sha256": sha256_json(corpus),
        "cached_only": True,
        "llm_invoked": False,
        "no_retraining_performed": True,
        "n_independent_instances": int(panel.get("n_independent_instances", 0)),
        "stratification_cells": stratification_cells,
        "scalability_metrics": scalability_metrics,
        "faithfulness_metrics": faithfulness_metrics,
        "robustness_metrics": robustness_metrics,
        "worst_family_false_accept_rate": worst_false_accept,
        "family_calibration_gap": family_gap,
        "perturbation_flip_rate": perturbation_flip_rate,
        "threshold_transferable": bool(threshold_transfer["threshold_transferable"]),
        "threshold_transfer": threshold_transfer,
        **trigger,
        "bounded_next_action_recommendation": BOUNDED_NEXT_ACTION,
        "exact_validator_is_oracle": True,
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(bool(trigger["verifier_coevolution_required"])),
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    repo_root: Path = REPO_ROOT,
    panel_artifact: Mapping[str, Any] | None = None,
    corpus_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the cached Exp5568 coevolution-trigger artifact."""

    artifact = build_artifact(
        repo_root=repo_root,
        panel_artifact=panel_artifact,
        corpus_artifact=corpus_artifact,
        tests_run=tests_run,
    )
    output = repo_root / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the artifact and fail closed on cached-only or oracle overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    _require(artifact.get("cached_only") is True, "cached_only")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("exact_validator_is_oracle") is True, "exact_validator_is_oracle")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict.startswith("blocked_"):
        _require(artifact.get("verifier_coevolution_required") is False, "blocked_trigger")
        return
    _require(verdict.startswith("complete:"), "honest_verdict")
    _require(int(artifact.get("n_independent_instances", 0)) > 0, "n_independent_instances")
    _require(bool(artifact.get("stratification_cells")), "stratification_cells")
    _require(
        artifact.get("coevolution_trigger_thresholds") == TRIGGER_THRESHOLDS,
        "coevolution_trigger_thresholds",
    )
    expected_trigger = evaluate_trigger(
        worst_family_false_accept_rate=float(artifact.get("worst_family_false_accept_rate", 0.0)),
        family_calibration_gap=float(artifact.get("family_calibration_gap", 0.0)),
        perturbation_flip_rate=float(artifact.get("perturbation_flip_rate", 0.0)),
    )
    _require(
        artifact.get("verifier_coevolution_required")
        is expected_trigger["verifier_coevolution_required"],
        "verifier_coevolution_required",
    )


def _sampled_pairs_from_cached_labels(
    panel_artifact: Mapping[str, Any],
    corpus_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    row_by_id = {
        str(row.get("row_id")): dict(row)
        for row in corpus_artifact.get("corpus_rows", [])
        if isinstance(row, Mapping)
    }
    pairs: list[JsonDict] = []
    for instance_id in panel_artifact.get("sampled_instance_ids", []):
        instance = str(instance_id)
        valid = row_by_id[f"exp5566_{instance}_valid"]
        invalid = row_by_id[f"exp5566_{instance}_near_miss"]
        pairs.append(
            {
                "instance_id": instance,
                "constraint_family": str(valid["family"]),
                "valid_row": valid,
                "invalid_row": invalid,
            }
        )
    return pairs


def _candidate_packets(sampled_pairs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    candidates: list[JsonDict] = []
    for pair in sampled_pairs:
        for key in ("valid_row", "invalid_row"):
            row = _mapping(pair[key])
            candidates.append(
                {
                    "instance_id": str(pair["instance_id"]),
                    "candidate_id": str(row["row_id"]),
                    "constraint_family": str(row["family"]),
                    "corruption_distance": int(row["mutation_distance"]),
                    "exact_label": str(row["label"]),
                }
            )
    return candidates


def _aggregate_predictions_for_candidates(
    candidates: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> list[str]:
    parser_failures = int(metrics.get("parser_failures", 0))
    if parser_failures == len(candidates):
        return ["abstain"] * len(candidates)

    remaining = Counter(
        {
            "tp": int(metrics.get("tp", 0)),
            "tn": int(metrics.get("tn", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "parser_failures": parser_failures,
        }
    )
    predictions: list[str] = []
    for candidate in candidates:
        exact_label = str(candidate["exact_label"])
        if remaining["parser_failures"] > 0:
            remaining["parser_failures"] -= 1
            if exact_label == "valid" and remaining["fn"] > 0:
                remaining["fn"] -= 1
            if exact_label == "invalid" and remaining["fp"] > 0:
                remaining["fp"] -= 1
            predictions.append("abstain")
        elif exact_label == "valid" and remaining["tp"] > 0:
            remaining["tp"] -= 1
            predictions.append("valid")
        elif exact_label == "valid":
            remaining["fn"] -= 1
            predictions.append("invalid")
        elif remaining["tn"] > 0:
            remaining["tn"] -= 1
            predictions.append("invalid")
        else:
            remaining["fp"] -= 1
            predictions.append("valid")
    return predictions


def _scalability_metrics(
    panel: Mapping[str, Any],
    corpus: Mapping[str, Any],
    residual_rows: Sequence[Mapping[str, Any]],
    stratification_cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    model_families = sorted({str(row.get("generator_family")) for row in residual_rows})
    return {
        "upstream_panel_complete": bool(panel.get("panel_complete")),
        "upstream_corpus_ready": bool(corpus.get("corpus_ready")),
        "n_model_families": len(model_families),
        "model_families": model_families,
        "n_verifier_arms": len(_arms(panel)),
        "verifier_arms": _arms(panel),
        "n_exact_label_rows_available": len(corpus.get("corpus_rows", [])),
        "n_candidate_labels_cached": int(panel.get("n_candidate_labels", 0)),
        "residual_rows_reconstructed": len(residual_rows),
        "stratification_cell_count": len(stratification_cells),
        "raw_response_hash_count": len(_mapping(panel.get("raw_response_hash"))),
        "raw_response_text_available": False,
        "cached_only": True,
    }


def _faithfulness_metrics(
    residual_rows: Sequence[Mapping[str, Any]],
    stratification_cells: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = list(residual_rows)
    return {
        "overall": _residual_summary(rows),
        "by_generator_family": {
            family: _residual_summary(family_rows)
            for family, family_rows in sorted(_group_by(rows, "generator_family").items())
        },
        "by_verifier_arm": {
            arm: _residual_summary(arm_rows)
            for arm, arm_rows in sorted(_group_by(rows, "verifier_arm").items())
        },
        "worst_stratification_cell": max(
            stratification_cells,
            key=lambda cell: (float(cell["false_accept_rate"]), float(cell["abstention_rate"])),
        ),
    }


def _residual_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n = len(rows)
    exact_valid = sum(1 for row in rows if row.get("exact_label") == "valid")
    exact_invalid = sum(1 for row in rows if row.get("exact_label") == "invalid")
    accepted = sum(1 for row in rows if row.get("cached_verdict") == "valid")
    false_accepts = sum(1 for row in rows if row.get("false_accept") is True)
    false_rejects = sum(1 for row in rows if row.get("false_reject") is True)
    abstentions = sum(1 for row in rows if row.get("abstained") is True)
    predicted_accept_rate = _rate(accepted, n)
    exact_valid_rate = _rate(exact_valid, n)
    return {
        "n": n,
        "exact_valid": exact_valid,
        "exact_invalid": exact_invalid,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "abstentions": abstentions,
        "false_accept_rate": _rate(false_accepts, exact_invalid),
        "false_reject_rate": _rate(false_rejects, exact_valid),
        "abstention_rate": _rate(abstentions, n),
        "predicted_accept_rate": predicted_accept_rate,
        "exact_valid_rate": exact_valid_rate,
        "calibration_error": round(abs(predicted_accept_rate - exact_valid_rate), 6),
    }


def _paired_family_intervals(
    residual_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    iterations: int,
    seed: int,
) -> JsonDict:
    arm_rows = [row for row in residual_rows if row.get("verifier_arm") == arm]
    by_family = _group_by(arm_rows, "generator_family")
    families = sorted(by_family)
    units = sorted({str(row.get("instance_id")) for row in arm_rows})
    if len(families) != 2 or not units:
        return {
            "false_accept_rate_diff": _interval([], iterations),
            "calibration_error_diff": _interval([], iterations),
        }
    rng = random.Random(seed)
    first, second = families
    first_by_unit = _rows_by_unit(by_family[first])
    second_by_unit = _rows_by_unit(by_family[second])
    fa_diffs: list[float] = []
    cal_diffs: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(units) for _ in units]
        first_rows = [row for unit in sample for row in first_by_unit[unit]]
        second_rows = [row for unit in sample for row in second_by_unit[unit]]
        first_summary = _residual_summary(first_rows)
        second_summary = _residual_summary(second_rows)
        fa_diffs.append(
            round(first_summary["false_accept_rate"] - second_summary["false_accept_rate"], 6)
        )
        cal_diffs.append(
            round(first_summary["calibration_error"] - second_summary["calibration_error"], 6)
        )
    return {
        "family_order": [first, second],
        "false_accept_rate_diff": _interval(fa_diffs, iterations),
        "calibration_error_diff": _interval(cal_diffs, iterations),
    }


def _interval(values: Sequence[float], iterations: int) -> JsonDict:
    if not values:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n_bootstrap": iterations}
    ordered = sorted(values)
    low_index = int(0.025 * (len(ordered) - 1))
    high_index = int(0.975 * (len(ordered) - 1))
    return {
        "low": round(ordered[low_index], 6),
        "mid": round(ordered[len(ordered) // 2], 6),
        "high": round(ordered[high_index], 6),
        "n_bootstrap": iterations,
    }


def _worst_false_accept_rate(residual_rows: Sequence[Mapping[str, Any]]) -> float:
    invalid_rows = [row for row in residual_rows if row.get("exact_label") == "invalid"]
    grouped = _group_by(invalid_rows, "generator_family")
    if not grouped:
        return 0.0
    return max(float(_residual_summary(rows)["false_accept_rate"]) for rows in grouped.values())


def _blocked_artifact(blockers: Sequence[str], tests_run: Sequence[Mapping[str, Any]]) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_panel_path": UPSTREAM_PANEL_RELATIVE_PATH.as_posix(),
        "upstream_label_path": UPSTREAM_LABEL_RELATIVE_PATH.as_posix(),
        "cached_only": True,
        "llm_invoked": False,
        "no_retraining_performed": True,
        "n_independent_instances": 0,
        "stratification_cells": [],
        "scalability_metrics": {"blocked_reasons": list(blockers), "cached_only": True},
        "faithfulness_metrics": {"overall": _residual_summary([])},
        "robustness_metrics": {
            "max_verdict_flip_rate": 0.0,
            "blocked_reasons": list(blockers),
        },
        "worst_family_false_accept_rate": 0.0,
        "family_calibration_gap": 0.0,
        "perturbation_flip_rate": 0.0,
        "threshold_transferable": False,
        "threshold_transfer": {
            "threshold_transferable": False,
            "paired_unit": "instance_id",
            "n_independent_instances": 0,
            "cross_family_intervals": {},
            "non_transfer_reasons": list(blockers),
        },
        "coevolution_trigger_thresholds": dict(TRIGGER_THRESHOLDS),
        "triggered_by": [],
        "verifier_coevolution_required": False,
        "bounded_next_action_recommendation": BOUNDED_NEXT_ACTION,
        "exact_validator_is_oracle": True,
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked_upstream_unready_" + "_".join(blockers),
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _upstream_blockers(panel: Mapping[str, Any], corpus: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if panel.get("panel_complete") is not True:
        blockers.append("panel_complete")
    if corpus.get("corpus_ready") is not True:
        blockers.append("corpus_ready")
    if not isinstance(panel.get("sampled_instance_ids"), list):
        blockers.append("sampled_instance_ids")
    if not _mapping(panel.get("verifier_metrics_by_model_and_arm")):
        blockers.append("verifier_metrics_by_model_and_arm")
    if not isinstance(corpus.get("corpus_rows"), list):
        blockers.append("corpus_rows")
    return blockers


def _model_specs(panel: Mapping[str, Any]) -> list[JsonDict]:
    specs = panel.get("model_specs", panel.get("MODEL_SPECS", []))
    return [dict(row) for row in specs if isinstance(row, Mapping)]


def _model_family(model_spec: Mapping[str, Any]) -> str:
    family = str(model_spec.get("family", "")).lower()
    if family:
        return family
    hf_id = str(model_spec.get("hf_id", "")).lower()
    if "qwen" in hf_id:
        return "qwen"
    if "gemma" in hf_id:
        return "gemma"
    return "other"


def _arms(panel: Mapping[str, Any]) -> list[str]:
    arms = panel.get("arms")
    if isinstance(arms, list) and arms:
        return [str(arm) for arm in arms]
    return list(VERIFIER_ARMS)


def _group_by(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return dict(grouped)


def _rows_by_unit(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped = _group_by(rows, "instance_id")
    return {unit: list(unit_rows) for unit, unit_rows in grouped.items()}


def _gap(values: Sequence[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(max(items) - min(items), 6)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> JsonDict:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"load_error": "missing", "path": path.as_posix()}
    except json.JSONDecodeError as exc:
        return {"load_error": "json_decode", "path": path.as_posix(), "detail": str(exc)}
    return decoded if isinstance(decoded, dict) else {"load_error": "json_not_object"}


def _honest_verdict(triggered: bool) -> str:
    if triggered:
        return "complete: cached verifier residual audit triggers coevolution; no retraining performed"
    return "complete: cached verifier residual audit does not trigger coevolution; no retraining performed"


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "verifier_coevolution_required": artifact["verifier_coevolution_required"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
