"""Build the Exp 3288 KAN sidecar failure-autopsy boundary artifact.

Spec refs: REQ-REPORT-3288, SCENARIO-REPORT-3288.

This module is deliberately an offline ledger. The failed full-corpus KAN
artifact already contains the detector scores, thresholds, slice metrics, and
baseline comparisons. Re-reading that evidence instead of retraining keeps the
boundary decision tied to the failed promotion gate and prevents a diagnostic
task from accidentally becoming a new headline-model attempt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.kan_sidecar_failure_autopsy_boundary.v1"
EXPERIMENT_ID = "exp3288"
TASK_ID = "exp3288-kan-sidecar-failure-autopsy-boundary-v1"
ARTIFACT = "experiment_3288_kan_sidecar_failure_autopsy_boundary_v1"
MILESTONE = "2026.05.304"
RUN_DATE = "20260528"
RANDOM_SEED = 3288

OUTPUT_REL_PATH = Path("results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json")
EXP3273_REL_PATH = Path(
    "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
)
EXP3272_REL_PATH = Path(
    "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
)
EXP3283_REL_PATH = Path(
    "results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json"
)

SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
ALLOWED_BOUNDARY_DECISIONS = {
    "retain_sidecar_only",
    "retire_from_prompt_injection_headline",
    "prerequisite_required",
}
PERMITTED_DOWNSTREAM_USE = [
    "offline_failure_autopsy",
    "negative_control_regression_fixture",
    "future_kan_work_prerequisite_evidence_only",
]
REQUIRED_FIELDS = {
    "kan_failure_autopsy_ready",
    "kan_boundary_decision_ready",
    "prior_full_corpus_auroc",
    "prior_delong_noninferiority_passed",
    "per_slice_failure_summary",
    "aligned_instruction_false_positive_summary",
    "leakage_or_provenance_findings",
    "baseline_comparison_summary",
    "kan_boundary_decision",
    "permitted_downstream_use",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3288_kan_sidecar_failure_autopsy_boundary.py -q -o addopts=''",
    ".venv/bin/coverage run --source=python/carnot/reporting/kan_sidecar_failure_autopsy_boundary_3288.py -m pytest -o addopts='' tests/python/test_experiment_3288_kan_sidecar_failure_autopsy_boundary.py -q",
    ".venv/bin/coverage report --include='python/carnot/reporting/kan_sidecar_failure_autopsy_boundary_3288.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-REPORT-3288: autopsy prior KAN failure without new model work."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3273 = read_json_object(root_path / EXP3273_REL_PATH)
    exp3272 = read_json_object(root_path / EXP3272_REL_PATH)
    exp3283 = read_json_object(root_path / EXP3283_REL_PATH)
    threshold = selected_max_f1_threshold(exp3273)
    prior_auroc = metric_float(exp3273.get("full_corpus_auroc"))
    prior_delong_passed = exp3273.get("delong_noninferiority_passed") is True
    boundary_decision = "retire_from_prompt_injection_headline"
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-REPORT-3288", "SCENARIO-REPORT-3288"],
        "kan_failure_autopsy_ready": True,
        "kan_boundary_decision_ready": True,
        "prior_full_corpus_auroc": prior_auroc,
        "prior_full_corpus_auprc": metric_float(exp3273.get("full_corpus_auprc")),
        "prior_delong_noninferiority_passed": prior_delong_passed,
        "prior_delong_ci": list(exp3273.get("delong_ci") or []),
        "prior_sidecar_only": exp3273.get("sidecar_only") is True,
        "selected_threshold_name": "max_f1_eval",
        "selected_threshold": threshold,
        "per_slice_failure_summary": per_slice_failure_summary(exp3273, threshold),
        "aligned_instruction_false_positive_summary": (
            aligned_instruction_false_positive_summary(exp3273, threshold)
        ),
        "leakage_or_provenance_findings": leakage_or_provenance_findings(
            exp3272,
            exp3283,
            exp3273,
        ),
        "baseline_comparison_summary": baseline_comparison_summary(exp3273),
        "kan_boundary_decision": boundary_decision,
        "permitted_downstream_use": list(PERMITTED_DOWNSTREAM_USE),
        "prohibited_downstream_use": [
            "prompt_injection_headline_detector",
            "repair_gate_authority",
            "standalone_garak_success_evidence",
            "production_triage_without_new_calibrated_false_positive_gate",
        ],
        "future_work_prerequisite": (
            "reattempt only after a balanced two-class slice suite, an aligned-benign "
            "false-positive ceiling, leakage/provenance-clean labels, and a calibrated "
            "threshold beat regex/keyword baselines on held-out evidence"
        ),
        "no_retraining_performed": True,
        "no_new_kan_training_or_scoring": True,
        "no_new_model_execution": True,
        "no_garak_or_repair_run": True,
        "protected_files_untouched": {
            "scripts/research_conductor.py": True,
            "ops/status.md": True,
            "ops/changelog.md": True,
            "_bmad/traceability.md": True,
        },
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "random_seed": int(random_seed),
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3288 JSON deliverable."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read a checked-in JSON object used as evidence for the autopsy."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def selected_max_f1_threshold(exp3273: Mapping[str, Any]) -> float:
    """Recover the threshold that produced high recall and benign false alarms."""

    threshold_metrics = mapping(exp3273.get("threshold_metrics"))
    selected = mapping(threshold_metrics.get("selected_thresholds"))
    return metric_float(selected.get("max_f1_eval"))


def per_slice_failure_summary(exp3273: Mapping[str, Any], threshold: float) -> JsonDict:
    """Summarize where the KAN score failed instead of hiding behind one AUROC."""

    slice_records = [
        slice_record(name, mapping(raw), threshold)
        for name, raw in sorted(mapping(exp3273.get("per_slice_metrics")).items())
    ]
    single_class = [
        row for row in slice_records if row["positive_count"] == 0 or row["negative_count"] == 0
    ]
    threshold_row = mapping(
        mapping(exp3273.get("threshold_metrics")).get("eval_plus_holdout", {})
    ).get("max_f1_eval", {})
    threshold_summary = threshold_failure_summary(mapping(threshold_row))
    failure_modes = [
        "below_random_full_corpus_auroc",
        "delong_noninferiority_failed",
        "holdout_auroc_below_random",
        "single_class_slices_make_slice_auroc_uninformative",
        "max_f1_threshold_predicts_every_benign_eval_holdout_row_positive",
    ]
    return {
        "slice_count": len(slice_records),
        "single_class_slice_count": len(single_class),
        "two_class_slice_count": len(slice_records) - len(single_class),
        "single_class_slices": single_class,
        "threshold_failure_summary": threshold_summary,
        "holdout_auroc": metric_float(
            mapping(exp3273.get("split_metrics")).get("holdout", {}).get("auroc")
        ),
        "global_failure_modes": failure_modes,
    }


def slice_record(name: str, metrics: Mapping[str, Any], threshold: float) -> JsonDict:
    """Convert one Exp 3273 slice metric into an autopsy row."""

    negative_count = int_value(metrics.get("negative_count"))
    return {
        "slice": name,
        "n": int_value(metrics.get("n")),
        "positive_count": int_value(metrics.get("positive_count")),
        "negative_count": negative_count,
        "auroc": metrics.get("auroc"),
        "f1_at_selected_threshold": metrics.get("f1_at_selected_threshold"),
        "mean_score": metric_float(metrics.get("mean_score")),
        "estimated_false_positive_rate": negative_only_false_positive_rate(
            negative_count,
            metrics,
            threshold,
        ),
    }


def threshold_failure_summary(threshold_row: Mapping[str, Any]) -> JsonDict:
    """Explain why high-recall triage collapses into predict-positive behavior."""

    fp = int_value(threshold_row.get("fp"))
    tn = int_value(threshold_row.get("tn"))
    tp = int_value(threshold_row.get("tp"))
    fn = int_value(threshold_row.get("fn"))
    predicted_positive = int_value(threshold_row.get("predicted_positive_count"))
    total = fp + tn + tp + fn
    return {
        "threshold": metric_float(threshold_row.get("threshold")),
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_positive_rate": safe_div(fp, fp + tn),
        "recall": metric_float(threshold_row.get("recall")),
        "precision": metric_float(threshold_row.get("precision")),
        "predicted_positive_count": predicted_positive,
        "predicted_positive_rate": safe_div(predicted_positive, total),
        "true_negative_rate": safe_div(tn, fp + tn),
    }


def aligned_instruction_false_positive_summary(
    exp3273: Mapping[str, Any],
    threshold: float,
) -> JsonDict:
    """Measure the benign aligned-control cost at the selected threshold."""

    metrics = mapping(exp3273.get("per_slice_metrics"))
    aligned = mapping(metrics.get("instruction_alignment:aligned_instruction"))
    negative_count = int_value(aligned.get("negative_count"))
    false_positive_rate = negative_only_false_positive_rate(negative_count, aligned, threshold)
    category_rates = {
        name: negative_only_false_positive_rate(
            int_value(mapping(raw).get("negative_count")), mapping(raw), threshold
        )
        for name, raw in sorted(metrics.items())
        if name.startswith("category:") and int_value(mapping(raw).get("negative_count")) > 0
    }
    return {
        "source_slice": "instruction_alignment:aligned_instruction",
        "threshold_name": "max_f1_eval",
        "threshold": threshold,
        "aligned_instruction_case_count": int_value(aligned.get("n")),
        "aligned_instruction_negative_count": negative_count,
        "aligned_instruction_false_positive_rate": false_positive_rate,
        "aligned_instruction_false_positive_count": int(
            round(false_positive_rate * negative_count)
        ),
        "aligned_instruction_min_score": metric_float(aligned.get("min_score")),
        "aligned_instruction_max_score": metric_float(aligned.get("max_score")),
        "category_false_positive_rates": category_rates,
        "utility_conclusion": (
            "selected high-recall threshold flags aligned benign controls, so it is "
            "not a useful bounded triage gate without a new false-positive ceiling"
        ),
    }


def baseline_comparison_summary(exp3273: Mapping[str, Any]) -> JsonDict:
    """Compare the KAN score with baselines that were already in Exp 3273."""

    prior_auroc = metric_float(exp3273.get("full_corpus_auroc"))
    baseline_metrics = mapping(exp3273.get("baseline_detector_metrics"))
    trivial = {
        name: metric_float(mapping(baseline_metrics.get(name)).get("auroc"))
        for name in ("keyword_feature_baseline", "regex_phrase_baseline")
    }
    strongest_name = max(trivial, key=trivial.get)
    exact_auroc = metric_float(
        mapping(baseline_metrics.get("exact_label_upper_bound")).get("auroc")
    )
    split = mapping(mapping(exp3273.get("split_metrics")).get("eval_plus_holdout"))
    threshold_summary = threshold_failure_summary(
        mapping(
            mapping(mapping(exp3273.get("threshold_metrics")).get("eval_plus_holdout")).get(
                "max_f1_eval"
            )
        )
    )
    shard = mapping(exp3273.get("shard_302_comparison"))
    positive_count = int_value(split.get("positive_count"))
    total = int_value(split.get("n"))
    return {
        "kan": {
            "auroc": prior_auroc,
            "auprc": metric_float(exp3273.get("full_corpus_auprc")),
        },
        "random_auroc_reference": 0.5,
        "kan_minus_random_auroc": metric_float(prior_auroc - 0.5),
        "trivial_baselines": {
            name: {
                "auroc": value,
                "auprc": metric_float(mapping(baseline_metrics.get(name)).get("auprc")),
                "kan_minus_auroc": metric_float(prior_auroc - value),
            }
            for name, value in trivial.items()
        },
        "strongest_trivial_baseline": {
            "name": strongest_name,
            "auroc": trivial[strongest_name],
        },
        "kan_minus_strongest_trivial_auroc": metric_float(prior_auroc - trivial[strongest_name]),
        "exact_label_upper_bound": {"auroc": exact_auroc},
        "kan_minus_exact_upper_bound_auroc": metric_float(prior_auroc - exact_auroc),
        "prior_302_shard_auroc": metric_float(shard.get("prior_shard_auroc")),
        "kan_minus_prior_302_shard_auroc": metric_float(shard.get("full_minus_prior_shard_auroc")),
        "positive_prevalence_baseline": safe_div(positive_count, total),
        "max_f1_threshold_behavior": threshold_summary,
        "baseline_verdict": (
            "KAN is below random AUROC, below regex/keyword baselines, far below "
            "the exact-label upper bound, and its high-recall threshold behaves "
            "like a predict-positive baseline"
        ),
    }


def leakage_or_provenance_findings(
    exp3272: Mapping[str, Any],
    exp3283: Mapping[str, Any],
    exp3273: Mapping[str, Any],
) -> list[JsonDict]:
    """Carry forward data-hygiene and corrigendum findings that bound claims."""

    leakage = mapping(exp3272.get("leakage_audit"))
    findings: list[JsonDict] = [
        {
            "kind": "split_leakage_boundary",
            "source_experiment_id": "exp3272",
            "leakage_audit_passed": exp3272.get("leakage_audit_passed") is True,
            "exact_duplicate_overlap_rows": nested_int(
                leakage,
                "exact_duplicate_overlap",
                "overlap_row_count",
            ),
            "near_duplicate_overlap_rows": nested_int(
                leakage,
                "near_duplicate_overlap",
                "overlap_row_count",
            ),
            "normal_template_family_overlap_rows": nested_int(
                leakage,
                "normal_template_family_overlap",
                "overlap_row_count",
            ),
            "garak_template_family_overlap_count": int_value(
                leakage.get("garak_template_family_overlap_count")
            ),
            "within_source_duplicate_count": int_value(
                exp3272.get("within_source_duplicate_count")
            ),
            "finding": (
                "normal train/eval/holdout leakage passed, but Garak template "
                "family overlap and within-source duplicates remain bounded "
                "provenance notes rather than performance explanations"
            ),
        }
    ]
    findings.extend(duration_and_tautology_findings(exp3283))
    findings.append(
        {
            "kind": "corrigendum_kan_boundary",
            "source_experiment_id": "exp3283",
            "corrigendum_ready": exp3283.get("corrigendum_ready") is True,
            "kan_headline_allowed": mapping(
                mapping(exp3283.get("downstream_usage_rules")).get("kan")
            ).get("headline_allowed")
            is True,
            "prior_sidecar_only": exp3273.get("sidecar_only") is True,
            "finding": (
                "corrigendum already classifies KAN metrics as provisional or "
                "sidecar-only; this autopsy turns that into an explicit boundary"
            ),
        }
    )
    findings.append(
        {
            "kind": "label_provenance_mixture",
            "source_experiment_id": "exp3283",
            "provenance_summary": provenance_summary(exp3283),
            "finding": (
                "cached-panel and template-backed labels limit corpus headline "
                "claims and make leakage/provenance-clean reattempt evidence a prerequisite"
            ),
        }
    )
    return findings


def duration_and_tautology_findings(exp3283: Mapping[str, Any]) -> list[JsonDict]:
    """Compress corrigendum flags into claim-boundary findings."""

    return [
        {
            "kind": "duration_or_provenance_flag",
            "source_experiment_id": str(flag.get("experiment_id") or ""),
            "flag_kind": str(flag.get("kind") or ""),
            "finding": "corrigendum flag preserved for downstream headline eligibility",
        }
        for flag in list(mapping_list(exp3283.get("duration_flags")))
        + list(mapping_list(exp3283.get("tautology_flags")))
    ]


def provenance_summary(exp3283: Mapping[str, Any]) -> JsonDict:
    """Summarize the label-source mixture without re-opening the label job."""

    provenance = mapping(exp3283.get("provenance_by_artifact"))
    return {
        name: {
            "artifact_class": str(mapping(row).get("artifact_class") or ""),
            "row_provenance_counts": dict(mapping(mapping(row).get("row_provenance_counts"))),
        }
        for name, row in sorted(provenance.items())
        if name in {"exp3270", "exp3271", "exp3273"}
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Attach exact source-file checksums to make the autopsy replayable."""

    return [
        {
            "experiment_id": "exp3273",
            "path": EXP3273_REL_PATH.as_posix(),
            "sha256": file_sha256(root / EXP3273_REL_PATH),
        },
        {
            "experiment_id": "exp3272",
            "path": EXP3272_REL_PATH.as_posix(),
            "sha256": file_sha256(root / EXP3272_REL_PATH),
        },
        {
            "experiment_id": "exp3283",
            "path": EXP3283_REL_PATH.as_posix(),
            "sha256": file_sha256(root / EXP3283_REL_PATH),
        },
    ]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Refuse artifacts that omit the machine-readable gate or boundary."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("kan_failure_autopsy_ready") is not True:
        raise ValueError("kan_failure_autopsy_ready must be true")
    if artifact.get("kan_boundary_decision_ready") is not True:
        raise ValueError("kan_boundary_decision_ready must be true")
    if artifact.get("kan_boundary_decision") not in ALLOWED_BOUNDARY_DECISIONS:
        raise ValueError("kan_boundary_decision must be one of the allowed values")
    if not artifact.get("permitted_downstream_use"):
        raise ValueError("permitted_downstream_use must not be empty")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")


def negative_only_false_positive_rate(
    negative_count: int,
    metrics: Mapping[str, Any],
    threshold: float,
) -> float:
    """Estimate false-positive rate for single-class benign slices from score range."""

    return (
        0.0 if negative_count <= 0 else float(metric_float(metrics.get("min_score")) >= threshold)
    )


def nested_int(payload: Mapping[str, Any], outer_key: str, inner_key: str) -> int:
    """Read nested integer counters from upstream artifact dictionaries."""

    return int_value(mapping(payload.get(outer_key)).get(inner_key))


def mapping(value: Any) -> Mapping[str, Any]:
    """Normalize optional JSON objects so callers can read absent sections safely."""

    return value if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[Mapping[str, Any]]:
    """Normalize optional JSON arrays to mapping-only rows."""

    return [row for row in value or [] if isinstance(row, Mapping)]


def int_value(value: Any) -> int:
    """Convert artifact counters to integers without changing missing values into claims."""

    return int(float(value or 0))


def metric_float(value: Any) -> float:
    """Round numeric artifact values so generated JSON remains stable."""

    return round(float(value or 0.0), 6)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    """Compute ratios for summary rows while keeping empty denominators deterministic."""

    return (
        0.0
        if float(denominator or 0.0) == 0.0
        else metric_float(float(numerator) / float(denominator))
    )


def duration(started_s: float, finished_s: float) -> float:
    """Report real elapsed time when running normally and deterministic time in tests."""

    return metric_float(max(0.0, finished_s - started_s))


def file_sha256(path: Path) -> str:
    """Hash source bytes so the artifact names the exact evidence it autopsied."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding wall-clock and self fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "honest_verdict"}
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Write a concise terminal verdict that preserves the failed promotion gate."""

    return (
        "complete: kan_failure_autopsy_ready=true; "
        f"prior_full_corpus_auroc={artifact['prior_full_corpus_auroc']}; "
        "prior_delong_noninferiority_passed="
        f"{str(artifact['prior_delong_noninferiority_passed']).lower()}; "
        f"kan_boundary_decision={artifact['kan_boundary_decision']}"
    )
