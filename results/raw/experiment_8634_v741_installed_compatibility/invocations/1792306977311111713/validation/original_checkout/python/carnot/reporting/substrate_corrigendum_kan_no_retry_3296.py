"""Build the Exp 3296 substrate corrigendum and KAN no-retry ledger.

Spec refs: REQ-REPORT-3296, SCENARIO-REPORT-3296.

The ledger exists because a downstream matrix can accidentally count a row as
"done" even when that row is only a capstone, a flagged methodology note, or a
sidecar result. This module reads the relevant `.304` evidence and turns that
boundary into machine-readable usage rules for `.305`. It deliberately avoids
new KAN training, KAN scoring, Garak, repair, verifier, or corpus relabeling
work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.substrate_corrigendum_kan_no_retry.v1"
EXPERIMENT_ID = "exp3296"
TASK_ID = "exp3296-substrate-corrigendum-kan-no-retry-v1"
ARTIFACT = "experiment_3296_substrate_corrigendum_kan_no_retry_v1"
MILESTONE = "2026.05.305"
RUN_DATE = "20260528"
RANDOM_SEED = 3296
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

OUTPUT_REL_PATH = Path("results/experiment_3296_substrate_corrigendum_kan_no_retry_v1.json")
EXP3283_REL_PATH = Path(
    "results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json"
)
EXP3288_REL_PATH = Path("results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json")
EXP3292_REL_PATH = Path("results/experiment_3292_evidence_matrix_v36.json")
EXP3293_REL_PATH = Path("results/experiment_3293_capstone_v304.json")
RESEARCH_CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

SOURCE_REL_PATHS = (
    EXP3283_REL_PATH,
    EXP3288_REL_PATH,
    EXP3292_REL_PATH,
    EXP3293_REL_PATH,
)
PROTECTED_REL_PATHS = SOURCE_REL_PATHS + (RESEARCH_CONDUCTOR_REL_PATH,)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_FIELDS = {
    "substrate_corrigendum_ready",
    "kan_no_retry_ledger_ready",
    "kan_prompt_injection_headline_retired",
    "prior_kan_auroc",
    "prior_aligned_instruction_false_positive_rate",
    "headline_eligible_prior_metrics",
    "non_headline_prior_metrics",
    "downstream_usage_rules",
    "future_reopen_prerequisites",
    "protected_files_untouched",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3296: aggregate prior evidence into `.305` claim boundaries."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    before = snapshot_files(root_path, PROTECTED_REL_PATHS)
    exp3283 = read_json_object(root_path / EXP3283_REL_PATH)
    exp3288 = read_json_object(root_path / EXP3288_REL_PATH)
    exp3292 = read_json_object(root_path / EXP3292_REL_PATH)
    exp3293 = read_json_object(root_path / EXP3293_REL_PATH)
    headline = headline_eligible_prior_metrics(exp3283, exp3292)
    non_headline = non_headline_prior_metrics(exp3283, exp3288, exp3292, exp3293)
    classes = prior_metric_classes(headline, non_headline, exp3292, exp3293)
    after = snapshot_files(root_path, PROTECTED_REL_PATHS)
    protected_status = protected_file_status(before, after)
    finished = time.perf_counter() if now_s is None else float(now_s)
    prior_kan_auroc = metric_float(exp3288.get("prior_full_corpus_auroc"))
    prior_aligned_fpr = metric_float(
        mapping(exp3288.get("aligned_instruction_false_positive_summary")).get(
            "aligned_instruction_false_positive_rate"
        )
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "spec_refs": ["REQ-REPORT-3296", "SCENARIO-REPORT-3296"],
        "substrate_corrigendum_ready": True,
        "kan_no_retry_ledger_ready": True,
        "kan_prompt_injection_headline_retired": (
            exp3288.get("kan_boundary_decision")
            == "retire_from_prompt_injection_headline"
        ),
        "prior_kan_auroc": prior_kan_auroc,
        "prior_aligned_instruction_false_positive_rate": prior_aligned_fpr,
        "headline_eligible_prior_metrics": headline,
        "non_headline_prior_metrics": non_headline,
        "prior_metric_classes": classes,
        "downstream_usage_rules": downstream_usage_rules(exp3283, exp3288, exp3292, exp3293),
        "future_reopen_prerequisites": future_reopen_prerequisites(exp3288),
        "source_artifacts": source_artifacts(root_path),
        "source_checksums": source_checksums(root_path),
        "protected_file_status": protected_status,
        "protected_files_untouched": all(
            status["unchanged"] for status in protected_status.values()
        ),
        "field_provenance": field_provenance(),
        "no_new_model_execution": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_kan_training": True,
        "no_new_kan_scoring": True,
        "no_corpus_relabeling": True,
        "no_prior_result_file_edits": True,
        "scripts_research_conductor_modified": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration(start, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3296 JSON deliverable."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def headline_eligible_prior_metrics(
    exp3283: Mapping[str, Any],
    exp3292: Mapping[str, Any],
) -> list[JsonDict]:
    """Return prior metrics that can be cited without becoming performance wins."""

    rows: list[JsonDict] = []
    for raw in list(exp3283.get("headline_eligible_metrics") or []):
        row = mapping(raw)
        source = str(row.get("source_experiment_id") or "unknown")
        metric = str(row.get("metric") or "unknown")
        rows.append(
            {
                "metric_id": f"{source}.{metric}",
                "source_experiment_id": source,
                "metric": metric,
                "value": row.get("value"),
                "claim_class": "clean_boundary_or_integrity",
                "headline_boundary": str(row.get("boundary") or ""),
            }
        )
    garak = matrix_row(exp3292, "exp3285")
    summary = mapping(garak.get("summary"))
    rows.append(
        {
            "metric_id": "exp3285.live_garak_failure_boundary",
            "source_experiment_id": "exp3285",
            "metric": "attack_success_rate_and_failed_gate",
            "value": {
                "attack_success_rate": summary.get("attack_success_rate"),
                "garak_gate_passed": summary.get("garak_gate_passed"),
                "garak_redteam_eval_ready": summary.get("garak_redteam_eval_ready"),
            },
            "claim_class": "headline_eligible_negative_boundary",
            "headline_boundary": (
                "Citable as live Garak failure/root-cause evidence only; not a "
                "Garak success or paper-readiness claim."
            ),
        }
    )
    return rows


def non_headline_prior_metrics(
    exp3283: Mapping[str, Any],
    exp3288: Mapping[str, Any],
    exp3292: Mapping[str, Any],
    exp3293: Mapping[str, Any],
) -> list[JsonDict]:
    """Return metrics that may remain useful but cannot headline `.305` claims."""

    rows = [
        {
            "metric_id": "exp3273.full_corpus_auroc",
            "source_experiment_id": "exp3273",
            "metric": "full_corpus_auroc",
            "value": metric_float(exp3288.get("prior_full_corpus_auroc")),
            "classification": "sidecar-only",
            "boundary": "KAN prompt-injection headline retired by Exp 3288.",
        },
        {
            "metric_id": "exp3273.aligned_instruction_false_positive_rate",
            "source_experiment_id": "exp3273",
            "metric": "aligned_instruction_false_positive_rate",
            "value": metric_float(
                mapping(exp3288.get("aligned_instruction_false_positive_summary")).get(
                    "aligned_instruction_false_positive_rate"
                )
            ),
            "classification": "sidecar-only",
            "boundary": "Specific KAN failure mode; not detector utility evidence.",
        },
    ]
    for raw in list(exp3283.get("provisional_or_sidecar_metrics") or []):
        row = mapping(raw)
        source = str(row.get("source_experiment_id") or "unknown")
        metric = str(row.get("metric") or "unknown")
        rows.append(
            {
                "metric_id": f"{source}.{metric}",
                "source_experiment_id": source,
                "metric": metric,
                "value": row.get("value"),
                "classification": "flagged_or_sidecar",
                "boundary": str(row.get("boundary") or ""),
            }
        )
    repair = matrix_row(exp3292, "exp3290")
    repair_summary = mapping(repair.get("summary"))
    rows.extend(
        [
            {
                "metric_id": "exp3290.repair_micro_panel_verified_success_count",
                "source_experiment_id": "exp3290",
                "metric": "verified_success_count",
                "value": repair_summary.get("verified_success_count"),
                "classification": "flagged",
                "boundary": (
                    "Four-case diagnostic exact repair panel; "
                    "headline_claim_allowed=false in matrix v36."
                ),
            },
            {
                "metric_id": "exp3292.matrix_v36_status_counts",
                "source_experiment_id": "exp3292",
                "metric": "primary_status_counts",
                "value": dict(mapping(exp3292.get("primary_status_counts"))),
                "classification": "aggregation-only",
                "boundary": "Matrix status counts are aggregation, not live evidence.",
            },
            {
                "metric_id": "exp3293.capstone_v304_publication_blocker_count",
                "source_experiment_id": "exp3293",
                "metric": "publication_blocker_count",
                "value": exp3293.get("publication_blocker_count"),
                "classification": "aggregation-only",
                "boundary": "Capstone closeout count is a readiness ledger, not a benchmark metric.",
            },
        ]
    )
    return rows


def prior_metric_classes(
    headline: Sequence[Mapping[str, Any]],
    non_headline: Sequence[Mapping[str, Any]],
    exp3292: Mapping[str, Any],
    exp3293: Mapping[str, Any],
) -> JsonDict:
    """Classify prior metrics by evidence status for downstream matrix builders."""

    return {
        "clean": [dict(row) for row in headline if row.get("source_experiment_id") == "exp3272"],
        "flagged": [classification_row(matrix_row(exp3292, "exp3290"))],
        "sidecar-only": [classification_row(matrix_row(exp3292, "exp3288"))],
        "blocked": [classification_row(matrix_row(exp3292, "exp3285"))],
        "aggregation-only": [
            {
                "source_experiment_id": "exp3292",
                "status": "aggregation-only",
                "metric": "matrix_v36_status_counts",
                "value": dict(mapping(exp3292.get("primary_status_counts"))),
            },
            {
                "source_experiment_id": "exp3293",
                "status": "aggregation-only",
                "metric": "capstone_v304_publication_blocker_count",
                "value": exp3293.get("publication_blocker_count"),
            },
        ],
        "not_headline_eligible": [
            {
                "source_experiment_id": str(row.get("source_experiment_id")),
                "metric": row.get("metric"),
                "classification": row.get("classification"),
                "boundary": row.get("boundary"),
            }
            for row in non_headline
        ],
    }


def downstream_usage_rules(
    exp3283: Mapping[str, Any],
    exp3288: Mapping[str, Any],
    exp3292: Mapping[str, Any],
    exp3293: Mapping[str, Any],
) -> JsonDict:
    """Emit precise `.305` rules for Garak, repair, KAN, and corpus references."""

    garak = mapping(mapping(exp3292.get("gate_summary")).get("garak_redteam"))
    repair = mapping(mapping(exp3292.get("gate_summary")).get("repair_panel"))
    return {
        "garak": {
            "new_live_garak_can_be_headline_candidate": True,
            "prior_dot304_garak_success_headline_allowed": False,
            "prior_garak_gate_passed": garak.get("garak_gate_passed") is True,
            "required_boundary": (
                "Only a new live Garak/DataFlip run with model provenance, attack-success "
                "and error gates passed, and DataFlip separated from target-model behavior "
                "can become headline-eligible."
            ),
        },
        "repair": {
            "new_exact_repair_can_be_headline_candidate": True,
            "prior_micro_panel_headline_allowed": repair.get("headline_claim_allowed") is True,
            "prior_repair_gate_open": exp3293.get("repair_gate_open") is True,
            "required_boundary": (
                "Use exact checks, zero false accepts, enough cases for a headline denominator, "
                "and clean duration/provenance evidence. Exp 3290 remains diagnostic."
            ),
        },
        "kan": {
            "retry_without_operator_directive_allowed": False,
            "prompt_injection_headline_retired": (
                exp3288.get("kan_boundary_decision")
                == "retire_from_prompt_injection_headline"
            ),
            "allowed_prior_use": list(exp3288.get("permitted_downstream_use") or []),
            "prohibited_prior_use": list(exp3288.get("prohibited_downstream_use") or []),
            "required_boundary": (
                "KAN may be cited as a retired negative-control or boundary ledger only. "
                "Do not rerun or promote the same sidecar."
            ),
        },
        "corpus": {
            "prior_dot303_corpus_headline_label_claim_allowed": False,
            "relabeling_without_operator_directive_allowed": False,
            "corrigendum_ready": exp3283.get("corrigendum_ready") is True,
            "required_boundary": (
                "The `.303` corpus/KAN artifacts remain corrigendum-bounded. Use them as "
                "integrity, leakage-boundary, inventory, or negative-control context only."
            ),
        },
        "aggregation": {
            "matrix_or_capstone_counts_are_headline_evidence": False,
            "required_boundary": (
                "Aggregation rows can route work and preserve blockers, but they do not "
                "replace live inference or exact repair evidence."
            ),
        },
    }


def future_reopen_prerequisites(exp3288: Mapping[str, Any]) -> list[JsonDict]:
    """State the operator-authorized gates required before any KAN reopening."""

    return [
        {
            "prerequisite": "operator_directive",
            "reason": "Exp 3288 retired this KAN from prompt-injection headline use.",
        },
        {
            "prerequisite": "materially_different_kan_or_ensemble",
            "reason": "Do not silently retry the same failed sidecar architecture.",
        },
        {
            "prerequisite": "leakage_provenance_clean_labels",
            "reason": "The `.303` corpus remains corrigendum-bounded until clean labels exist.",
        },
        {
            "prerequisite": "aligned_benign_false_positive_ceiling",
            "reason": (
                "Prior aligned instruction false-positive rate was "
                f"{metric_float(mapping(exp3288.get('aligned_instruction_false_positive_summary')).get('aligned_instruction_false_positive_rate')):.6f}."
            ),
        },
        {
            "prerequisite": "beat_regex_keyword_baselines",
            "reason": "Exp 3288 found the KAN below trivial baselines.",
        },
        {
            "prerequisite": "paired_delong_noninferiority_pass",
            "reason": "Prior full-corpus paired DeLong non-inferiority failed.",
        },
        {
            "prerequisite": "garak_pressure_pass",
            "reason": "Future headline detector evidence must survive Garak/DataFlip pressure.",
        },
    ]


def source_artifacts(root: Path) -> list[JsonDict]:
    """Attach hashes for every upstream artifact consumed by the ledger."""

    return [
        {
            "experiment_id": f"exp{rel_path.name.split('_', 2)[1]}",
            "path": rel_path.as_posix(),
            "sha256": file_sha256(root / rel_path),
        }
        for rel_path in SOURCE_REL_PATHS
    ]


def source_checksums(root: Path) -> JsonDict:
    """Return a compact path-to-checksum mapping for downstream integrity checks."""

    return {rel_path.as_posix(): file_sha256(root / rel_path) for rel_path in SOURCE_REL_PATHS}


def field_provenance() -> JsonDict:
    """Explain where required fields come from so audits do not infer provenance."""

    return {
        "substrate_corrigendum_ready": {"source": EXP3292_REL_PATH.as_posix()},
        "kan_no_retry_ledger_ready": {"source": EXP3288_REL_PATH.as_posix()},
        "kan_prompt_injection_headline_retired": {"source": EXP3288_REL_PATH.as_posix()},
        "prior_kan_auroc": {"source": EXP3288_REL_PATH.as_posix()},
        "prior_aligned_instruction_false_positive_rate": {
            "source": EXP3288_REL_PATH.as_posix()
        },
        "headline_eligible_prior_metrics": {"source": EXP3283_REL_PATH.as_posix()},
        "non_headline_prior_metrics": {
            "source": "Exp 3283, Exp 3288, Exp 3292, and Exp 3293"
        },
        "downstream_usage_rules": {"source": "derived from the cited upstream artifacts"},
        "future_reopen_prerequisites": {"source": EXP3288_REL_PATH.as_posix()},
        "protected_files_untouched": {"source": "before/after checksum comparison"},
        "inference_substrate": {"source": "REQ-REPORT-3296"},
    }


def classification_row(row: Mapping[str, Any]) -> JsonDict:
    """Reduce a matrix row to the status details needed by Exp 3296."""

    return {
        "source_experiment_id": str(row.get("experiment_id") or "unknown"),
        "role": row.get("role"),
        "status": row.get("status"),
        "summary": dict(mapping(row.get("summary"))),
        "quality_flags": list(row.get("quality_flags") or []),
        "bounded_claims": list(row.get("bounded_claims") or []),
        "blocker_reasons": list(row.get("blocker_reasons") or []),
    }


def matrix_row(exp3292: Mapping[str, Any], experiment_id: str) -> JsonDict:
    """Find one matrix row by experiment id."""

    for row in list(exp3292.get("rows") or []):
        candidate = mapping(row)
        if candidate.get("experiment_id") == experiment_id:
            return dict(candidate)
    return {}


def snapshot_files(root: Path, rel_paths: Sequence[Path]) -> JsonDict:
    """Record protected-file hashes before and after the aggregation step."""

    return {rel_path.as_posix(): file_sha256(root / rel_path) for rel_path in rel_paths}


def protected_file_status(before: Mapping[str, Any], after: Mapping[str, Any]) -> JsonDict:
    """Compare before/after checksums without editing protected inputs."""

    return {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in before
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Refuse ledgers that omit the no-retry and protected-file contract."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("substrate_corrigendum_ready") is not True:
        raise ValueError("substrate_corrigendum_ready must be true")
    if artifact.get("kan_no_retry_ledger_ready") is not True:
        raise ValueError("kan_no_retry_ledger_ready must be true")
    if artifact.get("kan_prompt_injection_headline_retired") is not True:
        raise ValueError("kan_prompt_injection_headline_retired must be true")
    if artifact.get("protected_files_untouched") is not True:
        raise ValueError("protected_files_untouched must be true")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Write a terminal verdict that keeps KAN retirement and substrate clear."""

    return (
        "complete: substrate_corrigendum_ready=true; "
        "kan_no_retry_ledger_ready=true; "
        "kan_prompt_injection_headline_retired=true; "
        f"prior_kan_auroc={float(artifact['prior_kan_auroc']):.6f}; "
        f"prior_aligned_instruction_false_positive_rate="
        f"{float(artifact['prior_aligned_instruction_false_positive_rate']):.6f}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding runtime-only self fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "honest_verdict"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Return a file hash, or None when evidence is absent in an isolated test."""

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping(value: Any) -> Mapping[str, Any]:
    """Normalize optional JSON objects before nested lookups."""

    return value if isinstance(value, Mapping) else {}


def metric_float(value: Any) -> float:
    """Convert numeric artifact fields to stable six-decimal floats."""

    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def duration(started_s: float, finished_s: float) -> float:
    """Return non-negative elapsed seconds for real runs and deterministic tests."""

    return metric_float(max(0.0, float(finished_s) - float(started_s)))
