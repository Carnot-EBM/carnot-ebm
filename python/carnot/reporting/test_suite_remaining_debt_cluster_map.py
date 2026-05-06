"""Build the Exp 1426 remaining test-suite debt cluster map.

Spec: REQ-REPORT-034, SCENARIO-REPORT-034.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260506"
EXPERIMENT = "1426_test_suite_remaining_debt_cluster_map"
SCHEMA = "test_suite_remaining_debt_cluster_map_v1"
EXP1421_FILE = "experiment_1421_test_suite_execution_debt_v1.json"
DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1426_test_suite_remaining_debt_cluster_map.json"
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "failure_cluster_map_complete",
    "collection_clean_confirmed",
    "failure_clusters_identified",
    "next_cluster_recommended",
    "spec_coverage_debt_count",
    "commands_run",
    "honest_verdict",
}

_SPEC_COUNT_RE = re.compile(r"(\d+)\s+test\(s\)\s+missing spec traceability")

_RUNTIME_CLUSTER_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "cluster_id": "publication_paper_v5_integrity_assertions",
        "priority": 2,
        "failure_category": "paper v5 issue checks",
        "likely_owner": "publication artifact/reporting owners",
        "path_patterns": (
            "test_experiment_1182_paper_v5_medium_low_issues_11_18.py",
            "test_figure_integrity_audit.py",
        ),
        "bounded_fix_hint": (
            "Reconcile the affected publication audit fixtures with the current paper/artifact "
            "metadata, then run only the publication audit tests before broader validation."
        ),
    },
    {
        "cluster_id": "live_data_model_name_drift",
        "priority": 3,
        "failure_category": "stale live-data model-name assertions",
        "likely_owner": "live-data experiment fixtures",
        "path_patterns": (
            "test_experiment_578_live_data_a_v3.py",
            "test_experiment_579_live_data_c.py",
            "test_experiment_295_apple_verify_repair.py",
            "test_experiment_294_gpu_baseline_apple.py",
        ),
        "bounded_fix_hint": (
            "Update the stale expected model metadata or fixture source for live-data artifacts; "
            "do not run live models while repairing assertion drift."
        ),
    },
    {
        "cluster_id": "legacy_retro_schema_drift",
        "priority": 4,
        "failure_category": "older operational retro schema assertions",
        "likely_owner": "milestone retrospective/reporting owners",
        "path_patterns": (
            "test_retro_689.py",
            "test_retro_702.py",
            "test_experiment_805_milestone_retro.py",
            "test_experiment_867_milestone_retro.py",
            "test_experiment_337_retro.py",
        ),
        "bounded_fix_hint": (
            "Normalize the legacy retrospective fixtures or schema adapters in one retro-era "
            "slice, preserving historical artifacts rather than rewriting conclusions."
        ),
    },
    {
        "cluster_id": "arbiter_warmstart_expectations",
        "priority": 5,
        "failure_category": "arbiter warmstart expectations",
        "likely_owner": "multi-agent arbiter / Gibbs warmstart",
        "path_patterns": (
            "test_experiment_846_arbiter_warmstart.py",
            "test_experiment_822_arbiter_fix_v2_agent_auditor.py",
        ),
        "bounded_fix_hint": (
            "Re-run the arbiter warmstart tests in isolation, then decide whether the energy "
            "threshold or deterministic fixture has drifted."
        ),
    },
    {
        "cluster_id": "fpga_vitis_hls_environment_verdicts",
        "priority": 6,
        "failure_category": "Vitis/HLS verdict checks",
        "likely_owner": "FPGA/HLS experiment harnesses",
        "path_patterns": (
            "test_experiment_750_vitis_hls.py",
            "test_experiment_859_ice40_n8_combinational.py",
            "test_experiment_714_npu_iron.py",
            "test_experiment_1048_kv260_diagnosis.py",
        ),
        "bounded_fix_hint": (
            "Separate no-toolchain environment verdicts from synthesis-result assertions and "
            "verify the HLS verdict mapper without invoking vendor tools."
        ),
    },
    {
        "cluster_id": "gpu_live_model_memory_watchdog",
        "priority": 7,
        "failure_category": "GPU/live-model memory-watchdog errors",
        "likely_owner": "live model, GPU harness, and memory watchdog owners",
        "path_patterns": (
            "test_memory_watchdog.py",
            "test_gpu_acceleration.py",
            "test_experiment_368_precision_live.py",
            "test_experiment_431_eorm_jepa_retrain.py",
            "test_experiment_443_eorm_jepa_live_retrain.py",
            "test_experiment_772_semantic_energy_probe.py",
            "test_experiment_1029_fover_expansion_v2.py",
            "test_experiment_1031_energy_ssd_v3.py",
            "test_experiment_1043_fover_expansion_v3.py",
            "test_experiment_1163_nrgpt_energy.py",
        ),
        "bounded_fix_hint": (
            "Partition true memory-watchdog assertions from tests that accidentally load live "
            "model state, then force deterministic CPU/mock paths for unit scope."
        ),
    },
    {
        "cluster_id": "conductor_infrastructure_assertions",
        "priority": 8,
        "failure_category": "conductor/infrastructure assertions",
        "likely_owner": "research conductor infrastructure",
        "path_patterns": (
            "test_conductor_supervisor.py",
            "quarantine/test_conductor_supervisor.py",
            "test_infrastructure_hardening_v3.py",
            "test_failure_ledger.py",
            "test_failure_ledger_v2_cap.py",
            "test_pick_next_task_gate_block.py",
        ),
        "bounded_fix_hint": (
            "Fix infrastructure tests as a separate conductor-harness slice; this task did not "
            "edit scripts/research_conductor.py."
        ),
    },
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-034: create the bootstrap artifact before evidence is reconciled."""

    artifact = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "failure_cluster_map_complete": False,
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _normalize_test_path(value: object) -> str:
    text = str(value).strip()
    if text.startswith("- "):
        text = text[2:].strip()
    marker = "tests/python/"
    if marker in text:
        return text[text.index(marker) :]
    return text


def _parse_spec_coverage_debt_count(output: str) -> int | None:
    match = _SPEC_COUNT_RE.search(output)
    if match:
        return int(match.group(1))
    return None


def _group_spec_coverage_violations(lines: Sequence[object]) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        normalized = _normalize_test_path(line)
        parts = normalized.split("::", 1)
        if len(parts) == 2:
            grouped[parts[0]].append(normalized)
    return [
        {
            "path": path,
            "missing_count": len(tests),
            "representative_tests": tests[:5],
        }
        for path, tests in sorted(grouped.items())
    ]


def _match_representative_tests(
    candidates: Sequence[object],
    patterns: Sequence[object],
) -> list[str]:
    normalized_candidates = [_normalize_test_path(candidate) for candidate in candidates]
    matches = [
        candidate
        for candidate in normalized_candidates
        if any(str(pattern) in candidate for pattern in patterns)
    ]
    return sorted(dict.fromkeys(matches))[:5]


def _spec_coverage_cluster(
    debt_count: int,
    violations: Sequence[object],
    recommended: bool,
) -> dict[str, object]:
    groups = _group_spec_coverage_violations(violations)
    representative = [test for group in groups for test in group["representative_tests"]][:5]
    return {
        "cluster_id": "spec_coverage_traceability_metadata",
        "priority": 1,
        "failure_category": "spec coverage missing REQ/SCENARIO references",
        "evidence_source": "current scripts/check_spec_coverage.py output",
        "representative_tests": representative,
        "grouped_missing_reference_files": groups,
        "likely_owner": "test hygiene and owners of the listed test files",
        "count_estimate": debt_count,
        "bounded_fix_hint": (
            "Add accurate file-level Spec lines or per-test REQ/SCENARIO references in the "
            "listed files, then rerun scripts/check_spec_coverage.py before any full-suite "
            "retry."
        ),
        "recommended_next": recommended,
    }


def _runtime_cluster(
    template: Mapping[str, object],
    lastfailed_keys: Sequence[object],
) -> dict[str, object]:
    patterns = tuple(template["path_patterns"])  # type: ignore[arg-type]
    representative = _match_representative_tests(lastfailed_keys, patterns)
    count_estimate: int | str = (
        len(representative) if representative else "unknown_from_exp1421_summary"
    )
    return {
        "cluster_id": template["cluster_id"],
        "priority": template["priority"],
        "failure_category": template["failure_category"],
        "evidence_source": (
            "Exp1421 remaining_debt summary, with local pytest lastfailed cache used only "
            "for representative paths"
        ),
        "representative_tests": representative,
        "likely_owner": template["likely_owner"],
        "count_estimate": count_estimate,
        "bounded_fix_hint": template["bounded_fix_hint"],
        "recommended_next": False,
    }


def _exp1421_remaining_categories(exp1421: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in exp1421.get("remaining_debt", [])]


def build_artifact(
    *,
    exp1421: Mapping[str, Any],
    collection_clean_confirmed: bool,
    collection_outcome: str,
    spec_coverage_debt_count: int | None,
    spec_coverage_violations: Sequence[object],
    lastfailed_keys: Sequence[object],
    commands_run: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """REQ-REPORT-034: combine cheap diagnostics into one bounded debt map."""

    complete = collection_clean_confirmed and spec_coverage_debt_count is not None
    next_cluster = "spec_coverage_traceability_metadata" if complete else None
    clusters = [
        _spec_coverage_cluster(
            spec_coverage_debt_count or 0,
            spec_coverage_violations,
            recommended=next_cluster == "spec_coverage_traceability_metadata",
        )
    ]
    clusters.extend(
        _runtime_cluster(template, lastfailed_keys) for template in _RUNTIME_CLUSTER_TEMPLATES
    )
    if not complete:
        for cluster in clusters:
            cluster["recommended_next"] = False

    honest_verdict = (
        "diagnostic_cluster_map_blocked_collection_not_clean_or_spec_count_unknown"
        if not complete
        else (
            "diagnostic_cluster_map_complete_collection_clean_spec_coverage_red_"
            f"{spec_coverage_debt_count}_full_suite_not_rerun_exp1421_runtime_debt_partitioned"
        )
    )
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete" if complete else "blocked",
        "failure_cluster_map_complete": complete,
        "collection_clean_confirmed": collection_clean_confirmed,
        "collection_outcome": collection_outcome,
        "failure_clusters_identified": clusters,
        "next_cluster_recommended": next_cluster,
        "spec_coverage_debt_count": spec_coverage_debt_count,
        "commands_run": [dict(command) for command in commands_run],
        "full_suite_rerun": False,
        "full_suite_health_claimed": False,
        "source_artifacts_checked": [
            {
                "path": f"results/{EXP1421_FILE}",
                "exists": bool(exp1421),
                "honest_verdict": exp1421.get("honest_verdict"),
            }
        ],
        "exp1421_remaining_failure_categories": _exp1421_remaining_categories(exp1421),
        "fixed_cluster_excluded": "embedding_constraint_store",
        "honest_verdict": honest_verdict,
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    collection_clean_confirmed: bool,
    collection_outcome: str,
    spec_coverage_output: str,
    spec_coverage_violations: Sequence[object],
    lastfailed_keys: Sequence[object],
    commands_run: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """SCENARIO-REPORT-034: write bootstrap and terminal artifact from observed checks."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    spec_count = _parse_spec_coverage_debt_count(spec_coverage_output)
    artifact = build_artifact(
        exp1421=_read_json(root_path / "results" / EXP1421_FILE),
        collection_clean_confirmed=collection_clean_confirmed,
        collection_outcome=collection_outcome,
        spec_coverage_debt_count=spec_count,
        spec_coverage_violations=spec_coverage_violations,
        lastfailed_keys=lastfailed_keys,
        commands_run=commands_run,
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - CLI requires observed command inputs.
    raise SystemExit(
        "Use carnot.reporting.test_suite_remaining_debt_cluster_map.run(...) with "
        "explicit observed diagnostics."
    )
