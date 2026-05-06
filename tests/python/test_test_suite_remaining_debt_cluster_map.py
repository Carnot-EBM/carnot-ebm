"""Tests for the Exp 1426 remaining test-suite debt cluster map.

Spec: REQ-REPORT-034, SCENARIO-REPORT-034.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.test_suite_remaining_debt_cluster_map import (
    REQUIRED_ARTIFACT_FIELDS,
    _group_spec_coverage_violations,
    _normalize_test_path,
    _parse_spec_coverage_debt_count,
    _read_json,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _exp1421_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "collection_clean_confirmed": True,
        "execution_failures_after": [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "result": (
                    "interrupted after 2045.45s with 69 failed, 15454 passed, "
                    "66 skipped, 74 warnings, 4 errors"
                ),
            }
        ],
        "remaining_debt": [
            "Representative remaining full-suite failures include paper v5 issue checks, "
            "stale live-data model-name assertions, older operational retro schema assertions, "
            "arbiter warmstart expectations, Vitis/HLS verdict checks, GPU/live-model "
            "memory-watchdog errors, and conductor/infrastructure assertions."
        ],
        "honest_verdict": (
            "focused_embedding_store_runtime_failures_fixed_collection_clean_targeted_tests_green_"
            "100pct_store_coverage_full_suite_and_preexisting_spec_coverage_debt_remain"
        ),
    }


def _commands() -> list[dict[str, str]]:
    return [
        {
            "command": ".venv/bin/pytest tests/python --collect-only -q --no-header "
            "--disable-warnings --no-cov",
            "outcome": "22311 tests collected in 8.77s",
        },
        {
            "command": ".venv/bin/python scripts/check_spec_coverage.py",
            "outcome": "71 test(s) missing spec traceability",
        },
    ]


def test_scenario_report_034_builds_complete_cluster_map_without_full_suite_claim() -> None:
    """SCENARIO-REPORT-034: cheap current checks produce a bounded cluster map."""

    artifact = build_artifact(
        exp1421=_exp1421_payload(),
        collection_clean_confirmed=True,
        collection_outcome="22311 tests collected in 8.77s",
        spec_coverage_debt_count=71,
        spec_coverage_violations=[
            "tests/python/test_conductor_supervisor.py::test_heartbeat_fresh_no_alert",
            "tests/python/test_experiment_1297_sota_gguf_preflight.py::test_ready",
            "tests/python/test_experiment_972_kan_milp.py::test_pwa_segments_count",
        ],
        lastfailed_keys=[
            "tests/python/test_experiment_1182_paper_v5_medium_low_issues_11_18.py::test_issue_13",
            "tests/python/test_experiment_578_live_data_a_v3.py::TestBuildLiveDataArtifact::test_models_field",
            "tests/python/test_experiment_846_arbiter_warmstart.py::test_cold_start_vs_warm_start_energy_magnitude",
            "tests/python/test_experiment_750_vitis_hls.py::TestHonestVerdictLogic::test_verdict_hls_synthesized",
            "tests/python/test_memory_watchdog.py::test_watchdog_detects_leak",
        ],
        commands_run=_commands(),
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["failure_cluster_map_complete"] is True
    assert artifact["collection_clean_confirmed"] is True
    assert artifact["spec_coverage_debt_count"] == 71
    assert artifact["next_cluster_recommended"] == "spec_coverage_traceability_metadata"
    assert "full_suite_not_rerun" in artifact["honest_verdict"]
    assert not any(
        "embedding" in cluster["cluster_id"] for cluster in artifact["failure_clusters_identified"]
    )
    assert len(artifact["failure_clusters_identified"]) >= 8
    assert (
        sum(1 for cluster in artifact["failure_clusters_identified"] if cluster["recommended_next"])
        == 1
    )


def test_req_report_034_blocks_completion_when_collection_is_not_clean() -> None:
    """REQ-REPORT-034: collection must be clean before the map claims completion."""

    artifact = build_artifact(
        exp1421=_exp1421_payload(),
        collection_clean_confirmed=False,
        collection_outcome="collection failed before completion",
        spec_coverage_debt_count=71,
        spec_coverage_violations=[],
        lastfailed_keys=[],
        commands_run=_commands(),
    )

    assert artifact["status"] == "blocked"
    assert artifact["failure_cluster_map_complete"] is False
    assert artifact["collection_clean_confirmed"] is False
    assert artifact["next_cluster_recommended"] is None
    assert "collection_not_clean" in artifact["honest_verdict"]


def test_req_report_034_parses_spec_coverage_count_and_groups_violations(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-034: spec-coverage output is normalized into owner-sized groups."""

    output = """
ERROR: The following tests lack spec references (REQ-* or SCENARIO-*):
  - /repo/tests/python/test_conductor_supervisor.py::test_heartbeat_fresh_no_alert
  - /repo/tests/python/test_conductor_supervisor.py::test_heartbeat_missing_fires_alert
  - /repo/tests/python/test_experiment_972_kan_milp.py::test_pwa_segments_count

3 test(s) missing spec traceability.
"""

    assert _parse_spec_coverage_debt_count(output) == 3
    assert _parse_spec_coverage_debt_count("OK: All tests reference specifications.") is None
    assert _read_json(tmp_path / "missing.json") == {}
    assert _normalize_test_path("/repo/tests/python/test_conductor_supervisor.py::test_x") == (
        "tests/python/test_conductor_supervisor.py::test_x"
    )
    grouped = _group_spec_coverage_violations(output.splitlines())

    assert grouped == [
        {
            "path": "tests/python/test_conductor_supervisor.py",
            "missing_count": 2,
            "representative_tests": [
                "tests/python/test_conductor_supervisor.py::test_heartbeat_fresh_no_alert",
                "tests/python/test_conductor_supervisor.py::test_heartbeat_missing_fires_alert",
            ],
        },
        {
            "path": "tests/python/test_experiment_972_kan_milp.py",
            "missing_count": 1,
            "representative_tests": [
                "tests/python/test_experiment_972_kan_milp.py::test_pwa_segments_count"
            ],
        },
    ]


def test_req_report_034_run_writes_bootstrap_and_terminal_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-034: run writes the in-progress and complete JSON artifacts."""

    out_path = tmp_path / "results" / "experiment_1426_test_suite_remaining_debt_cluster_map.json"
    exp1421_path = tmp_path / "results" / "experiment_1421_test_suite_execution_debt_v1.json"
    exp1421_path.parent.mkdir(parents=True)
    exp1421_path.write_text(json.dumps(_exp1421_payload()), encoding="utf-8")

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        collection_clean_confirmed=True,
        collection_outcome="22311 tests collected in 8.77s",
        spec_coverage_output="71 test(s) missing spec traceability.",
        spec_coverage_violations=[
            "tests/python/test_figure_integrity_audit.py::test_audit_flags_untraced_constant"
        ],
        lastfailed_keys=["tests/python/test_conductor_supervisor.py::test_pid_file_written"],
        commands_run=_commands(),
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact == written
    assert written["status"] == "complete"
    assert written["failure_cluster_map_complete"] is True
    assert written["source_artifacts_checked"][0]["exists"] is True
