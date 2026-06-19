"""Tests for the ARC artifact discipline lint.

Spec refs: REQ-VERIFY-4437, SCENARIO-VERIFY-4437.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.agentic import arc_solve_artifact_discipline as discipline
from scripts import arc_artifact_lint as lint


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_req_verify_4437_discovers_arc_solve_and_config_rule_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-4437: lint discovery targets ARC/solve/config-rule result paths."""

    results = tmp_path / "results"
    arc_path = _write_json(
        results / "experiment_9001_arc_solve.json",
        {
            "honest_verdict": "complete: offline_replay",
            "duration_s": 0.01,
            "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
        },
    )
    config_path = _write_json(
        results / "nested" / "experiment_9002_config_rule.json",
        {
            "honest_verdict": "complete: config_rule_replay",
            "duration_s": 0.01,
            "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
        },
    )
    _write_json(
        results / "experiment_9003_unrelated.json",
        {"honest_verdict": "complete: unrelated", "duration_s": 0.01},
    )

    discovered = lint.discover_candidate_artifacts(results)

    assert discovered == [arc_path, config_path]
    assert lint.lint_results_dir(results) == []


def test_scenario_verify_4437_flags_missing_substrate_and_partial_verdict(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4437: missing substrate and partial verdicts fail lint."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_9004_config_rule_solve.json",
        {
            "honest_verdict": "partial: config_rule_not_done",
            "duration_s": 0.01,
        },
    )

    issues = lint.lint_paths([artifact])
    kinds = {issue.kind for issue in issues}

    assert "MISSING_INFERENCE_SUBSTRATE" in kinds
    assert "NON_TERMINAL_PARTIAL_VERDICT" in kinds
    assert all(issue.path == artifact for issue in issues)


def test_scenario_verify_4437_live_llm_requires_allowlist_and_duration_floor(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4437: live ARC induction is explicit and allow-listed."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_9005_arc_live_solve.json",
        {
            "honest_verdict": "success: live_llm_induction_solved",
            "duration_s": 61.0,
            "inference_substrate": discipline.LIVE_LLM_SUBSTRATE,
        },
    )

    blocked = lint.lint_paths([artifact])
    assert [issue.kind for issue in blocked] == ["LIVE_LLM_NOT_ALLOWLISTED"]

    allowed = lint.lint_paths([artifact], allow_live_artifacts=[artifact])
    assert allowed == []

    too_short = _write_json(
        tmp_path / "results" / "experiment_9006_arc_live_solve.json",
        {
            "honest_verdict": "success: live_llm_induction_claimed",
            "duration_s": 5.0,
            "inference_substrate": discipline.LIVE_LLM_SUBSTRATE,
        },
    )
    short_issues = lint.lint_paths([too_short], allow_live_artifacts=[too_short])
    assert [issue.kind for issue in short_issues] == ["DURATION_BELOW_SUBSTRATE_FLOOR"]


def test_req_verify_4437_lint_main_returns_nonzero_and_json_report(
    tmp_path: Path,
    capsys,
) -> None:
    """REQ-VERIFY-4437: CLI emits machine-readable failures for conductor use."""

    results = tmp_path / "results"
    _write_json(
        results / "experiment_9007_arc_solve.json",
        {
            "honest_verdict": "complete: offline_replay",
            "duration_s": 0.01,
            "inference_substrate": "offline_arc_solver_kit_reproduce_no_3090",
        },
    )

    exit_code = lint.main(["--results-dir", str(results), "--json"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["ok"] is False
    assert output["issue_count"] == 1
    assert output["issues"][0]["kind"] == "INVALID_INFERENCE_SUBSTRATE"

    fixed = _write_json(
        results / "experiment_9008_arc_solve.json",
        {
            "honest_verdict": "complete: offline_replay",
            "duration_s": 0.01,
            "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
        },
    )
    assert lint.main(["--json", str(fixed)]) == 0
    ok_output = json.loads(capsys.readouterr().out)
    assert ok_output == {"ok": True, "issue_count": 0, "issues": []}
