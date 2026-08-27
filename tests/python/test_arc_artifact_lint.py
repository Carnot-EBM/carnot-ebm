"""Tests for the ARC artifact discipline lint.

Spec refs: REQ-VERIFY-4437, SCENARIO-VERIFY-4437.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from carnot.agentic import arc_solve_artifact_discipline as discipline
from scripts import arc_artifact_lint as lint


REPO = Path(__file__).resolve().parents[2]
EXP4433_PATH = REPO / "results" / "experiment_4433_example_conditioned_win_induction.json"
PRE_COMMIT_PATH = REPO / ".pre-commit-config.yaml"
ALLOWLIST_PATH = REPO / "ops" / "arc_artifact_live_allowlist.txt"


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


def test_req_verify_4437_lint_accepts_arc_live_agent_no_llm_substrate(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4437: no-LLM live ARC receipts pass the ARC artifact lint."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_9005_arc_live_patch_receipt.json",
        {
            "honest_verdict": "complete: level_delta=0 patch_retired",
            "duration_s": 0.02,
            "inference_substrate": discipline.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
            "solve_provenance": "live_agent_self_discovery",
            "target_game": "zz99_exp5253_live_receipt_probe",
            "reproduction_gate": {"reproduced": False},
        },
    )

    assert lint.lint_paths([artifact]) == []


def test_req_arc_wmte_6681_lint_accepts_canonical_outcome_transport_substrate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6681: canonical live outcome transport passes ARC lint."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_6681_arc_post_redirect_outcomes.json",
        {
            "honest_verdict": "complete: exact live outcomes joined",
            "duration_s": 0.02,
            "inference_substrate": (discipline.ARC_CANONICAL_OUTCOME_TRANSPORT_NO_LLM_SUBSTRATE),
            "solve_claim_scope": "none",
        },
    )

    assert lint.lint_paths([artifact]) == []


def test_req_arc_wmte_6682_lint_accepts_supervisor_ab_substrate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6682: held-family supervisor A/B passes ARC lint."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_6682_arc_held_family_supervisor_ab.json",
        {
            "honest_verdict": "complete: exact held-family comparison measured",
            "duration_s": 0.02,
            "inference_substrate": discipline.ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE,
            "solve_claim_scope": "none",
        },
    )

    assert lint.lint_paths([artifact]) == []


def test_req_verify_4437_lint_accepts_arc_supervisor_receipt_replay_substrate(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4437: Exp6524 blocked no-solve receipt replay passes ARC lint."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_6524_arc_supervisor_redirect_generalization.json",
        {
            "honest_verdict": "blocked: missing outcome-bearing live trajectory-supervisor receipts",
            "duration_s": 0.02,
            "inference_substrate": discipline.ARC_SUPERVISOR_RECEIPT_REPLAY_SUBSTRATE,
            "verifier_is_oracle": False,
            "solve_claim": False,
        },
    )

    assert lint.lint_paths([artifact]) == []


def test_req_verify_4437_lint_accepts_principle_wrapped_arc_fields(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4437: principle-wrapped ARC receipt fields still lint cleanly."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_9005_arc_wrapped_receipt.json",
        {
            "honest_verdict": {
                "value": "complete: level_delta=0 patch_decision=retire",
                "principle": "terminal-prefixed",
            },
            "duration_s": {"value": 0.02, "principle": "measured wall-clock"},
            "inference_substrate": {
                "value": discipline.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
                "principle": "canonical substrate",
            },
            "target_game": "zz99_exp5253_live_receipt_probe",
            "reproduction_gate": {"reproduced": False},
        },
    )

    assert lint.lint_paths([artifact]) == []


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


def test_scenario_verify_4450_exp4433_class_missing_substrate_is_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4450: exp4433-class ARC solves cannot omit substrate."""

    payload = json.loads(EXP4433_PATH.read_text(encoding="utf-8"))
    payload["inference_substrate"] = None
    artifact = _write_json(
        tmp_path / "results" / EXP4433_PATH.name,
        payload,
    )

    issues = lint.lint_paths([artifact])
    issue_kinds = {issue.kind for issue in issues}

    assert "MISSING_INFERENCE_SUBSTRATE" in issue_kinds
    assert any("inference_substrate" in issue.detail for issue in issues)


def test_scenario_verify_4450_exp4433_class_passes_with_declared_substrate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4450: the same ARC solve passes with a valid substrate."""

    payload = json.loads(EXP4433_PATH.read_text(encoding="utf-8"))
    payload["inference_substrate"] = discipline.AGGREGATION_SUBSTRATE
    artifact = _write_json(
        tmp_path / "results" / EXP4433_PATH.name,
        payload,
    )

    assert (
        payload["duration_s"]
        >= discipline.SUBSTRATE_DURATION_FLOORS[discipline.AGGREGATION_SUBSTRATE]
    )
    assert lint.lint_paths([artifact]) == []


def test_req_verify_4450_cli_accepts_live_allowlist_file(
    tmp_path: Path,
    capsys,
) -> None:
    """REQ-VERIFY-4450: legitimately live ARC artifacts use an allow-list file."""

    artifact = _write_json(
        tmp_path / "results" / "experiment_9010_arc_live_solve.json",
        {
            "honest_verdict": "success: live_llm_induction_solved",
            "duration_s": 61.0,
            "inference_substrate": discipline.LIVE_LLM_SUBSTRATE,
        },
    )
    allowlist = tmp_path / "arc_live_allowlist.txt"
    allowlist.write_text(f"{artifact}\n", encoding="utf-8")

    blocked_exit = lint.main(["--json", str(artifact)])
    blocked_report = json.loads(capsys.readouterr().out)
    allowed_exit = lint.main(["--allow-live-file", str(allowlist), "--json", str(artifact)])
    allowed_report = json.loads(capsys.readouterr().out)

    assert blocked_exit == 1
    assert blocked_report["issues"][0]["kind"] == "LIVE_LLM_NOT_ALLOWLISTED"
    assert allowed_exit == 0
    assert allowed_report == {"ok": True, "issue_count": 0, "issues": []}


def test_req_verify_4450_precommit_hook_is_scoped_and_allowlisted() -> None:
    """REQ-VERIFY-4450: pre-commit runs the ARC lint on result artifacts."""

    config = PRE_COMMIT_PATH.read_text(encoding="utf-8")
    hook_block = config.split("- id: arc-artifact-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)

    assert "scripts/arc_artifact_lint.py" in hook_block
    assert f"--allow-live-file {ALLOWLIST_PATH.relative_to(REPO).as_posix()}" in hook_block
    assert files_match is not None
    files_re = re.compile(files_match.group(1))
    assert files_re.search("results/experiment_9011_arc_solve.json")
    assert files_re.search("results/experiment_9012_config_rule_solve.json")
    assert files_re.search("results/nested/experiment_9013_world_model_report.json")
    assert not files_re.search("results/experiment_9014_capstone.json")
    assert ALLOWLIST_PATH.exists()


def test_req_verify_4450_lint_guard_edge_paths_assert(
    tmp_path: Path,
    capsys,
) -> None:
    """REQ-VERIFY-4450: candidate detection and allow-list edge paths assert."""

    missing_results = tmp_path / "missing-results"
    noncandidate = _write_json(
        tmp_path / "results" / "experiment_9015_unrelated.json",
        {"honest_verdict": "complete: unrelated", "duration_s": 0.01},
    )
    tagged_candidate = _write_json(
        tmp_path / "results" / "experiment_9016_metadata_tagged.json",
        {
            "tags": ["arc"],
            "honest_verdict": "complete: tagged_arc_artifact",
            "duration_s": 0.01,
            "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
        },
    )
    live_artifact = _write_json(
        tmp_path / "results" / "experiment_9017_arc_live_solve.json",
        {
            "honest_verdict": "success: live_llm_induction_solved",
            "duration_s": 61.0,
            "inference_substrate": discipline.LIVE_LLM_SUBSTRATE,
        },
    )
    nonpartial_bad_verdict = {
        "honest_verdict": "ongoing",
        "duration_s": 0.01,
        "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
    }

    assert lint.discover_candidate_artifacts(missing_results) == []
    assert lint.lint_paths([noncandidate]) == []
    assert lint.lint_paths([tagged_candidate]) == []
    assert [
        issue.kind for issue in lint.lint_artifact(tagged_candidate, nonpartial_bad_verdict)
    ] == ["NON_TERMINAL_HONEST_VERDICT"]
    assert (
        lint.main(
            [
                "--allow-live-file",
                str(tmp_path / "does-not-exist.txt"),
                "--json",
                str(live_artifact),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["issues"][0]["kind"] == "LIVE_LLM_NOT_ALLOWLISTED"
