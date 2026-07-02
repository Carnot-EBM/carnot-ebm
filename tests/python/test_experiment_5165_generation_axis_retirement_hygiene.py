"""Tests for Exp 5165 generation-axis exploration-signal retirement hygiene.

Spec refs: REQ-REPORT-5165, SCENARIO-REPORT-5165-LINT,
SCENARIO-REPORT-5165-NARROW-SCOPE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5165_generation_axis_retirement_hygiene_v473 as mod


GREEN_VERIFY = mod.CommandResult(
    command=("python", "scripts/adversarial_verify.py"),
    exit_code=0,
    stdout='{"flags":[]}',
    stderr="",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_with_entry(entry: dict | None = None) -> dict:
    return {
        "retired": [],
        "retired_experiments": [],
        "retired_extras": [entry or mod.expected_manifest_entry()],
    }


def _roadmap_yaml(*, include_match: bool = False) -> str:
    tasks = [
        {
            "id": "exp5157-deepen-warmstart-replay-ablation-v473",
            "title": "PHASE A1 deepen-wall warm-start replay ablation",
            "prompt": (
                "Test within-game cross-level state carryover on already-contacted "
                "games; this is not first-contact exploration."
            ),
        },
        {
            "id": "exp5158-deepen-goal-energy-ranker-replay-v473",
            "title": "PHASE A2 deepen-wall goal-energy ranker replay",
            "prompt": "Rank already-contacted cross-level frontiers for deepening.",
        },
        {
            "id": "exp5159-deepen-live-levelup-attempt-v473",
            "title": "PHASE A3 deepen-wall live level-up attempt",
            "prompt": "Wire validated warm-start carryover into live level-up path.",
        },
    ]
    if include_match:
        tasks.append(
            {
                "id": "exp9999-curiosity-signal-rerun",
                "title": "curiosity driven first contact exploration signal rerun",
                "prompt": "Try another better exploration signal over first-contact generation.",
            }
        )
    return yaml.safe_dump({"milestone": "2026.07.473", "tasks": tasks}, sort_keys=False)


def make_repo(
    tmp_path: Path,
    *,
    manifest_entry: dict | None = None,
    known_issue_note: bool = True,
    roadmap_match: bool = False,
) -> Path:
    root = tmp_path
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_manifest_with_entry(manifest_entry), sort_keys=False),
        encoding="utf-8",
    )
    note = mod.KNOWN_ISSUES_NOTE_MARKER if known_issue_note else "no retirement note"
    (root / "ops" / "known-issues.md").write_text(note + "\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        _roadmap_yaml(include_match=roadmap_match),
        encoding="utf-8",
    )
    _write_json(
        root / mod.SOURCE_RESULT_PATHS["exp4688"],
        {
            "experiment_id": 4688,
            "honest_verdict": "complete: controllable_novelty_no_new_level_residual",
            "generic_agent_reached_level": 0,
            "offline_reproduced": False,
        },
    )
    _write_json(
        root / mod.SOURCE_RESULT_PATHS["exp4689"],
        {
            "experiment_id": 4689,
            "honest_verdict": "complete: program_synthesis_filter_no_coverage_gain",
            "candidate_generation_coverage_filter": 0.0,
            "coverage_delta": 0.0,
        },
    )
    _write_json(
        root / mod.SOURCE_RESULT_PATHS["exp5154"],
        {
            "experiment": "experiment_5154_energy_fitness_directed_exploration_v472",
            "honest_verdict": (
                "complete: energy_fitness_qd_winning_trajectory_not_surfaced_reproducible_delta_0"
            ),
            "winning_trajectory_surfaced": False,
            "reproducible_levels_delta": 0,
        },
    )
    return root


def test_req_report_5165_spec_declares_retirement_contract() -> None:
    """REQ-REPORT-5165: OpenSpec anchors manifest retirement and lint checks."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5165",
        "SCENARIO-REPORT-5165-LINT",
        "SCENARIO-REPORT-5165-NARROW-SCOPE",
        mod.ENTRY_ID,
        "results/experiment_5165_generation_axis_retirement_hygiene_v473.json",
        "BLOCKED_PATTERN_MATCHED",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_req_report_5165_manifest_entry_and_doc_note_are_valid(tmp_path: Path) -> None:
    """REQ-REPORT-5165: required manifest entry and dated gap note are detected."""

    root = make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        duration_s=1.25,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )

    mod.validate_artifact(artifact)
    assert artifact["exclusion_manifest_entry_added"] is True
    assert artifact["entry_id"] == mod.ENTRY_ID
    assert artifact["known_issues_or_gaps_md_updated"] is True
    assert artifact["false_positive_check_against_this_milestone"] is True
    assert artifact["synthetic_match_check_passed"] is True
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["manifest_entry_audit"]["errors"] == []
    assert artifact["source_artifact_summary"]["exp5154"]["reproducible_levels_delta"] == 0


def test_scenario_report_5165_lint_blocks_synthetic_but_not_deepen_wall(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5165-LINT and NARROW-SCOPE: linter is load-bearing and narrow."""

    root = make_repo(tmp_path)

    current = mod.check_current_roadmap_false_positive(root)
    synthetic = mod.check_synthetic_match(root)

    assert current["passed"] is True
    assert current["entry_blocked_pattern_task_ids"] == []
    assert set(current["deepen_wall_task_ids"]) == {
        "exp5157-deepen-warmstart-replay-ablation-v473",
        "exp5158-deepen-goal-energy-ranker-replay-v473",
        "exp5159-deepen-live-levelup-attempt-v473",
    }
    assert synthetic["passed"] is True
    assert synthetic["matched_risks"][0]["violation_class"] == "BLOCKED_PATTERN_MATCHED"
    assert synthetic["matched_risks"][0]["severity"] == "HARD"
    assert mod.ENTRY_ID in synthetic["matched_risks"][0]["detail"]


def test_req_report_5165_validation_edges_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5165: validation fails closed and CLI writes the artifact."""

    valid = mod.build_artifact(
        root=make_repo(tmp_path / "valid"),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    mod.validate_artifact(valid)

    mutations = [
        ("exclusion_manifest_entry_added", False),
        ("entry_id", "wrong"),
        ("false_positive_check_against_this_milestone", False),
        ("synthetic_match_check_passed", False),
        ("known_issues_or_gaps_md_updated", False),
        ("honest_verdict", "blocked_bad"),
        ("inference_substrate", "live_llm_inference"),
        ("duration_s", 0.0),
        ("tests_run", []),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in mutations:
        payload = copy.deepcopy(valid)
        payload[key] = value
        with pytest.raises(ValueError):
            mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload.pop("tests_run")
    with pytest.raises(ValueError, match="invalid Exp 5165 retirement artifact"):
        mod.validate_artifact(payload)

    payload = copy.deepcopy(valid)
    payload["field_principles"]["entry_id"] = "wrong"
    with pytest.raises(ValueError, match="invalid Exp 5165 retirement artifact"):
        mod.validate_artifact(payload)

    assert mod._load_yaml_mapping(tmp_path / "missing.yaml") == {}
    no_entry_root = tmp_path / "no_entry"
    (no_entry_root / "ops").mkdir(parents=True)
    (no_entry_root / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump({"retired_extras": []}),
        encoding="utf-8",
    )
    assert mod._find_manifest_entry(no_entry_root) == {}

    with monkeypatch.context() as patch:
        patch.setattr(mod.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError):
            mod._load_linter_module()

    missing = mod.build_artifact(
        root=make_repo(tmp_path / "missing_note", known_issue_note=False),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    assert missing["known_issues_or_gaps_md_updated"] is False
    assert missing["honest_verdict"] == mod.INCOMPLETE_VERDICT

    bad_entry = mod.expected_manifest_entry()
    bad_entry["blocked_patterns"] = []
    bad = mod.build_artifact(
        root=make_repo(tmp_path / "bad_entry", manifest_entry=bad_entry),
        duration_s=1.0,
        run_date="20260702",
        verification=mod.verification_payload(GREEN_VERIFY),
        tests_run=["unit-test-placeholder"],
    )
    assert bad["exclusion_manifest_entry_added"] is False
    assert bad["synthetic_match_check_passed"] is False
    assert bad["honest_verdict"] == mod.INCOMPLETE_VERDICT

    root = make_repo(tmp_path / "cli_repo")
    (root / "scripts" / "adversarial_verify.py").write_text(
        "import json\nprint(json.dumps({'flags': []}))\n",
        encoding="utf-8",
    )
    output = root / "module_cli_result.json"
    assert mod.main(["--root", str(root), "--output", str(output), "--date", "20260702"]) == 0
    assert output.exists()
