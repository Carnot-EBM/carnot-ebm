"""Tests for Exp 4835 .445 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4835, SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4835")
    end = spec.index("### REQ-ARC-WMTE-4775", start)
    return spec[start:end]


def _a1_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4831_amortized_incontext_exploration_prior_live",
        "honest_verdict": "complete_amortized_prior_no_first_win_lift_l1_wall_survives",
        "go_explore_archive_alive": {
            "alive": True,
            "observations": 2,
            "stored_cells": 2,
            "prefixes_injected": 1,
            "actions_injected": 1,
        },
        "prior_changed_proposals": True,
        "prior_change_diagnostics": {
            "changed": True,
            "no_prior_order": [1, 2, 3],
            "with_prior_order": [1, 3, 2],
        },
        "prior_diagnostics": {"proposal_changes": 1, "rank_calls": 12, "context_hits": 12},
        "first_win_rate_with_prior": 0.0,
        "first_win_rate_no_prior_ablation": 0.0,
        "first_win_delta_ci95": {"low": 0.0, "high": 0.0},
        "imitation_control_heldout_games": {
            "distillation_games": ["cd82", "cn04"],
            "heldout_games": ["bp35"],
            "heldout_not_in_distillation_set": True,
            "first_win_rate_with_prior": 0.0,
            "first_win_rate_no_prior_ablation": 0.0,
            "lift_holds": False,
        },
        "live_path_reachable": True,
        "inference_substrate": "live_llm_inference",
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4832_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "target_game": "ka59",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "schema_errors": [],
        "preconditions_checked": {"offline_arcade": {"ok": True}},
        "attempted_games": [
            {
                "game": "ka59",
                "prior_level": 1,
                "target_level": 2,
                "reached_level": 1,
                "new_levels_banked": 0,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
                "reproduction_gate": {
                    "reproduced": True,
                    "reached_level": 1,
                    "claimed_level": 1,
                },
                "solution_labels": ["4", "3"],
            }
        ],
    }


def _firstwin_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4834_heldout_first_win_readiness",
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "parity_test": {"passed": True},
        "null_delta_methodology_note": "flat 0.04 genuine null",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "live_agent_ran": False,
        "heldout_proxy_summary": {"proxy_cache_used": True},
        "preconditions_checked": {"ok": True},
    }


def _write_json(root: Path, rel: str, payload: dict[str, Any]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fixture_artifacts(root: Path) -> None:
    _write_json(
        root,
        "results/experiment_4831_amortized_incontext_exploration_prior_live.json",
        _a1_payload(),
    )
    _write_json(root, "results/experiment_4832_levelup_attempt.json", _levelup_payload())
    _write_json(
        root,
        "results/experiment_4834_heldout_first_win_readiness.json",
        _firstwin_payload(),
    )


def test_req_arc_wmte_4835_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4835: OpenSpec declares the .445 audit contract."""

    from carnot import experiment_4835_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4835" in section
    assert "SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4835_a1_archive_prior_and_imitation_guards() -> None:
    """SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT: A1 needs alive archive, exercised prior, and held-out imitation control."""

    from carnot import experiment_4835_silent_bug_audit as mod

    good = mod.audit_null_artifact(
        "experiment_4831_amortized_incontext_exploration_prior_live",
        _a1_payload(),
    )
    dead = _a1_payload()
    dead["go_explore_archive_alive"] = {
        "alive": False,
        "observations": 0,
        "stored_cells": 0,
        "prefixes_injected": 0,
    }
    noop = _a1_payload()
    noop["prior_changed_proposals"] = False
    noop["prior_change_diagnostics"] = {
        "changed": False,
        "no_prior_order": [1, 2, 3],
        "with_prior_order": [1, 2, 3],
    }
    noop["prior_diagnostics"] = {"proposal_changes": 0}
    overlap = _a1_payload()
    overlap["imitation_control_heldout_games"] = {
        "distillation_games": ["bp35"],
        "heldout_games": ["bp35"],
        "heldout_not_in_distillation_set": False,
    }

    dead_result = mod.audit_null_artifact(
        "experiment_4831_amortized_incontext_exploration_prior_live",
        dead,
    )
    noop_result = mod.audit_null_artifact(
        "experiment_4831_amortized_incontext_exploration_prior_live",
        noop,
    )
    overlap_result = mod.audit_null_artifact(
        "experiment_4831_amortized_incontext_exploration_prior_live",
        overlap,
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["a1_control_check"]["archive_alive"] is True
    assert good["a1_control_check"]["prior_changed"] is True
    assert good["a1_control_check"]["imitation_control_confirmed"] is True
    assert good["a1_archive_alive_and_prior_exercised"] is True

    assert dead_result["verdict"] == "silent_bug_must_reopen"
    assert "dead_go_explore_archive" in dead_result["silent_bug_signatures"]
    assert noop_result["verdict"] == "silent_bug_must_reopen"
    assert "prior_no_op" in noop_result["silent_bug_signatures"]
    assert overlap_result["verdict"] == "silent_bug_must_reopen"
    assert "a1_imitation_control_missing" in overlap_result["silent_bug_signatures"]


def test_req_arc_wmte_4835_other_arc_nulls_require_exercise_evidence() -> None:
    """REQ-ARC-WMTE-4835: non-A1 nulls are trusted only with exercised levers."""

    from carnot import experiment_4835_silent_bug_audit as mod

    levelup = mod.audit_null_artifact("experiment_4832_levelup_attempt", _levelup_payload())
    firstwin = mod.audit_null_artifact(
        "experiment_4834_heldout_first_win_readiness",
        _firstwin_payload(),
    )
    bad_levelup_payload = _levelup_payload()
    bad_levelup_payload["attempted_games"] = []
    bad_firstwin_payload = _firstwin_payload()
    bad_firstwin_payload["positive_control_passed"] = False
    bad_firstwin_payload["null_delta_methodology_note"] = ""
    bad_firstwin_payload["heldout_variant_attempts"] = 1
    unknown = mod.audit_null_artifact("experiment_unknown", {})

    assert levelup["verdict"] == "trustworthy_null"
    assert firstwin["verdict"] == "trustworthy_null"
    assert "levelup_attempts_missing" in mod.audit_null_artifact(
        "experiment_4832_levelup_attempt",
        bad_levelup_payload,
    )["silent_bug_signatures"]
    bad_firstwin = mod.audit_null_artifact(
        "experiment_4834_heldout_first_win_readiness",
        bad_firstwin_payload,
    )
    assert "firstwin_positive_control_missing" in bad_firstwin["silent_bug_signatures"]
    assert "firstwin_0_04_tautology_unannotated" in bad_firstwin["silent_bug_signatures"]
    assert "heldout_attempt_floor_not_met" in bad_firstwin["silent_bug_signatures"]
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]


def test_req_arc_wmte_4835_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4835: checked-in .445 artifacts produce a complete audit."""

    from carnot import experiment_4835_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_0_reopen"
    assert artifact["a1_archive_alive_and_prior_exercised"] is True
    assert artifact["a1_control_check"]["archive_alive"] is True
    assert artifact["a1_control_check"]["prior_changed"] is True
    assert artifact["a1_control_check"]["imitation_control_confirmed"] is True
    assert artifact["nulls_audited"] == 3
    assert artifact["silent_bugs_found"] == []
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4831_amortized_incontext_exploration_prior_live",
        "experiment_4832_levelup_attempt",
        "experiment_4834_heldout_first_win_readiness",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_445_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4835_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4835: complete audits write JSON and append the ops report."""

    from carnot import experiment_4835_silent_bug_audit as mod

    _write_fixture_artifacts(tmp_path)
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True)

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4835 .445 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4831_amortized_incontext_exploration_prior_live`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4835_blocked_paths_and_schema_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4835_silent_bug_audit as mod

    blocked = mod.run(root=tmp_path, write=True)
    assert blocked["honest_verdict"] == "blocked_missing_source_artifacts"
    assert blocked["nulls_audited"] == 0
    assert blocked["silent_bugs_found"] == []
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert not (tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(blocked) == []

    invalid = dict(blocked)
    invalid["honest_verdict"] = "not terminal"
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    assert "honest_verdict_missing_terminal_prefix" in mod.artifact_schema_errors(invalid)

    full = mod.run(root=REPO, write=False)
    invalids: list[dict[str, Any]] = [
        full | {"field_principles": {}},
        full | {"inference_substrate": "wrong"},
        full | {"a1_archive_alive_and_prior_exercised": "true"},
        full | {"nulls_audited": "3"},
        full | {"silent_bugs_found": {}},
        full | {"per_null_verdicts": {}},
        full | {"a1_control_check": []},
        full | {"duration_s": 0.0},
        full | {"nulls_audited": 99},
    ]
    for invalid_artifact in invalids:
        invalid_artifact["reproducibility_checksum"] = mod.payload_checksum(invalid_artifact)
        assert mod.artifact_schema_errors(invalid_artifact)

    bad_checksum = copy.deepcopy(full)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.artifact_schema_errors(bad_checksum)
    with pytest.raises(ValueError):
        mod.write_artifact(bad_checksum, root=tmp_path)

    rendered = mod.render_markdown_section(full | {"per_null_verdicts": [None]})
    assert "Experiment 4835" in rendered
    rendered_without_a1 = mod.render_markdown_section(full | {"a1_control_check": None})
    assert "### A1 Control Check" not in rendered_without_a1

    assert mod._string_set("not-a-list") == set()
    fresh_report_root = tmp_path / "fresh_report"
    mod.append_markdown_report(full, root=fresh_report_root)
    assert (fresh_report_root / mod.AUDIT_REPORT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False)
