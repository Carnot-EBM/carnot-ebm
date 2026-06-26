"""Tests for Exp 4825 .444 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4825, SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4825")
    end = spec.index("### REQ-ARC-WMTE-4781", start)
    return spec[start:end]


def _lint_result(passed: bool = True) -> dict[str, Any]:
    return {
        "command": "python scripts/arc_orphan_solver_lint.py",
        "passed": passed,
        "returncode": 0 if passed else 1,
        "stdout_tail": "OK: all solver-like ARC modules are reachable from the live agent path."
        if passed
        else "ORPHANED ARC SOLVER / WORLD-MODEL MODULES",
        "stderr_tail": "",
    }


def _s3_row(
    game: str,
    *,
    reachable: bool = True,
    e_banked: bool = False,
    bare_banked: bool = False,
    already_bare: bool = False,
    lambda0_attempts: int = 4,
    e_guided_attempts: int = 4,
) -> dict[str, Any]:
    return {
        "game": game,
        "winner_rank": {"rank": None, "candidate_count": None, "source": "fixture"},
        "winner-rank": {"rank": None, "candidate_count": None, "source": "fixture"},
        "banked_by_E": e_banked,
        "banked-by-E": e_banked,
        "banked_by_bare": bare_banked,
        "banked-by-bare": bare_banked,
        "was_already_in_bare_pool": already_bare,
        "was-already-in-bare-pool": already_bare,
        "winner_newly_entered_pool": bool(e_banked and not bare_banked and not already_bare),
        "positive_control_reachable": reachable,
        "bare_reached_level": 1 if bare_banked else 0,
        "e_guided_reached_level": 1 if e_banked else 0,
        "bare_offline_reproduced": bare_banked,
        "e_guided_offline_reproduced": e_banked,
        "lambda0_attempts": lambda0_attempts,
        "e_guided_attempts": e_guided_attempts,
    }


def _s3_payload(*, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    games = [f"g{i}" for i in range(5)]
    game_results = rows if rows is not None else [_s3_row(game) for game in games]
    return {
        "schema": "carnot.arc_structural_energy_s3_generation_lift_4821.v1",
        "experiment": "experiment_4821_structural_energy_s3_generation_lift",
        "experiment_id": 4821,
        "spec_refs": ["REQ-ARC-WMTE-4821"],
        "result_path": "results/experiment_4821_structural_energy_s3_generation_lift.json",
        "honest_verdict": "complete_structural_energy_s3_bounded_no_generation_lift",
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "preconditions_checked": {
            "arc_orphan_solver_lint": {"passed": True},
            "source_artifacts_present": {
                "results/experiment_4737_goal_energy_candidate_generation_valid_test.json": True
            },
        },
        "n_headroom_games": len(game_results),
        "min_headroom_games": 5,
        "winners_newly_entering_pool_delta": 0.0,
        "winners_newly_entering_pool_delta_ci95": [0.0, 0.0],
        "new_levels_not_in_bare_pool": [],
        "solve_provenance": "live_agent_self_discovery",
        "game_results": game_results,
        "positive_control_passed": True,
        "retire_if_same_verdict": True,
        "lambda_guidance": 1.0,
        "lambda0_control": {
            "lambda": 0.0,
            "matched_control": True,
            "description": "goal_guidance_lambda=0 disables plan_in_model goal_energy guidance",
        },
        "energy_config": {},
        "source_artifacts": {
            "matched_generation_measurement": (
                "results/experiment_4737_goal_energy_candidate_generation_valid_test.json"
            )
        },
        "field_principles": {},
        "duration_s": 1.0,
        "random_seed": 4821,
        "reproducibility_checksum": "sha256:fixture",
    }


def _generation_source(*, exercised: bool = True) -> dict[str, Any]:
    baseline_rows: list[dict[str, Any]] = []
    guided_rows: list[dict[str, Any]] = []
    signatures: list[str] = []
    for game_index in range(5):
        game = f"g{game_index}"
        for variant in range(4):
            signature = f"{game}~v{variant}"
            signatures.append(signature)
            baseline_rows.append(
                {
                    "game": game,
                    "variant": variant,
                    "variant_signature": signature,
                    "attempted": True,
                    "policy_mode": "baseline",
                    "solution_labels": [],
                    "reached_level": 0,
                    "reproduction_gate": {"reproduced": False, "reached_level": 0},
                }
            )
            diagnostics = {
                "enabled": True,
                "candidate_pool_differs_from_baseline": exercised and game_index == 0,
                "candidate_states_scored": 12 if exercised else 0,
                "candidate_count": 12 if exercised else 0,
                "real_candidate_state_evidence": exercised,
                "goal_energy_score_variance": 0.5 if exercised else 0.0,
                "score_min": 0.1,
                "score_max": 0.9,
            }
            guided_rows.append(
                {
                    "game": game,
                    "variant": variant,
                    "variant_signature": signature,
                    "attempted": True,
                    "policy_mode": "goal_energy",
                    "solution_labels": [],
                    "reached_level": 0,
                    "reproduction_gate": {"reproduced": False, "reached_level": 0},
                    "goal_candidate_guidance_diagnostics": diagnostics,
                }
            )
    return {
        "experiment": "experiment_4737_goal_energy_candidate_generation_valid_test",
        "random_seed": 4737,
        "arms_non_degenerate": exercised,
        "candidate_pool_differs_from_baseline": exercised,
        "goal_energy_score_variance": 0.5 if exercised else 0.0,
        "positive_control_passed": True,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "baseline_measurement": {
            "variant_attempts": baseline_rows,
            "variant_attempts_count": len(baseline_rows),
            "variant_signatures": signatures,
        },
        "goal_energy_measurement": {
            "variant_attempts": guided_rows,
            "variant_attempts_count": len(guided_rows),
            "variant_signatures": signatures,
        },
        "nondegeneracy": {
            "arms_non_degenerate": exercised,
            "candidate_pool_differs_from_baseline": exercised,
            "candidate_states_scored": 12 if exercised else 0,
            "goal_energy_score_variance": 0.5 if exercised else 0.0,
            "diagnostics": {
                "enabled": True,
                "real_candidate_state_evidence": exercised,
            },
        },
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4822_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "target_game": "ka59",
        "solve_provenance": "live_agent_self_discovery",
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
                "residual_cause": "reproduced_existing_or_lower_level",
            }
        ],
    }


def _firstwin_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4824_heldout_first_win_readiness",
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


def _write_fixture_artifacts(root: Path, *, exercised: bool = True) -> None:
    _write_json(
        root,
        "results/experiment_4821_structural_energy_s3_generation_lift.json",
        _s3_payload(),
    )
    _write_json(root, "results/experiment_4822_levelup_attempt.json", _levelup_payload())
    _write_json(
        root,
        "results/experiment_4824_heldout_first_win_readiness.json",
        _firstwin_payload(),
    )
    _write_json(
        root,
        "results/experiment_4737_goal_energy_candidate_generation_valid_test.json",
        _generation_source(exercised=exercised),
    )


def test_req_arc_wmte_4825_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4825: OpenSpec declares the .444 audit contract."""

    from carnot import experiment_4825_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4825" in section
    assert "SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4825_s3_controls_and_guidance_noop(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT: S3 needs matched controls and exercised guidance."""

    from carnot import experiment_4825_silent_bug_audit as mod

    source = _generation_source(exercised=True)
    good = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(),
        source_artifacts={"matched_generation_measurement": source},
        lint_result=_lint_result(True),
    )
    noop = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(),
        source_artifacts={"matched_generation_measurement": _generation_source(exercised=False)},
        lint_result=_lint_result(True),
    )
    unmatched_budget = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(rows=[_s3_row(f"g{i}", e_guided_attempts=3) for i in range(5)]),
        source_artifacts={"matched_generation_measurement": source},
        lint_result=_lint_result(True),
    )
    missing_reachability = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(rows=[_s3_row("g0", reachable=False)] + [_s3_row(f"g{i}") for i in range(1, 5)]),
        source_artifacts={"matched_generation_measurement": source},
        lint_result=_lint_result(True),
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["s3_controls_verified"] is True
    assert good["s3_guidance_exercised"] is True
    assert good["s3_control_check"]["matched_lambda0_control"] is True
    assert good["s3_control_check"]["same_games_seeds_budget"] is True
    assert good["s3_control_check"]["guidance_exercise_check"]["differing_headroom_games"]

    assert noop["verdict"] == "inconclusive_guidance_no_op"
    assert noop["s3_controls_verified"] is True
    assert noop["s3_guidance_exercised"] is False
    assert "s3_guidance_no_op" in noop["silent_bug_signatures"]

    assert unmatched_budget["verdict"] == "silent_bug_must_reopen"
    assert "s3_lambda0_control_not_matched" in unmatched_budget["silent_bug_signatures"]
    assert missing_reachability["verdict"] == "silent_bug_must_reopen"
    assert "s3_positive_control_missing_for_headroom" in missing_reachability[
        "silent_bug_signatures"
    ]


def test_req_arc_wmte_4825_other_arc_nulls_require_exercise_evidence() -> None:
    """REQ-ARC-WMTE-4825: non-S3 nulls are trusted only with exercised levers."""

    from carnot import experiment_4825_silent_bug_audit as mod

    levelup = mod.audit_null_artifact("experiment_4822_levelup_attempt", _levelup_payload())
    firstwin = mod.audit_null_artifact(
        "experiment_4824_heldout_first_win_readiness",
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
        "experiment_4822_levelup_attempt",
        bad_levelup_payload,
    )["silent_bug_signatures"]
    bad_firstwin = mod.audit_null_artifact(
        "experiment_4824_heldout_first_win_readiness",
        bad_firstwin_payload,
    )
    assert "firstwin_positive_control_missing" in bad_firstwin["silent_bug_signatures"]
    assert "firstwin_0_04_tautology_unannotated" in bad_firstwin["silent_bug_signatures"]
    assert "heldout_attempt_floor_not_met" in bad_firstwin["silent_bug_signatures"]
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]


def test_req_arc_wmte_4825_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4825: checked-in .444 artifacts produce a complete audit."""

    from carnot import experiment_4825_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False, lint_runner=lambda _root: _lint_result(True))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_0_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["s3_controls_verified"] is True
    assert artifact["s3_guidance_exercised"] is True
    assert artifact["s3_control_check"]["n_headroom_games"] == 24
    assert artifact["s3_control_check"]["same_games_seeds_budget"] is True
    assert artifact["s3_control_check"]["guidance_exercise_check"][
        "candidate_pool_differs_from_baseline"
    ] is True
    assert artifact["s3_control_check"]["guidance_exercise_check"][
        "differing_headroom_games"
    ]
    assert artifact["silent_bugs_found"] == []
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4821_structural_energy_s3_generation_lift",
        "experiment_4822_levelup_attempt",
        "experiment_4824_heldout_first_win_readiness",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_444_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4825_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4825: complete audits write JSON and append the ops report."""

    from carnot import experiment_4825_silent_bug_audit as mod

    _write_fixture_artifacts(tmp_path, exercised=True)
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True, lint_runner=lambda _root: _lint_result(True))

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4825 .444 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4821_structural_energy_s3_generation_lift`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4825_blocked_paths_and_schema_guards(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4825_silent_bug_audit as mod

    blocked = mod.run(root=tmp_path, write=True, lint_runner=lambda _root: _lint_result(True))
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


def test_req_arc_wmte_4825_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4825: malformed null evidence is not trusted."""

    from carnot import experiment_4825_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)
    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    assert mod._list_of_mappings("not-a-list") == []
    assert mod._format_number(None) == "None"
    assert mod._format_number(1.25) == "1.25"
    assert mod._tail("abcdef", limit=3) == "def"
    assert mod._truthy_passed(True) is True
    assert mod._per_game_budget_matched([]) is False
    assert mod._variant_signatures([{"game": "g0", "variant": 7}]) == {
        '["g0",7]'
    }

    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="x" * 2100, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    lint = mod.run_arc_orphan_solver_lint(REPO)
    assert lint["passed"] is True
    assert len(lint["stdout_tail"]) == 2000

    row = _s3_row("g0", e_banked=True, already_bare=True)
    rerank = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(rows=[row] + [_s3_row(f"g{i}") for i in range(1, 5)]),
        source_artifacts={"matched_generation_measurement": _generation_source(exercised=True)},
        lint_result=_lint_result(True),
    )
    assert "s3_new_level_is_reranking" in rerank["silent_bug_signatures"]

    bad_lint = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(),
        source_artifacts={"matched_generation_measurement": _generation_source(exercised=True)},
        lint_result=_lint_result(False),
    )
    assert "s3_goal_energy_not_live_path_reachable" in bad_lint["silent_bug_signatures"]

    missing_source = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(),
        source_artifacts={},
        lint_result=_lint_result(True),
    )
    assert missing_source["verdict"] == "inconclusive_guidance_no_op"
    assert "s3_guidance_evidence_missing" in missing_source["silent_bug_signatures"]

    fingerprint_source = _generation_source(exercised=True)
    fingerprint_source["baseline_measurement"]["variant_attempts"][0][
        "proposal_fingerprint"
    ] = "bare"
    fingerprint_source["goal_energy_measurement"]["variant_attempts"][0][
        "proposal_fingerprint"
    ] = "guided"
    fingerprint = mod.audit_null_artifact(
        "experiment_4821_structural_energy_s3_generation_lift",
        _s3_payload(),
        source_artifacts={"matched_generation_measurement": fingerprint_source},
        lint_result=_lint_result(True),
    )
    assert "g0:proposal_fingerprint" in fingerprint["s3_control_check"][
        "guidance_exercise_check"
    ]["fingerprint_differences"]

    artifact = mod.run(root=REPO, write=False, lint_runner=lambda _root: _lint_result(True))
    invalids: list[dict[str, Any]] = [
        artifact | {"field_principles": {}},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"nulls_audited": "3"},
        artifact | {"s3_controls_verified": "true"},
        artifact | {"s3_guidance_exercised": "true"},
        artifact | {"silent_bugs_found": {}},
        artifact | {"per_null_verdicts": {}},
        artifact | {"s3_control_check": []},
        artifact | {"duration_s": 0.0},
        artifact | {"nulls_audited": 99},
    ]
    for invalid_artifact in invalids:
        invalid_artifact["reproducibility_checksum"] = mod.payload_checksum(invalid_artifact)
        assert mod.artifact_schema_errors(invalid_artifact)

    bad_checksum = copy.deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.artifact_schema_errors(bad_checksum)
    with pytest.raises(ValueError):
        mod.write_artifact(bad_checksum, root=tmp_path)

    rendered = mod.render_markdown_section(artifact | {"per_null_verdicts": [None]})
    assert "Experiment 4825" in rendered
    rendered_without_s3 = mod.render_markdown_section(artifact | {"s3_control_check": None})
    assert "### S3 Control Check" not in rendered_without_s3

    fresh_report_root = tmp_path / "fresh_report"
    mod.append_markdown_report(artifact, root=fresh_report_root)
    assert (fresh_report_root / mod.AUDIT_REPORT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False, lint_runner=lambda _root: _lint_result(True))
