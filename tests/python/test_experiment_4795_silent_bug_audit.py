"""Tests for Exp 4795 .441 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4795, SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4795")
    end = spec.index("### REQ-ARC-WMTE-4731", start)
    return spec[start:end]


def _lint_result(passed: bool = True) -> dict[str, Any]:
    return {
        "command": "python scripts/arc_orphan_solver_lint.py",
        "passed": passed,
        "returncode": 0 if passed else 1,
        "stdout_tail": (
            "OK: all solver-like ARC modules are reachable from the live agent path "
            "(39 modules in the live closure)."
            if passed
            else "ORPHANED ARC SOLVER / WORLD-MODEL MODULES"
        ),
        "stderr_tail": "",
    }


def _s2_payload(
    *,
    live_path: bool = True,
    real_control: bool = True,
    identical_sets: bool = True,
) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    candidate_sets: dict[str, list[str]] = {}
    for idx in range(5):
        game = f"g{idx}"
        candidates = [f"{game}/candidate_a.py", f"{game}/candidate_b.py"]
        first_passes = idx % 2 == 0
        rows = [
            {
                "candidate_name": candidates[0],
                "binary_gate_pass": first_passes,
                "prefix_accuracy": 0.75 if first_passes else 0.25,
                "heldout_accuracy": 0.5,
                "heldout_cell_recall": 0.5,
                "offpath_structural_energy": 20.0,
            },
            {
                "candidate_name": candidates[1],
                "binary_gate_pass": False,
                "prefix_accuracy": 0.25,
                "heldout_accuracy": 0.5,
                "heldout_cell_recall": 0.5,
                "offpath_structural_energy": 10.0,
            },
        ]
        games.append(
            {
                "game": game,
                "n_candidates": 2,
                "energy_selected_candidate": candidates[1],
                "accuracy_gate_selected_candidate": candidates[0],
                "accuracy_gate_exact_accuracy": 0.75 if first_passes else 0.25,
                "accuracy_gate_passed": first_passes,
                "candidate_rows": rows,
            }
        )
        candidate_sets[game] = candidates if identical_sets else list(reversed(candidates))
    return {
        "experiment": "experiment_4791_structural_energy_s2_offpath_trust_gate",
        "honest_verdict": "complete_structural_energy_s2_no_live_trust_value",
        "live_path_reachable": live_path,
        "verifier_is_oracle": False,
        "binary_accuracy_control": {
            "selection_rule": (
                "first_candidate_with_WorldModelVerifier_exact_accuracy_ge_0.5_else_first"
                if real_control
                else "best_prefix_accuracy_strawman"
            ),
            "threshold": 0.5,
            "verifier_is_oracle": True,
        },
        "preconditions_checked": {"arc_orphan_solver_lint_passed": live_path},
        "energy_config": {"source_experiment": 4781},
        "n_heldout_games": 5,
        "min_heldout_games": 5,
        "candidate_sets": candidate_sets,
        "game_results": games,
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4792_levelup_attempt",
        "honest_verdict": "complete_sb26_no_new_level_residual_existing_depth",
        "target_game": "sb26",
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "schema_errors": [],
        "preconditions_checked": {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_offline_env": {"ok": True, "game": "sb26"},
        },
        "attempted_games": [
            {
                "game": "sb26",
                "prior_level": 2,
                "target_level": 3,
                "reached_level": 2,
                "new_levels_banked": 0,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
                "residual_cause": "reproduced_existing_or_lower_level",
                "reproduction_gate": {
                    "reproduced": True,
                    "reached_level": 2,
                    "claimed_level": 2,
                },
                "solution_labels": ["{\"action\":6}", "{\"action\":5}"],
            }
        ],
    }


def _firstwin_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4794_heldout_first_win_readiness",
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
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


def test_req_arc_wmte_4795_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4795: OpenSpec declares the .441 audit contract."""

    from carnot import experiment_4795_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4795" in section
    assert "SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.AUDIT_REPORT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4795_reconstructs_s2_real_binary_control() -> None:
    """SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT: S2 uses live path and real binary control."""

    from carnot import experiment_4795_silent_bug_audit as mod

    good = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        _s2_payload(),
        lint_result=_lint_result(True),
    )
    weakened = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        _s2_payload(real_control=False),
        lint_result=_lint_result(True),
    )
    mismatched = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        _s2_payload(identical_sets=False),
        lint_result=_lint_result(True),
    )
    orphaned = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        _s2_payload(live_path=False),
        lint_result=_lint_result(False),
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["s2_live_path_reachable_confirmed"] is True
    assert good["s2_control_check"]["real_binary_accuracy_control"] is True
    assert good["s2_control_check"]["identical_candidate_sets"] is True
    assert good["s2_control_check"]["binary_selection_reconstructed"] is True
    assert good["s2_control_check"]["checked_games"] == 5

    assert weakened["verdict"] == "silent_bug_must_reopen"
    assert "s2_binary_control_not_real_incumbent" in weakened["silent_bug_signatures"]
    assert mismatched["verdict"] == "silent_bug_must_reopen"
    assert "s2_candidate_sets_not_identical" in mismatched["silent_bug_signatures"]
    assert orphaned["verdict"] == "silent_bug_must_reopen"
    assert "s2_energy_gate_not_live_path_reachable" in orphaned["silent_bug_signatures"]


def test_req_arc_wmte_4795_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4795: checked-in .441 artifacts produce a complete audit."""

    from carnot import experiment_4795_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_0_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["s2_live_path_reachable_confirmed"] is True
    assert artifact["s2_control_check"]["real_binary_accuracy_control"] is True
    assert artifact["s2_control_check"]["identical_candidate_sets"] is True
    assert artifact["s2_control_check"]["binary_selection_reconstructed"] is True
    assert artifact["arc_orphan_solver_lint"]["passed"] is True
    assert artifact["silent_bugs_found"] == []
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        "experiment_4792_levelup_attempt",
        "experiment_4794_heldout_first_win_readiness",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_441_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4795_write_artifact_and_append_markdown(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4795: complete audits write JSON and append the ops report."""

    from carnot import experiment_4795_silent_bug_audit as mod

    _write_json(
        tmp_path,
        "results/experiment_4791_structural_energy_s2_offpath_trust_gate.json",
        _s2_payload(),
    )
    _write_json(tmp_path, "results/experiment_4792_levelup_attempt.json", _levelup_payload())
    _write_json(
        tmp_path,
        "results/experiment_4794_heldout_first_win_readiness.json",
        _firstwin_payload(),
    )
    report = tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# ARC Null Silent-Bug Audit\n", encoding="utf-8")

    artifact = mod.run(root=tmp_path, write=True, lint_runner=lambda _root: _lint_result(True))

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert loaded == artifact
    text = report.read_text(encoding="utf-8")
    assert "## Experiment 4795 .441 ARC Null Silent-Bug Audit" in text
    assert "`experiment_4791_structural_energy_s2_offpath_trust_gate`" in text
    size_after_first = len(text)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert len(report.read_text(encoding="utf-8")) == size_after_first


def test_req_arc_wmte_4795_blocked_paths_and_schema_guards(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION: missing sources fail closed."""

    from carnot import experiment_4795_silent_bug_audit as mod

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


def test_req_arc_wmte_4795_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4795: malformed null evidence is not trusted."""

    from carnot import experiment_4795_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)
    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    assert mod._list_of_mappings("not-a-list") == []
    deduped = ["existing"]
    mod._append_unique(deduped, "existing")
    mod._append_unique(deduped, "")
    assert deduped == ["existing"]

    bad_s2 = _s2_payload()
    bad_s2["n_heldout_games"] = 1
    bad_s2["game_results"][0]["accuracy_gate_selected_candidate"] = "wrong.py"
    bad_s2["game_results"][0]["candidate_rows"] = []
    bad_s2_result = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        bad_s2,
        lint_result=_lint_result(True),
    )
    assert "s2_heldout_game_floor_not_met" in bad_s2_result["silent_bug_signatures"]
    assert "s2_candidate_rows_missing" in bad_s2_result["silent_bug_signatures"]
    assert "s2_binary_selection_not_reconstructed" in bad_s2_result["silent_bug_signatures"]

    malformed_s2 = _s2_payload()
    malformed_s2["game_results"][0]["accuracy_gate_selected_candidate"] = "wrong.py"
    malformed_s2["game_results"][0]["accuracy_gate_passed"] = False
    malformed_s2["game_results"][0]["energy_selected_candidate"] = "missing.py"
    malformed_s2["game_results"][0]["candidate_rows"][0]["offpath_structural_energy"] = None
    malformed_s2_result = mod.audit_null_artifact(
        "experiment_4791_structural_energy_s2_offpath_trust_gate",
        malformed_s2,
        lint_result=_lint_result(True),
    )
    assert "s2_energy_scores_missing" in malformed_s2_result["silent_bug_signatures"]
    assert "g0:accuracy_gate_passed_mismatch" in malformed_s2_result["s2_control_check"]["mismatches"]
    assert "g0:energy_selected_not_in_candidate_set" in malformed_s2_result["s2_control_check"]["mismatches"]

    bad_levelup = mod.audit_null_artifact(
        "experiment_4792_levelup_attempt",
        {"attempted_games": [], "schema_errors": ["bad"]},
    )
    assert "levelup_attempts_missing" in bad_levelup["silent_bug_signatures"]
    assert "levelup_schema_errors_present" in bad_levelup["silent_bug_signatures"]

    inconsistent_levelup_payload = _levelup_payload()
    inconsistent_levelup_payload["offline_reproduced"] = True
    inconsistent_levelup_payload["preconditions_checked"]["offline_arcade"]["ok"] = False
    inconsistent_levelup_payload["attempted_games"] = [
        {
            "game": "sb26",
            "prior_level": 2,
            "target_level": 3,
            "reached_level": 3,
            "new_levels_banked": 0,
            "offline_reproduced_existing_depth": True,
            "offline_reproduced_new_depth": True,
            "reproduction_gate": {},
            "solution_labels": [],
        }
    ]
    inconsistent_levelup = mod.audit_null_artifact(
        "experiment_4792_levelup_attempt",
        inconsistent_levelup_payload,
    )
    assert "levelup_solution_labels_missing" in inconsistent_levelup["silent_bug_signatures"]
    assert "levelup_reproduction_gate_missing" in inconsistent_levelup["silent_bug_signatures"]
    assert "levelup_same_depth_evidence_missing" in inconsistent_levelup["silent_bug_signatures"]
    assert "levelup_depth_accounting_inconsistent" in inconsistent_levelup["silent_bug_signatures"]
    assert (
        "levelup_offline_reproduced_true_without_new_bank"
        in inconsistent_levelup["silent_bug_signatures"]
    )
    assert "levelup_offline_arcade_failed" in inconsistent_levelup["silent_bug_signatures"]

    bad_firstwin = mod.audit_null_artifact(
        "experiment_4794_heldout_first_win_readiness",
        {
            "heldout_first_win_rate": 0.04,
            "first_win_baseline": 0.04,
            "heldout_variant_attempts": 1,
            "positive_control_passed": False,
            "parity_test_green": False,
            "null_delta_methodology_note": "",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "live_agent_ran": True,
            "heldout_proxy_summary": {"proxy_cache_used": False},
        },
    )
    assert "heldout_attempt_floor_not_met" in bad_firstwin["silent_bug_signatures"]
    assert "firstwin_positive_control_missing" in bad_firstwin["silent_bug_signatures"]
    assert "firstwin_0_04_tautology_unannotated" in bad_firstwin["silent_bug_signatures"]
    assert "parity_test_not_green" in bad_firstwin["silent_bug_signatures"]
    assert (
        "firstwin_substrate_declares_aggregation_but_live_ran"
        in bad_firstwin["silent_bug_signatures"]
    )
    assert "firstwin_aggregation_cache_evidence_missing" in bad_firstwin["silent_bug_signatures"]

    unknown = mod.audit_null_artifact("experiment_unknown", {})
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]

    artifact = mod.run(root=REPO, write=False)
    invalids: list[dict[str, Any]] = [
        artifact | {"field_principles": {}},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"nulls_audited": "3"},
        artifact | {"s2_live_path_reachable_confirmed": "true"},
        artifact | {"silent_bugs_found": {}},
        artifact | {"per_null_verdicts": {}},
        artifact | {"s2_control_check": []},
        artifact | {"duration_s": 0.0},
        artifact | {"nulls_audited": 99},
    ]
    for invalid_artifact in invalids:
        invalid_artifact["reproducibility_checksum"] = mod.payload_checksum(invalid_artifact)
        assert mod.artifact_schema_errors(invalid_artifact)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.artifact_schema_errors(bad_checksum)
    with pytest.raises(ValueError):
        mod.write_artifact(bad_checksum, root=tmp_path)

    rendered = mod.render_markdown_section(artifact | {"per_null_verdicts": [None]})
    assert "Experiment 4795" in rendered
    rendered_without_s2 = mod.render_markdown_section(artifact | {"s2_control_check": None})
    assert "### S2 Control Check" not in rendered_without_s2

    fresh_report_root = tmp_path / "fresh_report"
    mod.append_markdown_report(artifact, root=fresh_report_root)
    assert (fresh_report_root / mod.AUDIT_REPORT_RELATIVE_PATH).exists()

    blocked_no_write = mod.run(
        root=tmp_path / "missing_no_write",
        write=False,
        lint_runner=lambda _root: _lint_result(True),
    )
    assert blocked_no_write["honest_verdict"] == "blocked_missing_source_artifacts"

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False)
