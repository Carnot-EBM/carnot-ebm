"""Tests for Exp 4815 .443 ARC null silent-bug audit.

Spec refs: REQ-ARC-WMTE-4815, SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT,
SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION.
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
    start = spec.index("### REQ-ARC-WMTE-4815")
    end = spec.index("### REQ-ARC-WMTE-4768", start)
    return spec[start:end]


def _s2v3_payload(
    *,
    available: int = 25,
    attempted: int = 25,
    effective: int = 15,
    logged: bool = True,
) -> dict[str, Any]:
    game_results: list[dict[str, Any]] = []
    for idx in range(attempted):
        game = f"g{idx:02d}"
        diverse = idx < effective
        recalls = [0.25, 0.70, 0.80] if diverse else [0.50, 0.50, 0.50]
        candidates = [
            {
                "candidate_name": f"{game}/accuracy_seed",
                "candidate_source": "deterministic_transition_induction",
                "genuinely_induced": True,
                "prefix_accuracy": 0.60,
                "heldout_cell_recall": recalls[0],
                "offpath_structural_energy": 3.0,
                "binary_gate_pass": True,
            },
            {
                "candidate_name": f"{game}/energy_seed",
                "candidate_source": "programmatic_expert_induction",
                "genuinely_induced": True,
                "prefix_accuracy": 0.40,
                "heldout_cell_recall": recalls[1],
                "offpath_structural_energy": 1.0,
                "binary_gate_pass": False,
            },
            {
                "candidate_name": f"{game}/headroom_seed",
                "candidate_source": "deterministic_transition_induction",
                "genuinely_induced": True,
                "prefix_accuracy": 0.20,
                "heldout_cell_recall": recalls[2],
                "offpath_structural_energy": 2.0,
                "binary_gate_pass": False,
            },
        ]
        row: dict[str, Any] = {
            "game": game,
            "effective": diverse,
            "n_candidates": len(candidates),
            "candidate_rows": candidates,
            "energy_selected_candidate": f"{game}/energy_seed",
            "accuracy_gate_selected_candidate": f"{game}/accuracy_seed",
            "energy_selected_offpath_cell_recall": recalls[1],
            "accuracy_gate_selected_offpath_cell_recall": recalls[0],
            "selection_candidates_differ": True,
        }
        if not logged:
            row.pop("energy_selected_candidate")
        game_results.append(row)
    return {
        "experiment": "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        "honest_verdict": "complete_structural_energy_s2v3_bounded_corpus_wide",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_available_games": available,
        "n_games_attempted": attempted,
        "n_effective_games": effective,
        "min_heldout_games": 15,
        "required_effective_games": 15,
        "energy_minus_accuracy_delta": 0.0,
        "positive_control_passed": True,
        "game_results": game_results,
        "candidate_sets": {
            row["game"]: [candidate["candidate_name"] for candidate in row["candidate_rows"]]
            for row in game_results
        },
    }


def _levelup_payload() -> dict[str, Any]:
    return {
        "experiment": "experiment_4812_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "target_game": "ka59",
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "new_levels_banked": 0,
        "schema_errors": [],
        "preconditions_checked": {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_offline_env": {"ok": True, "game": "ka59"},
        },
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
        "experiment": "experiment_4814_heldout_first_win_readiness",
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


def test_req_arc_wmte_4815_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4815: OpenSpec declares the .443 audit contract."""

    from carnot import experiment_4815_silent_bug_audit as mod

    section = _spec_section()

    assert "REQ-ARC-WMTE-4815" in section
    assert "SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT" in section
    assert "SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4815_s2v3_corpus_coverage_guard() -> None:
    """SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT: S2-v3 must cover the real corpus."""

    from carnot import experiment_4815_silent_bug_audit as mod

    good = mod.audit_null_artifact(
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        _s2v3_payload(),
        real_corpus_size=25,
    )
    under_declared = mod.audit_null_artifact(
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        _s2v3_payload(available=5, attempted=5, effective=5),
        real_corpus_size=25,
    )
    subset = mod.audit_null_artifact(
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        _s2v3_payload(attempted=5, effective=5),
        real_corpus_size=25,
    )
    degenerate = mod.audit_null_artifact(
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        _s2v3_payload(effective=9),
        real_corpus_size=25,
    )
    unlogged = mod.audit_null_artifact(
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        _s2v3_payload(logged=False),
        real_corpus_size=25,
    )

    assert good["verdict"] == "trustworthy_null"
    assert good["s2v3_corpus_coverage_verified"] is True
    assert good["s2v3_corpus_coverage_check"]["degenerate_candidate_pool_flagged"] is False
    assert good["s2v3_corpus_coverage_check"]["required_effective_games"] == 15

    assert under_declared["verdict"] == "silent_bug_must_reopen"
    assert "s2v3_n_available_games_mismatch_real_corpus" in under_declared[
        "silent_bug_signatures"
    ]
    assert "DEGENERATE_CANDIDATE_POOL" in under_declared[
        "s2v3_corpus_coverage_check"
    ]["flag_kinds"]

    assert subset["verdict"] == "silent_bug_must_reopen"
    assert "s2v3_not_corpus_wide_attempt" in subset["silent_bug_signatures"]
    assert degenerate["verdict"] == "silent_bug_must_reopen"
    assert "s2v3_degenerate_candidate_pool" in degenerate["silent_bug_signatures"]
    assert "s2v3_effective_game_floor_not_met" in degenerate["silent_bug_signatures"]
    assert unlogged["verdict"] == "silent_bug_must_reopen"
    assert "s2v3_per_game_selection_logging_missing" in unlogged["silent_bug_signatures"]


def test_req_arc_wmte_4815_other_arc_nulls_require_exercise_evidence() -> None:
    """REQ-ARC-WMTE-4815: non-S2 nulls are trusted only with exercised levers."""

    from carnot import experiment_4815_silent_bug_audit as mod

    levelup = mod.audit_null_artifact("experiment_4812_levelup_attempt", _levelup_payload())
    firstwin = mod.audit_null_artifact(
        "experiment_4814_heldout_first_win_readiness",
        _firstwin_payload(),
    )
    bad_levelup_payload = _levelup_payload()
    bad_levelup_payload["attempted_games"] = []
    bad_firstwin_payload = _firstwin_payload()
    bad_firstwin_payload["positive_control_passed"] = False
    bad_firstwin_payload["null_delta_methodology_note"] = ""
    bad_firstwin_payload["heldout_variant_attempts"] = 1

    bad_levelup = mod.audit_null_artifact(
        "experiment_4812_levelup_attempt",
        bad_levelup_payload,
    )
    bad_firstwin = mod.audit_null_artifact(
        "experiment_4814_heldout_first_win_readiness",
        bad_firstwin_payload,
    )
    unknown = mod.audit_null_artifact("experiment_unknown", {})

    assert levelup["verdict"] == "trustworthy_null"
    assert levelup["null_id"] == "experiment_4812_levelup_attempt"
    assert firstwin["verdict"] == "trustworthy_null"
    assert firstwin["null_id"] == "experiment_4814_heldout_first_win_readiness"
    assert "levelup_attempts_missing" in bad_levelup["silent_bug_signatures"]
    assert "firstwin_positive_control_missing" in bad_firstwin["silent_bug_signatures"]
    assert "firstwin_0_04_tautology_unannotated" in bad_firstwin["silent_bug_signatures"]
    assert "heldout_attempt_floor_not_met" in bad_firstwin["silent_bug_signatures"]
    assert unknown["silent_bug_signatures"] == ["unknown_null_artifact"]


def test_req_arc_wmte_4815_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4815: checked-in .443 artifacts produce a complete audit."""

    from carnot import experiment_4815_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_arc_null_silent_bug_audit_3_nulls_0_reopen"
    assert artifact["nulls_audited"] == 3
    assert artifact["s2v3_corpus_coverage_verified"] is True
    assert artifact["s2v3_corpus_coverage_check"]["real_corpus_size"] == 25
    assert artifact["s2v3_corpus_coverage_check"]["n_available_games"] == 25
    assert artifact["s2v3_corpus_coverage_check"]["n_games_attempted"] == 25
    assert artifact["s2v3_corpus_coverage_check"]["n_effective_games"] == 23
    assert artifact["s2v3_corpus_coverage_check"]["required_effective_games"] == 15
    assert artifact["s2v3_corpus_coverage_check"]["flag_kinds"] == []
    assert artifact["silent_bugs_found"] == []
    assert set(artifact["trusted_nulls"]) == {
        "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        "experiment_4812_levelup_attempt",
        "experiment_4814_heldout_first_win_readiness",
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["milestone_443_artifacts_present"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4815_write_artifact_and_blocked_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION: writes fail closed."""

    from carnot import experiment_4815_silent_bug_audit as mod

    env = tmp_path / "environment_files"
    env.mkdir()
    for idx in range(25):
        (env / f"g{idx:02d}.json").write_text("{}", encoding="utf-8")
    _write_json(
        tmp_path,
        "results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json",
        _s2v3_payload(),
    )
    _write_json(tmp_path, "results/experiment_4812_levelup_attempt.json", _levelup_payload())
    _write_json(
        tmp_path,
        "results/experiment_4814_heldout_first_win_readiness.json",
        _firstwin_payload(),
    )

    artifact = mod.run(root=tmp_path, write=True)
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["s2v3_corpus_coverage_verified"] is True
    assert mod.real_corpus_size(tmp_path) == 25

    blocked = mod.run(root=tmp_path / "missing", write=True)
    assert blocked["honest_verdict"] == "blocked_missing_source_artifacts"
    assert blocked["nulls_audited"] == 0
    assert blocked["silent_bugs_found"] == []
    assert (tmp_path / "missing" / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(blocked) == []


def test_req_arc_wmte_4815_schema_and_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4815: schema validation rejects malformed audit artifacts."""

    from carnot import experiment_4815_silent_bug_audit as mod

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    assert mod._list_of_mappings("not-a-list") == []
    assert mod.required_effective_games(25, 25) == 15
    assert mod.required_effective_games(8, 8) == 10
    assert mod.required_effective_games(12, 20) == 12
    assert mod.real_corpus_size(tmp_path / "no-corpus") == 0
    assert mod._game_selection_logged({"candidate_rows": []}) is False
    logged_row = _s2v3_payload()["game_results"][0]
    bad_name_row = copy.deepcopy(logged_row)
    bad_name_row["energy_selected_candidate"] = "missing"
    bad_energy_recall_row = copy.deepcopy(logged_row)
    bad_energy_recall_row["energy_selected_offpath_cell_recall"] = None
    bad_accuracy_recall_row = copy.deepcopy(logged_row)
    bad_accuracy_recall_row["accuracy_gate_selected_offpath_cell_recall"] = None
    assert mod._game_selection_logged(bad_name_row) is False
    assert mod._game_selection_logged(bad_energy_recall_row) is False
    assert mod._game_selection_logged(bad_accuracy_recall_row) is False

    artifact = mod.run(root=REPO, write=False)
    invalids: list[dict[str, Any]] = [
        artifact | {"honest_verdict": "not terminal"},
        artifact | {"field_principles": {}},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"s2v3_corpus_coverage_verified": "true"},
        artifact | {"nulls_audited": "3"},
        artifact | {"silent_bugs_found": {}},
        artifact | {"per_null_verdicts": {}},
        artifact | {"s2v3_corpus_coverage_check": []},
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

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(root=REPO, write=False)
