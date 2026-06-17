"""Tests for Exp 4357 .402 milestone capstone.

Spec refs: REQ-CAPSTONE-4357, SCENARIO-CAPSTONE-4357.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import capstone_v402_4357 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path, total: int = 23, games: int = 14) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_levels": total,
                "reproducible_total_games": games,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _publication_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _minimal_payloads(*, flagged_s3: bool = True, s3_win: bool = False) -> dict[str, JsonDict]:
    return {
        "4348_s3_search": {
            "honest_verdict": "controls_not_differentiable",
            "acceptance_gate": True,
            "s3_guided_beats_control": s3_win,
            "controls_differentiated": s3_win,
            "scorer_leak_recheck_passed": True,
            "nfe_budget": 16,
            "s3_gain_ci95": [0.208333, 0.329167] if s3_win else [-0.02, 0.01],
            "s3_minus_best_of_k_delta": 0.266667 if s3_win else 0.0,
            "s3_minus_self_reward_smc_delta": 0.266667 if s3_win else 0.0,
            "s3_minus_unguided_delta": 0.266667 if s3_win else 0.0,
            "flagged_adversarial": flagged_s3,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4349_reward_state_alignment": {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp4348-s3-stratified-verifier-guided-search.s3_guided_beats_control"
            ),
        },
        "4350_e3_ka59": {
            "honest_verdict": "success_e3_ka59_L1_reproduced",
            "game": "ka59",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4351_e3_deeper": {
            "honest_verdict": "success_e3_deeper_tn36_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 23,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "sc25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                },
                {
                    "game": "tn36",
                    "offline_reproduced": True,
                    "new_reproduced_level": 7,
                },
                {
                    "game": "ar25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                },
            ],
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4352_e3_tr87_ft09": {
            "honest_verdict": "complete_e3_tr87_ft09_partial",
            "new_levels_reproduced": 0,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {"game": "tr87", "offline_reproduced": False, "reproduced_levels": 0},
                {"game": "ft09", "offline_reproduced": False, "reproduced_levels": 0},
            ],
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4353_action_efficiency": {
            "honest_verdict": "success: learned_action_cost_reduces_actions_25_to_16",
            "acceptance_gate_passed": True,
            "action_efficiency_improves": True,
            "held_out_actions_baseline": 25,
            "held_out_actions_learned": 16,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4357_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4357: OpenSpec declares the .402 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4357" in spec
    assert "SCENARIO-CAPSTONE-4357" in spec
    assert "experiment_4357_capstone_v402.json" in spec
    assert "blocked_publication_gate_unrunnable" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "publication_gate.py --json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4357_current_artifacts_report_v402_scorecard() -> None:
    """SCENARIO-CAPSTONE-4357: current .402 artifacts report the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v402_s3_open_arc_levels_23_action_efficiency_improves_publication_ready"
    )
    assert artifact["s3_moat_utility"] == "open"
    assert artifact["s3_utility"]["status"] == "open_flagged_or_missing_s3"
    assert artifact["s3_search"]["status"] == "excluded_flagged_adversarial"
    assert artifact["reward_state_alignment"]["status"] == "blocked"
    assert artifact["reproducible_total_levels"] == 23
    assert artifact["arc_reproducible_progress"]["new_levels_since_prior"] == 2
    assert artifact["arc_reproducible_progress"]["new_games_since_prior"] == 1
    assert artifact["arc_e3_outcomes"]["new_levels_reproduced_from_artifacts"] == 2
    assert artifact["action_efficiency_improves"] is True
    assert artifact["verifier_thesis_state"] == ("moat_proven_leak_robust_but_s3_utility_open")
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is True

    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert "4348_s3_search" not in cited
    assert cited["4353_action_efficiency"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4353_action_efficiency"]
    )
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert "4348_s3_search" in excluded


def test_req_capstone_4357_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4357: missing artifacts become per-axis gaps only."""

    _write_support_files(tmp_path, total=24, games=15)
    payloads = _minimal_payloads()
    payloads.pop("4352_e3_tr87_ft09")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["s3_moat_utility"] == "open"
    assert artifact["reproducible_total_levels"] == 24
    assert artifact["action_efficiency_improves"] is True
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {"axis": "arc", "artifact_key": "4352_e3_tr87_ft09", "experiment_id": 4352}
    ]


def test_req_capstone_4357_useful_and_not_useful_decisions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4357: S3 utility decisions require clean S3 plus alignment."""

    _write_support_files(tmp_path)
    useful_payloads = _minimal_payloads(flagged_s3=False, s3_win=True)
    useful_payloads["4349_reward_state_alignment"] = {
        "honest_verdict": "success: reward_state_alignment_passed",
        "reward_state_alignment_passed": True,
        "alignment_delta": 0.11,
        "reproducibility_checksum": "sha256:" + "f" * 64,
    }
    _write_default_artifacts(tmp_path, useful_payloads)
    useful = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(useful)
    assert useful["s3_moat_utility"] == "useful_generation_gain"
    assert useful["verifier_thesis_state"] == "moat_proven_useful"
    assert useful["verifier_is_oracle"] is False

    null_root = tmp_path / "null"
    _write_support_files(null_root)
    null_payloads = _minimal_payloads(flagged_s3=False, s3_win=False)
    null_payloads["4348_s3_search"]["controls_differentiated"] = True
    null_payloads["4348_s3_search"]["s3_gain_ci95"] = [-0.03, 0.02]
    null_payloads["4349_reward_state_alignment"] = {
        "honest_verdict": "complete: no aligned gain because S3 null",
        "reward_state_alignment_passed": False,
    }
    _write_default_artifacts(null_root, null_payloads)
    null = mod.build_artifact(
        null_root,
        started_s=4.0,
        now_s=4.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(null)
    assert null["s3_moat_utility"] == "proven_but_not_useful"
    assert null["verifier_thesis_state"] == "moat_proven_not_useful"


def test_req_capstone_4357_publication_gate_unrunnable_blocks(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4357: an unrunnable publication gate stops honestly."""

    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")

    def broken_gate(_: Path) -> JsonDict:
        raise RuntimeError("fixture gate failed")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=broken_gate,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_publication_gate_unrunnable"
    assert artifact["s3_moat_utility"] == "open"
    assert artifact["reproducible_total_levels"] == 0
    assert artifact["action_efficiency_improves"] is False
    assert artifact["verifier_thesis_state"] == "blocked_publication_gate_unrunnable"
    assert artifact["publication_gate"]["error"].startswith("RuntimeError")
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is False


def test_req_capstone_4357_write_validate_and_strict_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4357: wrapper output and validation stay strict."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _minimal_payloads())
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4357_capstone_v402.json"),
        started_s=6.0,
        now_s=6.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.checksum_from_provenance(
        written["upstream_provenance"]
    )

    assert mod.s3_search_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.s3_search_read(None, False)["status"] == "missing_or_excluded"
    assert (
        mod.s3_search_read({"s3_guided_beats_control": True, "verifier_is_oracle": False}, False)[
            "status"
        ]
        == "measured_unresolved"
    )
    assert mod.reward_alignment_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.reward_alignment_read(None, False)["status"] == "missing_or_excluded"
    assert mod.reward_alignment_read({}, False)["status"] == "missing_measurement"
    assert (
        mod.s3_utility_read(
            {"status": "measured_unresolved"},
            {"reward_state_alignment_passed": False},
        )["status"]
        == "open_alignment_not_validated"
    )
    assert (
        mod.s3_utility_read(
            {"status": "measured_unresolved"},
            {"reward_state_alignment_passed": True},
        )["status"]
        == "open"
    )
    assert mod.arc_ka59_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_ka59_read(None, False)["status"] == "missing_or_excluded"
    assert mod.arc_deeper_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_deeper_read(None, False)["status"] == "missing_or_excluded"
    assert mod.arc_tr87_ft09_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_tr87_ft09_read(None, False)["status"] == "missing_or_excluded"
    deeper_with_bad_row = mod.arc_deeper_read(
        {
            "new_levels_reproduced": 1,
            "per_target_scorecard": [
                None,
                {"game": "tn36", "offline_reproduced": True, "new_reproduced_level": 7},
            ],
        },
        False,
    )
    assert deeper_with_bad_row["games_with_new_reproducible_levels"] == ["tn36"]
    assert mod.action_efficiency_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.action_efficiency_read(None, False)["status"] == "missing_or_excluded"
    assert mod.read_registry_progress(tmp_path / "missing")["status"] == "missing"
    bad_registry = tmp_path / "bad_registry"
    _write_support_files(bad_registry)
    (bad_registry / "ops" / "arc_solve_registry.yaml").write_text("[bad\n", encoding="utf-8")
    assert mod.read_registry_progress(bad_registry)["status"] == "unparseable"
    non_mapping_registry = tmp_path / "non_mapping_registry"
    _write_support_files(non_mapping_registry)
    (non_mapping_registry / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")
    assert mod.read_registry_progress(non_mapping_registry)["error"] == "non-mapping registry"
    bad_total_registry = tmp_path / "bad_total_registry"
    _write_support_files(bad_total_registry)
    (bad_total_registry / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: nope\nreproducible_total_games: nope\n",
        encoding="utf-8",
    )
    bad_total = mod.read_registry_progress(bad_total_registry)
    assert bad_total["reproducible_total_levels"] == 0
    assert bad_total["reproducible_total_games"] == 0
    assert mod.checksum_from_provenance([]) == mod.EMPTY_UPSTREAM_CHECKSUM
    assert (
        mod._cited_upstream_artifacts(  # noqa: SLF001
            [
                {"skipped": False, "fields_imported": []},
                {"skipped": False, "fields_imported": "not-list"},
            ]
        )
        == []
    )

    missing_gate = mod.build_artifact(
        tmp_path / "no_gate",
        started_s=7.0,
        now_s=7.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(missing_gate)
    assert missing_gate["honest_verdict"] == "blocked_publication_gate_unrunnable"

    non_object_gate_root = tmp_path / "non_object_gate"
    (non_object_gate_root / "scripts").mkdir(parents=True, exist_ok=True)
    (non_object_gate_root / "scripts" / "publication_gate.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    non_object_gate = mod.build_artifact(
        non_object_gate_root,
        started_s=7.5,
        now_s=7.75,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: [],  # type: ignore[return-value]
    )
    mod.validate_artifact(non_object_gate)
    assert non_object_gate["publication_gate"]["error"] == "publication_gate returned non-object"

    parse_root = tmp_path / "parse_error"
    _write_support_files(parse_root)
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4352_e3_tr87_ft09"].path
    bad_artifact.parent.mkdir(parents=True, exist_ok=True)
    bad_artifact.write_text("[]\n", encoding="utf-8")
    parse_artifact = mod.build_artifact(
        parse_root,
        started_s=8.0,
        now_s=8.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(parse_artifact)
    assert parse_artifact["flagged_artifacts_excluded"][0]["reason"] == ("unparsable_or_non_object")

    for field, value, pattern in [
        ("honest_verdict", "not_terminal", "terminal-prefixed"),
        ("s3_moat_utility", "maybe", "s3_moat_utility"),
        ("reproducible_total_levels", True, "bare int"),
        ("action_efficiency_improves", None, "bare bool"),
        ("verifier_thesis_state", "unknown", "verifier_thesis_state"),
        ("publication_gate", [], "publication_gate"),
        ("verifier_is_oracle", None, "verifier_is_oracle"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
    ]:
        bad = json.loads(json.dumps(written))
        bad[field] = value
        with pytest.raises(ValueError, match=pattern):
            mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"] = {}
    with pytest.raises(ValueError, match="upstream_provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"] = ["bad-row"]
    with pytest.raises(ValueError, match="upstream provenance row"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="upstream provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["skipped"] = True
    bad["upstream_provenance"][0]["fields_imported"] = ["s3_guided_beats_control"]
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad)

    wrapper = Path("results/experiment_4357_capstone_v402.py").read_text(encoding="utf-8")
    assert "capstone_v402_4357" in wrapper
