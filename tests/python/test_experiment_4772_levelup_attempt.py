"""Tests for Exp 4772 ARC level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4772,
SCENARIO-ARC-WMTE-4772-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4772-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4772-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4772_levelup_attempt as exp4772


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - r11l: prefix_rooted_graph_search_stalled_at_L1
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - g50t: clone_replay_L2_route_reached_distance_12_no_bank
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - ka59_l2_not_reproduced_after_step_counter_state_key
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 65
"""


def _loop_result(game: str = "ka59", reached_level: int = 1, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "learned_verifier_checkpoint": None,
        "reproduction_gate": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["4", "4", "C:1"],
    }


def test_req_arc_wmte_4772_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4772: OpenSpec declares the 4772 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4772.SPEC_REFS:
        assert ref in spec
    assert exp4772.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4772.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4772_public_unsolved_takes_priority() -> None:
    """SCENARIO-ARC-WMTE-4772-ROTATION-TARGET: unreproduced public games are first."""

    registry = yaml.safe_load(_registry_text())
    registry["games"][0]["levels_reproduced"] = 0

    selection = exp4772.select_rotation_target(registry, adaptered_games={"ka59"})

    assert selection["game"] == "re86"
    assert selection["prior_level"] == 0
    assert selection["target_level"] == 1
    assert selection["reason"] == "preferred_public_first_contact"


def test_scenario_arc_wmte_4772_selects_shallowest_adaptered_deepen() -> None:
    """SCENARIO-ARC-WMTE-4772-ROTATION-TARGET: L2 public games fall back to L1 deepen."""

    registry = yaml.safe_load(_registry_text())

    selection = exp4772.select_rotation_target(registry, adaptered_games={"ka59", "dc22"})

    assert selection["game"] == "ka59"
    assert selection["prior_level"] == 1
    assert selection["target_level"] == 2
    assert selection["reason"] == "shallowest_standing_loop_deepen"
    assert {row["game"] for row in selection["public_rotation"]} == {"re86", "sb26", "bp35", "lf52"}
    assert {row["game"] for row in selection["skipped_shallow_non_adaptered"]} == {"g50t", "r11l"}
    assert "ka59" in exp4772._adaptered_games()


def test_scenario_arc_wmte_4772_same_depth_reproduction_is_not_a_bank() -> None:
    """SCENARIO-ARC-WMTE-4772-REPRODUCTION-GATE: same-depth gates retire with no bank."""

    selection = {
        "game": "ka59",
        "prior_level": 1,
        "target_level": 2,
        "reason": "shallowest_standing_loop_deepen",
    }

    attempt = exp4772.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("ka59", 1),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )

    assert attempt["offline_reproduced_existing_depth"] is True
    assert attempt["offline_reproduced_new_depth"] is False
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "reproduced_existing_or_lower_level"
    assert "same-depth" in attempt["dead_end"]
    assert attempt["target_selection_reason"] == "shallowest_standing_loop_deepen"


def test_req_arc_wmte_4772_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4772: no-bank artifact preserves the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4772.select_rotation_target(registry, adaptered_games={"ka59"})
    attempt = exp4772.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("ka59", 1),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )

    artifact = exp4772.build_artifact(
        registry=registry,
        selection=selection,
        attempt=attempt,
        preconditions_checked={
            "AGENTS.md": True,
            "CODEX.md": True,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "induction_needed": False,
        },
    )

    assert artifact["honest_verdict"] == "complete_ka59_no_new_level_residual_existing_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["schema_errors"] == []
    assert exp4772.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4772_success_requires_strictly_new_reproduced_level() -> None:
    """REQ-ARC-WMTE-4772: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4772.select_rotation_target(registry, adaptered_games={"ka59"})
    attempt = exp4772.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("ka59", 2),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )

    artifact = exp4772.build_artifact(
        registry=registry,
        selection=selection,
        attempt=attempt,
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    assert artifact["honest_verdict"] == "success_ka59_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["registry_update"]["updated"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 66
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4772_schema_guards_required_fields() -> None:
    """REQ-ARC-WMTE-4772: schema validation rejects overclaims and bad checksums."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4772.select_rotation_target(registry, adaptered_games={"ka59"})
    attempt = exp4772.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("ka59", 1),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )
    artifact = exp4772.build_artifact(
        registry=registry,
        selection=selection,
        attempt=attempt,
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    assert "missing_field:honest_verdict" in exp4772.artifact_schema_errors(missing)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["solve_provenance"] = "wrong"
    assert "missing_principle:solve_provenance" in exp4772.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4772.artifact_schema_errors(bad_checksum)

    stale_checksum = dict(artifact)
    stale_checksum["reproducibility_checksum"] = "0" * 64
    assert "checksum_mismatch" in exp4772.artifact_schema_errors(stale_checksum)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_ka59"
    assert "honest_verdict_missing_terminal_prefix" in exp4772.artifact_schema_errors(bad_prefix)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4772.artifact_schema_errors(bad_provenance)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    assert "verifier_is_oracle_must_be_true" in exp4772.artifact_schema_errors(bad_oracle)

    fabricated_bank = dict(artifact)
    fabricated_bank["new_levels_banked"] = 1
    fabricated_bank["reproducibility_checksum"] = exp4772.stable_checksum(fabricated_bank)
    assert "bank_without_offline_reproduction" in exp4772.artifact_schema_errors(fabricated_bank)

    fabricated_repro = dict(artifact)
    fabricated_repro["offline_reproduced"] = True
    fabricated_repro["reproducibility_checksum"] = exp4772.stable_checksum(fabricated_repro)
    assert "offline_reproduced_true_without_new_bank" in exp4772.artifact_schema_errors(fabricated_repro)


def test_req_arc_wmte_4772_edge_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4772: malformed registries and missing loop outputs do not fabricate progress."""

    registry = {"games": [{"game": "re86", "levels_reproduced": "bad"}], "reproducible_total_levels": object()}

    assert exp4772.registry_levels(registry) == {"re86": 0}
    assert exp4772.registry_total_levels(registry) == 0

    selection = exp4772.select_rotation_target(yaml.safe_load(_registry_text()), adaptered_games=set())
    assert selection["game"] == "none"
    assert selection["reason"] == "no_reproduced_standing_loop_target"

    missing = exp4772.collect_attempt(
        {"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert missing["residual_cause"] == "missing_loop_result"
    assert missing["new_levels_banked"] == 0
    missing_artifact = exp4772.build_artifact(
        registry=yaml.safe_load(_registry_text()),
        selection={"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        attempt=missing,
        preconditions_checked={"offline_arcade": {"ok": True}},
    )
    assert missing_artifact["honest_verdict"] == "complete_ka59_no_new_level_residual_missing_loop_result"

    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4772.load_registry(registry_file) == {}


def test_scenario_arc_wmte_4772_stable_artifact_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4772-STABLE-ARTIFACT: writer emits deterministic JSON."""

    payload = {"z": 1, "a": {"b": 2}}
    out = tmp_path / exp4772.RESULT_RELATIVE_PATH

    exp4772.write_artifact(payload, path=out)

    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert out.read_text(encoding="utf-8").startswith("{\n  \"a\"")


def test_req_arc_wmte_4772_main_writes_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4772: main writes a stable terminal artifact from loop results."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4772.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_ka59.json").write_text(
        json.dumps(_loop_result("ka59", 1)),
        encoding="utf-8",
    )

    monkeypatch.setattr(exp4772, "REPO", tmp_path)
    monkeypatch.setattr(exp4772, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4772, "REGISTRY", tmp_path / exp4772.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp4772, "ARTIFACT", tmp_path / exp4772.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4772, "_adaptered_games", lambda: {"ka59"})
    monkeypatch.setattr(
        exp4772,
        "check_preconditions",
        lambda: {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    assert exp4772.main([]) == 0

    written = json.loads((tmp_path / exp4772.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "complete_ka59_no_new_level_residual_existing_depth"
    assert written["target_game"] == "ka59"
    assert written["schema_errors"] == []
