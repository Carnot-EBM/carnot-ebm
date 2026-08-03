"""Tests for Exp5970 ARC strip-swap convention sentinel.

Spec refs: REQ-ARC-CPTB-5970,
SCENARIO-ARC-CPTB-5970-LOSSLESS-STRIP-SWAPS,
SCENARIO-ARC-CPTB-5970-STATIC-DOSE-MATRIX,
SCENARIO-ARC-CPTB-5970-BOUNDED-LIVE-PATH.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_strip_swap_sentinel as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"


def test_req_arc_cptb_5970_spec_declares_strip_swap_contract() -> None:
    """REQ-ARC-CPTB-5970: OpenSpec freezes transform, live path, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-CPTB-5970") :]

    for marker in (
        "SCENARIO-ARC-CPTB-5970-LOSSLESS-STRIP-SWAPS",
        "SCENARIO-ARC-CPTB-5970-STATIC-DOSE-MATRIX",
        "SCENARIO-ARC-CPTB-5970-BOUNDED-LIVE-PATH",
        mod.RESULT_RELATIVE_PATH,
        "t >= EDGE_BAR_EDGE_TOLERANCE",
        "make_carnot_agent",
        "E3AgentPolicy",
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_arc_cptb_5970_lossless_row_and_column_swaps() -> None:
    """SCENARIO-ARC-CPTB-5970-LOSSLESS-STRIP-SWAPS: swaps are exact involutions."""

    grid = np.arange(8 * 9, dtype=np.int16).reshape(8, 9)
    specs = (
        mod.StripSwapSpec(axis="row", edge="top", width=2),
        mod.StripSwapSpec(axis="row", edge="bottom", width=2),
        mod.StripSwapSpec(axis="col", edge="left", width=2),
        mod.StripSwapSpec(axis="col", edge="right", width=2),
    )

    for spec in specs:
        swapped = mod.strip_swap_grid(grid, spec)
        restored = mod.inverse_strip_swap_grid(swapped, spec)
        receipt = mod.inverse_and_multiset_receipt(grid, spec)

        assert np.array_equal(restored, grid)
        assert sorted(swapped.ravel().tolist()) == sorted(grid.ravel().tolist())
        assert receipt["round_trip_equal"] is True
        assert receipt["multiset_equal"] is True
        assert receipt["outside_band_unchanged"] is True
        assert receipt["changed_cell_count"] == 2 * spec.width * (
            grid.shape[1] if spec.axis == "row" else grid.shape[0]
        )


def test_req_arc_cptb_5970_invalid_parameters_fail_closed() -> None:
    """REQ-ARC-CPTB-5970: invalid dimensions and degenerate slices are rejected."""

    grid = np.arange(16, dtype=np.uint8).reshape(4, 4)
    bad_specs = (
        mod.StripSwapSpec(axis="row", edge="top", width=1),
        mod.StripSwapSpec(axis="row", edge="left", width=2),
        mod.StripSwapSpec(axis="col", edge="top", width=2),
        mod.StripSwapSpec(axis="diag", edge="top", width=2),
        mod.StripSwapSpec(axis="row", edge="top", width=True),
        mod.StripSwapSpec(axis="row", edge="top", width=2.5),
        mod.StripSwapSpec(axis="row", edge="top", width=3),
    )

    for spec in bad_specs:
        with pytest.raises(ValueError):
            mod.strip_swap_grid(grid, spec)

    with pytest.raises(ValueError, match="2-D"):
        mod.strip_swap_grid(np.arange(8), mod.StripSwapSpec(axis="row", edge="top", width=2))


def test_req_arc_cptb_5970_defensive_helpers_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-CPTB-5970: defensive helper branches stay explicit and tested."""

    with pytest.raises(ValueError, match="unknown sentinel"):
        mod._sentinel_grid("missing")
    with pytest.raises(ValueError, match="unknown strip-swap condition"):
        mod._condition_by_id("missing")

    monkeypatch.setattr(mod, "_permutation_indices", lambda dim, edge, width: [0, 0, 2])
    with pytest.raises(ValueError, match="lose or duplicate"):
        mod._validate_permutation(3, "top", 2)

    monkeypatch.setattr(mod, "_permutation_indices", lambda dim, edge, width: list(range(dim)))
    monkeypatch.setattr(mod, "_index_map", lambda index, dim, edge, width: (index + 1) % dim)
    with pytest.raises(ValueError, match="inverse is not exact"):
        mod._validate_permutation(3, "top", 2)

    frame = mod.SimpleNamespace(
        frame=np.zeros((2, 2), dtype=np.uint8),
        available_actions=[1],
        levels_completed=2,
        state="RUNNING",
    )
    copied = mod._copy_frame_with_grid(frame, np.ones((2, 2), dtype=np.uint8))
    assert copied.frame == [[1, 1], [1, 1]]
    assert copied.available_actions == [1]
    assert copied.levels_completed == 2
    assert np.array_equal(mod._grid_of_frame(frame), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D grid"):
        mod._grid_of_frame(mod.SimpleNamespace(frame=[]))

    assert mod._action_id(mod.SimpleNamespace(name="ACTION6")) == 6
    assert mod._action_id(mod.SimpleNamespace(name="RESET")) is None
    assert mod._action_data_dict(None) is None
    assert mod._action_data_dict({"x": 1}) == {"x": 1}
    assert mod._action_data_dict(mod.SimpleNamespace(game_id="g", x=2, y=3)) == {
        "game_id": "g",
        "x": 2,
        "y": 3,
    }
    assert mod._available_action_ids(
        mod.SimpleNamespace(available_actions=[1, mod.SimpleNamespace(name="ACTION2")])
    ) == {1, 2}

    proposer = mod._NoLLMProposer()
    with pytest.raises(RuntimeError, match="disables induction"):
        proposer.generate()
    assert proposer.calls == 1
    agent = mod._SentinelBaseAgent("g")
    agent.cleanup()
    assert agent._cleanup is True

    monkeypatch.setenv("CARNOT_ARC_ACTIVE_PROBE", "1")
    with mod._disabled_escape_hatches():
        assert mod.os.environ["CARNOT_ARC_ACTIVE_PROBE"] == "0"
    assert mod.os.environ["CARNOT_ARC_ACTIVE_PROBE"] == "1"

    monkeypatch.setattr(mod.os, "sysconf", lambda name: (_ for _ in ()).throw(OSError(name)))
    assert mod._resource_receipt(REPO)["ram_available_bytes"] is None


def test_scenario_arc_cptb_5970_static_dose_matrix_has_target_and_non_target_receipts() -> None:
    """SCENARIO-ARC-CPTB-5970-STATIC-DOSE-MATRIX: target dose and collateral are explicit."""

    matrix = mod.build_static_dose_matrix()

    for sentinel in ("top_hud", "bottom_hud", "left_hud", "right_hud"):
        matching = [
            row
            for row in matrix["rows"]
            if row["sentinel"] == sentinel and row["condition_edge"] == sentinel.split("_")[0]
        ]
        assert len(matching) == 1
        row = matching[0]
        assert row["target_predicate_before"] is True
        assert row["target_predicate_after"] is False
        assert row["hud_mask_pixels_moved"] > 0
        assert row["outside_band_unchanged"] is True
        assert row["multiset_equal"] is True

    for row in matrix["rows"]:
        assert row["frontier_predicate_dose"] >= 0.0
        assert row["grid_difference_localized_to_swapped_bands"] is True
        if row["sentinel"] in {"no_hud", "frontier_only"}:
            assert row["target_predicate_before"] is False
            assert row["target_predicate_after"] is False
            assert row["hud_mask_pixels_moved"] == 0
            assert row["outside_band_unchanged"] is True


def test_req_arc_cptb_5970_condition_metadata_is_deterministic_and_targets_hud() -> None:
    """REQ-ARC-CPTB-5970: condition ids, parameters, and declared targets are versioned."""

    receipt = mod.transform_schema_parameters_and_hash()
    labels = [row["condition_id"] for row in receipt["conditions"]]

    assert labels == [
        "C4_strip_swap_rows_top_t2",
        "C5_strip_swap_rows_bottom_t2",
        "C6_strip_swap_cols_left_t2",
        "C7_strip_swap_cols_right_t2",
    ]
    assert receipt["schema_hash"].startswith("sha256:")
    assert receipt == mod.transform_schema_parameters_and_hash()
    for row in receipt["conditions"]:
        assert row["declared_targets"] == ["hud_edge_adjacency"]
        assert row["width"] >= mod.EDGE_BAR_EDGE_TOLERANCE


def test_scenario_arc_cptb_5970_bounded_live_path_has_valid_support() -> None:
    """SCENARIO-ARC-CPTB-5970-BOUNDED-LIVE-PATH: transformed observations use live E3 path."""

    receipt = mod.run_live_strip_swap_sentinel(
        root=REPO,
        anchor_games=("tn36",),
        control_games=("lp85",),
        conditions=("C5_strip_swap_rows_bottom_t2", "C4_strip_swap_rows_top_t2"),
        action_budget=1,
        seed=5970,
    )

    assert receipt["normal_path"] == "make_carnot_agent/E3AgentPolicy.choose_action"
    assert receipt["adapter_disabled"] is True
    assert receipt["source_bfs_adapter_prior_game_hidden_state_access_count"] == 0
    assert receipt["llm_induction_disabled"] is True
    assert receipt["valid_live_support"] is True
    assert receipt["valid_action_count"] >= 1
    assert any(row["hud_target_predicate_violated"] for row in receipt["rows"])


def test_req_arc_cptb_5970_artifact_schema_and_ready_gates(tmp_path: Path) -> None:
    """REQ-ARC-CPTB-5970: artifact validation rejects overclaims and checksum drift."""

    artifact = mod.build_artifact(
        root=REPO,
        result_output_path=tmp_path / "experiment_5970.json",
        test_exit_codes={"focused_unit": 0},
        live_action_budget=1,
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["strip_swap_sentinel_ready_score"] == 1.0
    assert artifact["registry_precheck_and_hash"]["public_solve_target_selected"] is False
    assert artifact["shipped_flag_and_registry_immutability"]["registry_unchanged"] is True
    assert artifact["no_solve_credit_receipt"]["solve_credit_claimed"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})
    with pytest.raises(ValueError, match="missing required fields"):
        missing = dict(artifact)
        del missing["status"]
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact({**artifact, "inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": True})
    with pytest.raises(ValueError, match="solve_credit"):
        bad = json.loads(json.dumps(artifact))
        bad["no_solve_credit_receipt"]["solve_credit_claimed"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="complete_ready status"):
        bad = json.loads(json.dumps(artifact))
        bad["status"] = "complete_null"
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="ready_score"):
        bad = json.loads(json.dumps(artifact))
        bad["anchor_support_and_behavioral_validity"]["valid_live_support"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="registry immutability"):
        bad = json.loads(json.dumps(artifact))
        bad["shipped_flag_and_registry_immutability"]["registry_unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="protected files"):
        bad = json.loads(json.dumps(artifact))
        bad["protected_files_unchanged"]["all_unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        bad = json.loads(json.dumps(artifact))
        bad["status"] = "complete_null"
        bad["strip_swap_sentinel_ready_score"] = 0.0
        bad["honest_verdict"] = "ready: invalid prefix"
        mod.validate_artifact(bad)


def test_req_arc_cptb_5970_writer_round_trips_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5970: writer emits the artifact returned by the builder."""

    payload = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    payload.update({"status": "complete_ready", "honest_verdict": "complete_ready: fixture"})
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: payload)

    out = tmp_path / "experiment_5970.json"
    written = mod.write_artifact(root=REPO, result_output_path=out, test_exit_codes={"unit": 0})

    assert written is payload
    assert json.loads(out.read_text(encoding="utf-8"))["honest_verdict"] == "complete_ready: fixture"
