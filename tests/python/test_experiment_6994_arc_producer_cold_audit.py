"""Independent ARC producer cold-audit tests.

Spec refs: REQ-ARC-WMTE-6994 and SCENARIO-ARC-WMTE-6994-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6994_arc_producer_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fixture_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    root = tmp_path_factory.mktemp("exp6994") / "fixtures"
    rows = exp.materialize_fixtures(root)
    return root, rows


def _passing_receipt() -> dict:
    return {
        "fresh_process": True,
        "network_namespace_isolated": True,
        "gpu_devices_visible": [],
        "llm_modules_loaded": [],
        "arc_service_disabled": True,
        "game_source_disabled": True,
        "source_tree_read_only": True,
        "output_is_temporary": True,
        "passed": True,
    }


def test_req_arc_wmte_6994_spec_precedes_implementation() -> None:
    """REQ-ARC-WMTE-6994 defines every cold-audit evidence surface."""
    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-ARC-WMTE-6994", 1)[1]
    for marker in (
        "SCENARIO-ARC-WMTE-6994-PRECONDITIONS",
        "SCENARIO-ARC-WMTE-6994-HASH-REPLAY",
        "SCENARIO-ARC-WMTE-6994-ELIGIBILITY",
        "SCENARIO-ARC-WMTE-6994-ATOMICITY",
        "SCENARIO-ARC-WMTE-6994-FACTORY-ROUTING",
        "SCENARIO-ARC-WMTE-6994-NO-SOLVE",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_6994_hashes_inputs_before_source_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-6994-PRECONDITIONS rejects changed bytes before parsing."""
    source = tmp_path / "source.json"
    source.write_text('{"arc_producer_contract_complete_score": 1}\n', encoding="utf-8")
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "fixture_index.json").write_text("[]\n", encoding="utf-8")

    def forbidden_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("aggregate claim parsed before its hash passed")

    monkeypatch.setattr(exp.json, "loads", forbidden_parse)
    loaded = exp.load_hashed_inputs(
        tmp_path,
        fixture_root,
        source_paths={"exp6993": Path("source.json")},
        expected_source_hash="sha256:wrong",
    )

    assert loaded["passed"] is False
    assert loaded["sources"] == {}
    assert loaded["preconditions"][0]["check"] == "source_hash:exp6993"


def test_scenario_6994_raw_fixture_matrix_replays_fail_closed(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """SCENARIO-ARC-WMTE-6994-HASH-REPLAY covers changed, reordered, and missing bytes."""
    root, rows = fixture_store
    replay = exp.replay_fixtures(root, rows)
    summaries = {row["fixture_id"]: row for row in replay["per_fixture_rows"]}

    assert summaries["success"]["eligible"] is True
    assert [row["fixture_id"] for row in replay["eligibility_replay_rows"] if row["eligible"]] == [
        "success"
    ]
    prompt = next(
        row for row in replay["prompt_hash_replay_rows"] if row["fixture_id"] == "tamper_prompt"
    )
    assert prompt["passed"] is False
    assert prompt["expected_hash"] != prompt["observed_hash"]
    missing = next(
        row for row in replay["prompt_hash_replay_rows"] if row["fixture_id"] == "missing_prompt"
    )
    assert missing["observed_hash"] is None
    reordered = next(
        row
        for row in replay["transition_order_rows"]
        if row["fixture_id"] == "reordered_transitions"
    )
    assert reordered["observed_order"] == [1, 0]
    assert reordered["passed"] is False
    assert summaries["unreachable_engine"]["eligible"] is False
    assert "canonical_engine_reachable" in summaries["unreachable_engine"]["failed_checks"]
    assert all(row["terminal"] for row in replay["rows"])


def test_scenario_6994_legacy_duplicate_and_fake_sources_are_ineligible(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """SCENARIO-ARC-WMTE-6994-ELIGIBILITY rejects old, repeated, and fake provenance."""
    root, rows = fixture_store
    replay = exp.replay_fixtures(root, rows)
    summaries = {row["fixture_id"]: row for row in replay["per_fixture_rows"]}

    assert summaries["legacy"]["classification"] == "legacy"
    assert summaries["legacy"]["eligible"] is False
    assert summaries["duplicate"]["eligible"] is False
    assert "unique_run_id" in summaries["duplicate"]["failed_checks"]
    fake_rows = [
        row for row in replay["source_kind_rows"] if row["fixture_id"].startswith("source_")
    ]
    assert not any(row["passed"] for row in fake_rows)
    assert all(summaries[row["fixture_id"]]["eligible"] is False for row in fake_rows)


def test_scenario_6994_interrupted_atomic_writes_have_no_marker(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """SCENARIO-ARC-WMTE-6994-ATOMICITY keeps all stopped writes ineligible."""
    root, rows = fixture_store
    replay = exp.replay_fixtures(root, rows)
    interrupted = [
        row for row in replay["atomicity_replay_rows"] if row["scenario"] == "interrupted"
    ]

    assert len(interrupted) == 4
    assert all(row["manifest_marker_count"] == 0 for row in interrupted)
    assert all(row["eligible_count"] == 0 for row in interrupted)
    success = next(row for row in replay["atomicity_replay_rows"] if row["fixture_id"] == "success")
    assert success["manifest_marker_count"] == 1
    assert success["eligible_count"] == 1


@pytest.mark.memory_watchdog_skip
def test_scenario_6994_shipped_factory_routes_and_changes_action_score(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """SCENARIO-ARC-WMTE-6994-FACTORY-ROUTING executes the shipped routing seam."""
    root, rows = fixture_store
    replay = exp.replay_fixtures(root, rows)
    trace, routing, influence = exp.trace_factory_route(root, "success")

    assert (
        next(row for row in replay["per_fixture_rows"] if row["fixture_id"] == "success")[
            "eligible"
        ]
        is True
    )
    assert [row["symbol"] for row in trace] == [
        "make_carnot_agent",
        "E3AgentPolicy",
        "load_engine",
        "_world_model_candidates",
    ]
    assert all(row["called"] for row in trace)
    assert routing["selected_candidate_name"] == "loaded_world_model.py"
    assert routing["engine_hash_matches_envelope"] is True
    assert influence["score_changed"] is True
    assert influence["control_score"] == 0
    assert influence["routed_score"] == 1


def test_req_6994_full_replay_confirms_source_without_solve_claim(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """REQ-ARC-WMTE-6994 emits a complete audit and mandatory false claim fields."""
    root, _rows = fixture_store
    loaded = exp.load_hashed_inputs(REPO_ROOT, root)
    artifact = exp.build_from_sources(
        REPO_ROOT,
        root,
        run_date="20260904",
        loaded=loaded,
        read_only_receipt=_passing_receipt(),
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["arc_contract_audit_complete_score"] == 1
    assert artifact["arc_producer_contract_confirmed_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["source_disagreement_rows"] == []
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["solve_provenance_applicable"] is False
    for field in (
        "solve_claimed",
        "level_claimed",
        "registry_updated",
        "submitted_to_leaderboard",
        "model_quality_claimed",
        "verifier_is_oracle",
    ):
        assert artifact[field] is False


def test_scenario_6994_failed_precondition_is_blocked_not_partial() -> None:
    """SCENARIO-ARC-WMTE-6994-PRECONDITIONS reports missing upstream data as blocked."""
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        preconditions_checked=[exp.gate_check("raw_fixture_rows", "complete", "missing")],
        evidence={},
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_arc_producer_cold_audit"
    assert artifact["arc_contract_audit_complete_score"] == 0
    assert artifact["arc_producer_contract_confirmed_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "raw_fixture_rows"
    assert artifact["gate_check_summary"]["expected_value"] == "complete"
    assert artifact["gate_check_summary"]["observed_value"] == "missing"


def test_scenario_6994_artifact_validation_rejects_overclaims(
    fixture_store: tuple[Path, list[dict]],
) -> None:
    """SCENARIO-ARC-WMTE-6994-NO-SOLVE makes claim and checksum drift invalid."""
    root, _rows = fixture_store
    artifact = exp.build_from_sources(
        REPO_ROOT,
        root,
        run_date="20260904",
        loaded=exp.load_hashed_inputs(REPO_ROOT, root),
        read_only_receipt=_passing_receipt(),
    )
    changed = deepcopy(artifact)
    changed["solve_claimed"] = True
    changed["field_principles"].pop("rows")
    changed["verdict_class"] = "unknown"
    changed["honest_verdict"] = "partial_wrong"
    changed["arc_contract_audit_complete_score"] = True
    changed["arc_producer_contract_confirmed_score"] = 1

    errors = exp.validate_artifact(changed)
    assert "field_principles_mismatch" in errors
    assert "solve_claimed_must_be_false" in errors
    assert "verdict_class_invalid" in errors
    assert "arc_contract_audit_complete_score_not_bare_int" in errors
    assert "reproducibility_checksum_mismatch" in errors

    blocked = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        preconditions_checked=[exp.gate_check("fixture", True, False)],
        evidence={},
    )
    blocked["honest_verdict"] = "partial_wrong"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "verdict_prefix_mismatch" in exp.validate_artifact(blocked)


def test_scenario_6994_fresh_command_disables_external_capabilities(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6994-PRECONDITIONS creates a restricted fresh child."""
    command = exp.fresh_process_command(
        executable=Path("/venv/python"),
        wrapper=Path("/repo/wrapper.py"),
        repo_root=Path("/repo"),
        fixture_root=Path("/fixtures"),
        writable_root=tmp_path,
        output_path=tmp_path / "child.json",
        run_date="20260904",
        parent_netns="net:[1]",
    )
    joined = " ".join(str(value) for value in command)

    assert "--unshare-net" in command
    assert "--unshare-pid" in command
    assert "CUDA_VISIBLE_DEVICES  " in joined
    assert "CARNOT_DISABLE_LLM 1" in joined
    assert "CARNOT_DISABLE_ARC_SERVICE 1" in joined
    assert "CARNOT_DISABLE_GAME_SOURCE 1" in joined
    assert "--fresh-child" in command
    assert str(tmp_path / "child.json") in command
