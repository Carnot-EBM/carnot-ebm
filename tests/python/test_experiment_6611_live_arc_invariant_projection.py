"""Tests for live E3 invariant projection over sealed transition archives.

Spec refs: REQ-ARC-WMTE-6611, REQ-ARC-WMTE-6611-LIVE,
REQ-ARC-WMTE-6611-FEATURES, REQ-ARC-WMTE-6611-SPLIT,
REQ-ARC-WMTE-6611-ARCHIVE, REQ-ARC-WMTE-6611-CONTROLS,
REQ-ARC-WMTE-6611-ROWS, REQ-ARC-WMTE-6611-VERDICT,
REQ-ARC-WMTE-6611-FAILURES, REQ-ARC-WMTE-6611-ATOMIC,
SCENARIO-ARC-WMTE-6611-LIVE, SCENARIO-ARC-WMTE-6611-SPLIT,
SCENARIO-ARC-WMTE-6611-ORACLE, SCENARIO-ARC-WMTE-6611-ROWS,
SCENARIO-ARC-WMTE-6611-ATTACKS, SCENARIO-ARC-WMTE-6611-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as live
from carnot.agentic import arc_invariant_projector as projector
from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot import experiment_6611_live_arc_invariant_projection as mod
import scripts.adversarial_verify as adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _test_receipts() -> list[dict[str, object]]:
    return [
        {"command": command, "exit_code": 0, "duration_s": 0.01}
        for command in mod.VALIDATION_COMMANDS
    ]


def _checksum(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_arc_wmte_6611_spec_owns_live_archive_contract() -> None:
    """REQ-ARC-WMTE-6611: OpenSpec pins the live, split, and artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-6611:") :]
    for marker in (
        "REQ-ARC-WMTE-6611-LIVE",
        "REQ-ARC-WMTE-6611-FEATURES",
        "REQ-ARC-WMTE-6611-SPLIT",
        "REQ-ARC-WMTE-6611-ARCHIVE",
        "REQ-ARC-WMTE-6611-CONTROLS",
        "REQ-ARC-WMTE-6611-ROWS",
        "REQ-ARC-WMTE-6611-VERDICT",
        "REQ-ARC-WMTE-6611-FAILURES",
        "REQ-ARC-WMTE-6611-ATOMIC",
        "SCENARIO-ARC-WMTE-6611-ORACLE",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in mod.FIELD_PRINCIPLES


def test_req_arc_wmte_6611_mandated_substrate_is_reviewed_by_arc_validators() -> None:
    """REQ-ARC-WMTE-6611-ATOMIC: the exact no-LLM substrate is allowlisted."""

    payload = {
        "honest_verdict": "complete_held_prediction_only_no_solve_claim",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "duration_s": 0.02,
    }
    assert discipline.duration_floor_s(mod.INFERENCE_SUBSTRATE) == 0.01
    assert discipline.validate_arc_solve_artifact(payload) == []
    classification = adversarial_verify._classify_inference_substrate(payload)
    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["source"] == "top_level_inference_substrate"
    assert adversarial_verify.duration_floor_for_artifact(payload) == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": adversarial_verify.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    flags: list[object] = []
    adversarial_verify._emit_no_llm_by_name_warning(payload, flags)
    assert flags == []


def test_req_arc_wmte_6611_general_projector_math_and_edges() -> None:
    """REQ-ARC-WMTE-6611-FEATURES: projection math is bounded and game blind."""

    state = np.asarray([2.0, 1.0])
    result = projector.project_to_level_set(state, np.eye(2), target=1.0)
    assert result.converged is True
    assert result.iterations <= projector.DEFAULT_MAX_ITERATIONS
    assert result.distance > 0.0
    assert abs(projector.quadratic_value(result.state, np.eye(2)) - 1.0) <= 1e-6

    already = projector.project_to_level_set(np.ones(2), np.eye(2), target=2.0)
    assert already.state.tolist() == [1.0, 1.0]
    assert already.iterations == 0
    zero = projector.project_to_level_set(np.zeros(2), np.eye(2), target=1.0)
    assert zero.failure == "zero_gradient"
    maxed = projector.project_to_level_set(
        np.ones(2), np.eye(2), target=3.0, max_iterations=0
    )
    assert maxed.failure == "max_iterations"
    with pytest.raises(ValueError, match="quadratic matrix"):
        projector.project_to_level_set(np.ones(2), np.ones((2, 3)), target=1.0)
    with pytest.raises(ValueError, match="finite"):
        projector.project_to_level_set(np.asarray([np.nan, 1.0]), np.eye(2), target=1.0)
    budgeted = projector.project_to_level_set(
        np.ones(2), np.eye(2), target=8.0, max_distance=1e-9
    )
    assert budgeted.failure == "cost_budget_exceeded"
    with pytest.raises(ValueError, match="two-dimensional"):
        projector.grid_features(np.ones((2, 2, 1)))
    with pytest.raises(ValueError, match="finite and non-empty"):
        projector.grid_features(np.empty((0, 0)))
    with pytest.raises(ValueError, match="share one two-dimensional shape"):
        projector.project_prediction(
            np.ones((2, 2)), np.ones((3, 3)), projector.InvariantProjectionConfig()
        )
    disabled = projector.project_prediction(
        np.zeros((2, 2)), np.ones((2, 2)), projector.InvariantProjectionConfig()
    )
    assert disabled.grid.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(ValueError, match="requires a quadratic matrix"):
        projector.project_prediction(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            projector.InvariantProjectionConfig(enabled=True),
        )
    assert "game" not in inspect.signature(projector.grid_features).parameters
    assert "observed_next" not in inspect.signature(projector.project_prediction).parameters


def test_scenario_arc_wmte_6611_live_default_off_and_explicit_opt_in() -> None:
    """SCENARIO-ARC-WMTE-6611-LIVE: the scored wrapper is inert by default."""

    def engine(grid: np.ndarray, action: int, data: object) -> np.ndarray:
        del action, data
        return np.full_like(grid, 5)

    disabled = projector.InvariantProjectionConfig()
    assert projector.wrap_world_model_engine(engine, disabled) is engine
    assert projector.wrap_world_model_engine(engine) is engine
    baseline = engine(np.zeros((4, 4), dtype=np.int16), 1, None)

    enabled = projector.InvariantProjectionConfig(
        enabled=True,
        quadratic_matrix=((1.0, 0.0), (0.0, 0.0)),
        max_iterations=32,
    )
    wrapped = projector.wrap_world_model_engine(engine, enabled)
    changed = wrapped(np.zeros((4, 4), dtype=np.int16), 1, None)
    assert not np.array_equal(changed, baseline)

    policy = object.__new__(live.E3AgentPolicy)
    policy.proposer = None
    policy.invariant_projection_config = enabled
    candidates = policy._world_model_candidates(engine, None)
    assert len(candidates) == 1
    assert candidates[0].engine is not engine
    assert not np.array_equal(candidates[0].engine(np.zeros((4, 4)), 1, None), baseline)
    assert inspect.signature(live.make_carnot_agent).parameters[
        "invariant_projection_config"
    ].default is None
    assert inspect.signature(live.E3AgentPolicy).parameters[
        "invariant_projection_config"
    ].default is None
    with pytest.raises(ValueError, match="requires a quadratic matrix"):
        projector.wrap_world_model_engine(
            engine, projector.InvariantProjectionConfig(enabled=True)
        )
    with pytest.raises(ValueError, match="shape"):
        projector.norm_matched_random_matrix(np.ones((3, 3)), 1)


def test_req_arc_wmte_6611_random_zero_norm_and_helper_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-6611-CONTROLS: degenerate random controls fail closed."""

    class ZeroRng:
        def normal(self, *, size: tuple[int, int]) -> np.ndarray:
            return np.zeros(size)

    monkeypatch.setattr(projector.np.random, "default_rng", lambda _seed: ZeroRng())
    with pytest.raises(ValueError, match="zero norm"):
        projector.norm_matched_random_matrix(np.eye(2), 1)

    assert mod._exact_mismatch(np.zeros((1, 1)), np.zeros((2, 2))) == 4
    with pytest.raises(ValueError, match="at least two"):
        mod.fit_and_select_invariant([])
    transition = {"current_grid": np.zeros((2, 2)), "action": 1, "data": None}
    invalid, error = mod._valid_prediction(
        lambda *_args: np.zeros((3, 3)), transition
    )
    assert invalid is None and error == "prediction_shape_mismatch"
    invalid, error = mod._valid_prediction(
        lambda *_args: np.full((2, 2), np.nan), transition
    )
    assert invalid is None and error == "prediction_invalid_values"


def test_req_arc_wmte_6611_failure_boundaries_remain_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-6611-FAILURES: resource and projection errors remain rows."""

    monkeypatch.setattr(mod.os, "sysconf", lambda _name: (_ for _ in ()).throw(ValueError()))
    assert mod._resource_receipt()["ram_total_bytes"] is None
    transition = {"current_grid": np.zeros((2, 2)), "action": 1, "data": None}
    draft = mod._proposal_draft(
        engine=lambda *_args: np.ones((2, 2)),
        transition=transition,
        arm=mod.ARMS[1],
        config=projector.InvariantProjectionConfig(enabled=True),
        basis_sha256="sha256:test",
    )
    assert draft["runtime_valid"] is False
    assert "quadratic matrix" in draft["failure"]


def test_req_arc_wmte_6611_invalid_calibration_row_is_retained_in_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-6611-FAILURES: an invalid calibration proposal is not fitted."""

    original = mod._valid_prediction
    eligible_count = len(mod._world_model_sources(REPO, [row["game"] for row in mod._archive_sources(REPO)]))
    calls = 0

    def one_invalid(engine: object, transition: object) -> tuple[np.ndarray | None, str | None]:
        nonlocal calls
        calls += 1
        if calls == eligible_count + 1:
            return None, "forced_calibration_failure"
        return original(engine, transition)

    monkeypatch.setattr(mod, "_valid_prediction", one_invalid)
    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
        max_transitions_per_game=1,
        seeds=(6611,),
    )
    assert report["status"] == "complete_live_reachable_comparison"


def test_scenario_arc_wmte_6611_split_and_selection_are_held_blind() -> None:
    """SCENARIO-ARC-WMTE-6611-SPLIT: held contents cannot tune selection."""

    games = [f"g{i:02d}" for i in range(8)]
    split = mod.freeze_game_split(games, seed=6611)
    assert set(split["calibration_games"]).isdisjoint(split["held_games"])
    assert len(split["calibration_games"]) >= 2
    assert len(split["held_games"]) >= 2

    calibration = []
    for index in range(6):
        current = np.full((4, 4), index % 3, dtype=np.int16)
        predicted = np.full((4, 4), index % 3 + 2, dtype=np.int16)
        observed = current.copy()
        calibration.append(
            {"current_grid": current, "predicted_grid": predicted, "observed_next_grid": observed}
        )
    selected = mod.fit_and_select_invariant(calibration)
    assert selected["data_scope"] == "calibration_games_only"
    assert selected["held_outcomes_used"] == 0
    assert selected["basis_sha256"].startswith("sha256:")
    assert "held" not in inspect.signature(mod.fit_and_select_invariant).parameters


def test_req_arc_wmte_6611_complete_report_recomputes_all_rows() -> None:
    """REQ-ARC-WMTE-6611-ROWS: actual archives yield complete matched held rows."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
        max_transitions_per_game=2,
        seeds=(6611,),
    )
    assert report["status"] == "complete_live_reachable_comparison"
    assert report["verdict_class"] in {"null", "circular_positive"}
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert report["live_projection_contract_ready_score"] == 1.0
    assert "solve_provenance" not in report
    assert report["arc_scope_and_non_claims"]["game_solve_claim"] is False
    assert report["archive_and_split_receipts"]["game_disjoint"] is True
    assert len(report["archive_and_split_receipts"]["calibration_games"]) >= 2
    assert len(report["archive_and_split_receipts"]["held_games"]) >= 2
    assert all(row["held_outcomes_used"] == 0 for row in report["invariant_selection_rows"])
    assert all(row["observation_opened_after_prediction"] for row in report["per_unit_rows"])
    assert {row["arm"] for row in report["per_unit_rows"]} == set(mod.ARMS)

    held_games = report["archive_and_split_receipts"]["held_games"]
    expected = len(held_games) * 2 * len(mod.ARMS)
    assert len(report["per_unit_rows"]) == expected
    assert report["held_arm_summary"] == mod.summarize_held_rows(report["per_unit_rows"])
    assert all(row["detected"] and row["failed_closed"] for row in report["attack_rows"])
    assert mod.validate_report(report, REPO) == []


def test_scenario_arc_wmte_6611_attacks_and_validator_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-6611-ATTACKS: decision-bearing tamper is rejected."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
        max_transitions_per_game=1,
        seeds=(6611,),
    )
    assert {row["attack_id"] for row in report["attack_rows"]} == set(mod.ATTACK_IDS)

    bad = deepcopy(report)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "per_unit_rows coverage mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["live_import_reachability_receipts"]["default_enabled"] = True
    assert "projector default must be off" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["held_arm_summary"] = []
    assert "held_arm_summary mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["archive_and_split_receipts"]["game_disjoint"] = False
    assert "calibration and held games overlap" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["protected_files_unchanged"]["all_unchanged"] = False
    assert "protected files changed" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad, REPO)
    bad = deepcopy(report)
    bad["solve_provenance"] = {"forbidden": True}
    assert "solve_provenance forbidden" in mod.validate_report(_checksum(bad), REPO)[0]
    bad = deepcopy(report)
    bad["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["inference_substrate"] = "new_llm"
    assert "inference_substrate mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["per_unit_rows"][0]["observation_opened_after_prediction"] = False
    assert "observation opened before prediction" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    random_row = next(row for row in bad["per_unit_rows"] if row["arm"] == mod.ARMS[2])
    random_row["random_basis_norm_match_error"] = 1.0
    assert "random basis norm mismatch" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["invariant_selection_rows"][0]["held_outcomes_used"] = 1
    assert "held leakage in invariant selection" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["attack_rows"] = []
    assert "attack_rows incomplete" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["protected_files_unchanged"]["rows"][0]["after_sha256"] = "sha256:bad"
    assert "protected file current hash mismatch" in mod.validate_report(_checksum(bad), REPO)


def test_req_arc_wmte_6611_named_insufficiency_block() -> None:
    """REQ-ARC-WMTE-6611-ARCHIVE: too few games blocks without a proxy split."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.1,
        tests_run=_test_receipts(),
        precondition_overrides={"valid_game_count": 3},
    )
    assert report["status"] == "blocked_insufficient_game_disjoint_live_transitions"
    assert report["verdict_class"] == "blocked"
    assert report["per_unit_rows"] == []
    assert report["live_projection_contract_ready_score"] == 0.0
    failed = report["gate_check_summary"]["failed_checks"]
    assert failed[0]["check_id"] == "minimum_game_disjoint_world_model_archives"
    assert failed[0]["observed_value"] == 3
    assert failed[0]["required_value"] == 4
    assert mod.validate_report(report, REPO) == []

    bad = deepcopy(report)
    bad["gate_check_summary"]["failed_checks"] = []
    assert "blocked report lacks exact failed check" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["per_unit_rows"] = [{"fabricated": True}]
    assert "blocked report fabricated per_unit_rows" in mod.validate_report(_checksum(bad), REPO)
    bad = deepcopy(report)
    bad["live_projection_contract_ready_score"] = 1.0
    assert "blocked report cannot be contract ready" in mod.validate_report(_checksum(bad), REPO)


def test_scenario_arc_wmte_6611_atomic_write_and_receipt_reuse(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6611-ATOMIC: output validates and replaces atomically."""

    report = mod.build_report(
        REPO,
        date="20260825",
        duration_s=0.5,
        tests_run=_test_receipts(),
        max_transitions_per_game=1,
        seeds=(6611,),
    )
    target = tmp_path / "experiment_6611.json"
    receipt = mod.atomic_write_report(target, report, repo_root=REPO)
    assert receipt == {
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync": True,
    }
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert mod.existing_test_receipts(target) == _test_receipts()
    assert mod.existing_test_receipts(tmp_path / "missing.json") == list(mod.DEFAULT_TESTS_RUN)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.existing_test_receipts(malformed) == list(mod.DEFAULT_TESTS_RUN)
    invalid = deepcopy(report)
    del invalid["status"]
    with pytest.raises(ValueError, match="missing required field: status"):
        mod.atomic_write_report(tmp_path / "invalid.json", invalid, repo_root=REPO)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        mod.atomic_write_report(tmp_path / "replace_error.json", report, repo_root=REPO)
    assert not list(tmp_path.glob(".replace_error.json.*.tmp"))
    monkeypatch.undo()


def test_scenario_arc_wmte_6611_cli_boundary_uses_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-ARC-WMTE-6611-ATOMIC: CLI reuses receipts and delegates writing."""

    calls: dict[str, object] = {}
    fake = {
        "status": "complete_live_reachable_comparison",
        "verdict_class": "null",
        "per_unit_rows": [],
    }
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "existing_test_receipts", lambda _path: _test_receipts())

    def fake_build(_root: Path, *, date: str, tests_run: object) -> dict[str, object]:
        calls["date"] = date
        calls["tests_run"] = tests_run
        return fake

    def fake_write(path: Path, payload: object, *, repo_root: Path) -> dict[str, bool]:
        calls["path"] = path
        calls["payload"] = payload
        calls["repo_root"] = repo_root
        return {"file_fsync": True}

    monkeypatch.setattr(mod, "build_report", fake_build)
    monkeypatch.setattr(mod, "atomic_write_report", fake_write)
    assert mod.main(["--date", "20260825"]) == 0
    assert calls["date"] == "20260825"
    assert calls["payload"] is fake
    assert str(tmp_path) in str(calls["path"])
    assert '"verdict_class": "null"' in capsys.readouterr().out
