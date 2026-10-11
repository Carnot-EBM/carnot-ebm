"""REQ-REPORT-8398 / REQ-VERIFY-8398: fixtures qualify mechanics only."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_generalization_panel_8398 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from test_arc_supervisor_live_panel_8384 import Arcade, work


def test_frozen_panel_and_absent_slots() -> None:
    """SCENARIO-REPORT-8398-PANEL: metadata changes cannot replace an absent frozen game."""
    original = json.loads((p.ROOT / p.ORIGINAL).read_text())["frozen_panel"]
    panel = p.freeze([], [])
    assert panel["units"] == original["units"]
    assert panel["selected_games"] == original["selected_games"]
    assert panel["missing_games"] == original["selected_games"]
    present = p.freeze(original["selected_games"] + ["easier-new-game"], [])
    assert present["missing_games"] == []
    assert present["units"] == panel["units"]


def test_current_preconditions_and_scoped_plan(tmp_path: Path) -> None:
    """REQ-VERIFY-8398: current authority qualifies without a historical design repair."""
    checks, hashes, resources = p.preconditions(tmp_path, p.ROOT)
    assert not checks
    assert resources["task_sha256"] == p.TASK_PIN
    assert hashes[str(p.ROOT / p.ORIGINAL)] == p.ORIGINAL_PIN
    specs = p.plan(tmp_path)
    assert all(s["scope"] == "owned" for s in specs)
    assert any(s["name"] == "private_E2E011_017_wrapper" for s in specs)
    assert any(s["name"] == "coverage_report" for s in specs)
    checks, _, _ = p.preconditions(tmp_path, tmp_path)
    assert any(c["observed"] is None for c in checks)


def test_bound_identity_replay_and_real_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8398-REPLAY: no observation becomes a fabricated zero."""
    w = work(tmp_path)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["experiment_id"] == 8398 and value["task_id"] == p.TASK
    assert value["milestone"] == "2026.10.723"
    assert value["verdict_class"] == "blocked" and value["completed_count"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["llm_calls"] == 0
    assert set(value) <= set(value["field_principles"])
    assert p.replay(value)
    assert all(r["passed"] for r in p.controls(value, tmp_path))
    changed = deepcopy(value)
    changed["completed_count"] += 1
    assert not p.replay(changed)
    assert not p.replay({})


def test_original_hash_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8398-CONTROLS: substituted history cannot choose a new panel."""
    monkeypatch.setattr(p, "ORIGINAL_PIN", "sha256:wrong")
    with pytest.raises(ValueError, match="frozen_panel_hash"):
        p.freeze([], [])
    checks, _, _ = p.preconditions(tmp_path, p.ROOT)
    assert any(c["check"] == "frozen_panel_hash" for c in checks)


def test_bindings_restore() -> None:
    """REQ-REPORT-8398: invocation patches never change shipped defaults on disk."""
    original = p.base.NAME
    with p.bindings():
        assert p.base.NAME == p.NAME and p.base.TASK_PIN == p.TASK_PIN
        assert p.base.build is p.build
    assert p.base.NAME == original


def test_run_live_constructed_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8398-PANEL: only actual wrapper actions enter primitive reduction."""
    panel = p.freeze([], [])
    arcade = Arcade()
    arcade.available_environments = [SimpleNamespace(game_id=g) for g in panel["selected_games"]]
    monkeypatch.setattr(p, "sdk", lambda: arcade)
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p, "execute", lambda specs, logs: [dict(passed=True, scope="owned")])
    real_child = p.child

    def episode_child(name: str, argv: list[str], logs: Path, **kwargs: object) -> dict:
        if not name.startswith("episode_"):
            return real_child(name, argv, logs, **kwargs)
        unit = json.loads(Path(argv[argv.index("--episode") + 1]).read_text())
        raw = Path(argv[argv.index("--raw") + 1])
        test_unit = dict(unit, max_actions=4)
        p.runtime.episode(test_unit, arcade, raw)
        return dict(passed=True, scope="measurement", name=name)

    monkeypatch.setattr(p, "child", episode_child)
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_text())
    assert value["completed_count"] == 8 and value["verdict_class"] == "null"
    assert value["arc_panel_ready_score"] == 1
    assert all(r["action_count"] == 4 for r in value["rows"])
    assert arcade.seeds == [11, 11, 22, 22] * 2
    assert p.replay(value)
    with monkeypatch.context() as local:
        original_hash = p.sha256_file
        snapshot_path = value["source_input_receipts"][0]["snapshot_path"]
        local.setattr(
            p,
            "sha256_file",
            lambda path: "invalid" if str(path) == snapshot_path else original_hash(path),
        )
        assert not p.replay(value)
    with monkeypatch.context() as local:
        current = p.authority.authority
        local.setattr(
            p.authority, "authority", lambda root, raw: dict(current(root, raw), activated=False)
        )
        assert not p.replay(value)
    with monkeypatch.context() as local:
        freezing = p.freeze
        local.setattr(
            p,
            "freeze",
            lambda roster, registry: dict(freezing(roster, registry), installed_roster=[]),
        )
        assert not p.replay(value)


@pytest.mark.parametrize("mode", ["absent", "sdk_error", "qualification"])
def test_run_blocked_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-8398-PANEL: external absence blocks and owned failures disqualify."""

    def sdk() -> object:
        if mode == "sdk_error":
            raise ImportError("constructed absent SDK")
        return SimpleNamespace(available_environments=[])

    monkeypatch.setattr(p, "sdk", sdk)
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(
        p, "execute", lambda specs, logs: [dict(passed=mode != "qualification", scope="owned")]
    )
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_text())
    assert value["intended_count"] == 8 and value["completed_count"] == 0
    assert value["verdict_class"] == ("disqualified" if mode == "qualification" else "blocked")
    assert all(r["absolute_metric"] is None for r in value["rows"])


def test_real_cli_and_consumers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8398-CONTROLS: real children publish and replay private blocked bytes."""
    from carnot.reporting.primary_publication import reader_receipt

    output = tmp_path / (p.NAME + ".json")
    receipt = child(
        "private_cli",
        [
            str(p.ROOT / ".venv/bin/python"),
            "-u",
            str(p.ROOT / p.CLI),
            "--private-e2e",
            "--output",
            str(output),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"]
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    read = reader_receipt(p.TASK, tmp_path, field="arc_panel_ready_score", expected=0)
    assert read["passed"] and read["gate_sha256"] == sha256_file(output)
    assert p.replay(value)
    for name, argv, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        (
            "episode_error",
            ["--episode", str(tmp_path / "absent"), "--raw", str(tmp_path / "episode")],
            1,
        ),
    ]:
        checked = child(
            name,
            [str(p.ROOT / ".venv/bin/python"), "-u", str(p.ROOT / p.CLI), *argv],
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )
        assert checked["passed"]


def test_historical_preconditions_use_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-8398: frozen V722 tests retain their original assertions and task authority."""
    calls = []
    monkeypatch.setattr(pytest, "main", lambda args: calls.append(args) or 0)
    assert p.historical_checks(tmp_path) == 0
    assert p.base.TEST in calls[0]
    with monkeypatch.context() as local:
        local.setattr(p, "sha256_file", lambda path: "invalid")
        with pytest.raises(ValueError, match="historical_snapshot_hash"):
            p.historical_checks(tmp_path)


def test_reduction_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8398-CONTROLS: malformed operands and changed panel units reject."""
    monkeypatch.setattr(p, "BASE_FREEZE", lambda roster, registry: dict(units=[]))
    with pytest.raises(ValueError, match="frozen_panel_units"):
        p.freeze([], [])
    monkeypatch.setattr(p, "BASE_REPLAY", lambda value: True)
    assert not p.replay({})
