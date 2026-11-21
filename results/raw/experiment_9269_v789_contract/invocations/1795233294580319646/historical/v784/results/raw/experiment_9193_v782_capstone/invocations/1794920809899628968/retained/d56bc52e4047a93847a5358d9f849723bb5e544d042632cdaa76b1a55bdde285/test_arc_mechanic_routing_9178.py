"""REQ-REPORT-9178-V781 / SCENARIO-REPORT-9178-V781: routing controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.agentic import arc_solver_kit as kit

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "results/raw/experiment_9178_v781_arc_mechanic_routing/inputs/baseline.json"


def test_frozen_matrix():
    """Explicit mechanic routing must survive identity collisions without changing legacy rows."""
    baseline = json.loads(BASELINE.read_bytes())
    mismatches = []
    for row in baseline["rows"]:
        actual = [
            o.operator
            for o in kit.select_primitive_operators(
                mechanic_class=row["mechanic"],
                action_model=row["action_model"],
                game=row["game_name"],
            )
        ]
        reference = [
            o.operator
            for o in kit.select_primitive_operators(
                mechanic_class=row["mechanic"], action_model=row["action_model"], game=""
            )
        ]
        if row["recognized"]:
            if row["before"] != reference:
                mismatches.append(row["unit_id"])
            assert actual == reference, row
        else:
            assert actual == row["before"], row
    assert mismatches, "The source-visible pre-correction collision must be retained."


@pytest.mark.parametrize(
    "mechanic",
    [
        "cast_grid",
        "cast grid",
        "phase_fsm",
        "two_phase_cast_grid",
        "color_match",
        "color match",
        "slot_sequence",
        "slot sequence",
        "sprite_overlay",
        "sprite overlay",
        "pattern_match_sprite_resize",
        "sprite_resize",
        "resize",
        "config_substitution",
        "glyph",
        "config",
        "toggle",
        "constraint",
        "program_editor",
        "object_motion",
        "object motion",
        "reflection",
        "reflect",
        "push",
        "world_model",
        "e3",
        "graph",
    ],
)
def test_all_existing_aliases(mechanic):
    """Recognition uses existing substring semantics, including casing and overlapping labels."""
    reference = kit.select_primitive_operators(mechanic_class=mechanic.upper())
    for game in ["sc25", "sb26", "re86", "tr87", "unseen_9178"]:
        assert (
            kit.select_primitive_operators(mechanic_class=mechanic.upper(), game=game.upper())
            == reference
        )


def test_consumers(monkeypatch):
    """Both recommendation helper call sites receive controlled mechanics; no live episode runs."""
    from carnot.agentic import arc_solve_learning as learning
    from carnot.agentic import arc_strategy_router as strategy
    from carnot.agentic import arc_primitive_library as library

    monkeypatch.setattr(learning, "_registry", lambda: {"games": []})
    monkeypatch.setattr(learning, "_feature_router_payload", lambda **_: {})
    monkeypatch.setattr(learning, "typed_memory_provenance_guard", lambda: {})
    monkeypatch.setattr(library, "retrieve_primitives", lambda *_, **__: [])
    for mechanic, _, action in json.loads(BASELINE.read_bytes())["branches"]:
        monkeypatch.setattr(
            strategy, "route_for_game", lambda *_, **__: {"routed_mechanic": mechanic}
        )
        reference = [
            o.operator
            for o in kit.select_primitive_operators(mechanic_class=mechanic, action_model=action)
        ]
        for surveyed in [False, True]:
            for game in ["", "sc25", "sb26", "re86", "tr87", "unseen_9178"]:
                monkeypatch.setattr(
                    learning,
                    "_survey_features",
                    lambda: {game: {"action_type": action, "win_kw": set()}} if surveyed else {},
                )
                result = learning.recommend_approach(game, mechanic=mechanic)
                assert [o["operator"] for o in result["selected_generic_operators"]] == reference


def test_measure_and_build(tmp_path):
    """Current rows and source gates determine readiness without model activity."""
    from carnot.reporting import arc_mechanic_routing_9178 as m

    work = m.measure(ROOT, tmp_path / "raw", "20261117")
    assert all(c["passed"] for c in work["inspection"]["checks"])
    assert len(work["rows"]) == 124
    assert all(r["passed"] for r in work["rows"])
    assert sum(r["before_mismatch"] for r in work["rows"]) > 0
    value = m.build(work, {})
    assert value["verdict_class"] == "disqualified"
    assert value["mechanic_routing_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["consumer_reachability"]["classification"] == "offline_recommendation_helper_test"
    assert value["changed_behavior_rows"]
    blocked = m.measure(tmp_path / "missing", tmp_path / "blocked", "20261117")
    b = m.build(blocked, {})
    assert b["verdict_class"] == "blocked" and b["censored_count"] == 124
    assert b["inference_substrate_class"] == "blocked_no_run"
    assert any(c["observed"] == "missing_file" for c in b["gate_check_summary"])


def child(tmp_path, *args):
    """Actual private children import via the CLI and never inherit PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/experiments/experiment_9178_v781_arc_mechanic_routing.py"),
        *map(str, args),
    ]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run"]
    print("[9178] child_before", args, flush=True)
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("[9178] child_after", result.returncode, result.stderr[-300:], flush=True)
    return result


def test_private_cli(tmp_path):
    """E2E-018/021: success, external refusal, replay and rehashed tamper are real processes."""
    from carnot.reporting import arc_mechanic_routing_9178 as m

    output = tmp_path / (m.NAME + ".json")
    result = child(tmp_path, "--output", output, "--checks-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["rows"][0]["after"] = ["invented"]
    value["reproducibility_checksum"] = m.checksum(value)
    m.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad", "--output", output).returncode == 1
    assert (
        child(
            tmp_path, "--root", tmp_path / "absent", "--output", output, "--checks-only"
        ).returncode
        == 0
    )
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_replay_refusals(tmp_path, monkeypatch):
    """Source, primitive and row drift remain refusals even with freshly computed checksums."""
    from carnot.reporting import arc_mechanic_routing_9178 as m

    work = m.measure(ROOT, tmp_path / "work", "20261117")
    monkeypatch.setattr(m, "receipts_pass", lambda *_: True)
    value = m.build(work, {})
    assert value["verdict_class"] == "null" and value["mechanic_routing_ready_score"] == 1
    output = tmp_path / "experiment_9178_private.json"
    m.atomic_json(output, value)
    assert m.replay(output)["passed"]
    changed = deepcopy(value)
    changed["code_config_hashes"][0]["sha256"] = "forged"
    m.atomic_json(output, changed)
    with pytest.raises(ValueError, match="operand_hash"):
        m.replay(output)
    m.atomic_json(output, value)
    original_inspect = m.inspect
    monkeypatch.setattr(m, "inspect", lambda _: {})
    with pytest.raises(ValueError, match="source_drift"):
        m.replay(output)
    monkeypatch.setattr(m, "inspect", original_inspect)
    monkeypatch.setattr(m, "routing_rows", lambda _: [])
    with pytest.raises(ValueError, match="routing_drift"):
        m.replay(output)


def test_main_owned_path(tmp_path, monkeypatch):
    """The normal CLI executes the existing qualifier; private shortcuts cannot write a primary."""
    from carnot.reporting import arc_mechanic_routing_9178 as m

    calls = []
    monkeypatch.setattr(m.lifecycle, "qualify", lambda work: calls.append(work) or {})
    monkeypatch.setattr(m, "validate", lambda _: {"passed": True})
    assert m.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
    assert len(calls) == 1
    assert m.main(["--checks-only"]) == 1
    value = json.loads((tmp_path / (m.NAME + ".json")).read_bytes())
    assert value["mechanic_routing_ready_score"] == 0


def test_changed_selector_statement_coverage():
    """REQ-REPORT-9178-V781: every added selector statement is measured without exclusions."""
    import ast
    from coverage import Coverage

    tracer = Coverage.current()
    if tracer is None:
        return
    tree = ast.parse(Path(kit.__file__).read_text())
    selector = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_primitive_operators"
    )
    guard = next(n for n in selector.body if isinstance(n, ast.If))
    _, statements, excluded, missing, _ = tracer.analysis2(kit.__file__)
    owned = [n for n in statements if guard.lineno <= n <= guard.end_lineno]
    assert owned and not set(owned).intersection(missing)
    assert not set(owned).intersection(excluded)
    if os.environ.get("COVERAGE_FILE"):
        from carnot.reporting.current_work_receipt import atomic_json

        atomic_json(
            Path(os.environ["COVERAGE_FILE"]).parent / "selector_changed_statement_coverage.json",
            {
                "path": kit.__file__,
                "owned_statements": owned,
                "missing": [],
                "excluded": [],
                "percent": 100,
                "scope": "only statements added by9178; original selector remains out of owned percentage",
            },
        )
