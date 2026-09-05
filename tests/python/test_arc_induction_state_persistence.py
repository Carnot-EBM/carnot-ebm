"""REQ-ARC-WMTE-7040/7041/7042: CPU request tests, with no live model."""

from __future__ import annotations

import io
import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_induction_memory import InductionMemory

pytestmark = pytest.mark.memory_watchdog_skip

CODE = """import numpy as np
def engine(grid, action, data):
    return grid.copy()
def is_level_complete(grid):
    return False
"""


def rows(n=10, *, level=0, shape=(2, 2)):
    return [
        e3.Transition(
            np.full(shape, i, dtype=np.int16),
            1,
            None,
            np.full(shape, i + 1, dtype=np.int16),
            level,
            level,
        )
        for i in range(n)
    ]


@pytest.fixture
def transport(monkeypatch, tmp_path):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path / "engines")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_LOOP", raising=False)
    p = e3.LocalGGUFProposer(
        model_path="/unused.gguf",
        ffn_cpu_layers=0,
        mtp=False,
        n_ctx=98304,
        max_tokens=131072,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    monkeypatch.setattr(p, "observed_n_ctx", lambda: 98304)
    payloads = []
    replies = [CODE]

    def request(req, timeout=None):
        payloads.append(json.loads(req.data))
        code = replies.pop(0) if len(replies) > 1 else replies[0]
        response = (
            {
                "choices": [
                    {"message": {"content": f"```python\n{code}```"}, "finish_reason": "stop"}
                ]
            }
            if "/chat/completions" in req.full_url
            else {"content": f"```python\n{code}```", "stop_type": "eos"}
        )
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    return p, payloads, replies


def state_in(prompt):
    assert "PRIOR INDUCTION STATE\n" in prompt
    return json.loads(
        prompt.split("PRIOR INDUCTION STATE\n", 1)[1].split("\nEND PRIOR STATE", 1)[0]
    )


@pytest.mark.parametrize("flag", [None, "0", "true", "1"])
@pytest.mark.parametrize("think", [False, True])
def test_second_request_and_output_reserve(monkeypatch, transport, flag, think):
    """SCENARIO-ARC-WMTE-7040-A / 7041-B: inspect the actual HTTP body."""
    if flag is None:
        monkeypatch.delenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", raising=False)
    else:
        monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", flag)
    p, sent, _ = transport
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1" if think else "0")
    memory = InductionMemory()
    for _ in range(2):
        assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    assert len(sent) == 2
    if flag != "1":
        assert sent[0] == sent[1]
        assert memory.receipt()["deliveries"] == 0
        assert memory.receipt()["stored_source_bytes"] == 0
        assert memory.receipt()["calls"] == 0
    else:
        prompts = [s["messages"][0]["content"] if think else s["prompt"] for s in sent]
        state = state_in(prompts[1])
        assert state["source"] == CODE.strip()
        assert state["refutations"][-1]["wrong_cells"][-1][-2:] == [9, 10]
        assert state["refutations"][-1]["action"] == 1
        added = len(prompts[1].encode()) - len(prompts[0].encode())
        assert 0 < added <= 4096
        budget = "max_tokens" if think else "n_predict"
        assert sent[1][budget] == sent[0][budget] - added
        assert memory.receipt()["added_bytes"] == added


@pytest.mark.parametrize("change", ["game", "level", "shape", "cell"])
def test_scope_change_clears_source_and_refutations(monkeypatch, transport, change):
    """SCENARIO-ARC-WMTE-7040-B: similar games and levels must not share claims."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, replies = transport
    memory = InductionMemory()
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    args = (
        "other" if change == "game" else "g",
        rows(level=1 if change == "level" else 0, shape=(3, 3) if change == "shape" else (2, 2)),
        2 if change == "cell" else 1,
    )
    replies[:] = [CODE.replace("grid.copy()", "grid + 1")]
    assert p.induce(*args, induction_memory=memory)[0]
    assert "PRIOR INDUCTION STATE" not in sent[-1]["prompt"]
    assert memory.receipt()["scope_resets"] == 1

    assert p.induce(*args, induction_memory=memory)[0]
    assert state_in(sent[-1]["prompt"])["refutations"] == []


def test_compaction_and_dedup_at_request_site(monkeypatch, transport):
    """SCENARIO-ARC-WMTE-7041-A: history, JSON, and complete source stay bounded."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, replies = transport
    memory = InductionMemory()
    replies[:] = [CODE + "#" + "x" * 6000]
    for _ in range(12):
        assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert state["source"] is None
    assert state["source_omitted_bytes"] > 6000
    assert len(state["refutations"]) == 4
    assert memory.receipt()["stored_refutations"] == 4
    assert memory.receipt()["last_added_bytes"] <= 4096
    assert memory.receipt()["compactions"] > 0
    replies[:] = [CODE + "#" + "é" * 20000]
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    assert memory.receipt()["stored_source_bytes"] == 0
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    assert state_in(sent[-1]["prompt"])["source_omitted_bytes"] > 32768
    assert state_in(sent[-1]["prompt"])["execution_error"] is None


def test_probe_population_and_boundary_exclusion(monkeypatch, transport):
    """SCENARIO-ARC-WMTE-7040-B: only eight non-boundary proposal rows are probed."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    current = rows(20)
    current[-1].level_after = 1
    current[-2].level_before = current[-2].level_after = 7
    current[-3].grid = np.full((3, 3), 17, dtype=np.int16)
    current[-3].next_grid = np.full((3, 3), 18, dtype=np.int16)
    assert p.induce("g", current, 1, induction_memory=memory)[0]
    assert memory.receipt()["probed_transitions"] == 8
    assert all(r["wrong_cells"][0][-1] < 18 for r in state_in(sent[-1]["prompt"])["refutations"])


@pytest.mark.parametrize("where", ["compile", "engine"])
def test_execution_error_is_not_a_refuted_dynamics_claim(monkeypatch, transport, where):
    """REQ-ARC-WMTE-7040: a timed-out engine has no measured predictions."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    slow = (
        "import time\ntime.sleep(0.4)\n" + CODE
        if where == "compile"
        else CODE.replace("return grid.copy()", "__import__('time').sleep(0.4); return grid.copy()")
    )
    memory.remember(slow)
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert state["execution_error"] == "EngineCallTimeout"
    assert state["refutations"] == []


def test_two_policy_instances_do_not_share_state(monkeypatch, transport):
    """SCENARIO-ARC-WMTE-7040-B: the server may be shared; memory may not."""
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    a = E3AgentPolicy("g", proposer=p, value_head=None)
    b = E3AgentPolicy("g", proposer=p, value_head=None)
    for owner in (a, a, b):
        assert p.induce("g", rows(), 1, induction_memory=owner._induction_memory)[0]
    assert "PRIOR INDUCTION STATE" in sent[1]["prompt"]
    assert "PRIOR INDUCTION STATE" not in sent[2]["prompt"]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_live_policy_branches_deliver_memory(monkeypatch, transport, bounded, enabled):
    """SCENARIO-ARC-WMTE-7042-A: start at the scored policy, inspect HTTP requests."""
    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_llm_reinduction as reinduce
    from scripts.arc_leaderboard_eval import _policy_diagnostics

    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1" if enabled else "0")
    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "1" if bounded else "0")
    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
    p, sent, _ = transport
    received = []
    induce = p.induce

    def recording(*args, **kwargs):
        received.append(kwargs)
        return induce(*args, **kwargs)

    monkeypatch.setattr(p, "induce", recording)
    policy = agent.E3AgentPolicy("g", proposer=p, value_head=None)
    policy.transitions = rows(12)
    policy.root_grid = policy.transitions[0].grid
    policy.cell = 1
    policy._episode_transition_start = 0
    policy.program_synthesis_filter_enabled = False
    policy.active_probe_controller_enabled = False
    policy.think_arm_fallback_enabled = False
    policy.max_refinement_rounds = 1
    # Other model tiers are independent of the induction interface under test.
    monkeypatch.setenv("CARNOT_ARC_STRUCTURED_NAV", "0")
    monkeypatch.setenv("CARNOT_ARC_LIVE_TTT", "0")
    monkeypatch.setattr(reinduce, "MAX_REFINEMENT_ROUNDS", 1)
    for _ in range(2):
        policy._induce_and_plan()
    assert sent, policy.induction_attempts
    with_state = [s for s in sent if "PRIOR INDUCTION STATE" in s.get("prompt", "")]
    assert all(("induction_memory" in kw) == enabled for kw in received)
    if not enabled:
        assert not with_state
        return
    assert with_state, policy.induction_attempts
    state = state_in(with_state[-1]["prompt"])
    assert state["source"] == CODE.strip()
    # The acceptance tail's values 11 and 12 must never become mismatch evidence.
    assert all(r["wrong_cells"][0][-1] < 11 for r in state["refutations"])
    assert policy.induction_attempts[0].get("stall_refactor_loop_used", False) is bounded
    assert _policy_diagnostics(policy)["induction_memory"]
    assert _policy_diagnostics(policy)["induction_memory"]["deliveries"] > 0
    assert (
        policy.generator_liveness_witness()["induction_memory"]
        == _policy_diagnostics(policy)["induction_memory"]
    )


def test_legacy_proposer_accepts_memory_enabled_policy(monkeypatch):
    """REQ-ARC-WMTE-7042: older proposer signatures still receive their normal arguments."""
    from carnot.agentic.arc_llm_reinduction import _call_induce

    class Legacy:
        def induce(self, game, transitions, cell):
            assert game == "g" and len(transitions) == 2 and cell == 1
            return True, "legacy"

    try:
        outcome = _call_induce(Legacy(), "g", rows(2), 1, None, InductionMemory())
    except Exception as exc:
        outcome = type(exc).__name__
    assert outcome == (True, "legacy")


def test_insufficient_room_never_sends_memory(monkeypatch, transport):
    """REQ-ARC-WMTE-7041: refuse locally when extra context leaves no usable output budget."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    assert p.induce("g", rows(), 1, induction_memory=memory)[0]
    p.max_tokens = 1024
    ok, note = p.induce("g", rows(), 1, induction_memory=memory)
    assert not ok and "insufficient completion space" in note
    assert len(sent) == 1
    assert memory.receipt()["deliveries"] == 0


def test_offline_twin_dispatch_and_output(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-7042-B: the actual CLI branch calls the scored runner."""
    from scripts import arc_leaderboard_eval as evaluate
    from scripts import arc_loop_solve as twin
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    def run(game, policy, *, budget):
        assert game == "r11l" and isinstance(policy, E3AgentPolicy) and budget == 7
        return {"game": game, "policy_diagnostics": evaluate._policy_diagnostics(policy)}

    monkeypatch.setattr(evaluate, "run_game", run)
    output = tmp_path / "eval.json"
    assert (
        twin.main(
            ["--game", "r11l", "--mechanism", "e3", "--max-actions", "7", "--output", str(output)]
        )
        == 0
    )
    receipt = json.loads(output.read_text())
    assert receipt.get("policy_diagnostics", {}).get("induction_memory", {}).get("deliveries") == 0


@pytest.mark.parametrize("missing", [False, True])
def test_offline_twin_requires_output_outside_evidence(monkeypatch, tmp_path, missing):
    """REQ-ARC-WMTE-7042: use only disposable paths, including during mutation."""
    from scripts import arc_loop_solve as twin
    from scripts import arc_leaderboard_eval as evaluate

    monkeypatch.setattr(twin, "REPO", tmp_path)
    monkeypatch.setattr(evaluate, "run_game", lambda *a, **kw: {})
    args = ["--game", "r11l", "--mechanism", "e3"]
    if not missing:
        args += ["--output", str(tmp_path / "results" / "forbidden.json")]
    try:
        twin.main(args)
        status = "returned"
    except SystemExit as exc:
        status = exc.code
    except Exception as exc:
        status = type(exc).__name__
    assert status == 2
    assert not (tmp_path / "results").exists()


def test_duplicate_and_correct_predictions(monkeypatch, transport):
    """REQ-ARC-WMTE-7040: reuse one refutation; exact predictions add none."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    for _ in range(4):
        assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert len(state["refutations"]) == 1
    old_sha = state["source_sha256"]
    memory.remember(CODE.replace("return grid.copy()", "return grid + 1"))
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert state["source_sha256"] != old_sha
    assert len(state["refutations"]) == 1
    assert state["refutations"][0]["source_sha256"] == old_sha


@pytest.mark.parametrize(
    "prediction", ["grid + 0.1", "np.full(grid.shape, 10**5000, dtype=object)", "grid + 256"]
)
def test_malformed_predictions_are_errors(monkeypatch, transport, prediction):
    """REQ-ARC-WMTE-7040: malformed values cannot falsify or crash feedback."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    memory.remember(CODE.replace("grid.copy()", prediction))
    try:
        ok, _ = p.induce("g", rows(1), 1, induction_memory=memory)
        state = state_in(sent[-1]["prompt"])
    except Exception as exc:
        ok, state = False, {"escaped": type(exc).__name__}
    assert ok, state
    assert state["execution_error"] == "ValueError"
    assert state["refutations"] == []


def test_probe_does_not_mutate_inputs(monkeypatch, transport):
    """REQ-ARC-WMTE-7040: generated code cannot alter subsequent evidence."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    evidence = rows(1)
    evidence[0].data = {"x": 1, "nested": [2]}
    assert p.induce("g", evidence, 1, induction_memory=memory)[0]
    memory.remember(
        CODE.replace("return grid.copy()", "grid[:] = 8; data['nested'][0] = 99; return grid")
    )
    assert p.induce("g", evidence, 1, induction_memory=memory)[0]
    assert evidence[0].grid.tolist() == [[0, 0], [0, 0]]
    assert evidence[0].data == {"x": 1, "nested": [2]}
    assert state_in(sent[-1]["prompt"])["refutations"]
    assert state_in(sent[-1]["prompt"])["refutations"][0]["data"] == {"x": 1}


def test_action_coordinate_bound(monkeypatch, transport):
    """REQ-ARC-WMTE-7041: malformed metadata cannot create unbounded JSON."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    evidence = rows(1)
    assert p.induce("g", evidence, 1, induction_memory=memory)[0]
    evidence[0].data = {"x": 10**100}
    assert p.induce("g", evidence, 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert state["execution_error"] == "ValueError"
    assert state["refutations"] == []


def test_chat_continuation_counts_each_response(monkeypatch, transport):
    """REQ-ARC-WMTE-7042: continuation repeats memory and pays twice."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    monkeypatch.setenv("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION", "1")
    p, _, _ = transport
    sent = []

    def request(req, timeout=None):
        sent.append(json.loads(req.data))
        msg = {"content": f"```python\n{CODE}```"}
        if len(sent) == 2:
            msg = {"content": "", "reasoning_content": "The prior identity rule missed a change."}
        return io.BytesIO(
            json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]}).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", request)
    memory = InductionMemory()
    for _ in range(2):
        assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert len(sent) == 3
    assert len(sent[-1]["messages"]) == 2
    assert memory.receipt()["deliveries"] == 2
    assert memory.receipt()["added_bytes"] == 2 * memory.receipt()["last_added_bytes"]
    assert state_in(sent[-1]["messages"][0]["content"])["refutations"]


@pytest.mark.parametrize("full_engine_reply", [False, True])
def test_split_fallback_delivers_and_retains_state(monkeypatch, transport, full_engine_reply):
    """REQ-ARC-WMTE-7040/7041: engine retries share state; focused goals do not."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, replies = transport
    memory = InductionMemory()
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    monkeypatch.setenv("CARNOT_ARC_GOAL_DEDUP", "1")
    engine = (
        CODE.replace("return False", "return bool(np.all(grid == 1))")
        if full_engine_reply
        else CODE.split("def is_level_complete")[0]
    )
    replies[:] = ["pass", engine, "def is_level_complete(grid):\n    return False\n"]
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert len(sent) == (3 if full_engine_reply else 4)
    for s in sent[1:3]:
        assert state_in(s["prompt"])["source"] == CODE.strip()
        assert s["n_predict"] == sent[0]["n_predict"] - memory.last_added_bytes
    if not full_engine_reply:
        assert "PRIOR INDUCTION STATE" not in sent[-1]["prompt"]
    assert memory.deliveries == 2
    assert "def engine" in memory.source and "def is_level_complete" in memory.source
    memory.remember("")
    replies[:] = ["pass", engine, "def is_level_complete(grid):\n    return False\n"]
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert "def engine" in memory.source and "def is_level_complete" in memory.source


def test_failed_publication_does_not_replace_source(monkeypatch, transport):
    """REQ-ARC-WMTE-7040: only published candidates become the prior source."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, _, replies = transport
    memory = InductionMemory()
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    replies[:] = [CODE.replace("grid.copy()", "grid + 1")]
    monkeypatch.setattr(p, "_write_world_model", lambda *a, **kw: (False, "failed write"))
    assert not p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert memory.source == CODE.strip()


def test_successful_tool_loop_reports_unsupported(monkeypatch, transport):
    """REQ-ARC-WMTE-7042: an excluded transport must not appear as a measured null."""
    from carnot.agentic import arc_induction_tool_loop as loop

    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    monkeypatch.setattr(loop, "induce_with_tool_loop", lambda *a, **kw: (True, "tool success"))
    p, sent, _ = transport
    memory = InductionMemory()
    assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert memory.receipt()["unsupported_tool_calls"] == 1
    assert memory.receipt()["deliveries"] == 0
    assert sent == []


def test_vllm_completion_counts_memory(monkeypatch, transport):
    """REQ-ARC-WMTE-7041/7042: the optional raw backend pays and records the same addition."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    monkeypatch.setattr(e3, "_vllm_backend_active", lambda: True)
    p, _, _ = transport
    sent = []

    def request(req, timeout=None):
        sent.append(json.loads(req.data))
        assert req.full_url.endswith("/v1/completions")
        return io.BytesIO(
            json.dumps(
                {"choices": [{"text": f"```python\n{CODE}```", "finish_reason": "stop"}]}
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", request)
    memory = InductionMemory()
    for _ in range(2):
        assert p.induce("g", rows(1), 1, induction_memory=memory)[0]
    assert state_in(sent[1]["prompt"])["source"] == CODE.strip()
    assert sent[1]["max_tokens"] == sent[0]["max_tokens"] - memory.last_added_bytes
    assert memory.deliveries == 1
    assert memory.added_bytes == memory.last_added_bytes


def test_large_grid_feedback_samples_stay_bounded(monkeypatch, transport):
    """REQ-ARC-WMTE-7041: 4096 wrong cells still produce four samples per record."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    for _ in range(2):
        assert p.induce("g", rows(10, shape=(64, 64)), 1, induction_memory=memory)[0]
    state = state_in(sent[-1]["prompt"])
    assert all(
        r["wrong_cell_count"] == 4096 and len(r["wrong_cells"]) == 4 for r in state["refutations"]
    )
    assert memory.last_added_bytes <= 4096


def test_bounded_helper_with_previous_grid(monkeypatch, transport):
    """REQ-ARC-WMTE-7042: level-up induction forwards both the grid and memory."""
    from carnot.agentic.arc_llm_reinduction import _call_induce

    monkeypatch.setenv("CARNOT_ARC_INDUCE_STATE_PERSISTENCE", "1")
    p, sent, _ = transport
    memory = InductionMemory()
    for _ in range(2):
        assert _call_induce(p, "g", rows(1), 1, np.zeros((2, 2), dtype=np.int16), memory)[0]
    assert state_in(sent[-1]["prompt"])["source"] == CODE.strip()
