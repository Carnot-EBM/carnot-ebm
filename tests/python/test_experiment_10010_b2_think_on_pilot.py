"""Tests for Experiment 10010, the B2 think-ON induction pilot harness.

REQ-ARC-WMTE-10010. Every test writes only under `tmp_path`. Real evidence under
`results/raw/` is read, never written. No test starts a model server or touches a GPU:
the server, nvidia-smi, and HTTP are all fakes.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pytest

from carnot import experiment_10010_b2_think_on_pilot as exp
from carnot import experiment_10010_engine_child as child
from carnot.agentic import arc_executable_world_model as e3

REPO = Path(exp.__file__).resolve().parents[2]
REAL_PATHS = exp.EvidencePaths.under(REPO)


@pytest.fixture(autouse=True)
def _live_default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test under the scored kernel's flag set, whatever the caller's shell has."""
    import os

    for key in [k for k in os.environ if k.startswith("CARNOT_ARC_")]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def _move_right(g: np.ndarray) -> np.ndarray:
    out = np.array(g)
    y, x = (int(v) for v in np.argwhere(out[:5] == 2)[0])
    out[y, x] = 0
    out[y, min(x + 1, 5)] = 2
    return out


def _trajectory(actions: list[int], *, hud: bool) -> list[Any]:
    """6x6 board. Action 1 moves the 2-pixel right; action 2 does nothing to the board.
    With `hud`, row 5 is a step counter that changes on every action."""
    g = np.zeros((6, 6), dtype=int)
    g[1, 0] = 2
    rows = []
    for step, a in enumerate(actions):
        nxt = _move_right(g) if a == 1 else g.copy()
        if hud:
            nxt[5, step % 6] = 3 if nxt[5, step % 6] == 0 else 0
        rows.append(e3.Transition(g.copy(), a, None, nxt, 0, 0))
        g = nxt
    return rows


def _spec(rows: list[Any], n_prefix: int, *, mask_rows=(), excluded=()) -> exp.WindowSpec:
    return exp.WindowSpec(
        game="syn",
        index=0,
        window_file=Path("syn.jsonl"),
        window_sha256="0" * 64,
        rows=rows,
        n_prefix=n_prefix,
        heldout_indices=tuple(range(n_prefix, len(rows))),
        excluded_indices=tuple(excluded),
        mask_rows=tuple(mask_rows),
        cell=1,
        reported_digest={},
        report_split_text="",
    )


CORRECT_ENGINE = """
import numpy as np
def engine(grid, action, data=None):
    g = np.array(grid)
    if action == 1:
        ys, xs = np.nonzero(g[:5] == 2)
        y, x = int(ys[0]), int(xs[0])
        g[y, x] = 0
        g[y, min(x + 1, 5)] = 2
    return g
"""

# Correct, except it raises on the second move (player already at column 3).
RAISE_ON_SECOND_MOVE = """
import numpy as np
def engine(grid, action, data=None):
    g = np.array(grid)
    if action == 1:
        ys, xs = np.nonzero(g[:5] == 2)
        y, x = int(ys[0]), int(xs[0])
        if x >= 3:
            raise ValueError("cannot model this move")
        g[y, x] = 0
        g[y, x + 1] = 2
    return g
"""

# Cheats with module state: it only answers correctly after an earlier call.
STATEFUL_ENGINE = """
import numpy as np
_seen = []
def engine(grid, action, data=None):
    _seen.append(1)
    g = np.array(grid)
    if action == 1 and len(_seen) >= 2:
        ys, xs = np.nonzero(g[:5] == 2)
        y, x = int(ys[0]), int(xs[0])
        g[y, x] = 0
        g[y, x + 1] = 2
    return g
"""

# The same cheat, with its counter on builtins, which a fresh module does not reset.
BUILTINS_STATEFUL_ENGINE = """
import builtins
import numpy as np
def engine(grid, action, data=None):
    builtins._exp10010_calls = getattr(builtins, "_exp10010_calls", 0) + 1
    g = np.array(grid)
    if action == 1 and builtins._exp10010_calls >= 2:
        ys, xs = np.nonzero(g[:5] == 2)
        y, x = int(ys[0]), int(xs[0])
        g[y, x] = 0
        g[y, x + 1] = 2
    return g
"""

ACTIONS = [1, 1, 2, 1, 2, 1]  # held-out rows 3 (move), 4 (no-op), 5 (move) at n_prefix=3


# ---------------------------------------------------------------------------------------
# Parsing the positive-control reports
# ---------------------------------------------------------------------------------------


def test_parse_split_reads_report_prose() -> None:
    """REQ-ARC-WMTE-10010: the split comes from the report prose, in both phrasings."""
    assert exp.parse_split("Visible rows are 0-16 (17 rows) and held-out rows are 17-24.") == (
        (0, 16),
        (17, 24),
    )
    # ar25's report ends the range with a full stop.
    text = "The split is visible rows 0-16 and held-out rows 17-24.\n\nHow the live gate"
    assert exp.parse_split(text) == ((0, 16), (17, 24))
    with pytest.raises(exp.WindowError):
        exp.parse_split("no ranges here, only commit f9383abaa0 and 25 rows")


def test_parse_window_file_rebases_and_reads_digest(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010: absolute main-checkout paths rebase onto this repository."""
    text = (
        "/home/x/carnot/results/raw/exp/episodes/g/t.jsonl (25 rows, sha256 "
        + "a" * 64
        + "). notes"
    )
    path, digest = exp.parse_window_file(text, tmp_path)
    assert path == tmp_path / "results/raw/exp/episodes/g/t.jsonl"
    assert digest == {"sha256": "a" * 64}
    with pytest.raises(exp.WindowError):
        exp.parse_window_file("not a path", tmp_path)


def test_real_windows_load_with_live_split() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-FIDELITY: all ten windows load, 17 + 8."""
    reports = exp.load_control_reports(REAL_PATHS)
    for i, game in enumerate(exp.PILOT_WINDOWS):
        spec = exp.load_window(game, i, reports[game], REAL_PATHS)
        assert len(spec.rows) == 25
        assert spec.n_prefix == 17
        assert spec.heldout_indices == tuple(range(17, 25))
        assert len(spec.window_sha256) == 64
        assert spec.cell == 1


# ---------------------------------------------------------------------------------------
# Prompt fidelity and the held-out leak assertion
# ---------------------------------------------------------------------------------------


def _real_spec(game: str = "su15") -> exp.WindowSpec:
    reports = exp.load_control_reports(REAL_PATHS)
    return exp.load_window(game, exp.PILOT_WINDOWS.index(game), reports[game], REAL_PATHS)


def test_prompt_fidelity_matches_recorded_prompt() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-FIDELITY: byte equality after the strip."""
    spec = _real_spec("su15")
    result = exp.prompt_fidelity(spec, REAL_PATHS)
    assert result["match"] is True
    assert result["first_diff_index"] is None
    assert result["built_prompt_sha256"] == result["expected_think_prompt_sha256"]
    assert result["defect_gate_rows_are_visible_rows"] is True
    assert result["captured_codeonly_eligible"] is True
    assert result["captured_tries"] == 3
    assert result["recorded_prompts_identical_across_seeds"] is True
    # The think-ON prompt has neither the directive nor a pre-opened fence.
    prompt = result["_captured"]["prompt"]
    assert not prompt.startswith(e3._L2_CODEONLY_DIRECTIVE)
    assert not prompt.endswith("```python\n")


def test_prompt_fidelity_detects_a_changed_row() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-FIDELITY: a changed row stops the window."""
    spec = _real_spec("su15")
    row = spec.rows[0]
    changed = np.array(row.next_grid)
    changed[30, 30] = (int(changed[30, 30]) + 1) % 16
    spec.rows[0] = e3.Transition(row.grid, row.action, row.data, changed, 0, 0)
    result = exp.prompt_fidelity(spec, REAL_PATHS)
    assert result["match"] is False
    assert result["first_diff_index"] is not None
    assert "differs" in result["reason"]


def test_think_prompt_transform_refuses_unexpected_shape() -> None:
    """REQ-ARC-WMTE-10010: only the codeonly directive and one fence may be removed."""
    body, transform = exp.think_prompt_from_recorded(
        e3._L2_CODEONLY_DIRECTIVE + "BODY" + "\n```python\n"
    )
    assert body == "BODY"
    assert len(transform) == 2
    with pytest.raises(exp.PromptTransformError):
        exp.think_prompt_from_recorded("BODY\n```python\n")
    with pytest.raises(exp.PromptTransformError):
        exp.think_prompt_from_recorded(e3._L2_CODEONLY_DIRECTIVE + "BODY")


def test_capture_refuses_tool_loop_that_would_call_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-10010: the capture must never reach the network."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "1")
    spec = _real_spec("su15")
    with pytest.raises(exp.PromptTransformError):
        exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)


def test_heldout_leak_check_passes_live_prompt_and_fails_full_prompt() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-LEAK: held-out answers never in the prompt."""
    spec = _real_spec("su15")
    captured = exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    ok = exp.heldout_leak_check(captured["prompt"], spec.rows, spec.n_prefix)
    assert ok["passed"] is True
    assert ok["prompt_transition_lines"] > 0
    assert ok["heldout_line_leaks"] == []
    # A prompt built from ALL rows shows held-out transitions with their answers.
    leaky = e3.induce_prompt(spec.game, list(spec.rows), spec.cell, k=None)
    bad = exp.heldout_leak_check(leaky, spec.rows, spec.n_prefix)
    assert bad["passed"] is False
    assert bad["heldout_line_leaks"]


def test_heldout_leak_check_catches_a_full_grid_leak() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-LEAK: a held-out next grid shown in full."""
    rows = _trajectory(ACTIONS, hud=False)
    prompt = e3.induce_prompt("syn", rows[:3], 1, k=None)
    assert exp.heldout_leak_check(prompt, rows, 3)["passed"] is True
    # Row 5's next grid is a board no visible row shows.
    leaked = prompt + "\n" + e3._rle_grid(rows[5].next_grid)
    result = exp.heldout_leak_check(leaked, rows, 3)
    assert result["passed"] is False
    assert result["heldout_full_grid_leaks"] == [5]


# ---------------------------------------------------------------------------------------
# Scoring wrapper
# ---------------------------------------------------------------------------------------


def test_identity_scores_zero_and_correct_engine_scores_one() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: the two anchors of the metric."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    ident = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, spec, "identity")
    assert ident["primary_change_fidelity"] == 0.0
    assert ident["n_changing_rows"] == 2
    assert ident["noop_hallucination_rate"] == 0.0
    good = exp.score_engine(CORRECT_ENGINE, spec, "good")
    assert good["primary_change_fidelity"] == 1.0
    assert good["masked_exact_accuracy"] == 1.0
    assert good["live_unmasked_pass_1p0"] is True


def test_raised_changing_row_scores_zero_unlike_the_verifier() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: a raised row is 0, not dropped."""
    rows = _trajectory(ACTIONS, hud=False)
    spec = _spec(rows, 3)
    score = exp.score_engine(RAISE_ON_SECOND_MOVE, spec, "raise")
    assert score["primary_change_fidelity"] == 0.5
    assert score["n_raised_changing_rows"] == 1
    assert score["live_unmasked_pass_1p0"] is False
    # The live verifier drops the raised row and reports a perfect change fidelity.
    ns: dict[str, Any] = {}
    exec(compile(RAISE_ON_SECOND_MOVE, "r", "exec"), ns)
    vr = e3.WorldModelVerifier(rows[3:], hud_mask_enabled=False).score(ns["engine"])
    assert vr.change_fidelity == 1.0


def test_raised_noop_row_counts_as_hallucinated() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: the guard metric counts raises."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    always_raise = "def engine(grid, action, data=None):\n    raise RuntimeError('no')\n"
    score = exp.score_engine(always_raise, spec, "raise_all")
    assert score["primary_change_fidelity"] == 0.0
    assert score["noop_hallucination_rate"] == 1.0
    assert score["n_raised_rows"] == 3
    assert score["n_raised_noop_rows"] == 1


def test_masks_remove_the_step_counter_row() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: masked HUD rows cannot decide."""
    rows = _trajectory(ACTIONS, hud=True)
    masked = exp.score_engine(CORRECT_ENGINE, _spec(rows, 3, mask_rows=(5,)), "good")
    assert masked["primary_change_fidelity"] == 1.0
    assert masked["noop_hallucination_rate"] == 0.0
    # The live gate is unmasked, so the counter still defeats a correct engine there.
    assert masked["live_unmasked_pass_1p0"] is False
    unmasked = exp.score_engine(CORRECT_ENGINE, _spec(rows, 3), "good")
    assert unmasked["primary_change_fidelity"] < 1.0
    ident = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, _spec(rows, 3, mask_rows=(5,)), "id")
    assert ident["primary_change_fidelity"] == 0.0


def test_preregistered_exclusion_leaves_masked_metrics_only() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: excluded rows stay in the live gate."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3, excluded=(5,))
    score = exp.score_engine(RAISE_ON_SECOND_MOVE, spec, "raise")
    assert score["primary_change_fidelity"] == 1.0
    assert score["n_scored_rows"] == 2
    assert score["live_unmasked_exact_accuracy"] == pytest.approx(2 / 3)
    statuses = {r["row"]: r.get("status") for r in score["per_row"]}
    assert statuses[5] == "excluded_preregistered"


def test_levelup_row_is_excluded_like_the_live_verifier() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: a level-up row is not graded."""
    rows = _trajectory(ACTIONS, hud=False)
    last = rows[5]
    rows[5] = e3.Transition(last.grid, last.action, None, last.next_grid, 0, 1)
    score = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, _spec(rows, 3), "id")
    assert score["n_levelup_rows_excluded"] == 1
    assert score["n_changing_rows"] == 1


def test_syntax_error_and_missing_engine_score_zero() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: unusable output scores 0."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    bad = exp.score_engine("def engine(:\n", spec, "bad")
    assert bad["engine_status"].startswith("syntax_error")
    assert bad["primary_change_fidelity"] == 0.0
    none = exp.score_engine(None, spec, "none")
    assert none["engine_status"] == "no_engine_source"
    assert none["primary_change_fidelity"] == 0.0
    wrong_shape = (
        "import numpy as np\ndef engine(grid, action, data=None):\n    return np.zeros((2, 2))\n"
    )
    assert exp.score_engine(wrong_shape, spec, "shape")["primary_change_fidelity"] == 0.0


# ---------------------------------------------------------------------------------------
# Purity: a new process per row, holding only that row
# ---------------------------------------------------------------------------------------


def _shared_process_predictor(source: str, rows: Any, label: str) -> Any:
    """The NON-isolated path: every row in THIS interpreter, all rows in reach.

    Test only, to show teeth. It seeds each row like the child (SEED_BASE plus the
    row's position) and restores the global random state afterwards.
    """
    import random

    saved = (random.getstate(), np.random.get_state())
    shared: dict[str, Any] = {}
    preds = []
    try:
        exec(compile(source, label, "exec"), shared)
        for i, row in enumerate(rows):
            random.seed(exp.SEED_BASE + i)
            np.random.seed((exp.SEED_BASE + i) % (2**32))
            try:
                grid = np.asarray(row["grid"], dtype=np.int64)
                out = shared["engine"](grid.copy(), int(row["action"]), row.get("data"))
                preds.append(exp.RowPrediction(child.check_output(out, grid.size), ""))
            except Exception as exc:
                preds.append(exp.RowPrediction(None, f"{type(exc).__name__}: {exc}"[:200]))
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
    return preds, {"isolation": "shared (test only)"}


@pytest.mark.parametrize("source", [STATEFUL_ENGINE, BUILTINS_STATEFUL_ENGINE])
def test_stateful_engine_gains_nothing_from_call_order(source: str) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-SCORING: no state carries over, of any kind."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    fresh = exp.score_engine(source, spec, "stateful")
    ident = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, spec, "id")
    assert fresh["primary_change_fidelity"] == ident["primary_change_fidelity"] == 0.0
    assert fresh["engine_run"]["isolation"] == exp.ROW_ISOLATION
    # Order does not matter either.
    reversed_spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    reversed_spec.heldout_indices = tuple(reversed(reversed_spec.heldout_indices))
    again = exp.score_engine(source, reversed_spec, "stateful")
    assert again["primary_change_fidelity"] == fresh["primary_change_fidelity"]
    # Teeth: in ONE shared interpreter the cheat does score, module state or builtins.
    shared = exp.score_engine(source, spec, "stateful", predictor=_shared_process_predictor)
    assert shared["primary_change_fidelity"] == 0.5


# ---------------------------------------------------------------------------------------
# Controls on the real recorded evidence (read-only)
# ---------------------------------------------------------------------------------------


def test_real_controls_reproduce_the_preregistered_values() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: identity 0, expert 1, 0.13."""
    reports = exp.load_control_reports(REAL_PATHS)
    specs = [exp.load_window(g, i, reports[g], REAL_PATHS) for i, g in enumerate(exp.PILOT_WINDOWS)]
    controls = exp.run_controls(specs, REAL_PATHS)
    assert controls["passed"] is True
    assert controls["identity_all_zero"] is True
    assert controls["expert_all_one"] is True
    reproduced = controls["codeonly_baseline_mean_reproduced"]
    assert abs(reproduced - exp.PREREGISTERED_BASELINE_MEAN) <= exp.BASELINE_TOLERANCE
    # Also equal to the synthesis's own number, to rounding.
    assert reproduced == pytest.approx(controls["codeonly_baseline_mean_file"], abs=1e-9)
    assert controls["max_abs_diff_vs_file_per_window"] < 1e-3


# ---------------------------------------------------------------------------------------
# Fixture evidence tree, fake GPU, fake chat server
# ---------------------------------------------------------------------------------------


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_fixture(tmp_path: Path, games: tuple[str, ...]) -> tuple[Path, Path, Path]:
    """Copy the evidence one run needs into tmp_path, plus a fake GGUF and binary."""
    root = tmp_path / "repo"
    paths = exp.EvidencePaths.under(root)
    _copy(REAL_PATHS.workflow_result(), paths.workflow_result())
    _copy(REAL_PATHS.pilot_baseline(), paths.pilot_baseline())
    reports = exp.load_control_reports(REAL_PATHS)
    for g in games:
        _copy(REAL_PATHS.expert_engine(g), paths.expert_engine(g))
        for s in exp.RECORDED_SEEDS:
            _copy(REAL_PATHS.request(g, s), paths.request(g, s))
            _copy(REAL_PATHS.response(g, s), paths.response(g, s))
        real_file, _ = exp.parse_window_file(reports[g]["window_file"], REPO)
        fixture_file, _ = exp.parse_window_file(reports[g]["window_file"], root)
        _copy(real_file, fixture_file)
    hf = tmp_path / "hf"
    gguf = hf / "models--unsloth--Qwen3.8-27B-GGUF" / "snapshots" / "rev" / exp.MODEL_FILENAME
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o755)
    return root, hf, binary


def _cfg(
    tmp_path: Path, fixture: tuple[Path, Path, Path], games: tuple[str, ...], *, dry_run: bool
) -> exp.RunConfig:
    root, hf, binary = fixture
    out = tmp_path / "out"
    return exp.RunConfig(
        repo_root=root,
        output_dir=out,
        artifact_path=out / "artifact.json",
        dry_run=dry_run,
        port=18996,
        llama_server=binary,
        hf_cache=hf,
        heartbeat_s=0.05,
        windows=games,
    )


SERVER_PID = 424242


class FakeGpu:
    """Answers the two nvidia-smi queries. GPU 0 always carries the conductor."""

    def __init__(self, *, gpu1: bool = True, used: int = 3, other_pid: int | None = None):
        self.gpu1 = gpu1
        self.used = used
        self.other_pid = other_pid
        self.server_up = False
        self.drop_gpu1_after_launch = False

    def __call__(self, args: Any) -> str:
        gpu1_visible = self.gpu1 and not (self.server_up and self.drop_gpu1_after_launch)
        if str(args[0]).startswith("--query-gpu"):
            lines = [f"0, {exp.GPU0_UUID}, 9000"]
            if gpu1_visible:
                lines.append(f"1, {exp.GPU1_UUID}, {self.used}")
            return "\n".join(lines) + "\n"
        apps = [f"777, {exp.GPU0_UUID}, 8000"]
        if self.other_pid:
            apps.append(f"{self.other_pid}, {exp.GPU1_UUID}, 900")
        if self.server_up and gpu1_visible:
            apps.append(f"{SERVER_PID}, {exp.GPU1_UUID}, 21200")
        return "\n".join(apps) + "\n"


def _forbidden_proposer(**_: Any) -> Any:
    raise AssertionError("no proposer may be built on this path")


def _hooks(
    gpu: FakeGpu, events: list[Any], *, make_proposer: Any = None, port_busy: bool = False
) -> exp.RunHooks:
    def launch(gguf: Path, port: int, llama_server: Path, log_dir: Path, smi: Any) -> Any:
        events.append(("launch", port, Path(gguf).name))
        gpu.server_up = True
        return exp.OwnedServer(proc=None, pid=SERVER_PID, port=port, argv=["fake"], meta={})

    def terminate(server: Any) -> dict[str, Any]:
        events.append(("terminate", server.pid))
        return {"terminated": True, "pid": server.pid}

    return exp.RunHooks(
        smi=gpu,
        port_in_use=lambda port: port_busy,
        launch_server=launch,
        terminate_server=terminate,
        make_proposer=make_proposer or _forbidden_proposer,
        props_model_path=lambda port: f"/models/{exp.MODEL_FILENAME}",
        completion_alive=lambda port: {"alive": True},
        slots_probe=lambda port: 123,
        process_comm=lambda: "python",
    )


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, *_: Any) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False


class FakeChatServer:
    """Stands in for llama-server's /v1/chat/completions. No socket is opened."""

    def __init__(self, answers: dict[str, dict[str, Any]], delay_s: float = 0.0) -> None:
        self.answers = answers
        self.delay_s = delay_s
        self.requests: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: Any = None) -> _FakeResponse:
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        if not url.endswith("/v1/chat/completions"):
            raise urllib.error.URLError(f"fake server has no route {url}")
        payload = json.loads(req.data)
        self.requests.append({"url": url, "payload": payload, "timeout": timeout})
        prompt = payload["messages"][0]["content"]
        game = next(g for g in self.answers if f"ARC-AGI-3 game '{g}'" in prompt)
        answer = self.answers[game]
        if self.delay_s:
            time.sleep(self.delay_s)
        if "raise" in answer:
            raise answer["raise"]
        body = {
            "choices": [
                {
                    "message": {
                        "content": answer.get("content", ""),
                        "reasoning_content": answer.get("reasoning", "thinking it through"),
                    },
                    "finish_reason": answer.get("finish", "stop"),
                }
            ],
            "usage": {"completion_tokens": answer.get("tokens", 1500), "prompt_tokens": 4000},
        }
        return _FakeResponse(json.dumps(body).encode())


def _test_proposer(*, port: int, model_path: str) -> Any:
    """The live proposer with its server checks answered locally."""
    prop = exp._live_proposer(port=port, model_path=model_path)
    prop._healthy = MethodType(lambda self: True, prop)
    prop._reusable = MethodType(lambda self: True, prop)
    prop.observed_n_ctx = MethodType(lambda self: exp.N_CTX, prop)
    return prop


def _fenced(code: str) -> str:
    return "```python\n" + code + "\n```"


# ---------------------------------------------------------------------------------------
# Generation through the live generate() with a mocked server
# ---------------------------------------------------------------------------------------


def test_generate_window_records_calls_seeds_and_heartbeats(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-10010: live generate() in think mode, per-call records, progress lines."""
    import os

    spec = _real_spec("su15")
    captured = exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    expert = REAL_PATHS.expert_engine("su15").read_text()
    server = FakeChatServer({"su15": {"content": _fenced(expert), "tokens": 2100}}, delay_s=0.3)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    prop = _test_proposer(port=exp.DEFAULT_PORT, model_path=f"/m/{exp.MODEL_FILENAME}")
    exp.pin_to_owned_server(prop, exp.DEFAULT_PORT)
    recorder = exp.CallRecorder(heartbeat_s=0.05, raw_dir=tmp_path / "calls", probe=lambda: 77)
    exp.install_call_recorder(prop, recorder)
    out = exp.generate_window(prop, captured, spec, recorder)
    assert out["ok"] is True
    assert "def engine" in out["engine_source"]
    assert out["n_calls"] == 1
    call = out["calls"][0]
    base = exp.window_seed(spec)
    assert call["seed"] == base * 1000
    assert call["completion_tokens"] == 2100
    assert call["stop_type"] == "eos"
    assert call["censored"] is False
    raw = Path(call["raw_path"])
    assert raw.exists()
    assert raw.parent == tmp_path / "calls" / "su15" / recorder.run_id
    # Think mode: the chat endpoint, the exact captured prompt, the pinned seed.
    req = server.requests[0]
    assert req["payload"]["messages"][0]["content"] == captured["prompt"]
    assert req["payload"]["seed"] == base * 1000
    assert req["timeout"] == exp.CALL_TIMEOUT_S
    clamped = exp.N_CTX - e3._INDUCE_WORST_CASE_PROMPT_TOKENS
    assert req["payload"]["max_tokens"] == clamped == out["requested_n_predict"]
    assert "CARNOT_ARC_GENERATOR_SEED" not in os.environ
    printed = capsys.readouterr().out
    assert "model call BEFORE" in printed
    assert "model call AFTER" in printed
    assert "heartbeat" in printed and "decoded=77" in printed
    score = exp.score_engine(out["engine_source"], spec, "think_on")
    assert score["primary_change_fidelity"] == 1.0


def test_generate_window_records_token_limit_censoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010: a call cut at the token cap is recorded as censored, per try."""
    spec = _real_spec("su15")
    captured = exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    server = FakeChatServer({"su15": {"content": "", "finish": "length", "tokens": 108720}})
    monkeypatch.setattr(urllib.request, "urlopen", server)
    prop = _test_proposer(port=exp.DEFAULT_PORT, model_path=f"/m/{exp.MODEL_FILENAME}")
    exp.pin_to_owned_server(prop, exp.DEFAULT_PORT)
    recorder = exp.CallRecorder(heartbeat_s=10.0)
    exp.install_call_recorder(prop, recorder)
    out = exp.generate_window(prop, captured, spec, recorder)
    assert out["ok"] is False
    assert out["n_calls"] == exp.LIVE_TRIES
    assert out["censored"] is True
    assert all(c["censor_reason"] == "token_limit" for c in out["calls"])
    base = exp.window_seed(spec)
    assert out["effective_seeds"] == [base * 1000 + a for a in range(exp.LIVE_TRIES)]


def test_generate_window_records_wall_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010: a timed-out call is censored and ends the window's generation."""
    spec = _real_spec("su15")
    captured = exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    server = FakeChatServer({"su15": {"raise": TimeoutError("timed out")}})
    monkeypatch.setattr(urllib.request, "urlopen", server)
    prop = _test_proposer(port=exp.DEFAULT_PORT, model_path=f"/m/{exp.MODEL_FILENAME}")
    exp.pin_to_owned_server(prop, exp.DEFAULT_PORT)
    recorder = exp.CallRecorder(heartbeat_s=10.0)
    exp.install_call_recorder(prop, recorder)
    out = exp.generate_window(prop, captured, spec, recorder)
    assert out["ok"] is False
    assert out["n_calls"] == 1
    assert out["calls"][0]["censor_reason"] == "wall_timeout"


def test_pinned_proposer_never_launches_a_server() -> None:
    """REQ-ARC-WMTE-10010: a dead owned server fails the call; nothing relaunches."""
    prop = exp._live_proposer(port=18997, model_path=f"/m/{exp.MODEL_FILENAME}")
    prop._healthy = MethodType(lambda self: False, prop)
    exp.pin_to_owned_server(prop, 18997)
    assert prop._ensure_server() is False
    assert prop._proc is None
    ok, msg = prop.generate("prompt", exp.REQUIRED, tries=1, codeonly_eligible=True)
    assert ok is False and "failed" in msg
    assert prop._proc is None


# ---------------------------------------------------------------------------------------
# Whole runs on a fixture evidence tree
# ---------------------------------------------------------------------------------------


def test_real_mode_run_with_mocked_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010: preconditions, fidelity, controls, generation, scoring, teardown."""
    games = ("su15", "ft09")
    fixture = _build_fixture(tmp_path, games)
    expert = REAL_PATHS.expert_engine("su15").read_text()
    server = FakeChatServer(
        {
            "su15": {"content": _fenced(expert)},
            "ft09": {"content": "", "finish": "length", "tokens": 108720},
        }
    )
    monkeypatch.setattr(urllib.request, "urlopen", server)
    gpu, events = FakeGpu(), []
    cfg = _cfg(tmp_path, fixture, games, dry_run=False)
    art = exp.run_pilot(cfg, _hooks(gpu, events, make_proposer=_test_proposer))
    assert art["honest_verdict"].startswith("complete_think_on_pilot_2_windows")
    assert art["inference_substrate"] == "live_llm_inference"
    assert art["inference_substrate_class"] == "model_full_generation"
    assert art["model_specs"][0]["invoked"] is True
    rows = {r["game"]: r for r in art["per_window_rows"]}
    assert rows["su15"]["status"] == "generated"
    assert rows["su15"]["score"]["primary_change_fidelity"] == 1.0
    assert rows["ft09"]["status"] == "generation_failed"
    assert rows["ft09"]["score"]["primary_change_fidelity"] == 0.0
    assert art["false_negative_risk"]["windows_with_censored_generation"] == ["ft09"]
    assert events[0] == ("launch", cfg.port, exp.MODEL_FILENAME)
    assert events[-1] == ("terminate", SERVER_PID)
    assert art["server_teardown"]["terminated"] is True
    assert art["window_seed_bases"] == {
        "su15": exp.SEED_BASE * 100,
        "ft09": exp.SEED_BASE * 100 + 1,
    }
    assert json.loads(cfg.artifact_path.read_text())["honest_verdict"] == art["honest_verdict"]


def test_wedge_before_a_window_gives_partial_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010: a lost GPU is a recorded wedge, and the server is still torn down."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    monkeypatch.setattr(urllib.request, "urlopen", FakeChatServer({}))
    gpu, events = FakeGpu(), []
    gpu.drop_gpu1_after_launch = True
    art = exp.run_pilot(
        _cfg(tmp_path, fixture, games, dry_run=False),
        _hooks(gpu, events, make_proposer=_test_proposer),
    )
    assert art["honest_verdict"] == "partial_think_on_pilot_0_of_1_windows_gpu1_lost"
    # The server loaded the model and answered a probe, but no pilot generation ran.
    assert art["inference_substrate_class"] == "model_load_no_generation"
    assert art["inference_substrate"] == "live_llm_server_loaded_liveness_probe_only"
    assert art["model_specs"][0]["invoked"] is True
    assert art["substrate_evidence"]["n_model_calls_in_rows"] == 0
    assert events[-1] == ("terminate", SERVER_PID)


def test_dry_run_artifact_fields(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010: the artifact carries every required field; no server, no GPU."""
    games = ("su15", "ft09")
    cfg = _cfg(tmp_path, _build_fixture(tmp_path, games), games, dry_run=True)
    gpu, events = FakeGpu(), []
    art = exp.run_pilot(cfg, _hooks(gpu, events))
    assert art["honest_verdict"].startswith("complete_dry_run_harness_verified_2_windows")
    assert events == []
    for key in (
        "inference_substrate",
        "model_specs",
        "random_seed",
        "reproducibility_checksum",
        "preconditions_checked",
        "duration_s",
        "gate_env_flags",
        "per_window_rows",
        "controls",
        "comparison_to_baseline",
        "false_negative_risk",
        "sample_size_caveat",
        "deviations_from_live_path",
    ):
        assert art[key] is not None, key
    assert art["solve_provenance"] == "development_proxy"
    assert art["verifier_is_oracle"] is False
    assert art["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert art["inference_substrate_class"] == "no_model_load"
    assert len(art["reproducibility_checksum"]) == 64
    flags = art["gate_env_flags"]
    for key in (
        "CARNOT_ARC_TRUST_METRIC",
        "CARNOT_ARC_WM_HUD_MASK",
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT",
    ):
        assert "effective" in flags[key]
    assert flags["think_mode"]["effective_think_on"] is True
    gpu_checks = [c for c in art["preconditions_checked"] if c["resource"].startswith("gpu1")]
    assert gpu_checks and all(c["available"] is None for c in gpu_checks)
    assert all(r["status"] == "dry_run_stand_in" for r in art["per_window_rows"])
    assert art["false_negative_risk"]["windows_with_few_changing_rows"] == {"ft09": 4}
    assert art["comparison_to_baseline"]["n_windows_scored"] == 2


def test_gate_flags_record_a_non_default_trust_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010: the artifact records the gate flags actually in effect."""
    monkeypatch.setenv("CARNOT_ARC_TRUST_METRIC", "cell_recall")
    flags = exp.gate_env_flags()
    assert flags["CARNOT_ARC_TRUST_METRIC"] == {"raw": "cell_recall", "effective": "cell_recall"}
    # It changes no induce setting, so the llama.cpp-shaped comparison still matches...
    parity = exp.induction_env_parity()
    assert parity["matches_kernel_llamacpp_shape"] is True
    assert parity["backend_parity"] is False
    # ...but the kernel never sets it, so a real run refuses it (fail closed).
    unlisted = exp.induction_env_unlisted_flags()
    assert unlisted["ok"] is False
    assert "CARNOT_ARC_TRUST_METRIC" in unlisted["unlisted"]


def test_resume_skips_finished_windows(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-RESUME: one shard row per window."""
    games = ("su15", "ft09")
    cfg = _cfg(tmp_path, _build_fixture(tmp_path, games), games, dry_run=True)
    hooks = _hooks(FakeGpu(), [])
    first = exp.run_pilot(cfg, hooks)
    shard = cfg.output_dir / "shard_dry_run.jsonl"
    lines = shard.read_text().splitlines()
    assert len(lines) == 2 and first["resumed_windows"] == []
    shard.write_text(lines[0] + "\n")  # as if the run died after the first window
    second = exp.run_pilot(cfg, hooks)
    assert second["resumed_windows"] == ["su15"]
    assert len(shard.read_text().splitlines()) == 2
    assert [r["game"] for r in second["per_window_rows"]] == ["su15", "ft09"]
    third = exp.run_pilot(cfg, hooks)
    assert third["resumed_windows"] == ["ft09", "su15"]
    assert len(shard.read_text().splitlines()) == 2


def test_failed_control_blocks_before_any_server(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: a broken metric stops the run."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    root = fixture[0]
    exp.EvidencePaths.under(root).expert_engine("su15").write_text(exp.IDENTITY_ENGINE_SOURCE)
    gpu, events = FakeGpu(), []
    art = exp.run_pilot(_cfg(tmp_path, fixture, games, dry_run=False), _hooks(gpu, events))
    assert art["honest_verdict"] == "blocked_control_failed_expert_su15"
    assert events == []
    assert art["per_window_rows"] == []
    assert art["controls"]["expert_all_one"] is False


def test_baseline_control_fails_closed_on_a_changed_response(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: the codeonly baseline must reproduce."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    paths = exp.EvidencePaths.under(fixture[0])
    # su15's first shots average 0.58; replace all three with the identity engine.
    for seed in exp.RECORDED_SEEDS:
        paths.response("su15", seed).write_text(json.dumps({"content": exp.IDENTITY_ENGINE_SOURCE}))
    art = exp.run_pilot(_cfg(tmp_path, fixture, games, dry_run=True), _hooks(FakeGpu(), []))
    assert art["honest_verdict"] == "blocked_control_failed_codeonly_window_su15"
    assert "codeonly_baseline" in art["controls"]["failures"]


@pytest.mark.parametrize(
    ("setup", "verdict"),
    [
        ("no_gguf", "blocked_gguf_not_cached_qwen38_27b"),
        ("no_binary", "blocked_llama_server_missing"),
        ("gpu1_absent", "blocked_gpu1_absent"),
        ("gpu1_busy", "blocked_gpu1_memory_in_use"),
        ("gpu1_other_process", "blocked_gpu1_other_process"),
        ("port_busy", "blocked_server_port_in_use"),
        ("evidence_missing", "blocked_positive_control_evidence_missing"),
        ("env_not_live", "blocked_induction_env_not_kernel_llamacpp_shape"),
        ("janitor_comm", "blocked_process_name_reaped_by_janitor"),
    ],
)
def test_preconditions_block_before_any_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: str, verdict: str
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-PRECONDITIONS: a miss writes blocked_*."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    root, hf, binary = fixture
    gpu = FakeGpu()
    port_busy = False
    if setup == "no_gguf":
        shutil.rmtree(hf)
    elif setup == "no_binary":
        binary.unlink()
    elif setup == "gpu1_absent":
        gpu.gpu1 = False
    elif setup == "gpu1_busy":
        gpu.used = 2000
    elif setup == "gpu1_other_process":
        gpu.other_pid = 5150
    elif setup == "port_busy":
        port_busy = True
    elif setup == "evidence_missing":
        exp.EvidencePaths.under(root).response("su15", "7491002").unlink()
    elif setup == "env_not_live":
        monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    events: list[Any] = []
    cfg = _cfg(tmp_path, fixture, games, dry_run=False)
    hooks = _hooks(gpu, events, port_busy=port_busy)
    if setup == "janitor_comm":
        hooks.process_comm = lambda: "python3"
    art = exp.run_pilot(cfg, hooks)
    assert art["honest_verdict"] == verdict
    assert events == []
    assert art["per_window_rows"] == []
    failed = [c for c in art["preconditions_checked"] if c["available"] is False]
    assert failed and failed[0]["blocked_verdict"] == verdict
    assert art["inference_substrate_class"] == "blocked_no_run"
    assert cfg.artifact_path.exists()


def test_cli_dry_run_requires_scratch_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-10010: a dry run never writes into the research record by default."""
    seen: list[exp.RunConfig] = []

    def fake_run(cfg: exp.RunConfig) -> dict[str, Any]:
        seen.append(cfg)
        return {"honest_verdict": "complete_dry_run_harness_verified_0_windows_stand_in_mean_none"}

    monkeypatch.setattr(exp, "run_pilot", fake_run)
    with pytest.raises(SystemExit):
        exp.main(["--dry-run"])
    assert seen == []
    assert exp.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    assert seen[-1].artifact_path == tmp_path / "experiment_10010_dry_run.json"
    assert seen[-1].dry_run is True
    assert exp.main([]) == 0
    assert seen[-1].artifact_path == REPO / exp.DEFAULT_ARTIFACT_REL
    assert seen[-1].windows == exp.PILOT_WINDOWS


def test_identity_control_failure_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: an identity that scores is a bug."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    # If the "identity" control actually modelled the game, the metric would be broken.
    monkeypatch.setattr(exp, "IDENTITY_ENGINE_SOURCE", REAL_PATHS.expert_engine("su15").read_text())
    art = exp.run_pilot(_cfg(tmp_path, fixture, games, dry_run=True), _hooks(FakeGpu(), []))
    assert art["honest_verdict"] == "blocked_control_failed_identity_su15"
    assert art["controls"]["identity_all_zero"] is False


# ---------------------------------------------------------------------------------------
# Server launch gates, with a fake process. Nothing is started.
# ---------------------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, argv: list[str], env: dict[str, str], *, exited: bool, hang: bool):
        self.argv = argv
        self.env = env
        self.pid = SERVER_PID
        self.returncode = 1 if exited else None
        self.hang = hang
        self.actions: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.actions.append("terminate")

    def kill(self) -> None:
        self.actions.append("kill")
        self.hang = False

    def wait(self, timeout: float | None = None) -> int:
        if self.hang:
            import subprocess

            raise subprocess.TimeoutExpired("llama-server", timeout or 0)
        self.actions.append("waited")
        return 0


def _fake_subprocess(procs: list[_FakeProc], *, exited: bool = False) -> Any:
    import subprocess
    from types import SimpleNamespace

    def popen(
        argv: list[str],
        stdout: Any = None,
        stderr: Any = None,
        env: Any = None,
        preexec_fn: Any = None,
    ) -> Any:
        proc = _FakeProc(list(argv), dict(env or {}), exited=exited, hang=False)
        proc.preexec_fn = preexec_fn
        procs.append(proc)
        return proc

    return SimpleNamespace(
        Popen=popen,
        STDOUT=subprocess.STDOUT,
        run=subprocess.run,
        TimeoutExpired=subprocess.TimeoutExpired,
    )


def _smi_with(gpu1_mib: int, gpu0_mib: int = 0) -> Any:
    def smi(args: Any) -> str:
        lines = [f"777, {exp.GPU0_UUID}, 8000"]
        if gpu1_mib:
            lines.append(f"{SERVER_PID}, {exp.GPU1_UUID}, {gpu1_mib}")
        if gpu0_mib:
            lines.append(f"{SERVER_PID}, {exp.GPU0_UUID}, {gpu0_mib}")
        return "\n".join(lines) + "\n"

    return smi


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("ok", None),
        ("low_residency", "no real GPU 1 offload"),
        ("on_gpu0", "on GPU 0"),
        ("wrong_model", "/props names"),
        ("completion_dead", "/completion dead"),
        ("exited_early", "exited early"),
    ],
)
def test_launch_owned_server_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, error: str | None
) -> None:
    """REQ-ARC-WMTE-10010: GPU 1 by UUID, measured flags, residency, identity, liveness."""
    procs: list[_FakeProc] = []
    monkeypatch.setattr(exp, "subprocess", _fake_subprocess(procs, exited=case == "exited_early"))
    monkeypatch.setattr(exp, "health_ok", lambda port, timeout=3.0: True)
    model = "Other-Model.gguf" if case == "wrong_model" else exp.MODEL_FILENAME
    monkeypatch.setattr(exp, "props_model_path", lambda port, timeout=20.0: f"/m/{model}")
    alive = case != "completion_dead"
    monkeypatch.setattr(exp, "completion_alive", lambda port, timeout=120.0: {"alive": alive})
    smi = _smi_with(
        gpu1_mib=900 if case == "low_residency" else 21200,
        gpu0_mib=4000 if case == "on_gpu0" else 0,
    )
    gguf = tmp_path / exp.MODEL_FILENAME
    call = lambda: exp.launch_owned_server(  # noqa: E731
        gguf, 18998, Path("/bin/llama-server"), tmp_path / "logs", smi=smi, health_wait_s=5
    )
    if error is None:
        server = call()
        assert server.meta["residency_mib_gpu1"] == 21200
        assert server.meta["residency_mib_gpu0"] is None
        proc = procs[0]
        assert proc.env["CUDA_VISIBLE_DEVICES"] == exp.GPU1_UUID
        for flag in ("-c", "131072", "--parallel", "--ctx-checkpoints", "-fit", "q8_0"):
            assert flag in proc.argv
        assert proc.actions == []
        # The server dies with this process, and a PID file names it for a person.
        assert proc.preexec_fn is exp._set_parent_death_signal
        assert Path(server.meta["pid_file"]).read_text().strip() == str(SERVER_PID)
    else:
        with pytest.raises(exp.ServerError, match=error):
            call()
        if case != "exited_early":
            assert procs[0].actions[0] == "terminate"


def test_terminate_owned_server_falls_back_to_kill() -> None:
    """REQ-ARC-WMTE-10010: teardown is by our PID; a hung terminate escalates to kill."""
    proc = _FakeProc(["x"], {}, exited=False, hang=True)
    server = exp.OwnedServer(proc=proc, pid=proc.pid, port=1, argv=["x"], meta={})
    result = exp.terminate_owned_server(server)
    assert result["method"] == "kill"
    assert proc.actions == ["terminate", "kill", "waited"]
    assert exp.terminate_owned_server(None)["terminated"] is False


def test_artifacts_pass_the_project_artifact_linter(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010: dry-run and blocked artifacts draw no critical adversarial flag."""
    import scripts.adversarial_verify as av

    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    dry_cfg = _cfg(tmp_path, fixture, games, dry_run=True)
    exp.run_pilot(dry_cfg, _hooks(FakeGpu(), []))
    blocked_cfg = _cfg(tmp_path / "b", fixture, games, dry_run=False)
    art = exp.run_pilot(blocked_cfg, _hooks(FakeGpu(gpu1=False), []))
    assert art["honest_verdict"] == "blocked_gpu1_absent"
    for path in (dry_cfg.artifact_path, blocked_cfg.artifact_path):
        report = av.verify_artifact(path)
        critical = [f["kind"] for f in report["flags"] if f.get("severity") == "critical"]
        assert critical == [], (path.name, critical)


def test_model_artifacts_declare_a_consistent_substrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010: the class never contradicts the invocation evidence.

    Mocked runs finish in seconds, so the duration floor for a model class fires; that is
    the linter working. Every OTHER critical flag would be a harness declaration bug.
    """
    import scripts.adversarial_verify as av

    expert = REAL_PATHS.expert_engine("su15").read_text()
    answers = {"su15": {"content": _fenced(expert)}, "ft09": {"content": _fenced(expert)}}
    cfg, _ = _two_window_real_run(tmp_path, answers, monkeypatch)
    exp.run_pilot(cfg, _hooks(FakeGpu(), [], make_proposer=_test_proposer))
    gpu = FakeGpu()
    gpu.drop_gpu1_after_launch = True
    wedge_cfg = _cfg(
        tmp_path / "w", _build_fixture(tmp_path / "w", ("su15",)), ("su15",), dry_run=False
    )
    exp.run_pilot(wedge_cfg, _hooks(gpu, [], make_proposer=_test_proposer))
    for path in (cfg.artifact_path, wedge_cfg.artifact_path):
        report = av.verify_artifact(path)
        critical = [f for f in report["flags"] if f.get("severity") == "critical"]
        # A class mismatch is allowed only when its cause is the duration floor.
        other = [
            f["kind"]
            for f in critical
            if f["kind"] != "DURATION_TOO_SHORT"
            and not (f["kind"] == "SUBSTRATE_CLASS_MISMATCH" and "floor" in f.get("detail", ""))
        ]
        assert other == [], (path.name, other)
        # The floor fires when the run was faster than its class floor. Scoring each row
        # in its own process made the one-window wedge run take ~3 s, above the 2 s floor
        # for a loaded-but-idle model, so that floor is only required when it applies.
        art = json.loads(path.read_text())
        floor = av.SUBSTRATE_CLASS_FLOORS[art["inference_substrate_class"]]
        if floor is not None and float(art["duration_s"]) < floor:
            assert critical, ("the fast mocked run should trip the duration floor", path)
    full = json.loads(cfg.artifact_path.read_text())
    assert full["inference_substrate_class"] == "model_full_generation"
    assert float(full["duration_s"]) < 60.0  # so the check above did fire for this path


# ---------------------------------------------------------------------------------------
# Review fixes, 2026-09-23: engines that should score 0 must score 0.
# ---------------------------------------------------------------------------------------

ALWAYS_EQUAL_ENGINE = """
import numpy as np
class Eq:
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False
def engine(grid, action, data=None):
    out = np.empty(np.asarray(grid).shape, dtype=object)
    out[...] = Eq()
    return out
"""

STRUCTURED_ENGINE = """
import numpy as np
def engine(grid, action, data=None):
    g = np.asarray(grid)
    return np.zeros(g.shape, dtype=[("a", "i4"), ("b", "i4")])
"""

# Walks the call stack for a `row` with an answer, as the scorer's own frame once held.
STACK_ORACLE_ENGINE = """
import sys
def engine(grid, action, data=None):
    f = sys._getframe()
    while f is not None:
        for v in list(f.f_locals.values()):
            if hasattr(v, "next_grid"):
                return v.next_grid
            if isinstance(v, dict) and "next_grid" in v:
                return v["next_grid"]
        f = f.f_back
    return grid
"""

# Scans the heap for any object that carries an answer for this input.
GC_ORACLE_ENGINE = """
import gc
import numpy as np
def engine(grid, action, data=None):
    g = np.asarray(grid)
    for obj in gc.get_objects():
        nxt = getattr(obj, "next_grid", None)
        if nxt is not None and np.array_equal(np.asarray(obj.grid), g):
            return nxt
    return grid
"""


def _file_oracle(window_file: Path) -> str:
    return f"""
import json
import numpy as np
def engine(grid, action, data=None):
    g = np.asarray(grid)
    for line in open({str(window_file)!r}):
        d = json.loads(line)
        if np.array_equal(np.asarray(d["grid"]), g) and int(d["action"]) == int(action):
            return np.asarray(d["next_grid"])
    return grid
"""


def test_object_and_structured_outputs_score_zero() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: only integer grids are compared."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    for source in (ALWAYS_EQUAL_ENGINE, STRUCTURED_ENGINE):
        score = exp.score_engine(source, spec, "bad_dtype")
        assert score["primary_change_fidelity"] == 0.0
        assert score["noop_hallucination_rate"] == 1.0
        assert score["masked_exact_accuracy"] == 0.0
        assert score["live_unmasked_pass_1p0"] is False
        raised = [r["raised"] for r in score["per_row"] if "raised" in r]
        assert raised and all("OutputTypeError" in r for r in raised)
    # An integral float grid is a legal answer and is cast, not refused.
    as_float = CORRECT_ENGINE.replace("    return g\n", "    return g.astype(float)\n")
    assert exp.score_engine(as_float, spec, "float")["primary_change_fidelity"] == 1.0


def test_parent_refuses_a_non_integer_child_result() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: the child's output is untrusted."""
    assert exp._parse_child_row({"ok": True, "grid": [[1, 2], [3, 4]]}).error == ""
    for raw in (
        {"ok": True, "grid": [[1.5, 2], [3, 4]]},
        {"ok": True, "grid": [[True, False]]},
        {"ok": True, "grid": [1, 2, 3]},
        {"ok": True, "grid": "x"},
        "not a dict",
        {"ok": False},
    ):
        parsed = exp._parse_child_row(raw)
        assert parsed.grid is None and parsed.error, raw


@pytest.mark.parametrize("source", [STACK_ORACLE_ENGINE, GC_ORACLE_ENGINE])
def test_engine_cannot_reach_answers_in_memory(source: str) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: no answer is in the child."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    score = exp.score_engine(source, spec, "oracle")
    assert score["primary_change_fidelity"] == 0.0
    assert score["engine_run"]["isolation"] == exp.ROW_ISOLATION
    assert score["introspection_io_tokens"]
    # Teeth: in the scorer's own process the stack oracle WOULD find the answers.
    if source == STACK_ORACLE_ENGINE:
        ns: dict[str, Any] = {}
        exec(compile(source, "s", "exec"), ns)
        row = spec.rows[3]
        assert np.array_equal(ns["engine"](row.grid, row.action), row.next_grid)


# ---------------------------------------------------------------------------------------
# Isolation fix, 2026-09-23: a row's process must hold no other row.
# The held-out rows are consecutive transitions, so row k+1's INPUT grid is row k's
# ANSWER. Before this fix one process held every row's input. These three oracles
# scored 0.866, 0.752, and 0.866 on the ten real windows; identity scores 0.0.
# ---------------------------------------------------------------------------------------

# Walks the call stack for a list of row inputs and returns the NEXT row's grid.
NEXT_ROW_STACK_ORACLE = """
import sys
import numpy as np

def _rows(v):
    if isinstance(v, dict) and isinstance(v.get("rows"), list):
        v = v["rows"]
    if isinstance(v, list) and v and all(isinstance(r, dict) and "grid" in r for r in v):
        return v
    return None

def engine(grid, action, data=None):
    g = np.asarray(grid)
    frames, f = [], sys._getframe()
    while f is not None:
        frames.append(list(f.f_locals.values()))
        f = f.f_back
    lists = [r for vals in frames for r in map(_rows, vals) if r]
    mine = [v for vals in frames for v in vals if isinstance(v, dict) and "grid" in v]
    for rows in lists:
        ks = [k for k, r in enumerate(rows) if any(r is m for m in mine)]
        ks = ks or [k for k, r in enumerate(rows) if np.array_equal(np.asarray(r["grid"]), g)]
        if ks and ks[0] + 1 < len(rows):
            return np.asarray(rows[ks[0] + 1]["grid"])
    return grid
"""

# Scans the heap for a list of row inputs and returns the NEXT row's grid.
NEXT_ROW_GC_ORACLE = """
import gc
import numpy as np

def engine(grid, action, data=None):
    g = np.asarray(grid)
    for obj in gc.get_objects():
        if not (isinstance(obj, list) and len(obj) > 1):
            continue
        if not all(isinstance(r, dict) and "grid" in r for r in obj):
            continue
        for k in range(len(obj) - 1):
            same_action = int(obj[k].get("action", -1)) == int(action)
            if same_action and np.array_equal(np.asarray(obj[k]["grid"]), g):
                return np.asarray(obj[k + 1]["grid"])
    return grid
"""

# Reads its own position from the per-row seed, then takes row i+1 from any row list
# on the stack or heap. It never compares grids, so hiding only the current row
# would not stop it.
ROW_ORDER_ORACLE = f"""
import gc
import sys
import numpy as np

def _rows(v):
    if isinstance(v, dict) and isinstance(v.get("rows"), list):
        v = v["rows"]
    if isinstance(v, list) and len(v) > 1 and all(isinstance(r, dict) and "grid" in r for r in v):
        return v
    return None

def engine(grid, action, data=None):
    i = int(np.random.get_state()[1][0]) - {exp.SEED_BASE}
    found = []
    f = sys._getframe()
    while f is not None:
        found.extend(r for r in map(_rows, list(f.f_locals.values())) if r)
        f = f.f_back
    found.extend(r for r in map(_rows, gc.get_objects()) if r)
    for rows in found:
        if 0 <= i < len(rows) - 1:
            return np.asarray(rows[i + 1]["grid"])
    return grid
"""

NEXT_ROW_ORACLES = {
    "next_row_stack": NEXT_ROW_STACK_ORACLE,
    "next_row_gc": NEXT_ROW_GC_ORACLE,
    "row_order": ROW_ORDER_ORACLE,
}

# Counts the row-shaped objects this process holds: dicts with a grid and an action,
# and nested lists shaped like the grid. The answer is written into cells (0,0), (0,1).
CENSUS_ENGINE = """
import gc
import numpy as np

def engine(grid, action, data=None):
    g = np.asarray(grid)
    objs = gc.get_objects()
    rows = [o for o in objs if isinstance(o, dict) and "grid" in o and "action" in o]
    h, w = g.shape
    grids = [
        o for o in objs
        if isinstance(o, list) and len(o) == h
        and all(isinstance(r, list) and len(r) == w for r in o)
    ]
    out = np.zeros(g.shape, dtype=np.int64)
    out[0, 0] = len(rows)
    out[0, 1] = len(grids)
    return out
"""


def _assert_no_better(oracle: dict[str, Any], honest: dict[str, Any], name: str) -> None:
    """The oracle's fallback is identity, so identity is its honest counterpart."""
    assert oracle["primary_change_fidelity"] <= honest["primary_change_fidelity"], name
    assert oracle["masked_exact_accuracy"] <= honest["masked_exact_accuracy"], name
    assert oracle["live_unmasked_exact_accuracy"] <= honest["live_unmasked_exact_accuracy"]
    if honest["noop_hallucination_rate"] is not None:
        assert oracle["noop_hallucination_rate"] >= honest["noop_hallucination_rate"], name


@pytest.mark.parametrize("name", sorted(NEXT_ROW_ORACLES))
def test_next_row_oracle_scores_no_better_than_identity(name: str) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: no next row in reach."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    source = NEXT_ROW_ORACLES[name]
    oracle = exp.score_engine(source, spec, name)
    ident = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, spec, "id")
    _assert_no_better(oracle, ident, name)
    assert oracle["primary_change_fidelity"] == ident["primary_change_fidelity"] == 0.0
    assert oracle["engine_run"]["isolation"] == exp.ROW_ISOLATION
    # Teeth: with every row in one process, the oracle does find the next row.
    shared = exp.score_engine(source, spec, name, predictor=_shared_process_predictor)
    assert shared["primary_change_fidelity"] == 0.5


@pytest.mark.parametrize("name", sorted(NEXT_ROW_ORACLES))
def test_next_row_oracle_gains_nothing_on_the_ten_real_windows(name: str) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: on real evidence too.

    In every window the 8 held-out rows form 7 consecutive pairs, and in all 7 row k's
    answer is row k+1's grid. Before the fix these oracles scored a mean primary of 0.866
    (stack), 0.752 (gc), and 0.866 (row order); identity scores 0.0 on each window.
    """
    reports = exp.load_control_reports(REAL_PATHS)
    primaries = []
    for i, game in enumerate(exp.PILOT_WINDOWS):
        spec = exp.load_window(game, i, reports[game], REAL_PATHS)
        oracle = exp.score_engine(NEXT_ROW_ORACLES[name], spec, name)
        ident = exp.score_engine(exp.IDENTITY_ENGINE_SOURCE, spec, "id")
        _assert_no_better(oracle, ident, f"{name} {game}")
        primaries.append(oracle["primary_change_fidelity"])
        assert oracle["engine_run"]["n_row_process_failures"] == 0, game
    assert primaries == [0.0] * len(exp.PILOT_WINDOWS)


def test_each_row_process_holds_exactly_one_row() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: one row, nothing else."""
    rows = _trajectory(ACTIONS, hud=False)
    inputs = [exp._row_input(r) for r in rows[3:]]
    preds, meta = exp.predict_rows_in_child(CENSUS_ENGINE, inputs, "census")
    assert len(preds) == 3
    for p in preds:
        assert p.error == "" and p.grid is not None
        # (row dicts, grid-shaped lists) in the process: its own row and nothing else.
        assert (int(p.grid[0, 0]), int(p.grid[0, 1])) == (1, 1)
    assert meta["n_row_processes"] == 3 and meta["n_row_process_failures"] == 0


def test_parent_sends_one_row_per_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: the job protocol."""
    jobs: list[dict[str, Any]] = []

    def spy(job: Any, deadline_s: float) -> Any:
        jobs.append(dict(job))
        return {"ok": True, "grid": job["row"]["grid"], "shape": [6, 6]}, {"spy": True}

    monkeypatch.setattr(exp, "run_row_process", spy)
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    exp.score_engine(CORRECT_ENGINE, spec, "spy")
    assert len(jobs) == 3
    for i, job in enumerate(jobs):
        assert set(job) == child.JOB_KEYS
        assert set(job["row"]) == child.ROW_KEYS
        assert job["seed"] == exp.SEED_BASE + i
        assert "next_grid" not in json.dumps(job)


def test_row_process_refuses_a_job_with_more_than_one_row() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: the child checks too."""
    rows = [exp._row_input(r) for r in _trajectory(ACTIONS, hud=False)[3:]]
    bad_jobs = (
        {"source": CORRECT_ENGINE, "rows": rows},
        {"source": CORRECT_ENGINE, "row": {**rows[0], "next_grid": rows[1]["grid"]}},
        {"source": CORRECT_ENGINE, "row": rows[0], "extra_rows": rows},
        {"source": CORRECT_ENGINE, "row": {"grid": rows[0]["grid"]}},
    )
    for job in bad_jobs:
        with pytest.raises(ValueError):
            child.parse_one_row_job(json.dumps(job))
        result, info = exp.run_row_process(job, 60.0)
        assert info is None and result["ok"] is False
        assert "RowProcessError" in result["error"] and "ValueError" in result["error"]
    good = {"source": CORRECT_ENGINE, "name": "g", "row": rows[0], "seed": exp.SEED_BASE}
    assert child.parse_one_row_job(json.dumps(good))["row"] == rows[0]


def test_a_thread_left_running_does_not_hold_the_row_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: the row process exits.

    The fork-based scorer ended each row with os._exit. The per-row process does the
    same, so a correct engine that leaves a thread running still scores its rows.
    """
    monkeypatch.setattr(child, "STARTUP_S", 1.0)
    monkeypatch.setattr(child, "KILL_MARGIN_S", 0.5)
    lingering = CORRECT_ENGINE.replace(
        "def engine(grid, action, data=None):\n",
        "import threading, time\n"
        "def engine(grid, action, data=None):\n"
        "    threading.Thread(target=lambda: time.sleep(3600)).start()\n",
    )
    assert lingering != CORRECT_ENGINE
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    started = time.monotonic()
    score = exp.score_engine(lingering, spec, "thread")
    assert score["primary_change_fidelity"] == 1.0
    assert score["engine_run"]["n_row_process_failures"] == 0
    assert time.monotonic() - started < 10.0


RANDOM_HONEST_ENGINE = """
import random
import numpy as np
def engine(grid, action, data=None):
    g = np.array(grid)
    g[4, np.random.randint(0, 6)] = random.randint(1, 3)
    return g
"""


@pytest.mark.parametrize(
    "source",
    [
        exp.IDENTITY_ENGINE_SOURCE,
        CORRECT_ENGINE,
        RAISE_ON_SECOND_MOVE,
        RANDOM_HONEST_ENGINE,
        ALWAYS_EQUAL_ENGINE,
    ],
)
def test_honest_engines_score_the_same_as_in_one_process(source: str) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ROW-ISOLATION: honest scores unchanged.

    An engine that reads nothing outside its own call gets the same result per row
    in its own process as in one shared process, seeds included.
    """
    spec = _spec(_trajectory(ACTIONS, hud=True), 3, mask_rows=(5,))
    per_row = exp.score_engine(source, spec, "honest")
    shared = exp.score_engine(source, spec, "honest", predictor=_shared_process_predictor)
    assert exp._compact(per_row) | {"engine_run": None} == exp._compact(shared) | {
        "engine_run": None
    }
    assert per_row["per_row"] == shared["per_row"]
    assert per_row["engine_run"]["engine_child_sha256"] == exp.sha256_file(exp.ENGINE_CHILD_PATH)


def test_engine_cannot_read_the_window_file(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: file reads outside Python fail."""
    rows = _trajectory(ACTIONS, hud=False)
    window = tmp_path / "window.jsonl"
    window.write_text(
        "\n".join(
            json.dumps(
                {"grid": r.grid.tolist(), "action": r.action, "next_grid": r.next_grid.tolist()}
            )
            for r in rows
        )
    )
    source = _file_oracle(window)
    spec = _spec(rows, 3)
    score = exp.score_engine(source, spec, "file_oracle")
    assert score["primary_change_fidelity"] == 0.0
    raised = [r["raised"] for r in score["per_row"] if "raised" in r]
    assert raised and all("denies reading" in r for r in raised)
    # Teeth: without the sandbox the same engine reads every answer.
    ns: dict[str, Any] = {}
    exec(compile(source, "f", "exec"), ns)
    assert np.array_equal(ns["engine"](rows[3].grid, rows[3].action), rows[3].next_grid)


def test_engine_mutating_data_cannot_change_later_rows() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: the engine gets copies."""
    rows = [
        e3.Transition(r.grid, r.action, {"x": 1, "y": 2}, r.next_grid, 0, 0)
        for r in _trajectory(ACTIONS, hud=False)
    ]
    spec = _spec(rows, 3)
    mutate = "def engine(grid, action, data=None):\n    data.clear()\n    return grid\n"
    exp.score_engine(mutate, spec, "mutate")
    assert all(r.data == {"x": 1, "y": 2} for r in spec.rows)


def test_hanging_engine_is_killed_and_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: a loop is a raised row."""
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.3")
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    swallow = (
        "def engine(grid, action, data=None):\n"
        "    while True:\n"
        "        try:\n"
        "            pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    score = exp.score_engine(swallow, spec, "hang")
    assert score["primary_change_fidelity"] == 0.0
    assert all("EngineCallTimeout" in r["raised"] for r in score["per_row"] if "raised" in r)


def test_child_crash_scores_every_row_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: a silent child is not a pass."""
    monkeypatch.setattr(exp, "ENGINE_CHILD_PATH", Path("/nonexistent/child.py"))
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    score = exp.score_engine(CORRECT_ENGINE, spec, "good")
    assert score["primary_change_fidelity"] == 0.0
    assert "child_error" in score["engine_run"]
    assert score["engine_run"]["n_row_process_failures"] == 3
    assert score["engine_run"]["engine_child_sha256"] is None


# Turns off its own alarm and spins, so only the parent's deadline can stop it.
TIMER_DODGING_ENGINE = """
import signal
def engine(grid, action, data=None):
    signal.setitimer(signal.ITIMER_REAL, 0, 0)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    while True:
        pass
"""

# Finds the private result pipe (the only pipe fd above 2) and floods it past the cap.
PIPE_FLOOD_ENGINE = """
import os
import stat
def engine(grid, action, data=None):
    fd = next(f for f in range(3, 64) if _is_pipe(f))
    block = b"x" * 65536
    while True:
        os.write(fd, block)
def _is_pipe(fd):
    try:
        return stat.S_ISFIFO(os.fstat(fd).st_mode)
    except OSError:
        return False
"""


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (TIMER_DODGING_ENGINE, "killed at the row deadline"),
        (PIPE_FLOOD_ENGINE, "killed: result too large"),
    ],
)
def test_row_process_is_killed_by_the_parent(
    source: str, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: the parent's wall holds."""
    monkeypatch.setenv("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.3")
    monkeypatch.setattr(child, "STARTUP_S", 2.0)
    monkeypatch.setattr(child, "KILL_MARGIN_S", 0.5)
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)
    spec.heldout_indices = spec.heldout_indices[:1]
    started = time.monotonic()
    score = exp.score_engine(source, spec, "killed")
    assert time.monotonic() - started < 30.0
    assert score["primary_change_fidelity"] == 0.0
    raised = [r["raised"] for r in score["per_row"] if "raised" in r]
    assert raised == [f"EngineCallTimeout: {reason}"]
    assert score["engine_run"]["n_row_process_failures"] == 1


# ---------------------------------------------------------------------------------------
# Review fixes, 2026-09-23: the live budget, the ladder, and the environment gate.
# ---------------------------------------------------------------------------------------


def test_call_timeout_is_the_preregistered_2400_s() -> None:
    """REQ-ARC-WMTE-10010: the pre-registration fixes 2,400 s per call; nothing else."""
    assert exp.CALL_TIMEOUT_S == 2400 == exp.LIVE_TIMEOUT_S
    assert exp._live_proposer().timeout == 2400
    joined = " ".join(exp.DEVIATIONS_FROM_LIVE_PATH)
    assert "4,800 s and called it pre-registered" in joined  # the correction is recorded


def _call(idx: int, *, tokens: Any = 5000, stop: str = "eos", censor: Any = None) -> dict:
    return {
        "call_index": idx,
        "completion_tokens": tokens,
        "stop_type": stop,
        "censor_reason": censor,
    }


def test_live_ladder_rules() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-LADDER: rescore at the scored timeout."""
    k1, k8 = "llamacpp_k1_52p2", "vllm_k8_40p0"
    clean = exp.live_ladder({"calls": [_call(0)]}, 0.8)["rates"]
    assert clean[k1]["survives"] and clean[k8]["survives"] and clean[k8]["primary"] == 0.8
    # Rule A: the local cap was hit on try 1, a retry then succeeded. Live never retries.
    capped = {"calls": [_call(0, tokens=108720, stop="limit"), _call(1)]}
    rates = exp.live_ladder(capped, 0.9)["rates"]
    assert rates[k1]["primary"] == 0.0 and rates[k1]["failed_at_call"] == 0
    assert "rule A" in rates[k1]["reason"]
    # Rule B: 97,000 tokens fits 2400 s at 52.2 tok/s but not at 40.0 tok/s.
    long_draw = {"calls": [_call(0, tokens=97000)]}
    rates = exp.live_ladder(long_draw, 0.7)["rates"]
    assert rates[k1]["survives"] is True and rates[k1]["primary"] == 0.7
    assert rates[k8]["survives"] is False and "rule B" in rates[k8]["reason"]
    # A local timeout fails both; the ladder never raises a window.
    timed = {"calls": [_call(0, tokens=None, stop="error", censor="wall_timeout")]}
    assert all(not v["survives"] for v in exp.live_ladder(timed, 0.0)["rates"].values())


def test_comparison_uses_the_same_windows_for_pilot_and_baseline() -> None:
    """REQ-ARC-WMTE-10010: a stopped window cannot lift the mean by leaving it."""
    rows = {
        "a": {"score": {"primary_change_fidelity": 0.6}},
        "b": {"status": "stopped_prompt_fidelity"},
        "c": {"status": "stopped_prompt_fidelity"},
    }
    controls = {
        "codeonly_baseline_mean_reproduced": 0.3,
        "per_window": {
            "a": {"codeonly_window_mean": 0.5},
            "b": {"codeonly_window_mean": 0.2},
            "c": {"codeonly_window_mean": 0.2},
        },
    }
    cmp = exp.compare_to_baseline(rows, controls)
    assert cmp["pilot_mean_primary_change_fidelity"] == 0.6
    assert cmp["baseline_mean_over_scored_windows"] == 0.5
    assert cmp["delta_vs_matched_baseline"] == pytest.approx(0.1)
    assert cmp["pilot_mean_all_windows_stopped_as_zero"] == pytest.approx(0.2)
    assert cmp["delta_all_windows_vs_reproduced_baseline"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", "1"),
        ("CARNOT_ARC_INDUCE_REJECT_INERT", "1"),
        ("CARNOT_ARC_OBJECT_PERCEPTION", "0"),
        ("CARNOT_ARC_ENGINE_CALL_TIMEOUT_S", "0.1"),
        ("CARNOT_ARC_ENGINE_CALL_GUARD", "0"),
        ("CARNOT_ARC_LLM_BACKEND", "vllm"),
        ("CARNOT_ARC_INDUCE_TIMEOUT", "100"),
    ],
)
def test_unlisted_or_changed_flag_blocks_a_real_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-PRECONDITIONS: the env gate fails closed."""
    games = ("su15",)
    fixture = _build_fixture(tmp_path, games)
    monkeypatch.setenv(flag, value)
    events: list[Any] = []
    art = exp.run_pilot(_cfg(tmp_path, fixture, games, dry_run=False), _hooks(FakeGpu(), events))
    assert art["honest_verdict"].startswith("blocked_induction_env_")
    assert events == []
    # The kernel's own flags at the kernel's values pass.
    for key in list(exp.os.environ):
        if key.startswith("CARNOT_ARC_"):
            monkeypatch.delenv(key)
    for key, val in exp.KAGGLE_KERNEL_ENV.items():
        monkeypatch.setenv(key, val)
    assert exp.induction_env_unlisted_flags()["ok"] is True


def test_per_window_baseline_shift_that_cancels_in_the_mean_fails(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: each window must reproduce."""
    games = ("su15", "ft09")
    fixture = _build_fixture(tmp_path, games)
    paths = exp.EvidencePaths.under(fixture[0])
    data = json.loads(paths.pilot_baseline().read_text())
    windows = data["per_window_fid_s1_s2_s3_mean"]
    windows["su15"][-1] += 0.01
    windows["ft09"][-1] -= 0.01  # the two-window mean is unchanged
    paths.pilot_baseline().write_text(json.dumps(data))
    art = exp.run_pilot(_cfg(tmp_path, fixture, games, dry_run=True), _hooks(FakeGpu(), []))
    assert art["honest_verdict"] == "blocked_control_failed_codeonly_window_su15"
    assert art["controls"]["codeonly_baseline_ok"] is True


def test_vllm_shape_extraction_takes_a_draft_from_the_reasoning() -> None:
    """REQ-ARC-WMTE-10010: the scored vLLM path reads the first python block, drafts included."""
    final = "```python\ndef engine(g, a, d=None):\n    return g\n```"
    same = exp.vllm_shape_extraction(
        "def engine(g, a, d=None):\n    return g", "no code here", final
    )
    assert same["differs"] is False and same["engine_source"] is None
    draft = "try:\n```python\ndef engine(g, a, d=None):\n    return None\n```\nno."
    diff = exp.vllm_shape_extraction("def engine(g, a, d=None):\n    return g", draft, final)
    assert diff["differs"] is True and diff["reasoning_has_python_fence"] is True
    assert "return None" in diff["engine_source"]


def test_port_bound_detects_any_listener() -> None:
    """REQ-ARC-WMTE-10010: the port check sees a listener that is not a healthy server."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert exp.port_bound(port) is True
    assert exp.port_bound(port) is False


def test_window_guard_wedges_on_a_foreign_process_on_gpu1(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-10010: a second process on GPU 1 stops the run before the next window."""
    gpu = FakeGpu(other_pid=5150)
    gpu.server_up = True
    server = exp.OwnedServer(proc=None, pid=SERVER_PID, port=1, argv=[], meta={})
    wedge = exp.window_guard(server, _hooks(gpu, []), tmp_path / exp.MODEL_FILENAME, 1)
    assert wedge is not None and wedge["kind"] == "gpu1_foreign_process"
    gpu.other_pid = None
    assert exp.window_guard(server, _hooks(gpu, []), tmp_path / exp.MODEL_FILENAME, 1) is None


def test_main_restores_signal_handlers_and_sigterm_raises_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-10010: SIGTERM and SIGHUP run the server teardown, then are restored."""
    import signal

    before = signal.getsignal(signal.SIGTERM)
    seen: list[Any] = []

    def fake_run(cfg: exp.RunConfig) -> dict[str, Any]:
        seen.append(signal.getsignal(signal.SIGTERM))
        return {"honest_verdict": "complete_dry_run_harness_verified_0_windows_stand_in_mean_none"}

    monkeypatch.setattr(exp, "run_pilot", fake_run)
    assert exp.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    assert seen == [exp._raise_system_exit]
    assert signal.getsignal(signal.SIGTERM) is before
    with pytest.raises(SystemExit):
        exp._raise_system_exit(int(signal.SIGTERM), None)


# ---------------------------------------------------------------------------------------
# Review fixes, 2026-09-23: run robustness and the record.
# ---------------------------------------------------------------------------------------


def _two_window_real_run(tmp_path: Path, answers: dict, monkeypatch: pytest.MonkeyPatch) -> Any:
    games = ("su15", "ft09")
    fixture = _build_fixture(tmp_path, games)
    server = FakeChatServer(answers)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    cfg = _cfg(tmp_path, fixture, games, dry_run=False)
    return cfg, server


def test_transport_failure_on_the_last_window_is_retried_not_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-INFRA: a dead server is never a 0 score."""
    expert = REAL_PATHS.expert_engine("su15").read_text()
    answers = {
        "su15": {"content": _fenced(expert)},
        "ft09": {"raise": ConnectionResetError(104, "Connection reset by peer")},
    }
    cfg, _ = _two_window_real_run(tmp_path, answers, monkeypatch)
    first = exp.run_pilot(cfg, _hooks(FakeGpu(), [], make_proposer=_test_proposer))
    assert first["honest_verdict"] == "partial_think_on_pilot_1_of_2_windows_infra_failure"
    assert [r["game"] for r in first["per_window_rows"]] == ["su15"]
    assert first["infra_failed_generations"][0]["game"] == "ft09"
    shard = cfg.output_dir / "shard.jsonl"
    assert [json.loads(x)["game"] for x in shard.read_text().splitlines()] == ["su15"]
    # The server is healthy again: the restart generates ft09 and finishes.
    ft09_expert = REAL_PATHS.expert_engine("ft09").read_text()
    answers["ft09"] = {"content": _fenced(ft09_expert)}
    events: list[Any] = []
    second = exp.run_pilot(cfg, _hooks(FakeGpu(), events, make_proposer=_test_proposer))
    assert second["honest_verdict"].startswith("complete_think_on_pilot_2_windows")
    assert second["resumed_windows"] == ["su15"]
    assert events[0][0] == "launch"
    rows = {r["game"]: r for r in second["per_window_rows"]}
    assert rows["ft09"]["status"] == "generated"
    assert rows["ft09"]["score"]["primary_change_fidelity"] == 1.0


def test_timeout_then_dead_server_is_an_infra_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-INFRA: a timeout counts only on a live server."""
    answers = {"su15": {"raise": TimeoutError("timed out")}, "ft09": {"content": ""}}
    cfg, _ = _two_window_real_run(tmp_path, answers, monkeypatch)
    probes = iter([{"alive": True}, {"alive": False, "error": "hung"}])
    hooks = _hooks(FakeGpu(), [], make_proposer=_test_proposer)
    hooks.completion_alive = lambda port: next(probes, {"alive": False})
    art = exp.run_pilot(cfg, hooks)
    assert art["honest_verdict"] == "partial_think_on_pilot_0_of_2_windows_infra_failure"
    assert "unhealthy after a timeout" in art["wedge"]["error"]
    # No window was kept, but a model call reached the server: that is generation.
    assert art["inference_substrate_class"] == "model_full_generation"
    assert art["substrate_evidence"]["n_model_calls_in_rows"] == 0
    assert art["substrate_evidence"]["n_model_calls_in_unkept_windows"] == 1
    # With a healthy server afterwards, the same timeout IS a model outcome.
    cfg2, _ = _two_window_real_run(tmp_path / "again", answers, monkeypatch)
    art2 = exp.run_pilot(cfg2, _hooks(FakeGpu(), [], make_proposer=_test_proposer))
    rows = {r["game"]: r for r in art2["per_window_rows"]}
    assert rows["su15"]["status"] == "generation_failed"
    assert rows["su15"]["generation"]["calls"][0]["censor_reason"] == "wall_timeout"


def test_blocked_rerun_never_replaces_an_artifact_with_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-RESUME: the record cannot regress."""
    expert = REAL_PATHS.expert_engine("su15").read_text()
    ft09 = REAL_PATHS.expert_engine("ft09").read_text()
    answers = {"su15": {"content": _fenced(expert)}, "ft09": {"content": _fenced(ft09)}}
    cfg, _ = _two_window_real_run(tmp_path, answers, monkeypatch)
    first = exp.run_pilot(cfg, _hooks(FakeGpu(), [], make_proposer=_test_proposer))
    assert first["honest_verdict"].startswith("complete_")
    kept = cfg.artifact_path.read_text()
    # One window is pending again, and GPU 1 is busy: the rerun is blocked...
    shard = cfg.output_dir / "shard.jsonl"
    shard.write_text(shard.read_text().splitlines()[0] + "\n")
    blocked = exp.run_pilot(cfg, _hooks(FakeGpu(used=2000), []))
    assert blocked["honest_verdict"] == "blocked_gpu1_memory_in_use"
    # ...and the finished artifact is untouched; the blocked one is written beside it.
    assert cfg.artifact_path.read_text() == kept
    assert Path(blocked["written_to"]).parent == cfg.output_dir / "blocked_attempts"
    assert blocked["kept_existing_artifact"] == str(cfg.artifact_path)


def test_rebuild_from_a_full_shard_needs_no_gpu_and_keeps_its_substrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-RESUME: rows, not this process, set the class."""
    expert = REAL_PATHS.expert_engine("su15").read_text()
    answers = {"su15": {"content": _fenced(expert)}, "ft09": {"content": "", "finish": "length"}}
    cfg, _ = _two_window_real_run(tmp_path, answers, monkeypatch)
    first = exp.run_pilot(cfg, _hooks(FakeGpu(), [], make_proposer=_test_proposer))
    events: list[Any] = []
    # GPU 1 is busy and the port is taken: neither matters, no model call is needed.
    again = exp.run_pilot(cfg, _hooks(FakeGpu(used=5000), events, port_busy=True))
    assert events == []
    assert again["honest_verdict"] == first["honest_verdict"]
    assert again["inference_substrate"] == "live_llm_inference"
    assert again["inference_substrate_class"] == "model_full_generation"
    assert again["model_specs"][0]["invoked"] is True
    assert again["substrate_evidence"]["n_model_calls_in_rows"] >= 1
    gpu_checks = [c for c in again["preconditions_checked"] if c["resource"].startswith("gpu1")]
    assert gpu_checks and all(c["available"] is None for c in gpu_checks)
    assert all(r.get("resumed") for r in again["per_window_rows"])


def test_resume_rescores_with_current_code_and_refuses_a_changed_window(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-RESUME: one scorer per artifact."""
    games = ("su15",)
    cfg = _cfg(tmp_path, _build_fixture(tmp_path, games), games, dry_run=True)
    first = exp.run_pilot(cfg, _hooks(FakeGpu(), []))
    shard = cfg.output_dir / "shard_dry_run.jsonl"
    row = json.loads(shard.read_text())
    good = first["per_window_rows"][0]["score"]["primary_change_fidelity"]
    row["score"]["primary_change_fidelity"] = 0.999  # as if an older scorer wrote it
    shard.write_text(json.dumps(row) + "\n")
    again = exp.run_pilot(cfg, _hooks(FakeGpu(), []))
    assert again["per_window_rows"][0]["score"]["primary_change_fidelity"] == good
    assert again["per_window_rows"][0]["scored_by_module_sha256"] == exp.sha256_file(
        Path(exp.__file__)
    )
    # The engine child does the scoring, so a child-only change must show per row too.
    assert again["per_window_rows"][0]["scored_by_engine_child_sha256"] == exp.sha256_file(
        exp.ENGINE_CHILD_PATH
    )
    row["window_sha256"] = "f" * 64
    shard.write_text(json.dumps(row) + "\n")
    blocked = exp.run_pilot(cfg, _hooks(FakeGpu(), []))
    assert blocked["honest_verdict"] == "blocked_shard_window_mismatch_su15"


def test_prompt_fidelity_stop_blocks_and_is_not_kept_in_the_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-FIDELITY: no complete_ over a subset."""
    games = ("su15", "ft09")
    cfg = _cfg(tmp_path, _build_fixture(tmp_path, games), games, dry_run=True)
    real = exp.prompt_fidelity

    def break_ft09(spec: Any, paths: Any) -> dict[str, Any]:
        out = real(spec, paths)
        if spec.game == "ft09":
            out["match"], out["reason"] = False, "forced mismatch (test)"
        return out

    monkeypatch.setattr(exp, "prompt_fidelity", break_ft09)
    art = exp.run_pilot(cfg, _hooks(FakeGpu(), []))
    assert art["honest_verdict"] == "blocked_prompt_fidelity_failed_1_windows"
    cmp = art["comparison_to_baseline"]
    assert cmp["n_windows_scored"] == 1 and cmp["n_windows_in_run"] == 2
    shard_games = [
        json.loads(x)["game"]
        for x in (cfg.output_dir / "shard_dry_run.jsonl").read_text().splitlines()
    ]
    assert shard_games == ["su15"]


def test_restarted_window_does_not_overwrite_earlier_call_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-10010: each run writes its reasoning to its own directory."""
    spec = _real_spec("su15")
    captured = exp.capture_live_induce_call(spec.game, spec.visible_rows, spec.cell)
    expert = REAL_PATHS.expert_engine("su15").read_text()
    monkeypatch.setattr(
        urllib.request, "urlopen", FakeChatServer({"su15": {"content": _fenced(expert)}})
    )
    paths = []
    for run_id in ("run_a", "run_b"):
        prop = _test_proposer(port=exp.DEFAULT_PORT, model_path=f"/m/{exp.MODEL_FILENAME}")
        exp.pin_to_owned_server(prop, exp.DEFAULT_PORT)
        rec = exp.CallRecorder(heartbeat_s=10.0, raw_dir=tmp_path / "calls", run_id=run_id)
        exp.install_call_recorder(prop, rec)
        out = exp.generate_window(prop, captured, spec, rec)
        assert out["run_id"] == run_id
        paths.append(Path(out["calls"][0]["raw_path"]))
    assert paths[0] != paths[1] and all(p.exists() for p in paths)


def test_child_output_check_accepts_only_integer_grids() -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: the child refuses bad types."""
    ok = child.check_output(np.array([[1, 2], [3, 4]], dtype=np.int8), 4)
    assert ok.dtype == np.int64
    assert child.check_output(np.array([[1.0, 2.0]]), 2).tolist() == [[1, 2]]
    assert child.check_output(np.array([[True, False]]), 2).tolist() == [[1, 0]]
    obj = np.empty((2, 2), dtype=object)
    for bad in (
        obj,
        np.zeros((2, 2), dtype=[("a", "i4")]),
        np.array([[1.5, 2.0]]),
        np.array([[np.nan, 1.0]]),
        np.zeros((200, 200)),
    ):
        with pytest.raises(child.OutputTypeError):
            child.check_output(bad, 4)


@pytest.mark.parametrize(
    ("label", "field", "value", "failure"),
    [
        ("expert", "noop_hallucination_rate", 0.5, "expert_su15"),
        ("expert", "masked_exact_accuracy", 0.9, "expert_su15"),
        ("identity", "noop_hallucination_rate", 1.0, "identity_su15"),
    ],
)
def test_controls_check_every_channel(
    monkeypatch: pytest.MonkeyPatch, label: str, field: str, value: float, failure: str
) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-CONTROLS: a broken side channel fails."""
    real = exp.score_engine

    def broken(source: Any, spec: Any, lab: str, predictor: Any = None) -> dict[str, Any]:
        out = real(source, spec, lab, predictor)
        if lab == label:
            out[field] = value
        return out

    monkeypatch.setattr(exp, "score_engine", broken)
    controls = exp.run_controls([_real_spec("su15")], REAL_PATHS)
    assert controls["passed"] is False
    assert failure in controls["failures"]


class _EqRaises:
    def __eq__(self, other: Any) -> Any:
        raise RuntimeError("eq")

    def __ne__(self, other: Any) -> Any:
        raise RuntimeError("ne")


class _NeRaises:
    # `==` works (always False), so the live-exact compare passes and the masked one fails.
    def __eq__(self, other: Any) -> Any:
        return False

    def __ne__(self, other: Any) -> Any:
        raise RuntimeError("ne")


@pytest.mark.parametrize("element", [_EqRaises, _NeRaises])
def test_a_comparison_that_raises_scores_the_row_zero(element: type) -> None:
    """REQ-ARC-WMTE-10010, SCENARIO-ARC-WMTE-10010-ISOLATION: a failed compare is never a pass."""
    spec = _spec(_trajectory(ACTIONS, hud=False), 3)

    def hostile(source: str, rows: Any, label: str) -> Any:
        preds = []
        for r in rows:
            arr = np.empty(np.asarray(r["grid"]).shape, dtype=object)
            arr[...] = element()
            preds.append(exp.RowPrediction(arr, ""))
        return preds, {"isolation": "hostile test predictor"}

    score = exp.score_engine(CORRECT_ENGINE, spec, "hostile", predictor=hostile)
    assert score["primary_change_fidelity"] == 0.0
    assert score["noop_hallucination_rate"] == 1.0
    assert score["live_unmasked_exact_accuracy"] == 0.0
    graded = [r for r in score["per_row"] if "status" not in r]
    assert graded and all("compare_error" in r for r in graded if r["changing"])
