"""Round rows carry emission provenance an audit can resolve (REQ-ARC-WMTE-6643).

SCENARIO-ARC-WMTE-6643-ROUND-PROVENANCE: a successful proposer round records
`engine_emitted_at` and `prompt_sha256`; the induce round also records
`transition_source_path`.
SCENARIO-ARC-WMTE-6643-TRANSITION-SOURCE: the persisted file carries the rows,
the shown/held-out split, and the engine binding hash.

Origin: exp6968 blocked with zero resolvable engine candidates because the round
rows carried only a sha16 — no timestamp, no prompt hash, no transition source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

# Not because these tests leak: the first reinduction call in a fresh worker pays the
# heavy module import, same as test_arc_gateway_card_ground_truth.py (its precedent).
pytestmark = pytest.mark.memory_watchdog_skip

GAME = "prov"
ACTIONS = [1, 3, 0, 2, 1, 3, 0, 2, 3]
_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

ENGINE_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    return False
"""


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def _corpus() -> tuple[list[Transition], np.ndarray]:
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for action in ACTIONS:
        nxt = _true_next(grid, action)
        rows.append(
            Transition(
                grid=grid.copy(),
                action=action,
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        grid = nxt
    return rows, root


class _ArchivingProposer:
    """A scripted proposer that archives through the REAL attempt archiver,
    exactly as `_write_world_model` does on the live path."""

    model_specs = "provenance-test-proposer"

    def __init__(self, store: Path, game: str, source: str) -> None:
        self.store = Path(store)
        self.game = game
        self.source = source
        self.last_prompt_sha256 = hashlib.sha256(b"the induce prompt").hexdigest()

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
        path = self.store / self.game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source)
        self.last_attempt_archive = e3._archive_engine_attempt(
            game,
            self.source,
            writer="test",
            prompt_sha256=self.last_prompt_sha256,
        )
        return True, "wrote"

    def refactor(self, game, vr):  # pragma: no cover - single-round test
        return False, "no refactor in this test"


def _run(monkeypatch, tmp_path, proposer):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    transitions, root = _corpus()
    result = execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda engine, goal, grid: None,
        max_rounds=1,
        min_heldout_accuracy=1.0,
    )
    return result, transitions


def test_round_row_carries_emission_provenance(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-6643-ROUND-PROVENANCE, positive path."""

    proposer = _ArchivingProposer(tmp_path, GAME, ENGINE_SRC)
    result, transitions = _run(monkeypatch, tmp_path, proposer)

    row = result.rounds[0]
    assert row["action"] == "induce"
    assert row["engine_emitted_at"], "archive timestamp must reach the round row"
    assert row["prompt_sha256"] == proposer.last_prompt_sha256
    assert row["transition_source_path"], "induce round must persist its evidence"


def test_transition_source_file_schema_and_binding(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-6643-TRANSITION-SOURCE: the file re-derives the split."""

    proposer = _ArchivingProposer(tmp_path, GAME, ENGINE_SRC)
    result, transitions = _run(monkeypatch, tmp_path, proposer)

    recorded = result.rounds[0]["transition_source_path"]
    # A redirected store records an absolute path; the live store records repo-relative.
    path = Path(recorded)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    payload = json.loads(path.read_text())

    rows = payload["rows"]
    assert len(rows) == len(transitions)
    ids = [row["transition_id"] for row in rows]
    assert len(ids) == len(set(ids)), "transition ids must be unique"
    for row in rows:
        assert np.asarray(row["grid"]).ndim == 2
        assert np.asarray(row["next_grid"]).ndim == 2
    shown = payload["prompt_row_ids"]
    assert 0 < len(shown) < len(rows), "shown prefix must be a strict subset"
    assert set(shown) <= set(ids)
    assert payload["repair_feedback_row_ids"] == []
    assert payload["prompt_sha256"] == proposer.last_prompt_sha256
    # The binding hash is the full digest of the engine bytes, matching the
    # archived engine file byte-for-byte (exp6968's transition_source_engine_hash gate).
    expected = hashlib.sha256(ENGINE_SRC.encode("utf-8", "replace")).hexdigest()
    assert payload["attempt_engine_sha256"] == expected
    archived = next((tmp_path / GAME / "attempts").glob("wm_*.py"))
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected


def test_stale_archive_entry_is_refused(monkeypatch, tmp_path) -> None:
    """A stale archive entry (sha mismatch) must not be attributed to this round."""

    class _StaleProposer(_ArchivingProposer):
        def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
            path = self.store / self.game / "world_model.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.source)
            # A leftover entry from some EARLIER emission: wrong sha16 on purpose.
            self.last_attempt_archive = {
                "archived": True,
                "sha256_16": "0" * 16,
                "ts": "20260101T000000_000000",
                "prompt_sha256": self.last_prompt_sha256,
            }
            return True, "wrote"

    proposer = _StaleProposer(tmp_path, GAME, ENGINE_SRC)
    result, _ = _run(monkeypatch, tmp_path, proposer)

    row = result.rounds[0]
    assert "engine_emitted_at" not in row
    assert "transition_source_path" not in row


def test_archiver_returns_provenance_fields(monkeypatch, tmp_path) -> None:
    """The archive info and manifest line both carry ts / full sha / prompt sha."""

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    prompt_sha = hashlib.sha256(b"p").hexdigest()
    info = e3._archive_engine_attempt(GAME, "x = 1\n", writer="test", prompt_sha256=prompt_sha)

    assert info["archived"] is True
    assert info["ts"]
    assert info["sha256_full"] == hashlib.sha256(b"x = 1\n").hexdigest()
    assert info["sha256_16"] == info["sha256_full"][:16]
    assert info["prompt_sha256"] == prompt_sha
    manifest = (tmp_path / GAME / "attempts" / "manifest.jsonl").read_text().splitlines()
    assert json.loads(manifest[-1])["prompt_sha256"] == prompt_sha


def test_generate_records_prompt_sha(monkeypatch) -> None:
    """`generate()` pins the prompt hash before any transport can fail."""

    proposer = object.__new__(e3.LocalGGUFProposer)
    proposer.n_completion_calls = 0
    monkeypatch.setattr(type(proposer), "_ensure_server", lambda self: False, raising=False)
    monkeypatch.setattr(
        type(proposer), "_effective_model_label", lambda self: "stub", raising=False
    )
    monkeypatch.setattr(
        type(proposer), "_note_server_failure", lambda self, msg: None, raising=False
    )
    ok, _msg = proposer.generate("PROMPT BYTES")
    assert ok is False
    assert proposer.last_prompt_sha256 == hashlib.sha256(b"PROMPT BYTES").hexdigest()
