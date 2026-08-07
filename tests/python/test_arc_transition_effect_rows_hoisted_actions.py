"""REQ-ARC-FCP-TRANSITION-ROWS-1 (2026-08-07): `load_cached_transition_effect_rows`
indexed `data["actions"][index]` straight off the lazy `.npz` loader INSIDE the
per-row loop, unlike its six sibling arrays (`grids`, `next_grids`, `xs`, `ys`, `lb`,
`la`), which were all hoisted out of the loop once per file. `NpzFile.__getitem__` is
a decompress-and-cache lookup per key, so this re-paid that lookup once per ROW
instead of once per FILE -- found while profiling a live ft09 run (part of the
sp80/ft09 Kaggle gate-timeout investigation; this function runs once per game-eval
process at agent setup, a fixed cost on ft09's tight timeout margin).

Verified byte-identical output on the real `data/arc_transition_corpus/` corpus
(sha256 of the JSON-serialized first 500 rows, unchanged across the fix) before this
test was written; this test locks that property in with a synthetic, self-contained
fixture so it does not depend on the corpus's current real content.

Spec: REQ-ARC-FCP-TRANSITION-ROWS-1,
SCENARIO-ARC-FCP-TRANSITION-ROWS-1-HOISTED-ACTIONS-MATCH-PER-ROW-LOOKUP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from carnot.agentic.arc_frame_change_predictor import load_cached_transition_effect_rows

TRANSITION_CORPUS_RELATIVE_DIR = "data/arc_transition_corpus"


def _write_synthetic_transition_npz(root: Path, game: str, *, n_rows: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    corpus_dir = root / TRANSITION_CORPUS_RELATIVE_DIR
    corpus_dir.mkdir(parents=True, exist_ok=True)
    grids = rng.randint(0, 10, size=(n_rows, 8, 8)).astype(np.int16)
    next_grids = grids.copy()
    # Mutate roughly half the rows so `changed`/`frame_delta` aren't all-zero/all-one.
    for i in range(0, n_rows, 2):
        next_grids[i, 0, 0] = (int(next_grids[i, 0, 0]) + 1) % 10
    actions = rng.randint(1, 7, size=(n_rows,)).astype(np.int32)
    xs = rng.randint(0, 8, size=(n_rows,)).astype(np.int32)
    ys = rng.randint(0, 8, size=(n_rows,)).astype(np.int32)
    lb = rng.randint(0, 3, size=(n_rows,)).astype(np.int16)
    la = np.clip(lb + rng.randint(0, 2, size=(n_rows,)).astype(np.int16), 0, 5)
    np.savez(
        corpus_dir / f"{game}.npz",
        grids=grids,
        next_grids=next_grids,
        actions=actions,
        xs=xs,
        ys=ys,
        lb=lb,
        la=la,
    )


class TestHoistedActionsMatchPerRowLookup:
    def test_rows_match_a_reference_that_indexes_actions_per_row(self, tmp_path):
        # SCENARIO-ARC-FCP-TRANSITION-ROWS-1-HOISTED-ACTIONS-MATCH-PER-ROW-LOOKUP
        _write_synthetic_transition_npz(tmp_path, "synthgame", n_rows=40, seed=1)

        rows = load_cached_transition_effect_rows(tmp_path)
        assert len(rows) == 40

        # Independent oracle: re-read the same file and index `actions` PER ROW straight
        # off the lazy loader (the pre-fix pattern), and confirm every row's action_id
        # matches -- the exact property whose absence would mean the hoist introduced a
        # stale/misaligned read.
        data = np.load(tmp_path / TRANSITION_CORPUS_RELATIVE_DIR / "synthgame.npz")
        for i, row in enumerate(rows):
            assert row["action_id"] == int(data["actions"][i])
            assert row["step_index"] == i

    def test_action_6_rows_carry_xy_matching_a_per_row_reference(self, tmp_path):
        _write_synthetic_transition_npz(tmp_path, "synthgame", n_rows=40, seed=2)
        rows = load_cached_transition_effect_rows(tmp_path)

        data = np.load(tmp_path / TRANSITION_CORPUS_RELATIVE_DIR / "synthgame.npz")
        for i, row in enumerate(rows):
            if row["action_id"] == 6 and int(data["xs"][i]) >= 0 and int(data["ys"][i]) >= 0:
                assert row["x"] == int(data["xs"][i])
                assert row["y"] == int(data["ys"][i])
            else:
                assert "x" not in row and "y" not in row

    def test_multiple_files_each_keep_their_own_actions_array(self, tmp_path):
        """The bug class this guards against most directly: if a hoisted `actions`
        array leaked ACROSS files (e.g. hoisted outside the file loop instead of
        inside it), a second file's rows would silently read the first file's
        actions. Two files, deliberately different seeds/lengths, must each report
        their OWN action ids."""
        _write_synthetic_transition_npz(tmp_path, "gamea", n_rows=10, seed=3)
        _write_synthetic_transition_npz(tmp_path, "gameb", n_rows=15, seed=4)

        rows = load_cached_transition_effect_rows(tmp_path)
        assert len(rows) == 25

        data_a = np.load(tmp_path / TRANSITION_CORPUS_RELATIVE_DIR / "gamea.npz")
        data_b = np.load(tmp_path / TRANSITION_CORPUS_RELATIVE_DIR / "gameb.npz")
        rows_a = [r for r in rows if r["game"] == "gamea"]
        rows_b = [r for r in rows if r["game"] == "gameb"]
        assert len(rows_a) == 10
        assert len(rows_b) == 15
        for i, row in enumerate(rows_a):
            assert row["action_id"] == int(data_a["actions"][i])
        for i, row in enumerate(rows_b):
            assert row["action_id"] == int(data_b["actions"][i])

    def test_limit_still_returns_the_correct_prefix_after_hoisting(self, tmp_path):
        _write_synthetic_transition_npz(tmp_path, "synthgame", n_rows=40, seed=5)
        limited = load_cached_transition_effect_rows(tmp_path, limit=7)
        full = load_cached_transition_effect_rows(tmp_path)
        assert len(limited) == 7
        assert limited == full[:7]

    def test_real_corpus_output_hash_is_unchanged_by_the_hoist(self):
        """Direct regression anchor: this exact sha256 was captured on the REAL
        data/arc_transition_corpus/ corpus (first 500 rows, JSON-serialized,
        sort_keys=True) before and after the hoist, and matched byte-for-byte. If
        the real corpus has since changed (a legitimate future data update), this
        test will fail and should be re-anchored -- it is not testing "this hash is
        eternal", it is testing "this specific corpus snapshot round-trips
        identically", which is what actually matters for a behavior-preservation
        change landing on top of a live, evolving corpus."""
        rows = load_cached_transition_effect_rows(limit=500)
        import hashlib
        import json

        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
        assert digest == "711f974227f02d80b5c9e8d14a2585118c91d02869ed35ecbf9f98b9f88f8cfc"
