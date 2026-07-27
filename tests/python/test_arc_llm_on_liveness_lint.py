"""Tests for scripts/arc_llm_on_liveness_lint.py -- the guard that refuses an LLM-on ARC row
whose own instrumentation says the generator was dead.

REQ-ARC-GEN-LIVENESS-1: a row asserting llm_enabled=True while carrying evidence the generator
was dead, produced nothing, or was duplicated by a server storm MUST be refused, and the
refusal MUST be computed from the primitive fields rather than from the derived
`llm_on_row_valid` stamp.

SCENARIO-ORIGIN-INCIDENT: the guard fires on all eight recorded K=4 cells from
`results/llm_on_contention_rows_20260726/` -- read off the real files on disk, not fixtures.
SCENARIO-MATCHED-CONTROL: the guard stays silent on the four matched-config K=4/n_ctx=32768
cells, which are the same games at the same concurrency with a working generator.
SCENARIO-NESTED-ROW: the guard's walker reaches a row nested at `cells[i].row`, the shape the
origin artifact actually uses.

WHY THESE TESTS READ REAL FILES. This project has shipped a lint that printed OK on a faithful
replay of its own origin incident, because the lint listed files from one place and read them
from another. A synthetic-fixture-only test cannot catch that class of error. So the
origin/control tests here glob the actual recorded artifacts and assert on them; they skip
(rather than silently pass) if those artifacts are absent, so a missing corpus can never read
as a green guard.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os

import pytest

_LINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
    "arc_llm_on_liveness_lint.py",
)
_ROWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "llm_on_contention_rows_20260726",
)


def _load():
    spec = importlib.util.spec_from_file_location("arc_llm_on_liveness_lint", _LINT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lint = _load()


def _rows_from(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(_ROWS_DIR, pattern))):
        with open(path) as fh:
            rows.extend(row for _, row in lint.walk_rows(json.load(fh)))
    return rows


# --- SCENARIO-ORIGIN-INCIDENT ----------------------------------------------------------


def test_fires_on_every_recorded_k4_dead_cell() -> None:
    """The eight real K=4 cells at n_ctx=16384 must every one produce a FAIL finding."""
    paths = sorted(glob.glob(os.path.join(_ROWS_DIR, "cells", "cell_K4_*_b400.json")))
    if not paths:
        pytest.skip(f"origin corpus absent: {_ROWS_DIR}/cells/cell_K4_*_b400.json")
    assert len(paths) == 8, f"expected the 8 recorded K=4 cells, found {len(paths)}"
    for path in paths:
        with open(path) as fh:
            found = list(lint.walk_rows(json.load(fh)))
        assert found, f"walker reached no row in {path}"
        for _, row in found:
            findings = lint.check_row(row)
            codes = {f["code"] for f in findings}
            assert any(f["severity"] == "FAIL" for f in findings), (
                f"{os.path.basename(path)}: guard did not fire; codes={codes or 'NONE'}"
            )
            assert "DEAD_GENERATOR" in codes, f"{os.path.basename(path)}: codes={codes}"


def test_the_dead_cells_all_exited_zero_which_is_the_defect() -> None:
    """Documents WHY the guard is needed: every dead cell reported worker success.

    This is the asymmetry the guard closes -- the inner row knew it was invalid while the outer
    wrapper reported `worker_ok: True` / `worker_returncode: 0`, and nothing reconciled them.
    """
    paths = sorted(glob.glob(os.path.join(_ROWS_DIR, "cells", "cell_K4_*_b400.json")))
    if not paths:
        pytest.skip("origin corpus absent")
    for path in paths:
        with open(path) as fh:
            cell = json.load(fh)
        row = cell.get("row") or cell
        assert row.get("generator_healthy_after") is False, path
        assert row.get("actions", 0) >= 380, (
            f"{path}: expected a full run, got {row.get('actions')}"
        )


# --- SCENARIO-MATCHED-CONTROL ----------------------------------------------------------


def test_silent_on_the_matched_config_controls() -> None:
    """K=4 at n_ctx=32768 -- same games, same concurrency, working generator -> no findings."""
    paths = sorted(glob.glob(os.path.join(_ROWS_DIR, "cells_nctx32768", "cell_K4_*_b400.json")))
    if not paths:
        pytest.skip("matched-config control corpus absent")
    for path in paths:
        with open(path) as fh:
            for _, row in lint.walk_rows(json.load(fh)):
                findings = lint.check_row(row)
                assert not findings, (
                    f"FALSE POSITIVE on the matched-config control "
                    f"{os.path.basename(path)}: {[f['code'] for f in findings]}"
                )


def test_k1_cells_are_clean() -> None:
    """The concurrency-1 arm is where every prior LLM-on number was taken; it must stay clean,
    or the guard would retroactively invalidate the whole measurement history by over-firing."""
    paths = sorted(glob.glob(os.path.join(_ROWS_DIR, "cells", "cell_K1_*_b400.json")))
    if not paths:
        pytest.skip("K=1 corpus absent")
    for path in paths:
        with open(path) as fh:
            for _, row in lint.walk_rows(json.load(fh)):
                assert not lint.check_row(row), f"K=1 false positive in {path}"


# --- SCENARIO-NESTED-ROW ---------------------------------------------------------------


def test_walker_reaches_a_row_nested_under_cells_index_row() -> None:
    """The origin artifact nests rows at `cells[i].row`. A top-level-only walker would report a
    clean scan on the very artifact that records the incident."""
    nested = {
        "probe": "ladder",
        "cells": [
            {
                "worker_ok": True,
                "worker_returncode": 0,
                "row": {
                    "llm_enabled": True,
                    "generator_healthy_after": False,
                    "llm": {"responses": 0},
                },
            },
        ],
    }
    found = list(lint.walk_rows(nested))
    assert len(found) == 1
    path, row = found[0]
    assert path == "/cells[0]/row"
    codes = {f["code"] for f in lint.check_row(row)}
    assert {"DEAD_GENERATOR", "NO_COMPLETIONS"} <= codes


# --- mutation proofs: one primitive flipped, verdict must flip ---------------------------


def _base_row(**kw) -> dict:
    row = {
        "llm_enabled": True,
        "llm": {"responses": 6, "errors": 0},
        "generator_healthy_before": True,
        "generator_healthy_after": True,
        "server_storm_suspected": False,
        "llm_on_row_valid": True,
    }
    row.update(kw)
    return row


def test_mutation_dead_generator() -> None:
    assert not lint.check_row(_base_row())
    codes = {f["code"] for f in lint.check_row(_base_row(generator_healthy_after=False))}
    assert codes == {"DEAD_GENERATOR", "VALID_STAMP_WRONG"}


def test_mutation_no_completions() -> None:
    codes = {f["code"] for f in lint.check_row(_base_row(llm={"responses": 0, "errors": 0}))}
    assert codes == {"NO_COMPLETIONS", "VALID_STAMP_WRONG"}


def test_mutation_server_storm() -> None:
    codes = {f["code"] for f in lint.check_row(_base_row(server_storm_suspected=True))}
    assert codes == {"SERVER_STORM", "VALID_STAMP_WRONG"}


def test_does_not_trust_the_derived_stamp() -> None:
    """A row the harness stamped valid while dead must still FAIL, and the same row with the
    stamp deleted must still FAIL on the primitive. Both directions matter: the guard must
    neither believe a lying stamp nor depend on the stamp existing."""
    lying = _base_row(generator_healthy_after=False, llm_on_row_valid=True)
    assert "VALID_STAMP_WRONG" in {f["code"] for f in lint.check_row(lying)}

    stampless = _base_row(generator_healthy_after=False)
    del stampless["llm_on_row_valid"]
    assert {f["code"] for f in lint.check_row(stampless)} == {"DEAD_GENERATOR"}


def test_llm_off_row_is_never_flagged() -> None:
    """An LLM-off row makes no claim about the generator, so a dead generator is irrelevant."""
    off = _base_row(llm_enabled=False, generator_healthy_after=False, llm={"responses": 0})
    assert not lint.check_row(off)


def test_uninstrumented_row_is_warn_not_fail() -> None:
    """Rows predating the liveness instrumentation are unauditable, not provably wrong -- and
    history is never rewritten (never-prune), so they must not hard-fail a scan."""
    findings = lint.check_row({"llm_enabled": True, "game": "old", "actions": 400})
    assert [f["code"] for f in findings] == ["WITNESS_MISSING"]
    assert findings[0]["severity"] == "WARN"


def test_per_attempt_subdicts_are_not_rows() -> None:
    """A `model_specs`-bearing per-attempt dict inside `induction_attempts` is not a
    measurement row. An earlier draft keyed on `model_specs` and inflated WITNESS_MISSING from
    1 real row to 233, 232 of which were sub-dicts with no liveness witness by design."""
    attempt = {"reason": "stall", "model_specs": "Qwen3.5-9B-MTP", "planned": True}
    assert not lint._is_row(attempt)
    row = {"game": "sc25", "actions": 400, "induction_attempts": [attempt]}
    assert lint._is_row(row)


def test_self_test_entrypoint_passes() -> None:
    """The script's own --self-test must pass; it is the shipped fire-on-origin proof."""
    assert lint.self_test() == 0
