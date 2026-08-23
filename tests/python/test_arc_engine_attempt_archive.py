"""REQ-ARC-WMTE-6690: per-attempt retention of induced world models.

THE DEFECT UNDER TEST. The live store `E3_DIR/<game>/world_model.py` is keyed by game
only, so every re-induction overwrites the previous model. Measured on the 2026-08-22
25-game baseline run: 40 induction attempts -> 25 surviving files, 15 induced models
destroyed by last-write-wins. These tests pin the fix: every producer write is archived
under `attempts/`, and NOTHING about the canonical write, return values, or read path
changes.

Every assertion here was proven by deletion (mutation): removing the archive call, the
dedup check, the kill-switch check, the fail-open try, or the harness wiring each turns
at least one test RED. Evidence in the commit message.

Spec: REQ-ARC-WMTE-6690, SCENARIO-ARC-WMTE-6690-1..6
(openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

_ENGINE_A = "import numpy as np\ndef engine(grid, action, data):\n    return np.asarray(grid)\n\ndef is_level_complete(grid):\n    return False\n"
_ENGINE_B = _ENGINE_A.replace("return False", "return True")
_ENGINE_C = _ENGINE_A.replace("np.asarray(grid)", "np.asarray(grid) * 1")

_HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py"


def _sha16(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()[:16]


def _proposer_with_stub(code: str) -> LocalGGUFProposer:
    """A proposer whose generate() returns `code` without any server or GPU."""
    prop = LocalGGUFProposer()
    prop.generate = lambda *a, **k: (True, code)  # type: ignore[method-assign]
    return prop


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Redirect the engine store to tmp_path -- the documented test pattern; the
    _guard_engine_write test-guard refuses the tracked store otherwise."""
    monkeypatch.setattr(awm, "E3_DIR", tmp_path)
    monkeypatch.delenv("CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE", raising=False)
    return tmp_path


def _manifest_lines(store: Path, game: str) -> list[dict]:
    p = store / game / "attempts" / "manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines()]


# SCENARIO-ARC-WMTE-6690-1: canonical write unchanged, attempt archived.
def test_gen_to_file_archives_and_leaves_canonical_write_unchanged(store):
    prop = _proposer_with_stub(_ENGINE_A)
    ok, msg = prop._gen_to_file("gme", "prompt")
    assert ok is True
    # The canonical contract, byte for byte and message for message.
    assert (store / "gme" / "world_model.py").read_text() == _ENGINE_A
    assert msg == "local gguf (GPU server) wrote world_model.py"
    # The retained attempt: one archived copy whose content and hash match the canonical file.
    archived = list((store / "gme" / "attempts").glob("wm_*.py"))
    assert len(archived) == 1
    assert archived[0].read_text() == _ENGINE_A
    assert archived[0].name.endswith(f"__{_sha16(_ENGINE_A)}.py")
    lines = _manifest_lines(store, "gme")
    assert len(lines) == 1
    assert lines[0]["writer"] == "gen_to_file"
    assert lines[0]["sha256_16"] == _sha16(_ENGINE_A)
    assert lines[0]["deduplicated"] is False
    assert prop.last_attempt_archive["archived"] is True


# SCENARIO-ARC-WMTE-6690-1 for the second producer seam.
def test_write_world_model_archives_with_note(store):
    prop = LocalGGUFProposer()
    ok, msg = prop._write_world_model("gme", _ENGINE_A, note="split induce: x")
    assert ok is True and "split induce: x" in msg
    assert (store / "gme" / "world_model.py").read_text() == _ENGINE_A
    lines = _manifest_lines(store, "gme")
    assert len(lines) == 1
    assert lines[0]["writer"] == "write_world_model"
    assert lines[0]["note"] == "split induce: x"


# NO-BEHAVIOUR-CHANGE PROOF: archive ON vs OFF give identical (ok, msg, canonical bytes).
def test_archive_on_off_equivalence(tmp_path, monkeypatch):
    results = {}
    for setting, sub in ((None, "on"), ("0", "off")):
        root = tmp_path / sub
        monkeypatch.setattr(awm, "E3_DIR", root)
        if setting is None:
            monkeypatch.delenv("CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE", raising=False)
        else:
            monkeypatch.setenv("CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE", setting)
        prop = _proposer_with_stub(_ENGINE_A)
        ok, msg = prop._gen_to_file("gme", "prompt")
        results[sub] = (ok, msg, (root / "gme" / "world_model.py").read_text())
    assert results["on"] == results["off"]


# SCENARIO-ARC-WMTE-6690-3: kill switch -- no attempts dir at all, canonical unchanged.
def test_kill_switch_writes_no_archive(store, monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE", "0")
    prop = _proposer_with_stub(_ENGINE_A)
    ok, _ = prop._gen_to_file("gme", "prompt")
    assert ok is True
    assert (store / "gme" / "world_model.py").read_text() == _ENGINE_A
    assert not (store / "gme" / "attempts").exists()
    assert prop.last_attempt_archive == {
        "enabled": False,
        "archived": False,
        "deduplicated": False,
    }


# SCENARIO-ARC-WMTE-6690-2: dedup by content hash -- an attempt LOG, not a blob store.
def test_same_content_twice_archives_one_file_two_manifest_lines(store):
    prop = LocalGGUFProposer()
    prop._write_world_model("gme", _ENGINE_A)
    prop._write_world_model("gme", _ENGINE_A)
    archived = list((store / "gme" / "attempts").glob("wm_*.py"))
    assert len(archived) == 1
    lines = _manifest_lines(store, "gme")
    assert len(lines) == 2
    assert lines[0]["deduplicated"] is False and lines[0]["file"]
    assert lines[1]["deduplicated"] is True and lines[1]["file"] is None


# SCENARIO-ARC-WMTE-6690-5: the defect's shape, reduced to zero loss.
def test_three_attempts_all_retained_canonical_holds_last(store):
    prop = LocalGGUFProposer()
    for code in (_ENGINE_A, _ENGINE_B, _ENGINE_C):
        prop._write_world_model("gme", code)
    # Canonical: last-write-wins, exactly as shipped.
    assert (store / "gme" / "world_model.py").read_text() == _ENGINE_C
    # Retention: all three attempts survive, none destroyed.
    archived = {p.read_text() for p in (store / "gme" / "attempts").glob("wm_*.py")}
    assert archived == {_ENGINE_A, _ENGINE_B, _ENGINE_C}
    assert len(_manifest_lines(store, "gme")) == 3


# SCENARIO-ARC-WMTE-6690-4: archive failure cannot fail the induction (fail-open past
# the guard, direction stated in the helper's docstring).
def test_archive_failure_does_not_fail_the_write(store):
    (store / "gme").mkdir(parents=True)
    # A FILE occupying the attempts name makes mkdir raise inside the archive helper.
    (store / "gme" / "attempts").write_text("not a directory")
    prop = LocalGGUFProposer()
    ok, msg = prop._write_world_model("gme", _ENGINE_A)
    assert ok is True and msg == "local gguf (GPU server) wrote world_model.py"
    assert (store / "gme" / "world_model.py").read_text() == _ENGINE_A
    info = prop.last_attempt_archive
    assert info["archived"] is False and "error" in info


# The test-guard half stays FAIL-CLOSED: a test reaching the TRACKED store must blow up
# loudly before any write, exactly like the canonical write path (2026-07-30 guard).
# The "tracked store" here is a STAND-IN under tmp_path, patched into BOTH globals the
# guard compares. Aiming this test at the real results/arc_e3 would mean a broken guard
# WRITES THERE -- which happened during this REQ's own mutation run (guard deleted ->
# untracked results/arc_e3/gme/attempts/ appeared and had to be cleaned by hand).
def test_archive_refuses_tracked_evidence_store_from_tests(tmp_path, monkeypatch):
    fake_tracked = tmp_path / "tracked_store"
    monkeypatch.setattr(awm, "E3_DIR", fake_tracked)
    monkeypatch.setattr(awm, "_TRACKED_E3_EVIDENCE_DIR", fake_tracked)
    monkeypatch.delenv("CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE", raising=False)
    with pytest.raises(RuntimeError, match="read-only evidence"):
        awm._archive_engine_attempt("gme", _ENGINE_A, writer="test")
    assert not (fake_tracked / "gme" / "attempts").exists()


# The codex path (dev-only) archives post-hoc: codex writes the file out-of-band.
def test_archive_codex_engine_reads_back_and_archives(store):
    (store / "gme").mkdir(parents=True)
    (store / "gme" / "world_model.py").write_text(_ENGINE_A)
    info = awm._archive_codex_engine("gme")
    assert info["archived"] is True
    lines = _manifest_lines(store, "gme")
    assert len(lines) == 1 and lines[0]["writer"] == "codex"


def test_archive_codex_engine_missing_file_is_soft(store):
    info = awm._archive_codex_engine("gme")
    assert info["archived"] is False and info["error"] == "no_file_after_codex"


# CodexProposer wiring: both methods archive on ok. Source-level pin, proven by deletion.
def test_codex_proposer_methods_call_posthoc_archive():
    for meth in (awm.CodexProposer.induce, awm.CodexProposer.refactor):
        assert "_archive_codex_engine" in inspect.getsource(meth)


# ---------------------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6690-6: harness row provenance.
# ---------------------------------------------------------------------------------------


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lever_harness_under_test", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_attempt_archive_delta_reports_only_this_cells_lines(tmp_path):
    hs = _harness()
    man = tmp_path / "manifest.jsonl"
    man.write_text(
        json.dumps({"sha256_16": "old1"}) + "\n" + json.dumps({"sha256_16": "old2"}) + "\n"
    )
    before = hs._manifest_line_count(man)
    with man.open("a") as fh:
        for sha in ("new1", "new2", "new3"):
            fh.write(json.dumps({"sha256_16": sha}) + "\n")
    delta = hs._attempt_archive_delta(man, before)
    assert delta == {"before": 2, "added": 3, "sha256_16": ["new1", "new2", "new3"]}


def test_attempt_archive_delta_missing_manifest_is_zero(tmp_path):
    hs = _harness()
    man = tmp_path / "absent" / "manifest.jsonl"
    assert hs._manifest_line_count(man) == 0
    assert hs._attempt_archive_delta(man, 0) == {"before": 0, "added": 0, "sha256_16": []}


def test_run_cell_wires_induction_archive_row_field():
    """run_cell snapshots the manifest before the game and records the delta on the row.
    Source pin, proven by deletion: removing either wiring line turns this RED."""
    hs = _harness()
    src = inspect.getsource(hs.run_cell)
    assert 'row["induction_archive"]' in src
    assert "_attempt_archive_delta(_man_path, _man_before)" in src
    assert "_manifest_line_count(_man_path)" in src
