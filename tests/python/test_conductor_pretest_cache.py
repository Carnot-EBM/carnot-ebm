"""Tests for the pre-test fingerprint cache in scripts/research_conductor.py.

Coverage targets
----------------
- _compute_pretest_fingerprint: stable across calls when nothing changes
- _compute_pretest_fingerprint: 64-char hex output (sha256 contract)
- _pretest_cache_satisfies: empty cache always misses
- _pretest_cache_satisfies: full cache satisfies both subset and full requests
- _pretest_cache_satisfies: subset cache satisfies subset only, not full
- _pretest_cache_satisfies: fingerprint mismatch never satisfies
- _load_pretest_cache: returns empty dict when file missing or malformed
- _save_pretest_cache then _load_pretest_cache: roundtrips a payload

Background: the cache short-circuits run_tests() when the fingerprint of
all tracked .py files matches the last green pre-test for the requested
(or stronger) mode. This is the load-bearing optimization that turns
~17 min/iteration pre-tests into ~5 s when no source file has changed.

Spec: REQ-INFRA-039, SCENARIO-INFRA-047, SCENARIO-INFRA-048
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_conductor():
    """Import scripts/research_conductor.py without executing __main__."""
    spec = importlib.util.spec_from_file_location(
        "_rc_under_test", _REPO_ROOT / "scripts" / "research_conductor.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestComputePretestFingerprint:
    """The fingerprint must be stable when nothing changes and reflect
    changes when tracked files are modified."""

    def test_fingerprint_is_stable_across_calls(self) -> None:
        rc = _load_conductor()
        fp1 = rc._compute_pretest_fingerprint()
        fp2 = rc._compute_pretest_fingerprint()
        assert fp1 == fp2, "fingerprint must be deterministic for unchanged repo"

    def test_fingerprint_is_64_char_hex(self) -> None:
        rc = _load_conductor()
        fp = rc._compute_pretest_fingerprint()
        assert len(fp) == 64
        int(fp, 16)  # raises if not hex


class TestPretestCacheSatisfies:
    """Decision matrix between cached mode and requested mode."""

    def test_empty_cache_never_satisfies(self) -> None:
        rc = _load_conductor()
        fp = "a" * 64
        assert rc._pretest_cache_satisfies("subset", fp, {}) is False
        assert rc._pretest_cache_satisfies("full", fp, {}) is False

    def test_full_cache_satisfies_both_modes(self) -> None:
        rc = _load_conductor()
        fp = "b" * 64
        cache = {"fingerprint": fp, "mode": "full"}
        assert rc._pretest_cache_satisfies("subset", fp, cache) is True
        assert rc._pretest_cache_satisfies("full", fp, cache) is True

    def test_subset_cache_satisfies_subset_only(self) -> None:
        rc = _load_conductor()
        fp = "c" * 64
        cache = {"fingerprint": fp, "mode": "subset"}
        assert rc._pretest_cache_satisfies("subset", fp, cache) is True
        assert rc._pretest_cache_satisfies("full", fp, cache) is False

    def test_fingerprint_mismatch_never_satisfies(self) -> None:
        rc = _load_conductor()
        cache = {"fingerprint": "a" * 64, "mode": "full"}
        assert rc._pretest_cache_satisfies("subset", "b" * 64, cache) is False
        assert rc._pretest_cache_satisfies("full", "b" * 64, cache) is False

    def test_unknown_mode_in_cache_does_not_satisfy(self) -> None:
        rc = _load_conductor()
        fp = "d" * 64
        cache = {"fingerprint": fp, "mode": "garbage"}
        assert rc._pretest_cache_satisfies("subset", fp, cache) is False
        assert rc._pretest_cache_satisfies("full", fp, cache) is False


class TestLoadPretestCache:
    """The loader must never raise — it returns {} on any failure."""

    def test_load_returns_empty_when_file_missing(self, tmp_path, monkeypatch) -> None:
        rc = _load_conductor()
        monkeypatch.setattr(rc, "PRETEST_CACHE_FILE", tmp_path / "missing.json")
        assert rc._load_pretest_cache() == {}

    def test_load_returns_empty_when_file_malformed(self, tmp_path, monkeypatch) -> None:
        rc = _load_conductor()
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        monkeypatch.setattr(rc, "PRETEST_CACHE_FILE", bad)
        assert rc._load_pretest_cache() == {}


class TestSavePretestCache:
    """save → load roundtrip must preserve all fields."""

    def test_save_then_load_roundtrips(self, tmp_path, monkeypatch) -> None:
        rc = _load_conductor()
        cache_path = tmp_path / "subdir" / ".pretest-cache.json"
        monkeypatch.setattr(rc, "PRETEST_CACHE_FILE", cache_path)
        rc._save_pretest_cache("a" * 64, "517 passed in 1036.91s", "full")
        loaded = rc._load_pretest_cache()
        assert loaded["fingerprint"] == "a" * 64
        assert loaded["summary"] == "517 passed in 1036.91s"
        assert loaded["mode"] == "full"
        assert "saved_at" in loaded

    def test_save_creates_parent_dir(self, tmp_path, monkeypatch) -> None:
        rc = _load_conductor()
        nested = tmp_path / "a" / "b" / "c" / "cache.json"
        monkeypatch.setattr(rc, "PRETEST_CACHE_FILE", nested)
        rc._save_pretest_cache("e" * 64, "ok", "subset")
        assert nested.exists()
        payload = json.loads(nested.read_text())
        assert payload["mode"] == "subset"


# Spec refs: REQ-CONDUCTOR-PRETEST-1.
#
# 2026-09-08: the fingerprint globbed only "*.py". 20 test files import
# carnot._rust, which is python/carnot/_rust.*.so -- an untracked build
# artifact. A rebuilt extension left the fingerprint identical, so the gate
# that runs before every task launch reported a cache hit on a stale green.
# We hash what pytest LOADS (the .so), not the .rs sources behind it.


def test_a_rebuilt_compiled_extension_changes_the_fingerprint(tmp_path, monkeypatch) -> None:
    rc = _load_conductor()
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_DIRS", ("python/carnot",))
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_FILES", ())
    pkg = tmp_path / "python" / "carnot"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1\n")
    so = pkg / "_rust.cpython-312-x86_64-linux-gnu.so"
    so.write_bytes(b"\x00" * 16)
    before = rc._compute_pretest_fingerprint()
    so.write_bytes(b"\x00" * 32)  # a rebuild changes size and mtime
    assert rc._compute_pretest_fingerprint() != before


def test_a_python_edit_still_changes_the_fingerprint(tmp_path, monkeypatch) -> None:
    rc = _load_conductor()
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_DIRS", ("python/carnot",))
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_FILES", ())
    pkg = tmp_path / "python" / "carnot"
    pkg.mkdir(parents=True)
    src = pkg / "mod.py"
    src.write_text("x = 1\n")
    before = rc._compute_pretest_fingerprint()
    src.write_text("x = 22\n")
    assert rc._compute_pretest_fingerprint() != before


def test_an_unrelated_suffix_is_ignored(tmp_path, monkeypatch) -> None:
    # Widening must stay bounded: a README or a .rs source must not churn the
    # fingerprint, or every doc edit forces a full pre-test.
    rc = _load_conductor()
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_DIRS", ("python/carnot",))
    monkeypatch.setattr(rc, "PRETEST_FINGERPRINT_FILES", ())
    pkg = tmp_path / "python" / "carnot"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1\n")
    before = rc._compute_pretest_fingerprint()
    (pkg / "notes.md").write_text("hello")
    (pkg / "lib.rs").write_text("fn main() {}")
    assert rc._compute_pretest_fingerprint() == before
