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
