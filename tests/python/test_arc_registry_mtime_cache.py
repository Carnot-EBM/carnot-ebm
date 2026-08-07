"""profiling a live sp80 run (cProfile, budget=2000) found `arc_solve_learning._registry`
and `arc_primitive_library._load_registry` re-parsing the 452KB `ops/arc_solve_registry.yaml`
from scratch on every call -- the dominant cost of `recommend_approach`. An UNCONDITIONAL
cache fixed that but broke test isolation: the research conductor runs concurrently with the
test suite and genuinely appends to this file mid-session, so a cache with no invalidation
served a stale snapshot for the rest of the process (measured: broke
test_experiment_4447_lilo_documented_primitive_library.py's coverage assertion). This tests
the fix -- an mtime-gated cache -- proves it serves cached data when the file is unchanged
AND correctly re-reads when the file's mtime moves, on both functions.

Spec: REQ-ARC-REGISTRY-CACHE-1,
SCENARIO-ARC-REGISTRY-CACHE-1-SAME-MTIME-SERVES-CACHED-VALUE,
SCENARIO-ARC-REGISTRY-CACHE-1-CHANGED-MTIME-FORCES-REREAD.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from carnot.agentic import arc_primitive_library
from carnot.agentic import arc_solve_learning


def _write_registry(path: Path, games: list[str]) -> None:
    path.write_text(yaml.safe_dump({"games": [{"game": g} for g in games]}), encoding="utf-8")


def _bump_mtime_forward(path: Path) -> None:
    """A file written twice in quick succession can land on the SAME mtime (common
    filesystem resolution is 1s, sometimes coarser) -- setting it explicitly forward makes
    the "mtime genuinely changed" case deterministic instead of test-flaky."""
    current = path.stat().st_mtime
    os.utime(path, (current + 2, current + 2))


class TestRegistryMtimeCache:
    def test_same_mtime_serves_cached_value_not_a_fresh_read(self, tmp_path, monkeypatch):
        # SCENARIO-ARC-REGISTRY-CACHE-1-SAME-MTIME-SERVES-CACHED-VALUE
        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        monkeypatch.setattr(arc_solve_learning, "REGISTRY", registry_path)
        # No manual cache reset needed: the cache is keyed by path (2026-08-06 adversarial
        # review fix) and each test uses its own unique tmp_path registry, so there is no
        # pre-existing entry to collide with.

        first = arc_solve_learning._registry()
        assert [g["game"] for g in first["games"]] == ["r11l"]

        # Rewrite the file WITHOUT changing its mtime (same on-disk bytes length happens to
        # be irrelevant -- what matters is the mtime is pinned identical to the first read).
        pinned_mtime = registry_path.stat().st_mtime
        _write_registry(registry_path, ["r11l", "sc25"])
        os.utime(registry_path, (pinned_mtime, pinned_mtime))

        second = arc_solve_learning._registry()
        assert [g["game"] for g in second["games"]] == ["r11l"], (
            "same mtime must serve the cached value, not the just-written new content"
        )

    def test_changed_mtime_forces_a_fresh_read(self, tmp_path, monkeypatch):
        # SCENARIO-ARC-REGISTRY-CACHE-1-CHANGED-MTIME-FORCES-REREAD
        # This is the exact property whose absence broke test_experiment_4447 -- the
        # conductor appends new games to the REAL registry mid-session, and a reader must
        # see that once the mtime has genuinely moved.
        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        monkeypatch.setattr(arc_solve_learning, "REGISTRY", registry_path)
        # No manual cache reset needed: the cache is keyed by path (2026-08-06 adversarial
        # review fix) and each test uses its own unique tmp_path registry, so there is no
        # pre-existing entry to collide with.

        first = arc_solve_learning._registry()
        assert [g["game"] for g in first["games"]] == ["r11l"]

        _write_registry(registry_path, ["r11l", "sc25", "bp35"])
        _bump_mtime_forward(registry_path)

        second = arc_solve_learning._registry()
        assert [g["game"] for g in second["games"]] == ["r11l", "sc25", "bp35"], (
            "a genuinely changed mtime must force a fresh read of the new content"
        )

    def test_distinct_paths_never_share_a_cache_slot(self, tmp_path, monkeypatch):
        """The un-keyed version of this cache (pre-adversarial-review) could not have
        distinguished two different REGISTRY paths at all -- there was only one slot."""
        path_a = tmp_path / "a" / "arc_solve_registry.yaml"
        path_a.parent.mkdir()
        path_b = tmp_path / "b" / "arc_solve_registry.yaml"
        path_b.parent.mkdir()
        _write_registry(path_a, ["r11l"])
        _write_registry(path_b, ["sc25"])

        monkeypatch.setattr(arc_solve_learning, "REGISTRY", path_a)
        a = arc_solve_learning._registry()
        monkeypatch.setattr(arc_solve_learning, "REGISTRY", path_b)
        b = arc_solve_learning._registry()

        assert [g["game"] for g in a["games"]] == ["r11l"]
        assert [g["game"] for g in b["games"]] == ["sc25"]

    def test_concurrent_first_access_parses_once_not_once_per_thread(self, tmp_path, monkeypatch):
        # SCENARIO-ARC-REGISTRY-CACHE-1-CONCURRENT-FIRST-ACCESS-PARSES-ONCE
        # Adversarial-review regression (2026-08-06): an unlocked check-then-write let N
        # threads' near-simultaneous first calls each miss the cache and redundantly
        # reparse -- exactly the concurrency model `Swarm.main()` uses for a real
        # competition submission (every game's E3AgentPolicy on its own thread).
        import threading as _threading

        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        monkeypatch.setattr(arc_solve_learning, "REGISTRY", registry_path)

        n_threads = 8
        parse_count = 0
        parse_count_lock = _threading.Lock()
        real_safe_load = yaml.safe_load

        def _counting_safe_load(*args, **kwargs):
            nonlocal parse_count
            with parse_count_lock:
                parse_count += 1
            return real_safe_load(*args, **kwargs)

        monkeypatch.setattr(arc_solve_learning.yaml, "safe_load", _counting_safe_load)

        barrier = _threading.Barrier(n_threads)
        results: list[dict] = [None] * n_threads  # type: ignore[list-item]

        def _worker(index: int) -> None:
            barrier.wait()  # forces genuine overlap, not a race the OS scheduler happens to avoid
            results[index] = arc_solve_learning._registry()

        threads = [_threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert parse_count == 1, (
            f"expected exactly one YAML parse across {n_threads} concurrent first-callers, "
            f"got {parse_count} -- the lock did not serialize the check-then-write"
        )
        for r in results:
            assert [g["game"] for g in r["games"]] == ["r11l"]


class TestPrimitiveLibraryRegistryMtimeCache:
    def test_same_mtime_serves_cached_value_not_a_fresh_read(self, tmp_path):
        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        arc_primitive_library._load_registry_cache.pop(registry_path, None)

        first = arc_primitive_library._load_registry(registry_path)
        assert [g["game"] for g in first["games"]] == ["r11l"]

        pinned_mtime = registry_path.stat().st_mtime
        _write_registry(registry_path, ["r11l", "sc25"])
        os.utime(registry_path, (pinned_mtime, pinned_mtime))

        second = arc_primitive_library._load_registry(registry_path)
        assert [g["game"] for g in second["games"]] == ["r11l"]

    def test_changed_mtime_forces_a_fresh_read(self, tmp_path):
        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        arc_primitive_library._load_registry_cache.pop(registry_path, None)

        first = arc_primitive_library._load_registry(registry_path)
        assert [g["game"] for g in first["games"]] == ["r11l"]

        _write_registry(registry_path, ["r11l", "sc25", "bp35"])
        _bump_mtime_forward(registry_path)

        second = arc_primitive_library._load_registry(registry_path)
        assert [g["game"] for g in second["games"]] == ["r11l", "sc25", "bp35"]

    def test_distinct_paths_never_share_a_cache_slot(self, tmp_path):
        """The bug class this guards: a global (path-unaware) cache would let one test's
        tmp_path registry leak into another's. Two DIFFERENT paths, same mtime-instant,
        different content -- each must read its own file."""
        path_a = tmp_path / "a" / "arc_solve_registry.yaml"
        path_a.parent.mkdir()
        path_b = tmp_path / "b" / "arc_solve_registry.yaml"
        path_b.parent.mkdir()
        _write_registry(path_a, ["r11l"])
        _write_registry(path_b, ["sc25"])
        arc_primitive_library._load_registry_cache.pop(path_a, None)
        arc_primitive_library._load_registry_cache.pop(path_b, None)

        a = arc_primitive_library._load_registry(path_a)
        b = arc_primitive_library._load_registry(path_b)
        assert [g["game"] for g in a["games"]] == ["r11l"]
        assert [g["game"] for g in b["games"]] == ["sc25"]

    def test_missing_file_returns_empty_and_is_not_cached_as_a_permanent_miss(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        arc_primitive_library._load_registry_cache.pop(missing, None)
        assert arc_primitive_library._load_registry(missing) == {"games": []}

        # If the file appears AFTER the first (missing-file) call, a later call must see it
        # -- a cache keyed on a successful stat() must not have recorded a false "cached"
        # empty result for a path that raised OSError.
        _write_registry(missing, ["r11l"])
        assert [g["game"] for g in arc_primitive_library._load_registry(missing)["games"]] == [
            "r11l"
        ]

    def test_concurrent_first_access_parses_once_not_once_per_thread(self, tmp_path, monkeypatch):
        # SCENARIO-ARC-REGISTRY-CACHE-1-CONCURRENT-FIRST-ACCESS-PARSES-ONCE
        import threading as _threading

        registry_path = tmp_path / "arc_solve_registry.yaml"
        _write_registry(registry_path, ["r11l"])
        arc_primitive_library._load_registry_cache.pop(registry_path, None)

        n_threads = 8
        parse_count = 0
        parse_count_lock = _threading.Lock()
        real_safe_load = yaml.safe_load

        def _counting_safe_load(*args, **kwargs):
            nonlocal parse_count
            with parse_count_lock:
                parse_count += 1
            return real_safe_load(*args, **kwargs)

        monkeypatch.setattr(arc_primitive_library.yaml, "safe_load", _counting_safe_load)

        barrier = _threading.Barrier(n_threads)
        results: list[dict] = [None] * n_threads  # type: ignore[list-item]

        def _worker(index: int) -> None:
            barrier.wait()
            results[index] = arc_primitive_library._load_registry(registry_path)

        threads = [_threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert parse_count == 1, (
            f"expected exactly one YAML parse across {n_threads} concurrent first-callers, "
            f"got {parse_count}"
        )
        for r in results:
            assert [g["game"] for g in r["games"]] == ["r11l"]
