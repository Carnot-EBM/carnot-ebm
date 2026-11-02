"""REQ-VERIFY-8723: collection must validate before touching private scratch."""

from pathlib import Path

import pytest

from carnot.reporting import v747_contract_runner as runner


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    """A fixed child stream isolates filesystem ownership from pytest discovery."""
    private, raw = tmp_path / "private", tmp_path / "raw"
    private.mkdir()
    raw.mkdir()
    stream = tmp_path / "nodes"
    suites = runner.m.TESTS + runner.old.CONSUMERS
    nodes = [s + "::test_actual" for s in suites]
    stream.write_text("\n".join(nodes))
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=True, stdout_path=str(stream)))
    return private, raw, stream, nodes


def test_repeated_collection_and_sentinel(inventory):
    """SCENARIO-VERIFY-8723-COLLECTION: no invocation inherits another's files."""
    private, raw, _, _ = inventory
    sentinel = private / "0"
    sentinel.mkdir()
    (sentinel / ".coverage").write_bytes(b"stale coverage must not qualify")
    first, _ = runner.collect(private, raw)
    second, _ = runner.collect(private, raw)
    assert {s["scratch"] for s in first}.isdisjoint(s["scratch"] for s in second)
    assert all(not list(Path(s["scratch"]).glob(".coverage*")) for s in first + second)
    assert (sentinel / ".coverage").read_bytes() == b"stale coverage must not qualify"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "empty", "malformed", "foreign"])
def test_inventory_rejected_without_partial_effects(inventory, mode):
    """REQ-VERIFY-8723: the original second collection must preserve its error."""
    private, raw, stream, nodes = inventory
    contents = {
        "missing": nodes[:1],
        "duplicate": nodes + nodes[:1],
        "empty": [],
        "malformed": nodes[:-1] + [nodes[-1].split("::")[0] + "::"],
        "foreign": nodes + ["tests/python/test_foreign.py::test_stale"],
    }[mode]
    stream.write_text("\n".join(contents))
    before = sorted(p.relative_to(raw) for p in raw.rglob("*"))
    for _ in range(2):
        with pytest.raises(
            ValueError, match="missing_collection_nodes" if mode == "missing" else "collection"
        ):
            runner.collect(private, raw)
        assert not list(private.iterdir())
        assert sorted(p.relative_to(raw) for p in raw.rglob("*")) == before


@pytest.mark.parametrize("where", ["scratch", "config", "inventory"])
def test_collection_collision(inventory, monkeypatch, where):
    """REQ-VERIFY-8723: detecting a collision never overwrites another owner."""
    from types import SimpleNamespace
    from carnot.reporting import collection_contract as c

    private, raw, _, _ = inventory
    monkeypatch.setattr(c, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    target = {
        "scratch": private / "collection-fixed",
        "config": raw / "coverage-fixed-0.ini",
        "inventory": raw / "collection-fixed-inventory.json",
    }[where]
    target.write_bytes(b"sentinel")
    before = sorted(p.relative_to(raw) for p in raw.rglob("*"))
    with pytest.raises(ValueError, match="collection_scratch_collision"):
        runner.collect(private, raw)
    assert target.read_bytes() == b"sentinel"
    assert sorted(p.relative_to(raw) for p in raw.rglob("*")) == before
    if where != "scratch":
        assert not list(private.iterdir())


def test_failed_collection(inventory, monkeypatch):
    """REQ-VERIFY-8723: a nonzero child cannot consume pre-existing coverage."""
    private, raw, stream, _ = inventory
    sentinel = private / ".coverage"
    sentinel.write_bytes(b"unrelated SQLite")
    monkeypatch.setattr(
        runner, "child", lambda *a, **k: dict(passed=False, stdout_path=str(stream))
    )
    with pytest.raises(ValueError, match="single_exact_collection_failed"):
        runner.collect(private, raw)
    assert list(private.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"unrelated SQLite"
    assert not list(raw.iterdir())


def test_original_second_collect(inventory):
    """SCENARIO-VERIFY-8723-COLLECTION: preserve the original missing-node assertion."""
    private, raw, stream, nodes = inventory
    first, _ = runner.collect(private, raw)
    before = sorted(p.relative_to(private) for p in private.rglob("*"))
    configs = {p: p.read_bytes() for p in raw.iterdir() if p.is_file()}
    stream.write_text(nodes[0] + "\n")
    for _ in range(2):
        with pytest.raises(ValueError, match="missing_collection_nodes"):
            runner.collect(private, raw)
    assert sorted(p.relative_to(private) for p in private.rglob("*")) == before
    assert all(p.read_bytes() == data for p, data in configs.items())
    assert all(not list(Path(s["scratch"]).glob(".coverage*")) for s in first)
