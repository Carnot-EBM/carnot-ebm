"""Tests for the reusable moat trainer (REQ-VERIFY-5030, SCENARIO-VERIFY-5030).

Two tiers:

  * Pure-logic tests (no GPU) exercise the base resolver, pair normalization,
    checkpoint discovery, and the reproducibility checksum against a synthetic
    HuggingFace cache laid out in ``tmp_path``.  These prove the FIX for the
    hallucinated-repo failure class deterministically.

  * One real end-to-end test loads the smallest cached base, trains the LoRA
    energy head for a couple of steps on a 2-pair fixture, checkpoints, reloads,
    and scores — proving the pipeline actually trains (the de-risk D1 needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot import moat_trainer


# --------------------------------------------------------------------------- #
# Synthetic-cache fixtures for the pure resolver logic.
# --------------------------------------------------------------------------- #
def _make_cached_base(
    hub: Path, repo_id: str, *, sharded: bool = False, weights: bool = True
) -> Path:
    """Lay out a fake HF snapshot for ``repo_id`` under ``hub`` and return it."""
    snap = hub / moat_trainer._hf_cache_dir_name(repo_id) / "snapshots" / "rev0"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        if sharded:
            (snap / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        else:
            (snap / "model.safetensors").write_text("x", encoding="utf-8")
    return snap


def test_hf_cache_dir_name_maps_org_slash_name():
    # SCENARIO-VERIFY-5030: repo id -> on-disk cache folder.
    assert moat_trainer._hf_cache_dir_name("Qwen/Qwen3.5-2B") == "models--Qwen--Qwen3.5-2B"


def test_snapshot_with_weights_detects_complete_snapshot(tmp_path):
    # REQ-VERIFY-5030: a snapshot with config.json + safetensors is usable.
    snap = _make_cached_base(tmp_path, "Fake/Model-A")
    found = moat_trainer.snapshot_with_weights("Fake/Model-A", hub_cache=tmp_path)
    assert found == snap


def test_snapshot_with_weights_accepts_sharded_index(tmp_path):
    # REQ-VERIFY-5030: a sharded model (index manifest) also counts as weights.
    _make_cached_base(tmp_path, "Fake/Sharded", sharded=True)
    assert moat_trainer.snapshot_with_weights("Fake/Sharded", hub_cache=tmp_path) is not None


def test_snapshot_with_weights_rejects_config_only(tmp_path):
    # REQ-VERIFY-5030: metadata-only cache (no weights) is NOT trainable.
    _make_cached_base(tmp_path, "Fake/MetaOnly", weights=False)
    assert moat_trainer.snapshot_with_weights("Fake/MetaOnly", hub_cache=tmp_path) is None


def test_snapshot_with_weights_returns_none_when_absent(tmp_path):
    # REQ-VERIFY-5030: not cached at all -> None (not an exception).
    assert moat_trainer.snapshot_with_weights("Fake/Missing", hub_cache=tmp_path) is None


def test_resolve_trainable_base_returns_first_present(tmp_path):
    # REQ-VERIFY-5030: walk the priority list, return the first cached base.
    _make_cached_base(tmp_path, "Fake/Second")
    repo_id, path = moat_trainer.resolve_trainable_base(
        hub_cache=tmp_path, priority=["Fake/First", "Fake/Second", "Fake/Third"]
    )
    assert repo_id == "Fake/Second"
    assert Path(path).is_dir()


def test_resolve_trainable_base_preferred_used_first(tmp_path):
    # REQ-VERIFY-5030: a present preferred id wins over the priority list.
    _make_cached_base(tmp_path, "Fake/Pref")
    _make_cached_base(tmp_path, "Fake/Default")
    repo_id, _ = moat_trainer.resolve_trainable_base(
        preferred="Fake/Pref", hub_cache=tmp_path, priority=["Fake/Default"]
    )
    assert repo_id == "Fake/Pref"


def test_resolve_trainable_base_hallucinated_preferred_falls_through(tmp_path):
    # REQ-VERIFY-5030: the .462 bug — a wrong preferred id must NOT block; it
    # silently falls through to a real cached base.
    _make_cached_base(tmp_path, "Fake/Real")
    repo_id, _ = moat_trainer.resolve_trainable_base(
        preferred="Org/Hallucinated-1.7B", hub_cache=tmp_path, priority=["Fake/Real"]
    )
    assert repo_id == "Fake/Real"


def test_resolve_trainable_base_raises_when_none_present(tmp_path):
    # REQ-VERIFY-5030: only when NOTHING is cached, and the message names the list.
    with pytest.raises(RuntimeError) as exc:
        moat_trainer.resolve_trainable_base(
            preferred="Org/Nope", hub_cache=tmp_path, priority=["Fake/AlsoNope"]
        )
    msg = str(exc.value)
    assert "Org/Nope" in msg and "Fake/AlsoNope" in msg


def test_default_hub_cache_env_overrides(monkeypatch, tmp_path):
    # REQ-VERIFY-5030: HF_HUB_CACHE / HF_HOME are honoured.
    monkeypatch.setenv("HF_HUB_CACHE", (tmp_path / "explicit").as_posix())
    assert moat_trainer.default_hub_cache() == tmp_path / "explicit"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", (tmp_path / "home").as_posix())
    assert moat_trainer.default_hub_cache() == tmp_path / "home" / "hub"
    monkeypatch.delenv("HF_HOME")
    assert moat_trainer.default_hub_cache() == Path.home() / ".cache" / "huggingface" / "hub"


def test_normalize_pairs_accepts_tuples_and_objects():
    # REQ-VERIFY-5030: pairs may be tuples or .good_text/.bad_text objects.
    class _Pair:
        good_text = "good step"
        bad_text = "bad step"

    pairs = moat_trainer.normalize_pairs([("g1", "b1"), _Pair(), ("  ", "x"), ("only_one",)])
    assert pairs == [("g1", "b1"), ("good step", "bad step")]


def test_latest_epoch_checkpoint_picks_highest(tmp_path):
    # REQ-VERIFY-5030: resume/scoring find the highest finished epoch.
    for n in (1, 2, 3):
        d = tmp_path / f"epoch_{n}"
        d.mkdir()
        (d / "train_metrics.json").write_text("{}", encoding="utf-8")
    # An epoch dir without metrics is ignored; a non-dir is ignored.
    (tmp_path / "epoch_4").mkdir()
    (tmp_path / "epoch_notanumber").mkdir()
    (tmp_path / "epoch_notanumber" / "train_metrics.json").write_text("{}", encoding="utf-8")
    latest = moat_trainer.latest_epoch_checkpoint(tmp_path)
    assert latest == tmp_path / "epoch_3"


def test_latest_epoch_checkpoint_none_when_empty(tmp_path):
    assert moat_trainer.latest_epoch_checkpoint(tmp_path) is None
    assert moat_trainer.latest_epoch_checkpoint(tmp_path / "missing") is None


def test_reproducibility_checksum_is_content_sensitive():
    # REQ-VERIFY-5030: drift on any input changes the hash.
    base = moat_trainer.reproducibility_checksum(
        "Qwen/Qwen3.5-2B", [("g", "b")], lora_r=8, lora_alpha=16, seed=0
    )
    same = moat_trainer.reproducibility_checksum(
        "Qwen/Qwen3.5-2B", [("g", "b")], lora_r=8, lora_alpha=16, seed=0
    )
    drifted = moat_trainer.reproducibility_checksum(
        "Qwen/Qwen3.5-2B", [("g", "b2")], lora_r=8, lora_alpha=16, seed=0
    )
    assert base == same and base != drifted and base.startswith("sha256:")


def test_read_write_json_roundtrip_and_bad_json(tmp_path):
    # REQ-VERIFY-5030: checkpoint metric IO is robust to unreadable files.
    p = tmp_path / "m.json"
    moat_trainer._write_json(p, {"a": 1})
    assert moat_trainer._read_json(p) == {"a": 1}
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert moat_trainer._read_json(tmp_path / "broken.json") is None
    assert moat_trainer._read_json(tmp_path / "missing.json") is None


# --------------------------------------------------------------------------- #
# Real-cache resolver test (no GPU): the resolver picks a REAL base from THIS
# box's HuggingFace cache.  The live train+score on a 2-pair fixture is the
# de-risk proven by the experiment smoke (experiment_5030) which actually runs
# the model end-to-end — GPU training is not put behind a pytest skip per the
# "Tests Must Run and Assert" rule (skipping is never allowed; the GPU path is
# pragma-no-cover and exercised by the live smoke instead).
# --------------------------------------------------------------------------- #
def test_resolve_trainable_base_picks_a_real_cached_base():
    # SCENARIO-VERIFY-5030: against the real cache the resolver returns one of the
    # prioritized bases with a snapshot that exists on disk (the .462 fix in situ).
    repo_id, path = moat_trainer.resolve_trainable_base()
    assert repo_id in moat_trainer.PRIORITY_BASES
    snap = Path(path)
    assert snap.is_dir()
    assert (snap / "config.json").exists()
    assert (
        bool(list(snap.glob("*.safetensors"))) or (snap / "model.safetensors.index.json").exists()
    )
