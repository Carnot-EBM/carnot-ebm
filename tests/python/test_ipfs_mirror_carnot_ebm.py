"""Tests for scripts/ipfs_mirror_carnot_ebm.py.

Covers the pure-Python paths that don't require a live IPFS daemon or
HuggingFace network access:
  - MirrorEntry JSON serialization
  - manifest load/save round-trip
  - Markdown table emission from a synthetic manifest

The network-touching paths (_list_hf_repos, _hf_last_modified, _hf_tree_stats,
_snapshot_download, _ipfs_add_dir) are exercised end-to-end by running the
script in dry-run mode against the live HF API in CI; they are not unit-
tested here because they have no semantic surface beyond "the HTTP call
succeeded."

Spec coverage: CLAUDE.md "Decentralization-Respecting Design Constraints"
Rule 3 (distribution mirroring via IPFS as secondary channel) +
feedback_ipfs_over_gitea_for_mirror_channel.md.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_module():
    """Load scripts/ipfs_mirror_carnot_ebm.py without executing main()."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "ipfs_mirror_carnot_ebm.py"
    spec = importlib.util.spec_from_file_location(
        "ipfs_mirror_carnot_ebm", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ipfs_mirror_carnot_ebm"] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module()


class TestMirrorEntry:
    """MirrorEntry is the per-repo record written to ipfs_mirrors.json."""

    def test_to_json_includes_all_required_fields(self) -> None:
        entry = _MOD.MirrorEntry(
            repo_id="Carnot-EBM/carnot-thinkprm-v3",
            repo_type="model",
            cid="Qmf3oPRWBiq5mJAwounw2wWLKxoMHxRitd8PGHmvvGntbn",
            hf_last_modified="2026-05-26T19:27:52.000Z",
            file_count=3,
            total_bytes=80169,
        )
        d = entry.to_json()
        for required in (
            "repo_id",
            "repo_type",
            "cid",
            "hf_last_modified",
            "file_count",
            "total_bytes",
            "ipfs_gateway_url",
            "pinned_at",
        ):
            assert required in d, f"missing required field {required}"

    def test_gateway_url_defaults_to_ipfs_io(self) -> None:
        entry = _MOD.MirrorEntry(
            repo_id="Carnot-EBM/x",
            repo_type="model",
            cid="QmABC",
            hf_last_modified="2026-05-27T00:00:00.000Z",
            file_count=1,
            total_bytes=100,
        )
        d = entry.to_json()
        assert d["ipfs_gateway_url"] == "https://ipfs.io/ipfs/QmABC"

    def test_pinned_at_defaults_to_now(self) -> None:
        entry = _MOD.MirrorEntry(
            repo_id="Carnot-EBM/x",
            repo_type="model",
            cid="QmABC",
            hf_last_modified="2026-05-27T00:00:00.000Z",
            file_count=1,
            total_bytes=100,
        )
        d = entry.to_json()
        # parse the ISO-8601 string
        parsed = datetime.fromisoformat(d["pinned_at"])
        # accept any value within the last hour
        now = datetime.now(timezone.utc)
        delta = (now - parsed).total_seconds()
        assert -5 < delta < 3600, f"pinned_at not 'now-ish': {d['pinned_at']}"


class TestManifestRoundTrip:
    """Manifest load/save preserves structure."""

    def test_load_missing_file_returns_bootstrap_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_MOD, "MIRROR_MANIFEST", tmp_path / "nonexistent.json")
        m = _MOD._load_manifest()
        assert m == {"updated_at": "", "entries": {}}

    def test_load_then_save_preserves_entries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mf = tmp_path / "manifest.json"
        seed = {
            "updated_at": "2026-05-26T00:00:00+00:00",
            "entries": {
                "Carnot-EBM/x": {
                    "repo_id": "Carnot-EBM/x",
                    "repo_type": "model",
                    "cid": "QmABC",
                    "hf_last_modified": "2026-05-26T00:00:00.000Z",
                    "file_count": 1,
                    "total_bytes": 100,
                    "ipfs_gateway_url": "https://ipfs.io/ipfs/QmABC",
                    "pinned_at": "2026-05-26T00:00:00+00:00",
                }
            },
        }
        mf.write_text(json.dumps(seed))
        monkeypatch.setattr(_MOD, "MIRROR_MANIFEST", mf)
        loaded = _MOD._load_manifest()
        assert loaded["entries"]["Carnot-EBM/x"]["cid"] == "QmABC"
        _MOD._save_manifest(loaded)
        reloaded = json.loads(mf.read_text())
        # _save_manifest stamps a fresh updated_at; entries unchanged
        assert reloaded["entries"]["Carnot-EBM/x"]["cid"] == "QmABC"
        assert reloaded["updated_at"] != seed["updated_at"]


class TestMarkdownTableEmission:
    """The rendered table at docs/ipfs_mirror_table.md is human-readable."""

    def test_table_contains_repo_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        md = tmp_path / "ipfs_mirror_table.md"
        monkeypatch.setattr(_MOD, "MARKDOWN_TABLE", md)
        manifest = {
            "updated_at": "2026-05-27T15:39:15+00:00",
            "entries": {
                "Carnot-EBM/carnot-thinkprm-v3": {
                    "repo_id": "Carnot-EBM/carnot-thinkprm-v3",
                    "repo_type": "model",
                    "cid": "Qmf3oPRWBiq5mJAwounw2wWLKxoMHxRitd8PGHmvvGntbn",
                    "hf_last_modified": "2026-05-26T19:27:52.000Z",
                    "file_count": 3,
                    "total_bytes": 80169,
                },
                "Carnot-EBM/token-activations": {
                    "repo_id": "Carnot-EBM/token-activations",
                    "repo_type": "dataset",
                    "cid": "QmWe1LvmGFG7ja2LeGjUUxSxCeC8JBE8QgwYA9cd4VmXCB",
                    "hf_last_modified": "2026-05-26T19:27:01.000Z",
                    "file_count": 12,
                    "total_bytes": 968826880,
                },
            },
        }
        _MOD._emit_markdown_table(manifest)
        rendered = md.read_text()
        assert "Carnot-EBM IPFS Mirror Manifest" in rendered
        assert "Carnot-EBM/carnot-thinkprm-v3" in rendered
        assert "Carnot-EBM/token-activations" in rendered
        assert "Qmf3oPRWBiq5mJAwounw2wWLKxoMHxRitd8PGHmvvGntbn" in rendered
        # Size column should render in MB
        assert "MB" in rendered

    def test_table_skips_malformed_entries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Entries missing 'cid' (e.g., legacy partial records) are dropped."""
        md = tmp_path / "ipfs_mirror_table.md"
        monkeypatch.setattr(_MOD, "MARKDOWN_TABLE", md)
        manifest = {
            "updated_at": "now",
            "entries": {
                "Carnot-EBM/legacy-broken": {"some_other_field": "value"},
                "Carnot-EBM/good": {
                    "repo_id": "Carnot-EBM/good",
                    "repo_type": "model",
                    "cid": "QmGood",
                    "hf_last_modified": "2026-05-27T00:00:00.000Z",
                    "file_count": 1,
                    "total_bytes": 100,
                },
            },
        }
        _MOD._emit_markdown_table(manifest)
        rendered = md.read_text()
        assert "legacy-broken" not in rendered
        assert "QmGood" in rendered

    def test_table_handles_empty_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        md = tmp_path / "ipfs_mirror_table.md"
        monkeypatch.setattr(_MOD, "MARKDOWN_TABLE", md)
        _MOD._emit_markdown_table({"updated_at": "now", "entries": {}})
        rendered = md.read_text()
        assert "Carnot-EBM IPFS Mirror Manifest" in rendered
        # No data rows, just the header row
        data_rows = [
            l
            for l in rendered.splitlines()
            if l.startswith("|") and "Carnot-EBM" in l
        ]
        assert data_rows == []
