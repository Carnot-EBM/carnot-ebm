#!/usr/bin/env python3
"""IPFS mirror maintenance for the Carnot-EBM HuggingFace org (Rule 3).

Implements CLAUDE.md "Decentralization-Respecting Design Constraints" Rule 3:
"Distribution mirroring for any published artifact. ... Preferred:
HuggingFace as primary channel + IPFS as secondary channel."

This script:
  1. Enumerates all Carnot-EBM models, datasets, and (optionally) spaces
     from the HuggingFace API.
  2. For each, downloads the repo into a temp dir via
     `huggingface_hub.snapshot_download`.
  3. Runs `ipfs add -r` to add it to the local IPFS daemon and capture
     the CID.
  4. Writes/updates results/ipfs_mirrors.json with per-repo CID,
     HuggingFace last-modified timestamp, file count, and total bytes.
  5. Emits a Markdown table at docs/ipfs_mirror_table.md for use in
     README + model cards.

Idempotency: each entry stores the HuggingFace `lastModified` field. On
re-run, if HF lastModified <= manifest lastModified, the entry is
skipped. This makes the script safe to re-run on a cron / CI cadence.

What this script does NOT do:
  - Set up a Filecoin-backed pinning service (web3.storage / Storj /
    Filebase). That requires operator credentials and account creation.
    Without paid pinning, IPFS reachability depends on the local node
    staying online. Document the limitation; defer the operator action.
  - Update individual HF model cards or pyproject.toml with CIDs. The
    Markdown table can be sourced into those manually by the operator.

Usage:
  python scripts/ipfs_mirror_carnot_ebm.py
  python scripts/ipfs_mirror_carnot_ebm.py --include carnot-thinkprm-v3
  python scripts/ipfs_mirror_carnot_ebm.py --dry-run

Spec refs: CLAUDE.md Decentralization-Respecting Design Constraints
Rule 3 + feedback_ipfs_over_gitea_for_mirror_channel.md memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIRROR_MANIFEST = PROJECT_ROOT / "results" / "ipfs_mirrors.json"
MARKDOWN_TABLE = PROJECT_ROOT / "docs" / "ipfs_mirror_table.md"

HF_API_BASE = "https://huggingface.co/api"
HF_RESOLVE_BASE = "https://huggingface.co"
ORG = "Carnot-EBM"


@dataclass
class MirrorEntry:
    """One mirrored artifact: HF repo + IPFS CID + provenance."""

    repo_id: str
    repo_type: str  # "model", "dataset", or "space"
    cid: str
    hf_last_modified: str
    file_count: int
    total_bytes: int
    ipfs_gateway_url: str = field(default="")
    pinned_at: str = field(default="")

    def to_json(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "cid": self.cid,
            "hf_last_modified": self.hf_last_modified,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "ipfs_gateway_url": self.ipfs_gateway_url
            or f"https://ipfs.io/ipfs/{self.cid}",
            "pinned_at": self.pinned_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def _http_get_json(url: str) -> object:
    """Fetch JSON from a URL with a sensible timeout."""
    with urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted HF endpoint)
        return json.loads(resp.read().decode("utf-8"))


def _check_ipfs_daemon() -> None:
    """Refuse to proceed if the local IPFS daemon is unreachable."""
    try:
        subprocess.run(
            ["ipfs", "id"], check=True, capture_output=True, timeout=10
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(
            "ERROR: local IPFS daemon unreachable. Start with `ipfs daemon` "
            "and retry. Original error:",
            exc,
            file=sys.stderr,
        )
        sys.exit(2)


def _list_hf_repos(repo_type: str) -> list[dict]:
    """List all Carnot-EBM repos of a given type via the HF API."""
    if repo_type == "model":
        url = f"{HF_API_BASE}/models?author={ORG}&limit=200"
    elif repo_type == "dataset":
        url = f"{HF_API_BASE}/datasets?author={ORG}&limit=200"
    elif repo_type == "space":
        url = f"{HF_API_BASE}/spaces?author={ORG}&limit=200"
    else:
        raise ValueError(f"unknown repo_type {repo_type!r}")
    data = _http_get_json(url)
    assert isinstance(data, list)
    return data


def _hf_last_modified(repo_id: str, repo_type: str) -> str:
    """Fetch the most-recent commit timestamp for an HF repo."""
    if repo_type == "model":
        url = f"{HF_API_BASE}/models/{repo_id}"
    elif repo_type == "dataset":
        url = f"{HF_API_BASE}/datasets/{repo_id}"
    elif repo_type == "space":
        url = f"{HF_API_BASE}/spaces/{repo_id}"
    else:
        raise ValueError(f"unknown repo_type {repo_type!r}")
    data = _http_get_json(url)
    assert isinstance(data, dict)
    # HF returns lastModified in some endpoints, sha+createdAt in others.
    return str(data.get("lastModified") or data.get("sha") or "")


def _hf_tree_stats(repo_id: str, repo_type: str) -> tuple[int, int]:
    """Return (file_count, total_bytes) for the repo's main branch tree."""
    if repo_type == "model":
        url = f"{HF_API_BASE}/models/{repo_id}/tree/main"
    elif repo_type == "dataset":
        url = f"{HF_API_BASE}/datasets/{repo_id}/tree/main"
    elif repo_type == "space":
        url = f"{HF_API_BASE}/spaces/{repo_id}/tree/main"
    else:
        raise ValueError(f"unknown repo_type {repo_type!r}")
    data = _http_get_json(url)
    assert isinstance(data, list)
    total = sum(int(f.get("size", 0)) for f in data if isinstance(f, dict))
    return len(data), total


def _snapshot_download(repo_id: str, repo_type: str, dest: Path) -> None:
    """Mirror a HF repo to a local directory via huggingface_hub."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )


def _ipfs_add_dir(path: Path) -> str:
    """Run `ipfs add -r -Q` on a directory and return its root CID."""
    result = subprocess.run(
        ["ipfs", "add", "-r", "-Q", "--pin", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    cid = result.stdout.strip().splitlines()[-1].strip()
    if not cid:
        raise RuntimeError(f"`ipfs add` returned no CID for {path}")
    return cid


def _load_manifest() -> dict:
    """Load the existing ipfs_mirrors.json or return a bootstrap dict."""
    if MIRROR_MANIFEST.exists():
        try:
            return json.loads(MIRROR_MANIFEST.read_text())
        except json.JSONDecodeError:
            print(
                f"WARNING: {MIRROR_MANIFEST} is not valid JSON; starting fresh",
                file=sys.stderr,
            )
    return {"updated_at": "", "entries": {}}


def _save_manifest(manifest: dict) -> None:
    """Write the manifest with a fresh updated_at timestamp."""
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    MIRROR_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _emit_markdown_table(manifest: dict) -> None:
    """Render docs/ipfs_mirror_table.md from the manifest."""
    entries = manifest.get("entries", {})
    rows = sorted(entries.items())
    body = [
        "# Carnot-EBM IPFS Mirror Manifest",
        "",
        "Per CLAUDE.md Rule 3 (distribution mirroring), all Carnot-EBM",
        "HuggingFace artifacts are content-addressed on IPFS as a second,",
        "vendor-independent distribution channel.",
        "",
        "Mirror state and the canonical CIDs are stored in",
        "[`results/ipfs_mirrors.json`](../results/ipfs_mirrors.json) and",
        "regenerated by [`scripts/ipfs_mirror_carnot_ebm.py`](../scripts/ipfs_mirror_carnot_ebm.py).",
        "",
        f"Last updated: {manifest.get('updated_at', '?')}.",
        "",
        "## Fetching an artifact via IPFS",
        "",
        "If you have an IPFS client locally:",
        "",
        "```bash",
        "ipfs get <CID> -o ./carnot-artifact",
        "```",
        "",
        "Without a local IPFS node, use a public gateway:",
        "",
        "```bash",
        "curl -L https://ipfs.io/ipfs/<CID>/ > carnot-artifact.tar.gz",
        "# or, for a single file under the directory:",
        "curl -L https://cloudflare-ipfs.com/ipfs/<CID>/<filename>",
        "```",
        "",
        "## Sovereignty caveat",
        "",
        "CIDs in this manifest are pinned by the operator's local IPFS",
        "node. For full Rule 3 sovereignty, the operator should additionally",
        "pin each CID via at least one Filecoin-backed service",
        "(web3.storage / Storj / Filebase) so the artifacts remain reachable",
        "if the local node goes offline. That step requires operator-side",
        "credentials and is documented in",
        "[`docs/ipfs_anchor_placeholder.md`](ipfs_anchor_placeholder.md).",
        "",
        "## Mirrored artifacts",
        "",
        "| Repo | Type | CID | Files | Size | HF Last-Modified |",
        "|---|---|---|---|---|---|",
    ]
    for repo_id, entry in rows:
        if not isinstance(entry, dict) or "cid" not in entry:
            continue  # skip legacy / partial entries
        cid = entry["cid"]
        rt = entry.get("repo_type", "?")
        nfiles = entry.get("file_count", "?")
        size_mb = entry.get("total_bytes", 0) / 1024 / 1024
        last_mod = (entry.get("hf_last_modified") or "?")[:19]
        body.append(
            f"| `{repo_id}` | {rt} | `{cid}` | {nfiles} | {size_mb:.1f} MB | {last_mod} |"
        )
    body.append("")
    MARKDOWN_TABLE.write_text("\n".join(body))


def _mirror_one(
    repo_id: str,
    repo_type: str,
    manifest: dict,
    dry_run: bool,
) -> Optional[MirrorEntry]:
    """Mirror one repo to IPFS, updating the manifest in place."""
    existing = manifest.get("entries", {}).get(repo_id)
    try:
        last_mod = _hf_last_modified(repo_id, repo_type)
        file_count, total_bytes = _hf_tree_stats(repo_id, repo_type)
    except Exception as exc:  # noqa: BLE001 (network failures are expected)
        print(f"  {repo_id}: cannot fetch HF metadata: {exc}", file=sys.stderr)
        return None
    if (
        existing
        and isinstance(existing, dict)
        and existing.get("hf_last_modified") == last_mod
        and existing.get("cid")
    ):
        print(f"  {repo_id}: up to date (cid={existing['cid'][:12]}...)")
        return None
    if dry_run:
        print(
            f"  {repo_id}: would re-mirror ({file_count} files, "
            f"{total_bytes / 1024 / 1024:.1f} MB; HF lastModified={last_mod})"
        )
        return None
    print(
        f"  {repo_id}: downloading {file_count} files, "
        f"{total_bytes / 1024 / 1024:.1f} MB ..."
    )
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / repo_id.replace("/", "_")
        dest.mkdir()
        try:
            _snapshot_download(repo_id, repo_type, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  {repo_id}: snapshot_download failed: {exc}", file=sys.stderr)
            return None
        try:
            cid = _ipfs_add_dir(dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  {repo_id}: ipfs add failed: {exc}", file=sys.stderr)
            return None
    entry = MirrorEntry(
        repo_id=repo_id,
        repo_type=repo_type,
        cid=cid,
        hf_last_modified=last_mod,
        file_count=file_count,
        total_bytes=total_bytes,
        ipfs_gateway_url=f"https://ipfs.io/ipfs/{cid}",
        pinned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest.setdefault("entries", {})[repo_id] = entry.to_json()
    print(f"  {repo_id}: CID={cid}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        action="append",
        help="Repo basename (without org prefix). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="Do not mirror model repos"
    )
    parser.add_argument(
        "--skip-datasets", action="store_true", help="Do not mirror dataset repos"
    )
    parser.add_argument(
        "--skip-spaces", action="store_true", help="Do not mirror space repos"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be mirrored without downloading or pinning",
    )
    args = parser.parse_args()

    if not args.dry_run:
        _check_ipfs_daemon()

    manifest = _load_manifest()

    targets: list[tuple[str, str]] = []
    if not args.skip_models:
        for repo in _list_hf_repos("model"):
            repo_id = repo.get("id")
            if repo_id and isinstance(repo_id, str):
                if args.include and repo_id.split("/")[-1] not in args.include:
                    continue
                targets.append((repo_id, "model"))
    if not args.skip_datasets:
        for repo in _list_hf_repos("dataset"):
            repo_id = repo.get("id")
            if repo_id and isinstance(repo_id, str):
                if args.include and repo_id.split("/")[-1] not in args.include:
                    continue
                targets.append((repo_id, "dataset"))
    if not args.skip_spaces:
        for repo in _list_hf_repos("space"):
            repo_id = repo.get("id")
            if repo_id and isinstance(repo_id, str):
                if args.include and repo_id.split("/")[-1] not in args.include:
                    continue
                targets.append((repo_id, "space"))

    print(f"plan: {len(targets)} repos under consideration")
    new_or_updated = 0
    for repo_id, repo_type in targets:
        result = _mirror_one(repo_id, repo_type, manifest, args.dry_run)
        if result is not None:
            new_or_updated += 1

    if not args.dry_run:
        _save_manifest(manifest)
        _emit_markdown_table(manifest)
    print(f"done: {new_or_updated} entries added or refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
