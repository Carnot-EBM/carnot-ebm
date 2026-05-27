# IPFS Distribution — Carnot (Rule 3 Mirror State)

CLAUDE.md "Decentralization-Respecting Design Constraints" Rule 3 mandates
that every published artifact have at least two independent distribution
channels, with the preferred secondary being content-addressed IPFS.
This document tracks where we are against that obligation.

## Mirrored today (operator-authorized 2026-05-27)

All 31 Carnot-EBM models, the 1 dataset, and the 2 Spaces are pinned to
the operator's local IPFS node. See
[`docs/ipfs_mirror_table.md`](ipfs_mirror_table.md) for the full per-repo
CID list and [`results/ipfs_mirrors.json`](../results/ipfs_mirrors.json)
for the machine-readable manifest.

Total: 34 artifacts, 159 files, ~1 GB pinned.

Refresh-on-demand:

```bash
python3 scripts/ipfs_mirror_carnot_ebm.py            # incremental
python3 scripts/ipfs_mirror_carnot_ebm.py --dry-run  # plan only
python3 scripts/ipfs_mirror_carnot_ebm.py --include carnot-thinkprm-v3
```

The script is idempotent — it compares HuggingFace `lastModified`
timestamps against the manifest and only re-mirrors changed repos.

## Reachability — full Rule 3 vs current state

| Channel | State |
|---|---|
| HuggingFace (primary) | live, 31 models + 1 dataset + 2 Spaces |
| IPFS (secondary, local pin) | live as of 2026-05-27, all CIDs reachable while local node is online |
| **Filecoin-backed durable pinning** | **operator action needed** — see below |
| Cloudflare / ipfs.io gateways (low-friction fallback) | usable today: CIDs resolvable via public gateways while the local node is reachable |

The current state satisfies Rule 3 *operationally* — anyone can fetch
any Carnot-EBM artifact via IPFS without depending on HuggingFace. But
if the operator's local node goes offline, the CIDs become unreachable
unless another node has fetched and re-pinned them. For full sovereignty
the next step is durable pinning via a Filecoin-backed service.

## Operator action — durable pinning

Pick one of the following pinning services (any one provides durability;
two provides redundancy):

- **web3.storage** — Filecoin-backed; free tier covers ~5 GB.
- **Storj** — distributed cloud, S3-compatible API plus an IPFS pinning
  service.
- **Filebase** — S3-compatible with IPFS pinning.

Once a token is in hand, pin each CID listed in
`docs/ipfs_mirror_table.md`. For web3.storage:

```bash
# install the w3 CLI once
npm install -g @web3-storage/w3cli
w3 login your@email.com  # follow the email-link flow
w3 space create carnot-ebm-mirror

# pin every CID in the manifest
python3 -c "
import json, subprocess
m = json.load(open('results/ipfs_mirrors.json'))
for repo_id, e in m['entries'].items():
    subprocess.run(['w3', 'space', 'add', e['cid']], check=False)
"
```

After durable pinning, edit this file to record the pinning service and
date, and add a 'durably-pinned' column to the IPFS mirror table.

## PyPI sdist mirroring (Rule 3 for the framework itself)

The 34 entries above cover trained weights, the dataset, and the
interactive Spaces. The `carnot-ebm` PyPI package itself is a separate
artifact governed by Rule 3. Releases are cut by tagging `v<version>`
and pushing; CI publishes to PyPI via OIDC trusted publishing
(feedback_pypi_publish_via_ci_tagged_release.md).

To mirror a release to IPFS after it lands on PyPI:

```bash
VER=0.1.0b1
mkdir -p /tmp/carnot-sdist
pip download --no-deps --no-binary :all: -d /tmp/carnot-sdist carnot-ebm==$VER
CID=$(ipfs add -r -Q --pin /tmp/carnot-sdist)
echo "carnot-ebm-$VER sdist CID: $CID"
# append to results/ipfs_mirrors.json under entries.{pypi-carnot-ebm-$VER}
```

`scripts/ipfs_mirror_carnot_ebm.py` will be extended to handle PyPI
sdist mirroring once a stable release tag exists; the project ships
beta tags only right now.

## Cross-references

- CLAUDE.md "Decentralization-Respecting Design Constraints" Rule 3
- `feedback_ipfs_over_gitea_for_mirror_channel.md` (operator directive
  2026-05-08 selecting IPFS over gitea for the secondary channel)
- `scripts/ipfs_mirror_carnot_ebm.py` (the mirror-automation script)
- `results/ipfs_mirrors.json` (live CID manifest)
- `docs/ipfs_mirror_table.md` (human-readable CID table)
