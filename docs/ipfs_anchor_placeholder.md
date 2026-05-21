# IPFS Distribution Placeholder — Carnot Phase 1

Per CLAUDE.md Rule 3 (distribution mirroring), the carnot-ebm source distribution
will be pinned to IPFS after the PyPI release is confirmed.

## Steps for operator:
1. After PyPI publish (via CI): download carnot-ebm-{version}.tar.gz from PyPI
2. `ipfs add carnot-ebm-{version}.tar.gz` → note the CID
3. Pin via Filecoin-backed service (web3.storage / Storj / Filebase)
4. Update README.md IPFS section with CID
5. Update HuggingFace model card with CID

## CID: [TO BE FILLED BY OPERATOR AFTER IPFS PIN]