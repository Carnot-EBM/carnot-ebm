# Reproduce the FoVer headline in one command (help close gate G2)

We are looking for **one person who is not the project operator** to independently reproduce the headline result of the Carnot-EBM verifier ensemble. It is **CPU-only**, needs **no GPU, no large model, and no API keys**, and takes a couple of minutes (mostly `pip install`). You do not need any prior knowledge of the project.

**Option A — one-click (no checkout):** on the project's GitHub, open the Actions tab, pick **"FoVer G2 One-Click Reproduction"**, and press **Run workflow**. A green run is the reproduction.

**Option B — self-contained tarball (one command):** download `g2-fover-repro.tar.gz` and run:

```bash
tar xzf g2-fover-repro.tar.gz && cd g2-fover-repro && bash run.sh
```

A zero exit (`echo $?` -> `0`) is the pass: the harness exits non-zero unless condition-A mean AUROC lands in `[0.9027, 0.9235]` and the FR-11 learning contribution lands in `[0.0125, 0.0245]`, over n=1,000 and 5 seeds. **Please report back** the two printed numbers, your platform, and your Python/library versions — that report is what closes G2.

## Integrity

- Package sha256: `521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be`
- IPFS (content-addressed fetch): `ipfs get QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c`
