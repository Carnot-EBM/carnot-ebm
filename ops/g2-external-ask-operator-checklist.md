# Operator checklist — close gate G2 (one external action)

G2 (independent reproduction) is the SOLE remaining blocker to `paper_ready` (`ops/north-star.md` §2). Autonomous work has verified everything that can be verified without an external run; the terminal step below is reserved for you (Operator-Only External Publication).

## Autonomous-verified preconditions (all green before the ask)

- [x] Self-contained package present: `dist/g2-fover-repro.tar.gz`
- [x] Package sha256 re-verified against the recorded checksum: `521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be`
- [x] Package re-run in an isolated environment (`isolated_dir`) reproduced condition-A AUROC `0.9131` within the published CI `[0.9027, 0.9235]`
- [x] Content-addressed fetch: `QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c`
- [x] One-click workflow committed to the working tree: `.github/workflows/fover-g2-repro.yml`
- [x] Reproducer invite drafted: `docs/g2-reproducer-invite.md`

## TERMINAL STEP — the single external action (operator-only)

1. Review and push the prepared files to the canonical remote (`github.com/Carnot-EBM/carnot-ebm`).
2. **Send the invite** in `docs/g2-reproducer-invite.md` to one non-operator reproducer, **or** open the Actions tab and press **Run workflow** on **"FoVer G2 One-Click Reproduction"** yourself from a non-operator account.
3. When the external/CI run lands condition-A AUROC in `[0.9027, 0.9235]`, record it per `ops/reproduction-runbook-fover-headline.md` and flip G2 to met. Only this confirmed non-operator run closes G2 — autonomous work never does.
