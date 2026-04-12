# Epic: REPORT-001 - Result Provenance Cleanup and Honest Reporting

**Status:** Completed
**Goal:** Audit every `results/experiment_*_results.json` artifact, label
validated live versus simulated or unverified results, and update the public
README/report/landing page so benchmark claims are presented honestly.
**Rationale:** The repo preserves valuable research history, but several public
documents were summarizing simulated or unverified artifacts as if they were
validated real-world results. Exp 209 cleans that up without deleting any
historical data.

## Stories
- [x] Add a capability spec for result provenance cleanup and reporting disclosure
- [x] Write tests first for result normalization, warning headers, and doc rewrites
- [x] Implement `scripts/experiment_209_cleanup.py`
- [x] Run the cleanup script against the repository artifacts
- [x] Update `README.md`, `docs/technical-report.md`, and `docs/index.html` with provenance-aware claims
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`, and `ops/metrics.md`
- [x] Run the required validation commands, including targeted 100% coverage for the new script
