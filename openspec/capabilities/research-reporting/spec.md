# Research Reporting Capability Specification

**Capability:** research-reporting
**Version:** 0.1.0
**Status:** Draft
**Traces to:** NFR-03

## Overview

Defines how Carnot records experiment-result provenance and how public-facing
documentation must distinguish validated live artifacts from simulated or
otherwise unverified results. The goal is to preserve the research record
without presenting exploratory or simulated artifacts as validated real-world
performance.

## Requirements

### REQ-REPORT-001: Result Provenance Audit

The repository shall provide a cleanup workflow that scans
`results/experiment_*_results.json` and determines each artifact's inference
provenance from:

- the top-level `inference_mode`, if present
- otherwise `metadata.inference_mode`
- otherwise `statistics.inference_mode`
- otherwise known nested experiment metadata fields documented by the cleanup
  implementation

The workflow shall normalize the detected provenance into a top-level summary
without deleting any historical result data.

### REQ-REPORT-002: Result Headers

The cleanup workflow shall annotate each scanned result artifact with:

- a human-readable header stating whether the artifact is validated live,
  simulated, or missing explicit live provenance
- a machine-readable provenance summary describing the normalized mode, the
  source field used, and the resulting status

Artifacts with `live_gpu` provenance shall be marked as validated. Artifacts
with `simulated`, `simulation`, or missing provenance shall receive a warning
header rather than being removed.

### REQ-REPORT-003: README Provenance Disclosure

`README.md` shall present key benchmark claims with explicit provenance labels.
Headline result tables shall distinguish:

- validated live results
- simulated results kept for historical record
- results whose inference provenance is missing or otherwise unverified

The README shall add honest caveats where prior headline improvements were not
validated with explicit live inference provenance.

### REQ-REPORT-004: Report and Landing-Page Disclosure

`docs/technical-report.md` and `docs/index.html` shall include an explicit
"Simulation vs Reality" disclosure that:

- summarizes the audit counts for validated live, simulated, and unverified
  artifacts
- marks headline benchmark claims as live, simulated, or unverified
- revises top-level improvement claims so they no longer imply that simulated
  or unverified benchmarks were validated live results

### REQ-REPORT-005: Curated Research Scan Artifact

The repository shall provide a workflow that writes
`results/experiment_210_results.json` for the focused literature scan on
constraint extraction for instruction-tuned models. The artifact shall include:

- the query themes searched
- a ranked list of papers or benchmark assets
- why each item matters to Carnot's current constraint-extraction gap
- concrete proposed experiments for the next milestone

### REQ-REPORT-006: Research References Update

The research-scan workflow shall update `research-references.md` with a dated,
idempotent section containing:

- direct papers on constraint extraction or verification for instruction
  following and chain-of-thought reasoning
- benchmark assets for evaluating fine-grained constraint extraction
- risk evidence on chain-of-thought monitorability when that evidence changes
  the recommended Carnot direction

### REQ-REPORT-007: Research Studying Queue Update

The research-scan workflow shall update `research-studying.md` with a dated,
idempotent section that:

- ranks the new findings by relevance to Carnot's current gap
- records the main takeaways for experiment planning
- proposes the concrete follow-on experiments targeted for 2026-04-15

### REQ-REPORT-008: Idempotent Research Refresh

Re-running the research-scan workflow shall refresh the Exp 210 sections in
`research-references.md`, `research-studying.md`, and
`results/experiment_210_results.json` without duplicating the section body in
the markdown documents.

## Scenarios

### SCENARIO-REPORT-001: Nested Live Provenance Is Promoted

**Given** a result artifact with no top-level `inference_mode`
**And** `metadata.inference_mode` is `live_gpu`
**When** the cleanup workflow runs
**Then** the artifact is marked as a validated live result
**And** the top-level provenance summary records `live_gpu`
**And** the header states that the artifact is validated

### SCENARIO-REPORT-002: Simulated Artifact Receives Warning

**Given** a result artifact whose detected provenance is `simulated` or
`simulation`
**When** the cleanup workflow runs
**Then** the artifact is preserved
**And** a warning header is added
**And** public documentation labels the referenced benchmark as simulated

### SCENARIO-REPORT-003: Missing Provenance Is Disclosed

**Given** a result artifact with no detectable `inference_mode`
**When** the cleanup workflow runs
**Then** the artifact is preserved
**And** a warning header is added
**And** public documentation labels any referenced benchmark as unverified or
missing provenance rather than as a validated live result

### SCENARIO-REPORT-004: Exp 210 Scan Writes the Curated Artifact

**Given** the research docs exist
**When** the Exp 210 research-scan workflow runs
**Then** `results/experiment_210_results.json` is written
**And** it includes ranked papers, benchmark assets, and proposed experiments
**And** the research docs gain a dated Exp 210 section

### SCENARIO-REPORT-005: Exp 210 Rerun Refreshes In Place

**Given** the Exp 210 sections already exist in the research docs
**When** the research-scan workflow runs again
**Then** the Exp 210 section bodies are replaced in place
**And** the docs do not accumulate duplicate Exp 210 blocks

### REQ-PUBLISH-003: HuggingFace README Accuracy Audit

All HuggingFace model READMEs for Phase 1 per-token activation EBMs shall
be audited and updated to:

- Include a Phase 1 disclaimer stating that the models detect model confidence,
  not factual correctness.
- Point users to `pip install carnot` for production use.
- Include a link to the latest full-scale benchmark results (Exp 316) where
  available.

The update operation shall be idempotent: re-running against a README that
already contains the Phase 1 patch shall skip that repo without error.

The carnot-joint-constraint-v1 model card shall include an honest
"RESEARCH PROTOTYPE — weights not published" label when the trained weights
artifact is absent.

### SCENARIO-PUBLISH-005: Phase 1 Patch Is Idempotent

**Given** a model README that already contains the Phase 1 disclaimer section
**When** `model_card_update(repo_id, patch)` is called
**Then** the README is not modified
**And** the repo is recorded in `models_skipped` (not `models_updated`)
**And** no HuggingFace API upload is made for that repo

### SCENARIO-PUBLISH-006: Blocked When Credentials Absent

**Given** HuggingFace credentials are not available (CLI and Python API both fail)
**When** `run_experiment_317()` is called
**Then** the function returns an artifact with `blocked == True`
**And** the artifact includes `exp_317_next_action` with login instructions
**And** `models_updated` is empty
**And** no HuggingFace API calls are made

## Implementation Status

| Requirement | Implementation | Tests | Status |
|------------|----------------|-------|--------|
| REQ-REPORT-001 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-002 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-003 | `scripts/experiment_209_cleanup.py`, `README.md` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-004 | `scripts/experiment_209_cleanup.py`, `docs/technical-report.md`, `docs/index.html` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-005 | `scripts/experiment_210_research_scan.py`, `results/experiment_210_results.json` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-006 | `scripts/experiment_210_research_scan.py`, `research-references.md` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-007 | `scripts/experiment_210_research_scan.py`, `research-studying.md` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-008 | `scripts/experiment_210_research_scan.py` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-PUBLISH-003 | `scripts/experiment_317_hf_publish.py` | `tests/python/test_experiment_317_hf_publish.py` | Implemented |
