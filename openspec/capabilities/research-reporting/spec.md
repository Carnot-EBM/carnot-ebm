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

## Implementation Status

| Requirement | Implementation | Tests | Status |
|------------|----------------|-------|--------|
| REQ-REPORT-001 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-002 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-003 | `scripts/experiment_209_cleanup.py`, `README.md` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-004 | `scripts/experiment_209_cleanup.py`, `docs/technical-report.md`, `docs/index.html` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
