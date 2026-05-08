"""Tests for Exp 1563 SpecAnn Phase 3 rejection record.

Spec: REQ-HARNESS-SAMPLER-NO-SPECANN,
SCENARIO-HARNESS-SAMPLER-1, SCENARIO-HARNESS-SAMPLER-2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.specann_rejection_architecture_record import (
    ARCHITECTURE_HEADING,
    EXCLUSION_MARKER,
    REQUIRED_ARTIFACT_FIELDS,
    SPEC_ENTRY_MARKERS,
    _relative_path,
    build_artifact,
    count_spec_entries,
    ensure_architecture_record,
    ensure_exclusion_manifest_entry,
    ensure_spec_entries,
    run,
)


BASE_ARCHITECTURE = """# Carnot - Architecture

## Asymptotic Hardware Mandate (Phase 2 -> Phase 3)

### Hardware path

| Phase | Hardware | Samples/sec | Convergence depth |
|-------|----------|-------------|-------------------|
| 3 | photonic / Ising-machine cluster | >=10^13 | foundation-model regime |

### Active hardware tracks (Exp 1460)

Hardware track table.
"""


BASE_SPEC = """# Research Harnesses Capability Specification

## Requirements

### REQ-HARNESS-014: Generated Test Import Guard

The repository shall provide a lightweight audit.

## Scenarios

### SCENARIO-HARNESS-009: Orphan Generated Test Import Is Blocked

**Given** a generated pytest imports a missing module
**When** the generated-test import guard runs
**Then** the audit fails before pytest collection.

## Implementation Status

| Requirement | Documentation | Artifact |
|-------------|---------------|----------|
| REQ-HARNESS-014 | Implemented (`scripts/audit_orphan_test_imports.py`) | Implemented |
"""


def test_req_harness_sampler_no_specann_builds_complete_artifact() -> None:
    """REQ-HARNESS-SAMPLER-NO-SPECANN: the terminal artifact records the ban."""

    architecture_text, architecture_added = ensure_architecture_record(BASE_ARCHITECTURE)
    spec_text, spec_entries_added = ensure_spec_entries(BASE_SPEC)
    manifest_text, manifest_added = ensure_exclusion_manifest_entry("retired_extras:\n")
    artifact = build_artifact(
        architecture_text=architecture_text,
        spec_text=spec_text,
        manifest_text=manifest_text,
    )

    assert architecture_added is True
    assert spec_entries_added == len(SPEC_ENTRY_MARKERS)
    assert manifest_added is True
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["architecture_record_updated"] is True
    assert artifact["spec_requirements_added"] == 3
    assert artifact["exclusion_manifest_updated"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "unreduced HUBO" in artifact["honest_verdict"]


def test_scenario_harness_sampler_entries_are_idempotent_and_specific() -> None:
    """SCENARIO-HARNESS-SAMPLER-2: future SpecAnn proposals must rebut the record."""

    architecture_text, _ = ensure_architecture_record(BASE_ARCHITECTURE)
    architecture_again, architecture_added_again = ensure_architecture_record(architecture_text)
    spec_text, _ = ensure_spec_entries(BASE_SPEC)
    spec_again, spec_added_again = ensure_spec_entries(spec_text)
    manifest_text, _ = ensure_exclusion_manifest_entry("retired_extras:\n")
    manifest_again, manifest_added_again = ensure_exclusion_manifest_entry(manifest_text)

    assert architecture_again == architecture_text
    assert architecture_added_again is False
    assert architecture_text.count(ARCHITECTURE_HEADING) == 1
    assert "HUBO→QUBO reduction injects gadgets+penalties" in architecture_text
    assert "Gadget-Induced Mean-Field Collapse" in architecture_text

    assert spec_again == spec_text
    assert spec_added_again == 0
    assert count_spec_entries(spec_text) == len(SPEC_ENTRY_MARKERS)
    assert "n≥128" in spec_text
    assert "MUST document why the rejection rationale no longer" in spec_text
    assert "applies" in spec_text

    assert manifest_again == manifest_text
    assert manifest_added_again is False
    assert manifest_text.count(EXCLUSION_MARKER) == 1
    assert "Spectral Annealing" in manifest_text


def test_scenario_harness_sampler_run_writes_record_manifest_and_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HARNESS-SAMPLER-1: direct HUBO path is recorded end to end."""

    architecture_path = tmp_path / "_bmad" / "architecture.md"
    spec_path = tmp_path / "openspec" / "capabilities" / "research-harnesses" / "spec.md"
    manifest_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    out_path = tmp_path / "results" / "experiment_1563_specann_rejection_architecture_record.json"
    architecture_path.parent.mkdir(parents=True)
    spec_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    architecture_path.write_text(BASE_ARCHITECTURE, encoding="utf-8")
    spec_path.write_text(BASE_SPEC, encoding="utf-8")
    manifest_path.write_text("retired_extras:\n", encoding="utf-8")

    artifact = run(
        root=tmp_path,
        architecture_path=architecture_path,
        spec_path=spec_path,
        manifest_path=manifest_path,
        out_path=out_path,
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert ARCHITECTURE_HEADING in architecture_path.read_text(encoding="utf-8")
    assert count_spec_entries(spec_path.read_text(encoding="utf-8")) == 3
    assert EXCLUSION_MARKER in manifest_path.read_text(encoding="utf-8")
    assert _relative_path(architecture_path, root=tmp_path) == "_bmad/architecture.md"


def test_req_harness_sampler_no_specann_requires_all_acceptance_evidence() -> None:
    """REQ-HARNESS-SAMPLER-NO-SPECANN: incomplete evidence is rejected."""

    architecture_text, _ = ensure_architecture_record(BASE_ARCHITECTURE)
    spec_text, _ = ensure_spec_entries(BASE_SPEC)
    manifest_text, _ = ensure_exclusion_manifest_entry("retired_extras:\n")

    with pytest.raises(ValueError, match="architecture record"):
        build_artifact(
            architecture_text=BASE_ARCHITECTURE,
            spec_text=spec_text,
            manifest_text=manifest_text,
        )

    with pytest.raises(ValueError, match="spec"):
        build_artifact(
            architecture_text=architecture_text,
            spec_text=BASE_SPEC,
            manifest_text=manifest_text,
        )

    with pytest.raises(ValueError, match="exclusion manifest"):
        build_artifact(
            architecture_text=architecture_text,
            spec_text=spec_text,
            manifest_text="retired_extras:\n",
        )

    with pytest.raises(ValueError, match="Cannot insert"):
        ensure_architecture_record("# missing active hardware anchor\n")

    with pytest.raises(ValueError, match="implementation table"):
        ensure_spec_entries(
            "\n".join(
                [
                    SPEC_ENTRY_MARKERS[0],
                    SPEC_ENTRY_MARKERS[1],
                    SPEC_ENTRY_MARKERS[2],
                    "## Implementation Status",
                    "",
                ]
            )
        )
    assert SPEC_ENTRY_MARKERS[0] in spec_text

    manifest_without_bucket, _ = ensure_exclusion_manifest_entry("retired:\n")
    assert "retired_extras:" in manifest_without_bucket

    outside_path = Path("/tmp/specann-outside-root.md")
    assert _relative_path(outside_path, root=Path("/tmp/carnot-root")) == outside_path.as_posix()
