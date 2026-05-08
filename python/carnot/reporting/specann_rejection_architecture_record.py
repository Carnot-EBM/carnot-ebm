"""Write the Exp 1563 SpecAnn rejection architecture-record artifact.

Spec: REQ-HARNESS-SAMPLER-NO-SPECANN,
SCENARIO-HARNESS-SAMPLER-1, SCENARIO-HARNESS-SAMPLER-2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "2026-05-08"
SCHEMA = "specann_rejection_architecture_record_v1"
EXPERIMENT_ID = "1563"

DEFAULT_ARCHITECTURE_PATH = REPO_ROOT / "_bmad" / "architecture.md"
DEFAULT_SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "research-harnesses" / "spec.md"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ops" / "exclusion_manifest.yaml"
DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1563_specann_rejection_architecture_record.json"
)

ARCHITECTURE_HEADING = "### Sampler-Optimization Decision Record"
ARCHITECTURE_ANCHOR = "\n### Active hardware tracks"
ARCHITECTURE_BLOCK = f"""{ARCHITECTURE_HEADING}

SpecAnn rejected for Phase 3 inference-time argmin. Rationale: (a)
HUBO→QUBO reduction injects gadgets+penalties that fracture SpecAnn's spectral
homotopy path; (b) phase-transition level-crossings during training force
catastrophic cold-restarts; (c) three-paper composition
(SpecAnn+BRAIN+MCMC Layers) triggers Gadget-Induced Mean-Field Collapse (Deep
Think DT-COMPOSITION (f), 2026-05-08). Carnot retains existing
Gibbs-heuristic argmin on unreduced HUBO energy.
"""

REQ_MARKER = "REQ-HARNESS-SAMPLER-NO-SPECANN"
SCENARIO_1_MARKER = "SCENARIO-HARNESS-SAMPLER-1"
SCENARIO_2_MARKER = "SCENARIO-HARNESS-SAMPLER-2"
SPEC_ENTRY_MARKERS = (REQ_MARKER, SCENARIO_1_MARKER, SCENARIO_2_MARKER)

REQ_BLOCK = f"""### {REQ_MARKER}: Phase 3 SpecAnn Ban

Phase 3 substrate sampler MUST NOT use Spectral Annealing (Deep Think
DT-COMPOSITION 2026-05-08).
"""

SCENARIO_1_BLOCK = f"""### {SCENARIO_1_MARKER}: Direct HUBO Evaluation At Production Scale

**Given** a Phase 3 substrate sampler evaluates an unreduced HUBO energy
**And** the production-scale problem has n≥128 variables
**When** the sampler performs inference-time argmin
**Then** HUBO direct evaluation MUST succeed without QUBO reduction at
production scale (n≥128).
"""

SCENARIO_2_BLOCK = f"""### {SCENARIO_2_MARKER}: Future SpecAnn Proposals Rebut Rejection

**Given** a future Phase 3 planning proposal recommends SpecAnn or Spectral
Annealing
**When** it enters the research harness
**Then** the proposal MUST document why the rejection rationale no longer
applies.
"""

STATUS_ROW = (
    "| REQ-HARNESS-SAMPLER-NO-SPECANN | Implemented (`_bmad/architecture.md`, "
    "`ops/exclusion_manifest.yaml`) | Implemented "
    "(`results/experiment_1563_specann_rejection_architecture_record.json`) |"
)
STATUS_TABLE_HEADER = (
    "| Requirement | Documentation | Artifact |\n|-------------|---------------|----------|\n"
)

EXCLUSION_MARKER = "specann_phase3_sampler_optimization_rejected"
EXCLUSION_BLOCK = f"""  # Added by Exp 1563 - SpecAnn Phase 3 sampler rejection.
  - id: {EXCLUSION_MARKER}
    reason: |
      Spectral Annealing is retired for Phase 3 inference-time argmin after
      Deep Think DT-COMPOSITION (e)/(f): HUBO→QUBO reductions inject gadgets
      and penalties that fracture the spectral homotopy path, phase-transition
      level-crossings force cold-restarts, and SpecAnn+BRAIN+MCMC Layers
      triggers Gadget-Induced Mean-Field Collapse. Carnot retains the existing
      Gibbs-heuristic argmin on unreduced HUBO energy.
    experiment_ids:
      - exp1563
    blocked_patterns:
      - Spectral Annealing
      - SpecAnn
      - SpecAnn+BRAIN+MCMC Layers
      - HUBO→QUBO Spectral Annealing
    retired_milestone: "2026.05.120"
    retired_by_artifact: "results/experiment_1563_specann_rejection_architecture_record.json"
    operator_reopen_required: true
    retire_if_same_verdict: true
"""

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "schema",
    "experiment_id",
    "run_date",
    "architecture_record_updated",
    "spec_requirements_added",
    "exclusion_manifest_updated",
    "honest_verdict",
}


def _relative_path(path: Path, *, root: Path = REPO_ROOT) -> str:
    """Return a stable repository-relative path for artifact metadata."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _insert_before(text: str, anchor: str, block: str) -> str:
    if anchor not in text:
        raise ValueError(f"Cannot insert block because anchor is missing: {anchor.strip()}")
    return text.replace(anchor, f"\n{block.rstrip()}\n{anchor}", 1)


def ensure_architecture_record(text: str) -> tuple[str, bool]:
    """Ensure the Phase 3 substrate architecture carries the SpecAnn rejection."""

    if ARCHITECTURE_HEADING in text:
        return text, False
    return _insert_before(text, ARCHITECTURE_ANCHOR, ARCHITECTURE_BLOCK), True


def count_spec_entries(text: str) -> int:
    """Count the required REQ/SCENARIO markers present in the harness spec."""

    return sum(1 for marker in SPEC_ENTRY_MARKERS if marker in text)


def ensure_spec_entries(text: str) -> tuple[str, int]:
    """Ensure the harness spec carries the SpecAnn ban and review scenarios."""

    updated = text
    entries_added = 0
    for marker, block, anchor in (
        (REQ_MARKER, REQ_BLOCK, "\n## Scenarios"),
        (SCENARIO_1_MARKER, SCENARIO_1_BLOCK, "\n## Implementation Status"),
        (SCENARIO_2_MARKER, SCENARIO_2_BLOCK, "\n## Implementation Status"),
    ):
        if marker not in updated:
            updated = _insert_before(updated, anchor, block)
            entries_added += 1

    if STATUS_ROW not in updated:
        if STATUS_TABLE_HEADER not in updated:
            raise ValueError("Cannot insert status row because implementation table is missing")
        updated = updated.replace(
            STATUS_TABLE_HEADER,
            f"{STATUS_TABLE_HEADER}{STATUS_ROW}\n",
            1,
        )

    return updated, entries_added


def ensure_exclusion_manifest_entry(text: str) -> tuple[str, bool]:
    """Ensure the conductor exclusion manifest retires SpecAnn proposals."""

    if EXCLUSION_MARKER in text:
        return text, False
    updated = text if text.strip() else "retired_extras:\n"
    if "retired_extras:" not in updated:
        updated = f"{updated.rstrip()}\n\nretired_extras:\n"
    separator = "\n" if updated.endswith("\n") else "\n\n"
    return f"{updated.rstrip()}{separator}{EXCLUSION_BLOCK.rstrip()}\n", True


def build_artifact(
    *,
    architecture_text: str,
    spec_text: str,
    manifest_text: str,
) -> dict[str, Any]:
    """Build the terminal artifact after validating the acceptance evidence."""

    architecture_record_updated = ARCHITECTURE_HEADING in architecture_text
    spec_requirements_added = count_spec_entries(spec_text)
    exclusion_manifest_updated = EXCLUSION_MARKER in manifest_text

    if not architecture_record_updated:
        raise ValueError("architecture record is missing the SpecAnn rejection subsection")
    if spec_requirements_added != len(SPEC_ENTRY_MARKERS):
        raise ValueError("spec is missing the SpecAnn REQ/SCENARIO entries")
    if not exclusion_manifest_updated:
        raise ValueError("exclusion manifest is missing the SpecAnn retirement entry")

    return {
        "status": "complete",
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "architecture_record_updated": architecture_record_updated,
        "spec_requirements_added": spec_requirements_added,
        "exclusion_manifest_updated": exclusion_manifest_updated,
        "honest_verdict": (
            "complete: SpecAnn rejected for Phase 3 inference-time argmin; "
            "Carnot retains Gibbs-heuristic argmin on unreduced HUBO energy"
        ),
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path = REPO_ROOT,
    architecture_path: Path | None = None,
    spec_path: Path | None = None,
    manifest_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the record updates and write the terminal Exp 1563 JSON artifact."""

    architecture_path = architecture_path or root / "_bmad" / "architecture.md"
    spec_path = spec_path or root / "openspec" / "capabilities" / "research-harnesses" / "spec.md"
    manifest_path = manifest_path or root / "ops" / "exclusion_manifest.yaml"
    out_path = out_path or root / "results" / DEFAULT_OUT_PATH.name

    architecture_text, _ = ensure_architecture_record(architecture_path.read_text(encoding="utf-8"))
    spec_text, _ = ensure_spec_entries(spec_path.read_text(encoding="utf-8"))
    manifest_text, _ = ensure_exclusion_manifest_entry(manifest_path.read_text(encoding="utf-8"))

    _write_text(architecture_path, architecture_text)
    _write_text(spec_path, spec_text)
    _write_text(manifest_path, manifest_text)

    artifact = build_artifact(
        architecture_text=architecture_text,
        spec_text=spec_text,
        manifest_text=manifest_text,
    )
    artifact["paths"] = {
        "architecture": _relative_path(architecture_path, root=root),
        "spec": _relative_path(spec_path, root=root),
        "exclusion_manifest": _relative_path(manifest_path, root=root),
    }
    _write_json(out_path, artifact)
    return artifact
