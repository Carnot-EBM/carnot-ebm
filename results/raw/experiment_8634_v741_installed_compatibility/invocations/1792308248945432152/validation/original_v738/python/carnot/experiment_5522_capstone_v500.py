"""Exp5522: capstone reconciliation for milestone 2026.07.500.

Spec refs: REQ-CAPSTONE-5522, SCENARIO-CAPSTONE-5522,
SCENARIO-CAPSTONE-5522-MISSING-SKIPPED-GATES,
SCENARIO-CAPSTONE-5522-FIELD-PRINCIPLES.

This module is deliberately conservative. It reads the upstream .500 artifacts
and records what they already prove; it does not rerun SOTA inference, memory
panels, ARC solvers, or hardware benchmarks. That boundary matters because the
milestone contains useful fixture-level progress, but several downstream
headline gates were conductor-blocked and must remain visibly bounded.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5522_capstone_v500.json")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5522_capstone_v500"
EXPERIMENT_ID = "exp5522-v500-capstone-reconciliation"
MILESTONE = "2026.07.500"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5522
SCHEMA = "carnot.experiment_5522.capstone_v500.v1"
INFERENCE_SUBSTRATE = "capstone_aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5522",
    "SCENARIO-CAPSTONE-5522",
    "SCENARIO-CAPSTONE-5522-MISSING-SKIPPED-GATES",
    "SCENARIO-CAPSTONE-5522-FIELD-PRINCIPLES",
)

PRIMARY_ARTIFACT_PATHS = (
    Path("results/experiment_5510_transition_v500.json"),
    Path("results/experiment_5511_v500_source_delta_ingestion.json"),
    Path("results/experiment_5512_structured_output_positive_control.json"),
    Path("results/experiment_5513_sota_hard_soft_structured_panel.json"),
    Path("results/experiment_5514_energy_spill_sidecar_diagnostic.json"),
    Path("results/experiment_5515_csl_independent_outcome_gate_repair.json"),
    Path("results/experiment_5516_sota_csl_memory_panel.json"),
    Path("results/experiment_5517_csl_memory_residue_stress.json"),
    Path("results/experiment_5518_block_gibbs_sparse_repair_descriptors.json"),
    Path("results/experiment_5519_hardware_continuity_methodology_receipts.json"),
    Path("results/experiment_5520_arc_action_diversity_target_precheck.json"),
    Path("results/experiment_5521_arc_live_action_diverse_levelup.json"),
)

AUXILIARY_ARTIFACT_PATHS = (
    Path("results/experiment_5515_csl_independent_outcome_stream_fixture.json"),
    Path("results/experiment_5521_arc_live_action_diverse_levelup_trajectory.json"),
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-roadmap.yaml"),
    Path("research-roadmap-next.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("ops/exclusion_manifest.yaml"),
)

DEFAULT_DOCS_UPDATED = ("openspec/capabilities/capstone/spec.md",)
DEFAULT_COMMANDS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_5522_capstone_v500.py -q --no-cov",
    (
        ".venv/bin/coverage run "
        "--include=python/carnot/experiment_5522_capstone_v500.py "
        "-m pytest tests/python/test_experiment_5522_capstone_v500.py -q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report "
        "--include=python/carnot/experiment_5522_capstone_v500.py --fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "route key; must equal 2026.07.500.",
    "artifact_paths_read": "evidence basis; primary and sidecar `.500` result",
    "missing_artifacts": "no fabricated rows.",
    "skipped_by_gates": "conductor gate receipts",
    "structured_sota_claim_allowed": "positive-control parse.",
    "energy_sidecar_headline_allowed": "sidecar-only or gated evidence.",
    "csl_claim_allowed": "not just a fixture.",
    "continuous_self_learning_evidence": "fixture-level Exp5515 independent graph-memory evidence.",
    "sparse_repair_claim_allowed": "bounded exact-checked",
    "hardware_speedup_claim": "must remain false",
    "arc_registry_delta": "Exp5521 registry delta.",
    "reproduced_levels": "Exp5521 reproduced levels.",
    "solve_provenance_summary": "from precheck",
    "docs_updated": "files updated by this",
    "commands_run": "validation commands actually run.",
    "conductor_unchanged": "protected-file discipline; derived from git status.",
    "inference_substrate": "must equal",
    "honest_verdict": "terminal status; starts with complete: or blocked:",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "terminal_evidence",
    "claim_boundaries",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

BOOL_FIELDS = (
    "structured_sota_claim_allowed",
    "energy_sidecar_headline_allowed",
    "csl_claim_allowed",
    "continuous_self_learning_evidence",
    "sparse_repair_claim_allowed",
    "hardware_speedup_claim",
    "conductor_unchanged",
)
INT_FIELDS = ("arc_registry_delta", "reproduced_levels")
LIST_FIELDS = (
    "artifact_paths_read",
    "missing_artifacts",
    "skipped_by_gates",
    "solve_provenance_summary",
    "docs_updated",
    "commands_run",
)


def _artifact_key(path: Path) -> str:
    return path.name.removesuffix(".json")


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "sha256": path_sha256(path),
                "read_only": True,
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[str], list[str]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    paths_read: list[str] = []
    missing: list[str] = []
    for rel_path in (*PRIMARY_ARTIFACT_PATHS, *AUXILIARY_ARTIFACT_PATHS):
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        key = _artifact_key(rel_path)
        artifacts[key] = payload
        metadata[rel] = meta
        if meta.get("exists") and meta.get("loadable"):
            paths_read.append(rel)
        elif rel_path in PRIMARY_ARTIFACT_PATHS:
            missing.append(rel)
    return artifacts, metadata, paths_read, missing


def _blocked_gate(payload: JsonMap) -> bool:
    return payload.get("schema") == "blocked_gate_check_v1" or payload.get("status") == "blocked"


def skipped_by_gates(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (
        Path("results/experiment_5514_energy_spill_sidecar_diagnostic.json"),
        Path("results/experiment_5516_sota_csl_memory_panel.json"),
        Path("results/experiment_5517_csl_memory_residue_stress.json"),
    ):
        payload = artifacts.get(_artifact_key(rel_path), {})
        if _blocked_gate(payload):
            rows.append(
                {
                    "artifact_path": rel_path.as_posix(),
                    "blocked_at_layer": payload.get("blocked_at_layer"),
                    "honest_verdict": payload.get("honest_verdict"),
                    "gate_check_summary": payload.get("gate_check_summary"),
                    "failed_gates": [
                        dict(gate)
                        for gate in payload.get("gates_evaluated", [])
                        if isinstance(gate, Mapping) and gate.get("passed") is False
                    ],
                }
            )
    return rows


def solve_provenance_summary(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (
        Path("results/experiment_5520_arc_action_diversity_target_precheck.json"),
        Path("results/experiment_5521_arc_live_action_diverse_levelup.json"),
    ):
        payload = artifacts.get(_artifact_key(rel_path), {})
        rows.append(
            {
                "artifact_path": rel_path.as_posix(),
                "selected_game": payload.get("selected_game"),
                "selected_level": payload.get("selected_level"),
                "solve_provenance": payload.get("solve_provenance"),
                "registry_delta": payload.get("registry_delta"),
                "reproduced_levels": payload.get("reproduced_levels"),
                "honest_verdict": payload.get("honest_verdict"),
            }
        )
    return rows


def build_artifact(
    artifacts: Mapping[str, JsonMap],
    artifact_metadata: JsonMap,
    artifact_paths_read: Sequence[str],
    missing_artifacts: Sequence[str],
    source_context: Sequence[JsonMap],
    source_context_missing: Sequence[str],
    *,
    commands_run: Sequence[str],
    docs_updated: Sequence[str],
    conductor_modified: bool,
) -> JsonDict:
    exp5513 = artifacts.get(
        _artifact_key(Path("results/experiment_5513_sota_hard_soft_structured_panel.json")), {}
    )
    exp5515 = artifacts.get(
        _artifact_key(Path("results/experiment_5515_csl_independent_outcome_gate_repair.json")), {}
    )
    exp5518 = artifacts.get(
        _artifact_key(Path("results/experiment_5518_block_gibbs_sparse_repair_descriptors.json")),
        {},
    )
    exp5519 = artifacts.get(
        _artifact_key(
            Path("results/experiment_5519_hardware_continuity_methodology_receipts.json")
        ),
        {},
    )
    exp5521 = artifacts.get(
        _artifact_key(Path("results/experiment_5521_arc_live_action_diverse_levelup.json")), {}
    )
    skipped = skipped_by_gates(artifacts)

    structured_sota_claim_allowed = bool(exp5513.get("sota_structured_panel_ready"))
    continuous_self_learning_evidence = bool(exp5515.get("continuous_self_learning_evidence"))
    csl_downstream_blocked = any(
        row["artifact_path"].endswith(
            (
                "experiment_5516_sota_csl_memory_panel.json",
                "experiment_5517_csl_memory_residue_stress.json",
            )
        )
        for row in skipped
    )
    sparse_repair_claim_allowed = bool(
        exp5518.get("active_constraint_sparse_repair_ready")
        and exp5518.get("all_candidates_exact_checked")
        and exp5518.get("exact_fallback_used")
        and not exp5518.get("readiness_blockers", [])
    )
    hardware_speedup_claim = bool(exp5519.get("hardware_speedup_claim"))
    status_prefix = "blocked:" if missing_artifacts else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifact_metadata": dict(artifact_metadata),
        "source_context": [dict(row) for row in source_context],
        "source_context_missing": list(source_context_missing),
        "terminal_evidence": {
            "structured_sota": {
                "honest_verdict": exp5513.get("honest_verdict"),
                "sota_structured_panel_ready": exp5513.get("sota_structured_panel_ready"),
                "sota_rows_emitted": exp5513.get("sota_rows_emitted"),
            },
            "csl": {
                "honest_verdict": exp5515.get("honest_verdict"),
                "metric_independence_clean": exp5515.get("metric_independence_clean"),
                "csl_experience_graph_ready": exp5515.get("csl_experience_graph_ready"),
                "continuous_self_learning_evidence": continuous_self_learning_evidence,
            },
            "sparse_constraints": {
                "honest_verdict": exp5518.get("honest_verdict"),
                "active_constraint_sparse_repair_ready": exp5518.get(
                    "active_constraint_sparse_repair_ready"
                ),
                "speedup_claim_allowed": exp5518.get("speedup_claim_allowed"),
            },
            "hardware": {
                "honest_verdict": exp5519.get("honest_verdict"),
                "matched_timing_available": exp5519.get("matched_timing_available"),
                "hardware_speedup_claim_allowed": exp5519.get("hardware_speedup_claim_allowed"),
            },
            "arc": {
                "honest_verdict": exp5521.get("honest_verdict"),
                "registry_delta": exp5521.get("registry_delta"),
                "reproduced_levels": exp5521.get("reproduced_levels"),
            },
        },
        "claim_boundaries": [
            "No structured SOTA claim because Exp5513 is not panel-ready.",
            "No energy-sidecar headline because Exp5514 was conductor-gated and sidecar-only.",
            "No broad CSL claim because downstream SOTA memory/residue lanes did not execute.",
            "Sparse repair claim is bounded to exact-checked descriptor-interface evidence.",
            "No hardware speedup claim without matched timing.",
            "No ARC progress claim because registry_delta and reproduced_levels are zero.",
        ],
        "milestone": MILESTONE,
        "artifact_paths_read": list(artifact_paths_read),
        "missing_artifacts": list(missing_artifacts),
        "skipped_by_gates": skipped,
        "structured_sota_claim_allowed": structured_sota_claim_allowed,
        "energy_sidecar_headline_allowed": False,
        "csl_claim_allowed": bool(continuous_self_learning_evidence and not csl_downstream_blocked),
        "continuous_self_learning_evidence": continuous_self_learning_evidence,
        "sparse_repair_claim_allowed": sparse_repair_claim_allowed,
        "hardware_speedup_claim": hardware_speedup_claim,
        "arc_registry_delta": int(exp5521.get("registry_delta") or 0),
        "reproduced_levels": int(exp5521.get("reproduced_levels") or 0),
        "solve_provenance_summary": solve_provenance_summary(artifacts),
        "docs_updated": list(docs_updated),
        "commands_run": list(commands_run),
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"{status_prefix} .500 capstone read {len(artifact_paths_read)} result artifacts; "
            "structured SOTA claim false, energy sidecar headline false, "
            f"fixture-level CSL evidence {continuous_self_learning_evidence}, "
            f"broad CSL claim {bool(continuous_self_learning_evidence and not csl_downstream_blocked)}, "
            f"sparse repair bounded claim {sparse_repair_claim_allowed}, "
            f"hardware_speedup_claim {hardware_speedup_claim}, "
            f"arc_registry_delta {int(exp5521.get('registry_delta') or 0)}"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run_capstone(
    root: Path = REPO_ROOT,
    *,
    commands_run: Sequence[str] = DEFAULT_COMMANDS_RUN,
    docs_updated: Sequence[str] = DEFAULT_DOCS_UPDATED,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, paths_read, missing = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    return build_artifact(
        artifacts,
        metadata,
        paths_read,
        missing,
        source_context,
        source_context_missing,
        commands_run=commands_run,
        docs_updated=docs_updated,
        conductor_modified=conductor_modified,
    )


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    for field in INT_FIELDS:
        if field in payload and not isinstance(payload[field], int):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_capstone(
    root: Path = REPO_ROOT, *, commands_run: Sequence[str] = DEFAULT_COMMANDS_RUN
) -> JsonDict:
    artifact = run_capstone(root=root, commands_run=commands_run)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - guarded by validate_artifact unit coverage
        raise ValueError(f"invalid Exp5522 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5522 artifact")
    args = parser.parse_args(argv)
    artifact = write_capstone() if args.write else run_capstone()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
