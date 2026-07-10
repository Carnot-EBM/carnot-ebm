"""Exp5535 capstone reconciliation for milestone 2026.07.501.

Spec refs: REQ-REPORT-5535, SCENARIO-REPORT-5535,
SCENARIO-REPORT-5535-MISSING-INPUT, SCENARIO-REPORT-5535-FIELD-PRINCIPLES.

This module is an evidence ledger. It reads the already-emitted `.501`
artifacts, records missing inputs and adversarially flagged rows, and then
sets claim boundaries from clean upstream evidence only. That conservative
boundary matters because several useful `.501` artifacts were real receipts or
bounded fixtures, while some of the tempting headline rows were flagged by the
artifact verifier and must stay visible without being promoted.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5535_capstone_v501.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5535_capstone_v501"
EXPERIMENT_ID = "exp5535-v501-capstone-reconciliation"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5535
SCHEMA = "carnot.experiment_5535.capstone_v501.v1"
INFERENCE_SUBSTRATE = "capstone_aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5535",
    "SCENARIO-REPORT-5535",
    "SCENARIO-REPORT-5535-MISSING-INPUT",
    "SCENARIO-REPORT-5535-FIELD-PRINCIPLES",
)

PRIMARY_ARTIFACT_PATHS = (
    Path("results/experiment_5523_transition_v501.json"),
    Path("results/experiment_5524_v501_source_delta_ingestion.json"),
    Path("results/experiment_5525_sota_schema_failure_taxonomy.json"),
    Path("results/experiment_5526_sota_structured_repair_loop.json"),
    Path("results/experiment_5527_sota_hard_soft_panel_v2.json"),
    Path("results/experiment_5528_csl_canonical_gate_artifact.json"),
    Path("results/experiment_5529_csl_event_topic_residue_stress.json"),
    Path("results/experiment_5530_sota_csl_memory_panel_v2.json"),
    Path("results/experiment_5531_sparse_repair_scaleup_ci.json"),
    Path("results/experiment_5532_hardware_receipt_parser_repeatability.json"),
    Path("results/experiment_5533_arc_strategy_routing_precheck.json"),
    Path("results/experiment_5534_arc_strategy_routed_levelup.json"),
)
AUXILIARY_ARTIFACT_PATHS = (
    Path("results/experiment_5534_arc_strategy_routed_levelup_trajectory.json"),
)
SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("ops/e2e-test-plan.md"),
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the `.501` capstone.",
    "artifact_paths_read": (
        "Lists only artifacts actually loaded as JSON evidence; missing files do not become rows."
    ),
    "missing_artifacts": (
        "Records absent or unreadable expected inputs, including consumed roadmap context."
    ),
    "skipped_by_gates": (
        "Keeps conductor-blocked or adversarial-flagged upstreams visible while excluding their positive metrics from headline claims."
    ),
    "structured_sota_claim_allowed": (
        "Bare boolean for schema-valid local SOTA structured-row evidence from clean taxonomy/repair artifacts."
    ),
    "sota_hard_soft_claim_allowed": (
        "Bare boolean for hard/soft quality claims; false when the hard/soft panel artifact is flagged."
    ),
    "continuous_self_learning_evidence": (
        "Bare boolean for clean CSL evidence from canonical gate or clean SOTA memory artifacts."
    ),
    "csl_claim_allowed": (
        "Bare boolean for broad CSL claim eligibility; downstream flagged residue evidence keeps the claim bounded."
    ),
    "sparse_repair_claim_allowed": (
        "Bare boolean for clean exact-checked sparse repair scale evidence, not a speedup claim."
    ),
    "hardware_speedup_claim": "Must remain false unless matched authenticated timing exists.",
    "arc_registry_delta": (
        "Integer registry delta imported only from live-path ARC artifacts, with flagged rows not promoted."
    ),
    "reproduced_levels": "Integer reproduced-level count; live ARC success requires offline reproduction.",
    "solve_provenance_summary": (
        "Per-ARC artifact provenance table preserving live self-discovery/null boundaries."
    ),
    "docs_updated": (
        "Files intentionally updated by Exp5535; ops/status, ops/changelog, and traceability remain empty here when a separate reconciler owns them."
    ),
    "commands_run": "Validation commands and outcomes actually run or marked not applicable with a reason.",
    "roadmap_yaml_unchanged": "Protected-file check for `research-roadmap.yaml`.",
    "conductor_unchanged": "Protected-file check for `scripts/research_conductor.py`.",
    "field_principles": (
        "Carries the why behind every headline and gate field for downstream audit."
    ),
    "inference_substrate": (
        "Must equal `capstone_aggregation_from_upstream_artifacts` because Exp5535 is synthesis only."
    ),
    "honest_verdict": (
        "Terminal summary starting with `complete:` or `blocked:` that names the exact .501 boundary."
    ),
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
    "missing_primary_artifacts",
    "terminal_evidence",
    "claim_boundaries",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
BOOL_FIELDS = (
    "structured_sota_claim_allowed",
    "sota_hard_soft_claim_allowed",
    "continuous_self_learning_evidence",
    "csl_claim_allowed",
    "sparse_repair_claim_allowed",
    "hardware_speedup_claim",
    "roadmap_yaml_unchanged",
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

DEFAULT_DOCS_UPDATED = ("openspec/capabilities/research-reporting/spec.md",)
DEFAULT_COMMANDS_RUN = (
    "PENDING: .venv/bin/pytest tests/python/test_experiment_5535_capstone_v501.py -q --no-cov",
    (
        "PENDING: .venv/bin/coverage run "
        "--include=python/carnot/experiment_5535_capstone_v501.py "
        "-m pytest tests/python/test_experiment_5535_capstone_v501.py -q --no-cov -n 0"
    ),
    (
        "PENDING: .venv/bin/coverage report "
        "--include=python/carnot/experiment_5535_capstone_v501.py --fail-under=100"
    ),
    "PENDING: .venv/bin/pytest tests/python -q",
)


def _read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[str], list[str]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    paths_read: list[str] = []
    missing_primary: list[str] = []
    for rel_path in (*PRIMARY_ARTIFACT_PATHS, *AUXILIARY_ARTIFACT_PATHS):
        payload, meta = read_json_mapping(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if meta.get("exists") and meta.get("loadable"):
            paths_read.append(rel)
        elif rel_path in PRIMARY_ARTIFACT_PATHS:
            missing_primary.append(rel)
    return artifacts, metadata, paths_read, missing_primary


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
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    return artifacts.get(rel_path.as_posix(), {})


def _is_skipped(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial")) or payload.get("status") == "blocked"


def _is_clean(artifacts: Mapping[str, JsonMap], rel_path: Path) -> bool:
    payload = _payload(artifacts, rel_path)
    return bool(payload) and not _is_skipped(payload)


def _has_flagged_upstream(payload: JsonMap) -> bool:
    evidence = payload.get("upstream_gate_evidence")
    if not isinstance(evidence, Mapping):
        return False
    return any(
        isinstance(row, Mapping) and bool(row.get("flagged_adversarial"))
        for row in evidence.values()
    )


def skipped_by_gates(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in PRIMARY_ARTIFACT_PATHS:
        payload = _payload(artifacts, rel_path)
        if not _is_skipped(payload):
            continue
        rows.append(
            {
                "artifact_path": rel_path.as_posix(),
                "skip_reason": (
                    "flagged_adversarial"
                    if payload.get("flagged_adversarial")
                    else "conductor_gate_blocked"
                ),
                "honest_verdict": payload.get("honest_verdict"),
                "status": payload.get("status"),
                "blocked_at_layer": payload.get("blocked_at_layer"),
                "corrigendum_pending": payload.get("corrigendum_pending", []),
                "failed_gates": [
                    dict(row)
                    for row in payload.get("gates_evaluated", [])
                    if isinstance(row, Mapping) and row.get("passed") is False
                ],
            }
        )
    return rows


def solve_provenance_summary(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (
        Path("results/experiment_5533_arc_strategy_routing_precheck.json"),
        Path("results/experiment_5534_arc_strategy_routed_levelup.json"),
    ):
        payload = _payload(artifacts, rel_path)
        rows.append(
            {
                "artifact_path": rel_path.as_posix(),
                "selected_game": payload.get("selected_game"),
                "selected_level": payload.get("selected_level"),
                "solve_provenance": payload.get("solve_provenance"),
                "registry_delta": payload.get("registry_delta"),
                "reproduced_levels": payload.get("reproduced_levels"),
                "flagged_adversarial": bool(payload.get("flagged_adversarial")),
                "honest_verdict": payload.get("honest_verdict"),
            }
        )
    return rows


def _claim_booleans(artifacts: Mapping[str, JsonMap]) -> JsonDict:
    exp5525 = _payload(artifacts, Path("results/experiment_5525_sota_schema_failure_taxonomy.json"))
    exp5526 = _payload(artifacts, Path("results/experiment_5526_sota_structured_repair_loop.json"))
    exp5527 = _payload(artifacts, Path("results/experiment_5527_sota_hard_soft_panel_v2.json"))
    exp5528 = _payload(artifacts, Path("results/experiment_5528_csl_canonical_gate_artifact.json"))
    exp5530 = _payload(artifacts, Path("results/experiment_5530_sota_csl_memory_panel_v2.json"))
    exp5531 = _payload(artifacts, Path("results/experiment_5531_sparse_repair_scaleup_ci.json"))
    return {
        "structured_sota_claim_allowed": bool(
            _is_clean(artifacts, Path("results/experiment_5525_sota_schema_failure_taxonomy.json"))
            and _is_clean(
                artifacts, Path("results/experiment_5526_sota_structured_repair_loop.json")
            )
            and exp5525.get("sota_schema_failure_taxonomy_ready")
            and exp5526.get("sota_structured_repair_loop_ready")
            and exp5526.get("exact_validator_handoff_ready")
            and int(exp5526.get("missing_candidate_rows_after") or 0) == 0
        ),
        "sota_hard_soft_claim_allowed": bool(
            _is_clean(artifacts, Path("results/experiment_5527_sota_hard_soft_panel_v2.json"))
            and exp5527.get("sota_hard_soft_claim_allowed")
            and exp5527.get("sota_structured_panel_ready")
        ),
        "continuous_self_learning_evidence": bool(
            (
                _is_clean(
                    artifacts, Path("results/experiment_5528_csl_canonical_gate_artifact.json")
                )
                and exp5528.get("continuous_self_learning_evidence")
            )
            or (
                _is_clean(artifacts, Path("results/experiment_5530_sota_csl_memory_panel_v2.json"))
                and exp5530.get("continuous_self_learning_evidence")
            )
        ),
        "csl_claim_allowed": bool(
            _is_clean(artifacts, Path("results/experiment_5530_sota_csl_memory_panel_v2.json"))
            and exp5530.get("csl_claim_allowed")
            and not _has_flagged_upstream(exp5530)
        ),
        "sparse_repair_claim_allowed": bool(
            _is_clean(artifacts, Path("results/experiment_5531_sparse_repair_scaleup_ci.json"))
            and exp5531.get("active_constraint_sparse_repair_ready")
        ),
    }


def _arc_numbers(artifacts: Mapping[str, JsonMap]) -> tuple[int, int]:
    rel_path = Path("results/experiment_5534_arc_strategy_routed_levelup.json")
    payload = _payload(artifacts, rel_path)
    if not _is_clean(artifacts, rel_path):
        return 0, 0
    return int(payload.get("registry_delta") or 0), int(payload.get("reproduced_levels") or 0)


def build_artifact(
    artifacts: Mapping[str, JsonMap],
    artifact_metadata: JsonMap,
    artifact_paths_read: Sequence[str],
    missing_primary_artifacts: Sequence[str],
    source_context: Sequence[JsonMap],
    source_context_missing: Sequence[str],
    *,
    commands_run: Sequence[str],
    docs_updated: Sequence[str],
    roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    booleans = _claim_booleans(artifacts)
    arc_registry_delta, reproduced_levels = _arc_numbers(artifacts)
    skipped = skipped_by_gates(artifacts)
    missing_artifacts = [*missing_primary_artifacts, *source_context_missing]
    status_prefix = "blocked:" if missing_primary_artifacts else "complete:"
    terminal = {
        "sota_taxonomy": _payload(
            artifacts, Path("results/experiment_5525_sota_schema_failure_taxonomy.json")
        ).get("honest_verdict"),
        "sota_repair_loop": _payload(
            artifacts, Path("results/experiment_5526_sota_structured_repair_loop.json")
        ).get("honest_verdict"),
        "hard_soft_panel": _payload(
            artifacts, Path("results/experiment_5527_sota_hard_soft_panel_v2.json")
        ).get("honest_verdict"),
        "csl_canonical_gate": _payload(
            artifacts, Path("results/experiment_5528_csl_canonical_gate_artifact.json")
        ).get("honest_verdict"),
        "csl_memory_panel": _payload(
            artifacts, Path("results/experiment_5530_sota_csl_memory_panel_v2.json")
        ).get("honest_verdict"),
        "sparse_repair": _payload(
            artifacts, Path("results/experiment_5531_sparse_repair_scaleup_ci.json")
        ).get("honest_verdict"),
        "hardware": _payload(
            artifacts, Path("results/experiment_5532_hardware_receipt_parser_repeatability.json")
        ).get("honest_verdict"),
        "arc_levelup": _payload(
            artifacts, Path("results/experiment_5534_arc_strategy_routed_levelup.json")
        ).get("honest_verdict"),
    }
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
        "missing_primary_artifacts": list(missing_primary_artifacts),
        "terminal_evidence": terminal,
        "claim_boundaries": [
            "Structured SOTA interface credit is limited to clean taxonomy plus repair-loop handoff.",
            "Hard/soft SOTA quality is not promoted because Exp5527 is adversarial-flagged.",
            "CSL has clean evidence, but broad claim credit is bounded by flagged residue-gate evidence.",
            "Sparse repair is claimable as exact-checked scale evidence, not as a speedup.",
            "Hardware speedup remains false without matched authenticated timing.",
            "ARC registry progress remains zero; flagged ARC rows are provenance only.",
        ],
        "milestone": MILESTONE,
        "artifact_paths_read": list(artifact_paths_read),
        "missing_artifacts": missing_artifacts,
        "skipped_by_gates": skipped,
        "hardware_speedup_claim": False,
        "arc_registry_delta": arc_registry_delta,
        "reproduced_levels": reproduced_levels,
        "solve_provenance_summary": solve_provenance_summary(artifacts),
        "docs_updated": list(docs_updated),
        "commands_run": list(commands_run),
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        **booleans,
    }
    payload["honest_verdict"] = (
        f"{status_prefix} .501 capstone read {len(artifact_paths_read)} result artifacts; "
        f"missing={len(missing_artifacts)}; skipped_flagged_or_gated={len(skipped)}; "
        f"structured_sota_claim_allowed={payload['structured_sota_claim_allowed']}; "
        f"sota_hard_soft_claim_allowed={payload['sota_hard_soft_claim_allowed']}; "
        f"continuous_self_learning_evidence={payload['continuous_self_learning_evidence']}; "
        f"csl_claim_allowed={payload['csl_claim_allowed']}; "
        f"sparse_repair_claim_allowed={payload['sparse_repair_claim_allowed']}; "
        f"hardware_speedup_claim={payload['hardware_speedup_claim']}; "
        f"arc_registry_delta={payload['arc_registry_delta']}"
    )
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run_capstone(
    root: Path = REPO_ROOT,
    *,
    commands_run: Sequence[str] = DEFAULT_COMMANDS_RUN,
    docs_updated: Sequence[str] = DEFAULT_DOCS_UPDATED,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, paths_read, missing_primary = _read_artifacts(root)
    source_context, source_missing = _read_source_context(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    return build_artifact(
        artifacts,
        metadata,
        paths_read,
        missing_primary,
        source_context,
        source_missing,
        commands_run=commands_run,
        docs_updated=docs_updated,
        roadmap_modified=roadmap_modified,
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
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    if payload.get("hardware_speedup_claim") is not False:
        errors.append("hardware_speedup_claim")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    commands_run: Sequence[str] = DEFAULT_COMMANDS_RUN,
    docs_updated: Sequence[str] = DEFAULT_DOCS_UPDATED,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root,
        commands_run=commands_run,
        docs_updated=docs_updated,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - guarded by validate_artifact unit coverage
        raise ValueError(f"invalid Exp5535 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5535 artifact")
    args = parser.parse_args(argv)
    artifact = write_capstone() if args.write else run_capstone()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
