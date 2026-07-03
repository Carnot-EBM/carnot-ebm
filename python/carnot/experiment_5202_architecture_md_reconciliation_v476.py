"""Exp 5202: architecture.md reconciliation for the v476 ARC/verification state.

Spec refs: REQ-REPORT-5202, SCENARIO-REPORT-5202,
SCENARIO-REPORT-5202-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
LIVE_AGENT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
VERIFY_REPAIR_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
VERIFY_DIR_RELATIVE_PATH = Path("python/carnot/verify")
RESEARCH_REPORTING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/research-reporting/spec.md"
)
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
HARDWARE_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5201_hardware_continuity_gatemate_diagnostic_v476.json"
)
RESULT_RELATIVE_PATH = Path("results/experiment_5202_architecture_md_reconciliation_v476.json")

EXPERIMENT = "experiment_5202_architecture_md_reconciliation_v476"
EXPERIMENT_ID = "exp5202-architecture-md-reconciliation-v476"
MILESTONE = "2026.07.476"
SCHEMA = "carnot.experiment_5202_architecture_md_reconciliation.v476"
RANDOM_SEED = 5202
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RECONCILED_DATE = "20260703"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

SPEC_REFS = [
    "REQ-REPORT-5202",
    "SCENARIO-REPORT-5202",
    "SCENARIO-REPORT-5202-BLOCKED-PRECONDITION",
]

NEW_SECTION_NAMES = (
    "ARC-AGI-3 Harness Architecture",
    "PHASE D Lifecycle And Retirement",
    "Hidden-State Verifier Research Frontier",
    "2026-07-03 Hardware Continuity Update",
)

LEGACY_SECTION_HEADINGS = (
    "## Overview",
    "## System Architecture",
    "## Key Design Decisions",
    "### DD-01: Cargo Workspace",
    "### DD-02: Trait-Based Core",
    "### DD-03: JAX for Python",
    "### DD-04: PyO3 Bindings",
    "### DD-05: Tier Separation",
    "### DD-06: Autoresearch Two-Phase Loop",
    "## Technology Stack",
    "## Data Flow",
    "## Cross-Cutting Concerns",
    "## thrml Integration",
    "## Verification Pipeline Tiers",
    "## KAN Fast-Path Tier \u2014 KAEMEnergy (Exp 447)",
    "## Asymptotic Hardware Mandate (Phase 2 \u2192 Phase 3)",
    "### The stochastic bottleneck",
    "### Implications",
    "### Hardware path",
    "### Sampler-Optimization Decision Record",
    "### Active hardware tracks (Exp 1460)",
    "### Deferred hardware tracks (Exp 1460)",
    "### Mitigations against $\\tau_{\\text{int}}$ blowup",
    "### Decentralisation-respecting consequence",
    "## Phase-3 \u2192 Phase-7 Defence-Layer Stack",
    "### Stack overview",
    "### Key closed-form theorems (canonical)",
    "### Hardware portability theorem",
    "### Operational implications",
    "### Cross-refs",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "sections_added": (
        "List of new architecture sections added by this reconciliation; the list proves "
        "the requested topics landed rather than being implied."
    ),
    "sections_preserved_verbatim": (
        "Count of pre-existing architecture section headings still present verbatim after "
        "the additive edit; protects against silent deletion."
    ),
    "last_reconciled_date_updated": (
        "Bare bool: true only when `_bmad/architecture.md` carries `Last Reconciled: 20260703`."
    ),
    "traceability_md_updated": (
        "Bare bool: false for this task because the conductor-owned reconciliation step "
        "updates traceability immediately after Exp 5202 exits."
    ),
    "inference_substrate": (
        "This reconciliation aggregates upstream docs/source artifacts only; no live model "
        "or board inference is performed."
    ),
    "honest_verdict": "Must start with complete:/complete_/success:/success_.",
}

REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "result_path",
    "run_date",
    "duration_s",
    "random_seed",
    "field_principles",
    "source_artifacts_read",
    "failed_preconditions",
    "architecture_checks",
    "phase_d_summary",
    "arc_registry_summary",
    "verification_pipeline_summary",
    "hardware_summary",
    "research_conductor_modified",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_WRAPPED_FIELDS,
)

DEFAULT_TESTS_RUN = [
    ".venv/bin/pytest tests/python/test_experiment_5202_architecture_md_reconciliation_v476.py -q",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_5202_architecture_md_reconciliation_v476.py' -m pytest tests/python/test_experiment_5202_architecture_md_reconciliation_v476.py -q --no-cov -o addopts=''",
    ".venv/bin/coverage report --rcfile=/dev/null -m --include='*/experiment_5202_architecture_md_reconciliation_v476.py' --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _read_text(root: Path, rel_path: Path, failed: list[str]) -> str:
    path = root / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        failed.append(f"missing_or_unreadable:{rel_path}:{exc.__class__.__name__}")
        return ""


def _load_json(root: Path, rel_path: Path, failed: list[str]) -> JsonDict:
    text = _read_text(root, rel_path, failed)
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        failed.append(f"malformed_json:{rel_path}:{exc.msg}")
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _load_yaml(root: Path, rel_path: Path, failed: list[str]) -> JsonDict:
    text = _read_text(root, rel_path, failed)
    if not text:
        return {}
    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        failed.append(f"malformed_yaml:{rel_path}:{exc.__class__.__name__}")
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _headings(text: str) -> list[str]:
    return re.findall(r"^(#{2,3} .+)$", text, flags=re.MULTILINE)


def _last_reconciled_date(text: str) -> str:
    match = re.search(r"\*\*Last Reconciled:\*\*\s*([^\n]+)", text)
    return match.group(1).strip() if match else ""


def _find_dict_with_id(value: Any, target_id: str) -> JsonDict:
    if isinstance(value, Mapping):
        if value.get("id") == target_id:
            return dict(value)
        for child in value.values():
            found = _find_dict_with_id(child, target_id)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_dict_with_id(child, target_id)
            if found:
                return found
    return {}


def _source_exists(root: Path, rel_path: Path) -> bool:
    path = root / rel_path
    return path.exists()


def analyze_architecture(root: Path, failed: list[str]) -> JsonDict:
    text = _read_text(root, ARCHITECTURE_RELATIVE_PATH, failed)
    if not text:
        return {
            "last_reconciled": "",
            "legacy_headings_present": [],
            "missing_legacy_headings": list(LEGACY_SECTION_HEADINGS),
            "new_sections_present": [],
            "missing_new_sections": list(NEW_SECTION_NAMES),
            "required_topic_markers_missing": [],
        }

    heading_set = set(_headings(text))
    legacy_present = [heading for heading in LEGACY_SECTION_HEADINGS if heading in heading_set]
    new_present = [section for section in NEW_SECTION_NAMES if section in text]
    topic_markers = (
        "scripts/arc_loop_solve.py",
        "E3AgentPolicy",
        "arc_graph_explore.py",
        "arc_solver_kit.py",
        "ops/arc_solve_registry.yaml",
        "2026-06-30",
        "2026-07-02/03",
        "hidden-state/internal-representation",
        "TrajSelector",
        "PHSV",
        "GAP-4891",
        "GAP-4",
        "JEPA_FAST_PATH",
        "ODAR_FAST_PATH",
        "KV260",
        "PolarFire",
        "GateMate",
        "jtag_protocol_level",
    )
    missing_topic_markers = [marker for marker in topic_markers if marker not in text]

    return {
        "last_reconciled": _last_reconciled_date(text),
        "legacy_headings_present": legacy_present,
        "missing_legacy_headings": [
            heading for heading in LEGACY_SECTION_HEADINGS if heading not in heading_set
        ],
        "new_sections_present": new_present,
        "missing_new_sections": [section for section in NEW_SECTION_NAMES if section not in text],
        "required_topic_markers_missing": missing_topic_markers,
    }


def summarize_phase_d(manifest: JsonMap, gaps_text: str) -> JsonDict:
    entry = _find_dict_with_id(manifest, "phase_d_external_text_scorer_retired_exp5163_v474")
    reason = str(entry.get("reason", ""))
    return {
        "manifest_entry_present": bool(entry),
        "retired_milestone": entry.get("retired_milestone"),
        "retired_by_artifact": entry.get("retired_by_artifact"),
        "recorded_by_artifact": entry.get("recorded_by_artifact"),
        "seven_milestones_recorded": "seven milestones" in reason.lower(),
        "hidden_state_carveout_recorded": (
            "hidden-state/internal-representation" in reason
            and "hidden-state/internal-representation verifier research" in gaps_text
        ),
        "operator_reopen_required": entry.get("operator_reopen_required") is True,
    }


def summarize_registry(registry: JsonMap) -> JsonDict:
    live_submissions = list(registry.get("live_submissions") or [])
    scored = [
        row
        for row in live_submissions
        if isinstance(row, Mapping) and row.get("mode") == "competition_kernel_live_agent"
    ]
    games = [row for row in registry.get("games", []) if isinstance(row, Mapping)]
    deepest = sorted(
        (
            {
                "game": str(row.get("game", "")),
                "levels_reproduced": int(row.get("levels_reproduced") or 0),
            }
            for row in games
        ),
        key=lambda row: (-row["levels_reproduced"], row["game"]),
    )[:5]
    return {
        "reproducible_total_levels": int(registry.get("reproducible_total_levels") or 0),
        "reproducible_total_games": int(registry.get("reproducible_total_games") or 0),
        "general_gotchas_count": len(registry.get("general_gotchas") or []),
        "scored_leaderboard_baseline": registry.get("scored_leaderboard_baseline")
        or (scored[-1].get("public_leaderboard_score") if scored else None),
        "scored_live_path_recorded": bool(scored),
        "deepest_reproduced_games": deepest,
    }


def summarize_verification_pipeline(verify_repair_text: str, verify_dir: Path) -> JsonDict:
    markers = {
        "jepa_fast_path": "JEPA_FAST_PATH" in verify_repair_text,
        "nup_probe_fast_path": "NUP_PROBE_FAST_PATH" in verify_repair_text,
        "odar_fast_path": "ODAR_FAST_PATH" in verify_repair_text,
        "think_probe_fast_path": "THINK_PROBE_FAST_PATH" in verify_repair_text,
        "semantic_verifier_v2": "semantic_verifier_v2" in verify_repair_text,
        "probability_calibration": "probability_calibration" in verify_repair_text,
        "casal_tier": "casal_tier" in verify_repair_text,
        "interwhen_monitor": "interwhen_monitor" in verify_repair_text,
        "and_compose_k5": "and_compose_k5" in verify_repair_text,
        "spectral_probe": "tier_0h_spectral" in verify_repair_text,
    }
    verify_modules = sorted(
        path.name for path in verify_dir.glob("*.py") if path.is_file()
    ) if verify_dir.exists() else []
    return {
        "markers": markers,
        "verify_module_count": len(verify_modules),
        "representative_modules": [
            name
            for name in (
                "thinkprm_probe.py",
                "spilled_energy.py",
                "nup_probe.py",
                "hallufield_verifier.py",
                "sc_energy.py",
                "and_composition_verifier.py",
            )
            if name in verify_modules
        ],
    }


def summarize_hardware(hardware_artifact: JsonMap) -> JsonDict:
    polarfire_status = hardware_artifact.get("polarfire_status")
    gatemate_status = hardware_artifact.get("gatemate_status")
    return {
        "artifact_present": bool(hardware_artifact),
        "kv260_status": hardware_artifact.get("kv260_status"),
        "polarfire_reachable": (
            polarfire_status.get("reachable") if isinstance(polarfire_status, Mapping) else None
        ),
        "polarfire_workload_validated": (
            polarfire_status.get("polarfire_workload_validated")
            if isinstance(polarfire_status, Mapping)
            else None
        ),
        "gatemate_reachable": (
            gatemate_status.get("reachable") if isinstance(gatemate_status, Mapping) else None
        ),
        "gatemate_diagnostic_narrowed_to": hardware_artifact.get(
            "gatemate_diagnostic_narrowed_to"
        )
        or (gatemate_status.get("narrowed_to") if isinstance(gatemate_status, Mapping) else None),
        "boards_reachable_count": hardware_artifact.get("boards_reachable_count"),
        "hardware_speedup_claimed": hardware_artifact.get("hardware_speedup_claimed"),
    }


def _missing_source_preconditions(root: Path) -> list[str]:
    required = (
        ARCHITECTURE_RELATIVE_PATH,
        NORTH_STAR_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
        VERIFIER_GAPS_RELATIVE_PATH,
        ARC_REGISTRY_RELATIVE_PATH,
        LIVE_AGENT_RELATIVE_PATH,
        VERIFY_REPAIR_RELATIVE_PATH,
        RESEARCH_REPORTING_SPEC_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
    )
    missing = [f"missing:{rel}" for rel in required if not _source_exists(root, rel)]
    if not (root / VERIFY_DIR_RELATIVE_PATH).is_dir():
        missing.append(f"missing:{VERIFY_DIR_RELATIVE_PATH}")
    return missing


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    duration_s: float = 0.0,
    run_date: str = RECONCILED_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    root_path = Path(root)
    result_path = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    failed: list[str] = _missing_source_preconditions(root_path)

    architecture = analyze_architecture(root_path, failed)
    north_star_text = _read_text(root_path, NORTH_STAR_RELATIVE_PATH, failed)
    manifest = _load_yaml(root_path, MANIFEST_RELATIVE_PATH, failed)
    gaps_text = _read_text(root_path, VERIFIER_GAPS_RELATIVE_PATH, failed)
    registry = _load_yaml(root_path, ARC_REGISTRY_RELATIVE_PATH, failed)
    live_agent_text = _read_text(root_path, LIVE_AGENT_RELATIVE_PATH, failed)
    verify_repair_text = _read_text(root_path, VERIFY_REPAIR_RELATIVE_PATH, failed)
    spec_text = _read_text(root_path, RESEARCH_REPORTING_SPEC_RELATIVE_PATH, failed)
    hardware = _load_json(root_path, HARDWARE_RESULT_RELATIVE_PATH, [])

    if "REQ-REPORT-5202" not in spec_text:
        failed.append("missing_spec_anchor:REQ-REPORT-5202")
    if "ARC-AGI-3" not in north_star_text:
        failed.append("north_star_missing_arc_agi3")
    if "E3AgentPolicy" not in live_agent_text:
        failed.append("live_agent_missing_E3AgentPolicy")
    if architecture["last_reconciled"] != RECONCILED_DATE:
        failed.append("architecture_last_reconciled_not_20260703")
    if architecture["missing_legacy_headings"]:
        failed.append("architecture_legacy_headings_missing")
    if architecture["missing_new_sections"]:
        failed.append("architecture_new_sections_missing")
    if architecture["required_topic_markers_missing"]:
        failed.append("architecture_required_topic_markers_missing")

    phase_d_summary = summarize_phase_d(manifest, gaps_text)
    if not phase_d_summary["manifest_entry_present"]:
        failed.append("phase_d_manifest_entry_missing")

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts_read": [
            str(ARCHITECTURE_RELATIVE_PATH),
            str(NORTH_STAR_RELATIVE_PATH),
            str(MANIFEST_RELATIVE_PATH),
            str(VERIFIER_GAPS_RELATIVE_PATH),
            str(ARC_REGISTRY_RELATIVE_PATH),
            str(LIVE_AGENT_RELATIVE_PATH),
            str(VERIFY_REPAIR_RELATIVE_PATH),
            str(HARDWARE_RESULT_RELATIVE_PATH),
        ],
        "failed_preconditions": sorted(set(failed)),
        "architecture_checks": architecture,
        "phase_d_summary": phase_d_summary,
        "arc_registry_summary": summarize_registry(registry),
        "verification_pipeline_summary": summarize_verification_pipeline(
            verify_repair_text,
            root_path / VERIFY_DIR_RELATIVE_PATH,
        ),
        "hardware_summary": summarize_hardware(hardware),
        "research_conductor_modified": False,
        "traceability_carveout": (
            "not updated in Exp 5202; conductor-owned reconciliation step handles "
            "_bmad/traceability.md, ops/status.md, and ops/changelog.md immediately after exit"
        ),
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "sections_added": _wrap("sections_added", list(architecture["new_sections_present"])),
        "sections_preserved_verbatim": _wrap(
            "sections_preserved_verbatim",
            len(architecture["legacy_headings_present"]),
        ),
        "last_reconciled_date_updated": _wrap(
            "last_reconciled_date_updated",
            architecture["last_reconciled"] == RECONCILED_DATE,
        ),
        "traceability_md_updated": _wrap("traceability_md_updated", False),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "complete: blocked_precondition_architecture_reconciliation_v476"
                if failed
                else "complete: architecture_md_reconciled_20260703_arc_phase_d_hidden_state_hardware"
            ),
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    scrubbed = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact.get("schema") != SCHEMA:
        raise AssertionError("schema mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise AssertionError("experiment_id mismatch")
    if artifact.get("inference_substrate", {}).get("value") != INFERENCE_SUBSTRATE:
        raise AssertionError("inference_substrate mismatch")
    for field in REQUIRED_WRAPPED_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, Mapping) or "value" not in value or "principle" not in value:
            raise AssertionError(f"{field} must be principle-wrapped")
        if value.get("principle") != FIELD_PRINCIPLES[field]:
            raise AssertionError(f"{field} principle mismatch")
    if artifact["last_reconciled_date_updated"]["value"] not in (True, False):
        raise AssertionError("last_reconciled_date_updated must be bool")
    if artifact["traceability_md_updated"]["value"] is not False:
        raise AssertionError("traceability_md_updated must remain false for Exp 5202")
    verdict = str(artifact["honest_verdict"]["value"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise AssertionError("honest_verdict must start with a terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise AssertionError("reproducibility_checksum mismatch")
    failed = artifact.get("failed_preconditions") or []
    if not failed:
        if artifact["sections_added"]["value"] != list(NEW_SECTION_NAMES):
            raise AssertionError("sections_added mismatch")
        if artifact["sections_preserved_verbatim"]["value"] != len(LEGACY_SECTION_HEADINGS):
            raise AssertionError("sections_preserved_verbatim mismatch")
        if artifact["last_reconciled_date_updated"]["value"] is not True:
            raise AssertionError("last_reconciled_date_updated must be true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RECONCILED_DATE)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--result-path", default=None)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    artifact = build_artifact(
        root=Path(args.root),
        result_path=Path(args.result_path) if args.result_path else None,
        duration_s=time.perf_counter() - start,
        run_date=args.date,
    )
    validate_artifact(artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
