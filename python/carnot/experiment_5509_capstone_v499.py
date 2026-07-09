"""Exp5509 capstone synthesis for milestone 2026.07.499.

Spec refs: REQ-REPORT-5509, SCENARIO-REPORT-5509,
SCENARIO-REPORT-5509-MISSING-INPUT.

This module is an evidence ledger, not a new science run. It reads the actual
Exp5496 through Exp5508 artifacts, keeps missing inputs visible, and converts
the milestone into a compact verdict table. That boundary matters because this
milestone contains several useful but claim-limited results: exact hard/soft
fixtures landed, the local SOTA panel mostly abstained, CSL memory replay
improved a cached replay fixture while the headline SOTA CSL gate stayed
blocked, hardware produced receipts but no timing-speedup evidence, and ARC
ran a changed live path without banking a new level.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5509_capstone_v499.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

EXPERIMENT = "experiment_5509_capstone_v499"
EXPERIMENT_ID = "exp5509-capstone-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5509
SCHEMA = "carnot.experiment_5509.capstone_v499.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = (
    "REQ-REPORT-5509",
    "SCENARIO-REPORT-5509",
    "SCENARIO-REPORT-5509-MISSING-INPUT",
)

EXPECTED_ARTIFACTS: dict[int, Path] = {
    5496: Path("results/experiment_5496_transition_v499.json"),
    5497: Path("results/experiment_5497_pretest_cascade_diagnostic_v499.json"),
    5498: Path("results/experiment_5498_source_delta_v499.json"),
    5499: Path("results/experiment_5499_preference_maxsat_minimal_fixture_v499.json"),
    5500: Path("results/experiment_5500_sota_concept_claim_panel_v499.json"),
    5501: Path("results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json"),
    5502: Path("results/experiment_5502_csl_tautology_static_corrigendum_v499.json"),
    5503: Path("results/experiment_5503_csl_experience_graph_replay_v499.json"),
    5504: Path("results/experiment_5504_sota_csl_memory_panel_v499.json"),
    5505: Path("results/experiment_5505_active_constraint_milp_descriptor_v499.json"),
    5506: Path("results/experiment_5506_hardware_multiboard_receipts_v499.json"),
    5507: Path("results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json"),
    5508: Path("results/experiment_5508_arc_live_perception_generation_levelup_v499.json"),
}

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "Route key for the .499 capstone.",
    "artifacts_expected": "Primary Exp5496-Exp5508 artifact paths required by the roadmap task.",
    "artifacts_found": "Primary expected artifacts actually read as JSON evidence.",
    "artifacts_missing": "Primary expected artifacts absent or unreadable; missing never counts as success.",
    "pretest_cascade_resolved": "Bare boolean imported from Exp5497, preserving full-suite caveats in supporting fields.",
    "hard_soft_core_verdict": "Evidence boundary for Exp5499, Exp5500, Exp5501, and Exp5505.",
    "csl_verdict": "Evidence boundary for Exp5502, Exp5503, and Exp5504.",
    "hardware_verdict": "Board-status and receipt boundary from Exp5506.",
    "arc_verdict": "Registry/provenance/methodology boundary from Exp5507 and Exp5508.",
    "arc_registry_delta": "Bare integer from Exp5508 registry before/after totals or artifact delta.",
    "hardware_speedup_claim": "False unless authenticated matched board timing exists.",
    "guided_decoding_quarantine_status": "Token-steering and guided-decoding quarantine boundary.",
    "prd_gap_table": "PRD requirement gaps grounded in upstream artifact fields.",
    "next_recommendations": "Next experiments or retirements grounded in observed evidence.",
    "roadmap_yaml_unchanged": "Protected-file check for research-roadmap.yaml.",
    "conductor_unchanged": "Protected-file check for scripts/research_conductor.py.",
    "inference_substrate": "Aggregation only; no hidden live inference, solver, or hardware run.",
    "honest_verdict": "Terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5509_capstone_v499.py -q --no-cov",
        "outcome": "expected",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/experiment_5509_capstone_v499.py "
            "-m pytest tests/python/test_experiment_5509_capstone_v499.py -q --no-cov -n 0"
        ),
        "outcome": "expected",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5509_capstone_v499.py --fail-under=100"
        ),
        "outcome": "expected",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "expected"},
)


def _artifact_path(exp_id: int) -> str:
    return EXPECTED_ARTIFACTS[exp_id].as_posix()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_artifacts(root: Path) -> tuple[dict[int, JsonDict], list[str], list[str], dict[str, JsonDict]]:
    artifacts: dict[int, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    metadata: dict[str, JsonDict] = {}
    for exp_id, rel_path in EXPECTED_ARTIFACTS.items():
        payload, meta = read_json_mapping(root / rel_path)
        artifacts[exp_id] = payload
        rel = rel_path.as_posix()
        metadata[rel] = meta
        target = found if meta["exists"] and meta["loadable"] else missing
        target.append(rel)
    return artifacts, found, missing, metadata


def _source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        exists = (root / rel_path).exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(root / rel_path),
            }
        )
        missing.extend([] if exists else [rel_path.as_posix()])
    return records, missing


def _discover_sidecars(root: Path) -> list[str]:
    expected = {path.as_posix() for path in EXPECTED_ARTIFACTS.values()}
    results_dir = root / "results"
    paths = list(results_dir.glob("experiment_*_v499.json")) if results_dir.exists() else []
    sidecars: list[str] = []
    for path in sorted(paths):
        parts = path.name.split("_")
        exp_id = _to_int(parts[1]) if len(parts) > 1 else 0
        rel = path.relative_to(root).as_posix()
        sidecars.extend([rel] if 5496 <= exp_id <= 5508 and rel not in expected else [])
    return sidecars


def _guided_decoding_status(transition: JsonMap) -> str:
    for row in transition.get("blocked_lanes", []) or []:
        if isinstance(row, Mapping) and "guided_decoding" in str(row.get("lane", "")):
            evidence = row.get("evidence")
            if isinstance(evidence, Mapping):
                return str(evidence.get("quarantine_status") or "quarantined")
    return "quarantined"


def _hard_soft_verdict(artifacts: Mapping[int, JsonMap], missing: set[str]) -> str:
    hard_missing = [_artifact_path(exp_id) for exp_id in (5499, 5500, 5501, 5505) if _artifact_path(exp_id) in missing]
    if hard_missing:
        return "blocked: hard/soft core incomplete because primary artifacts are missing: " + ", ".join(hard_missing)
    exp5499 = artifacts[5499]
    exp5500 = artifacts[5500]
    exp5501 = artifacts[5501]
    exp5505 = artifacts[5505]
    exact_core_ready = bool(exp5499.get("preference_maxsat_fixture_ready")) and bool(
        exp5501.get("helper_contract_fixture_ready")
    ) and bool(exp5505.get("descriptor_ready_for_hardware"))
    abstentions = _to_int(exp5500.get("abstention_count"))
    rows = _to_int(exp5500.get("concept_claim_telemetry_rows"))
    accuracy = _to_float(exp5500.get("exact_validator_accuracy"))
    if exact_core_ready and (abstentions > 0 or accuracy < 1.0):
        return (
            "bounded: exact hard/soft core landed with Exp5499 false_accept_rate="
            f"{_to_float(exp5499.get('false_accept_rate')):.1f}, Exp5501 claim/verdict accuracy=1.0, "
            f"and Exp5505 {exp5505.get('num_descriptor_rows')} descriptors; SOTA panel abstained "
            f"{abstentions}/{rows} rows and exact_validator_accuracy={accuracy:.6f}"
        )
    if exact_core_ready:
        return "headline_ready: exact hard/soft core and SOTA panel all passed their bounded fixtures"
    return "blocked: exact hard/soft core did not satisfy fixture, helper, and descriptor readiness gates"


def _csl_verdict(artifacts: Mapping[int, JsonMap], missing: set[str]) -> str:
    csl_missing = [_artifact_path(exp_id) for exp_id in (5502, 5503, 5504) if _artifact_path(exp_id) in missing]
    if csl_missing:
        return "blocked: CSL synthesis incomplete because primary artifacts are missing: " + ", ".join(csl_missing)
    exp5502 = artifacts[5502]
    exp5503 = artifacts[5503]
    exp5504 = artifacts[5504]
    if bool(exp5504.get("status") == "blocked") or not bool(exp5502.get("metric_independence_clean")):
        return (
            "blocked: Exp5502 resolves the static TAUTOLOGY audit but metric_independence_clean=false, "
            f"Exp5503 replay is bounded with heldout_delta={_to_float(exp5503.get('heldout_delta')):.1f}, "
            f"and Exp5504 is {exp5504.get('honest_verdict')}; Exp5474-style CSL scale headlines remain blocked"
        )
    return "bounded: CSL metric independence and experience-graph replay are clean enough for a next SOTA memory panel"


def _hardware_verdict(hardware: JsonMap, missing: set[str]) -> str:
    if _artifact_path(5506) in missing:
        return "blocked: hardware receipts missing"
    return (
        "bounded: CPU={cpu}, CUDA={cuda}, PolarFire={pf}, KV260={kv}, GateMate={gm}; "
        "matched_timing_available={timing}; hardware_speedup_claim={speedup}"
    ).format(
        cpu=hardware.get("cpu_status"),
        cuda=hardware.get("cuda_status"),
        pf=hardware.get("polar_fire_status"),
        kv=hardware.get("kv260_status"),
        gm=hardware.get("gatemate_status"),
        timing=bool(hardware.get("matched_timing_available")),
        speedup=bool(hardware.get("hardware_speedup_claim")),
    )


def _arc_delta(arc_attempt: JsonMap) -> int:
    if "arc_registry_delta" in arc_attempt:
        return _to_int(arc_attempt.get("arc_registry_delta"))
    return _to_int(arc_attempt.get("registry_after_levels")) - _to_int(
        arc_attempt.get("registry_before_levels")
    )


def _arc_verdict(artifacts: Mapping[int, JsonMap], missing: set[str]) -> str:
    arc_missing = [_artifact_path(exp_id) for exp_id in (5507, 5508) if _artifact_path(exp_id) in missing]
    if arc_missing:
        return "blocked: ARC synthesis incomplete because primary artifacts are missing: " + ", ".join(arc_missing)
    precheck = artifacts[5507]
    attempt = artifacts[5508]
    delta = _arc_delta(attempt)
    if delta > 0 and bool(attempt.get("offline_reproduced")):
        return (
            f"headline_ready: {attempt.get('selected_game')} {attempt.get('selected_level')} banked "
            f"{delta} reproduced level(s) via {attempt.get('solve_provenance')}"
        )
    return (
        f"honest_null: {attempt.get('selected_game') or precheck.get('selected_game')} "
        f"{attempt.get('selected_level') or precheck.get('selected_level')} used "
        f"{attempt.get('solve_provenance', 'unknown_provenance')} with live_agent_attempts="
        f"{_to_int(attempt.get('live_agent_attempts'))}; registry "
        f"{_to_int(attempt.get('registry_before_levels'))}->{_to_int(attempt.get('registry_after_levels'))}; "
        f"reproduced_levels={_to_int(attempt.get('reproduced_levels'))}; methodology supports an honest null, not a solve claim"
    )


def _prd_gap_table(artifacts: Mapping[int, JsonMap], arc_delta: int) -> list[JsonDict]:
    return [
        {
            "prd_item": "FR-11 autonomous self-learning",
            "status": "bounded_headline_blocked",
            "evidence": {
                "exp5502_metric_independence_clean": bool(
                    artifacts[5502].get("metric_independence_clean")
                ),
                "exp5503_heldout_delta": artifacts[5503].get("heldout_delta"),
                "exp5504_status": artifacts[5504].get("status"),
            },
            "gap": "Graph memory replay improved a cached fixture, but metric independence failed and the SOTA CSL panel was gate-blocked.",
            "next": "Run a renamed-field/independence-clean replay and retire same-scope Exp5474 scale headlines if the coupling repeats.",
        },
        {
            "prd_item": "FR-12 verifiable reasoning",
            "status": "bounded_core_ready",
            "evidence": {
                "exp5499_false_accept_rate": artifacts[5499].get("false_accept_rate"),
                "exp5501_rolled_up_verdict_accuracy": artifacts[5501].get(
                    "rolled_up_verdict_accuracy"
                ),
                "exp5505_exact_fallback_agreement_rate": artifacts[5505].get(
                    "exact_fallback_agreement_rate"
                ),
                "exp5500_exact_validator_accuracy": artifacts[5500].get(
                    "exact_validator_accuracy"
                ),
            },
            "gap": "Exact validators and helper contracts are clean, but the live SOTA panel did not generate usable positive claim states.",
            "next": "Use the exact fixture as a generator-format diagnostic before spending more live SOTA panel budget.",
        },
        {
            "prd_item": "NFR-01 performance and hardware",
            "status": "receipt_only_no_speedup",
            "evidence": {
                "cpu_status": artifacts[5506].get("cpu_status"),
                "cuda_status": artifacts[5506].get("cuda_status"),
                "polar_fire_status": artifacts[5506].get("polar_fire_status"),
                "kv260_status": artifacts[5506].get("kv260_status"),
                "gatemate_status": artifacts[5506].get("gatemate_status"),
                "matched_timing_available": artifacts[5506].get("matched_timing_available"),
            },
            "gap": "Receipts prove reachability/hash continuity on some substrates but not the 10x performance requirement.",
            "next": "Only reopen speedup language after authenticated matched timing on the same descriptor workload.",
        },
        {
            "prd_item": "ARC north-star live path",
            "status": "honest_null_no_registry_delta",
            "evidence": {
                "registry_before_levels": artifacts[5508].get("registry_before_levels"),
                "registry_after_levels": artifacts[5508].get("registry_after_levels"),
                "arc_registry_delta": arc_delta,
                "solve_provenance": artifacts[5508].get("solve_provenance"),
            },
            "gap": "Perception-generation changed the mechanism and met prohibited-input discipline, but did not bank dc22 L3.",
            "next": "Change candidate generation beyond repeated coordinate-action alternation before another dc22 L3 attempt.",
        },
        {
            "prd_item": "FR-09/FR-10 spec and test discipline",
            "status": "new_work_traced_repo_backlog_visible",
            "evidence": {
                "exp5497_spec_coverage_note": "current lane tests are REQ/SCENARIO anchored",
                "repo_wide_traceability_backlog_seen": True,
            },
            "gap": "New V499 lanes generally recorded focused coverage, while repo-wide historical spec/test backlog remains visible in upstream test logs.",
            "next": "Keep capstone modules narrow and traceable; do not chase old suite-wide backlog inside synthesis tasks.",
        },
    ]


def _failure_taxonomy(artifacts: Mapping[int, JsonMap], arc_delta: int) -> list[JsonDict]:
    return [
        {
            "failure_class": "historical_pretest_cascade_resolved_with_caveat",
            "evidence": {
                "pretest_cascade_resolved": bool(artifacts[5497].get("pretest_cascade_resolved")),
                "reproduced_pretest_failure": bool(
                    artifacts[5497].get("reproduced_pretest_failure")
                ),
                "recommendation": artifacts[5497].get("downstream_gate_recommendation"),
            },
            "classification": "bounded",
        },
        {
            "failure_class": "sota_abstention_panel",
            "evidence": {
                "abstention_count": _to_int(artifacts[5500].get("abstention_count")),
                "rows": _to_int(artifacts[5500].get("concept_claim_telemetry_rows")),
                "exact_validator_accuracy": artifacts[5500].get("exact_validator_accuracy"),
            },
            "classification": "bounded",
        },
        {
            "failure_class": "csl_metric_independence_blocker",
            "evidence": {
                "metric_independence_clean": bool(artifacts[5502].get("metric_independence_clean")),
                "csl_scale_headline_allowed": bool(
                    artifacts[5502].get("csl_scale_headline_allowed")
                ),
                "exp5504_status": artifacts[5504].get("status"),
                "gate_check_summary": artifacts[5504].get("gate_check_summary"),
            },
            "classification": "blocked",
        },
        {
            "failure_class": "hardware_methodology_flag",
            "evidence": {
                "flagged_adversarial": bool(artifacts[5506].get("flagged_adversarial")),
                "corrigendum_pending": artifacts[5506].get("corrigendum_pending", []),
                "matched_timing_available": bool(artifacts[5506].get("matched_timing_available")),
            },
            "classification": "bounded",
        },
        {
            "failure_class": "hardware_identity_blocks",
            "evidence": {
                "kv260_status": artifacts[5506].get("kv260_status"),
                "gatemate_status": artifacts[5506].get("gatemate_status"),
            },
            "classification": "blocked",
        },
        {
            "failure_class": "arc_no_bank",
            "evidence": {
                "arc_registry_delta": arc_delta,
                "live_agent_attempts": artifacts[5508].get("live_agent_attempts"),
                "reproduced_levels": artifacts[5508].get("reproduced_levels"),
                "trajectory_taxonomy_counts": artifacts[5508].get("trajectory_taxonomy_counts"),
            },
            "classification": "honest_null",
        },
    ]


def _lane_scorecard(
    artifacts: Mapping[int, JsonMap],
    verdicts: Mapping[str, str],
    artifacts_missing: Sequence[str],
) -> list[JsonDict]:
    return [
        {
            "lane": "pretest_recovery",
            "classification": "bounded",
            "source_artifacts": [_artifact_path(5497)],
            "evidence": {
                "pretest_cascade_resolved": bool(artifacts[5497].get("pretest_cascade_resolved")),
                "full_suite_caveat": True,
            },
        },
        {
            "lane": "hard_soft_verification_core",
            "classification": "bounded" if verdicts["hard_soft"].startswith("bounded:") else "blocked",
            "source_artifacts": [_artifact_path(exp_id) for exp_id in (5499, 5500, 5501, 5505)],
            "evidence": {"verdict": verdicts["hard_soft"]},
        },
        {
            "lane": "continuous_self_learning",
            "classification": "blocked" if verdicts["csl"].startswith("blocked:") else "bounded",
            "source_artifacts": [_artifact_path(exp_id) for exp_id in (5502, 5503, 5504)],
            "evidence": {"verdict": verdicts["csl"]},
        },
        {
            "lane": "hardware_receipts",
            "classification": "bounded",
            "source_artifacts": [_artifact_path(5506)],
            "evidence": {"verdict": verdicts["hardware"]},
        },
        {
            "lane": "arc_live_path",
            "classification": "honest_null" if verdicts["arc"].startswith("honest_null:") else "blocked",
            "source_artifacts": [_artifact_path(5507), _artifact_path(5508)],
            "evidence": {"verdict": verdicts["arc"]},
        },
        {
            "lane": "missing_primary_artifacts",
            "classification": "missing" if artifacts_missing else "complete",
            "source_artifacts": list(artifacts_missing),
            "evidence": {"missing_count": len(artifacts_missing)},
        },
    ]


def _next_recommendations() -> list[JsonDict]:
    return [
        {
            "rank": 1,
            "recommendation": "Retire Exp5474-style CSL scale headlines until metric independence is clean.",
            "evidence": "Exp5502 resolved the static TAUTOLOGY label but kept metric_independence_clean=false; Exp5504 gate-blocked.",
            "next_experiment": "Run a minimal independent-outcome CSL replay whose gate field names exactly match conductor expectations.",
        },
        {
            "rank": 2,
            "recommendation": "Keep the hard/soft core, but diagnose generation format before another SOTA panel.",
            "evidence": "Exp5499/Exp5501/Exp5505 exact checks are clean; Exp5500 abstained on all six model-instance rows.",
            "next_experiment": "Add a parser/format positive control over the Exp5499 fixture before spending flagship GGUF runtime.",
        },
        {
            "rank": 3,
            "recommendation": "Preserve hardware receipt discipline and do not make speedup claims.",
            "evidence": "Exp5506 has reachable CPU/CUDA/PolarFire smoke receipts, blocked KV260/GateMate identity, no matched timing, and hardware_speedup_claim=false.",
            "next_experiment": "Run authenticated same-workload timing only after board identity/workload gates are clean.",
        },
        {
            "rank": 4,
            "recommendation": "Change ARC candidate generation before another dc22 L3 live attempt.",
            "evidence": "Exp5508 ran 47 live-agent attempts with registry 69->69 and no reproduced level.",
            "next_experiment": "Target a mechanism that generates non-repeated action sequences rather than alternating the same coordinate probes.",
        },
    ]


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifacts_found, artifacts_missing, artifact_metadata = _read_artifacts(root)
    missing = set(artifacts_missing)
    source_context, source_context_missing = _source_context(root)
    sidecars = _discover_sidecars(root)
    arc_registry_delta = _arc_delta(artifacts[5508])
    hard_soft_verdict = _hard_soft_verdict(artifacts, missing)
    csl_verdict = _csl_verdict(artifacts, missing)
    hardware_verdict = _hardware_verdict(artifacts[5506], missing)
    arc_verdict = _arc_verdict(artifacts, missing)
    verdicts = {
        "hard_soft": hard_soft_verdict,
        "csl": csl_verdict,
        "hardware": hardware_verdict,
        "arc": arc_verdict,
    }
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    status_prefix = "blocked:" if artifacts_missing else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": "blocked" if artifacts_missing else "complete",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "artifact_metadata": artifact_metadata,
        "sidecar_artifacts_found": sidecars,
        "lane_scorecard": _lane_scorecard(artifacts, verdicts, artifacts_missing),
        "failure_taxonomy": _failure_taxonomy(artifacts, arc_registry_delta),
        "tests_run": list(tests_run),
        "reproducibility_checksum": "",
        "milestone": MILESTONE,
        "artifacts_expected": [path.as_posix() for path in EXPECTED_ARTIFACTS.values()],
        "artifacts_found": artifacts_found,
        "artifacts_missing": artifacts_missing,
        "pretest_cascade_resolved": bool(artifacts[5497].get("pretest_cascade_resolved")),
        "hard_soft_core_verdict": hard_soft_verdict,
        "csl_verdict": csl_verdict,
        "hardware_verdict": hardware_verdict,
        "arc_verdict": arc_verdict,
        "arc_registry_delta": arc_registry_delta,
        "hardware_speedup_claim": bool(artifacts[5506].get("hardware_speedup_claim")),
        "guided_decoding_quarantine_status": _guided_decoding_status(artifacts[5496]),
        "prd_gap_table": _prd_gap_table(artifacts, arc_registry_delta),
        "next_recommendations": _next_recommendations(),
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "protected_file_checks": [
            {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
                "git_status_clean": not roadmap_modified,
                "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            },
            {
                "path": CONDUCTOR_RELATIVE_PATH.as_posix(),
                "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
                "git_status_clean": not conductor_modified,
                "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
            },
        ],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"{status_prefix} .499 capstone read actual Exp5496-Exp5508 artifacts; "
            f"pretest_cascade_resolved={bool(artifacts[5497].get('pretest_cascade_resolved'))}; "
            "hard/soft core is bounded by SOTA abstentions; CSL headline claims remain blocked; "
            f"hardware_speedup_claim={bool(artifacts[5506].get('hardware_speedup_claim'))}; "
            f"arc_registry_delta={arc_registry_delta}; missing_primary_artifacts={len(artifacts_missing)}"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def write_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[Any] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    payload = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    write_report(args.root)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
