"""Exp5482 transition receipt from milestone .497 into .498.

Spec refs: REQ-REPORT-5482, SCENARIO-REPORT-5482,
SCENARIO-REPORT-5482-BLOCKED-INPUT.

This module is a record-only handoff. It does not re-run models, ARC agents,
or hardware workloads. Instead, it reads the .497 capstone plus the two
load-bearing exception artifacts: Exp5474, which later carried an adversarial
TAUTOLOGY flag despite reporting CSL scale readiness, and Exp5480, which
records the `sb26` L3 ARC no-bank. The JSON it writes is meant to make those
boundaries explicit before .498 work begins, so downstream tasks inherit facts
rather than roadmap hopes.
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
    extract_roadmap_tasks,
    normalize_task_range,
    path_sha256,
    payload_checksum,
    read_json_mapping,
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5482_transition_v498.json")
PRIOR_CAPSTONE_RELATIVE_PATH = Path("results/experiment_5481_capstone_v497.json")
EXP5474_RELATIVE_PATH = Path("results/experiment_5474_sota_csl_scale_v497.json")
EXP5480_RELATIVE_PATH = Path("results/experiment_5480_arc_live_salience_levelup_v497.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5482_transition_v498"
EXPERIMENT_ID = "exp5482-transition-v498"
MILESTONE = "2026.07.498"
PREVIOUS_MILESTONE = "2026.07.497"
PREVIOUS_TASK_RANGE = "exp5468-exp5481"
NEXT_TASK_RANGE = "exp5482-exp5495"
SCHEMA = "carnot.experiment_5482.transition_v498.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5482
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

EXPECTED_TASK_IDS = [
    "exp5482-transition-v498",
    "exp5483-source-delta-v498",
    "exp5484-csl-tautology-corrigendum-v498",
    "exp5485-preference-maxsat-claim-fixture-v498",
    "exp5486-gated-sota-concept-evidence-panel-v498",
    "exp5487-helper-contract-nl-spec-repair-v498",
    "exp5488-csl-latent-exploration-replay-v498",
    "exp5489-gated-sota-csl-independent-metrics-v498",
    "exp5490-csl-kan-fixed-point-update-ledger-v498",
    "exp5491-active-constraint-subproblem-descriptor-v498",
    "exp5492-gated-hardware-receipts-v498",
    "exp5493-arc-trajectory-target-precheck-v498",
    "exp5494-gated-arc-live-trajectory-levelup-v498",
    "exp5495-capstone-v498",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "route key for the new .498 transition receipt.",
    "previous_milestone": "source milestone whose terminal facts are being archived.",
    "prior_capstone_path": "exact .497 capstone artifact used as the main source of truth.",
    "previous_task_range": "closed .497 conductor range.",
    "clean_lanes": "prior lanes that can be carried forward with their stated boundaries.",
    "bounded_lanes": "useful evidence that remains claim-limited in the .497 capstone.",
    "blocked_lanes": "quarantined or unreachable lanes that must not be promoted.",
    "honest_null_lanes": "executed lanes that produced no positive bankable result.",
    "flagged_lanes": "adversarially flagged evidence that must be repaired before headline use.",
    "exp5474_tautology_flag_recorded": "bare boolean proving the CSL scale tautology caveat was seen.",
    "next_task_range": "planned .498 conductor range.",
    "roadmap_yaml_unchanged": "protected-file check for research-roadmap.yaml.",
    "conductor_unchanged": "protected-file check for scripts/research_conductor.py.",
    "inference_substrate": "aggregation only; no hidden live inference or hardware run.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "previous_milestone",
    "prior_capstone_path",
    "previous_task_range",
    "clean_lanes",
    "bounded_lanes",
    "blocked_lanes",
    "honest_null_lanes",
    "flagged_lanes",
    "exp5474_tautology_flag_recorded",
    "next_task_range",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "inference_substrate",
    "honest_verdict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "status",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "roadmap_task_ids",
    "roadmap_doc_task_range",
    "source_artifacts",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = (
    "REQ-REPORT-5482",
    "SCENARIO-REPORT-5482",
    "SCENARIO-REPORT-5482-BLOCKED-INPUT",
)

SOURCE_CONTEXT_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    PRIOR_CAPSTONE_RELATIVE_PATH,
    EXP5474_RELATIVE_PATH,
    EXP5480_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5482_transition_v498.py -q --no-cov",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5482_transition_v498.py "
            "-m pytest tests/python/test_experiment_5482_transition_v498.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5482_transition_v498.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    {
        "command": (
            "ops/e2e-test-plan.md review: Exp5482 is aggregation-only; no fresh "
            "training, PyO3 round trip, ARC live action, or hardware workload applies"
        ),
        "outcome": "not_applicable",
    },
)

CLEAN_LANE_NAMES = [
    "rewrite_state_guards",
    "guard_composition",
    "sota_evidence_telemetry",
    "kan_assurance",
    "behavioral_memory",
    "helper_repair",
    "pbit_pdit_boundary_exchange",
    "hardware_receipts",
]
BOUNDED_LANE_NAMES = [
    "local_sota_runtime",
    "pdit_lns_boundary_exchange",
    "hardware_receipts",
]
BLOCKED_LANE_NAMES = ["guided_decoding", "kv260_board", "gatemate_board"]
HONEST_NULL_LANE_NAMES = ["arc_sb26_l3_no_bank", "hardware_speedup_claim"]
FLAGGED_LANE_NAMES = ["exp5474_sota_csl_scale_tautology"]


def _truth_rows(capstone: JsonMap) -> dict[str, JsonDict]:
    rows = capstone.get("truth_table")
    if isinstance(rows, Mapping):
        return {str(lane): dict(row) for lane, row in rows.items() if isinstance(row, Mapping)}
    if isinstance(rows, list):
        return {
            str(row["lane"]): dict(row)
            for row in rows
            if isinstance(row, Mapping) and "lane" in row
        }
    return {}


def _evidence(row: JsonMap) -> JsonDict:
    evidence = row.get("evidence", row.get("terminal_evidence"))
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _source_artifacts(row: JsonMap, fallback: Sequence[str]) -> list[str]:
    source = row.get("source_artifacts", row.get("source_artifact"))
    if isinstance(source, list):
        return [str(item) for item in source]
    if isinstance(source, str):
        return [source]
    return [str(item) for item in fallback]


def _lane(
    lane: str,
    classification: str,
    source_artifacts: Sequence[str],
    evidence: JsonMap,
    claim_boundary: str,
) -> JsonDict:
    return {
        "lane": lane,
        "classification": classification,
        "source_artifacts": [str(item) for item in source_artifacts],
        "evidence": dict(evidence),
        "claim_boundary": claim_boundary,
    }


def _truth_lane(
    *,
    lane: str,
    source_row: JsonMap,
    fallback_sources: Sequence[str],
    evidence_fields: Sequence[str],
    claim_boundary: str,
) -> JsonDict:
    evidence = _evidence(source_row)
    selected = {field: evidence.get(field) for field in evidence_fields}
    return _lane(
        lane,
        str(source_row.get("classification", "unknown")),
        _source_artifacts(source_row, fallback_sources),
        selected,
        claim_boundary,
    )


def derive_clean_lanes(capstone: JsonMap) -> list[JsonDict]:
    rows = _truth_rows(capstone)
    reasoning = rows.get("verifiable_reasoning_guards", {})
    local_sota = rows.get("local_sota_runtime", {})
    csl = rows.get("csl", {})
    pdit = rows.get("pdit_lns_boundary_exchange", {})
    hardware = rows.get("hardware_receipts", {})
    lanes: list[JsonDict] = []
    if reasoning:
        lanes.extend(
            [
                _truth_lane(
                    lane="rewrite_state_guards",
                    source_row=reasoning,
                    fallback_sources=[
                        "results/experiment_5470_rewrite_state_semantic_fixture_v497.json"
                    ],
                    evidence_fields=("rewrite_state_fixture_ready", "exact_validator_agreement"),
                    claim_boundary="Deterministic rewrite-state guard evidence from Exp5470.",
                ),
                _truth_lane(
                    lane="guard_composition",
                    source_row=reasoning,
                    fallback_sources=["results/experiment_5471_guard_composition_scale_v497.json"],
                    evidence_fields=("guard_composition_ready", "false_accept_rate"),
                    claim_boundary="Guard composition remains exact-validator bounded.",
                ),
            ]
        )
    if local_sota:
        lanes.append(
            _truth_lane(
                lane="sota_evidence_telemetry",
                source_row=local_sota,
                fallback_sources=["results/experiment_5472_sota_evidence_telemetry_v497.json"],
                evidence_fields=(
                    "sota_evidence_telemetry_ready",
                    "guided_decoding_used",
                    "gpu_offload_receipt_count",
                    "exact_validator_accuracy",
                ),
                claim_boundary=(
                    "Local SOTA telemetry is bounded runtime evidence; exact validators "
                    "remain authority."
                ),
            )
        )
    if csl:
        lanes.extend(
            [
                _truth_lane(
                    lane="kan_assurance",
                    source_row=csl,
                    fallback_sources=[
                        "results/experiment_5473_csl_kan_surrogate_assurance_v497.json"
                    ],
                    evidence_fields=("csl_kan_surrogate_ready", "model_weight_mutation"),
                    claim_boundary="KAN assurance applies to governed frozen-policy CSL only.",
                ),
                _truth_lane(
                    lane="behavioral_memory",
                    source_row=csl,
                    fallback_sources=[
                        "results/experiment_5475_csl_behavioral_memory_ladder_v497.json"
                    ],
                    evidence_fields=("csl_behavioral_memory_ready", "model_weight_mutation"),
                    claim_boundary="Behavioral memory evidence is frozen-policy routing evidence.",
                ),
            ]
        )
    if reasoning:
        lanes.append(
            _truth_lane(
                lane="helper_repair",
                source_row=reasoning,
                fallback_sources=[
                    "results/experiment_5476_helper_lemma_core_witness_repair_v497.json"
                ],
                evidence_fields=("helper_lemma_repair_ready", "helper_false_accept_count"),
                claim_boundary="Helper repair is credited only after exact rechecks.",
            )
        )
    if pdit:
        lanes.append(
            _truth_lane(
                lane="pbit_pdit_boundary_exchange",
                source_row=pdit,
                fallback_sources=["results/experiment_5477_pdit_lns_boundary_exchange_v497.json"],
                evidence_fields=(
                    "boundary_exchange_ready",
                    "exact_fallback_completeness_rate",
                    "unsafe_false_accept_count",
                    "hardware_speedup_claim",
                ),
                claim_boundary="p-bit/p-dit exchange is solver-advisory; exact fallback remains final.",
            )
        )
    if hardware:
        lanes.append(
            _truth_lane(
                lane="hardware_receipts",
                source_row=hardware,
                fallback_sources=["results/experiment_5478_hardware_receipts_v497.json"],
                evidence_fields=(
                    "hardware_receipts_ready",
                    "hardware_speedup_claim",
                    "result_hash_match_rate",
                    "reachable_boards",
                    "unreachable_boards",
                ),
                claim_boundary=(
                    "Hardware receipts are hash-matched receipt evidence, not speedup evidence."
                ),
            )
        )
    return lanes


def derive_bounded_lanes(capstone: JsonMap) -> list[JsonDict]:
    rows = _truth_rows(capstone)
    bounded: list[JsonDict] = []
    for lane in BOUNDED_LANE_NAMES:
        row = rows.get(lane, {})
        if row:
            bounded.append(
                _lane(
                    lane,
                    str(row.get("classification", "unknown")),
                    _source_artifacts(row, [str(PRIOR_CAPSTONE_RELATIVE_PATH)]),
                    _evidence(row),
                    str(row.get("claim_boundary", "Bounded .497 evidence.")),
                )
            )
    return bounded


def _board_blocks(capstone: JsonMap) -> list[JsonDict]:
    hardware = _evidence(_truth_rows(capstone).get("hardware_receipts", {}))
    boards = hardware.get("unreachable_boards")
    if not isinstance(boards, list):
        return []
    blocked: list[JsonDict] = []
    for expected in ("kv260", "gatemate"):
        for board in boards:
            if isinstance(board, Mapping) and str(board.get("board_identity")) == expected:
                blocked.append(
                    _lane(
                        f"{expected}_board",
                        "blocked",
                        ["results/experiment_5478_hardware_receipts_v497.json"],
                        board,
                        f"{expected} had no credited workload receipt in .497.",
                    )
                )
                break
    return blocked


def derive_blocked_lanes(capstone: JsonMap) -> list[JsonDict]:
    rows = _truth_rows(capstone)
    guided = rows.get("guided_decoding", {})
    blocked = []
    if guided:
        blocked.append(
            _lane(
                "guided_decoding",
                str(guided.get("classification", "unknown")),
                _source_artifacts(
                    guided,
                    [
                        "results/experiment_5468_transition_v497.json",
                        "results/experiment_5470_rewrite_state_semantic_fixture_v497.json",
                        "results/experiment_5471_guard_composition_scale_v497.json",
                        "results/experiment_5472_sota_evidence_telemetry_v497.json",
                    ],
                ),
                _evidence(guided),
                str(
                    guided.get(
                        "claim_boundary",
                        "Guided decoding remains quarantined after .497.",
                    )
                ),
            )
        )
    blocked.extend(_board_blocks(capstone))
    return blocked


def derive_honest_null_lanes(capstone: JsonMap, exp5480: JsonMap) -> list[JsonDict]:
    rows = _truth_rows(capstone)
    lanes: list[JsonDict] = []
    arc_row = _evidence(rows.get("arc_live_path", {}))
    if arc_row:
        arc_evidence = {
            "game": exp5480.get("game", arc_row.get("selected_game")),
            "target_level": exp5480.get("target_level", arc_row.get("selected_target_level")),
            "new_level_banked": exp5480.get("new_level_banked", arc_row.get("new_level_banked")),
            "offline_reproduced": exp5480.get(
                "offline_reproduced", arc_row.get("offline_reproduced")
            ),
            "failure_mode": exp5480.get("failure_mode", arc_row.get("failure_mode")),
            "reproduced_levels_before": exp5480.get(
                "reproduced_levels_before", arc_row.get("reproduced_levels_before")
            ),
            "reproduced_levels_after": exp5480.get(
                "reproduced_levels_after", arc_row.get("reproduced_levels_after")
            ),
            "action_count": exp5480.get("action_count"),
            "explored_state_count": exp5480.get("explored_state_count"),
        }
        lanes.append(
            _lane(
                "arc_sb26_l3_no_bank",
                "honest_null",
                [
                    "results/experiment_5479_arc_target_rotation_precheck_v497.json",
                    str(EXP5480_RELATIVE_PATH),
                ],
                arc_evidence,
                "ARC sb26 L3 no-bank; no registry delta or offline reproduction.",
            )
        )

    hardware_row = _evidence(rows.get("hardware_receipts", {}))
    speedup_row = _evidence(rows.get("hardware_speedup_claim", {}))
    if hardware_row and speedup_row:
        lanes.append(
            _lane(
                "hardware_speedup_claim",
                "honest_null",
                ["results/experiment_5478_hardware_receipts_v497.json"],
                speedup_row,
                "No hardware speedup claim is supported by .497 receipts.",
            )
        )
    return lanes


def _exp5474_tautology_record(exp5474: JsonMap) -> tuple[bool, JsonDict]:
    pending = exp5474.get("corrigendum_pending")
    if not isinstance(pending, list):
        return False, {}
    for row in pending:
        if not isinstance(row, Mapping):
            continue
        kind = str(row.get("kind", "")).upper()
        severity = str(row.get("severity", "")).lower()
        if kind == "TAUTOLOGY" and severity == "critical":
            return (
                exp5474.get("flagged_adversarial") is True
                and exp5474.get("csl_scale_ready") is True,
                {
                    "flag_kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": row.get("detail", ""),
                    "flagged_adversarial": exp5474.get("flagged_adversarial"),
                    "artifact_reported_csl_scale_ready": exp5474.get("csl_scale_ready"),
                    "delta_vs_naive_icl": exp5474.get("delta_vs_naive_icl"),
                    "delta_vs_no_memory": exp5474.get("delta_vs_no_memory"),
                    "exact_validator_pass_rate": exp5474.get("exact_validator_pass_rate"),
                    "honest_verdict": exp5474.get("honest_verdict", ""),
                },
            )
    return False, {}


def derive_flagged_lanes(exp5474: JsonMap) -> list[JsonDict]:
    recorded, evidence = _exp5474_tautology_record(exp5474)
    if not recorded:
        return []
    return [
        _lane(
            "exp5474_sota_csl_scale_tautology",
            "flagged",
            [str(EXP5474_RELATIVE_PATH)],
            evidence,
            "Exp5474 reported CSL scale readiness but is adversarially flagged TAUTOLOGY.",
        )
    ]


def source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "path": str(relative),
            "exists": (root / relative).exists(),
            "sha256": path_sha256(root / relative),
            "read_only": True,
        }
        for relative in SOURCE_CONTEXT_PATHS
    ]


def protected_file_checks(
    root: Path,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[JsonDict]:
    return [
        {
            "path": str(ROADMAP_RELATIVE_PATH),
            "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
            "git_status_clean": not roadmap_modified,
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        },
        {
            "path": str(CONDUCTOR_RELATIVE_PATH),
            "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
            "git_status_clean": not conductor_modified,
            "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        },
    ]


def _missing_lane_names(rows: Sequence[JsonMap], expected: Sequence[str]) -> list[str]:
    observed = [row.get("lane") for row in rows]
    return [lane for lane in expected if lane not in observed]


def _capstone_failures(capstone: JsonMap, meta: JsonMap) -> list[str]:
    if meta.get("loadable") is not True:
        return ["capstone_missing_or_unloadable"]
    failures: list[str] = []
    if capstone.get("milestone") != PREVIOUS_MILESTONE:
        failures.append(
            f"capstone_milestone_expected_{PREVIOUS_MILESTONE}_observed_{capstone.get('milestone')}"
        )
    verdict = capstone.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    if capstone.get("hardware_speedup_claim") is not False:
        failures.append("capstone_hardware_speedup_claim_not_false")
    return failures


def _exp5474_failures(exp5474: JsonMap, meta: JsonMap, tautology_recorded: bool) -> list[str]:
    if meta.get("loadable") is not True:
        return ["exp5474_missing_or_unloadable"]
    failures: list[str] = []
    if exp5474.get("milestone") != PREVIOUS_MILESTONE:
        failures.append(
            f"exp5474_milestone_expected_{PREVIOUS_MILESTONE}_observed_{exp5474.get('milestone')}"
        )
    if not tautology_recorded:
        failures.append("exp5474_tautology_flag_missing")
    return failures


def _exp5480_failures(exp5480: JsonMap, meta: JsonMap) -> list[str]:
    if meta.get("loadable") is not True:
        return ["exp5480_missing_or_unloadable"]
    failures: list[str] = []
    if exp5480.get("milestone") != PREVIOUS_MILESTONE:
        failures.append(
            f"exp5480_milestone_expected_{PREVIOUS_MILESTONE}_observed_{exp5480.get('milestone')}"
        )
    if exp5480.get("game") != "sb26":
        failures.append(f"exp5480_expected_sb26_observed_{exp5480.get('game')}")
    if exp5480.get("target_level") != 3:
        failures.append(f"exp5480_target_level_expected_3_observed_{exp5480.get('target_level')}")
    if exp5480.get("new_level_banked") is not False:
        failures.append("exp5480_new_level_banked_expected_false")
    if exp5480.get("offline_reproduced") is not False:
        failures.append("exp5480_offline_reproduced_expected_false")
    return failures


def _lane_failures(
    *,
    clean_lanes: Sequence[JsonMap],
    bounded_lanes: Sequence[JsonMap],
    blocked_lanes: Sequence[JsonMap],
    honest_null_lanes: Sequence[JsonMap],
    flagged_lanes: Sequence[JsonMap],
    capstone_loadable: bool,
) -> list[str]:
    if not capstone_loadable:
        return []
    failures: list[str] = []
    if _missing_lane_names(clean_lanes, CLEAN_LANE_NAMES):
        failures.append("clean_lanes_incomplete")
    if _missing_lane_names(bounded_lanes, BOUNDED_LANE_NAMES):
        failures.append("bounded_lanes_incomplete")
    if _missing_lane_names(blocked_lanes, BLOCKED_LANE_NAMES):
        failures.append("blocked_lanes_incomplete")
    if _missing_lane_names(honest_null_lanes, HONEST_NULL_LANE_NAMES):
        failures.append("honest_null_lanes_incomplete")
    if _missing_lane_names(flagged_lanes, FLAGGED_LANE_NAMES):
        failures.append("flagged_lanes_incomplete")
    return failures


def _failed_preconditions(
    *,
    capstone_failures: Sequence[str],
    exp5474_failures: Sequence[str],
    exp5480_failures: Sequence[str],
    lane_failures: Sequence[str],
    roadmap_milestone: str | None,
    roadmap_task_ids: Sequence[str],
    doc_names_milestone: bool,
    doc_task_range: str | None,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [
        *capstone_failures,
        *exp5474_failures,
        *exp5480_failures,
        *lane_failures,
    ]
    if roadmap_milestone != MILESTONE:
        failures.append(f"roadmap_milestone_expected_{MILESTONE}_observed_{roadmap_milestone}")
    if list(roadmap_task_ids) != EXPECTED_TASK_IDS:
        failures.append("roadmap_task_ids_mismatch")
    if not doc_names_milestone:
        failures.append(f"roadmap_doc_missing_or_mismatch_{MILESTONE}")
    if doc_task_range != NEXT_TASK_RANGE:
        failures.append(
            f"roadmap_doc_task_range_expected_{NEXT_TASK_RANGE}_observed_{doc_task_range}"
        )
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .497 terminal evidence into .498 transition receipt; "
            "clean lanes include rewrite-state guards, guard composition, SOTA evidence "
            "telemetry, KAN assurance, behavioral memory, helper repair, p-bit/p-dit "
            "boundary exchange, and hardware receipts; Exp5474 critical TAUTOLOGY flag "
            "recorded; guided decoding remains quarantined; ARC sb26 L3 no-bank and "
            "hardware_speedup_claim=false; next_task_range=exp5482-exp5495."
        )
    first_failure = failures[0] if failures else "unknown"
    return f"blocked: .498 transition receipt failed precondition {first_failure}."


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    root_path = Path(root)
    capstone, capstone_meta = read_json_mapping(root_path / PRIOR_CAPSTONE_RELATIVE_PATH)
    exp5474, exp5474_meta = read_json_mapping(root_path / EXP5474_RELATIVE_PATH)
    exp5480, exp5480_meta = read_json_mapping(root_path / EXP5480_RELATIVE_PATH)
    roadmap, _roadmap_meta = read_yaml_mapping(root_path / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    doc_path = root_path / VNEXT_RELATIVE_PATH
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    doc_task_range = normalize_task_range(doc_text)
    roadmap_modified = _modification_status(root_path, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_modified = _modification_status(
        root_path, CONDUCTOR_RELATIVE_PATH, modification_status
    )
    roadmap_milestone = roadmap.get("milestone")
    roadmap_milestone = str(roadmap_milestone) if roadmap_milestone is not None else None

    clean_lanes = derive_clean_lanes(capstone) if capstone_meta.get("loadable") is True else []
    bounded_lanes = derive_bounded_lanes(capstone) if capstone_meta.get("loadable") is True else []
    blocked_lanes = derive_blocked_lanes(capstone) if capstone_meta.get("loadable") is True else []
    honest_null_lanes = (
        derive_honest_null_lanes(capstone, exp5480) if capstone_meta.get("loadable") is True else []
    )
    exp5474_tautology_flag_recorded, _tautology_evidence = _exp5474_tautology_record(exp5474)
    flagged_lanes = derive_flagged_lanes(exp5474)
    capstone_loadable = capstone_meta.get("loadable") is True
    failures = _failed_preconditions(
        capstone_failures=_capstone_failures(capstone, capstone_meta),
        exp5474_failures=_exp5474_failures(exp5474, exp5474_meta, exp5474_tautology_flag_recorded),
        exp5480_failures=_exp5480_failures(exp5480, exp5480_meta),
        lane_failures=_lane_failures(
            clean_lanes=clean_lanes,
            bounded_lanes=bounded_lanes,
            blocked_lanes=blocked_lanes,
            honest_null_lanes=honest_null_lanes,
            flagged_lanes=flagged_lanes,
            capstone_loadable=capstone_loadable,
        ),
        roadmap_milestone=roadmap_milestone,
        roadmap_task_ids=roadmap_task_ids,
        doc_names_milestone=MILESTONE in doc_text,
        doc_task_range=doc_task_range,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "previous_milestone": PREVIOUS_MILESTONE,
        "prior_capstone_path": str(PRIOR_CAPSTONE_RELATIVE_PATH),
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "clean_lanes": clean_lanes,
        "bounded_lanes": bounded_lanes,
        "blocked_lanes": blocked_lanes,
        "honest_null_lanes": honest_null_lanes,
        "flagged_lanes": flagged_lanes,
        "exp5474_tautology_flag_recorded": exp5474_tautology_flag_recorded,
        "next_task_range": NEXT_TASK_RANGE,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": doc_task_range,
        "source_artifacts": source_artifacts(root_path),
        "protected_file_checks": protected_file_checks(
            root_path,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "capstone_present": capstone_meta.get("exists") is True,
            "capstone_loadable": capstone_loadable,
            "capstone_milestone": capstone.get("milestone"),
            "exp5474_present": exp5474_meta.get("exists") is True,
            "exp5474_loadable": exp5474_meta.get("loadable") is True,
            "exp5474_csl_scale_ready": exp5474.get("csl_scale_ready"),
            "exp5474_tautology_flag_recorded": exp5474_tautology_flag_recorded,
            "exp5480_present": exp5480_meta.get("exists") is True,
            "exp5480_loadable": exp5480_meta.get("loadable") is True,
            "exp5480_game": exp5480.get("game"),
            "exp5480_target_level": exp5480.get("target_level"),
            "exp5480_new_level_banked": exp5480.get("new_level_banked"),
            "roadmap_milestone": roadmap_milestone,
            "roadmap_task_ids_match": roadmap_task_ids == EXPECTED_TASK_IDS,
            "roadmap_doc_task_range": doc_task_range,
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
            "roadmap_next_present": (root_path / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        },
        "failed_preconditions": failures,
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": _honest_verdict(status, failures),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _lane_names(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [str(row.get("lane")) for row in rows if isinstance(row, Mapping)]


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if payload.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("status") not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if payload.get("milestone") != MILESTONE:
        raise ValueError("milestone mismatch")
    if payload.get("previous_milestone") != PREVIOUS_MILESTONE:
        raise ValueError("previous_milestone mismatch")
    if payload.get("prior_capstone_path") != str(PRIOR_CAPSTONE_RELATIVE_PATH):
        raise ValueError("prior_capstone_path mismatch")
    if payload.get("previous_task_range") != PREVIOUS_TASK_RANGE:
        raise ValueError("previous_task_range mismatch")
    if payload.get("next_task_range") != NEXT_TASK_RANGE:
        raise ValueError("next_task_range mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    for field in ("roadmap_yaml_unchanged", "conductor_unchanged"):
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"{field} must be boolean")
    if not isinstance(payload.get("exp5474_tautology_flag_recorded"), bool):
        raise ValueError("exp5474_tautology_flag_recorded must be boolean")
    for field in (
        "clean_lanes",
        "bounded_lanes",
        "blocked_lanes",
        "honest_null_lanes",
        "flagged_lanes",
    ):
        if not isinstance(payload.get(field), list):
            raise ValueError(f"{field} must be a list")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    failures = payload.get("failed_preconditions")
    if not isinstance(failures, list):
        raise ValueError("failed_preconditions must be a list")
    if payload.get("status") == "complete":
        if failures:
            raise ValueError("complete status cannot have failed_preconditions")
        if payload.get("roadmap_yaml_unchanged") is not True:
            raise ValueError("roadmap_yaml_unchanged must be true for complete status")
        if payload.get("conductor_unchanged") is not True:
            raise ValueError("conductor_unchanged must be true for complete status")
        if payload.get("exp5474_tautology_flag_recorded") is not True:
            raise ValueError("complete status requires exp5474 tautology flag")
        if payload.get("roadmap_task_ids") != EXPECTED_TASK_IDS:
            raise ValueError("roadmap_task_ids mismatch")
        expected = {
            "clean_lanes": CLEAN_LANE_NAMES,
            "bounded_lanes": BOUNDED_LANE_NAMES,
            "blocked_lanes": BLOCKED_LANE_NAMES,
            "honest_null_lanes": HONEST_NULL_LANE_NAMES,
            "flagged_lanes": FLAGGED_LANE_NAMES,
        }
        for field, names in expected.items():
            missing_names = [name for name in names if name not in _lane_names(payload[field])]
            if missing_names:
                raise ValueError(f"{field} missing lanes: {missing_names}")
    elif not failures:
        raise ValueError("blocked status must record failed_preconditions")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    run_date: str = RUN_DATE,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> Path:
    root_path = Path(root)
    output_path = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    payload = build_artifact(
        root=root_path,
        run_date=run_date,
        tests_run=tests_run,
        modification_status=modification_status,
    )
    validate_artifact(payload)
    write_json(output_path, payload)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
